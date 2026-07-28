"""The transport breaker on the vision pass, and what a dead server costs.

R-40: three consecutive transport failures stop the run attempting further
**keyframes**; the remainder degrades to OCR-only and one **warning** records
the count. R-41's aggregation is checked in `tests/test_warning_aggregation.py`
- here only what the breaker itself emits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from distill.artifacts import FrameArtifact, Interpretation
from distill.local_vision import (
    CONSECUTIVE_TRANSPORT_FAILURE_LIMIT,
    FrameInterpreter,
    LocalVisionConfig,
    LocalVisionFailure,
    LocalVisionProbe,
    _TransportBreaker,
)

TIMEOUT = LocalVisionFailure(
    "local_vision_timeout",
    "Local vision timed out; continuing with OCR-only output.",
)
UNREACHABLE = LocalVisionFailure(
    "local_vision_rapid_mlx_unavailable",
    "Rapid-MLX local vision target was unreachable during generation.",
)
MALFORMED = LocalVisionFailure(
    "local_vision_malformed_response",
    "Rapid-MLX local vision returned a malformed interpretation.",
)


def _available_probe(config: LocalVisionConfig) -> LocalVisionProbe:
    return LocalVisionProbe(
        available=True,
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        code="local_vision_available",
        message="available",
        detail={},
    )


def _frames(tmp_path: Path, count: int) -> list[FrameArtifact]:
    """`count` **frame artifacts** carrying **extracted text** OCR already read."""
    frames = []
    for index in range(count):
        image = tmp_path / f"frame{index}.png"
        image.write_bytes(b"png")
        frames.append(
            FrameArtifact(
                index=index + 1,
                timestamp_sec=float(index),
                path=str(image),
                relative_path=f"frames/{image.name}",
                extracted_text=f"on-screen text {index}",
            )
        )
    return frames


def _reading(config: LocalVisionConfig, text: str) -> Interpretation:
    return Interpretation(
        visual_summary="A slide",
        detected_elements=("heading",),
        interpretation="It says something.",
        uncertainty="Low",
        backend=config.backend,
        model=config.model,
        prompt_profile="technical",
        verbatim_text=text,
        text_confidence="high",
    )


class _Server:
    """A fake vision server whose per-keyframe answers are scripted.

    `script` is read in call order; anything past its end repeats the last
    entry, which is how "the server died and stayed dead" is expressed without
    writing eighty entries.
    """

    def __init__(self, script: list[LocalVisionFailure | None]) -> None:
        self.script = script
        self.attempts: list[str] = []

    def __call__(
        self,
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation | None, dict[str, Any] | None]:
        self.attempts.append(image_path.name)
        index = min(len(self.attempts) - 1, len(self.script) - 1)
        answer = self.script[index]
        if answer is None:
            return _reading(config, f"read of {image_path.name}"), None
        return None, answer.warning()


def _interpreter(server: _Server, **kwargs: Any) -> FrameInterpreter:
    return FrameInterpreter(
        LocalVisionConfig(),
        probe=_available_probe,
        try_interpret=server,
        **kwargs,
    )


def test_a_server_dying_at_keyframe_3_does_not_produce_77_further_timeouts(
    tmp_path: Path,
) -> None:
    """FAILS FIRST: the finding, at the size that makes it one.

    Two **keyframes** are read, then the server dies. Every remaining keyframe
    was attempted and timed out, so an 80-keyframe run paid 78 timeouts to
    learn what the third one already said.
    """
    server = _Server([None, None, TIMEOUT])
    frames, _warnings = _interpreter(server).interpret(_frames(tmp_path, 80))

    assert len(server.attempts) == 5
    assert len(frames) == 80


def test_a_parallel_pool_stops_too_and_hands_every_keyframe_back(tmp_path: Path) -> None:
    """The pool submits every keyframe up front, so the breaker gates the worker.

    Each task asks to be admitted when it starts running rather than when it
    was queued, which is what makes a queue of eighty collapse the moment the
    third failure lands. The bound is the limit plus whatever was already in
    flight - three workers can be mid-request when the third failure returns -
    so at most six attempts, never the forty submitted.
    """
    server = _Server([TIMEOUT])
    frames, warnings = _interpreter(server, max_parallel=4).interpret(_frames(tmp_path, 40))

    assert 3 <= len(server.attempts) <= 6
    assert len(frames) == 40
    assert [w["code"] for w in warnings].count("local_vision_transport_breaker_open") == 1


def test_the_breaker_trips_on_the_third_consecutive_transport_failure(
    tmp_path: Path,
) -> None:
    server = _Server([TIMEOUT])
    _interpreter(server).interpret(_frames(tmp_path, 20))

    assert len(server.attempts) == CONSECUTIVE_TRANSPORT_FAILURE_LIMIT == 3


def test_a_refused_connection_is_a_transport_failure_like_a_timeout(
    tmp_path: Path,
) -> None:
    """The codes the transport raises are one class, not one code.

    A refused or reset connection arrives as the unavailable code and a
    timeout as its own; both say the same thing about the server.
    """
    server = _Server([UNREACHABLE, TIMEOUT, UNREACHABLE, TIMEOUT])
    _interpreter(server).interpret(_frames(tmp_path, 20))

    assert len(server.attempts) == 3


def test_a_read_that_lands_between_failures_resets_the_count(tmp_path: Path) -> None:
    """A success is evidence the transport works, so the count starts over."""
    server = _Server([TIMEOUT, TIMEOUT, None, TIMEOUT, TIMEOUT, TIMEOUT])
    _interpreter(server).interpret(_frames(tmp_path, 20))

    assert len(server.attempts) == 6


def test_a_malformed_response_is_not_a_transport_failure(tmp_path: Path) -> None:
    """A wrong server and a dead server are different findings.

    A malformed body was *delivered*: the transport carried it, so it says
    nothing about whether the server is reachable and must not push the run
    toward a breaker that exists to stop talking to one that is not. M7.1 owns
    what a malformed response costs; this test owns what it must not cost.
    """
    server = _Server([MALFORMED])
    _frames_out, warnings = _interpreter(server).interpret(_frames(tmp_path, 20))

    assert len(server.attempts) == 20
    assert [w["code"] for w in warnings if w["code"] == "local_vision_transport_breaker_open"] == []


def test_a_delivered_response_between_timeouts_resets_the_count(tmp_path: Path) -> None:
    """The same distinction where it actually bites: interleaved.

    Two timeouts, a malformed body, then two more timeouts is not three
    consecutive transport failures, because something was delivered in the
    middle of them.
    """
    server = _Server([TIMEOUT, TIMEOUT, MALFORMED, TIMEOUT, TIMEOUT, TIMEOUT])
    _interpreter(server).interpret(_frames(tmp_path, 20))

    assert len(server.attempts) == 6


def test_the_run_degrades_to_ocr_only_after_the_breaker_trips(tmp_path: Path) -> None:
    """ADR-0002 degradation: the keyframes past the trip keep their OCR text.

    The run continues and hands every frame back. The ones the breaker skipped
    carry no **interpretation** and were never attempted, but they still carry
    the **extracted text** OCR read, which is what OCR-only means.
    """
    server = _Server([None, None, TIMEOUT])
    frames, warnings = _interpreter(server).interpret(_frames(tmp_path, 10))

    assert len(frames) == 10
    assert [frame.reading is not None for frame in frames[:2]] == [True, True]
    skipped = frames[5:]
    assert [frame.reading for frame in skipped] == [None] * 5
    assert [frame.extracted_text for frame in skipped] == [
        f"on-screen text {index}" for index in range(5, 10)
    ]
    assert [w["code"] for w in warnings].count("local_vision_transport_breaker_open") == 1


def test_one_warning_records_the_transport_failure_count(tmp_path: Path) -> None:
    server = _Server([None, None, TIMEOUT])
    _frames_out, warnings = _interpreter(server).interpret(_frames(tmp_path, 80))

    breaker = [w for w in warnings if w["code"] == "local_vision_transport_breaker_open"]
    assert len(breaker) == 1
    assert breaker[0]["stage"] == "local_vision"
    assert "3 consecutive transport failures" in breaker[0]["message"]
    assert "75" in breaker[0]["message"]
    # The 75 are the ones the breaker refused to attempt. The three that timed
    # out before it opened also continue with OCR-only output and are not in
    # that number, so the sentence has to say which group it is counting.
    assert "not attempted" in breaker[0]["message"]


def test_a_breaker_that_skipped_nothing_does_not_warn_about_nothing(tmp_path: Path) -> None:
    """The trip lands on the last keyframe, so no keyframe was ever refused.

    The breaker exists to stop a run paying for a server that is gone, and its
    **warning** exists to say what that cost. Tripping on the final keyframe
    costs nothing - the three failures already have their own warnings - so a
    record here would report a **degradation** that did not happen, and would
    do it in a sentence saying zero of three keyframes were affected.
    """
    server = _Server([TIMEOUT])
    _frames_out, warnings = _interpreter(server).interpret(_frames(tmp_path, 3))

    assert len(server.attempts) == 3
    assert [w["code"] for w in warnings].count("local_vision_transport_breaker_open") == 0
    # The failures themselves are still reported - folded, as R-41 folds them;
    # it is only the summary of what the trip cost that has nothing to say.
    assert [(w["code"], w["occurrences"]) for w in warnings] == [(TIMEOUT.code, 3)]


def test_the_breaker_transition_is_emitted_with_the_attempt_count_that_tripped_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observability: the boundary log says when the breaker opened and on what.

    Emitted whether or not the debug toggle is on, because the run this
    explains is the one nobody re-ran with `--debug`.
    """
    caplog.set_level(logging.DEBUG, logger="distill.local_vision")
    server = _Server([None, None, TIMEOUT])
    _interpreter(server).interpret(_frames(tmp_path, 80))

    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "distill.local_vision"
    ]
    opened = [event for event in events if event["event"] == "breaker.open"]
    assert len(opened) == 1
    detail = opened[0]["detail"]
    assert detail["state"] == "open"
    assert detail["attempt"] == 5
    assert detail["consecutive_failures"] == 3
    assert detail["code"] == "local_vision_timeout"
    assert detail["frame"] == 5


def test_the_reset_of_a_partial_failure_run_is_emitted_too(tmp_path: Path) -> None:
    """The other transition: a count that was climbing and stopped.

    Without it the log shows two timeouts and then silence, and the reader
    cannot tell a recovery from a run that ended.
    """
    server = _Server([TIMEOUT, TIMEOUT, None])
    interpreter = _interpreter(server, debug=True)
    interpreter.interpret(_frames(tmp_path, 4))

    resets = [
        event
        for event in interpreter.debug_info()["trace_events"]
        if event["event"] == "breaker.reset"
    ]
    assert len(resets) == 1
    assert resets[0]["detail"] == {
        "state": "closed",
        "cleared_failures": 2,
        "attempt": 3,
        "frame": 3,
    }


def test_a_late_response_cannot_report_a_reset_for_an_open_breaker() -> None:
    """An attempt that was already in flight lands after the trip.

    With a parallel pool, a request that passed admission before the third
    failure returns after it. The breaker does not close, so nothing that
    arrives afterwards is a state transition - reporting one would put
    `breaker.reset` in the log of a run whose every remaining keyframe was
    still being skipped, and zero the count that explains why.
    """
    breaker = _TransportBreaker()
    for frame_number in (1, 2):
        assert breaker.record(frame_number=frame_number, code="local_vision_timeout") is None
    tripped = breaker.record(frame_number=3, code="local_vision_timeout")
    assert tripped is not None and tripped["state"] == "open"

    assert breaker.record(frame_number=4, code=None) is None
    assert breaker.record(frame_number=5, code="local_vision_malformed_response") is None

    state = breaker.state()
    assert state["open"] is True
    assert state["consecutive_failures"] == 3
