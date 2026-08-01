"""Re-asking the chain after a wait long enough to have outlived the answer.

Chain resolution happens during source resolution, which is *before* a run takes
the lock on the **bundle key** it resolved to. A run that then waits out a
contended lock arrives at its own work holding an availability answer it
gathered minutes ago - and the memo that answer came from has a life measured in
the same minutes (D-004).

The seam this file starts from is the threshold itself. It is a relationship
between three constants, two of which live in `bundle_store.py` and one in
`vision_chain.py`, so these tests import across that boundary deliberately: the
store must not learn what a vision endpoint is (D-016), but a test may relate
what two modules each know. Asserting the relation here is what keeps it from
becoming a coincidence nobody would notice breaking.

The second seam is `ProcessingRun.execute`, the one place that holds both the
wait `BundleStore.begin` measured and the inputs a second walk needs. The wait
is real here - a lock another descriptor holds, polled out on a clock that only
moves when the waiter sleeps - so what the run acts on is the number `_take_lock`
computed rather than one a test handed it. Nothing reaches a server: the chain's
probe is the module-level seam `source._probe_endpoint`, and the media stages are
fakes, because what they produce is not what is under test.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_bundle_locking import FakeClock, hold_the_lock

from distill import bundle_store, pipeline, source
from distill.artifacts import FrameArtifact, Provenance, RedactionState
from distill.bundle_store import (
    BATCH_ITEM_LOCK_WAIT_SEC,
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleStore,
)
from distill.local_vision import LocalVisionConfig
from distill.media_inspect import source_hash
from distill.options import DistillOptions
from distill.pipeline import ProcessingRun
from distill.progress import ProgressReporter
from distill.source import SourceInfo, _resolved_for
from distill.vision_chain import REVALIDATE_AFTER_WAIT_SEC


def test_a_contended_batch_item_can_never_wait_its_way_to_revalidation() -> None:
    """A batch item's whole lock budget is spent well short of the threshold.

    A playlist item gives up on a contended key after 5 s and lets the batch
    proceed, so the longest it can possibly have held a stale availability
    answer is 5 s against a memo measured in minutes (D-006). Re-keying such a
    run would buy nothing and charge every contended batch a second resolution,
    so the threshold has to sit strictly above the budget - not merely above
    the waits observed in practice.
    """
    assert BATCH_ITEM_LOCK_WAIT_SEC < REVALIDATE_AFTER_WAIT_SEC


def test_a_single_source_run_that_waits_it_out_lands_past_the_threshold() -> None:
    """Spending the whole single-source budget is enough to owe a second walk.

    The run a user is watching waits 300 s for a contended key rather than
    failing, and the memo its availability answer came from is trusted for the
    same 300 s - so the worst case is not a slightly stale answer but an expired
    one (D-004). The threshold has to be strictly inside that budget, or the
    exact case it exists for would be the one case that never triggers it.
    """
    assert REVALIDATE_AFTER_WAIT_SEC < SINGLE_SOURCE_LOCK_WAIT_SEC


ENDPOINT = LocalVisionConfig(model="a-served-reader", base_url="http://127.0.0.1:8000/v1")
FINGERPRINT = "the-fingerprint-of-a-fixture"


def _fake_transcribe(*_args: Any, **_kwargs: Any) -> tuple[None, list[dict[str, str]]]:
    return None, []


def _fake_select_keyframes(
    _video: Path,
    frames_dir: Path,
    *_args: Any,
    redaction: RedactionState,
    **_kwargs: Any,
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    image = frames_dir / "frame_0001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return [
        FrameArtifact(
            index=1,
            timestamp_sec=0.0,
            path=str(image),
            relative_path="frames/frame_0001.png",
            extracted_text="a frame this test never reads",
            redaction=redaction,
        )
    ], []


def _fake_ocr_frames(
    frames: list[FrameArtifact], *_args: Any
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    return frames, []


def _fake_interpret(
    frames: list[FrameArtifact], *_args: Any, **_kwargs: Any
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    return frames, []


@dataclass
class Contended:
    """One `execute()` that came out of a wait, and what the wait cost.

    `probes` holds only the endpoints asked *after* the run started: the walk
    that settled the **bundle key** happens before there is a run at all, so
    counting from zero at `execute` is what makes "asked a second time" a
    countable thing rather than an inference from a total.
    """

    response: dict[str, Any]
    probes: list[LocalVisionConfig]
    bundle_key: str
    lock_events: list[dict[str, Any]]
    pipeline_events: list[dict[str, Any]]


def _structured(caplog: pytest.LogCaptureFixture, logger: str) -> list[dict[str, Any]]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == logger and record.message.startswith("{")
    ]


def _run_that_waited(
    seconds: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> Contended:
    """A whole run of one source, held out of its own **bundle key** for `seconds`.

    The contention is the kernel's - a descriptor this process holds, which
    `flock` refuses to the run exactly as another process's would - and the
    clock is the store's injected one, so the budget is spent in the poll loop
    without a second of real time passing. `BundleStore.open` is replaced for
    the duration because `execute` opens its own store: production has no seam
    for a clock there, and the wait is the input under test.
    """
    root = tmp_path / "output"
    root.mkdir()
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"not a video, and never opened")

    probes: list[LocalVisionConfig] = []

    def probe(endpoint: LocalVisionConfig) -> bool:
        probes.append(endpoint)
        return True

    monkeypatch.setattr(source, "_probe_endpoint", probe)
    options = DistillOptions(
        caption_frames=True,
        local_vision_endpoints=(ENDPOINT,),
        job_id="job-1",
    )
    # The walk source resolution performs, run here because that is where it
    # happens: the run this test drives starts from its outcome, holding a
    # **bundle key** an endpoint answered for before the wait began.
    resolution = _resolved_for(options, FINGERPRINT, "local", root)
    bundle_key = source_hash(FINGERPRINT, resolution.opts_hash)
    probes.clear()

    holder: dict[str, int | None] = {"fd": hold_the_lock(root, bundle_key)}
    clock = FakeClock()

    def sleep(interval: float) -> None:
        clock.sleep(interval)
        fd = holder["fd"]
        if fd is not None and clock.now >= seconds:
            os.close(fd)
            holder["fd"] = None

    monkeypatch.setattr(
        BundleStore,
        "open",
        classmethod(lambda cls, root, **_kwargs: cls(Path(root).resolve(), clock.monotonic, sleep)),
    )
    monkeypatch.setattr(pipeline, "transcribe_with_imports", _fake_transcribe)
    monkeypatch.setattr(pipeline, "select_keyframes", _fake_select_keyframes)
    monkeypatch.setattr(pipeline, "ocr_frames", _fake_ocr_frames)
    monkeypatch.setattr(pipeline, "interpret_frames_with_local_vision", _fake_interpret)
    caplog.set_level(logging.DEBUG)

    run = ProcessingRun(
        source=SourceInfo(
            source_type="local",
            resolved_path=video,
            duration_sec=1.0,
            source_fingerprint=FINGERPRINT,
            source_hash=bundle_key,
            warnings=[],
            provenance=Provenance(
                title=video.name,
                duration_sec=1.0,
                processed_at="2026-08-01T09:00:00Z",
            ),
            resolved_options=resolution.options,
        ),
        options=resolution.options,
        output_root=root,
        progress=ProgressReporter(),
        tool="process_local_video",
    )
    try:
        response = run.execute()
    finally:
        fd = holder["fd"]
        if fd is not None:
            os.close(fd)
    return Contended(
        response=response,
        probes=probes,
        bundle_key=bundle_key,
        lock_events=_structured(caplog, bundle_store.LOGGER.name),
        pipeline_events=_structured(caplog, pipeline.LOGGER.name),
    )


def _counted(events: list[dict[str, Any]], name: str) -> int:
    return sum(1 for event in events if event["event"] == name)


def test_a_run_that_waited_past_the_threshold_re_asks_and_keeps_its_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FAILS FIRST: the wait is measured, and nothing acts on it.

    The run walked the chain before it queued, waited out a contended key for
    longer than the answer it was carrying is worth trusting, and then read
    against that answer anyway (D-004). Asking again is the whole feature.

    The second walk agrees here - the same endpoint answers, so it names the
    same key - and agreement must cost nothing: the lock in hand was taken on
    that key, so there is nothing to release and nothing to take again. One
    acquisition and one release over the whole run is what "kept the lock" is,
    stated in the only terms an operator could check.
    """
    contended = _run_that_waited(REVALIDATE_AFTER_WAIT_SEC + 50.0, tmp_path, monkeypatch, caplog)

    assert contended.probes == [ENDPOINT]
    assert contended.response["source_hash"] == contended.bundle_key
    assert _counted(contended.lock_events, "lock_acquired") == 1
    assert _counted(contended.lock_events, "lock_released") == 1


def test_a_wait_below_the_threshold_leaves_the_chain_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A whole batch-item budget spent waiting is still not worth re-asking.

    The guard on the other side of D-006. Revalidation is not free - the walk
    scans every candidate key against the store before it would probe - and an
    answer five seconds old is an answer, so a threshold that triggered on any
    non-zero wait would charge every contended item of every playlist for
    staleness that cannot have occurred.
    """
    contended = _run_that_waited(BATCH_ITEM_LOCK_WAIT_SEC, tmp_path, monkeypatch, caplog)

    assert contended.probes == []
    assert contended.response["source_hash"] == contended.bundle_key


def test_revalidation_says_which_key_waited_how_long_and_against_what(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second walk nobody can see is a second walk nobody can explain.

    Revalidation is invisible from the outside: it costs a cache scan, usually
    changes nothing, and happens only to runs that were already slow for another
    reason. Three fields make it accountable - which **bundle key** was being
    re-asked about, how long the wait was, and the threshold that wait was
    measured against - so a run that re-asked and a run that did not are told
    apart by the numbers rather than by inference from the total.

    `bundle_key` is the correlation field: the store logged `lock_acquired`
    under it moments earlier, and reading the two together is how an operator
    sees a wait turn into a decision.
    """
    contended = _run_that_waited(REVALIDATE_AFTER_WAIT_SEC + 50.0, tmp_path, monkeypatch, caplog)

    revalidated = [
        event for event in contended.pipeline_events if event["event"] == "chain_revalidated"
    ]
    assert len(revalidated) == 1
    detail = revalidated[0]["detail"]
    assert detail["bundle_key"] == contended.bundle_key
    assert detail["waited_sec"] == pytest.approx(REVALIDATE_AFTER_WAIT_SEC + 50.0, abs=0.05)
    assert detail["revalidate_after_sec"] == REVALIDATE_AFTER_WAIT_SEC

    acquired = [event for event in contended.lock_events if event["event"] == "lock_acquired"]
    assert [event["detail"]["bundle_key"] for event in acquired] == [contended.bundle_key]
