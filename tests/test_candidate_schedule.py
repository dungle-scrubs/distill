"""The **keyframe** candidate schedule: what it may contain, and that it ends.

R-48 - candidate generation must terminate for every option value that passes
validation - and R-41's **warning** for the candidates a run's frame budget
never reaches. These tests are about the schedule, not about extracting
anything from it.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

import pytest
from test_local_integration import fake_transcribe, make_short_screencast

from distill import frame_selection
from distill import pipeline as distill_session
from distill.errors import DistillError
from distill.frame_selection import (
    CANDIDATE_TIMESTAMP_QUANTUM_SEC as QUANTUM,
)
from distill.frame_selection import (
    filtered_candidates,
    select_keyframes,
)
from distill.options import DistillOptions

SWEEP_SEED = 20260728
"""Fixed, so a failing tuple is reproducible from the report alone."""


def test_a_sub_millisecond_static_window_is_refused_rather_than_spun_on() -> None:
    """finding 7: `--max-static-window-sec 0.0001` never returned.

    The gap-filling cursor advances by `max_static_window_sec` and is rounded
    to the millisecond, so a window below that quantum landed back on the
    value it started from and the walk toward the next candidate never
    arrived. It is refused at both ends: at the options boundary, where the
    number is still the operator's and nothing has been decoded yet, and at
    the schedule itself, which is the code that cannot honour it.
    """
    with pytest.raises(DistillError) as at_the_boundary:
        DistillOptions.from_args({"max_static_window_sec": 0.0001})

    assert at_the_boundary.value.code == "E_BAD_OPTIONS"
    assert at_the_boundary.value.stage == "options"
    # The floor is in the message, not only in the code: an operator who is
    # told 0.0001 is wrong still does not know what is right.
    assert at_the_boundary.value.message == (
        "max_static_window_sec must be a finite number 0.001 or greater"
    )

    with pytest.raises(DistillError) as at_the_schedule:
        filtered_candidates([0.0, 5.0], 10.0, 4.0, 0.0001)

    assert at_the_schedule.value.code == "E_BAD_OPTIONS"
    # Same code, different stage, and deliberately so: a **stage** names where
    # the failure was raised, not what kind of value caused it. The operator's
    # number is refused at `options` when it arrives and at `frames` when a
    # schedule is asked to honour it, and a reader of the second needs to know
    # the run got as far as building one.
    assert at_the_schedule.value.stage == "frames"
    assert "0.001 or greater" in at_the_schedule.value.message
    assert at_the_schedule.value.details == {"max_static_window_sec": "0.0001"}


def test_the_smallest_expressible_window_is_admitted_and_still_advances() -> None:
    """The floor is a floor, not a ban: one quantum is a schedule.

    Rejection was chosen over widening a sub-quantum window to the quantum,
    so the value just above the floor has to produce a well-formed schedule -
    strictly increasing, and no two entries collapsing onto the same
    millisecond seek.
    """
    # A bounded duration cap paired with the floor window, so the pair stays
    # under the candidate-count ceiling (M7.1): the floor is admitted, and it is
    # the combination with an uncapped duration that is unphysical, not the
    # window value itself.
    assert DistillOptions.from_args(
        {"max_static_window_sec": QUANTUM, "max_duration_sec": 100.0}
    ).max_static_window_sec == QUANTUM

    schedule = filtered_candidates([0.0], 0.01, 0.0, QUANTUM)

    assert schedule == sorted(set(schedule))
    assert schedule[0] == 0.0
    assert schedule[-1] <= 0.01
    assert len(schedule) > 1
    for earlier, later in itertools.pairwise(schedule):
        assert later - earlier >= QUANTUM * 0.9, schedule


def test_a_step_either_advances_or_the_walk_stops() -> None:
    """The property every loop in the module leans on to terminate.

    The last case is the one that cannot be reached from `filtered_candidates`
    in a test that finishes: past about 2e18 seconds a 90-second step is
    smaller than the gap between two representable floats, so the cursor would
    stand still. It answers with nothing instead.
    """
    assert frame_selection._step_from(0.0, 90.0, 1000.0) == 90.0
    assert frame_selection._step_from(950.0, 90.0, 1000.0) == 1000.0
    assert frame_selection._step_from(1000.0, 90.0, 1000.0) is None
    assert frame_selection._step_from(2e18, 90.0, 4e18) is None


def test_the_fallback_walk_samples_the_interval_it_was_given() -> None:
    """The schedule with no detector behind it, and the end of the source.

    An interval wider than the source is one sample, not two: the end of the
    source is where this walk stops rather than a place it is pulled back to.
    Each sample is on the millisecond it will be sought at, so a **frame
    artifact** records the timestamp ffmpeg was actually given.

    The sub-second intervals are the range the name only used to be true over.
    A one-second floor lived here, so an operator who asked for 0.002s was
    answered 500x coarser - the widening `ensure_static_window_is_expressible`
    refuses at the other door, done silently at this one.
    """
    assert frame_selection.fixed_interval_candidates(10.0004, 90.0) == [0.0]
    assert frame_selection.fixed_interval_candidates(4.1, 1.0004) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frame_selection.fixed_interval_candidates(4.0, 1.0) == [0.0, 1.0, 2.0, 3.0]
    assert frame_selection.fixed_interval_candidates(0.0, 1.0) == []
    assert frame_selection.fixed_interval_candidates(0.01, 0.002) == [
        0.0,
        0.002,
        0.004,
        0.006,
        0.008,
    ]
    assert frame_selection.fixed_interval_candidates(0.005, QUANTUM) == [
        0.0,
        0.001,
        0.002,
        0.003,
        0.004,
    ]
    assert frame_selection.fixed_interval_candidates(1.0, 0.25) == [0.0, 0.25, 0.5, 0.75]


def test_a_source_no_detector_answered_for_is_still_sampled_at_the_static_window() -> None:
    """The window is the schedule when the detector offered nothing.

    An empty candidate list is exactly the static-slide case
    `max_static_window_sec` exists for - one slide on screen for the whole
    source, no scene change to detect - so it is the last place the window may
    be reinterpreted. A validated 0.002s window answered as 1.0s is
    under-sampling by a factor of 500, reported as nothing.
    """
    assert filtered_candidates([], 0.01, 0.0, 0.002) == [0.0, 0.002, 0.004, 0.006, 0.008]
    assert filtered_candidates([], 2.0, 0.0, 0.5) == [0.0, 0.5, 1.0, 1.5]


# The sweep is executed, so it is bounded: a tuple whose schedule cannot
# exceed this many entries runs, and the closed-form bound is asserted for
# every tuple that does. Termination is the bound being finite and every step
# advancing, not the clock.
SWEEP_POINT_CEILING = 5_000

SWEEP_TUPLE_DEADLINE_SEC = 10.0
"""Generous: every tuple in the sweep returns in microseconds on this hardware.

It is not a performance assertion. It exists because the failure this sweep
guards against is a cursor that stops advancing, and a cursor that stops
advancing does not fail a test - it appends forever, and pytest hangs with no
tuple named and no seed printed. A test that cannot report is not a test, so
the stall is turned into an ordinary failure carrying the tuple that caused it.
"""


@contextmanager
def _fails_rather_than_hangs(report: str) -> Iterator[None]:
    """Turn a non-terminating schedule into a failure that names its tuple.

    `SIGALRM` rather than a thread or a subprocess, because the loop being
    watched is pure Python in this process: the handler runs between bytecodes
    and the `AssertionError` it raises unwinds through `filtered_candidates`
    with the tuple already in hand. On a platform without `SIGALRM` the sweep
    still runs and a stall still hangs - there is nothing to report from, and
    Distill is macOS-first.
    """
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - macOS and Linux have it
        yield
        return

    def stalled(_signum: int, _frame: FrameType | None) -> None:
        raise AssertionError(
            f"candidate generation did not terminate within "
            f"{SWEEP_TUPLE_DEADLINE_SEC}s: {report}"
        )

    previous = signal.signal(signal.SIGALRM, stalled)
    signal.setitimer(signal.ITIMER_REAL, SWEEP_TUPLE_DEADLINE_SEC)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _candidate_shapes(duration: float) -> tuple[tuple[float, ...], ...]:
    return (
        (),
        (0.0,),
        (duration,),
        (0.0, duration),
        (duration / 2,),
        # Out of range on both sides, unsorted, and repeated: the schedule
        # clamps and dedupes rather than trusting the detector.
        (duration * 3, -5.0, duration / 2, duration / 2),
    )


def _swept_tuples() -> Iterator[tuple[tuple[float, ...], float, float, float]]:
    """The boundary structure of the validated option space, then a sample.

    Boundaries first because that is where the arithmetic changes character:
    the quantum itself, a window that divides the duration, one that exactly
    equals it, and one wider than the source is long.

    A tuple whose schedule would hold more than `SWEEP_POINT_CEILING` entries is
    skipped, and what that skip is standing in front of is worth saying plainly:
    the schedule is bounded by `ceil(duration_sec / max_static_window_sec)`,
    which is finite for every validated tuple but is not a bound on *memory*.
    `--max-duration-sec 1e9` with a matching source and a one-second window is a
    billion-entry list of floats - tens of gigabytes, so an out-of-memory kill
    rather than a large schedule. R-48 is about termination and that walk does
    terminate; the size is a separate, documented, deliberately un-redesigned
    gap (bounding the schedule by `max_keyframes` was considered and rejected:
    candidates that dedupe on pHash are examined and dropped, so capping the
    list at the frame budget would cost real frames on the ordinary path).
    """
    # Durations and windows that are not whole milliseconds are in here on
    # purpose: a step of 1.0004s is quantized to 1.000s, so the schedule
    # advances by slightly less than the window it was given, and a duration of
    # 10.0004s is a ceiling no timestamp lands on.
    durations = (QUANTUM, 0.02, 0.5, 1.0, 4.1, 10.0004, 20.0, 90.0, 7200.0)
    for duration in durations:
        windows = {
            QUANTUM,
            2 * QUANTUM,
            0.0014,
            1.0,
            1.0004,
            90.0,
            duration / 1000,
            duration / 7,
            duration / 3,
            duration / 2,
            duration,
            duration * 2,
        }
        for window in sorted(w for w in windows if w >= QUANTUM):
            if duration / window > SWEEP_POINT_CEILING:
                continue
            for min_interval in (0.0, QUANTUM, duration / 4, duration, duration * 2):
                for candidates in _candidate_shapes(duration):
                    yield candidates, duration, min_interval, window

    sample = random.Random(SWEEP_SEED)
    drawn = 0
    while drawn < 200:
        duration = sample.uniform(QUANTUM, 5_000.0)
        window = sample.uniform(QUANTUM, duration * 2)
        if duration / window > SWEEP_POINT_CEILING:
            continue
        drawn += 1
        yield (
            tuple(sample.uniform(-10.0, duration * 1.5) for _ in range(sample.randint(0, 5))),
            duration,
            sample.uniform(0.0, duration),
            window,
        )


def test_candidate_generation_terminates_across_a_deterministic_boundary_and_sample_sweep() -> None:
    """R-48, structurally: a finite bound, and a step that always advances.

    A schedule holds at most one entry per candidate the detector offered plus
    one per step of source, because gap filling only ever moves forward and
    only ever toward `duration_sec`. That bound is what makes the loops end;
    asserting it for every tuple is what proves the walk has not started
    standing still again.

    A step is the window less at most half a quantum, because quantizing can
    round one down - a 0.0014s window advances the cursor by 0.001s. Dividing
    by the window itself would be a bound the code is not obliged to meet.

    What this sweeps, stated so the name is not read as more than it is: the
    boundary structure of the validated option space, plus a `Random(20260728)`
    sample - a few thousand tuples, not every validated tuple, which is an
    uncountable set. And tuples whose schedule would exceed
    `SWEEP_POINT_CEILING` entries are skipped rather than run: the sweep
    executes what it asserts about, and a validated tuple can name a schedule
    too large to hold in memory (see `_swept_tuples`). Those are not tuples the
    property is false for; they are tuples this test declines to build.
    """
    for candidates, duration, min_interval, window in _swept_tuples():
        tuple_report = (
            f"seed={SWEEP_SEED} candidates={candidates} duration={duration} "
            f"min_interval={min_interval} window={window}"
        )
        with _fails_rather_than_hangs(tuple_report):
            schedule = filtered_candidates(candidates, duration, min_interval, window)

        assert schedule == sorted(set(schedule)), tuple_report
        assert all(0.0 <= point <= duration for point in schedule), tuple_report
        for earlier, later in itertools.pairwise(schedule):
            assert later - earlier >= QUANTUM * 0.9, tuple_report
        step = max(window - QUANTUM / 2, QUANTUM)
        bound = math.ceil(duration / step) + len(set(candidates)) + 2
        assert len(schedule) <= bound, tuple_report
        if not candidates:
            # The bound above is an upper one, so it cannot see under-sampling:
            # a schedule of one entry satisfies it for every window. With no
            # detector behind it the whole schedule is the window, so this is
            # where a widened window is visible - every gap is at most the
            # window (plus the half quantum rounding may add), and there are at
            # least as many entries as the source has windows in it.
            for earlier, later in itertools.pairwise(schedule):
                assert later - earlier <= window + QUANTUM / 2, tuple_report
            assert len(schedule) >= duration / (window + QUANTUM / 2), tuple_report


def _four_candidates_and_a_frame_each(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pHash sequence whose neighbours are 64 bits apart, so nothing dedupes
    # and the frame budget is the only thing that can end the run early. Built
    # here rather than at module level: a shared cycle carries its position from
    # one test into the next, so a test that consumed an odd number of hashes
    # would hand the next one a starting hash that dedupes against its first
    # frame. Parity happens to work out today, which is not a property to leave
    # a test resting on.
    distinct_hashes = itertools.cycle(("0" * 16, "f" * 16))
    monkeypatch.setattr(
        frame_selection, "scene_midpoint_candidates", lambda _path, _duration: [0.0, 2.0, 4.0, 6.0]
    )
    monkeypatch.setattr(
        frame_selection,
        "extract_frame",
        lambda _video, _timestamp, output: (output.write_bytes(b"png"), (True, []))[1],
    )
    monkeypatch.setattr(frame_selection, "phash", lambda _path: next(distinct_hashes))


def test_the_candidates_a_frame_budget_never_reaches_are_warned_about(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-41: the **bundle** covers less than the schedule asked for, and says so.

    Without it a run that stopped at its budget and a run whose source held
    nothing more publish the same frames, and the operator has no way to know
    that raising `--max-keyframes` would have shown them more.
    """
    _four_candidates_and_a_frame_each(monkeypatch)

    frames, warnings = select_keyframes(
        Path("demo.mp4"),
        tmp_path,
        duration_sec=6.0,
        max_keyframes=2,
        min_interval_sec=1.0,
        max_static_window_sec=90.0,
    )

    assert len(frames) == 2
    assert warnings == [
        {
            "stage": "frames",
            "code": "keyframe_budget_reached",
            "message": (
                "stopped at max_keyframes=2; 2 of 4 candidate timestamps were not examined"
            ),
            "occurrences": 1,
        }
    ]


def test_a_run_that_examined_every_candidate_warns_about_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The warning fires on truncation, not on spending the budget exactly."""
    _four_candidates_and_a_frame_each(monkeypatch)

    frames, warnings = select_keyframes(
        Path("demo.mp4"),
        tmp_path,
        duration_sec=6.0,
        max_keyframes=4,
        min_interval_sec=1.0,
        max_static_window_sec=90.0,
    )

    assert len(frames) == 4
    assert warnings == []


def test_the_truncation_warning_reaches_the_published_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ADR-0002: a **warning** is only worth raising if a reader is given it.

    One **keyframe** out of a schedule of five, so the run really is cut
    short, and the record is read back off the published **manifest** rather
    than off the return value. The schedule is dictated rather than detected -
    the one-second fixture holds a single scene - but everything after it is
    the real run: real ffmpeg grabs, the real stage fold, a real publish.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)
    monkeypatch.setattr(
        frame_selection,
        "scene_midpoint_candidates",
        lambda _path, _duration: [0.0, 0.2, 0.4, 0.6, 0.8],
    )

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 1,
            "min_interval_sec": 0.1,
            "max_static_window_sec": 90,
        }
    )

    manifest = json.loads(Path(response["manifest_path"]).read_text())
    truncation = [
        item for item in manifest["warnings"] if item["code"] == "keyframe_budget_reached"
    ]
    assert len(truncation) == 1
    assert truncation[0]["stage"] == "frames"
    assert truncation[0]["occurrences"] == 1
    assert "max_keyframes=1" in truncation[0]["message"]
