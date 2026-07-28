from __future__ import annotations

from typing import Any, cast

import pytest

from distill.errors import DistillError
from distill.progress import (
    DEFAULT_MECHANISM_WEIGHTS,
    PROGRESS_STATUSES,
    MechanismState,
    MechanismWeight,
    OverallProgressAggregator,
    ProgressCounter,
    ProgressHeartbeat,
    ProgressReporter,
    clamp_percent,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_progress_heartbeat_resets_when_counter_advances() -> None:
    clock = FakeClock()
    counter = ProgressCounter()
    heartbeat = ProgressHeartbeat(
        counter,
        interval_sec=1.0,
        max_stalled_intervals=3,
        clock=clock,
    ).start()

    clock.advance(2.0)
    counter.increment()
    heartbeat.check()
    clock.advance(2.0)
    heartbeat.check()

    assert heartbeat.stopped is False


def test_stuck_progress_raises_e_stuck_and_stops_heartbeat() -> None:
    clock = FakeClock()
    counter = ProgressCounter()
    heartbeat = ProgressHeartbeat(
        counter,
        interval_sec=1.0,
        max_stalled_intervals=3,
        clock=clock,
    ).start()
    clock.advance(3.0)

    with pytest.raises(DistillError) as exc:
        heartbeat.check()

    assert exc.value.code == "E_STUCK"
    assert heartbeat.stopped is True
    assert counter.stopped is True


def test_progress_event_shape_and_json_line() -> None:
    clock = FakeClock()
    aggregator = OverallProgressAggregator((MechanismWeight("transcription", 1.0),))
    reporter = ProgressReporter(clock=clock, aggregator=aggregator)

    event = reporter.update(
        "transcription",
        percent=12.34567,
        overall_percent=9.87654,
        detail={"segments": 3},
    )

    assert event.sequence == 1
    assert event.mechanism == "transcription"
    assert event.status == "running"
    assert event.percent == 12.346
    assert event.overall_percent == 9.877
    assert event.timestamp == 0.0
    assert event.detail == {"segments": 3}
    assert event.to_dict() == {
        "type": "distill.progress",
        "sequence": 1,
        "mechanism": "transcription",
        "status": "running",
        "percent": 12.346,
        "overall_percent": 9.877,
        "timestamp": 0.0,
        "detail": {"segments": 3},
    }
    assert '"type": "distill.progress"' in event.to_json_line()


def test_progress_sequence_numbers_are_monotonic() -> None:
    reporter = ProgressReporter()

    first = reporter.update("cache_lookup")
    second = reporter.update("duration_probe")
    third = reporter.complete("duration_probe")

    assert [first.sequence, second.sequence, third.sequence] == [1, 2, 3]


def test_progress_percentages_are_clamped_and_completion_is_exact() -> None:
    aggregator = OverallProgressAggregator((MechanismWeight("download", 1.0),))
    reporter = ProgressReporter(aggregator=aggregator)

    low = reporter.update("download", percent=-10, overall_percent=-1)
    high = reporter.update("download", percent=123, overall_percent=500)
    done = reporter.complete("download")

    assert low.percent == 0.0
    assert low.overall_percent == 0.0
    assert high.percent == 100.0
    assert high.overall_percent == 100.0
    assert done.percent == 100.0
    assert done.overall_percent == 100.0
    assert clamp_percent(33.33333) == 33.333


def test_progress_status_enum_values() -> None:
    assert PROGRESS_STATUSES == (
        "pending",
        "running",
        "completed",
        "skipped_or_cached",
        "failed",
    )

    reporter = ProgressReporter()
    with pytest.raises(ValueError, match="invalid progress status"):
        reporter.update("bad", status=cast(Any, "unknown"))


def test_opaque_mechanisms_do_not_need_synthesized_percent() -> None:
    aggregator = OverallProgressAggregator((MechanismWeight("scene_detection", 1.0),))
    reporter = ProgressReporter(aggregator=aggregator)

    running = reporter.update("scene_detection", status="running")
    done = reporter.complete("scene_detection")

    assert running.percent is None
    assert running.overall_percent == 0.0
    assert done.percent == 100.0
    assert done.overall_percent == 100.0


def test_heartbeat_works_through_progress_reporter_counter() -> None:
    clock = FakeClock()
    reporter = ProgressReporter(clock=clock)
    heartbeat = ProgressHeartbeat(
        reporter.counter,
        interval_sec=1.0,
        max_stalled_intervals=3,
        clock=clock,
    ).start()

    clock.advance(2.0)
    reporter.update("ocr", percent=50)
    heartbeat.check()
    clock.advance(2.0)
    heartbeat.check()

    assert heartbeat.stopped is False


def test_progress_math_is_pure_and_inspectable() -> None:
    """Overall percent is one weighted share of the default table, derivable by hand.

    The constant is `ocr`'s weight of 10, half done, over the 107 the table
    totals: 10 * 0.5 / 107 = 4.673%. It is written out rather than recomputed
    from `DEFAULT_MECHANISM_WEIGHTS` so that a weight added or removed - as
    `redaction`'s was when D-019 stopped it being a stage - has to be
    acknowledged here instead of being absorbed silently.
    """
    reporter = ProgressReporter()
    reporter.update("ocr", percent=50, detail={"frame": 1, "frames": 2})

    assert sum(weight.weight for weight in DEFAULT_MECHANISM_WEIGHTS) == 107.0
    assert reporter.debug_info() == {
        "event_count": 1,
        "counter": 1,
        "last_sequence": 1,
        "overall_percent": 4.673,
        "mechanisms": {
            "ocr": {
                "status": "running",
                "percent": 50.0,
                "detail": {"frame": 1, "frames": 2},
            }
        },
    }


def test_weighted_aggregation_combines_mechanism_percentages() -> None:
    aggregator = OverallProgressAggregator(
        (
            MechanismWeight("fixed", 1.0),
            MechanismWeight("bytes", 2.0),
            MechanismWeight("duration", 3.0),
            MechanismWeight("frames", 4.0),
        )
    )
    states = {
        "fixed": MechanismState("fixed", status="completed"),
        "bytes": MechanismState("bytes", status="running", percent=50),
        "duration": MechanismState("duration", status="running", percent=25),
        "frames": MechanismState("frames", status="pending"),
    }

    assert aggregator.overall_percent(states) == 27.5


def test_weighted_aggregation_final_normalizes_to_100() -> None:
    aggregator = OverallProgressAggregator(
        (
            MechanismWeight("ocr", 1.0),
            MechanismWeight("rendering", 1.0),
        )
    )
    states = {
        "ocr": MechanismState("ocr", status="completed", percent=99.9),
        "rendering": MechanismState("rendering", status="skipped_or_cached"),
    }

    assert aggregator.overall_percent(states) == 100.0


def test_cache_hit_summary_marks_mechanisms_skipped_and_done() -> None:
    aggregator = OverallProgressAggregator(
        (
            MechanismWeight("transcription", 2.0),
            MechanismWeight("ocr", 1.0),
        )
    )

    summary = aggregator.cached_summary({"cache": "hit"})

    assert summary["overall_percent"] == 100.0
    assert summary["mechanisms"]["transcription"] == {
        "status": "skipped_or_cached",
        "percent": 100.0,
        "weight": 2.0,
        "detail": {"cache": "hit"},
    }
    assert summary["mechanisms"]["ocr"]["status"] == "skipped_or_cached"


def test_terminal_summary_fills_missing_mechanisms_as_skipped() -> None:
    aggregator = OverallProgressAggregator(
        (
            MechanismWeight("transcription", 2.0),
            MechanismWeight("ocr", 1.0),
        )
    )

    summary = aggregator.terminal_summary(
        {"transcription": MechanismState("transcription", status="completed")}
    )

    assert summary["overall_percent"] == 100.0
    assert summary["mechanisms"]["ocr"]["status"] == "skipped_or_cached"
    assert summary["mechanisms"]["ocr"]["detail"] == {"reason": "not_observed"}


def test_weights_are_inspectable() -> None:
    aggregator = OverallProgressAggregator(
        (
            MechanismWeight("transcription", 35.0),
            MechanismWeight("ocr", 10.0),
        )
    )

    assert aggregator.weights_summary() == [
        {"mechanism": "transcription", "weight": 35.0},
        {"mechanism": "ocr", "weight": 10.0},
    ]


def test_default_weights_include_local_vision_mechanism() -> None:
    mechanisms = {item["mechanism"] for item in OverallProgressAggregator().weights_summary()}

    assert "local_vision" in mechanisms


def test_weight_registry_rejects_duplicate_or_invalid_weights() -> None:
    with pytest.raises(ValueError, match="unique"):
        OverallProgressAggregator(
            (
                MechanismWeight("ocr", 1.0),
                MechanismWeight("ocr", 1.0),
            )
        )
    with pytest.raises(ValueError, match="positive"):
        OverallProgressAggregator((MechanismWeight("ocr", 0.0),))
