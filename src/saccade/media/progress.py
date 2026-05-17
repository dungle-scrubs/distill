"""Structured progress events for media-ingest apps."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import MediaIngestError

ProgressStatus = Literal[
    "pending",
    "running",
    "completed",
    "skipped_or_cached",
    "failed",
]

PROGRESS_STATUSES: tuple[ProgressStatus, ...] = (
    "pending",
    "running",
    "completed",
    "skipped_or_cached",
    "failed",
)

TERMINAL_PROGRESS_STATUSES = {"completed", "skipped_or_cached"}


@dataclass
class ProgressCounter:
    value: int = 0
    stopped: bool = False

    def increment(self, amount: int = 1) -> None:
        if self.stopped:
            raise RuntimeError("progress counter is stopped")
        self.value += amount


@dataclass(frozen=True)
class ProgressEvent:
    sequence: int
    mechanism: str
    status: ProgressStatus
    overall_percent: float
    timestamp: float
    event_type: str = "media.progress"
    percent: float | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.event_type,
            "sequence": self.sequence,
            "mechanism": self.mechanism,
            "status": self.status,
            "overall_percent": self.overall_percent,
            "timestamp": self.timestamp,
        }
        if self.percent is not None:
            payload["percent"] = self.percent
        if self.detail:
            payload["detail"] = self.detail
        return payload

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class MechanismState:
    mechanism: str
    status: ProgressStatus = "pending"
    percent: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MechanismWeight:
    mechanism: str
    weight: float


DEFAULT_MECHANISM_WEIGHTS: tuple[MechanismWeight, ...] = (
    MechanismWeight("cache_lookup", 1.0),
    MechanismWeight("source_resolution", 4.0),
    MechanismWeight("source_fingerprint", 4.0),
    MechanismWeight("transcript_discovery", 4.0),
    MechanismWeight("transcript_import", 12.0),
    MechanismWeight("audio_download", 15.0),
    MechanismWeight("duration_probe", 1.0),
    MechanismWeight("audio_extraction", 8.0),
    MechanismWeight("transcription", 35.0),
    MechanismWeight("rendering", 4.0),
    MechanismWeight("bundle_publish", 3.0),
)


def clamp_percent(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 3)


def mechanism_percent(state: MechanismState | None) -> float:
    if state is None:
        return 0.0
    if state.status in TERMINAL_PROGRESS_STATUSES:
        return 100.0
    if state.percent is None:
        return 0.0
    return clamp_percent(state.percent)


@dataclass
class OverallProgressAggregator:
    weights: tuple[MechanismWeight, ...] = DEFAULT_MECHANISM_WEIGHTS

    def __post_init__(self) -> None:
        mechanisms = [item.mechanism for item in self.weights]
        if len(mechanisms) != len(set(mechanisms)):
            raise ValueError("mechanism weights must be unique")
        if any(item.weight <= 0 for item in self.weights):
            raise ValueError("mechanism weights must be positive")

    def overall_percent(self, states: dict[str, MechanismState]) -> float:
        total_weight = sum(item.weight for item in self.weights)
        if total_weight <= 0:
            return 100.0
        weighted = sum(
            mechanism_percent(states.get(item.mechanism)) * item.weight for item in self.weights
        )
        percent = weighted / total_weight
        if self._all_weighted_mechanisms_terminal(states):
            return 100.0
        return clamp_percent(percent)

    def summary(self, states: dict[str, MechanismState]) -> dict[str, Any]:
        return {
            "overall_percent": self.overall_percent(states),
            "mechanisms": {
                item.mechanism: self._mechanism_summary(item, states.get(item.mechanism))
                for item in self.weights
            },
        }

    def cached_summary(self, detail: dict[str, Any] | None = None) -> dict[str, Any]:
        states = {
            item.mechanism: MechanismState(
                item.mechanism,
                status="skipped_or_cached",
                percent=100.0,
                detail=detail or {},
            )
            for item in self.weights
        }
        return self.summary(states)

    def terminal_summary(
        self,
        states: dict[str, MechanismState],
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        terminal_states = dict(states)
        for item in self.weights:
            if item.mechanism not in terminal_states:
                terminal_states[item.mechanism] = MechanismState(
                    item.mechanism,
                    status="skipped_or_cached",
                    percent=100.0,
                    detail=detail or {"reason": "not_observed"},
                )
        return self.summary(terminal_states)

    def weights_summary(self) -> list[dict[str, float | str]]:
        return [{"mechanism": item.mechanism, "weight": item.weight} for item in self.weights]

    def _all_weighted_mechanisms_terminal(
        self,
        states: dict[str, MechanismState],
    ) -> bool:
        return all(
            states.get(item.mechanism) is not None
            and states[item.mechanism].status in TERMINAL_PROGRESS_STATUSES
            for item in self.weights
        )

    @staticmethod
    def _mechanism_summary(
        weight: MechanismWeight,
        state: MechanismState | None,
    ) -> dict[str, Any]:
        if state is None:
            return {
                "status": "pending",
                "percent": 0.0,
                "weight": weight.weight,
                "detail": {},
            }
        return {
            "status": state.status,
            "percent": mechanism_percent(state),
            "weight": weight.weight,
            "detail": state.detail,
        }


@dataclass
class ProgressReporter:
    clock: Callable[[], float] = time.time
    counter: ProgressCounter = field(default_factory=ProgressCounter)
    states: dict[str, MechanismState] = field(default_factory=dict)
    events: list[ProgressEvent] = field(default_factory=list)
    emitter: Callable[[ProgressEvent], None] | None = None
    aggregator: OverallProgressAggregator = field(default_factory=OverallProgressAggregator)
    event_type: str = "media.progress"
    _sequence: int = 0

    def update(
        self,
        mechanism: str,
        *,
        status: ProgressStatus = "running",
        percent: float | None = None,
        overall_percent: float | None = None,
        detail: dict[str, Any] | None = None,
        advance: int = 1,
    ) -> ProgressEvent:
        if status not in PROGRESS_STATUSES:
            raise ValueError(f"invalid progress status: {status}")
        if percent is not None:
            percent = clamp_percent(percent)
        if status == "completed":
            percent = 100.0 if percent is None else percent
        if overall_percent is None:
            preview_states = {
                **self.states,
                mechanism: MechanismState(
                    mechanism,
                    status=status,
                    percent=percent,
                    detail=detail or {},
                ),
            }
            overall_percent = self.aggregator.overall_percent(preview_states)
        else:
            overall_percent = clamp_percent(overall_percent)

        self._sequence += 1
        event = ProgressEvent(
            sequence=self._sequence,
            mechanism=mechanism,
            status=status,
            percent=percent,
            overall_percent=overall_percent,
            timestamp=self.clock(),
            detail=detail,
            event_type=self.event_type,
        )
        self.states[mechanism] = MechanismState(
            mechanism=mechanism,
            status=status,
            percent=percent,
            detail=detail or {},
        )
        self.events.append(event)
        if advance:
            self.counter.increment(advance)
        if self.emitter:
            self.emitter(event)
        return event

    def increment(self, amount: int = 1) -> None:
        self.counter.increment(amount)

    def complete(
        self,
        mechanism: str,
        *,
        overall_percent: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ProgressEvent:
        return self.update(
            mechanism,
            status="completed",
            overall_percent=overall_percent,
            detail=detail,
        )

    def skip_cached(
        self,
        mechanism: str,
        *,
        overall_percent: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ProgressEvent:
        return self.update(
            mechanism,
            status="skipped_or_cached",
            overall_percent=overall_percent,
            detail=detail,
        )

    def fail(
        self,
        mechanism: str,
        *,
        overall_percent: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ProgressEvent:
        return self.update(
            mechanism,
            status="failed",
            overall_percent=overall_percent,
            detail=detail,
        )

    def debug_info(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "counter": self.counter.value,
            "last_sequence": self._sequence,
            "overall_percent": self.aggregator.overall_percent(self.states),
            "mechanisms": {
                name: {
                    "status": state.status,
                    "percent": state.percent,
                    "detail": state.detail,
                }
                for name, state in sorted(self.states.items())
            },
        }


@dataclass
class ProgressHeartbeat:
    counter: ProgressCounter
    interval_sec: float = 30.0
    max_stalled_intervals: int = 3
    clock: Callable[[], float] = time.monotonic
    last_value: int = 0
    last_change_sec: float = 0.0
    stopped: bool = False

    def start(self) -> ProgressHeartbeat:
        now = self.clock()
        self.last_value = self.counter.value
        self.last_change_sec = now
        self.stopped = False
        return self

    def check(self) -> None:
        if self.stopped:
            return
        now = self.clock()
        if self.counter.value != self.last_value:
            self.last_value = self.counter.value
            self.last_change_sec = now
            return
        stalled_for = now - self.last_change_sec
        if stalled_for >= self.interval_sec * self.max_stalled_intervals:
            self.stop()
            raise MediaIngestError(
                "E_STUCK",
                "progress",
                "pipeline progress stalled",
                {
                    "stalled_for_sec": round(stalled_for, 3),
                    "progress_value": self.counter.value,
                },
            )

    def stop(self) -> None:
        self.stopped = True
        self.counter.stopped = True
