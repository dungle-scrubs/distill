"""Saccade progress aliases for shared media-ingest progress helpers."""

from __future__ import annotations

from .errors import SaccadeError
from .media.progress import (
    PROGRESS_STATUSES,
    TERMINAL_PROGRESS_STATUSES,
    MechanismState,
    MechanismWeight,
    ProgressCounter,
    ProgressEvent,
    ProgressStatus,
    clamp_percent,
    mechanism_percent,
)
from .media.progress import (
    OverallProgressAggregator as _SharedOverallProgressAggregator,
)
from .media.progress import (
    ProgressHeartbeat as _SharedProgressHeartbeat,
)
from .media.progress import (
    ProgressReporter as _SharedProgressReporter,
)

DEFAULT_MECHANISM_WEIGHTS: tuple[MechanismWeight, ...] = (
    MechanismWeight("cache_lookup", 1.0),
    MechanismWeight("source_fingerprint", 4.0),
    MechanismWeight("youtube_download", 15.0),
    MechanismWeight("duration_probe", 1.0),
    MechanismWeight("audio_extraction", 8.0),
    MechanismWeight("transcription", 35.0),
    MechanismWeight("scene_detection", 4.0),
    MechanismWeight("frame_extraction", 12.0),
    MechanismWeight("ocr", 10.0),
    MechanismWeight("redaction", 3.0),
    MechanismWeight("local_vision", 10.0),
    MechanismWeight("rendering", 4.0),
    MechanismWeight("bundle_publish", 3.0),
)


class OverallProgressAggregator(_SharedOverallProgressAggregator):
    def __init__(
        self,
        weights: tuple[MechanismWeight, ...] = DEFAULT_MECHANISM_WEIGHTS,
    ) -> None:
        super().__init__(weights=weights)


class ProgressReporter(_SharedProgressReporter):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("event_type", "saccade.progress")
        kwargs.setdefault("aggregator", OverallProgressAggregator())
        super().__init__(*args, **kwargs)


class ProgressHeartbeat(_SharedProgressHeartbeat):
    def check(self) -> None:
        try:
            super().check()
        except Exception as exc:
            if getattr(exc, "code", None) == "E_STUCK":
                raise SaccadeError(
                    exc.code,
                    exc.stage,
                    exc.message,
                    exc.details,
                ) from exc
            raise


__all__ = [
    "DEFAULT_MECHANISM_WEIGHTS",
    "PROGRESS_STATUSES",
    "TERMINAL_PROGRESS_STATUSES",
    "MechanismState",
    "MechanismWeight",
    "OverallProgressAggregator",
    "ProgressCounter",
    "ProgressEvent",
    "ProgressHeartbeat",
    "ProgressReporter",
    "ProgressStatus",
    "clamp_percent",
    "mechanism_percent",
]
