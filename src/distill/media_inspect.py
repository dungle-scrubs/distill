"""Read source duration with ffprobe and duration recorded in a manifest.

Source fingerprints and bundle keys belong to source_identity. This module
performs media inspection without acquiring a source or writing a bundle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import DistillError, WarningRecord
from .run_command import run_json, silent_tool_timeouts

FFPROBE_TIMEOUTS = silent_tool_timeouts(60.0)


def probe_duration(path: Path) -> tuple[float, list[WarningRecord]]:
    """The source's duration, with any **warning** the probe itself recorded.

    The warnings are truncated capture (R-33). They are returned rather than
    dropped because this is the only place they exist, and a caller that took
    the duration alone would publish a **bundle** that never mentions the loss.
    """
    data, probe_warnings = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        stage="source",
        total_timeout_sec=FFPROBE_TIMEOUTS.total_sec,
        idle_timeout_sec=FFPROBE_TIMEOUTS.idle_sec,
    )
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DistillError("E_FFPROBE", "source", "could not read video duration") from exc
    return duration, list(probe_warnings)


def ensure_duration_allowed(duration_sec: float, max_duration_sec: float) -> None:
    """Refuse a **source** whose claimed duration is unusable or over the cap (R-47).

    Two refusals, and they are not the same kind. The cap is the operator's
    policy about a source that is genuinely too long. Usability is about the
    number itself: a duration arrives from ffprobe, so it is data from outside
    rather than operator error, and it is refused as `E_BAD_MEDIA` the way the
    acquisition path refuses media it cannot use.

    NaN is why this exists. It clears `> max_duration_sec` and it clears
    `<= 0`, so the cap silently stopped existing and every window, interval and
    percentage computed from the duration afterwards was NaN too.
    """
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise DistillError(
            "E_BAD_MEDIA",
            "source",
            "source reports an unusable duration",
            # Text, because a fatal error is published as JSON and a bare NaN
            # is not JSON a strict reader will parse.
            {"duration_sec": repr(duration_sec)},
        )
    if duration_sec > max_duration_sec:
        raise DistillError(
            "E_DURATION_CAP",
            "source",
            "video exceeds max_duration_sec",
            {"duration_sec": duration_sec, "max_duration_sec": max_duration_sec},
        )


def manifest_duration(manifest: Mapping[str, Any]) -> float | None:
    """The duration a **manifest** records, if it records a usable one.

    A manifest is a document another process wrote, so the number in it is input
    rather than a fact: `None` covers a missing field, a field that is not a
    number, and one that is a number no source can have. Every caller that
    reuses a published duration asks here, so "what makes a recorded duration
    usable" has one answer rather than one per cache path.

    `bool` is refused explicitly, because `True` is an `int` in Python and a
    manifest saying `"duration_sec": true` is not a one-second video.

    The conversion itself is guarded, because JSON has no integer ceiling and
    Python's `int` has none either: a manifest recording a 401-digit number is
    a document this function has to answer about, and `float()` on it raises
    `OverflowError`. An unusable number is a **cache miss**, whichever way it
    is unusable.
    """
    duration = manifest.get("duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return None
    try:
        duration = float(duration)
    except OverflowError:
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration
