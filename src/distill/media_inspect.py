"""Media inspection: probing a source's duration and fingerprinting its bytes.

This module owns the questions Distill asks about a media file or a manifest
without producing anything: how long a source runs (ffprobe), the content
fingerprint that identifies it, and the duration a manifest records. It reaches
no bundle write path; `source` orchestrates it and owns acquisition and the
bundle lifecycle.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import DistillError, WarningRecord
from .progress import ProgressReporter
from .run_command import run_json, silent_tool_timeouts

CONTENT_HASH_LIMIT_BYTES = 5 * 1024 * 1024 * 1024
FINGERPRINT_SAMPLE_BYTES = 64 * 1024
# Interior anchors sampled between the head and tail anchors. The sampled set is
# capped at FINGERPRINT_INTERIOR_ANCHORS + 2 anchors, so the default cache mode
# reads at most 576 KiB no matter how large the source is.
FINGERPRINT_INTERIOR_ANCHORS = 7
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


def fingerprint_anchor_offsets(size: int) -> list[int]:
    """Byte offsets the sampling cache mode reads, derived only from ``size``.

    The head anchor sits at 0 and the tail anchor at ``size - 64 KiB``, with
    ``FINGERPRINT_INTERIOR_ANCHORS`` interior anchors spread evenly between
    them. Offsets depend on nothing but the size, so the same file always yields
    the same anchor set; coinciding offsets collapse, which is why a file barely
    over one sample still reports two anchors rather than nine.
    """
    if size <= FINGERPRINT_SAMPLE_BYTES:
        return [0]
    last = size - FINGERPRINT_SAMPLE_BYTES
    offsets = {0, last}
    for index in range(1, FINGERPRINT_INTERIOR_ANCHORS + 1):
        offsets.add(last * index // (FINGERPRINT_INTERIOR_ANCHORS + 1))
    return sorted(offsets)


def _anchor_label(offset: int, offsets: list[int]) -> str:
    if offset == offsets[0]:
        return "first"
    if offset == offsets[-1]:
        return "last"
    return "interior"


def local_fingerprint(
    path: Path,
    cache_mode: str,
    progress: ProgressReporter | None = None,
) -> str:
    """Return the source fingerprint for a local file under ``cache_mode``.

    ``content`` hashes every byte, so it distinguishes any two files whose
    contents differ at all. It refuses files over 5 GB, because reading one is
    not a cost a cache lookup may impose.

    ``fingerprint`` - the default - samples instead. It hashes the file's size,
    its mtime in nanoseconds, and 64 KiB read at each offset in
    ``fingerprint_anchor_offsets``: the head, the tail, and seven evenly spread
    interior anchors. Each anchor's offset is hashed alongside its bytes, so
    overlapping or reordered anchors cannot alias. Reading is therefore bounded
    at 576 KiB regardless of source size, which is what makes the default cheap
    enough to run on every cache lookup of a file up to 5 GB.

    Sampling is what it sounds like, and the property it buys is worth stating
    plainly: two *distinct* sources collide iff they share a size, an mtime to
    the nanosecond, and every sampled anchor's bytes. Two independently produced
    videos do not do this - differing content gives differing sizes, and mtimes
    are not byte-identical by accident. Constructing such a pair on purpose is
    entirely possible, because the anchor offsets are public and deterministic;
    an attacker who can write both files can leave every anchor untouched and
    differ everywhere else. A colliding pair shares a bundle key, so one source
    is served the other's bundle. Where sources are untrusted or adversarial,
    ``content`` is the cache mode that removes the property; the default trades
    it for bounded cost. See R-38, R-51.
    """
    stat = path.stat()
    if cache_mode == "content":
        if stat.st_size > CONTENT_HASH_LIMIT_BYTES:
            raise DistillError(
                "E_CONTENT_HASH_TOO_LARGE",
                "source",
                "content cache mode refuses files over 5 GB",
                {"size_bytes": stat.st_size},
            )
        digest = hashlib.sha256()
        bytes_read = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
                if progress:
                    progress.update(
                        "source_fingerprint",
                        percent=(bytes_read / max(1, stat.st_size)) * 100,
                        detail={
                            "cache_mode": "content",
                            "bytes_read": bytes_read,
                            "total_bytes": stat.st_size,
                        },
                    )
        if progress and stat.st_size == 0:
            progress.complete(
                "source_fingerprint",
                detail={"cache_mode": "content", "total_bytes": 0},
            )
        return digest.hexdigest()

    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    offsets = fingerprint_anchor_offsets(stat.st_size)
    sample_total = len(offsets)
    sample_done = 0
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            # The offset is hashed with its bytes so two anchors reading the same
            # region, or the same regions in another order, cannot alias.
            digest.update(str(offset).encode())
            digest.update(handle.read(FINGERPRINT_SAMPLE_BYTES))
            sample_done += 1
            if progress:
                progress.update(
                    "source_fingerprint",
                    percent=(sample_done / sample_total) * 100,
                    detail={
                        "cache_mode": "fingerprint",
                        "sample": _anchor_label(offset, offsets),
                        "offset": offset,
                        "samples_done": sample_done,
                        "samples_total": sample_total,
                    },
                )
    return digest.hexdigest()


def source_hash(source_fingerprint: str, opts_hash: str) -> str:
    return hashlib.sha256(f"{source_fingerprint}:{opts_hash}".encode()).hexdigest()


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
