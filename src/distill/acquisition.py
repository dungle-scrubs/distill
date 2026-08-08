"""Effectful acquisition: lease, download, validation, promotion.

This module owns the **acquisition lease** (via ``flock`` on an
``ExclusiveLock``), staging a YouTube **source** download, validating the
produced media, and promoting it onto its immutable path with a single
``os.replace`` (R-35). The **acquisition lease** is held for the media file's
whole read lifetime, not only for the download (R-36), so the lease is handed
back inside an ``AcquiredSource`` and released by the caller when reading is
finished. Exclusion is the kernel's; a filesystem that cannot take the lock
fails closed with ``E_LOCK_UNSUPPORTED``.

Vocabulary per ``CONTEXT.md``: **source**, **source fingerprint**,
**lock key**, **bundle key**, **acquisition lease**, **staging directory**,
**bundle**, **generation**.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sys
import time as _real_time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


def _time_module():  # noqa: D401  hook for `distill.source` monkeypatch in tests
    """Return the ``time`` module the façade currently exposes, if patched.

    Tests monkeypatch ``distill.source.time`` to a fake clock (``BudgetClock``,
    ``ShiftedClock``) to advance the monotonic clock without sleeping. The
    **acquisition lease** wait loop lives here after the split, so the patch
    must be observed here as well. A direct ``import time`` would keep the
    original module and the fake wait would never fire. This indirection reads
    the façade's ``time`` attribute when it has been replaced, otherwise the
    real ``time``.
    """
    src = sys.modules.get("distill.source")
    if src is not None:
        candidate = getattr(src, "time", None)
        if candidate is not None and candidate is not _real_time:
            return candidate
    return _real_time


# Back-compat alias: `distill.acquisition.time` should mirror the façade's
# `time` for the same reason, so `monkeypatch.setattr(acquisition, "time", ...)`
# and `monkeypatch.setattr(source, "time", ...)` both work.
time = _real_time  # type: ignore[assignment]  overwritten below if façade exists
try:
    # Keep the two modules' `time` names in sync when the façade is already loaded.
    import distill.source as _source_mod  # noqa: F401  circular guard, imported lazily above

    time = getattr(_source_mod, "time", _real_time)  # type: ignore[assignment]
except Exception:
    time = _real_time  # type: ignore[assignment]

from .bundle_store import (  # noqa: E402  import after time-sync shim for test seam
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    ExclusiveLock,
    confined_path,
    ensure_safe_directory,
)
from .errors import DistillError, WarningRecord, warning  # noqa: E402
from .media_inspect import FFPROBE_TIMEOUTS  # noqa: E402
from .progress import ProgressReporter  # noqa: E402
from .run_command import CommandResult, run_json, stream  # noqa: E402
from .source_identity import youtube_lock_key as _identity_lock_key  # noqa: E402
from .youtube import (  # noqa: E402
    NO_PLAYLIST_ARG,
    YTDLP_DOWNLOAD_TIMEOUTS,
    YTDLP_SOCKET_TIMEOUT_SEC,
)

LOGGER = logging.getLogger("distill.source")

STAGING_DIR_NAME = "_youtube_staging"
MEDIA_DIR_NAME = "_youtube_sources"
LOCK_DIR_NAME = "_youtube_locks"
PROMOTED_MEDIA_STEM = "source"
MEDIA_CONTAINER_PREFERENCE = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
YOUTUBE_DISK_FLOOR_BYTES = 1024 * 1024 * 1024

BYTE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}

ACQUISITION_EVENT_TYPE = "distill.source"


def _acquisition_log(event: str, **detail: Any) -> None:
    """Emit one acquisition event: lease taken or released, verdict, promotion."""
    LOGGER.debug(
        json.dumps(
            {
                "type": ACQUISITION_EVENT_TYPE,
                "event": event,
                "detail": {"pid": os.getpid(), **detail},
            },
            sort_keys=True,
        )
    )


@dataclass
class AcquisitionLease:
    """The exclusive right to acquire and read one **source**'s media.

    Keyed by **lock key**, so it identifies the **source** and not the
    combination of **source** and options: two runs with different options
    contend for the same lease, which is exactly what stops the second from
    replacing media the first is still reading (R-36).

    Held until ``release``, which the reader calls when it is finished with the
    media — not when the download ends. ``release`` is idempotent.
    """

    lock_key: str
    lock_path: Path
    lock: ExclusiveLock
    warnings: list[WarningRecord] = field(default_factory=list)

    @property
    def released(self) -> bool:
        return self.lock.released

    @classmethod
    def take(cls, lock_key: str, lock_path: Path) -> AcquisitionLease | None:
        """Take the lease, or report ``None`` if another run holds it."""
        lock = ExclusiveLock.take(
            lock_key,
            lock_path,
            stage="youtube",
            message="filesystem cannot lock the YouTube source directory",
        )
        if lock is None:
            return None
        return cls(lock_key=lock_key, lock_path=lock_path, lock=lock)

    def release(self) -> None:
        """Give the lease up by closing the descriptor the kernel locked."""
        if self.lock.released:
            return
        self.lock.release()
        _acquisition_log(
            "lease_released",
            lock_key=self.lock_key,
            lock_path=str(self.lock_path),
        )


@dataclass(frozen=True)
class AcquiredSource:
    """A promoted media file and the **acquisition lease** that keeps it readable.

    The lease travels with the path because the two have the same lifetime: a
    caller holding this holds the right to read ``path``, and the moment it
    releases the lease another run may promote a replacement (R-36).
    """

    path: Path
    lease: AcquisitionLease
    warnings: list[WarningRecord] = field(default_factory=list)


class YouTubeDownloaderProtocol(Protocol):
    def acquire(
        self,
        url: str,
        lock_key: str,
        progress: ProgressReporter | None = None,
    ) -> AcquiredSource: ...


def youtube_lock_key(video_id: str) -> str:
    """The **lock key** for a YouTube **source** (delegated to pure identity)."""
    return _identity_lock_key(video_id)


def check_disk_floor(path: Path) -> None:
    usage = shutil.disk_usage(path)
    if usage.free < YOUTUBE_DISK_FLOOR_BYTES:
        raise DistillError(
            "E_DISK_SPACE",
            "youtube",
            "at least 1 GB free disk space is required",
            {"free_bytes": usage.free},
        )


def parse_byte_amount(value: str, unit: str) -> int:
    multiplier = BYTE_UNITS.get(unit)
    if multiplier is None:
        raise ValueError(f"unknown byte unit: {unit}")
    return int(float(value) * multiplier)


def parse_ytdlp_progress(line: str) -> dict[str, float | int] | None:
    if "[download]" not in line:
        return None
    result: dict[str, float | int] = {}
    percent = re.search(r"(?P<percent>\d+(?:\.\d+)?)%", line)
    if percent:
        result["percent"] = float(percent.group("percent"))

    downloaded = re.search(
        r"(?P<downloaded>\d+(?:\.\d+)?)\s*(?P<downloaded_unit>[KMGT]?i?B)\s+of\s+"
        r"(?P<total>\d+(?:\.\d+)?)\s*(?P<total_unit>[KMGT]?i?B)",
        line,
    )
    if downloaded:
        result["downloaded_bytes"] = parse_byte_amount(
            downloaded.group("downloaded"),
            downloaded.group("downloaded_unit"),
        )
        result["total_bytes"] = parse_byte_amount(
            downloaded.group("total"),
            downloaded.group("total_unit"),
        )
        if "percent" not in result and result["total_bytes"]:
            result["percent"] = (
                float(result["downloaded_bytes"]) / float(result["total_bytes"])
            ) * 100
    return result or {"indeterminate": 1}


def _container_rank(path: Path) -> tuple[int, str]:
    suffix = path.suffix.lower()
    if suffix in MEDIA_CONTAINER_PREFERENCE:
        return (MEDIA_CONTAINER_PREFERENCE.index(suffix), suffix)
    return (len(MEDIA_CONTAINER_PREFERENCE), suffix)


def select_downloaded_media(staging_dir: Path) -> Path:
    """Pick the completed container a download produced, deterministically (R-37)."""
    entries = sorted(staging_dir.iterdir()) if staging_dir.is_dir() else []
    candidates = [
        entry
        for entry in entries
        if entry.is_file() and not entry.is_symlink() and entry.stem == PROMOTED_MEDIA_STEM
    ]
    if not candidates:
        raise DistillError(
            "E_YTDLP",
            "youtube",
            "yt-dlp did not produce a source file",
            {"staging_dir": str(staging_dir), "produced": [entry.name for entry in entries]},
        )
    return min(candidates, key=_container_rank)


def _reject_media(path: Path, reason: str, message: str, **detail: Any) -> DistillError:
    """Record a rejection verdict and build the error that carries it."""
    _acquisition_log("media_validated", path=str(path), verdict="rejected", reason=reason, **detail)
    return DistillError("E_BAD_MEDIA", "youtube", message, {"path": str(path), **detail})


def _probed_codec_types(probe: Any) -> list[str]:
    streams = probe.get("streams") if isinstance(probe, dict) else None
    if not isinstance(streams, list):
        return []
    return [
        str(entry.get("codec_type"))
        for entry in streams
        if isinstance(entry, dict) and entry.get("codec_type")
    ]


def _probed_duration_sec(probe: Any) -> float:
    container = probe.get("format") if isinstance(probe, dict) else None
    value = container.get("duration") if isinstance(container, dict) else None
    if not isinstance(value, (str, int, float)):
        return 0.0
    try:
        duration = float(value)
    except ValueError:
        return 0.0
    return duration if math.isfinite(duration) else 0.0


def validate_media_file(path: Path) -> list[WarningRecord]:
    """Confirm a staged file is the media Distill asked for, before promoting it."""
    size = path.stat().st_size if path.is_file() else 0
    if size == 0:
        raise _reject_media(path, "empty_file", "downloaded source file is empty")
    try:
        probe, probe_warnings = run_json(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type:format=duration",
                "-of",
                "json",
                str(path),
            ],
            stage="youtube",
            total_timeout_sec=FFPROBE_TIMEOUTS.total_sec,
            idle_timeout_sec=FFPROBE_TIMEOUTS.idle_sec,
        )
    except DistillError:
        _acquisition_log("media_validated", path=str(path), verdict="rejected", reason="unreadable")
        raise
    codec_types = _probed_codec_types(probe)
    if "video" not in codec_types:
        raise _reject_media(
            path,
            "no_video_stream",
            "downloaded source file carries no video stream",
            codec_types=codec_types,
        )
    duration_sec = _probed_duration_sec(probe)
    if duration_sec <= 0:
        raise _reject_media(
            path,
            "no_duration",
            "downloaded source file reports no playable duration",
            duration_sec=duration_sec,
        )
    _acquisition_log(
        "media_validated",
        path=str(path),
        verdict="accepted",
        size_bytes=size,
        codec_types=codec_types,
        duration_sec=duration_sec,
    )
    return list(probe_warnings)


def promote_media(produced: Path, media_dir: Path, *, root: Path) -> Path:
    """Move a validated file onto its immutable path in one indivisible step.

    ``os.replace`` on one filesystem is the whole promotion: a reader either
    sees the previous media or this one, never a half-written file.
    """
    ensure_safe_directory(media_dir, root)
    produced = confined_path(produced, root)
    promoted = confined_path(media_dir / produced.name, root)
    os.replace(produced, promoted)
    _acquisition_log(
        "media_promoted",
        source=str(produced),
        path=str(promoted),
        size_bytes=promoted.stat().st_size,
    )
    return promoted


def release_acquisition_lease(source: Any, *, during: BaseException | None = None) -> None:
    """Release the lease a **source** carries, if it carries one."""
    lease = getattr(source, "acquisition_lease", None)
    if not isinstance(lease, AcquisitionLease):
        return
    if during is None:
        lease.release()
        return
    try:
        lease.release()
    except Exception as release_failure:
        _acquisition_log(
            "lease_release_failed",
            lock_key=lease.lock_key,
            lock_path=str(lease.lock_path),
            error=repr(release_failure),
            during=type(during).__name__,
        )


class YoutubeDownloader:
    def __init__(
        self,
        output_root: Path,
        *,
        lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
        lock_poll_sec: float = 0.25,
        lock_warn_after_sec: float = 5.0,
    ) -> None:
        """Acquire one remote **source** under an **acquisition lease**."""
        self.output_root = output_root
        self.lock_wait_sec = lock_wait_sec
        self.lock_poll_sec = lock_poll_sec
        self.lock_warn_after_sec = lock_warn_after_sec

    def acquire(
        self,
        url: str,
        lock_key: str,
        progress: ProgressReporter | None = None,
    ) -> AcquiredSource:
        """Stage a download, validate what it produced, and promote it (R-35).

        The **acquisition lease** is returned rather than released: the caller
        reads the media under it and releases it when finished (R-36).

        Helpers are looked up via the ``distill.source`` façade at call time
        so a test that monkeypatches ``distill.source.validate_media_file``
        (or ``promote_media``) sees that patch here. A direct top-level import
        would keep the original object and the patch would have no effect,
        after the split the **acquisition lease** wait loop moved here.
        """
        lease = self._take_lease(lock_key, progress)
        try:
            staging_dir = self._new_staging_dir(lock_key)
            try:
                result = self._download(url, staging_dir, progress)

                def _get_source_attr(name: str, fallback):  # noqa: D401
                    src = sys.modules.get("distill.source")
                    if src is not None:
                        cand = getattr(src, name, None)
                        if cand is not None and cand is not fallback:
                            return cand
                    return fallback

                _select = _get_source_attr("select_downloaded_media", select_downloaded_media)
                _validate = _get_source_attr("validate_media_file", validate_media_file)
                _promote = _get_source_attr("promote_media", promote_media)

                produced = _select(staging_dir)
                validation_warnings = _validate(produced)
                promoted = _promote(produced, self._media_dir(lock_key), root=self.output_root)
            finally:
                self._discard(staging_dir)
            if progress:
                progress.complete("youtube_download", detail={"path": str(promoted)})
        except BaseException:
            lease.release()
            raise
        return AcquiredSource(
            path=promoted,
            lease=lease,
            warnings=[*lease.warnings, *result.warnings, *validation_warnings],
        )

    def _media_dir(self, lock_key: str) -> Path:
        return self.output_root / MEDIA_DIR_NAME / lock_key

    def _discard(self, staging_dir: Path) -> None:
        confined_path(staging_dir, self.output_root)
        shutil.rmtree(staging_dir, ignore_errors=True)

    def _new_staging_dir(self, lock_key: str) -> Path:
        parent = ensure_safe_directory(
            self.output_root / STAGING_DIR_NAME / lock_key, self.output_root
        )
        for stale in sorted(parent.iterdir()):
            confined_path(stale, self.output_root)
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        staging_dir = confined_path(parent / f"{os.getpid()}-{uuid.uuid4().hex}", self.output_root)
        staging_dir.mkdir()
        return staging_dir

    def _take_lease(
        self,
        lock_key: str,
        progress: ProgressReporter | None,
    ) -> AcquisitionLease:
        locks = ensure_safe_directory(self.output_root / LOCK_DIR_NAME, self.output_root)
        lock = locks / f"{lock_key}.lock"
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "lock"})
        lease, lock_warnings = self._acquire(lock_key, lock)
        if lease is None:
            _acquisition_log("lease_denied", lock_key=lock_key, lock_path=str(lock), reason="held")
            raise DistillError("E_LOCKED", "youtube", "YouTube source is locked by another process")
        lease.warnings.extend(lock_warnings)
        _acquisition_log("lease_acquired", lock_key=lock_key, lock_path=str(lock))
        return lease

    def _download(
        self,
        url: str,
        staging_dir: Path,
        progress: ProgressReporter | None,
    ) -> CommandResult:
        out_template = str(staging_dir / f"{PROMOTED_MEDIA_STEM}.%(ext)s")
        command = [
            "yt-dlp",
            NO_PLAYLIST_ARG,
            "-f",
            "best[ext=mp4][height<=720]/best[height<=720]/best[ext=mp4]/best/bv*+ba/b",
            "--newline",
            "--progress",
            "--socket-timeout",
            str(YTDLP_SOCKET_TIMEOUT_SEC),
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "--concurrent-fragments",
            "8",
            "--merge-output-format",
            "mp4",
            "-o",
            out_template,
            "--",
            url,
        ]

        def report(line: str) -> None:
            parsed = parse_ytdlp_progress(line)
            if not parsed or progress is None:
                return
            progress.update(
                "youtube_download",
                percent=(float(parsed["percent"]) if "percent" in parsed else None),
                detail=parsed,
            )

        return stream(
            command,
            stage="youtube",
            total_timeout_sec=YTDLP_DOWNLOAD_TIMEOUTS.total_sec,
            idle_timeout_sec=YTDLP_DOWNLOAD_TIMEOUTS.idle_sec,
            on_stdout_line=report,
            error_code="E_YTDLP",
        )

    def _acquire(
        self,
        lock_key: str,
        lock: Path,
    ) -> tuple[AcquisitionLease | None, list[WarningRecord]]:
        started = _time_module().monotonic()
        warnings: list[WarningRecord] = []
        while True:
            confined_path(lock, self.output_root)
            lease = AcquisitionLease.take(lock_key, lock)
            if lease is not None:
                waited = _time_module().monotonic() - started
                if waited >= self.lock_warn_after_sec:
                    warnings.append(
                        warning(
                            "youtube",
                            "long_lock_wait",
                            f"waited {waited:.1f}s for YouTube source lock",
                        )
                    )
                return lease, warnings
            if _time_module().monotonic() - started >= self.lock_wait_sec:
                return None, warnings
            _time_module().sleep(self.lock_poll_sec)
