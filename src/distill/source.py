"""Source acquisition and fingerprinting for Distill.

This module owns local path resolution, duration probing, safe output root
validation, YouTube id lookup, the acquisition of a remote **source**, disk
checks, and **source fingerprints**.

Acquiring a remote source is staging, validation and promotion, in that order
(R-35). The download lands in a staging directory unique to the run; the media
file it produced is selected deterministically (R-37) and validated as the
expected media; only then is it promoted onto the immutable path with a single
`os.replace`. Nothing writes into the promoted path, and nothing clears it: the
previously promoted media stays readable until a proven replacement takes its
place atomically, so a download that dies mid-transfer costs a retry rather than
the only good copy.

The **acquisition lease** is held for the media file's whole read lifetime, not
only for the download (R-36). Two runs of one video under different options
share a **lock key** and not a **bundle key**, so a lease that ended with the
download would leave the second run free to write over media the first is still
reading. `acquire` therefore hands the lease back to its caller inside an
`AcquiredSource`, and the caller releases it when it is finished reading - see
`pipeline.process_resolved_source`.

It does not write final bundle artifacts, own bundle-level locking (that is
keyed by **bundle key** and belongs to the bundle store; the lease here is keyed
by **lock key** and answers only "is another run fetching this source?"), decide
what a **generation** contains, or install anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

from .errors import DistillError, warning
from .options import DistillOptions
from .progress import ProgressReporter
from .run_command import CommandResult, CommandTimeouts, run, run_json, stream

LOGGER = logging.getLogger(__name__)

CONTENT_HASH_LIMIT_BYTES = 5 * 1024 * 1024 * 1024
FINGERPRINT_SAMPLE_BYTES = 64 * 1024
# Interior anchors sampled between the head and tail anchors. The sampled set is
# capped at FINGERPRINT_INTERIOR_ANCHORS + 2 anchors, so the default cache mode
# reads at most 576 KiB no matter how large the source is.
FINGERPRINT_INTERIOR_ANCHORS = 7
YOUTUBE_DISK_FLOOR_BYTES = 1024 * 1024 * 1024
# Wall-clock ceilings so a wedged tool or a stalled network call cannot hang the
# whole run. yt-dlp additionally gets `--socket-timeout` so it aborts a stalled
# connection on its own rather than blocking until the outer timeout fires.
FFPROBE_TIMEOUTS = CommandTimeouts(total_sec=60.0, idle_sec=30.0)
YTDLP_METADATA_TIMEOUTS = CommandTimeouts(total_sec=120.0, idle_sec=60.0)
# A download is bounded by silence, not by length (R-30): a legitimate multi-GB
# fetch on a slow link may run for hours, while a wedged one stops emitting
# progress within seconds. The total is a backstop against a tool that reports
# progress forever without finishing.
YTDLP_DOWNLOAD_TIMEOUTS = CommandTimeouts(total_sec=6 * 60 * 60.0, idle_sec=120.0)
YTDLP_SOCKET_TIMEOUT_SEC = 30
# Where a run assembles a download, and where a proven one is promoted to. They
# are siblings under one output root so promotion is a rename on one filesystem
# rather than a copy across two, and so the promoted directory holds nothing but
# promoted media.
STAGING_DIR_NAME = "_youtube_staging"
MEDIA_DIR_NAME = "_youtube_sources"
LOCK_DIR_NAME = "_youtube_locks"
# The stem yt-dlp is told to write, and so the only stem a completed download
# has. A format fragment is `source.f140.m4a`, whose stem is `source.f140`, and
# an in-flight file is `source.mp4.part`, whose stem is `source.mp4`: matching
# the stem exactly is what separates the merged container from both (R-37).
PROMOTED_MEDIA_STEM = "source"
# Preference order when a staging directory somehow holds more than one complete
# container. Order is fixed rather than alphabetical so the choice is a stated
# preference; anything unlisted sorts after everything listed, by suffix.
MEDIA_CONTAINER_PREFERENCE = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SENSITIVE_COMPONENTS = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".config/1password",
    "library/keychains",
}
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}

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
    """Emit one acquisition event: lease taken or released, verdict, promotion.

    Metadata only, in the shape `run_command` uses for its boundary event, so
    one log stream answers both "which tool ran" and "what did acquisition do
    with what it produced". Paths are Distill's own; no **extracted text** is
    recorded here, because none of it has passed a **redaction sink**.
    """
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

    Keyed by **lock key**, so it identifies the source and not the combination
    of source and options: two runs with different options contend for the same
    lease, which is exactly what stops the second from replacing media the first
    is still reading (R-36).

    Held until `release`, which the reader calls when it is finished with the
    media - not when the download ends. `release` is idempotent, because the
    failure paths that release a lease early overlap with the caller's own
    cleanup.
    """

    lock_key: str
    lock_path: Path
    warnings: list[dict[str, str]] = field(default_factory=list)
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.lock_path.unlink(missing_ok=True)
        _acquisition_log(
            "lease_released",
            lock_key=self.lock_key,
            lock_path=str(self.lock_path),
        )


@dataclass(frozen=True)
class AcquiredSource:
    """A promoted media file and the lease that keeps it readable.

    The lease travels with the path because the two have the same lifetime: a
    caller holding this holds the right to read `path`, and the moment it
    releases the lease another run may promote a replacement.
    """

    path: Path
    lease: AcquisitionLease
    warnings: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SourceInfo:
    source_type: str
    resolved_path: Path
    duration_sec: float
    source_fingerprint: str
    source_hash: str
    warnings: list[dict[str, str]]
    youtube_video_id: str | None = None
    youtube_lock_key: str | None = None
    related_links: list[dict[str, str]] | None = None
    # Present only for a source this run acquired. A cache hit reads no media
    # and holds no lease; a local source has nothing to lease.
    acquisition_lease: AcquisitionLease | None = None


def release_acquisition_lease(source: Any) -> None:
    """Release the lease a source carries, if it carries one.

    Takes anything with the attribute rather than a `SourceInfo`, because the
    reader that calls this handles local sources, cache hits and test doubles
    through the same path, and none of those hold a lease.
    """
    lease = getattr(source, "acquisition_lease", None)
    if isinstance(lease, AcquisitionLease):
        lease.release()


@dataclass(frozen=True)
class SourceRequest:
    value: str
    options: DistillOptions
    output_root: Path | None = None
    progress: ProgressReporter | None = None


@dataclass(frozen=True)
class SourceResolution:
    source: SourceInfo
    output_root: Path | None
    progress: ProgressReporter | None = None


@dataclass(frozen=True)
class YouTubeMetadata:
    video_id: str
    description: str
    warnings: list[dict[str, str]]


class YouTubeDownloaderProtocol(Protocol):
    def acquire(
        self,
        url: str,
        lock_key: str,
        progress: ProgressReporter | None = None,
    ) -> AcquiredSource: ...


def probe_duration(path: Path) -> float:
    data = run_json(
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
    return duration


def ensure_duration_allowed(duration_sec: float, max_duration_sec: float) -> None:
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


class LocalSourceProvider:
    def resolve(self, request: SourceRequest) -> SourceInfo:
        path_text = request.value
        options = request.options
        progress = request.progress
        if not path_text:
            raise DistillError("E_BAD_SOURCE", "source", "path is required")
        original = Path(path_text).expanduser()
        if not original.exists() or not original.is_file():
            raise DistillError(
                "E_BAD_SOURCE", "source", "local video does not exist", {"path": path_text}
            )
        resolved = original.resolve()
        warnings: list[dict[str, str]] = []
        if resolved != original.absolute():
            warnings.append(
                warning(
                    "source",
                    "symlink_resolved",
                    f"source path resolved to {resolved}",
                )
            )
        if progress:
            progress.update("duration_probe", status="running")
        duration = probe_duration(resolved)
        if progress:
            progress.complete("duration_probe", detail={"duration_sec": duration})
        ensure_duration_allowed(duration, options.max_duration_sec)
        fingerprint = local_fingerprint(resolved, options.cache_mode, progress)
        opts_hash = options.opts_hash("local")
        return SourceInfo(
            source_type="local",
            resolved_path=resolved,
            duration_sec=duration,
            source_fingerprint=fingerprint,
            source_hash=source_hash(fingerprint, opts_hash),
            warnings=warnings,
        )


def resolve_local_source(
    path_text: str,
    options: DistillOptions,
    progress: ProgressReporter | None = None,
) -> SourceInfo:
    return LocalSourceProvider().resolve(SourceRequest(path_text, options, progress=progress))


def validate_output_root(output_dir: str | None) -> Path:
    root = Path(output_dir).expanduser() if output_dir else Path.home() / ".cache" / "distill"
    root = root.resolve()
    home = Path.home().resolve()
    temp = Path(tempfile.gettempdir()).resolve()
    if not (root in (home, temp) or root.is_relative_to(home) or root.is_relative_to(temp)):
        raise DistillError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "output_dir must be under $HOME or the system temp directory",
            {"output_dir": str(root)},
        )

    sensitive = sensitive_path_match(root)
    if sensitive:
        raise DistillError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "output_dir points inside a sensitive path",
            {"output_dir": str(root), "matched": sensitive},
        )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise DistillError("E_BAD_OUTPUT_DIR", "bundle", "output_dir must not be a symlink")
    return root


def sensitive_path_match(
    root: Path,
    *,
    case_sensitive: bool | None = None,
) -> str | None:
    if case_sensitive is None:
        case_sensitive = sys.platform != "darwin"
    root_text = str(root) if case_sensitive else str(root).lower()
    for sensitive in SENSITIVE_COMPONENTS:
        candidate = sensitive if case_sensitive else sensitive.lower()
        if candidate in root_text:
            return sensitive
    return None


YOUTUBE_STRIP_QUERY_KEYS = {"t", "start", "end", "time_continue"}


def normalize_youtube_url(url: str) -> str:
    """Drop player-state query params (e.g. `t=900s`) so the full video is consumed."""
    parsed = urlparse(url)
    if parsed.netloc.lower() not in YOUTUBE_HOSTS:
        return url
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    kept = [
        (k, v)
        for k, v in pairs
        if k not in YOUTUBE_STRIP_QUERY_KEYS
    ]
    if len(kept) == len(pairs):
        return url
    new_query = urlencode(kept)
    fragment = "" if parsed.fragment.startswith("t=") else parsed.fragment
    return urlunparse(parsed._replace(query=new_query, fragment=fragment))


def ensure_youtube_host(url: str) -> None:
    """Reject non-YouTube hosts (and option-injection values) before yt-dlp runs.

    Playlist/channel URLs carry no video id, so this host-only check is the guard
    the playlist path uses in place of ``parse_youtube_url``.
    """
    if urlparse(url).netloc.lower() not in YOUTUBE_HOSTS:
        raise DistillError("E_BAD_URL", "youtube", "only YouTube URLs are supported", {"url": url})


def parse_youtube_url(url: str) -> str:
    ensure_youtube_host(url)
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif any(parsed.path.startswith(prefix) for prefix in ("/shorts/", "/embed/", "/live/", "/v/")):
        video_id = parsed.path.strip("/").split("/")[1]
    else:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    if not video_id:
        raise DistillError(
            "E_BAD_URL",
            "youtube",
            f"could not find YouTube video id in URL: {url}",
        )
    return video_id


def _ytdlp_command(extra_args: list[str], url: str) -> list[str]:
    """Build a yt-dlp argv with a stall guard and a `--` terminator.

    The `--` before the URL stops a value that begins with `-` from being parsed
    as a yt-dlp option (argument injection), and `--socket-timeout` lets yt-dlp
    abort a stalled connection on its own.
    """
    return [
        "yt-dlp",
        "--socket-timeout",
        str(YTDLP_SOCKET_TIMEOUT_SEC),
        *extra_args,
        "--",
        url,
    ]


def _run_ytdlp(
    extra_args: list[str],
    url: str,
    *,
    timeouts: CommandTimeouts = YTDLP_METADATA_TIMEOUTS,
) -> CommandResult:
    """Run a non-downloading yt-dlp invocation and hand back what it produced.

    `check=False`: every caller inspects `returncode` itself, because "yt-dlp
    ran and said no" means something different at each one - an unresolvable id
    is fatal, an unreadable description is a **degradation**. A yt-dlp that is
    absent or wedged still raises, since there is no answer to inspect.
    """
    return run(
        _ytdlp_command(extra_args, url),
        stage="youtube",
        total_timeout_sec=timeouts.total_sec,
        idle_timeout_sec=timeouts.idle_sec,
        check=False,
        error_code="E_YTDLP",
    )


def youtube_description(url: str) -> tuple[str, list[dict[str, str]]]:
    try:
        proc = _run_ytdlp(["--skip-download", "--print", "%(description)s"], url)
    except DistillError:
        # Description is best-effort metadata; degrade rather than fail the run.
        return "", [
            warning(
                "youtube",
                "metadata_unavailable",
                "yt-dlp could not read YouTube description",
            )
        ]
    if proc.returncode != 0:
        return "", [
            warning(
                "youtube",
                "metadata_unavailable",
                "yt-dlp could not read YouTube description",
            )
        ]
    return proc.stdout.strip(), []


def youtube_metadata(url: str) -> YouTubeMetadata:
    proc = _run_ytdlp(["--skip-download", "--dump-json"], url)
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {}
        video_id = str(payload.get("id") or "").strip()
        if video_id:
            return YouTubeMetadata(
                video_id=video_id,
                description=str(payload.get("description") or "").strip(),
                warnings=[],
            )

    video_id = canonical_youtube_id(url)
    description, metadata_warnings = youtube_description(url)
    return YouTubeMetadata(video_id=video_id, description=description, warnings=metadata_warnings)


def canonical_youtube_id(url: str) -> str:
    parse_youtube_url(url)
    proc = _run_ytdlp(["--simulate", "--print", "id"], url)
    if proc.returncode != 0:
        raise DistillError(
            "E_YTDLP",
            "youtube",
            "yt-dlp could not resolve video id",
            {"stderr": proc.stderr.strip()},
        )
    video_id = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not video_id:
        raise DistillError("E_YTDLP", "youtube", "yt-dlp returned an empty video id")
    return video_id


def youtube_lock_key(video_id: str) -> str:
    return hashlib.sha256(video_id.encode()).hexdigest()


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
    """Pick the completed container a download produced, deterministically (R-37).

    An interrupted merge leaves the per-format fragments (`source.f140.m4a`,
    `source.f299.mp4`) and an in-flight `source.mp4.part` beside the container
    yt-dlp was asked for. Taking the first entry of a glob picks whichever of
    those sorts first, which is how a run ends up transcribing an audio fragment
    of a previous download - finding 16.

    The rule instead: a completed container is a regular file whose stem is
    exactly `source`, which every fragment and part-file fails by construction.
    Among those, the fixed container preference decides, and the suffix breaks
    any remaining tie, so the same staging directory always yields the same
    choice. Selecting is not trusting: the winner is still validated before it
    is promoted.
    """
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
    _acquisition_log(
        "media_validated", path=str(path), verdict="rejected", reason=reason, **detail
    )
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
        return float(value)
    except ValueError:
        return 0.0


def validate_media_file(path: Path) -> None:
    """Confirm a staged file is the media Distill asked for, before promoting it.

    A **fatal error**, not a **degradation** (ADR-0002): the media file is the
    input every later stage reads, so a file that is not playable video leaves
    no reduced-but-useful **bundle** to produce - there is no **transcript** and
    no **keyframe** without it. That makes usable media a **required
    capability** of the run rather than an optional one, and R-34 admits no
    third answer. Nothing here degrades; it promotes or it raises.

    Three questions, all answered by one ffprobe: does the file have content,
    does it carry a video stream, and does it have a playable duration. An
    audio-only format fragment fails the second and a truncated container fails
    the third, so neither can be promoted over media that answered all three.
    This is a separate probe from `probe_duration`, which reads the *promoted*
    path: the point of asking here is to ask before promotion.
    """
    size = path.stat().st_size if path.is_file() else 0
    if size == 0:
        raise _reject_media(path, "empty_file", "downloaded source file is empty")
    try:
        probe = run_json(
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
        # ffprobe could not read it at all, which is its own answer. The error it
        # raised already names the tool and the failure, so it travels as-is.
        _acquisition_log(
            "media_validated", path=str(path), verdict="rejected", reason="unreadable"
        )
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


def promote_media(produced: Path, media_dir: Path) -> Path:
    """Move a validated file onto its immutable path in one indivisible step.

    `os.replace` on one filesystem is the whole promotion: a reader either sees
    the previous media or this one, never a half-written file, and the previous
    media is not removed before its replacement exists. Copying, truncating or
    clearing the directory first would each reintroduce the window RV-3
    describes, in which the only good copy is already gone when the replacement
    turns out not to arrive.
    """
    media_dir.mkdir(parents=True, exist_ok=True)
    promoted = media_dir / produced.name
    os.replace(produced, promoted)
    _acquisition_log(
        "media_promoted",
        source=str(produced),
        path=str(promoted),
        size_bytes=promoted.stat().st_size,
    )
    return promoted


class YoutubeDownloader:
    def __init__(
        self,
        output_root: Path,
        stale_sec: float = 180.0,
        lock_wait_sec: float = 0.0,
        lock_poll_sec: float = 0.25,
        lock_warn_after_sec: float = 5.0,
    ) -> None:
        self.output_root = output_root
        self.stale_sec = stale_sec
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

        The lease is returned rather than released: the caller reads the media
        under it and releases it when finished (R-36). Every path out of here
        that does not return an `AcquiredSource` releases the lease itself, so
        a failure never strands one.
        """
        lease = self._take_lease(lock_key, progress)
        try:
            staging_dir = self._new_staging_dir(lock_key)
            try:
                result = self._download(url, staging_dir, progress)
                produced = select_downloaded_media(staging_dir)
                validate_media_file(produced)
                promoted = promote_media(produced, self._media_dir(lock_key))
            finally:
                # The staging directory is scratch: whatever survived the run -
                # a rejected file, an unmerged fragment, a `.part` - is
                # discarded with it. Nothing under the promoted path is touched.
                shutil.rmtree(staging_dir, ignore_errors=True)
            if progress:
                progress.complete("youtube_download", detail={"path": str(promoted)})
        except BaseException:
            lease.release()
            raise
        return AcquiredSource(
            path=promoted,
            lease=lease,
            warnings=[*lease.warnings, *result.warnings],
        )

    def _media_dir(self, lock_key: str) -> Path:
        return self.output_root / MEDIA_DIR_NAME / lock_key

    def _new_staging_dir(self, lock_key: str) -> Path:
        """A staging directory no other run can be writing into.

        Named by pid and a random token, so two runs of one source - which the
        lease already serializes - and two runs of different sources alike get
        directories that cannot collide. Created under the lease, which is also
        what makes discarding any staging directory left by an earlier run of
        this source safe: no live run can own one.
        """
        parent = self.output_root / STAGING_DIR_NAME / lock_key
        parent.mkdir(parents=True, exist_ok=True)
        for stale in parent.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        staging_dir = parent / f"{os.getpid()}-{uuid.uuid4().hex}"
        staging_dir.mkdir()
        return staging_dir

    def _take_lease(
        self,
        lock_key: str,
        progress: ProgressReporter | None,
    ) -> AcquisitionLease:
        locks = self.output_root / LOCK_DIR_NAME
        locks.mkdir(parents=True, exist_ok=True)
        lock = locks / f"{lock_key}.lock"
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "lock"})
        acquired, lock_warnings = self._acquire(lock)
        if not acquired:
            _acquisition_log(
                "lease_denied", lock_key=lock_key, lock_path=str(lock), reason="held"
            )
            raise DistillError("E_LOCKED", "youtube", "YouTube source is locked by another process")
        _acquisition_log("lease_acquired", lock_key=lock_key, lock_path=str(lock))
        return AcquisitionLease(lock_key=lock_key, lock_path=lock, warnings=lock_warnings)

    def _download(
        self,
        url: str,
        staging_dir: Path,
        progress: ProgressReporter | None,
    ) -> CommandResult:
        """Run yt-dlp with its output template pointed at the staging directory.

        The template names the staging directory and never the promoted path, so
        yt-dlp's habit of opening its destination before the transfer succeeds
        cannot truncate media a previous run proved good (RV-3).
        """
        out_template = str(staging_dir / f"{PROMOTED_MEDIA_STEM}.%(ext)s")
        command = [
            "yt-dlp",
            "-f",
            "best[ext=mp4][height<=720]/best[height<=720]/best[ext=mp4]/best/bv*+ba/b",
            "--newline",
            # The idle timeout is what bounds this download (R-30), and its
            # heartbeat is yt-dlp's progress output. `--progress` keeps that
            # output coming even under a user config that set `--quiet`,
            # which would otherwise starve the idle clock and have a healthy
            # download killed for saying nothing.
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

        # R-32: yt-dlp writes `--newline` download progress to stdout. It is
        # the only stream parsed here; stderr carries its diagnostics, which
        # run_command captures for the failure payload.
        return stream(
            command,
            stage="youtube",
            total_timeout_sec=YTDLP_DOWNLOAD_TIMEOUTS.total_sec,
            idle_timeout_sec=YTDLP_DOWNLOAD_TIMEOUTS.idle_sec,
            on_stdout_line=report,
            error_code="E_YTDLP",
        )

    def _acquire(self, lock: Path) -> tuple[bool, list[dict[str, str]]]:
        started = time.monotonic()
        warnings: list[dict[str, str]] = []
        while True:
            acquired = self._try_acquire_once(lock)
            if acquired:
                waited = time.monotonic() - started
                if waited >= self.lock_warn_after_sec:
                    warnings.append(
                        warning(
                            "youtube",
                            "long_lock_wait",
                            f"waited {waited:.1f}s for YouTube source lock",
                        )
                    )
                return True, warnings
            if time.monotonic() - started >= self.lock_wait_sec:
                return False, warnings
            time.sleep(self.lock_poll_sec)

    def _try_acquire_once(self, lock: Path) -> bool:
        now = time.monotonic()
        payload = {
            "pid": os.getpid(),
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_monotonic": now,
            "last_heartbeat_monotonic": now,
        }
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                current = json.loads(lock.read_text())
                heartbeat = float(current.get("last_heartbeat_monotonic", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                heartbeat = 0
            if now - heartbeat <= self.stale_sec:
                return False
            lock.unlink(missing_ok=True)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
        return True


class YouTubeSourceProvider:
    def cached(
        self,
        request: SourceRequest,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo | None:
        """Return a SourceInfo from a published manifest, or None on miss."""
        if request.output_root is None:
            raise DistillError("E_BAD_OUTPUT_DIR", "youtube", "output_root is required")
        video_id = metadata.video_id if metadata is not None else canonical_youtube_id(request.value)
        fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
        sh = source_hash(fingerprint, request.options.opts_hash("youtube"))
        manifest_path = request.output_root / sh / "_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        duration = manifest.get("duration_sec")
        resolved_path = manifest.get("source_resolved_path")
        if not isinstance(duration, (int, float)) or not isinstance(resolved_path, str):
            return None
        return SourceInfo(
            source_type="youtube",
            resolved_path=Path(resolved_path),
            duration_sec=float(duration),
            source_fingerprint=fingerprint,
            source_hash=sh,
            warnings=[],
            youtube_video_id=video_id,
            youtube_lock_key=youtube_lock_key(video_id),
            related_links=list(manifest.get("related_links", []))
            if isinstance(manifest.get("related_links"), list)
            else None,
        )

    def resolve(
        self,
        request: SourceRequest,
        downloader: YouTubeDownloaderProtocol | None = None,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo:
        if request.output_root is None:
            raise DistillError("E_BAD_OUTPUT_DIR", "youtube", "output_root is required")
        options = request.options
        output_root = request.output_root
        progress = request.progress
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "disk_precheck"})
        check_disk_floor(output_root)
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "resolve_id"})
        metadata = metadata or youtube_metadata(request.value)
        video_id = metadata.video_id
        lock_key = youtube_lock_key(video_id)
        fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
        source = source_hash(fingerprint, options.opts_hash("youtube"))
        from .links import extract_relevant_links

        downloader = downloader or YoutubeDownloader(output_root)
        acquired = downloader.acquire(request.value, lock_key, progress)
        # Everything from here reads the acquired media, so every failure has to
        # release the lease rather than strand it until the staleness window
        # expires. The success path deliberately does not: the caller reads the
        # media next and releases the lease when it is finished (R-36).
        try:
            warnings = [*metadata.warnings, *acquired.warnings]
            if progress:
                progress.update("duration_probe", status="running")
            duration = probe_duration(acquired.path)
            if progress:
                progress.complete("duration_probe", detail={"duration_sec": duration})
            ensure_duration_allowed(duration, options.max_duration_sec)
            if progress:
                progress.update(
                    "youtube_download", status="running", detail={"step": "disk_postcheck"}
                )
            check_disk_floor(output_root)
            if progress:
                progress.complete(
                    "youtube_download",
                    detail={"path": str(acquired.path.resolve()), "step": "complete"},
                )
            related_links = extract_relevant_links(
                metadata.description,
                source="youtube_description",
            )
        except BaseException:
            acquired.lease.release()
            raise
        return SourceInfo(
            source_type="youtube",
            resolved_path=acquired.path.resolve(),
            duration_sec=duration,
            source_fingerprint=fingerprint,
            source_hash=source,
            warnings=warnings,
            youtube_video_id=video_id,
            youtube_lock_key=lock_key,
            related_links=related_links,
            acquisition_lease=acquired.lease,
        )


class SourceResolver:
    def __init__(
        self,
        local: LocalSourceProvider | None = None,
        youtube: YouTubeSourceProvider | None = None,
    ) -> None:
        self.local = local or LocalSourceProvider()
        self.youtube = youtube or YouTubeSourceProvider()

    def local_source(
        self,
        path_text: str,
        options: DistillOptions,
        progress: ProgressReporter | None = None,
    ) -> SourceInfo:
        return self.local.resolve(SourceRequest(path_text, options, progress=progress))

    def cached_youtube_source(
        self,
        url: str,
        options: DistillOptions,
        output_root: Path,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo | None:
        return self.youtube.cached(
            SourceRequest(url, options, output_root=output_root),
            metadata=metadata,
        )

    def youtube_source(
        self,
        url: str,
        options: DistillOptions,
        output_root: Path,
        downloader: YouTubeDownloaderProtocol | None = None,
        progress: ProgressReporter | None = None,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo:
        return self.youtube.resolve(
            SourceRequest(url, options, output_root=output_root, progress=progress),
            downloader=downloader,
            metadata=metadata,
        )

    def resolve(
        self,
        source_type: str,
        value: str,
        options: DistillOptions,
        *,
        progress: ProgressReporter | None = None,
        downloader: YouTubeDownloaderProtocol | None = None,
    ) -> SourceResolution:
        if source_type == "local":
            return SourceResolution(
                self.local_source(value, options, progress=progress),
                output_root=None,
                progress=progress,
            )
        if source_type != "youtube":
            raise DistillError(
                "E_BAD_SOURCE",
                "source",
                "source_type must be 'local' or 'youtube'",
                {"source_type": source_type},
            )

        root = validate_output_root(options.output_dir)
        url = normalize_youtube_url(value)
        # Reject non-YouTube hosts (and option-injection values) before yt-dlp runs.
        parse_youtube_url(url)
        metadata = youtube_metadata(url)
        if not options.force_reprocess:
            cached = self.cached_youtube_source(url, options, root, metadata=metadata)
            if cached is not None:
                if progress:
                    progress.skip_cached(
                        "youtube_download",
                        detail={
                            "source": "cached_manifest",
                            "video_id": cached.youtube_video_id,
                        },
                    )
                return SourceResolution(cached, output_root=root, progress=progress)
        return SourceResolution(
            self.youtube_source(
                url,
                options,
                root,
                downloader=downloader,
                progress=progress,
                metadata=metadata,
            ),
            output_root=root,
            progress=progress,
        )


def cached_youtube_source(
    url: str,
    options: DistillOptions,
    output_root: Path,
) -> SourceInfo | None:
    return SourceResolver().cached_youtube_source(url, options, output_root)


def youtube_source_info(
    url: str,
    options: DistillOptions,
    output_root: Path,
    downloader: YouTubeDownloaderProtocol | None = None,
    progress: ProgressReporter | None = None,
) -> SourceInfo:
    return SourceResolver().youtube_source(url, options, output_root, downloader, progress)


def resolve_source_for_processing(
    source_type: str,
    value: str,
    options: DistillOptions,
    *,
    progress: ProgressReporter | None = None,
    downloader: YouTubeDownloaderProtocol | None = None,
) -> SourceResolution:
    return SourceResolver().resolve(
        source_type,
        value,
        options,
        progress=progress,
        downloader=downloader,
    )
