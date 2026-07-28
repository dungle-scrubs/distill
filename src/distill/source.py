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

Exclusion is the kernel's, via `flock` on a descriptor the lease keeps open for
as long as it is held. A filesystem that cannot take that lock is a filesystem
on which Distill cannot tell one run from two, so acquisition fails closed with
`E_LOCK_UNSUPPORTED` rather than falling back to a weaker scheme.

It does not write final bundle artifacts, own bundle-level locking (that is
keyed by **bundle key** and belongs to the bundle store; the lease here is keyed
by **lock key** and answers only "is another run fetching this source?"), decide
what a **generation** contains, or install anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

from .artifacts import RedactionState
from .bundle_store import (
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleStore,
    ExclusiveLock,
    confined_path,
    ensure_safe_directory,
)
from .capabilities import MISSING_TOOL_CODE, missing_tool_consequence
from .errors import DistillError, WarningRecord, warning
from .links import RelatedLink, extract_relevant_links
from .options import DistillOptions
from .progress import ProgressReporter
from .run_command import (
    CommandResult,
    CommandTimeouts,
    run,
    run_json,
    silent_tool_timeouts,
    stream,
)

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
# ffprobe is run as `-v error`, which is silent by construction: it prints its
# document when it has the answer and nothing before then, so the idle clock
# never resets - see `silent_tool_timeouts`, which is why one number governs
# here. A lower idle value would not catch a stall, it would just cut the probe's
# budget, and a probe that runs out is fatal (`E_COMMAND`), not a degradation.
FFPROBE_TIMEOUTS = silent_tool_timeouts(60.0)
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

    The exclusion itself is `bundle_store.ExclusiveLock` - one `flock` primitive
    for the whole package. The lease and the bundle lock stay two locks, because
    they answer different questions on different keys ("is another run fetching
    this source?" against "is another run producing this bundle?"), but they are
    not two mechanisms: the argument for why an open descriptor is exclusive and
    a lock file's *contents* are not is the argument finding 11 came from
    getting wrong, and a second copy of it is a second chance to get it wrong
    again.

    Held until `release`, which the reader calls when it is finished with the
    media - not when the download ends. `release` is idempotent, because the
    failure paths that release a lease early overlap with the caller's own
    cleanup.

    Does not own: the media it protects, the staging directory, or bundle-level
    locking, which is keyed by **bundle key** and belongs to the bundle store.
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
        """Take the lease, or report `None` if another run holds it.

        A filesystem that cannot take the lock (`flock` reporting anything other
        than "held") is fatal: `E_LOCK_UNSUPPORTED` rather than a fallback that
        would let two runs read one source without either of them knowing. The
        stage and message name *this* question, so a user downloading a video
        is told which lock could not be taken.
        """
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
        """Give the lease up by closing the descriptor the kernel locked.

        The lock file itself stays on disk, deliberately. Unlinking it is a
        second way to lose exclusivity: a waiter that has already opened the
        path holds a descriptor on an inode that now has no name, so it can lock
        that inode while the next run creates a fresh file at the same path and
        locks that - two runs, again, one lock key. Leaving the file costs one
        empty inode per **lock key** and keeps the path itself the identity of
        the lease.

        Nothing ever removes it, **prune** included: `_youtube_locks` is a
        reserved name the walk skips, so no rule proposes what is inside it
        (finding 7-opus). That is the intended end state - a lock file deleted
        while any run might be about to open its path is the same lost
        exclusivity - and the cost is bounded by the number of distinct sources
        this root has ever acquired.
        """
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
    """A promoted media file and the lease that keeps it readable.

    The lease travels with the path because the two have the same lifetime: a
    caller holding this holds the right to read `path`, and the moment it
    releases the lease another run may promote a replacement.
    """

    path: Path
    lease: AcquisitionLease
    warnings: list[WarningRecord] = field(default_factory=list)


@dataclass(frozen=True)
class SourceInfo:
    source_type: str
    resolved_path: Path
    duration_sec: float
    source_fingerprint: str
    source_hash: str
    warnings: list[WarningRecord]
    youtube_video_id: str | None = None
    youtube_lock_key: str | None = None
    related_links: list[RelatedLink] | None = None
    """The **related links** this source's metadata named, as carriers (R-21).

    Carriers and not documents, because every sink they reach - the **render**,
    the **manifest**, the caller's response - is a sink R-20 is stated about,
    and a document handed to one has bypassed the check that a document is all
    it can perform (finding 5). Serializing them is the sink's job.
    """
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
    lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC
    """How long this run waits for a **lock key** another run holds (D-044).

    Carried on the request because the budget is the *caller's* decision and
    acquisition is where a second run of one video meets the first: two runs of
    the same source share a lock key long before they contend for a **bundle
    key**, so a downloader constructed with its own idea of the budget makes
    D-044 unreachable for the case it was written for (finding 4-opus). The
    values are `bundle_store`'s, so the run a user is watching waits the same
    300 s at both locks and a batch item gives up after the same 5 s at both.
    """


@dataclass(frozen=True)
class SourceResolution:
    source: SourceInfo
    output_root: Path | None
    progress: ProgressReporter | None = None


@dataclass(frozen=True)
class YouTubeMetadata:
    video_id: str
    description: str
    warnings: list[WarningRecord]


class YouTubeDownloaderProtocol(Protocol):
    def acquire(
        self,
        url: str,
        lock_key: str,
        progress: ProgressReporter | None = None,
    ) -> AcquiredSource: ...


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
        warnings: list[WarningRecord] = []
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
        duration, probe_warnings = probe_duration(resolved)
        warnings.extend(probe_warnings)
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


def validate_output_root(output_dir: str | None, *, create: bool = True) -> Path:
    """Accept an output root, or refuse it before anything is written under it.

    `create=False` validates without creating, which is what a read-only caller
    needs: `cache-doctor` reports on a root and must not bring one into
    existence to say it is empty (R-57). Every writing caller takes the default.

    An output root is a place Distill owns: it creates bundles there and prunes
    them there. `$HOME` and the system temp directory are not such places - they
    hold the user's own files and other programs' - so R-15 admits only a
    subdirectory of either. Accepting `$HOME` itself is what gave finding 1 its
    blast radius, since every directory in the home directory then sat inside an
    output root.

    The root is resolved first, so confinement is decided on the real path a
    symlinked argument points at rather than on the name given.
    """
    root = Path(output_dir).expanduser() if output_dir else Path.home() / ".cache" / "distill"
    root = root.resolve()
    home = Path.home().resolve()
    temp = Path(tempfile.gettempdir()).resolve()
    if root in (home, temp):
        raise DistillError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "output_dir must be a subdirectory of $HOME or the system temp "
            "directory, not the directory itself",
            {"output_dir": str(root)},
        )
    if not (root.is_relative_to(home) or root.is_relative_to(temp)):
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
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def sensitive_path_match(
    root: Path,
    *,
    case_sensitive: bool | None = None,
) -> str | None:
    """The sensitive location `root` sits in, or `None`.

    Matching is by whole path components, not by substring: `.sshfs-mounts`
    contains `.ssh` and `my.aws-notes` contains `.aws`, yet neither is the
    directory the rule protects, and refusing them refuses output roots a user
    may legitimately ask for. Multi-component entries such as
    `.config/1password` match as a consecutive run of components.
    """
    if case_sensitive is None:
        case_sensitive = sys.platform != "darwin"
    parts = [part if case_sensitive else part.lower() for part in Path(root).parts]
    for sensitive in sorted(SENSITIVE_COMPONENTS):
        wanted = [
            component if case_sensitive else component.lower()
            for component in sensitive.split("/")
        ]
        for start in range(len(parts) - len(wanted) + 1):
            if parts[start : start + len(wanted)] == wanted:
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


def _metadata_unavailable() -> WarningRecord:
    return warning(
        "youtube",
        "metadata_unavailable",
        "yt-dlp could not read YouTube description",
    )


def youtube_description(url: str) -> tuple[str, list[WarningRecord]]:
    try:
        proc = _run_ytdlp(["--skip-download", "--print", "%(description)s"], url)
    except DistillError as exc:
        # An absent tool is the capability table's decision here too. The
        # description being best-effort does not make an absent yt-dlp a
        # degradation - yt-dlp is a **required capability**, so this raises
        # (ADR-0002, R-34). Anything else is yt-dlp having run and not answered,
        # which costs the description and not the run.
        if exc.code == MISSING_TOOL_CODE:
            return "", [missing_tool_consequence("youtube", "yt-dlp", cause=exc)]
        return "", [_metadata_unavailable()]
    if proc.returncode != 0:
        return "", [*proc.warnings, _metadata_unavailable()]
    return proc.stdout.strip(), list(proc.warnings)


def youtube_metadata(url: str) -> YouTubeMetadata:
    proc = _run_ytdlp(["--skip-download", "--dump-json"], url)
    # Carried whichever branch answers: a metadata invocation that lost part of
    # its own output (R-33) is the same loss whether or not the document parsed.
    probe_warnings = list(proc.warnings)
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
                warnings=probe_warnings,
            )

    video_id = canonical_youtube_id(url)
    description, metadata_warnings = youtube_description(url)
    return YouTubeMetadata(
        video_id=video_id,
        description=description,
        warnings=[*probe_warnings, *metadata_warnings],
    )


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
        duration = float(value)
    except ValueError:
        return 0.0
    # A header claiming NaN or inf is a container without a playable duration,
    # and reporting it as one is how it passed `duration_sec <= 0` (R-47).
    return duration if math.isfinite(duration) else 0.0


def validate_media_file(path: Path) -> list[WarningRecord]:
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

    It returns the probe's own **warnings** - truncated capture (R-33) - because
    a verdict of "accepted" is not the same as "nothing was lost", and this is
    the only place those warnings exist.
    """
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
    return list(probe_warnings)


def promote_media(produced: Path, media_dir: Path, *, root: Path) -> Path:
    """Move a validated file onto its immutable path in one indivisible step.

    `os.replace` on one filesystem is the whole promotion: a reader either sees
    the previous media or this one, never a half-written file, and the previous
    media is not removed before its replacement exists. Copying, truncating or
    clearing the directory first would each reintroduce the window RV-3
    describes, in which the only good copy is already gone when the replacement
    turns out not to arrive.

    Both ends of the rename are checked against `root` (R-16). A rename has two
    of them: a symlink at the media directory sends the destination outside the
    output root and replaces whatever it names, and a substituted staging
    directory makes the *source* a file of the user's that promotion then moves
    away from where they left it.
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


class YoutubeDownloader:
    def __init__(
        self,
        output_root: Path,
        *,
        lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
        lock_poll_sec: float = 0.25,
        lock_warn_after_sec: float = 5.0,
    ) -> None:
        """Acquire one remote **source** under an **acquisition lease**.

        `lock_wait_sec` defaults to the budget of the run a user is watching
        rather than to zero (D-044). A zero default was one production call site
        away from making the wait unreachable, and it was that call site: every
        second run of a video was denied on its first attempt, which is the case
        the budget exists for (finding 4-opus). A caller that means "do not
        wait" now says so.
        """
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
                validation_warnings = validate_media_file(produced)
                promoted = promote_media(
                    produced, self._media_dir(lock_key), root=self.output_root
                )
            finally:
                # The staging directory is scratch: whatever survived the run -
                # a rejected file, an unmerged fragment, a `.part` - is
                # discarded with it. Nothing under the promoted path is touched.
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
        """Remove a staging directory, having first proved it is one (R-16).

        The check is re-run here rather than trusted from creation, and it
        raises rather than skipping. A path that was inside the output root when
        it was made and is not when it is deleted is a substitution in progress,
        which is a sharper thing to report than whatever the download was doing
        when it happened - so this is allowed to replace an in-flight error.
        """
        confined_path(staging_dir, self.output_root)
        shutil.rmtree(staging_dir, ignore_errors=True)

    def _new_staging_dir(self, lock_key: str) -> Path:
        """A staging directory no other run can be writing into.

        Named by pid and a random token, so two runs of one source - which the
        lease already serializes - and two runs of different sources alike get
        directories that cannot collide. Created under the lease, which is also
        what makes discarding any staging directory left by an earlier run of
        this source safe: no live run can own one.

        The lease proves who is running, not what the path is, so every entry
        this walks is checked against the output root immediately before it is
        removed (R-16). Both halves matter: a symlink at `<lock key>` makes
        `iterdir` enumerate somebody else's directory, and a link planted among
        real entries names a directory this run never staged into. Each is a
        `rmtree` of the user's files, and neither is anything the lease can see.
        """
        parent = ensure_safe_directory(
            self.output_root / STAGING_DIR_NAME / lock_key, self.output_root
        )
        for stale in sorted(parent.iterdir()):
            confined_path(stale, self.output_root)
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        staging_dir = confined_path(
            parent / f"{os.getpid()}-{uuid.uuid4().hex}", self.output_root
        )
        staging_dir.mkdir()
        return staging_dir

    def _take_lease(
        self,
        lock_key: str,
        progress: ProgressReporter | None,
    ) -> AcquisitionLease:
        locks = ensure_safe_directory(self.output_root / LOCK_DIR_NAME, self.output_root)
        # The lock path itself is checked by `_acquire`, once per attempt, rather
        # than here: it is created by the attempt that takes it, and a wait that
        # can run for minutes is exactly the gap a check placed here would leave.
        lock = locks / f"{lock_key}.lock"
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "lock"})
        lease, lock_warnings = self._acquire(lock_key, lock)
        if lease is None:
            _acquisition_log(
                "lease_denied", lock_key=lock_key, lock_path=str(lock), reason="held"
            )
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

    def _acquire(
        self,
        lock_key: str,
        lock: Path,
    ) -> tuple[AcquisitionLease | None, list[WarningRecord]]:
        """Poll for the lease within this run's wait budget.

        The budget is the only thing that ends the wait: a lock is held until
        its holder gives it up or dies, and neither is something to time out on
        a holder's behalf.

        Each attempt creates the lock file if it is not there, so each attempt
        re-checks the path (R-16). A wait that can run for minutes and validates
        once at the top is a check separated from its use by the whole wait,
        which is the window the check exists to close.
        """
        started = time.monotonic()
        warnings: list[WarningRecord] = []
        while True:
            confined_path(lock, self.output_root)
            lease = AcquisitionLease.take(lock_key, lock)
            if lease is not None:
                waited = time.monotonic() - started
                if waited >= self.lock_warn_after_sec:
                    warnings.append(
                        warning(
                            "youtube",
                            "long_lock_wait",
                            f"waited {waited:.1f}s for YouTube source lock",
                        )
                    )
                return lease, warnings
            if time.monotonic() - started >= self.lock_wait_sec:
                return None, warnings
            time.sleep(self.lock_poll_sec)


def _manifest_related_links(
    manifest: Mapping[str, Any], options: DistillOptions
) -> list[RelatedLink] | None:
    """The **related links** a servable **manifest** recorded, as carriers again.

    A cache hit describes a source it did not read, so its links come off disk -
    and they come back as carriers because the run reusing them hands them to
    the same sinks a fresh run does, which are the sinks R-20 is stated about.

    The policy is this run's and never the document's claim about itself, for
    the reason `FrameArtifact.from_document` gives: a manifest is something
    another process wrote, and `redact_secrets` participates in the **options
    hash** and so in the **bundle key**, so a manifest under this bundle key was
    written by a run under this policy.

    An entry that is not a mapping is dropped rather than refused: a link that
    cannot be rebuilt costs that link, not the cache hit.
    """
    recorded = manifest.get("related_links")
    if not isinstance(recorded, list):
        return None
    policy = RedactionState.NOT_APPLIED if options.redact_secrets else RedactionState.DISABLED
    return [
        RelatedLink.from_document(entry, redaction=policy)
        for entry in recorded
        if isinstance(entry, Mapping)
    ]


class YouTubeSourceProvider:
    def cached(
        self,
        request: SourceRequest,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo | None:
        """Describe a source from a servable bundle, or `None` if there is none.

        The question is "can this download be skipped?", and the only answer
        that justifies skipping it is a **bundle** whose **active generation**
        is on disk (R-04). `BundleStore.load_active` is what proves that, so it
        is asked rather than the manifest file being read directly: a manifest
        naming a generation retention deleted is a promise, not evidence, and
        reusing the media path it records would resolve a source for a bundle
        nobody can serve (D-041).
        """
        if request.output_root is None:
            raise DistillError("E_BAD_OUTPUT_DIR", "youtube", "output_root is required")
        video_id = metadata.video_id if metadata is not None else canonical_youtube_id(request.value)
        fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
        sh = source_hash(fingerprint, request.options.opts_hash("youtube"))
        snapshot = BundleStore.open(request.output_root).load_active(sh)
        if snapshot is None:
            return None
        manifest = snapshot.manifest
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
            related_links=_manifest_related_links(manifest, request.options),
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
        downloader = downloader or YoutubeDownloader(
            output_root, lock_wait_sec=request.lock_wait_sec
        )
        acquired = downloader.acquire(request.value, lock_key, progress)
        # Everything from here reads the acquired media, so every failure has to
        # release the lease rather than strand it until the staleness window
        # expires. The success path deliberately does not: the caller reads the
        # media next and releases the lease when it is finished (R-36).
        try:
            warnings = [*metadata.warnings, *acquired.warnings]
            if progress:
                progress.update("duration_probe", status="running")
            duration, probe_warnings = probe_duration(acquired.path)
            warnings.extend(probe_warnings)
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
                redact=options.redact_secrets,
            )
            # Construction is where the **redaction** policy runs over a link's
            # label and destination, so the **warnings** it raises there are
            # this run's: a confusable-obfuscated key it had to normalize, a
            # label it truncated. They travel with the source's other warnings
            # rather than inside the link, which is bundle content and not a
            # place a **degradation** can be counted (finding 7).
            warnings.extend(dict(item) for link in related_links for item in link.warnings)
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
        lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
    ) -> SourceInfo:
        return self.youtube.resolve(
            SourceRequest(
                url,
                options,
                output_root=output_root,
                progress=progress,
                lock_wait_sec=lock_wait_sec,
            ),
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
        lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
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
                lock_wait_sec=lock_wait_sec,
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
    lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
) -> SourceResolution:
    """Resolve one **source** for a run, on that run's wait budget (D-044).

    `lock_wait_sec` is the caller's, and it belongs here rather than at the
    downloader because acquisition is the *first* lock a run meets: two runs of
    one video contend for a **lock key** before either reaches a **bundle key**,
    so a budget that stops short of this function is a budget the case it was
    written for never reaches (finding 4-opus).
    """
    return SourceResolver().resolve(
        source_type,
        value,
        options,
        progress=progress,
        downloader=downloader,
        lock_wait_sec=lock_wait_sec,
    )
