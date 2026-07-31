"""Source acquisition and fingerprinting for Distill.

This module owns local path resolution, duration probing, safe output root
validation, YouTube id lookup, the acquisition of a remote **source**, disk
checks, and **source fingerprints**.

The cache is consulted before any capability is demanded (R-49). A tool that
exists only to *produce* a **bundle** - yt-dlp to acquire a remote source,
ffprobe to read a duration - is not a tool a run needs in order to *serve* one
that is already published, so resolution derives the **bundle key** from inputs
it already has (the video id in the URL; the local file's own bytes) and asks
the store first. What the reorder leaves alone is **source fingerprint**
derivation: the same fingerprint functions are fed from cache-safe inputs, so
the fingerprint this version computes for a given video id or file is the one it
always computed, and reordering the lookup re-keys nothing.

It does not promise that a **bundle** an earlier Distill wrote is still found.
The **pipeline version** is in the **options hash** and so in every **bundle
key**, so raising it re-keys every bundle deliberately - that is how stale
output stops being served (D-015), and serving a bundle an older pipeline
produced as this version's output would reverse the mechanism.

A miss is where the capability rules apply, unweakened - an absent **required
capability** is still a **fatal error** (ADR-0002).

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
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from .artifacts import Provenance, RedactionState, document_carries_a_reading
from .bundle_store import (
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleStore,
    ExclusiveLock,
    confined_path,
    ensure_safe_directory,
)
from .errors import DistillError, WarningRecord, errno_name, warning
from .links import RelatedLink, extract_relevant_links
from .local_vision import LocalVisionConfig
from .media_inspect import (  # noqa: F401  re-exported: media inspection
    CONTENT_HASH_LIMIT_BYTES as CONTENT_HASH_LIMIT_BYTES,
)
from .media_inspect import (
    FFPROBE_TIMEOUTS as FFPROBE_TIMEOUTS,
)
from .media_inspect import (
    FINGERPRINT_INTERIOR_ANCHORS as FINGERPRINT_INTERIOR_ANCHORS,
)
from .media_inspect import (
    FINGERPRINT_SAMPLE_BYTES as FINGERPRINT_SAMPLE_BYTES,
)
from .media_inspect import (
    _anchor_label as _anchor_label,
)
from .media_inspect import (
    ensure_duration_allowed as ensure_duration_allowed,
)
from .media_inspect import (
    fingerprint_anchor_offsets as fingerprint_anchor_offsets,
)
from .media_inspect import (
    local_fingerprint as local_fingerprint,
)
from .media_inspect import (
    manifest_duration as manifest_duration,
)
from .media_inspect import (
    probe_duration as probe_duration,
)
from .media_inspect import (
    source_hash as source_hash,
)
from .options import DistillOptions
from .progress import ProgressReporter
from .redact_secrets import redact_text
from .run_command import (
    CommandResult,
    run_json,
    stream,
)
from .vision_chain import ResolvedRun, resolve_chain
from .youtube import (  # noqa: F401  re-exported: the YouTube client
    NO_PLAYLIST_ARG as NO_PLAYLIST_ARG,
)
from .youtube import (
    YOUTUBE_HOSTS as YOUTUBE_HOSTS,
)
from .youtube import (
    YOUTUBE_STRIP_QUERY_KEYS as YOUTUBE_STRIP_QUERY_KEYS,
)
from .youtube import (
    YOUTUBE_VIDEO_ID_PATTERN as YOUTUBE_VIDEO_ID_PATTERN,
)
from .youtube import (
    YTDLP_DOWNLOAD_TIMEOUTS as YTDLP_DOWNLOAD_TIMEOUTS,
)
from .youtube import (
    YTDLP_METADATA_TIMEOUTS as YTDLP_METADATA_TIMEOUTS,
)
from .youtube import (
    YTDLP_SOCKET_TIMEOUT_SEC as YTDLP_SOCKET_TIMEOUT_SEC,
)
from .youtube import (
    YouTubeMetadata as YouTubeMetadata,
)
from .youtube import (
    _first_description_paragraph as _first_description_paragraph,
)
from .youtube import (
    _metadata_text as _metadata_text,
)
from .youtube import (
    _metadata_unavailable as _metadata_unavailable,
)
from .youtube import (
    _run_ytdlp as _run_ytdlp,
)
from .youtube import (
    _validated_youtube_video_id as _validated_youtube_video_id,
)
from .youtube import (
    _ytdlp_command as _ytdlp_command,
)
from .youtube import (
    canonical_youtube_id as canonical_youtube_id,
)
from .youtube import (
    ensure_youtube_host as ensure_youtube_host,
)
from .youtube import (
    normalize_youtube_url as normalize_youtube_url,
)
from .youtube import (
    parse_youtube_url as parse_youtube_url,
)
from .youtube import (
    youtube_description as youtube_description,
)
from .youtube import (
    youtube_fast_path_video_id as youtube_fast_path_video_id,
)
from .youtube import (
    youtube_metadata as youtube_metadata,
)
from .youtube import (
    youtube_url_names_one_video as youtube_url_names_one_video,
)

LOGGER = logging.getLogger(__name__)

SourcePathKind = Literal["file", "directory", "other", "absent"]
"""What a caller found at a path a user named. See `source_path_kind`.

Four answers and not a boolean, because the two callers want different kinds -
a video is a regular file, a batch is a directory - and neither wants the third
thing a path can be. There is deliberately no "unreadable" member: a path that
could not be asked about is not a kind of answer, it is the absence of one.
"""

YOUTUBE_DISK_FLOOR_BYTES = 1024 * 1024 * 1024
# Wall-clock ceilings so a wedged tool or a stalled network call cannot hang the
# whole run. yt-dlp additionally gets `--socket-timeout` so it aborts a stalled
# connection on its own rather than blocking until the outer timeout fires.
# ffprobe is run as `-v error`, which is silent by construction: it prints its
# document when it has the answer and nothing before then, so the idle clock
# never resets - see `silent_tool_timeouts`, which is why one number governs
# here. A lower idle value would not catch a stall, it would just cut the probe's
# budget, and a probe that runs out is fatal (`E_COMMAND`), not a degradation.
# A download is bounded by silence, not by length (R-30): a legitimate multi-GB
# fetch on a slow link may run for hours, while a wedged one stops emitting
# progress within seconds. The total is a backstop against a tool that reports
# progress forever without finishing.
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
    provenance: Provenance | None = None
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


def release_acquisition_lease(source: Any, *, during: BaseException | None = None) -> None:
    """Release the lease a source carries, if it carries one.

    Takes anything with the attribute rather than a `SourceInfo`, because the
    reader that calls this handles local sources, cache hits and test doubles
    through the same path, and none of those hold a lease.

    `during` is the failure this release is cleaning up after, when there is
    one. Cleanup that raises while an exception travels *substitutes* that
    exception, so a lease that could not be given up turned an operator's
    `Ctrl-C` into an `E_INTERNAL` record about a descriptor - the wrong
    diagnosis, and the true one thrown away. With a failure named, the release
    failure is logged instead of raised; with none, it is raised, because a
    lease this process believes it released and did not is a **lock key** the
    next run of the same source will wait out.

    An `Exception` from the release, not a `BaseException`: a second `Ctrl-C`
    landing inside cleanup still propagates, because swallowing an interrupt to
    preserve an earlier one is not an improvement on losing the earlier one.
    """
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


def _processed_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SourceRequest:
    """One source resolution, with one capture time and one lock-wait budget.

    `lock_wait_sec` is the caller's decision (D-044). Acquisition is where a
    second run of one video meets the first: two runs of the same source share
    a lock key before they contend for a bundle key, so the budget must travel
    with the request. `processed_at` is captured once on that same request and
    passed into provenance construction.
    """

    value: str
    options: DistillOptions
    output_root: Path | None = None
    progress: ProgressReporter | None = None
    lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC
    processed_at: str = field(default_factory=_processed_at_utc)


@dataclass(frozen=True)
class SourceResolution:
    source: SourceInfo
    output_root: Path | None
    progress: ProgressReporter | None = None


class YouTubeDownloaderProtocol(Protocol):
    def acquire(
        self,
        url: str,
        lock_key: str,
        progress: ProgressReporter | None = None,
    ) -> AcquiredSource: ...


def servable_duration(output_root: Path, bundle_key: str) -> float | None:
    """The duration of the **bundle** `bundle_key` names, if it is servable.

    The question a cache lookup asks before it decides a tool need not run, and
    it is `BundleStore.load_active` that answers it for the reason
    `YouTubeSourceProvider.cached` gives: a **manifest** naming a **generation**
    retention deleted is a promise rather than evidence (D-041).

    The duration is what makes this worth asking. A run that serves an **active
    generation** produces nothing and reads no media, so the only thing the
    probe still supplied was a number the manifest already records - and the
    **options hash** contains `max_duration_sec`, so a bundle found under this
    **bundle key** was published by a run that already accepted that duration
    against this cap (R-47). Re-probing to re-decide a settled question is
    exactly the tool a cache hit should not need (R-49).
    """
    snapshot = BundleStore.open(output_root).load_active(bundle_key)
    if snapshot is None:
        return None
    return manifest_duration(snapshot.manifest)


def _probe_endpoint(endpoint: LocalVisionConfig) -> bool:
    """Whether this **vision endpoint** can serve this run.

    A module-level seam rather than an import at the call site, so a test can
    replace it without a live server and without reaching into
    `local_vision`'s internals. Imported lazily because `local_vision` pulls in
    the vision client, and source resolution is reached by runs that never
    caption a frame.
    """
    from .local_vision import probe_local_vision

    return probe_local_vision(endpoint).available


def _resolved_for(
    options: DistillOptions,
    fingerprint: str,
    source_type: str,
    output_root: Path | None,
) -> ResolvedRun:
    """Walk the **endpoint chain** and settle which bundle this run is about.

    <!-- D-033 --> Here rather than in the pipeline, because a candidate key is
    an **options hash** but whether one is *cached* is a question about
    `source_hash(fingerprint, opts_hash)` - and the fingerprint only exists once
    resolution is under way. Resolving earlier would look tidier and could only
    ever probe, never serve a hit, which is the cache-before-network property
    the walk exists to guarantee.

    A caller that named no **output root** has no store to ask, so every
    candidate reads as absent and the walk goes straight to probing.
    """
    return resolve_chain(
        options,
        options.local_vision_endpoints or (),
        source_type,
        cached=lambda opts_hash: (
            None
            if output_root is None
            else servable_interpretation_count(output_root, source_hash(fingerprint, opts_hash))
        ),
        probe=_probe_endpoint,
    )


def servable_interpretation_count(output_root: Path, bundle_key: str) -> int | None:
    """How many frames of the servable **generation** carry a reading.

    <!-- D-036 --> The question chain resolution asks before treating a
    candidate key as a hit: a `selected` key promises a reader's work is in
    there, so a generation holding none is not the bundle that key describes
    (P3-D-019).

    `None` and zero are different answers. `None` is "no generation" - nothing
    to serve. Zero is "a generation, holding no readings", which is a miss under
    a `selected` key and exactly what a disabled or exhausted bundle is expected
    to hold.

    Answered from the **manifest** alone, which already embeds the frame
    documents. Opening every frame of every candidate would make Phase 1's scan
    cost what it exists to avoid - the same reason `servable_duration` reads a
    recorded duration rather than re-probing the media.
    """
    snapshot = BundleStore.open(output_root).load_active(bundle_key)
    if snapshot is None:
        return None
    frames = snapshot.manifest.get("frames")
    if not isinstance(frames, list):
        # A manifest another process wrote is input rather than fact, and a
        # missing or malformed `frames` is not evidence of zero readings.
        return None
    return sum(
        1
        for frame in frames
        if isinstance(frame, dict) and document_carries_a_reading(frame.get("interpretation"))
    )


class LocalSourceProvider:
    """Resolve one local **source**: its path, its fingerprint, its duration.

    The order is the requirement (R-49). A local **source fingerprint** is the
    file's size, mtime and sampled bytes - or every byte, in `content` mode -
    and the **options hash** is the run's options plus the **pipeline version**.
    Neither reads anything ffprobe knows, so the **bundle key** is computable
    with no probe at all, in every cache mode: the enumeration of option
    combinations that genuinely need probed metadata to *find* a bundle is
    empty. What needed the probe was the duration, and a servable bundle already
    records one.

    So the fingerprint is computed first and the cache is asked next; ffprobe is
    run only when this run has to produce a **generation**. It does not decide
    whether that generation gets produced - `BundleStore.begin` re-asks under
    the run lock, and its answer is the one that counts.

    One cost moved, and it is worth naming rather than discovering. The duration
    cap used to fire before the file was hashed; it now fires after. In the
    default `fingerprint` mode that is 576 KiB of reading either way. In
    `content` mode a source over `max_duration_sec` is hashed in full - up to
    the 5 GB `local_fingerprint` refuses past - before `E_DURATION_CAP` is
    raised. There is no ordering that avoids this: the **bundle key** is a
    function of the fingerprint, so the cache cannot be asked before the
    fingerprint exists, and asking the cache first is the requirement.
    """

    def resolve(self, request: SourceRequest) -> SourceInfo:
        path_text = request.value
        options = request.options
        progress = request.progress
        if not path_text:
            raise DistillError("E_BAD_SOURCE", "source", "path is required")
        original = Path(path_text).expanduser()
        if source_path_kind(original) != "file":
            raise DistillError(
                "E_BAD_SOURCE", "source", "local video does not exist", {"path": path_text}
            )
        resolved = original.resolve()
        warnings: list[WarningRecord] = []
        if resolved != original.absolute():
            # The target's name, redacted, and not the absolute path it was
            # found at. A warning is rendered into the archive verbatim, so the
            # path form put the operator's home directory layout - and, when a
            # link pointed at a credential-shaped filename, the credential
            # itself - into a document gate 4->5 exists to keep both out of.
            # Which link resolved where is a fact about this machine; that the
            # source was a link, and what it named, is the diagnosis.
            warnings.append(
                warning(
                    "source",
                    "symlink_resolved",
                    f"source path resolved to {redact_text(resolved.name).text}",
                )
            )
        fingerprint = local_fingerprint(resolved, options.cache_mode, progress)
        # <!-- P3-D-015 --> The key comes from the resolution, not from the
        # options the run started with. Those still name whichever endpoint the
        # chain happened to list first, so deriving from them publishes entry
        # 0's key for a run that called entry 1.
        # Named `resolution`, not `resolved`: `resolved` is already the source
        # path in this scope, and shadowing it hands a ResolvedRun to
        # `probe_duration`.
        resolution = _resolved_for(options, fingerprint, "local", request.output_root)
        options = resolution.options
        bundle_key = source_hash(fingerprint, resolution.opts_hash)
        duration = self._served_duration(request, bundle_key)
        if duration is None:
            if progress:
                progress.update("duration_probe", status="running")
            duration, probe_warnings = probe_duration(resolved)
            warnings.extend(probe_warnings)
            if progress:
                progress.complete("duration_probe", detail={"duration_sec": duration})
            ensure_duration_allowed(duration, options.max_duration_sec)
        else:
            # The cap is re-applied to the number the manifest gave, not assumed
            # from the **bundle key**. `max_duration_sec` is in the **options
            # hash**, so a manifest under this key was published by a run that
            # accepted its duration - but a manifest is a document another
            # process wrote, and R-23's premise is that its claims are input
            # rather than facts. The probe is what a cache hit does not need;
            # the operator's policy is not.
            ensure_duration_allowed(duration, options.max_duration_sec)
            if progress:
                progress.skip_cached("duration_probe", detail={"source": "cached_manifest"})
        provenance = Provenance(
            title=original.name,
            duration_sec=duration,
            processed_at=request.processed_at,
            redaction=(
                RedactionState.NOT_APPLIED if options.redact_secrets else RedactionState.DISABLED
            ),
        )
        warnings.extend(dict(item) for item in provenance.warnings)
        return SourceInfo(
            source_type="local",
            resolved_path=resolved,
            duration_sec=duration,
            source_fingerprint=fingerprint,
            source_hash=bundle_key,
            warnings=warnings,
            provenance=provenance,
        )

    def _served_duration(self, request: SourceRequest, bundle_key: str) -> float | None:
        """The published duration this run may reuse, or `None` to probe.

        `None` for a caller that named no **output root**, because there is no
        store to ask; `None` for `force_reprocess`, because that run is going to
        produce a **generation** whatever is on disk and producing one needs a
        duration read from the media rather than from a manifest it is about to
        replace.
        """
        if request.output_root is None or request.options.force_reprocess:
            return None
        return servable_duration(request.output_root, bundle_key)


def resolve_local_source(
    path_text: str,
    options: DistillOptions,
    progress: ProgressReporter | None = None,
    output_root: Path | None = None,
) -> SourceInfo:
    return LocalSourceProvider().resolve(
        SourceRequest(path_text, options, output_root=output_root, progress=progress)
    )


def source_path_kind(path: Path) -> SourcePathKind:
    """What is at a path a *user* named, refusing to guess when it cannot be asked.

    One `stat`, following symlinks, because a symlinked source is a source (the
    resolver warns about it and goes on) - and one, because this is on the run
    path every single-video command takes.

    Here rather than at each caller because both callers say the same thing when
    the answer is "I may not look": `E_SOURCE_UNREADABLE`, stage `source`, the
    path and the symbolic errno. What differs is only which kind each wanted, so
    that is what they compare. The **output root** asks a related question and
    keeps its own answer in `bundle_store._root_directory_exists`: that is a
    directory Distill owns rather than one a user named, its refusal carries a
    different code, and this module already imports that one.

    Three refusals, and only one of them is "not there":

    - the not-there errnos (`ENOENT`, `ENOTDIR`) are `absent`, which is what a
      mistyped path is;
    - a `ValueError` is `absent` too, because a path holding a NUL is one no
      filesystem can hold and names nothing. Both `Path.exists()`
      implementations swallow it deliberately, and a guard replacing `exists()`
      that does not would turn an operator's typo into an internal fault;
    - anything else - `EACCES` above all - is refused, because "does not exist"
      about a directory this process may not search is a claim with nothing
      behind it (D-022). `Path.exists()` cannot make that distinction and does
      not even fail to make it consistently: it answers `False` on Python 3.14,
      where it delegates to `os.path.exists` and swallows every `OSError`, and
      raises `PermissionError` on 3.13.
    """
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return "absent"
    except OSError as exc:
        raise DistillError(
            "E_SOURCE_UNREADABLE",
            "source",
            "source path could not be read",
            {"path": str(path), "errno": errno_name(exc)},
        ) from exc
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    return "file" if stat.S_ISREG(info.st_mode) else "other"


def _default_cache_root() -> Path:
    """`$XDG_CACHE_HOME/distill`, or `~/.cache/distill` when it is unset.

    The bundle store is derived state a cache cleaner may reclaim, so it lives
    where the platform says caches live. Only an absolute `XDG_CACHE_HOME` is
    honoured: a relative one would put the store under the process working
    directory, which is how a cache ends up inside somebody's repository.
    """
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidate = Path(xdg_cache_home).expanduser()
        if candidate.is_absolute():
            return candidate / "distill"
    return Path.home() / ".cache" / "distill"


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

    A value that is not a path *at all* is refused before any of that, as a bad
    option rather than as a bad output directory: the other refusals here are
    policy about a location Distill understood, and this one is an argument it
    could not read. `--args` is a JSON document, so every tool argument arrives
    with a type the operator chose, and `{"output_dir": 5}` reached `Path` as an
    `int` and left as a `TypeError` - an operator's typo diagnosed as an
    internal defect, with the option's name thrown away.
    """
    if output_dir is not None and not isinstance(output_dir, str):
        raise DistillError(
            "E_BAD_OPTIONS",
            "options",
            "output_dir must be a path written as text",
            {"output_dir": repr(output_dir)},
        )
    root = Path(output_dir).expanduser() if output_dir else _default_cache_root()
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
            component if case_sensitive else component.lower() for component in sensitive.split("/")
        ]
        for start in range(len(parts) - len(wanted) + 1):
            if parts[start : start + len(wanted)] == wanted:
                return sensitive
    return None


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
                promoted = promote_media(produced, self._media_dir(lock_key), root=self.output_root)
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
        staging_dir = confined_path(parent / f"{os.getpid()}-{uuid.uuid4().hex}", self.output_root)
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
        """Run yt-dlp with its output template pointed at the staging directory.

        The template names the staging directory and never the promoted path, so
        yt-dlp's habit of opening its destination before the transfer succeeds
        cannot truncate media a previous run proved good (RV-3).
        """
        out_template = str(staging_dir / f"{PROMOTED_MEDIA_STEM}.%(ext)s")
        command = [
            "yt-dlp",
            # The URL names one video, whatever `list` parameter it carries.
            NO_PLAYLIST_ARG,
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


def _manifest_provenance(
    manifest: Mapping[str, Any],
    options: DistillOptions,
    *,
    duration_sec: float,
    processed_at: str,
    video_id: str,
) -> Provenance:
    """Rebuild cached provenance as a carrier under this run's policy."""
    recorded = manifest.get("provenance")
    fields = recorded if isinstance(recorded, Mapping) else {}

    def optional_string(name: str) -> str | None:
        value = fields.get(name)
        return value if isinstance(value, str) else None

    return Provenance(
        title=optional_string("title"),
        channel=optional_string("channel"),
        description=optional_string("description"),
        upload_date=optional_string("upload_date"),
        canonical_url=(
            optional_string("canonical_url") or f"https://www.youtube.com/watch?v={video_id}"
        ),
        duration_sec=duration_sec,
        processed_at=optional_string("processed_at") or processed_at,
        redaction=(
            RedactionState.NOT_APPLIED if options.redact_secrets else RedactionState.DISABLED
        ),
    )


class YouTubeSourceProvider:
    def cached(
        self,
        request: SourceRequest,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo | None:
        """Describe a source from a servable bundle, or `None` if there is none.

        Two ids can name one video: the one written in the URL, and the one
        yt-dlp reports. For an ordinary watch URL carrying an id-shaped value
        they are the same string, which is why the URL's own id is tried first -
        it costs nothing, and a cache hit found that way needs no tool at all
        (R-49). `youtube_fast_path_video_id` decides whether this URL is one of
        those. Otherwise, and whenever the first lookup misses, yt-dlp is asked
        for the canonical id and the lookup repeated, so a bundle keyed by an id
        the URL does not carry is still found, exactly as it was before.

        The **fingerprint** is the same function of the same video id either
        way. Nothing here derives identity a second way; what changed is only
        which cache-safe input it is fed.
        """
        if metadata is not None:
            return self.cached_for_video_id(request, metadata.video_id)
        url_video_id = youtube_fast_path_video_id(request.value)
        if url_video_id is not None:
            served = self.cached_for_video_id(request, url_video_id)
            if served is not None:
                return served
        return self.cached_for_video_id(request, canonical_youtube_id(request.value))

    def cached_for_video_id(self, request: SourceRequest, video_id: str) -> SourceInfo | None:
        """The servable **bundle** one video id names, or `None` for a miss.

        The question is "can this download be skipped?", and the only answer
        that justifies skipping it is a **bundle** whose **active generation**
        is on disk (R-04). `BundleStore.load_active` is what proves that, so it
        is asked rather than the manifest file being read directly: a manifest
        naming a generation retention deleted is a promise, not evidence, and
        reusing the media path it records would resolve a source for a bundle
        nobody can serve (D-041).

        The duration the manifest records is then put through the same cap the
        local hit applies, for the reason `LocalSourceProvider.resolve` gives:
        `max_duration_sec` is in the **options hash**, so a bundle under this
        **bundle key** was published by a run that accepted its duration - but
        a manifest is a document another process wrote, and R-23's premise is
        that its claims are input rather than facts. Skipping the probe is what
        a cache hit is entitled to; skipping the operator's policy is not, and
        one policy cannot answer differently by kind of **source**.
        """
        if request.output_root is None:
            raise DistillError("E_BAD_OUTPUT_DIR", "youtube", "output_root is required")
        fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
        sh = source_hash(fingerprint, request.options.opts_hash("youtube"))
        snapshot = BundleStore.open(request.output_root).load_active(sh)
        if snapshot is None:
            return None
        manifest = snapshot.manifest
        duration = manifest_duration(manifest)
        resolved_path = manifest.get("source_resolved_path")
        if duration is None or not isinstance(resolved_path, str):
            return None
        ensure_duration_allowed(duration, request.options.max_duration_sec)
        return SourceInfo(
            source_type="youtube",
            resolved_path=Path(resolved_path),
            duration_sec=duration,
            source_fingerprint=fingerprint,
            source_hash=sh,
            warnings=[],
            provenance=_manifest_provenance(
                manifest,
                request.options,
                duration_sec=duration,
                processed_at=request.processed_at,
                video_id=video_id,
            ),
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
            provenance = Provenance(
                title=metadata.title,
                channel=metadata.channel,
                description=_first_description_paragraph(metadata.description) or None,
                upload_date=metadata.upload_date,
                canonical_url=f"https://www.youtube.com/watch?v={video_id}",
                duration_sec=duration,
                processed_at=request.processed_at,
                redaction=(
                    RedactionState.NOT_APPLIED
                    if options.redact_secrets
                    else RedactionState.DISABLED
                ),
            )
            warnings.extend(dict(item) for item in provenance.warnings)
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
            provenance=provenance,
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
        output_root: Path | None = None,
    ) -> SourceInfo:
        return self.local.resolve(
            SourceRequest(path_text, options, output_root=output_root, progress=progress)
        )

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
            # The root is validated here, not only in the pipeline, because the
            # local resolution now asks the store whether a **bundle** for this
            # source is already servable - and it cannot ask without knowing
            # which store. It is the same path the caller validated, under the
            # same policy, so nothing new is created or accepted (R-15).
            local_root = validate_output_root(options.output_dir)
            return SourceResolution(
                self.local_source(value, options, progress=progress, output_root=local_root),
                output_root=local_root,
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
        # Rejects non-YouTube hosts (and option-injection values) before yt-dlp
        # runs. What it returns is the value written in the URL, which is not on
        # its own enough to key a **bundle** by - `youtube_fast_path_video_id`
        # decides that below.
        parse_youtube_url(url)
        request = SourceRequest(
            url,
            options,
            output_root=root,
            progress=progress,
            lock_wait_sec=lock_wait_sec,
        )

        # The cache before the capability (R-49, finding 22). yt-dlp exists to
        # acquire a **source**; a run that serves an **active generation**
        # acquires nothing, so demanding the tool first told a user their run
        # could not proceed over a bundle already on disk.
        #
        # Only where the URL's id is the id this run would publish under, which
        # `youtube_fast_path_video_id` decides. A URL that names a playlist as
        # well as a video, or carries a value yt-dlp's extractor would not match
        # whole, is resolved the way it always was.
        fast_path_video_id = youtube_fast_path_video_id(url)
        if fast_path_video_id is not None:
            served = self._served_from_cache(request, fast_path_video_id)
            if served is not None:
                return served
        metadata = youtube_metadata(url)
        # The resolved id is what the cache was always asked about, before the
        # reorder put a lookup in front of it. It is skipped only when the fast
        # path already asked this exact question and missed - so a URL the fast
        # path declined is looked up here, which is the behavior that predates
        # the reorder rather than a second chance added by it.
        if metadata.video_id != fast_path_video_id:
            served = self._served_from_cache(request, metadata.video_id)
            if served is not None:
                return served
        return SourceResolution(
            self.youtube.resolve(
                request,
                downloader=downloader,
                metadata=metadata,
            ),
            output_root=root,
            progress=progress,
        )

    def _served_from_cache(self, request: SourceRequest, video_id: str) -> SourceResolution | None:
        """This run's resolution if `video_id` names a servable bundle, else `None`.

        `force_reprocess` never consults the cache: that run is going to produce
        a **generation** whatever is on disk, so a hit would only let it skip
        the acquisition it is about to need.
        """
        if request.options.force_reprocess:
            return None
        cached = self.youtube.cached_for_video_id(request, video_id)
        if cached is None:
            return None
        if request.progress:
            request.progress.skip_cached(
                "youtube_download",
                detail={"source": "cached_manifest", "video_id": cached.youtube_video_id},
            )
        return SourceResolution(cached, output_root=request.output_root, progress=request.progress)


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
