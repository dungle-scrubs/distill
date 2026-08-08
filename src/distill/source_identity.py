"""Pure source identity: **source fingerprint**, **lock key**, **bundle key**, fast-path.

This module owns the **source fingerprint**, **lock key**, **bundle key** and
fast-path decision for a **source** without touching the filesystem or the
network. A **source fingerprint** names the media, a **lock key** names the
remote **source** for the **acquisition lease** (R-36), a **bundle key** names
the **bundle** (the fingerprint plus the **options hash**), and the fast-path
check decides whether the **bundle key** is knowable from the URL alone
(R-49, find-22).

No I/O is performed here: no file is opened, no path is stat-ed, no tool is
spawned. That lets the caller ask the **bundle store** whether the **bundle**
is already servable before any **required capability** is demanded. Every
helper is a pure function of its inputs.

Vocabulary is per ``CONTEXT.md``: **source**, **source fingerprint**,
**options hash**, **bundle key**, **lock key**, **acquisition lease**,
**bundle**, **generation**, **active generation**.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from .options import DistillOptions
from .youtube import (  # pure re-exports: no I/O, only URL parsing constants
    YOUTUBE_HOSTS as YOUTUBE_HOSTS,
)
from .youtube import (
    YOUTUBE_VIDEO_ID_PATTERN as YOUTUBE_VIDEO_ID_PATTERN,
)


@dataclass(frozen=True)
class SourceIdentity:
    """The derived identities for one **source** request, without I/O.

    For a YouTube **source** the **source fingerprint** and **lock key** are
    both ``sha256(video_id)`` - same bytes, different questions ("is this the
    same media?" vs "is another run fetching this?"). For a local **source**
    the fingerprint is not knowable without reading the file, so it is ``None``
    here and is derived by the I/O layer (``media_inspect.local_fingerprint``).
    A ``None`` **bundle key** likewise means the key cannot be known until the
    fingerprint exists (local) or the video id is resolved (YouTube without a
    fast-path id).

    ``opts_hash`` is the **options hash** for the **source** kind before the
    **endpoint chain** walk (``DistillOptions.opts_hash``). The walk may settle
    on a different ``opts_hash`` (the selected entry); the **bundle key** that
    incorporates that later choice is derived by the resolver, not here.

    ``fast_path_video_id`` is the id the URL itself names when
    ``youtube_fast_path_video_id`` can answer without yt-dlp (R-49). ``None``
    does not mean the URL is invalid, only that the shortcut is declined and
    the resolver must ask yt-dlp.
    """

    source_type: Literal["local", "youtube"]
    source_fingerprint: str | None
    lock_key: str | None
    bundle_key: str | None
    opts_hash: str | None
    video_id: str | None
    fast_path_video_id: str | None
    is_fast_path: bool

    @property
    def is_youtube_fast_path(self) -> bool:
        """Alias for ``is_fast_path`` for call-site readability."""
        return self.is_fast_path


# ---------------------------------------------------------------------------
# Fingerprint helpers (pure)
# ---------------------------------------------------------------------------


def youtube_fingerprint(video_id: str) -> str:
    """The **source fingerprint** for a YouTube **source**: ``sha256(video_id)``.

    Pure: the id alone names the media, so no file or network is consulted.
    """
    return hashlib.sha256(video_id.encode()).hexdigest()


def fingerprint_for_youtube(video_id: str) -> str:
    """Alias for ``youtube_fingerprint`` kept for the helper family name."""
    return youtube_fingerprint(video_id)


def source_fingerprint_for_youtube(video_id: str) -> str:
    """Another alias, matching the ``source_fingerprint`` phrasing in docs."""
    return youtube_fingerprint(video_id)


# ---------------------------------------------------------------------------
# Lock key helpers (pure)
# ---------------------------------------------------------------------------


def youtube_lock_key(video_id: str) -> str:
    """The **lock key** for a YouTube **source**: ``sha256(video_id)``.

    Same bytes as the **source fingerprint** for YouTube, but a distinct
    concept per ``CONTEXT.md``: the **source fingerprint** answers "is this the
    same media?" while the **lock key** answers "is another run fetching this?"
    They stay separate functions so a change to one identity does not silently
    change the other.
    """
    return hashlib.sha256(video_id.encode()).hexdigest()


def lock_key_for_youtube(video_id: str) -> str:
    """Alias for ``youtube_lock_key``."""
    return youtube_lock_key(video_id)


def lock_key_for_video_id(video_id: str) -> str:
    """Alias for ``youtube_lock_key``."""
    return youtube_lock_key(video_id)


# ---------------------------------------------------------------------------
# Bundle key helpers (pure)
# ---------------------------------------------------------------------------


def source_hash(source_fingerprint: str, opts_hash: str) -> str:
    """The **bundle key**: ``sha256(fingerprint + ':' + opts_hash)``.

    Pure combination of the **source fingerprint** and the **options hash**.
    Named ``source_hash`` for backward compatibility (the legacy name for the
    **bundle key**); ``bundle_key`` is the preferred name.
    """
    return hashlib.sha256(f"{source_fingerprint}:{opts_hash}".encode()).hexdigest()


def bundle_key(source_fingerprint: str, opts_hash: str) -> str:
    """The **bundle key** for a **source fingerprint** and **options hash**."""
    return source_hash(source_fingerprint, opts_hash)


def bundle_key_for(source_fingerprint: str, opts_hash: str) -> str:
    """Alias for ``bundle_key``."""
    return source_hash(source_fingerprint, opts_hash)


# Back-compat alias: older callers import ``source_hash`` from ``source``.
__all__ = [
    "SourceIdentity",
    "youtube_fingerprint",
    "fingerprint_for_youtube",
    "source_fingerprint_for_youtube",
    "youtube_lock_key",
    "lock_key_for_youtube",
    "lock_key_for_video_id",
    "source_hash",
    "bundle_key",
    "bundle_key_for",
    "youtube_fast_path_video_id",
    "is_youtube_fast_path",
    "youtube_url_names_one_video",
    "derive_source_identity",
    "derive_youtube_identity",
    "derive_local_identity",
]


# ---------------------------------------------------------------------------
# Fast-path decision (pure, no network)
# ---------------------------------------------------------------------------

YOUTUBE_STRIP_QUERY_KEYS = {"t", "start", "end", "time_continue"}

# Re-export the canonical fast-path decision from ``youtube`` so this module
# remains pure and the **bundle key** derived here is always the key the run
# would publish under. The implementation is pure URL parsing; no network.
from .youtube import (  # noqa: E402  after constants, before use
    youtube_fast_path_video_id as _yt_fast_path,
)
from .youtube import (  # noqa: E402  after constants, before use
    youtube_url_names_one_video as _yt_names_one,
)


def youtube_fast_path_video_id(url: str) -> str | None:
    """The id a **bundle** may be looked up by without asking yt-dlp, or ``None``."""
    return _yt_fast_path(url)


def is_youtube_fast_path(url: str) -> bool:
    """Whether the URL's own video id is the id yt-dlp will resolve for it."""
    return _yt_fast_path(url) is not None


def youtube_url_names_one_video(url: str) -> bool:
    """Whether the URL names exactly one video whose id is knowable without yt-dlp."""
    return _yt_names_one(url)


# ---------------------------------------------------------------------------
# High-level derivation
# ---------------------------------------------------------------------------


def derive_youtube_identity(
    url: str,
    options: DistillOptions,
    *,
    video_id: str | None = None,
) -> SourceIdentity:
    """Derive the YouTube **source fingerprint**, **lock key**, **bundle key** and fast-path.

    Pure: the **source fingerprint** and **lock key** are ``sha256(video_id)``,
    the **bundle key** is ``source_hash(fingerprint, opts_hash)``, and the
    fast-path decision is ``youtube_fast_path_video_id(url)``. No file is read
    and no tool is spawned. When ``video_id`` is ``None`` the fast-path id is
    tried; when that also declines the fingerprint/lock/bundle are ``None``
    (the resolver must ask yt-dlp for the canonical id).

    ``opts_hash`` here is ``options.opts_hash("youtube")`` — the hash before
    the **endpoint chain** walk. The resolver later re-derives the key from the
    chain-selected ``opts_hash``; this value is the candidate key the URL alone
    can address before that walk.
    """
    fast_path = youtube_fast_path_video_id(url)
    vid = video_id if video_id is not None else fast_path
    if vid is None:
        # Fast path declined and no canonical id supplied - identities not
        # knowable without yt-dlp.
        opts_hash = options.opts_hash("youtube")
        return SourceIdentity(
            source_type="youtube",
            source_fingerprint=None,
            lock_key=None,
            bundle_key=None,
            opts_hash=opts_hash,
            video_id=None,
            fast_path_video_id=fast_path,
            is_fast_path=fast_path is not None,
        )
    fingerprint = youtube_fingerprint(vid)
    lock_key = youtube_lock_key(vid)
    opts_hash = options.opts_hash("youtube")
    bkey = bundle_key(fingerprint, opts_hash)
    return SourceIdentity(
        source_type="youtube",
        source_fingerprint=fingerprint,
        lock_key=lock_key,
        bundle_key=bkey,
        opts_hash=opts_hash,
        video_id=vid,
        fast_path_video_id=fast_path,
        is_fast_path=fast_path is not None,
    )


def derive_local_identity(
    _path_text: str,
    options: DistillOptions,
) -> SourceIdentity:
    """Derive what is knowable about a local **source** without I/O.

    The **source fingerprint** for a local **source** is the file's sampled
    bytes (or every byte in ``content`` mode) — it cannot be derived without
    reading the file. This therefore returns the **options hash** and the
    fast-path fields as ``None``, leaving the fingerprint and **bundle key**
    for the I/O layer (``media_inspect.local_fingerprint`` + ``bundle_key``).
    """
    opts_hash = options.opts_hash("local")
    return SourceIdentity(
        source_type="local",
        source_fingerprint=None,
        lock_key=None,
        bundle_key=None,
        opts_hash=opts_hash,
        video_id=None,
        fast_path_video_id=None,
        is_fast_path=False,
    )


def derive_source_identity(
    source_type: str,
    value: str,
    options: DistillOptions,
    *,
    video_id: str | None = None,
) -> SourceIdentity:
    """Derive the **source fingerprint**, **lock key**, **bundle key** and fast-path.

    Pure dispatcher over ``source_type``. For ``"youtube"`` delegates to
    ``derive_youtube_identity`` (fast-path aware); for ``"local"`` delegates to
    ``derive_local_identity`` (fingerprint awaits I/O). No filesystem or network
    I/O is performed.

    ``value`` is the URL for YouTube and the path text for local. ``video_id``
    is the canonical id when already known (e.g. from ``youtube_metadata``);
    when ``None`` the fast-path id is used.
    """
    if source_type == "youtube":
        return derive_youtube_identity(value, options, video_id=video_id)
    if source_type == "local":
        return derive_local_identity(value, options)
    raise ValueError(f"source_type must be 'local' or 'youtube', got {source_type!r}")
