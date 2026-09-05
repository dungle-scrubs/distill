"""Source fingerprints, acquisition lock keys, and bundle keys.

Local fingerprints read bounded samples or complete content according to cache
mode. YouTube fingerprints and lock keys hash the resolved video id. Bundle
keys combine the source fingerprint and options hash; URL parsing lives in youtube.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .errors import DistillError
from .progress import ProgressReporter

CONTENT_HASH_LIMIT_BYTES = 5 * 1024 * 1024 * 1024
FINGERPRINT_SAMPLE_BYTES = 64 * 1024
FINGERPRINT_INTERIOR_ANCHORS = 7


def youtube_fingerprint(video_id: str) -> str:
    """Identify the media named by a resolved YouTube video id."""
    return hashlib.sha256(video_id.encode()).hexdigest()


def youtube_lock_key(video_id: str) -> str:
    """Coordinate acquisition of a resolved video, independent of processing options."""
    return youtube_fingerprint(video_id)


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


def bundle_key(source_fingerprint: str, opts_hash: str) -> str:
    return hashlib.sha256(f"{source_fingerprint}:{opts_hash}".encode()).hexdigest()
