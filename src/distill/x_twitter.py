"""The X client: URL parsing, metadata, and acquisition via yt-dlp.

This module owns how Distill names an X source (the accepted post URL forms and
the status id) and what it reads about one (title, uploader, post text, upload
date). The yt-dlp invocation is shared machinery - `acquisition` stages and
validates the download, and `run_command` owns the process boundary - so what
lives here is only what an X URL *means*.

Identity never comes from yt-dlp. The extractor resolves a post URL to a media
id that can differ from the id in the URL (a quote tweet resolves to the
quoted media's id), so the **source fingerprint** is keyed on the URL's own
status id - see ADR-0008. That keeps a cache hit free of yt-dlp (R-49), at the
recorded cost that a post and its quoted original produce two bundles.

One post, one video (ADR-0008): a post carrying several videos is processed as
its first video. Downloads pin `--playlist-items 1` so exactly one media file
is staged, and metadata reads the first document `--dump-json` emits, warning
when the count shows the post carried more.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .capabilities import MISSING_TOOL_CODE, missing_tool_consequence
from .errors import DistillError, WarningRecord, warning
from .run_command import CommandResult, run

# Shared with the YouTube client deliberately: one tool, one socket policy, and
# one rule (the `--` terminator) that keeps a URL from being parsed as an
# option. Everything else about an invocation is the client's own.
from .youtube import (
    NO_PLAYLIST_ARG,
    YTDLP_METADATA_TIMEOUTS,
    YTDLP_SOCKET_TIMEOUT_SEC,
)

X_HOSTS = {
    "x.com",
    "www.x.com",
    "mobile.x.com",
    "twitter.com",
    "www.twitter.com",
    "mobile.twitter.com",
    "m.twitter.com",
}

X_STATUS_ID_PATTERN = re.compile(r"[0-9]+")
"""The id shape the extractor's own `_VALID_URL` matches: digits, unconstrained.

There is no width rule to inherit. Unlike YouTube's eleven-character id, the X
extractor takes a digit run and resolves exactly what the URL carries, so the id read
here is the id a run publishes under by construction - the fast path needs no
narrower shape to stay sound (ADR-0008).
"""

# Accepted path forms, everything the extractor matches for status posts and
# nothing else. The id is the last segment; anything after it - `/video/2`,
# `/photo/1`, analytics suffixes - refuses, because a suffix naming a different
# video of the same post would key the wrong media under this URL's id.
_X_STATUS_PATH = re.compile(r"/(?:(?:i/web|[^/]+)/status(?:es)?)/(?P<status_id>[0-9]+)/?$")


@dataclass(frozen=True)
class XMetadata:
    status_id: str
    description: str
    warnings: list[WarningRecord]
    title: str | None = None
    uploader: str | None = None
    upload_date: str | None = None
    video_count: int = 1


def normalize_x_url(url: str) -> str:
    """Canonicalize an X post URL: strip query and fragment, keep the path.

    The sharing UI appends tracking query strings (`?s=20&t=...`); none of them
    name media. The canonical form keeps the scheme and host the URL came with -
    `x.com` and `twitter.com` name the same post, and the canonical URL written
    into **provenance** is built from the id alone, so nothing here needs to
    pick one host over the other.
    """
    parts = urlsplit(url)
    if parts.query or parts.fragment:
        return urlunsplit(parts._replace(query="", fragment=""))
    return url


def ensure_x_host(url: str) -> None:
    """Reject non-X hosts (and option-injection values) before yt-dlp runs."""
    host = urlsplit(url).netloc.lower()
    if host not in X_HOSTS:
        raise DistillError(
            "E_BAD_URL", "x", "only x.com or twitter.com post URLs are supported", {"url": url}
        )


def parse_x_status_id(url: str) -> str:
    """The status id an X post URL carries, or `E_BAD_URL`.

    Refusals carry the reason a form was refused, because the forms that fail
    here are ones a user arrives at deliberately: `/video/2` names a video the
    convention does not process (ADR-0008), and everything else is not a post
    URL at all. The id is what the **source fingerprint** hashes, so this - and
    not the extractor - is the authority on what a run names.
    """
    ensure_x_host(url)
    path = urlsplit(url).path
    match = _X_STATUS_PATH.fullmatch(path)
    if match is None:
        if re.fullmatch(r"/[^/]+/status(?:es)?/[0-9]+/(?:video|photo)/[0-9]+", path):
            raise DistillError(
                "E_BAD_URL",
                "x",
                "Distill processes the first video of a post; use the post URL "
                f"without the /video or /photo suffix: {url}",
                {"url": url},
            )
        raise DistillError(
            "E_BAD_URL",
            "x",
            f"could not find a post id in URL: {url}",
            {"url": url},
        )
    return match.group("status_id")


def x_fast_path_status_id(url: str) -> str | None:
    """The status id a **bundle** may be looked up by without asking yt-dlp.

    Sound by construction, where YouTube's fast path needs three refusals to
    stay sound: the extractor matches the same digit run this module validates, so
    the id in the URL is the id a run publishes under whenever the URL parses
    at all. `None` therefore means only "not an X post URL" - the caller's
    refusal, not a declined shortcut.
    """
    try:
        return parse_x_status_id(url)
    except DistillError:
        return None


def _x_command(extra_args: list[str], url: str) -> list[str]:
    """Build a yt-dlp argv for one X post, with a stall guard and a terminator.

    `NO_PLAYLIST_ARG` is carried for the same reason the YouTube client carries
    it - the default that has to be remembered is the one that gets forgotten -
    even though a post URL carries no `list` parameter for it to act on. The
    `--` before the URL stops a value that begins with `-` from being parsed as
    a yt-dlp option (argument injection).
    """
    return [
        "yt-dlp",
        "--socket-timeout",
        str(YTDLP_SOCKET_TIMEOUT_SEC),
        NO_PLAYLIST_ARG,
        *extra_args,
        "--",
        url,
    ]


def _run_x_metadata(args: list[str], url: str) -> CommandResult:
    """Run a non-downloading yt-dlp invocation and hand back what it produced.

    `check=False`: the caller inspects `returncode` itself, because a yt-dlp
    that ran and said no costs the metadata and not the run - only a tool that
    is absent or wedged raises, and that is the capability table's answer
    (`missing_tool_consequence` at the call site).
    """
    return run(
        _x_command(args, url),
        stage="x",
        total_timeout_sec=YTDLP_METADATA_TIMEOUTS.total_sec,
        idle_timeout_sec=YTDLP_METADATA_TIMEOUTS.idle_sec,
        check=False,
        error_code="E_YTDLP",
    )


def _metadata_unavailable() -> WarningRecord:
    return warning(
        "x",
        "metadata_unavailable",
        "yt-dlp could not read X post provenance metadata",
    )


def _metadata_text(payload: Mapping[str, Any], name: str) -> str:
    return str(payload.get(name) or "").strip()


def _documents(stdout: str) -> list[Mapping[str, Any]]:
    """The metadata documents yt-dlp emitted, one per video of the post.

    `--dump-json` prints one document per playlist entry, and a multi-video
    post is a playlist - so the line count is how the run learns the post
    carried more video than the convention processes. A line that does not
    parse is skipped rather than fatal: the metadata is best-effort, and the
    first parseable document is the one provenance reads.
    """
    documents: list[Mapping[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            documents.append(payload)
    return documents


def x_metadata(url: str, extra_args: list[str] | None = None) -> XMetadata:
    """Read one post's provenance metadata, best-effort.

    The status id comes from the URL, never from the document (ADR-0008): a
    quote tweet's document names the quoted media's id, and keying identity on
    it would make every cache lookup a question only yt-dlp can answer. The
    document contributes description and, when it answers, title, uploader and
    upload date.

    A post that carries several videos emits one document per video. The first
    is the one this run processes; the rest are counted into a **warning** so
    the bundle says what the convention left out. `extra_args` carries the
    run's authentication (`youtube.ytdlp_auth_args`), never identity.
    """
    status_id = parse_x_status_id(url)
    try:
        proc = _run_x_metadata([*(extra_args or []), "--skip-download", "--dump-json"], url)
    except DistillError as exc:
        # An absent tool is the capability table's decision here too. The
        # metadata being best-effort does not make an absent yt-dlp a
        # degradation - yt-dlp is a **required capability**, so this raises
        # (ADR-0002, R-34), with the table's own message. Anything else is a
        # yt-dlp that wedged, which is fatal here for the reason
        # `youtube_metadata` re-raises: a stalled extractor predicts a stalled
        # download.
        if exc.code == MISSING_TOOL_CODE:
            missing_tool_consequence("x", "yt-dlp", cause=exc)
        raise
    warnings = list(proc.warnings)
    if proc.returncode != 0:
        return XMetadata(
            status_id=status_id,
            description="",
            warnings=[*warnings, _metadata_unavailable()],
        )
    documents = _documents(proc.stdout)
    if not documents:
        return XMetadata(
            status_id=status_id,
            description="",
            warnings=[*warnings, _metadata_unavailable()],
        )
    payload = documents[0]
    if len(documents) > 1:
        warnings.append(
            warning(
                "x",
                "multi_video_post",
                f"post carries {len(documents)} videos; processing the first",
            )
        )
    return XMetadata(
        status_id=status_id,
        description=_metadata_text(payload, "description"),
        warnings=warnings,
        title=_metadata_text(payload, "title") or None,
        uploader=_metadata_text(payload, "uploader")
        or _metadata_text(payload, "uploader_id")
        or None,
        upload_date=_metadata_text(payload, "upload_date") or None,
        video_count=len(documents),
    )


# What pins the download to the first video of a multi-video post. A module
# constant rather than an inline literal because it is the convention's whole
# mechanism (ADR-0008): without it, yt-dlp stages every video of the post
# against one output template, and which one survives is whichever finished
# last. The downloader carries these args only for an X source.
PLAYLIST_ITEMS_FIRST = ("--playlist-items", "1")
