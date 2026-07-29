"""The YouTube client: URL parsing, metadata, and acquisition via yt-dlp.

This module owns how Distill names a YouTube source (the accepted URL forms and
the eleven-character id), what it reads about one (title, channel, description,
upload date) and how it fetches it (the yt-dlp invocation, its timeouts, and the
--no-playlist rule). It is a client for one source kind; `source` orchestrates
it and owns the bundle lifecycle around it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

from .capabilities import MISSING_TOOL_CODE, missing_tool_consequence
from .errors import DistillError, WarningRecord, warning
from .run_command import CommandResult, CommandTimeouts, run

YTDLP_METADATA_TIMEOUTS = CommandTimeouts(total_sec=120.0, idle_sec=60.0)
YTDLP_DOWNLOAD_TIMEOUTS = CommandTimeouts(total_sec=6 * 60 * 60.0, idle_sec=120.0)
YTDLP_SOCKET_TIMEOUT_SEC = 30
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}


@dataclass(frozen=True)
class YouTubeMetadata:
    video_id: str
    description: str
    warnings: list[WarningRecord]
    title: str | None = None
    channel: str | None = None
    upload_date: str | None = None


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


YOUTUBE_VIDEO_ID_PATTERN = re.compile(r"[0-9A-Za-z_-]{11}")
"""The id shape yt-dlp's YouTube extractor matches, and nothing wider.

Eleven characters of `[0-9A-Za-z_-]`, which is what the extractor's own regex
takes and what every published video id is. It is here because the **bundle
key** read off a URL has to be the key the run would publish under: yt-dlp
matches those eleven characters and ignores whatever follows, so a longer value
in `v=` names a video yt-dlp will report a *different* id for.
"""


def youtube_fast_path_video_id(url: str) -> str | None:
    """The id a **bundle** may be looked up by without asking yt-dlp, or `None`.

    `None` is not a refusal of the URL - it declines the shortcut, and the run
    resolves the id the way it always did. So this answers one question only:
    is the id written in this URL certainly the id yt-dlp resolves for it?

    It is not, in three ways, and each one would key a lookup on a string the
    run cannot publish under:

    - **A playlist is attached.** `watch?v=A&list=P` used to be the playlist to
      yt-dlp: `--dump-json` emitted one document per entry, which does not parse
      as one, and `canonical_youtube_id` fell back to `--print id` and took the
      *last* line, so the URL published under some other entry's id. Every
      single-video invocation now carries `NO_PLAYLIST_ARG`, so the id written
      in such a URL *is* the id a run publishes under, and the shortcut is
      declined here on the narrower ground that widening it changes which
      **bundle key** a URL is looked up by - a cache decision, taken
      deliberately or not at all. Declining still costs one resolution.
    - **The id is not id-shaped.** yt-dlp's extractor matches exactly eleven
      `[0-9A-Za-z_-]` characters and ignores trailing material, so
      `watch?v=YE7VzlLtp-4x` is published under `YE7VzlLtp-4`. Reading twelve
      characters off the URL keys a **bundle** nothing ever wrote - a false
      **cache miss**, and with yt-dlp uninstalled an `E_MISSING_TOOL` over data
      that is on disk. Truncating to eleven instead would be guessing at
      another tool's parser; declining costs one resolution.
    - **The URL names more than one.** `watch?v=A&v=B`, or a path with segments
      after the id. Which one yt-dlp picks is its business, not something to
      infer here.

    Case is preserved, because a video id is case-sensitive and `A` and `a` are
    different videos.
    """
    parsed = urlparse(url)
    if parsed.netloc.lower() not in YOUTUBE_HOSTS:
        return None
    # `keep_blank_values`, for the reason `normalize_youtube_url` twenty lines
    # above keeps blanks: `list=` with nothing after it is a `list` param, and
    # what yt-dlp does with an empty one is its business. The default drops it,
    # so `watch?v=A&list=` took the fast path while `watch?v=A&list=P` did not -
    # one URL form deciding a **bundle key** on a query-parser default.
    query = parse_qs(parsed.query, keep_blank_values=True)
    if "list" in query:
        return None
    segments = [segment for segment in parsed.path.split("/") if segment]
    if parsed.netloc.lower() == "youtu.be":
        candidates = segments[:1] if len(segments) == 1 else []
    elif any(parsed.path.startswith(prefix) for prefix in ("/shorts/", "/embed/", "/live/", "/v/")):
        candidates = segments[1:2] if len(segments) == 2 else []
    else:
        candidates = query.get("v", [])
    if len(candidates) != 1 or not YOUTUBE_VIDEO_ID_PATTERN.fullmatch(candidates[0]):
        return None
    return candidates[0]


def youtube_url_names_one_video(url: str) -> bool:
    """Whether the URL's own video id is the id yt-dlp will resolve for it.

    The boundary of the R-49 reorder rather than a detail of it. Reading a
    **bundle key** off the URL is only sound where the URL's id is the id a run
    would have published under; where it is not, the cache is asked the question
    it was always asked, after resolution, and yt-dlp is needed to ask it.
    Serving a bundle keyed by `A` for a URL this Distill would publish under `B`
    would be a cache hit for a bundle the same URL cannot produce.

    `youtube_fast_path_video_id` is the decision; this is the same answer as a
    question.
    """
    return youtube_fast_path_video_id(url) is not None


def _validated_youtube_video_id(value: object) -> str | None:
    video_id = str(value or "").strip()
    if YOUTUBE_VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        return None
    return video_id


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


NO_PLAYLIST_ARG = "--no-playlist"
"""What makes a `watch?v=A&list=P` URL name the video it says it names.

yt-dlp reads a `list` parameter as the thing being asked for, so without this a
URL an operator copied out of a playlist page resolves to every entry of the
playlist: `--dump-json` emits one document per entry, `--print id` one line per
entry, and the download points one output template at all of them. Which entry
the run then proceeds against is whichever landed on `source.mp4`.

Every yt-dlp invocation about *one video* carries it, and the one invocation
whose subject really is a playlist - the listing a playlist job starts with -
does not. That is why it is a parameter here rather than a constant folded into
the shared argv: a flag added unconditionally would make a playlist job
enumerate one video.
"""


def _ytdlp_command(extra_args: list[str], url: str, *, names_one_video: bool = True) -> list[str]:
    """Build a yt-dlp argv with a stall guard and a `--` terminator.

    The `--` before the URL stops a value that begins with `-` from being parsed
    as a yt-dlp option (argument injection), and `--socket-timeout` lets yt-dlp
    abort a stalled connection on its own.

    `names_one_video` defaults true because all but one caller is asking about a
    single video, and the default that has to be remembered is the one that gets
    forgotten.
    """
    return [
        "yt-dlp",
        "--socket-timeout",
        str(YTDLP_SOCKET_TIMEOUT_SEC),
        *([NO_PLAYLIST_ARG] if names_one_video else []),
        *extra_args,
        "--",
        url,
    ]


def _run_ytdlp(
    extra_args: list[str],
    url: str,
    *,
    timeouts: CommandTimeouts = YTDLP_METADATA_TIMEOUTS,
    names_one_video: bool = True,
) -> CommandResult:
    """Run a non-downloading yt-dlp invocation and hand back what it produced.

    `check=False`: every caller inspects `returncode` itself, because "yt-dlp
    ran and said no" means something different at each one - an unresolvable id
    is fatal, an unreadable description is a **degradation**. A yt-dlp that is
    absent or wedged still raises, since there is no answer to inspect.
    """
    return run(
        _ytdlp_command(extra_args, url, names_one_video=names_one_video),
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
        "yt-dlp could not read YouTube provenance metadata",
    )


def _metadata_text(payload: Mapping[str, Any], name: str) -> str:
    return str(payload.get(name) or "").strip()


def _first_description_paragraph(description: str) -> str:
    normalized = description.replace("\r\n", "\n").replace("\r", "\u2029")
    return re.split(r"(?:\u2029|\n[ \t]*\n)", normalized.strip(), maxsplit=1)[0]


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
    try:
        proc = _run_ytdlp(["--skip-download", "--dump-json"], url)
    except DistillError:
        video_id = youtube_fast_path_video_id(url)
        if video_id is None:
            raise
        return YouTubeMetadata(
            video_id=video_id,
            description="",
            warnings=[_metadata_unavailable()],
        )
    # Carried whichever branch answers: a metadata invocation that lost part of
    # its own output (R-33) is the same loss whether or not the document parsed.
    probe_warnings = list(proc.warnings)
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, Mapping):
            video_id = _validated_youtube_video_id(payload.get("id"))
            if video_id is not None:
                channel = _metadata_text(payload, "channel")
                if not channel:
                    channel = _metadata_text(payload, "uploader")
                return YouTubeMetadata(
                    video_id=video_id,
                    description=_metadata_text(payload, "description"),
                    warnings=probe_warnings,
                    title=_metadata_text(payload, "title") or None,
                    channel=channel or None,
                    upload_date=_metadata_text(payload, "upload_date") or None,
                )

    video_id = youtube_fast_path_video_id(url)
    if video_id is not None:
        return YouTubeMetadata(
            video_id=video_id,
            description="",
            warnings=[*probe_warnings, _metadata_unavailable()],
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
    raw_video_id = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    video_id = _validated_youtube_video_id(raw_video_id)
    if video_id is None:
        raise DistillError("E_YTDLP", "youtube", "yt-dlp returned an invalid video id")
    return video_id
