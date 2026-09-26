"""The X client: what a post URL names, and what a run of one records.

ADR-0008 is the design this file holds. The one property everything here
protects: a run keys the **bundle** on the status id the URL carries, never on
the media id yt-dlp's document reports - because the extractor may resolve a
quote tweet to the quoted media's id, and identity that needs yt-dlp to ask is
identity that makes every cache lookup cost a tool invocation (R-49).

Unit tests here parse URLs and read metadata off faked yt-dlp output; the
integration tests drive `process_x_video` end to end on faked tools, the way
`test_cache_before_capability.py` does for YouTube.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fake_tools import FAKE_FFMPEG_WRITES_A_REAL_PNG, FAKE_FFPROBE
from runtime_fakes import configure_run
from test_local_integration import fake_transcribe, make_short_screencast

from distill import pipeline as distill_session
from distill import source as distill_source
from distill import source_identity, x_twitter
from distill.acquisition import AcquiredSource, AcquisitionLease
from distill.artifacts import Provenance
from distill.bundle_store import validate_manifest_schema
from distill.errors import DistillError
from distill.progress import ProgressReporter
from distill.source import SourceInfo
from distill.x_twitter import XMetadata

STATUS_ID = "1234567890123456789"
URL = f"https://x.com/user/status/{STATUS_ID}"

# What a quote tweet's document reports instead of the URL's id: the *quoted*
# media's id. Identity keyed on this would make two URLs of one media collide
# and every cache hit cost a yt-dlp invocation - the failure ADR-0008 refuses.
DOCUMENT_ID = "999999999999999999"


X_FAKE_YTDLP = f"""
import json, pathlib, sys

argv = sys.argv[1:]
if "--dump-json" in argv:
    document = {{
        "id": "{DOCUMENT_ID}",
        "title": "Example - a post about a build",
        "uploader": "Example",
        "uploader_id": "example",
        "upload_date": "20240115",
        "description": "a post about a build",
    }}
    if pathlib.Path(sys.argv[0]).with_name("two_videos").exists():
        sys.stdout.write(json.dumps(document) + "\\n")
        sys.stdout.write(json.dumps({{**document, "title": document["title"] + " #2"}}) + "\\n")
    else:
        sys.stdout.write(json.dumps(document) + "\\n")
elif "-o" in argv:
    template = argv[argv.index("-o") + 1]
    media = pathlib.Path(template.replace("%(ext)s", "mp4"))
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"video")
else:
    pass  # the id line the YouTube client reads; X identity never asks yt-dlp
"""


def x_args(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "url": URL,
        "output_dir": str(tmp_path / "cache"),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        **overrides,
    }


def a_producing_path(fake_tool: Callable[[str, str], Path], *, two_videos: bool = False) -> Path:
    """Install every tool a producing X run needs; return yt-dlp's fake.

    The marker file `two_videos` sits beside the fake yt-dlp script rather than
    in the script text, so one fixture serves both the single- and multi-video
    tests without a second script to keep in step.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_A_REAL_PNG)
    fake_tool("ffprobe", FAKE_FFPROBE)
    ytdlp = fake_tool("yt-dlp", X_FAKE_YTDLP)
    if two_videos:
        ytdlp.with_name("two_videos").write_text("")
    return ytdlp


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (f"https://x.com/user/status/{STATUS_ID}", STATUS_ID),
        (f"https://twitter.com/user/status/{STATUS_ID}", STATUS_ID),
        (f"https://www.twitter.com/user/status/{STATUS_ID}", STATUS_ID),
        (f"https://mobile.x.com/user/status/{STATUS_ID}", STATUS_ID),
        (f"https://m.twitter.com/user/status/{STATUS_ID}", STATUS_ID),
        (f"https://x.com/user/statuses/{STATUS_ID}", STATUS_ID),
        (f"https://x.com/i/web/status/{STATUS_ID}", STATUS_ID),
        (f"https://x.com/user/status/{STATUS_ID}/", STATUS_ID),
        (f"https://x.com/user/status/{STATUS_ID}?s=20&t=abc", STATUS_ID),
        (f"https://x.com/user/status/{STATUS_ID}#m", STATUS_ID),
        ("https://x.com/user/status/1", "1"),
    ],
)
def test_accepted_post_forms_carry_their_status_id(url: str, expected: str) -> None:
    assert x_twitter.parse_x_status_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/user/status/1234567890123456789",
        "https://x.com/user/status/",
        "https://x.com/user/status/12abc34567890123456789",
        "https://x.com/user/status/1234567890123456789/video/2",
        "https://x.com/user/status/1234567890123456789/photo/1",
        "https://x.com/user",
        "https://x.com/i/videos/tweet/1234567890123456789",
        "relative/path",
    ],
)
def test_refused_forms_refuse_before_any_tool_runs(url: str) -> None:
    with pytest.raises(DistillError) as failure:
        x_twitter.parse_x_status_id(url)
    assert failure.value.code == "E_BAD_URL"


def test_a_video_suffix_refusal_names_the_convention() -> None:
    """`/video/2` names a video the first-video convention does not process.

    Keyed under the URL's id it would serve the wrong media's bundle - the
    hazard the YouTube fast path declines `list=` for - so the refusal says
    what to use instead rather than pretending the URL did not parse.
    """
    with pytest.raises(DistillError) as failure:
        x_twitter.parse_x_status_id(f"https://x.com/user/status/{STATUS_ID}/video/2")
    assert "without the /video or /photo suffix" in failure.value.message


def test_normalize_strips_tracking_query_and_fragment() -> None:
    normalized = x_twitter.normalize_x_url(f"{URL}?s=20&t=abc#m")
    assert normalized == URL


def test_the_fast_path_answers_every_parsing_url_and_only_those() -> None:
    """Sound by construction: the extractor matches the same digits.

    There is no narrower rule to inherit and no case to decline, so `None`
    means only "not an X post URL" - the one structural difference from the
    YouTube fast path, which declines three URL shapes to stay sound.
    """
    assert x_twitter.x_fast_path_status_id(URL) == STATUS_ID
    assert x_twitter.x_fast_path_status_id(f"{URL}/video/1") is None
    assert x_twitter.x_fast_path_status_id("https://youtu.be/abcdefghijk") is None


def test_x_fingerprints_live_in_their_own_domain() -> None:
    """An X id and a YouTube id that spelled the same string cannot share a key.

    Bare, `sha256(id)` would give both kinds the same **source fingerprint**
    for one spelling - and the same **bundle key** whenever their **options
    hashes** agreed. The `x:` prefix retires that; YouTube's function stays
    bare so existing bundles keep their addresses.
    """
    shared_spelling = "12345678901"  # eleven digits: a legal YouTube id shape
    assert source_identity.x_status_fingerprint(
        shared_spelling
    ) != source_identity.youtube_fingerprint(shared_spelling)
    assert source_identity.x_status_lock_key(
        shared_spelling
    ) == source_identity.x_status_fingerprint(shared_spelling)


def test_metadata_keeps_the_url_id_over_the_document_id(
    fake_tool: Callable[[str, str], Path],
) -> None:
    """The document names the quoted media; identity stays with the URL.

    This is ADR-0008's core property, exercised against a document that
    disagrees: `status_id` is the URL's, and the document contributes only
    provenance text.
    """
    a_producing_path(fake_tool)
    metadata = x_twitter.x_metadata(URL)
    assert metadata.status_id == STATUS_ID
    assert metadata.status_id != DOCUMENT_ID
    assert metadata.title == "Example - a post about a build"
    assert metadata.uploader == "Example"
    assert metadata.upload_date == "20240115"
    assert metadata.description == "a post about a build"
    assert metadata.video_count == 1
    assert metadata.warnings == []


def test_ytdlp_auth_args_reach_the_tool_and_the_identity_hash_not() -> None:
    """Cookies are **machine-local claims**: tool argv, never bundle identity.

    The mapping is the whole of the contract on the argv side; the options
    hash side is held by `cache_key=False`, and this is the assertion that
    a run with a cookie jar and a run without one publish under the same
    **bundle key**.
    """
    from distill.options import DistillOptions
    from distill.youtube import ytdlp_auth_args

    assert ytdlp_auth_args(None, None) == []
    assert ytdlp_auth_args("jar.txt", None) == ["--cookies", "jar.txt"]
    assert ytdlp_auth_args(None, "safari") == ["--cookies-from-browser", "safari"]
    assert ytdlp_auth_args("jar.txt", "safari") == [
        "--cookies",
        "jar.txt",
        "--cookies-from-browser",
        "safari",
    ]

    assert DistillOptions(cookies="jar.txt").opts_hash("x") == DistillOptions().opts_hash("x")
    assert DistillOptions(cookies_from_browser="safari").opts_hash("youtube") == (
        DistillOptions().opts_hash("youtube")
    )


def test_the_run_passes_its_cookie_options_to_yt_dlp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The argv the metadata invocation builds carries the run's auth options.

    Driven through the resolver's X branch with yt-dlp's process boundary
    captured, so the wiring from `DistillOptions.cookies` to the tool's argv
    is held end to end (the run then fails at the deliberately useless
    metadata answer, before any download is attempted).
    """
    from distill.configuration import resolve_run_config
    from distill.run_command import CommandResult

    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> CommandResult:
        captured["argv"] = argv
        return CommandResult(
            tool="yt-dlp",
            argv=tuple(argv),
            returncode=1,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_sec=0.0,
        )

    monkeypatch.setattr("distill.x_twitter.run", fake_run)
    monkeypatch.setattr("distill.acquisition.check_disk_floor", lambda _path, **_: None)

    options = resolve_run_config(
        {
            "url": URL,
            "output_dir": str(tmp_path / "cache"),
            "cache_mode": "fingerprint",
            "cookies": "jar.txt",
            "cookies_from_browser": "safari",
        }
    ).options

    with pytest.raises(DistillError):
        distill_source.resolve_source_for_processing("x", URL, options)

    argv = captured["argv"]
    assert argv[argv.index("--cookies") + 1] == "jar.txt"
    assert argv[argv.index("--cookies-from-browser") + 1] == "safari"
    assert "--dump-json" in argv
    assert argv[-1] == URL


def test_a_multi_video_post_reads_the_first_document_and_warns(
    fake_tool: Callable[[str, str], Path],
) -> None:
    a_producing_path(fake_tool, two_videos=True)
    metadata = x_twitter.x_metadata(URL)

    assert metadata.status_id == STATUS_ID
    assert metadata.video_count == 2
    assert metadata.title == "Example - a post about a build"
    assert [item["code"] for item in metadata.warnings] == ["multi_video_post"]
    assert "2 videos" in metadata.warnings[0]["message"]


def test_a_failed_metadata_probe_degrades_but_the_id_survives(
    fake_tool: Callable[[str, str], Path],
) -> None:
    """yt-dlp ran and said no: the run loses the provenance, not the identity."""
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_A_REAL_PNG)
    fake_tool("ffprobe", FAKE_FFPROBE)
    fake_tool(
        "yt-dlp",
        "import sys\\nsys.stderr.write('ERROR: unavailable\\\\n')\\nsys.exit(1)",
    )
    metadata = x_twitter.x_metadata(URL)

    assert metadata.status_id == STATUS_ID
    assert metadata.description == ""
    assert [item["code"] for item in metadata.warnings] == ["metadata_unavailable"]


def test_an_absent_yt_dlp_is_fatal_through_the_capability_table(
    fake_tool: Callable[[str, str], Path],
) -> None:
    """A required capability's absence is a **fatal error**, not a degradation.

    The metadata is best-effort; the tool is not. `fake_tool` replaces `PATH`
    with an empty directory beyond what each test installs, so yt-dlp is
    genuinely absent here and the table's own message is what raises.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_A_REAL_PNG)
    with pytest.raises(DistillError) as failure:
        x_twitter.x_metadata(URL)

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "yt-dlp"


def test_resolve_keys_identity_on_the_url_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution never asks the document who this source is.

    The downloader is handed the **lock key** of the URL's status id even while
    the metadata it accompanies reports a different media id, and the
    **source fingerprint** - the bundle key's first half - is that same id's.
    """
    from distill.options import DistillOptions

    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    lock = tmp_path / "lock.lock"
    lease = AcquisitionLease.take(STATUS_ID, lock)
    assert lease is not None
    acquired_urls: list[str] = []
    acquired_locks: list[str] = []

    class FakeDownloader:
        def acquire(
            self,
            url: str,
            lock_key: str,
            progress: ProgressReporter | None = None,
        ) -> AcquiredSource:
            acquired_urls.append(url)
            acquired_locks.append(lock_key)
            return AcquiredSource(path=video, lease=lease)

    monkeypatch.setattr("distill.acquisition.check_disk_floor", lambda _path, **_: None)
    monkeypatch.setattr("distill.media_inspect.probe_duration", lambda _path: (12.0, []))
    monkeypatch.setattr(
        "distill.x_twitter.x_metadata",
        lambda _url, *extra: XMetadata(
            status_id=STATUS_ID,
            description="a post about a build\n\nhttps://github.com/example/repo",
            title="Example - a post about a build",
            uploader="Example",
            upload_date="20240115",
            warnings=[],
        ),
    )

    output_root = tmp_path / "cache"
    output_root.mkdir()
    source = distill_source.SourceResolver().x_source(
        URL,
        DistillOptions(),
        output_root,
        FakeDownloader(),
        None,
    )

    assert acquired_urls == [URL]
    assert acquired_locks == [source_identity.x_status_lock_key(STATUS_ID)]
    assert source.source_type == "x"
    assert source.source_fingerprint == source_identity.x_status_fingerprint(STATUS_ID)
    assert source.resolved_options is not None
    assert source.source_hash == source_identity.bundle_key(
        source.source_fingerprint, source.resolved_options.opts_hash("x")
    )
    assert source.remote_source_id == STATUS_ID
    assert source.remote_lock_key == source_identity.x_status_lock_key(STATUS_ID)
    assert source.provenance is not None
    assert source.provenance.canonical_url == f"https://x.com/i/web/status/{STATUS_ID}"
    assert source.provenance.channel == "Example"
    assert source.provenance.description == "a post about a build"
    # The post text is **extracted text** and goes through the same related
    # link filter as a YouTube description; a code link survives, its label
    # redacted under the run's policy.
    assert [link.url for link in (source.related_links or [])] == [
        "https://github.com/example/repo"
    ]


def test_the_x_download_carries_the_first_video_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--playlist-items 1` is the convention's whole mechanism (ADR-0008).

    Without it yt-dlp stages every video of a multi-video post against one
    output template, and which one survives is whichever finished last. The
    command the downloader builds is the one place the convention could be
    lost, so the test reads the argv it hands to `stream`.
    """
    import distill.acquisition as acquisition_module
    from distill.run_command import CommandResult

    captured: dict[str, Any] = {}

    def fake_stream(command: list[str], **kwargs: Any) -> CommandResult:
        captured["command"] = command
        captured["stage"] = kwargs["stage"]
        return CommandResult(
            tool="yt-dlp",
            argv=tuple(command),
            returncode=0,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_sec=0.0,
        )

    monkeypatch.setattr(acquisition_module, "stream", fake_stream)
    downloader = acquisition_module.YtDlpDownloader(
        tmp_path,
        stage="x",
        label="X",
        download_args=x_twitter.PLAYLIST_ITEMS_FIRST,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "source.mp4").write_bytes(b"video")
    monkeypatch.setattr(acquisition_module, "validate_media_file", lambda _path, **_: [])
    downloader._download(URL, staging, None)

    command = captured["command"]
    assert "--playlist-items" in command
    assert command[command.index("--playlist-items") + 1] == "1"
    assert captured["stage"] == "x"
    assert command[-1] == URL


def _published_x_bundle(root: Path, bundle_key: str) -> Path:
    """A **bundle** on disk as a run of this URL would publish it."""
    bundle = root / bundle_key
    (bundle / "g1").mkdir(parents=True)
    (bundle / "g1" / "video.md").write_text("# cached render\n")
    (bundle / "g1" / "video.self-contained.md").write_text("# cached render\n")
    (bundle / "g1" / "transcript.json").write_text(json.dumps({"segments": []}))
    (bundle / "_manifest.json").write_text(
        json.dumps(
            {
                "pipeline_version": 1,
                "distill_version": "0.1.0",
                "source_type": "x",
                "bundle_key": bundle_key,
                "source_resolved_path": str(root / "source.mp4"),
                "duration_sec": 12.5,
                "options": {},
                "frame_count": 0,
                "transcript_present": True,
                "warning_count": 0,
                "frames": [],
                "warnings": [],
                "active_generation": "g1",
                "provenance": {
                    "canonical_url": f"https://x.com/i/web/status/{STATUS_ID}",
                    "duration_sec": 12.5,
                    "processed_at": "2026-07-29T14:20:00Z",
                },
            },
            indent=2,
        )
        + "\n"
    )
    return bundle


def test_a_published_x_bundle_is_served_without_yt_dlp(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-49 for the X kind: the cache answers before the capability is asked.

    The resolver never calls `x_metadata` on the hit path - the id came from
    the URL - so a run over a served bundle needs no yt-dlp installed. The
    miss path is the one that raises.
    """
    from distill.configuration import resolve_run_config

    a_producing_path(fake_tool)
    options = resolve_run_config({**x_args(tmp_path), "cache_mode": "fingerprint"}).options
    fingerprint = source_identity.x_status_fingerprint(STATUS_ID)
    assert options.output_dir is not None
    output_root = Path(options.output_dir)
    bundle_key = source_identity.bundle_key(fingerprint, options.opts_hash("x"))
    _published_x_bundle(output_root, bundle_key)
    (output_root / "source.mp4").write_bytes(b"video")

    metadata_calls: list[str] = []
    monkeypatch.setattr(
        "distill.x_twitter.x_metadata",
        lambda url: metadata_calls.append(url),
    )

    resolution = distill_source.resolve_source_for_processing("x", URL, options)

    assert resolution.source.source_type == "x"
    assert resolution.source.source_hash == bundle_key
    assert resolution.source.remote_source_id == STATUS_ID
    assert metadata_calls == []


def test_the_manifest_schema_accepts_an_x_canonical_url_and_refuses_other_urls() -> None:
    legacy = {
        "pipeline_version": 1,
        "distill_version": "0.1.0",
        "source_type": "x",
        "source_resolved_path": "source.mp4",
        "duration_sec": 12.5,
        "options": {},
        "frame_count": 0,
        "transcript_present": True,
        "warning_count": 0,
        "frames": [],
        "warnings": [],
        "bundle_key": "a" * 64,
    }

    def manifest_with(canonical_url: str) -> dict[str, Any]:
        return {
            **legacy,
            "provenance": {
                "canonical_url": canonical_url,
                "duration_sec": 12.5,
                "processed_at": "2026-07-29T14:20:00Z",
            },
        }

    validate_manifest_schema(
        manifest_with(f"https://x.com/i/web/status/{STATUS_ID}"),
        require_active_generation=False,
    )
    validate_manifest_schema(
        manifest_with("https://x.com/i/web/status/1"),
        require_active_generation=False,
    )
    for refused in (
        f"https://x.com/user/status/{STATUS_ID}",
        f"https://twitter.com/i/web/status/{STATUS_ID}",
        "https://example.com/status/1234567890123456789",
        f"https://x.com/i/web/status/{STATUS_ID}?s=20",
    ):
        with pytest.raises(DistillError) as raised:
            validate_manifest_schema(
                manifest_with(refused),
                require_active_generation=False,
            )
        assert raised.value.code == "E_BAD_MANIFEST"


def test_process_x_video_flows_through_the_shared_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The X command is the shared pipeline under an X resolution.

    Mirrors the YouTube integration test: a real screencast, a faked provider
    resolution, real transcription and frame work - so a wiring mistake in the
    command, the tool registry or the source type fails here and not in
    production.
    """
    video = tmp_path / "x.mp4"
    make_short_screencast(video)
    fingerprint = source_identity.x_status_fingerprint(STATUS_ID)

    def fake_resolve(
        _provider: object,
        request: Any,
        downloader: object = None,
        metadata: object = None,
    ) -> SourceInfo:
        _ = (downloader, metadata)
        return SourceInfo(
            source_type="x",
            resolved_path=video,
            duration_sec=1.0,
            source_fingerprint=fingerprint,
            source_hash=source_identity.bundle_key(fingerprint, request.options.opts_hash("x")),
            warnings=[],
            provenance=Provenance(
                canonical_url=f"https://x.com/i/web/status/{STATUS_ID}",
                duration_sec=1.0,
                processed_at="2026-07-29T14:20:00Z",
            ),
            remote_source_id=STATUS_ID,
            remote_lock_key=source_identity.x_status_lock_key(STATUS_ID),
        )

    monkeypatch.setattr(distill_source.XSourceProvider, "resolve", fake_resolve)
    configure_run(monkeypatch, transcribe=fake_transcribe)
    monkeypatch.setattr(
        "distill.x_twitter.x_metadata",
        lambda _url, *extra: XMetadata(STATUS_ID, "", []),
    )

    args = {
        "url": URL,
        "output_dir": str(tmp_path / "cache"),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        "max_keyframes": 3,
        "max_static_window_sec": 1,
    }

    response = distill_session.process_x_video(args)

    assert Path(response["transcript_path"]).exists()
    assert response["frames"]
    assert response["cached"] is False
    # The caller-facing extraction report. A run with the vision stage off is
    # told so, with the advice that says what a differently-configured run
    # would add - the gap the poteto-post proof run exposed.
    assert response["visual_extraction"] == {
        "mode": "none",
        "frames_total": len(response["frames"]),
        "frames_with_readings": 0,
        "frames_with_ocr_text": 0,
        "advice": response["visual_extraction"]["advice"],
    }
    assert "OCR enabled" in response["visual_extraction"]["advice"]

    second = distill_session.process_x_video(args)
    assert second["cached"] is True
