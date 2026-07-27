from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from ast import literal_eval
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import lease_is_held
from fake_tools import (
    FAKE_FFPROBE,
    FAKE_YTDLP_DOWNLOAD,
    FAKE_YTDLP_FAILING,
    fake_ffprobe_flooding_stderr,
)

from distill.errors import DistillError
from distill.options import OPTION_DEFAULTS, DistillOptions
from distill.progress import ProgressReporter
from distill.run_command import OUTPUT_CAP_BYTES, TRUNCATION_WARNING_CODE
from distill.source import (
    CONTENT_HASH_LIMIT_BYTES,
    FINGERPRINT_INTERIOR_ANCHORS,
    FINGERPRINT_SAMPLE_BYTES,
    AcquiredSource,
    AcquisitionLease,
    YoutubeDownloader,
    YouTubeMetadata,
    _ytdlp_command,
    local_fingerprint,
    normalize_youtube_url,
    parse_byte_amount,
    parse_youtube_url,
    parse_ytdlp_progress,
    probe_duration,
    sensitive_path_match,
    source_hash,
    validate_output_root,
    youtube_description,
    youtube_lock_key,
    youtube_metadata,
    youtube_source_info,
)


def test_ytdlp_command_guards_against_argument_injection() -> None:
    command = _ytdlp_command(["--skip-download", "--dump-json"], "--config-location=/etc/evil")
    # The URL sits after a literal `--`, so a value beginning with `-` cannot be
    # parsed as a yt-dlp option, and a socket timeout bounds a stalled connection.
    assert command[-2:] == ["--", "--config-location=/etc/evil"]
    assert "--socket-timeout" in command


def test_non_youtube_host_is_rejected_before_download() -> None:
    with pytest.raises(DistillError) as exc:
        parse_youtube_url("https://evil.example.com/watch?v=abc")
    assert exc.value.code == "E_BAD_URL"


def test_ensure_youtube_host_rejects_injection_and_foreign_hosts() -> None:
    from distill.source import ensure_youtube_host

    for bad in ("--exec=touch /tmp/pwned", "-J", "https://evil.example.com/playlist?list=x"):
        with pytest.raises(DistillError) as exc:
            ensure_youtube_host(bad)
        assert exc.value.code == "E_BAD_URL"
    # A real playlist URL (no video id) passes the host-only check.
    ensure_youtube_host("https://www.youtube.com/playlist?list=PLabc")


# A fake yt-dlp that records the argv it was handed, so a test can assert on the
# invocation that really happened rather than on a patched call. The recording
# path is passed out of band, in an environment variable Distill never reads.
FAKE_YTDLP_RECORDING_ARGV = """
import os, pathlib, sys

pathlib.Path(os.environ["FAKE_YTDLP_ARGV_FILE"]).write_text(repr(sys.argv[1:]))
sys.stdout.write("https://youtu.be/aaa\\n")
"""


def test_playlist_listing_uses_guarded_ytdlp_command(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The guard is on the argv yt-dlp actually receives, not on a patched call."""
    from distill.pipeline import youtube_playlist_urls

    fake_tool("yt-dlp", FAKE_YTDLP_RECORDING_ARGV)
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("FAKE_YTDLP_ARGV_FILE", str(argv_file))

    urls = youtube_playlist_urls("https://www.youtube.com/playlist?list=PLabc", 10)

    assert urls == ["https://youtu.be/aaa"]
    command = literal_eval(argv_file.read_text())
    # URL sits after a literal `--`, and yt-dlp gets a socket timeout.
    assert command[-2:] == ["--", "https://www.youtube.com/playlist?list=PLabc"]
    assert "--socket-timeout" in command


def test_non_positive_max_static_window_is_rejected() -> None:
    for bad in (0.0, -5.0):
        with pytest.raises(DistillError) as exc:
            DistillOptions.from_args({"max_static_window_sec": bad})
        assert exc.value.code == "E_BAD_OPTIONS"
        assert "max_static_window_sec" in exc.value.message


def test_dataclass_defaults_match_option_specs() -> None:
    # Guards against the two default tables (OPTION_SPECS and the dataclass
    # fields) drifting apart.
    defaults = DistillOptions()
    for name, expected in OPTION_DEFAULTS.items():
        assert getattr(defaults, name) == expected, name


def test_local_opts_hash_includes_cache_mode_but_youtube_excludes_it() -> None:
    fingerprint = DistillOptions(cache_mode="fingerprint")
    content = DistillOptions(cache_mode="content")
    assert fingerprint.opts_hash("local") != content.opts_hash("local")
    assert fingerprint.opts_hash("youtube") == content.opts_hash("youtube")


def test_local_vision_settings_affect_cache_key() -> None:
    no_caption = DistillOptions(caption_frames=False)
    caption = DistillOptions(caption_frames=True)
    larger_model = DistillOptions(caption_frames=True, local_vision_model="qwen3-vl:32b")

    assert no_caption.opts_hash("local") != caption.opts_hash("local")
    assert caption.opts_hash("local") != larger_model.opts_hash("local")


def test_source_hash_uses_fingerprint_and_options_hash() -> None:
    assert source_hash("abc", "def") == hashlib.sha256(b"abc:def").hexdigest()


def test_content_mode_refuses_files_over_5gb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"x")

    class FakeStat:
        st_size = CONTENT_HASH_LIMIT_BYTES + 1
        st_mtime_ns = 1

    monkeypatch.setattr(Path, "stat", lambda _self: FakeStat())
    with pytest.raises(DistillError) as exc:
        local_fingerprint(path, "content")
    assert exc.value.code == "E_CONTENT_HASH_TOO_LARGE"


def test_content_mode_reports_byte_accurate_fingerprint_progress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"a" * (1024 * 1024) + b"b")
    progress = ProgressReporter()

    local_fingerprint(path, "content", progress)

    events = [event for event in progress.events if event.mechanism == "source_fingerprint"]
    assert events
    assert events[-1].percent == 100.0
    assert events[-1].detail == {
        "cache_mode": "content",
        "bytes_read": path.stat().st_size,
        "total_bytes": path.stat().st_size,
    }


def test_fingerprint_mode_reports_bounded_sample_progress(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"a" * (1024 * 1024))
    progress = ProgressReporter()

    local_fingerprint(path, "fingerprint", progress)

    events = [event for event in progress.events if event.mechanism == "source_fingerprint"]
    anchors = FINGERPRINT_INTERIOR_ANCHORS + 2
    assert [event.detail["sample"] for event in events if event.detail] == [
        "first",
        *["interior"] * FINGERPRINT_INTERIOR_ANCHORS,
        "last",
    ]
    assert [event.percent for event in events] == [
        round((index + 1) / anchors * 100, 3) for index in range(anchors)
    ]
    assert events[0].detail == {
        "cache_mode": "fingerprint",
        "sample": "first",
        "offset": 0,
        "samples_done": 1,
        "samples_total": anchors,
    }
    assert events[-1].detail == {
        "cache_mode": "fingerprint",
        "sample": "last",
        "offset": 1024 * 1024 - FINGERPRINT_SAMPLE_BYTES,
        "samples_done": anchors,
        "samples_total": anchors,
    }


def test_fingerprint_mode_anchor_count_does_not_grow_with_source_size(
    tmp_path: Path,
) -> None:
    """Widening the anchors must not turn a cache lookup into a full read."""
    path = tmp_path / "huge.mp4"
    path.write_bytes(b"")
    os.truncate(path, 512 * 1024 * 1024)
    progress = ProgressReporter()

    local_fingerprint(path, "fingerprint", progress)

    events = [event for event in progress.events if event.mechanism == "source_fingerprint"]
    assert len(events) == FINGERPRINT_INTERIOR_ANCHORS + 2
    offsets = [event.detail["offset"] for event in events if event.detail]
    assert offsets == sorted(offsets)
    assert offsets[-1] == 512 * 1024 * 1024 - FINGERPRINT_SAMPLE_BYTES


def _write_crafted_collision_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two files sharing size, mtime, and their first and last 64 KiB.

    They differ across every byte of the interior, so any interior anchor at all
    separates them; only a fingerprint that reads nothing but the two ends can
    call them the same source.
    """
    head = b"H" * FINGERPRINT_SAMPLE_BYTES
    tail = b"T" * FINGERPRINT_SAMPLE_BYTES
    middle_len = 1024 * 1024 - 2 * FINGERPRINT_SAMPLE_BYTES
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    left.write_bytes(head + b"A" * middle_len + tail)
    right.write_bytes(head + b"B" * middle_len + tail)
    assert left.stat().st_size == right.stat().st_size
    stamp = 1_700_000_000
    for path in (left, right):
        os.utime(path, ns=(stamp * 1_000_000_000, stamp * 1_000_000_000))
    assert left.stat().st_mtime_ns == right.stat().st_mtime_ns
    return left, right


def test_distinct_sources_sharing_size_mtime_head_and_tail_do_not_collide(
    tmp_path: Path,
) -> None:
    """RV-4: sampling only the two ends hands one source another's bundle."""
    left, right = _write_crafted_collision_pair(tmp_path)

    assert local_fingerprint(left, "fingerprint") != local_fingerprint(right, "fingerprint")


def test_content_mode_distinguishes_the_crafted_pair(tmp_path: Path) -> None:
    """Content mode reads every byte, so no sampled anchor set can hide a difference."""
    left, right = _write_crafted_collision_pair(tmp_path)

    assert local_fingerprint(left, "content") != local_fingerprint(right, "content")


def test_output_dir_must_be_under_home_or_temp(tmp_path: Path) -> None:
    assert validate_output_root(str(tmp_path)).exists()
    with pytest.raises(DistillError) as exc:
        validate_output_root("/usr/local/distill")
    assert exc.value.code == "E_BAD_OUTPUT_DIR"


def test_output_dir_sensitive_paths_follow_platform_case_rules() -> None:
    assert sensitive_path_match(Path("/tmp/.ssh/cache"), case_sensitive=True) == ".ssh"
    assert sensitive_path_match(Path("/tmp/.SSH/cache"), case_sensitive=True) is None
    assert sensitive_path_match(Path("/tmp/.SSH/cache"), case_sensitive=False) == ".ssh"


def test_home_itself_is_rejected_as_an_output_root() -> None:
    """R-15, finding 1: a run rooted at `$HOME` puts every prune target in it.

    `$HOME` passed the confinement check because it is trivially under itself,
    which made the whole home directory an output root - the blast radius that
    turned finding 1 from a bug into a critical one.
    """
    with pytest.raises(DistillError) as failure:
        validate_output_root(str(Path.home()))

    assert failure.value.code == "E_BAD_OUTPUT_DIR"


def test_the_system_temp_directory_itself_is_rejected_as_an_output_root() -> None:
    """R-15: the shared temp directory holds other programs' files."""
    with pytest.raises(DistillError) as failure:
        validate_output_root(tempfile.gettempdir())

    assert failure.value.code == "E_BAD_OUTPUT_DIR"


def test_a_subdirectory_of_home_or_temp_is_accepted(tmp_path: Path) -> None:
    """R-15 rejects the roots themselves, not the tree under them."""
    under_home = validate_output_root(str(Path.home() / "distill-output"))
    under_temp = validate_output_root(str(tmp_path / "distill-output"))

    assert under_home.is_dir()
    assert under_temp.is_dir()


def test_sensitive_paths_match_on_a_path_component_not_a_substring() -> None:
    """A bare substring match rejects directories that are not sensitive at all.

    `.sshfs-mounts` contains `.ssh` and `my.aws-notes` contains `.aws`, yet
    neither is the directory the rule exists to protect. Matching whole path
    components keeps the guard from refusing output roots a user may legitimately
    ask for.
    """
    assert sensitive_path_match(Path("/tmp/.sshfs-mounts/cache")) is None
    assert sensitive_path_match(Path("/tmp/my.aws-notes/cache")) is None
    assert sensitive_path_match(Path("/tmp/keychains-backup/cache")) is None

    assert sensitive_path_match(Path("/tmp/.ssh/cache")) == ".ssh"
    assert sensitive_path_match(Path("/tmp/.config/1password/cache")) == ".config/1password"
    assert sensitive_path_match(Path("/tmp/Library/Keychains/x"), case_sensitive=False) == (
        "library/keychains"
    )


def test_youtube_url_parsing_and_lock_key() -> None:
    assert parse_youtube_url("https://youtu.be/abc123?t=1") == "abc123"
    assert parse_youtube_url("https://www.youtube.com/watch?v=abc123&feature=x") == "abc123"
    assert parse_youtube_url("https://m.youtube.com/shorts/abc123") == "abc123"
    assert parse_youtube_url("https://www.youtube.com/live/EB-OyTv11Q0") == "EB-OyTv11Q0"
    assert parse_youtube_url("https://www.youtube.com/v/abc123") == "abc123"
    with pytest.raises(DistillError):
        parse_youtube_url("https://example.com/watch?v=abc123")
    with pytest.raises(DistillError, match="https://www.youtube.com/feed/trending"):
        parse_youtube_url("https://www.youtube.com/feed/trending")
    assert youtube_lock_key("abc123") == hashlib.sha256(b"abc123").hexdigest()


def test_normalize_youtube_url_strips_timestamp() -> None:
    assert (
        normalize_youtube_url("https://www.youtube.com/watch?v=ow1we5PzK-o&t=900s")
        == "https://www.youtube.com/watch?v=ow1we5PzK-o"
    )
    assert normalize_youtube_url("https://youtu.be/abc123?t=42") == "https://youtu.be/abc123"
    assert (
        normalize_youtube_url("https://www.youtube.com/watch?v=abc&t=10&list=PL1")
        == "https://www.youtube.com/watch?v=abc&list=PL1"
    )
    unchanged = "https://www.youtube.com/watch?v=abc&list=PL1"
    assert normalize_youtube_url(unchanged) == unchanged
    assert (
        normalize_youtube_url("https://example.com/watch?v=abc&t=10")
        == "https://example.com/watch?v=abc&t=10"
    )


def test_ytdlp_progress_parses_percent_and_bytes() -> None:
    parsed = parse_ytdlp_progress("[download]  25.0% of 10.00MiB at 1.00MiB/s ETA 00:10")

    assert parsed == {"percent": 25.0}


def test_ytdlp_progress_parses_downloaded_of_total_without_percent() -> None:
    parsed = parse_ytdlp_progress("[download] 8.00MiB of 10.00MiB at 1.00MiB/s")

    assert parsed == {
        "downloaded_bytes": parse_byte_amount("8.00", "MiB"),
        "total_bytes": parse_byte_amount("10.00", "MiB"),
        "percent": 80.0,
    }


def test_ytdlp_progress_handles_indeterminate_download_line() -> None:
    assert parse_ytdlp_progress("[download] Destination: source.mp4") == {"indeterminate": 1}
    assert parse_ytdlp_progress("[info] unrelated") is None


def test_youtube_source_info_carries_the_lease_into_the_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-36: the resolved source hands its caller the lease, still held.

    The old contract had acquisition return a bare `Path` with the lease already
    released, after which the run probed and read the media unprotected. The
    lease now travels with the media as far as the reader, so a resolved YouTube
    source carries an unreleased lease and coarse progress alongside it.
    """
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    lock = tmp_path / "abc123.lock"
    lease = AcquisitionLease.take("abc123", lock)
    assert lease is not None
    progress = ProgressReporter()

    class FakeDownloader:
        def acquire(
            self,
            url: str,
            lock_key: str,
            progress: ProgressReporter | None = None,
        ) -> AcquiredSource:
            _ = (url, lock_key)
            if progress:
                progress.update(
                    "youtube_download",
                    percent=50,
                    detail={"downloaded_bytes": 5, "total_bytes": 10},
                )
            return AcquiredSource(path=video, lease=lease)

    monkeypatch.setattr("distill.source.check_disk_floor", lambda _path: None)
    monkeypatch.setattr(
        "distill.source.youtube_metadata",
        lambda _url: YouTubeMetadata(
            video_id="abc123",
            description=(
                "Skill repo: https://github.com/example/catch-me-up\n"
                "Speaker info: https://linkedin.com/in/example"
            ),
            warnings=[],
        ),
    )
    monkeypatch.setattr("distill.source.probe_duration", lambda _path: (12.0, []))

    source = youtube_source_info(
        "https://www.youtube.com/watch?v=abc123",
        DistillOptions(),
        tmp_path,
        FakeDownloader(),
        progress,
    )

    assert source.duration_sec == 12.0
    assert source.related_links == [
        {
            "url": "https://github.com/example/catch-me-up",
            "label": "Skill repo",
            "source": "youtube_description",
            "reason": "code_or_reference_domain",
            # R-21: a **related link** is **extracted text** on both halves, so
            # it is a carrier, and its document records the policy that produced
            # it exactly as a **frame artifact**'s does.
            "redaction": "applied",
            "warnings": [],
        }
    ]
    assert [event.detail for event in progress.events if event.mechanism == "youtube_download"][
        :2
    ] == [
        {"step": "disk_precheck"},
        {"step": "resolve_id"},
    ]
    assert source.acquisition_lease is lease
    assert lease.released is False
    assert lease_is_held("abc123", lock)
    assert any(
        event.mechanism == "duration_probe" and event.status == "completed"
        for event in progress.events
    )
    lease.release()


def test_options_payload_is_stable_json_hash() -> None:
    options = DistillOptions()
    raw = json.dumps(options.cache_payload("local"), sort_keys=True).encode()
    assert options.opts_hash("local") == hashlib.sha256(raw).hexdigest()


def test_youtube_lock_wait_ends_in_the_lease_and_a_warning(tmp_path: Path) -> None:
    """The wait budget is spent on a lease that is genuinely held, then reported.

    A run that queued behind another run's download waited, and a **warning** is
    how the run that waited says so. The held lease is released from a timer, so
    the poll loop is what ends the wait rather than the budget running out - a
    budget that expired would deny the lease instead (`E_LOCKED`).
    """
    lock = tmp_path / "video.lock"
    held = AcquisitionLease.take("video", lock)
    assert held is not None
    downloader = YoutubeDownloader(
        tmp_path,
        lock_wait_sec=10.0,
        lock_poll_sec=0.005,
        lock_warn_after_sec=0.0,
    )
    releasing = threading.Timer(0.02, held.release)
    releasing.start()

    try:
        lease, warnings = downloader._acquire("video", lock)
    finally:
        releasing.cancel()

    assert lease is not None
    assert warnings
    assert warnings[0]["code"] == "long_lock_wait"
    lease.release()


def test_youtube_lock_wait_that_runs_out_denies_the_lease(tmp_path: Path) -> None:
    """A budget that expires against a live holder is a denial, not a takeover.

    Nothing about a held lease becomes stealable by being waited on: the run
    that waited is told the source is locked and leaves the holder alone.
    """
    lock = tmp_path / "video.lock"
    held = AcquisitionLease.take("video", lock)
    assert held is not None
    downloader = YoutubeDownloader(
        tmp_path,
        lock_wait_sec=0.02,
        lock_poll_sec=0.005,
    )

    lease, warnings = downloader._acquire("video", lock)

    assert lease is None
    assert warnings == []
    assert lease_is_held("video", lock)
    held.release()


def test_ytdlp_download_progress_is_read_from_stdout(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 3, R-32): progress lives on stdout, not stderr.

    yt-dlp writes `--newline` progress to stdout. Reading stderr for it reports
    no progress at all, and - with both pipes not drained concurrently - is the
    deadlock finding 3 describes. The fake writes only to stdout, so a reader
    pointed at the wrong stream sees nothing.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    progress = ProgressReporter()

    YoutubeDownloader(tmp_path).acquire(
        "https://youtu.be/abc123", "abc123", progress
    ).lease.release()

    # The two leading Nones are the lock step and yt-dlp's indeterminate
    # "Destination:" line; every percent after them came off stdout.
    assert [
        event.percent
        for event in progress.events
        if event.mechanism == "youtube_download" and event.status == "running"
    ] == [None, None, 10.0, 55.0, 100.0]


def test_ytdlp_download_progress_percent_advances(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Percent advances while a download runs, and the media file is returned."""
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    # The media lands in a staging directory acquisition discards, so the argv
    # is recorded somewhere promotion does not sweep away.
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_YTDLP_ARGV_FILE", str(argv_file))
    progress = ProgressReporter()

    acquired = YoutubeDownloader(tmp_path).acquire("https://youtu.be/abc123", "abc123", progress)

    assert acquired.path.name == "source.mp4"
    argv = literal_eval(argv_file.read_text())
    # `--newline` is what makes yt-dlp emit one progress line per update rather
    # than rewriting a single line with carriage returns.
    assert "--newline" in argv
    assert argv[-2:] == ["--", "https://youtu.be/abc123"]
    percents = [
        event.percent
        for event in progress.events
        if event.mechanism == "youtube_download"
        and event.status == "running"
        and event.percent is not None
    ]
    assert percents == sorted(percents)
    assert percents[0] < percents[-1] == 100.0
    assert progress.events[-1].status == "completed"
    acquired.lease.release()


def test_ytdlp_download_emits_a_boundary_event(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """R-29: the invocation is visible at the boundary, named by its stage."""
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        YoutubeDownloader(tmp_path).acquire("https://youtu.be/abc123", "abc123").lease.release()

    events = [json.loads(record.message) for record in caplog.records]
    # yt-dlp fetches the media; ffprobe is the validation that gates promotion.
    assert [(event["detail"]["tool"], event["detail"]["stage"]) for event in events] == [
        ("yt-dlp", "youtube"),
        ("ffprobe", "youtube"),
    ]


def test_probe_duration_emits_a_boundary_event(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """The ffprobe call site goes through the same path as every other tool."""
    fake_tool("ffprobe", 'import sys\n\nsys.stdout.write(\'{"format": {"duration": "12.5"}}\')\n')

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        assert probe_duration(tmp_path / "video.mp4") == (12.5, [])

    events = [json.loads(record.message) for record in caplog.records]
    assert [(event["detail"]["tool"], event["detail"]["stage"]) for event in events] == [
        ("ffprobe", "source")
    ]


def test_youtube_downloader_preserves_ytdlp_error(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """A failed download raises E_YTDLP carrying run_command's failure payload."""
    fake_tool("yt-dlp", FAKE_YTDLP_FAILING)
    downloader = YoutubeDownloader(tmp_path)

    with pytest.raises(DistillError) as exc:
        downloader.acquire("https://youtu.be/abc123", "abc123", ProgressReporter())

    assert exc.value.code == "E_YTDLP"
    assert exc.value.details["tool"] == "yt-dlp"
    assert exc.value.details["exit_status"] == 1
    assert "fatal error" in exc.value.details["stderr_tail"]


def test_youtube_downloader_releases_its_lock_when_ytdlp_fails(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """A failure must free the lease, even though success deliberately does not.

    The lease outlives `acquire` only when there is media to read (R-36). When
    there is not, nothing is left holding it, so every failing path releases it
    rather than leaving the source locked until the staleness window expires.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_FAILING)

    with pytest.raises(DistillError):
        YoutubeDownloader(tmp_path).acquire("https://youtu.be/abc123", "abc123")

    # The lock file stays where it is; what a release gives up is the lock the
    # kernel granted, and that is what the next run needs to find free.
    assert not lease_is_held("abc123", tmp_path / "_youtube_locks" / "abc123.lock")


def test_probe_duration_reports_the_truncation_its_probe_recorded(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """R-33: `run_json` is the caller's only reach into the invocation.

    A duration returned on its own leaves the **warning** with nowhere to go,
    and the **bundle** would be published never mentioning that ffprobe's own
    output was cut short.
    """
    fake_tool("ffprobe", fake_ffprobe_flooding_stderr(OUTPUT_CAP_BYTES))
    (tmp_path / "video.mp4").write_bytes(b"video")

    duration, warnings = probe_duration(tmp_path / "video.mp4")

    assert duration == 12.5
    assert [item["code"] for item in warnings] == [TRUNCATION_WARNING_CODE]


# A fake yt-dlp that answers `--dump-json` and floods stderr past the capture cap
# while it does - yt-dlp is genuinely verbose there under a slow extractor.
FAKE_YTDLP_METADATA_FLOODS_STDERR = f"""
import json, sys

sys.stdout.write(json.dumps({{"id": "abc123", "description": "hello"}}))
sys.stderr.write("x" * ({OUTPUT_CAP_BYTES} + 1024))
"""


def test_youtube_metadata_carries_its_invocations_truncation_warning(
    fake_tool: Callable[[str, str], Path],
) -> None:
    """R-33: metadata that parsed is not proof the invocation lost nothing."""
    fake_tool("yt-dlp", FAKE_YTDLP_METADATA_FLOODS_STDERR)

    metadata = youtube_metadata("https://youtu.be/abc123")

    assert metadata.video_id == "abc123"
    assert [item["code"] for item in metadata.warnings] == [TRUNCATION_WARNING_CODE]


def test_an_absent_ytdlp_ends_the_run_even_on_the_best_effort_path(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
) -> None:
    """ADR-0002 / R-34: best-effort metadata does not downgrade a required tool.

    `youtube_description` degrades when yt-dlp runs and cannot answer. An absent
    yt-dlp is a different question, and the capability table is what answers it:
    yt-dlp is required, so the run ends rather than continuing without a source.
    """
    with pytest.raises(DistillError) as failure:
        youtube_description("https://youtu.be/abc123")

    assert failure.value.code == "E_MISSING_TOOL"
    assert "yt-dlp" in failure.value.message
    assert failure.value.details["requirement"] == "required"
