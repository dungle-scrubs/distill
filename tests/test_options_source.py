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

from distill.artifacts import RedactionState
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
    youtube_fast_path_video_id,
    youtube_lock_key,
    youtube_metadata,
    youtube_source_info,
    youtube_url_names_one_video,
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


VALID_ID = "YE7VzlLtp-4"
"""Eleven characters of `[0-9A-Za-z_-]`, which is the whole shape yt-dlp matches."""


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param(f"https://www.youtube.com/watch?v={VALID_ID}", VALID_ID, id="watch"),
        pytest.param(f"https://youtu.be/{VALID_ID}", VALID_ID, id="youtu.be"),
        pytest.param(f"https://m.youtube.com/shorts/{VALID_ID}", VALID_ID, id="shorts"),
        pytest.param(f"https://www.youtube.com/live/{VALID_ID}", VALID_ID, id="live"),
        pytest.param(f"https://www.youtube.com/embed/{VALID_ID}", VALID_ID, id="embed"),
        pytest.param(f"https://www.youtube.com/v/{VALID_ID}", VALID_ID, id="v"),
        pytest.param(
            f"https://www.youtube.com/watch?v={VALID_ID}&feature=share",
            VALID_ID,
            id="an_unrelated_query_param_is_not_trailing_junk",
        ),
        pytest.param(
            "https://www.youtube.com/watch?v=Ye7vZLltP-4", "Ye7vZLltP-4", id="case_is_preserved"
        ),
        pytest.param(
            f"https://www.youtube.com/watch?v={VALID_ID}x", None, id="twelve_characters"
        ),
        pytest.param(
            "https://www.youtube.com/watch?v=YE7VzlLtp", None, id="nine_characters"
        ),
        pytest.param("https://www.youtube.com/watch?v=YE7VzlLtp-", None, id="ten_characters"),
        pytest.param(f"https://youtu.be/{VALID_ID}/more", None, id="a_segment_after_the_id"),
        pytest.param(
            f"https://www.youtube.com/watch?v={VALID_ID}&v=aaaaaaaaaaa",
            None,
            id="duplicate_v_params",
        ),
        pytest.param(
            f"https://www.youtube.com/watch?v={VALID_ID}&list=PL1", None, id="a_playlist_attached"
        ),
        pytest.param(
            f"https://www.youtube.com/watch?v={VALID_ID}&list=", None, id="an_empty_playlist_param"
        ),
        pytest.param("https://www.youtube.com/playlist?list=PL1", None, id="no_video_id_at_all"),
        pytest.param(f"https://example.com/watch?v={VALID_ID}", None, id="not_a_youtube_host"),
    ],
)
def test_the_cache_fast_path_only_reads_an_id_yt_dlp_would_report(
    url: str, expected: str | None
) -> None:
    """The URL fast path trusts an id only where yt-dlp would resolve the same one.

    yt-dlp's extractor matches exactly eleven `[0-9A-Za-z_-]` characters and
    ignores what follows, so `watch?v=YE7VzlLtp-4x` is published under
    `YE7VzlLtp-4`. Keying a lookup on the twelve-character value names a
    **bundle** nothing ever wrote: a false **cache miss**, and with yt-dlp
    uninstalled an `E_MISSING_TOOL` raised over data that is on disk.

    Declining is the whole answer - the run then resolves the id the way it
    always did. Truncating to eleven characters here would be reimplementing
    another tool's parser against its own output.
    """
    assert youtube_fast_path_video_id(url) == expected
    assert youtube_url_names_one_video(url) is (expected is not None)


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
    # R-21: a **related link** is **extracted text** on both halves, so what the
    # resolver hands back is a carrier that records the policy which produced
    # it, exactly as a **frame artifact** does - and never a document, which is
    # what a sink can no longer refuse (finding 5).
    assert source.related_links is not None
    assert [
        (link.url, link.label, link.source, link.reason, link.redaction)
        for link in source.related_links
    ] == [
        (
            "https://github.com/example/catch-me-up",
            "Skill repo",
            "youtube_description",
            "code_or_reference_domain",
            RedactionState.APPLIED,
        )
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


def test_a_nan_max_duration_is_refused_rather_than_silencing_the_cap() -> None:
    """FAILS FIRST (finding 18, R-47): NaN cleared `<= 0` and then never compared true.

    A NaN is not merely a strange cap, it is a cap that cannot fire: every
    comparison against it is false, so `duration > max_duration_sec` answers no
    for a source of any length and the limit the operator asked for silently
    stops existing. It is refused at the boundary, where the value still has a
    name to report.
    """
    duration_sec, cap = 7200.0, float("nan")
    assert not (duration_sec > cap), "a NaN cap silently answers 'not over the limit'"

    with pytest.raises(DistillError) as exc:
        DistillOptions.from_args({"max_duration_sec": float("nan")})

    assert exc.value.code == "E_BAD_OPTIONS"
    assert "max_duration_sec" in exc.value.message


def test_a_negative_max_items_is_refused_rather_than_slicing_from_the_end(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """FAILS FIRST (nit, R-03): `items[:-1]` drops the last item and calls it a limit.

    `max_items` says how many items a batch may take. Negative, it became a
    negative slice - every item but the last few, taken from the wrong end -
    and the batch reported the count it produced as though the limit had been
    honoured.

    The playlist half also pins *when*: the refusal lands before yt-dlp is
    reached, so an unusable option is not reported as a missing tool.
    """
    from distill.pipeline import process_video_directory, process_youtube_playlist

    assert ["a", "b", "c"][:-1] == ["a", "b"], "what a negative max_items meant"
    directory = tmp_path / "videos"
    directory.mkdir()

    with pytest.raises(DistillError) as batch_exc:
        process_video_directory(
            {"path": str(directory), "output_dir": str(tmp_path / "cache"), "max_items": -1}
        )

    assert batch_exc.value.code == "E_BAD_OPTIONS"
    assert "max_items" in batch_exc.value.message

    with pytest.raises(DistillError) as playlist_exc:
        process_youtube_playlist(
            {
                "url": "https://www.youtube.com/playlist?list=PLabc",
                "output_dir": str(tmp_path / "cache"),
                "max_items": -1,
            }
        )

    assert playlist_exc.value.code == "E_BAD_OPTIONS"
    assert "max_items" in playlist_exc.value.message


def test_an_infinite_numeric_option_is_refused() -> None:
    """R-47: `inf` is finite-looking to `> 0` and means nothing as a limit.

    An infinite cap is the NaN failure with the sign flipped: the comparison
    answers, and always answers "under the limit", so the run is unbounded
    while reporting a bound. An infinite window or timeout is the same claim
    about time nobody has.
    """
    for name in ("max_duration_sec", "max_static_window_sec", "local_vision_timeout_sec"):
        for bad in (float("inf"), float("-inf")):
            with pytest.raises(DistillError) as exc:
                DistillOptions.from_args({name: bad})
            assert exc.value.code == "E_BAD_OPTIONS"
            assert name in exc.value.message


def test_a_zero_or_negative_duration_option_is_refused() -> None:
    """R-47: a limit of zero seconds admits no source, and a negative one no run.

    `max_static_window_sec` is already pinned non-positive by
    `test_non_positive_max_static_window_is_rejected`; this states the same for
    the run's duration cap and the vision timeout, which had no floor at all.
    """
    for name in ("max_duration_sec", "local_vision_timeout_sec"):
        for bad in (0.0, -1.0):
            with pytest.raises(DistillError) as exc:
                DistillOptions.from_args({name: bad})
            assert exc.value.code == "E_BAD_OPTIONS"
            assert name in exc.value.message


def test_every_numeric_option_declares_a_domain_and_refuses_a_non_finite_value() -> None:
    """R-47: finiteness is checked for the options that exist, not a list of them.

    The fields are read off `DistillOptions` rather than spelled out, so a
    numeric option added later is validated by construction: it either declares
    what values it may take or this test names it.
    """
    from dataclasses import fields

    from distill.options import NUMERIC_OPTION_DOMAINS

    numeric = [field.name for field in fields(DistillOptions) if field.type in {"int", "float"}]
    assert numeric, "the options dataclass has numeric fields to validate"

    for name in numeric:
        assert name in NUMERIC_OPTION_DOMAINS, f"{name} declares no numeric domain"
        for bad in (float("nan"), float("inf")):
            with pytest.raises(DistillError) as exc:
                DistillOptions.from_args({name: bad})
            assert exc.value.code == "E_BAD_OPTIONS", name
            assert name in exc.value.message


def test_a_numeric_option_refuses_a_value_that_is_not_a_number() -> None:
    """R-47: one error shape for a bad option, whatever made it bad.

    `int("lots")` and `int(float("nan"))` both raise, and neither raises
    anything the CLI reports as an option problem: they escaped as a
    `ValueError` traceback. A boolean is refused for the reason `PrunePolicy`
    refuses one - `True` meaning 1 is a coincidence, not an instruction.
    """
    for bad in ("lots", None, True):
        with pytest.raises(DistillError) as exc:
            DistillOptions.from_args({"max_keyframes": bad})
        assert exc.value.code == "E_BAD_OPTIONS"
        assert "max_keyframes" in exc.value.message


def test_max_keyframes_must_be_a_whole_number_of_frames() -> None:
    """R-03: keyframes are counted, so a fractional count is refused, not floored.

    `int(2.7)` silently answered 2. Truncating an operator's number is the same
    class of quiet reinterpretation this milestone removes, and a value that is
    already whole - `80`, or `80.0` from JSON - still passes.
    """
    with pytest.raises(DistillError) as exc:
        DistillOptions.from_args({"max_keyframes": 2.7})
    assert exc.value.code == "E_BAD_OPTIONS"
    assert "max_keyframes" in exc.value.message

    assert DistillOptions.from_args({"max_keyframes": 80.0}).max_keyframes == 80
    assert DistillOptions.from_args({"max_keyframes": "80"}).max_keyframes == 80


def test_a_counted_option_is_validated_without_being_rewritten() -> None:
    """R-03: an integral option survives validation as the integer it arrived as.

    Validation went through `float`, so `--max-keyframes 9007199254740993` was
    admitted as 9007199254740992: a rewritten option, and a different
    **bundle key** than the one the operator's number names. That is exactly
    what `test_validating_numbers_leaves_a_valid_option_tuple_hashing_as_before`
    promises does not happen, asserted there only for a value small enough for a
    float to carry.

    A decimal string is read as the integer it spells for the same reason. A
    float cannot be: `9007199254740993.0` is already 9007199254740992.0 by the
    time it reaches here, so honouring it would be publishing a number nobody
    typed - and 9007199254740992.0 is refused with it, because at that point the
    two are the same float and admitting it admits the one above.
    """
    beyond_float = 9007199254740993

    assert DistillOptions.from_args({"max_keyframes": beyond_float}).max_keyframes == beyond_float
    assert (
        DistillOptions.from_args({"max_keyframes": str(beyond_float)}).max_keyframes
        == beyond_float
    )
    assert json.dumps(
        DistillOptions.from_args({"max_keyframes": beyond_float}).cache_payload("local")[
            "max_keyframes"
        ]
    ) == str(beyond_float)

    for ambiguous in (1e300, float(2**53), float(beyond_float)):
        with pytest.raises(DistillError) as beyond_precision:
            DistillOptions.from_args({"max_keyframes": ambiguous})
        assert beyond_precision.value.code == "E_BAD_OPTIONS", ambiguous
        assert "max_keyframes" in beyond_precision.value.message, ambiguous


def test_a_numeric_option_too_large_for_a_float_is_an_option_error() -> None:
    """R-47: one error shape for a bad option, including the one float() refuses.

    JSON has no integer ceiling and Python's `int` has none either, so
    `10**400` is a value an operator or a config file can genuinely supply. It
    reached `float()` and came back as a bare `OverflowError` traceback -
    the same escape `"lots"` used to make as a `ValueError`.
    """
    for name in ("max_duration_sec", "min_interval_sec", "max_static_window_sec"):
        with pytest.raises(DistillError) as exc:
            DistillOptions.from_args({name: 10**400})
        assert exc.value.code == "E_BAD_OPTIONS", name
        assert exc.value.stage == "options", name
        assert name in exc.value.message, name

    with pytest.raises(DistillError) as as_text:
        DistillOptions.from_args({"max_duration_sec": "1" + "0" * 400})
    assert as_text.value.code == "E_BAD_OPTIONS"


def test_a_negative_zero_spacing_hashes_as_the_zero_it_is() -> None:
    """R-47: two spellings of one number are one **bundle key**, not two.

    `min_interval_sec` is the one option that admits zero, and `-0.0` clears
    every guard `0.0` clears - it is the same quantity, and a run configured
    with it asks for exactly the same schedule. `json.dumps` writes it as
    `-0.0`, so the **options hash** was over different text and the run
    published a second **bundle** for a run already on disk.
    """
    negative = DistillOptions.from_args({"min_interval_sec": -0.0})
    positive = DistillOptions.from_args({"min_interval_sec": 0.0})

    assert json.dumps(negative.cache_payload("local")["min_interval_sec"]) == "0.0"
    assert negative.opts_hash("local") == positive.opts_hash("local")


def test_the_spacing_floor_is_the_one_numeric_option_zero_still_means_something_to() -> None:
    """R-47: zero is rejected per option's meaning, not everywhere.

    `min_interval_sec` is the minimum gap between two **keyframes**; zero means
    no gap is required, which is a spacing policy and not a broken one - the
    production floor was already `< 0` rather than `<= 0`. Every other numeric
    option counts something a run needs at least one of.
    """
    from distill.options import NUMERIC_OPTION_DOMAINS

    assert DistillOptions.from_args({"min_interval_sec": 0.0}).min_interval_sec == 0.0
    with pytest.raises(DistillError) as exc:
        DistillOptions.from_args({"min_interval_sec": -0.5})
    assert exc.value.code == "E_BAD_OPTIONS"

    zero_admitted = [name for name, domain in NUMERIC_OPTION_DOMAINS.items() if domain.admits_zero]
    assert zero_admitted == ["min_interval_sec"]


def test_validating_numbers_leaves_a_valid_option_tuple_hashing_as_before() -> None:
    """R-47: validation refuses values, it does not rewrite the ones it admits.

    The **options hash** is sha256 over JSON, where `80` and `80.0` are
    different text and therefore a different **bundle key**. A validator that
    returned a float for a counted option would silently orphan every published
    **bundle**, so the payload is compared as the JSON it is hashed as.
    """
    from distill.local_vision import (
        DEFAULT_LOCAL_VISION_BACKEND,
        DEFAULT_LOCAL_VISION_BASE_URL,
        DEFAULT_LOCAL_VISION_MODEL,
        DEFAULT_TIMEOUT_SEC,
    )
    from distill.version import PIPELINE_VERSION

    options = DistillOptions.from_args(
        {"max_keyframes": 80, "min_interval_sec": 4, "max_duration_sec": 7200}
    )
    expected = {
        "caption_frames": True,
        "local_vision_allow_remote_endpoint": False,
        "local_vision_backend": DEFAULT_LOCAL_VISION_BACKEND,
        "local_vision_base_url": DEFAULT_LOCAL_VISION_BASE_URL,
        "local_vision_model": DEFAULT_LOCAL_VISION_MODEL,
        "local_vision_timeout_sec": DEFAULT_TIMEOUT_SEC,
        "max_duration_sec": 7200.0,
        "max_keyframes": 80,
        "max_static_window_sec": 90.0,
        "min_interval_sec": 4.0,
        "ocr": True,
        "ocr_language": "eng",
        "ocr_preprocess": True,
        "pipeline_version": PIPELINE_VERSION,
        "redact_secrets": True,
        "vad_filter": True,
        "whisper_language": "en",
        "whisper_model": "small",
    }
    payload = options.cache_payload("youtube")

    # Spelled as the JSON that is hashed: `80` and `80.0` compare equal in
    # Python and hash differently here.
    assert json.dumps(payload["max_keyframes"]) == "80"
    assert json.dumps(payload["min_interval_sec"]) == "4.0"
    assert json.dumps(payload["max_duration_sec"]) == "7200.0"
    assert json.dumps(payload, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert options.opts_hash("youtube") == hashlib.sha256(
        json.dumps(expected, sort_keys=True).encode()
    ).hexdigest()
