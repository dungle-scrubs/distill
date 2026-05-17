from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from saccade.errors import SaccadeError
from saccade.options import SaccadeOptions
from saccade.progress import ProgressReporter
from saccade.source import (
    CONTENT_HASH_LIMIT_BYTES,
    YoutubeDownloader,
    local_fingerprint,
    parse_byte_amount,
    parse_youtube_url,
    parse_ytdlp_progress,
    sensitive_path_match,
    source_hash,
    validate_output_root,
    youtube_lock_key,
    youtube_source_info,
)


def test_local_opts_hash_includes_cache_mode_but_youtube_excludes_it() -> None:
    fingerprint = SaccadeOptions(cache_mode="fingerprint")
    content = SaccadeOptions(cache_mode="content")
    assert fingerprint.opts_hash("local") != content.opts_hash("local")
    assert fingerprint.opts_hash("youtube") == content.opts_hash("youtube")


def test_local_vision_settings_affect_cache_key() -> None:
    no_caption = SaccadeOptions(caption_frames=False)
    caption = SaccadeOptions(caption_frames=True)
    larger_model = SaccadeOptions(caption_frames=True, local_vision_model="qwen3-vl:32b")

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
    with pytest.raises(SaccadeError) as exc:
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
    path.write_bytes(b"a" * (64 * 1024 + 1))
    progress = ProgressReporter()

    local_fingerprint(path, "fingerprint", progress)

    events = [event for event in progress.events if event.mechanism == "source_fingerprint"]
    assert [event.percent for event in events] == [50.0, 100.0]
    assert events[0].detail == {
        "cache_mode": "fingerprint",
        "sample": "first",
        "samples_done": 1,
        "samples_total": 2,
    }


def test_output_dir_must_be_under_home_or_temp(tmp_path: Path) -> None:
    assert validate_output_root(str(tmp_path)).exists()
    with pytest.raises(SaccadeError) as exc:
        validate_output_root("/usr/local/saccade")
    assert exc.value.code == "E_BAD_OUTPUT_DIR"


def test_output_dir_sensitive_paths_follow_platform_case_rules() -> None:
    assert sensitive_path_match(Path("/tmp/.ssh/cache"), case_sensitive=True) == ".ssh"
    assert sensitive_path_match(Path("/tmp/.SSH/cache"), case_sensitive=True) is None
    assert sensitive_path_match(Path("/tmp/.SSH/cache"), case_sensitive=False) == ".ssh"


def test_youtube_url_parsing_and_lock_key() -> None:
    assert parse_youtube_url("https://youtu.be/abc123?t=1") == "abc123"
    assert parse_youtube_url("https://www.youtube.com/watch?v=abc123&feature=x") == "abc123"
    assert parse_youtube_url("https://m.youtube.com/shorts/abc123") == "abc123"
    assert parse_youtube_url("https://www.youtube.com/live/EB-OyTv11Q0") == "EB-OyTv11Q0"
    assert parse_youtube_url("https://www.youtube.com/v/abc123") == "abc123"
    with pytest.raises(SaccadeError):
        parse_youtube_url("https://example.com/watch?v=abc123")
    with pytest.raises(SaccadeError, match="https://www.youtube.com/feed/trending"):
        parse_youtube_url("https://www.youtube.com/feed/trending")
    assert youtube_lock_key("abc123") == hashlib.sha256(b"abc123").hexdigest()


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


def test_youtube_source_info_emits_coarse_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    progress = ProgressReporter()

    class FakeDownloader:
        last_warnings: list[dict[str, str]] = []

        def download(
            self,
            _url: str,
            _lock_key: str,
            progress: ProgressReporter | None = None,
        ) -> Path:
            if progress:
                progress.update(
                    "youtube_download",
                    percent=50,
                    detail={"downloaded_bytes": 5, "total_bytes": 10},
                )
            return video

    monkeypatch.setattr("saccade.source.check_disk_floor", lambda _path: None)
    monkeypatch.setattr("saccade.source.canonical_youtube_id", lambda _url: "abc123")
    monkeypatch.setattr("saccade.source.probe_duration", lambda _path: 12.0)

    source = youtube_source_info(
        "https://www.youtube.com/watch?v=abc123",
        SaccadeOptions(),
        tmp_path,
        FakeDownloader(),
        progress,
    )

    assert source.duration_sec == 12.0
    assert [event.detail for event in progress.events if event.mechanism == "youtube_download"][
        :2
    ] == [
        {"step": "disk_precheck"},
        {"step": "resolve_id"},
    ]
    assert any(
        event.mechanism == "duration_probe" and event.status == "completed"
        for event in progress.events
    )


def test_options_payload_is_stable_json_hash() -> None:
    options = SaccadeOptions()
    raw = json.dumps(options.cache_payload("local"), sort_keys=True).encode()
    assert options.opts_hash("local") == hashlib.sha256(raw).hexdigest()


def test_youtube_stale_lock_takeover_is_covered(tmp_path: Path) -> None:
    lock = tmp_path / "video.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": 999999,
                "created_wall": "2026-01-01T00:00:00Z",
                "created_monotonic": 1.0,
                "last_heartbeat_monotonic": 1.0,
            }
        )
    )
    downloader = YoutubeDownloader(tmp_path, stale_sec=0.001)

    acquired, warnings = downloader._acquire(lock)

    assert acquired is True
    assert warnings == []
    payload = json.loads(lock.read_text())
    assert payload["pid"] != 999999


def test_youtube_long_lock_wait_emits_warning(tmp_path: Path) -> None:
    now = time.monotonic()
    lock = tmp_path / "video.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": 999999,
                "created_wall": "2026-01-01T00:00:00Z",
                "created_monotonic": now,
                "last_heartbeat_monotonic": now,
            }
        )
    )
    downloader = YoutubeDownloader(
        tmp_path,
        stale_sec=0.01,
        lock_wait_sec=0.05,
        lock_poll_sec=0.005,
        lock_warn_after_sec=0.0,
    )

    acquired, warnings = downloader._acquire(lock)

    assert acquired is True
    assert warnings
    assert warnings[0]["code"] == "long_lock_wait"


def test_youtube_downloader_parses_progress_and_preserves_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    class FakePopen:
        returncode = 0

        def __init__(self, command: list[str], **_kwargs: object) -> None:
            commands.append(command)
            out_template = Path(command[command.index("-o") + 1])
            (out_template.parent / "source.mp4").write_bytes(b"video")
            self.stderr = iter(
                [
                    "[download] Destination: source.mp4\n",
                    "[download] 50.0% of 10.00MiB at 1.00MiB/s ETA 00:05\n",
                ]
            )

        def communicate(self) -> tuple[str, str]:
            return "", ""

    monkeypatch.setattr("saccade.source.subprocess.Popen", FakePopen)
    progress = ProgressReporter()
    downloader = YoutubeDownloader(tmp_path)

    path = downloader.download("https://youtu.be/abc123", "abc123", progress)

    assert path.name == "source.mp4"
    assert "--newline" in commands[0]
    assert any(
        event.mechanism == "youtube_download" and event.percent == 50.0 for event in progress.events
    )
    assert progress.events[-1].status == "completed"


def test_youtube_downloader_preserves_ytdlp_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakePopen:
        returncode = 1

        def __init__(self, _command: list[str], **_kwargs: object) -> None:
            self.stderr = iter(["fatal error\n"])

        def communicate(self) -> tuple[str, str]:
            return "stdout text", ""

    monkeypatch.setattr("saccade.source.subprocess.Popen", FakePopen)
    downloader = YoutubeDownloader(tmp_path)

    with pytest.raises(SaccadeError) as exc:
        downloader.download("https://youtu.be/abc123", "abc123", ProgressReporter())

    assert exc.value.code == "E_YTDLP"
    assert exc.value.details["stderr"] == "fatal error"
    assert exc.value.details["stdout"] == "stdout text"
