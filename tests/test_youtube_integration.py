from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from test_local_integration import fake_transcribe, make_short_screencast

from saccade import pipeline as saccade_session
from saccade import source as saccade_source
from saccade.local_vision import LocalVisionProbe
from saccade.options import SaccadeOptions
from saccade.progress import ProgressReporter
from saccade.source import SourceInfo, source_hash, youtube_lock_key


def test_mocked_youtube_integration_passes_through_shared_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "youtube.mp4"
    make_short_screencast(video)
    lock_key = youtube_lock_key("mock-video")

    def fake_youtube_source_info(
        _url: str,
        options: SaccadeOptions,
        _output_root: Path,
        **_kwargs: object,
    ) -> SourceInfo:
        return SourceInfo(
            source_type="youtube",
            resolved_path=video,
            duration_sec=1.0,
            source_fingerprint="mock-video",
            source_hash=f"youtube-{options.opts_hash('youtube')}",
            warnings=[],
            youtube_video_id="mock-video",
            youtube_lock_key=lock_key,
        )

    monkeypatch.setattr(
        saccade_session,
        "youtube_source_info",
        fake_youtube_source_info,
    )
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)
    monkeypatch.setattr("saccade.source.canonical_youtube_id", lambda _url: "mock-video")

    response = saccade_session.process_youtube_video(
        {
            "url": "https://www.youtube.com/watch?v=mock-video",
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    assert Path(response["transcript_path"]).exists()
    assert response["frames"]
    assert response["cached"] is False


def test_cache_hit_skips_youtube_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "youtube.mp4"
    make_short_screencast(video)
    video_id = "mock-video"
    fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
    lock_key = youtube_lock_key(video_id)
    download_calls = {"count": 0}

    def fake_youtube_source_info(
        _url: str,
        options: SaccadeOptions,
        _output_root: Path,
        **_kwargs: object,
    ) -> SourceInfo:
        download_calls["count"] += 1
        return SourceInfo(
            source_type="youtube",
            resolved_path=video,
            duration_sec=1.0,
            source_fingerprint=fingerprint,
            source_hash=source_hash(fingerprint, options.opts_hash("youtube")),
            warnings=[],
            youtube_video_id=video_id,
            youtube_lock_key=lock_key,
        )

    monkeypatch.setattr(saccade_session, "youtube_source_info", fake_youtube_source_info)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)
    monkeypatch.setattr("saccade.source.canonical_youtube_id", lambda _url: video_id)

    args = {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "output_dir": str(tmp_path / "cache"),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        "max_keyframes": 3,
        "max_static_window_sec": 1,
    }

    first = saccade_session.process_youtube_video(args)
    assert first["cached"] is False
    assert download_calls["count"] == 1

    second = saccade_session.process_youtube_video(args)
    assert second["cached"] is True
    assert download_calls["count"] == 1


def test_successful_youtube_run_with_local_vision_warning_finishes_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "youtube.mp4"
    make_short_screencast(video)
    reporters: list[ProgressReporter] = []
    video_id = "mock-video"

    class FakeDownloader:
        last_warnings: list[dict[str, str]] = []

        def __init__(self, _output_root: Path) -> None:
            pass

        def download(
            self,
            _url: str,
            _lock_key: str,
            progress: ProgressReporter | None = None,
        ) -> Path:
            if progress:
                progress.update("youtube_download", percent=55.0)
                progress.complete("youtube_download", detail={"path": str(video)})
            return video

    def fake_probe(_config: object) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=False,
            backend="ollama",
            model="qwen3-vl:8b",
            base_url="http://127.0.0.1:11434",
            code="local_vision_ollama_unavailable",
            message="Ollama unavailable in test",
            detail={},
        )

    def make_reporter(**_kwargs: object) -> ProgressReporter:
        reporter = ProgressReporter()
        reporters.append(reporter)
        return reporter

    monkeypatch.setattr(saccade_session, "ProgressReporter", make_reporter)
    monkeypatch.setattr(saccade_source, "YoutubeDownloader", FakeDownloader)
    monkeypatch.setattr(saccade_source, "canonical_youtube_id", lambda _url: video_id)
    monkeypatch.setattr(saccade_source, "check_disk_floor", lambda _path: None)
    monkeypatch.setattr(saccade_source, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(saccade_session, "probe_local_vision", fake_probe)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    response = saccade_session.process_youtube_video(
        {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": True,
            "force_reprocess": True,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    assert response["cached"] is False
    assert Path(response["manifest_path"]).exists()
    assert any(
        warning["code"] == "local_vision_ollama_unavailable" for warning in response["warnings"]
    )
    assert response["progress"]["overall_percent"] == 100.0
    assert response["progress"]["mechanisms"]["youtube_download"]["status"] == "completed"
    assert response["progress"]["mechanisms"]["bundle_publish"]["status"] == "completed"
    assert response["progress"]["mechanisms"]["bundle_publish"]["detail"]["generation"] == "g1"

    manifest = json.loads(Path(response["manifest_path"]).read_text())
    assert manifest["progress"]["overall_percent"] == 100.0
    assert manifest["progress"]["mechanisms"]["youtube_download"]["status"] == "completed"
    assert manifest["progress"]["mechanisms"]["bundle_publish"]["status"] == "completed"
    assert reporters[0].states["youtube_download"].status == "completed"
    assert reporters[0].states["bundle_publish"].detail["generation"] == "g1"

    manifest["progress"] = {
        "overall_percent": 83.6,
        "mechanisms": {
            "youtube_download": {
                "status": "running",
                "percent": 0.0,
                "weight": 15.0,
                "detail": {},
            },
            "bundle_publish": {
                "status": "running",
                "percent": 0.0,
                "weight": 3.0,
                "detail": {"generation": None},
            },
        },
    }
    Path(response["manifest_path"]).write_text(json.dumps(manifest, indent=2) + "\n")

    cached = saccade_session.process_youtube_video(
        {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": True,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )
    cached_manifest = json.loads(Path(cached["manifest_path"]).read_text())

    assert cached["cached"] is True
    assert cached["progress"]["overall_percent"] == 100.0
    assert cached["progress"]["mechanisms"]["youtube_download"]["status"] == "skipped_or_cached"
    assert cached["progress"]["mechanisms"]["bundle_publish"]["status"] == "skipped_or_cached"
    assert cached_manifest["progress"]["overall_percent"] == 100.0
    assert (
        cached_manifest["progress"]["mechanisms"]["youtube_download"]["status"]
        == "skipped_or_cached"
    )


def test_concurrent_youtube_calls_for_same_video_share_lock_key() -> None:
    video_id = "same-video"
    first = youtube_lock_key(video_id)
    second = youtube_lock_key(video_id)

    assert first == second


def test_youtube_playlist_uses_playlist_output_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    url = "https://www.youtube.com/playlist?list=PLQHpFq3RA7fEJ0z3DABwTPvwre0Vu6OBH"
    child_urls = ["https://www.youtube.com/watch?v=one", "https://www.youtube.com/watch?v=two"]
    seen_child_args: list[dict[str, object]] = []

    monkeypatch.setattr(
        saccade_session,
        "youtube_playlist_urls",
        lambda _url, _max_items: child_urls,
    )

    def fake_process_youtube_video(args: dict[str, object]) -> dict[str, object]:
        seen_child_args.append(args)
        return {
            "markdown_path": str(Path(str(args["output_dir"])) / "hash" / "g1" / "video.md"),
            "transcript_path": str(Path(str(args["output_dir"])) / "hash" / "g1" / "transcript.json"),
            "manifest_path": str(Path(str(args["output_dir"])) / "hash" / "_manifest.json"),
            "frames": [],
            "duration_sec": 1.0,
            "frame_count": 0,
            "source_hash": "hash",
            "source_resolved_path": "source.mp4",
            "cached": False,
            "pipeline_version": 1,
            "saccade_version": "test",
            "job_id": str(args["job_id"]),
            "summary": "Processed 1.0s video with 0 keyframes",
            "warnings": [],
        }

    monkeypatch.setattr(saccade_session, "process_youtube_video", fake_process_youtube_video)

    response = saccade_session.process_youtube_playlist(
        {
            "url": url,
            "output_dir": str(tmp_path),
            "job_id": "playlist-job",
        }
    )

    expected_playlist_root = tmp_path / "playlists" / "PLQHpFq3RA7fEJ0z3DABwTPvwre0Vu6OBH"
    assert Path(response["playlist_output_dir"]) == expected_playlist_root
    assert Path(response["playlist_summary_path"]) == expected_playlist_root / "playlist.json"
    assert Path(response["playlist_summary_path"]).exists()
    assert [Path(str(args["output_dir"])) for args in seen_child_args] == [
        expected_playlist_root,
        expected_playlist_root,
    ]
    assert [args["job_id"] for args in seen_child_args] == ["playlist-job-1", "playlist-job-2"]


@pytest.mark.network
def test_youtube_network_smoke_is_skippable() -> None:
    if os.environ.get("SACCADE_RUN_NETWORK_TESTS") != "1":
        pytest.skip("set SACCADE_RUN_NETWORK_TESTS=1 to run YouTube network smoke")
    pytest.fail("network smoke needs an explicit allowed fixture URL before running")
