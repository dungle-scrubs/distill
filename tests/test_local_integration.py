from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from saccade import pipeline as saccade_session
from saccade.local_vision import LocalVisionProbe, LocalVisionResult
from saccade.options import SaccadeOptions
from saccade.progress import ProgressCounter, ProgressReporter


def make_short_screencast(path: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for local screencast fixture")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=5:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def fake_transcribe(
    _video_path: Path,
    _work_dir: Path,
    _options: SaccadeOptions,
    progress: ProgressCounter,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    progress.increment()
    return (
        {
            "language": "en",
            "language_probability": 0.99,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "short screencast",
                    "words": [
                        {"word": "short", "start": 0.0, "end": 0.4},
                        {"word": "screencast", "start": 0.4, "end": 0.9},
                    ],
                }
            ],
        },
        [],
    )


def test_short_local_screencast_fixture_produces_transcript_and_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    response = saccade_session.process_local_video(
        {
            "path": str(video),
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
    markdown = Path(response["markdown_path"]).read_text()
    assert "short screencast" in markdown
    assert "![Frame 1](frames/" in markdown


def test_cache_hit_returns_under_one_second(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    args = {
        "path": str(video),
        "output_dir": str(tmp_path / "cache"),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        "max_keyframes": 3,
        "max_static_window_sec": 1,
    }
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)
    first = saccade_session.process_local_video(args)
    assert first["cached"] is False

    def fail_transcribe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache hit should not transcribe")

    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fail_transcribe)
    started = time.perf_counter()
    cached = saccade_session.process_local_video(args)
    elapsed = time.perf_counter() - started

    assert cached["cached"] is True
    assert elapsed < 1.0


def test_pipeline_reports_redaction_render_and_publish_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    reporter = ProgressReporter()
    monkeypatch.setattr(saccade_session, "ProgressReporter", lambda **_kwargs: reporter)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    response = saccade_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": True,
            "caption_frames": False,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    assert response["cached"] is False
    mechanisms = [event.mechanism for event in reporter.events]
    assert "redaction" in mechanisms
    assert "rendering" in mechanisms
    assert "bundle_publish" in mechanisms
    assert reporter.states["bundle_publish"].status == "completed"


def test_local_e2e_emits_stderr_progress_and_final_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    response = saccade_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    captured = capsys.readouterr()
    assert '"type": "saccade.progress"' in captured.err
    assert response["progress"]["overall_percent"] == 100.0


def test_caption_frames_degrades_to_ocr_only_when_ollama_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

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

    monkeypatch.setattr(saccade_session, "probe_local_vision", fake_probe)

    response = saccade_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": True,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    assert response["frames"]
    assert response["warnings"][-1] == {
        "stage": "local_vision",
        "code": "local_vision_ollama_unavailable",
        "message": "Ollama unavailable in test",
    }


def test_caption_frames_store_visual_interpretation_without_changing_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    def fake_ocr(
        frames: list[dict],
        _language: str,
        _enabled: bool,
        _progress: object,
    ) -> tuple[list[dict], list[dict[str, str]]]:
        copied = []
        for frame in frames:
            item = dict(frame)
            item["ocr_text"] = "Visible OCR text"
            copied.append(item)
        return copied, []

    def fake_probe(_config: object) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=True,
            backend="ollama",
            model="qwen3-vl:8b",
            base_url="http://127.0.0.1:11434",
            code="local_vision_available",
            message="available",
            detail={},
        )

    seen_prompts: list[str] = []

    def fake_interpret(
        _config: object,
        _image_path: Path,
        prompt: str,
        *,
        prompt_profile: str,
    ) -> tuple[LocalVisionResult, None]:
        seen_prompts.append(prompt)
        return (
            LocalVisionResult(
                visual_summary="A settings screen",
                detected_elements=["toggle", "button"],
                interpretation="The user can save a setting.",
                uncertainty="Low",
                backend="ollama",
                model="qwen3-vl:8b",
                prompt_profile=prompt_profile,
            ),
            None,
        )

    monkeypatch.setattr(saccade_session, "ocr_frames", fake_ocr)
    monkeypatch.setattr(saccade_session, "probe_local_vision", fake_probe)
    monkeypatch.setattr(saccade_session, "try_interpret_image", fake_interpret)
    reporter = ProgressReporter()
    monkeypatch.setattr(saccade_session, "ProgressReporter", lambda **_kwargs: reporter)

    response = saccade_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": True,
            "redact_secrets": False,
            "caption_frames": True,
            "max_keyframes": 1,
            "max_static_window_sec": 1,
        }
    )

    frame = response["frames"][0]
    assert frame["ocr_text"] == "Visible OCR text"
    assert frame["visual_interpretation"]["visual_summary"] == "A settings screen"
    assert frame["visual_interpretation"]["backend"] == "ollama"
    assert reporter.states["local_vision"].status == "completed"
    assert reporter.states["local_vision"].percent == 100.0
    assert "Visible OCR text" in seen_prompts[0]
    manifest = Path(response["manifest_path"]).read_text()
    assert "visual_interpretation" in manifest

    cached = saccade_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": True,
            "redact_secrets": False,
            "caption_frames": True,
            "max_keyframes": 1,
            "max_static_window_sec": 1,
        }
    )
    assert cached["cached"] is True
    assert cached["frames"][0]["visual_interpretation"]["model"] == "qwen3-vl:8b"


def test_job_status_and_cache_cleanup_for_saccade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    cache = tmp_path / "cache"
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    response = saccade_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(cache),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "job_id": "saccade-job",
        }
    )

    status = saccade_session.get_job_status({"output_dir": str(cache), "job_id": "saccade-job"})
    assert status["result"]["manifest_path"] == response["manifest_path"]
    assert response["job_id"] == "saccade-job"

    cleanup = saccade_session.cleanup_saccade_cache(
        {"output_dir": str(cache), "dry_run": True, "keep_generations": 1}
    )
    assert cleanup["root"] == str(cache)


def test_directory_batch_processes_video_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    response = saccade_session.process_video_directory(
        {
            "path": str(tmp_path),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_items": 1,
            "job_id": "batch",
        }
    )

    assert response["processed_count"] == 1
    assert response["results"][0]["job_id"] == "batch-1"


def test_partial_resume_reuses_completed_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fake_transcribe)

    def fail_after_transcript(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom after transcript")

    monkeypatch.setattr(saccade_session, "select_keyframes", fail_after_transcript)
    args = {
        "path": str(video),
        "output_dir": str(tmp_path / "cache"),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        "max_keyframes": 1,
        "max_static_window_sec": 1,
    }
    with pytest.raises(RuntimeError):
        saccade_session.process_local_video(args)

    def fail_transcribe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("partial resume should not transcribe again")

    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fail_transcribe)
    monkeypatch.undo()
    monkeypatch.setattr(saccade_session, "transcribe_with_imports", fail_transcribe)
    response = saccade_session.process_local_video(args)

    assert response["cached"] is False
    assert Path(response["transcript_path"]).exists()
