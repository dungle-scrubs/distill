"""Command-level regressions for the RFC's failure reporting contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from runtime_fakes import configure_run
from test_local_integration import fake_transcribe, make_short_screencast

from distill import cli, pipeline
from distill.bundle_store import BundleStore
from distill.errors import DistillError


@pytest.fixture
def video_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    video = tmp_path / "video.mp4"
    make_short_screencast(video)
    configure_run(monkeypatch, transcribe=fake_transcribe)
    return {
        "path": str(video),
        "output_dir": str(tmp_path / "cache"),
        "artifact_dir": str(tmp_path / "cache" / "artifact"),
        "caption_frames": False,
        "ocr": False,
        "max_keyframes": 1,
        "job_id": "delivery",
    }


def test_delivery_failure_preserves_bundle_and_retry_uses_cache(
    video_args: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("DEBUG", logger="distill.bundle_store")
    generations = set()
    for _ in range(2):
        with pytest.raises(DistillError) as caught:
            pipeline.process_local_video(video_args)
        error = caught.value.to_dict()
        assert error["code"] == "E_ARTIFACT_WRITE"
        assert error["details"]["bundle_saved"] is True
        key = error["details"]["bundle_key"]
        generations.add(error["details"]["generation"])
        assert BundleStore.open(Path(video_args["output_dir"])).load_active(key) is not None
        job = pipeline.get_job_status(video_args)
        assert job["status"] == "failed"
        assert job["error"]["details"]["bundle_saved"] is True

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("cache retry must not run media stages")

    configure_run(monkeypatch, transcribe=unexpected, select_keyframes=unexpected)
    result = pipeline.process_local_video(
        {**video_args, "artifact_dir": str(tmp_path / "delivered")}
    )
    assert result["cached"] is True
    assert len(generations) == 1
    assert not any('"event": "run_abandoned"' in record.message for record in caplog.records)
    assert Path(result["artifact_path"]).is_file()


@pytest.mark.parametrize("surface", ["cli", "call-tool", "batch"])
def test_delivery_failure_reaches_each_command_surface(
    video_args: dict, surface: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if surface == "batch":
        result = pipeline.process_video_directory(
            {**video_args, "path": str(Path(video_args["path"]).parent)}
        )
        assert result["errors"][0]["code"] == "E_ARTIFACT_WRITE"
        assert not result["results"]
    else:
        if surface == "cli":
            args = [
                "process-local-video",
                video_args["path"],
                "--output-dir",
                video_args["output_dir"],
                "--artifact-dir",
                video_args["artifact_dir"],
                "--no-caption-frames",
                "--no-ocr",
            ]
        else:
            args = ["call-tool", "process_local_video", "--args", json.dumps(video_args)]
        with pytest.raises(SystemExit) as caught:
            cli.main(args)
        assert caught.value.code == 2
        error = json.loads(capsys.readouterr().err.splitlines()[-1])
        assert error["code"] == "E_ARTIFACT_WRITE"
        assert error["details"]["bundle_saved"] is True


@pytest.mark.parametrize("surface", ["call-tool", "batch"])
def test_traceback_opt_in_rethrows_original_failure(
    surface: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failure = RuntimeError("regression failure")

    def fail(*_args: object, **_kwargs: object) -> dict:
        raise failure

    monkeypatch.setenv("DISTILL_TRACEBACK", "1")
    if surface == "call-tool":
        monkeypatch.setattr(pipeline, "call_registered_tool", fail)
        args = ["call-tool", "cache_doctor"]
    else:
        (tmp_path / "video.mp4").write_bytes(b"video")
        monkeypatch.setattr(pipeline, "process_local_video", fail)
        args = ["process-video-directory", str(tmp_path), "--output-dir", str(tmp_path / "cache")]
    with pytest.raises(RuntimeError) as caught:
        cli.main(args)
    assert caught.value is failure


def test_scene_failure_is_in_the_bundle(
    video_args: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> list:
        raise RuntimeError("detector failed")

    monkeypatch.setitem(
        sys.modules,
        "scenedetect",
        SimpleNamespace(detect=fail, ContentDetector=object, AdaptiveDetector=object),
    )
    result = pipeline.process_local_video(
        {**video_args, "artifact_dir": str(tmp_path / "artifacts")}
    )
    assert result["frames"]
    assert any(w["code"] == "scene_detection_failed" for w in result["warnings"])
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert any(w["code"] == "scene_detection_failed" for w in manifest["warnings"])


def test_unexpected_job_failure_is_bounded(
    video_args: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("x" * 100_000)

    configure_run(monkeypatch, transcribe=fail)
    with pytest.raises(RuntimeError):
        pipeline.process_local_video(video_args)
    error = pipeline.get_job_status(video_args)["error"]
    assert len(json.dumps(error)) < 10_000
