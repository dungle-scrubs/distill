from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from distill import pipeline as distill_session
from distill.artifacts import FrameArtifact, Interpretation
from distill.local_vision import LocalVisionProbe
from distill.options import DistillOptions
from distill.progress import ProgressCounter, ProgressReporter
from distill.run_command import OUTPUT_CAP_BYTES, TRUNCATION_WARNING_CODE
from distill.source import probe_duration


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


# Durations seen by fake_transcribe, newest last. The fake stands in for the
# real transcriber in every local integration test, so without recording this it
# would silently absorb the pipeline dropping duration_sec - the argument that
# turns on ffmpeg's "-progress pipe:2" audio-extraction reporting.
RECORDED_DURATIONS: list[float] = []


def fake_transcribe(
    _video_path: Path,
    _work_dir: Path,
    _options: DistillOptions,
    progress: ProgressCounter,
    duration_sec: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    RECORDED_DURATIONS.append(duration_sec)
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
                        {"word": " screencast", "start": 0.4, "end": 0.9},
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
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)
    RECORDED_DURATIONS.clear()

    response = distill_session.process_local_video(
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
    # The transcriber must receive the probed media duration, not None and not 0:
    # extract_audio only asks ffmpeg for "-progress pipe:2" when it has one.
    probed_duration, _ = probe_duration(video)
    assert probed_duration > 0
    assert len(RECORDED_DURATIONS) == 1
    assert RECORDED_DURATIONS[0] == probed_duration


def test_a_fresh_run_writes_and_reports_both_render_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST: a generation carried and reported only `video.md`."""
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 1,
            "max_static_window_sec": 1,
        }
    )

    linked = Path(response["markdown_path"])
    self_contained = Path(response["self_contained_markdown_path"])
    assert linked.name == "video.md"
    assert self_contained.name == "video.self-contained.md"
    assert self_contained.parent == linked.parent
    assert linked.is_file()
    assert self_contained.is_file()


def test_a_credential_named_local_source_leaves_no_name_or_absolute_path_in_the_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate 4->5: only redacted provenance names the archived local source."""
    secret_name = f"ghp_{'a' * 36}"
    video = tmp_path / f"{secret_name}.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "caption_frames": False,
            "max_keyframes": 1,
            "max_static_window_sec": 1,
        }
    )

    archived = Path(response["self_contained_markdown_path"]).read_text()
    assert secret_name not in archived
    assert str(video) not in archived
    assert str(tmp_path) not in archived
    assert "[REDACTED].mp4" in archived


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
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)
    first = distill_session.process_local_video(args)
    assert first["cached"] is False

    def fail_transcribe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache hit should not transcribe")

    monkeypatch.setattr(distill_session, "transcribe_with_imports", fail_transcribe)
    started = time.perf_counter()
    cached = distill_session.process_local_video(args)
    elapsed = time.perf_counter() - started

    assert cached["cached"] is True
    assert elapsed < 1.0


def test_fresh_and_cache_hit_responses_report_the_same_frame_shape(
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
        "max_keyframes": 1,
        "max_static_window_sec": 1,
    }
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    fresh = distill_session.process_local_video(args)
    cached = distill_session.process_local_video(args)

    fresh_frame = fresh["frames"][0]
    cached_frame = cached["frames"][0]
    assert cached["cached"] is True
    assert cached["self_contained_markdown_path"] == fresh["self_contained_markdown_path"]
    assert Path(cached["self_contained_markdown_path"]).is_file()
    assert cached_frame.keys() == fresh_frame.keys()
    assert Path(cached_frame["path"]).is_absolute()
    assert cached_frame["path"] == fresh_frame["path"]

    manifest = json.loads(Path(cached["manifest_path"]).read_text())
    assert "path" not in manifest["frames"][0]


def test_pipeline_reports_render_and_publish_progress_and_no_redaction_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The mechanisms a run reports are the stages it has (D-019).

    Redaction was one of them, and it is not one any more: R-19 moved it into
    carrier construction, so it belongs to the stage whose text it redacts
    rather than being a step of its own. A mechanism nothing reports is a share
    of the progress bar that can never fill, which is why its weight went too.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    reporter = ProgressReporter()
    monkeypatch.setattr(distill_session, "ProgressReporter", lambda **_kwargs: reporter)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    response = distill_session.process_local_video(
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
    assert "redaction" not in mechanisms
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
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    response = distill_session.process_local_video(
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
    assert '"type": "distill.progress"' in captured.err
    assert response["progress"]["overall_percent"] == 100.0


def test_caption_frames_degrades_to_ocr_only_when_rapid_mlx_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    def fake_probe(_config: object) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=False,
            backend="rapid-mlx",
            model="qwen3-vl:8b",
            base_url="http://127.0.0.1:8000/v1",
            code="local_vision_rapid_mlx_unavailable",
            message="Rapid-MLX unavailable in test",
            detail={},
        )

    monkeypatch.setattr(distill_session, "probe_local_vision", fake_probe)

    response = distill_session.process_local_video(
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
    # Re-pinned by field rather than by whole-record equality: R-41 added the
    # occurrence count to every warning record, and the **degradation** this
    # test is about (R-34 - the vision target is absent, the run continues with
    # OCR-only output) is what the fields say.
    degradation = response["warnings"][-1]
    assert degradation["stage"] == "local_vision"
    assert degradation["code"] == "local_vision_rapid_mlx_unavailable"
    assert degradation["message"] == "Rapid-MLX unavailable in test"
    assert degradation["occurrences"] == 1


def test_caption_frames_store_visual_interpretation_without_changing_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    def fake_ocr(
        frames: list[FrameArtifact],
        _language: str,
        _enabled: bool,
        _progress: object,
        _preprocess: bool = True,
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        read = [frame.with_extracted_text("Visible OCR text")[0] for frame in frames]
        return read, []

    def fake_probe(_config: object) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=True,
            backend="rapid-mlx",
            model="qwen3-vl:8b",
            base_url="http://127.0.0.1:8000/v1",
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
    ) -> tuple[Interpretation, None]:
        seen_prompts.append(prompt)
        return (
            Interpretation(
                visual_summary="A settings screen",
                detected_elements=("toggle", "button"),
                interpretation="The user can save a setting.",
                uncertainty="Low",
                backend="rapid-mlx",
                model="qwen3-vl:8b",
                prompt_profile=prompt_profile,
            ),
            None,
        )

    monkeypatch.setattr(distill_session, "ocr_frames", fake_ocr)
    monkeypatch.setattr(distill_session, "probe_local_vision", fake_probe)
    monkeypatch.setattr(distill_session, "try_interpret_image_after_probe", fake_interpret)
    reporter = ProgressReporter()
    monkeypatch.setattr(distill_session, "ProgressReporter", lambda **_kwargs: reporter)

    response = distill_session.process_local_video(
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
    assert frame["visual_interpretation"]["backend"] == "rapid-mlx"
    assert reporter.states["local_vision"].status == "completed"
    assert reporter.states["local_vision"].percent == 100.0
    assert "Visible OCR text" in seen_prompts[0]
    manifest = Path(response["manifest_path"]).read_text()
    assert "visual_interpretation" in manifest

    cached = distill_session.process_local_video(
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


def test_job_status_and_cache_cleanup_for_distill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    cache = tmp_path / "cache"
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(cache),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "job_id": "distill-job",
        }
    )

    status = distill_session.get_job_status({"output_dir": str(cache), "job_id": "distill-job"})
    assert status["result"]["manifest_path"] == response["manifest_path"]
    assert response["job_id"] == "distill-job"

    cleanup = distill_session.cleanup_distill_cache(
        {"output_dir": str(cache), "dry_run": True, "keep_generations": 1}
    )
    assert cleanup["root"] == str(cache)


def test_directory_batch_processes_video_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    response = distill_session.process_video_directory(
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
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    def fail_after_transcript(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom after transcript")

    monkeypatch.setattr(distill_session, "select_keyframes", fail_after_transcript)
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
        distill_session.process_local_video(args)

    def fail_transcribe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("partial resume should not transcribe again")

    monkeypatch.setattr(distill_session, "transcribe_with_imports", fail_transcribe)
    monkeypatch.undo()
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fail_transcribe)
    response = distill_session.process_local_video(args)

    assert response["cached"] is False
    assert Path(response["transcript_path"]).exists()


# A fake tesseract that answers and floods stderr past run_command's capture cap
# while it does. Real tesseract writes leptonica diagnostics there, and a frame
# that produces megabytes of them is exactly what the cap exists for: the
# reading still arrives, and what was dropped has to be said out loud (R-33).
FAKE_TESSERACT_FLOODS_STDERR = f"""
import sys

sys.stdout.write("SLIDE TEXT\\n")
sys.stderr.write("x" * ({OUTPUT_CAP_BYTES} + 1024))
"""


def test_a_truncated_ocr_invocation_records_its_warning_in_the_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-33: truncation is a **warning**, not a silent loss - all the way out.

    The warning is created inside `run_command`, several hand-offs from the
    manifest: `ocr_frame` -> `ocr_frames` -> the OCR stage -> the run's
    warnings -> the published **bundle**. Asserting it at the bundle is what
    proves the migrated OCR call site carries `CommandResult.warnings` rather
    than reading `stdout` and dropping the rest.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)
    RECORDED_DURATIONS.clear()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    tesseract = fake_bin / "tesseract"
    tesseract.write_text(f"#!{sys.executable}\n{FAKE_TESSERACT_FLOODS_STDERR}")
    tesseract.chmod(0o755)
    # Prepended rather than replacing PATH: the run still needs the machine's
    # real ffmpeg to produce the keyframe this OCRs.
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": True,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 1,
            "max_static_window_sec": 1,
        }
    )

    truncations = [
        item for item in response["warnings"] if item["code"] == TRUNCATION_WARNING_CODE
    ]
    assert truncations, response["warnings"]
    assert truncations[0]["stage"] == "ocr"
    assert "stderr" in truncations[0]["message"]
    manifest = json.loads(Path(response["manifest_path"]).read_text())
    assert truncations[0] in manifest["warnings"]
    # The reading itself still arrived: truncation reduced the record of the
    # invocation, not its result.
    assert response["frames"][0]["ocr_text"] == "SLIDE TEXT"


# --- Run scratch never reaches the published bundle (finding 3-opus, R-13) ---


def transcribe_leaving_a_decode(
    _video_path: Path,
    work_dir: Path,
    _options: DistillOptions,
    progress: ProgressCounter,
    duration_sec: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """A transcriber that leaves behind what the real one leaves behind.

    `transcribe_video` decodes the source's audio to `work_dir/audio.wav` and
    keeps it, so an interrupted run does not decode it twice. That file is the
    largest thing a run writes that is not bundle content.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "audio.wav").write_bytes(b"RIFF" + b"\0" * 4096)
    return fake_transcribe(_video_path, work_dir, _options, progress, duration_sec)


def test_a_published_generation_carries_no_audio_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 3-opus, R-13): the decode is published with the bundle.

    The run handed the transcriber the **staging directory** itself as its work
    directory, and the strip that keeps scratch out of a **generation**
    recognizes stage results by name (`_*.json`). A 16 kHz decode of the whole
    source is neither, so it was renamed into `g1` and published - every bundle
    carrying a second copy of its source's audio for as long as it exists.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", transcribe_leaving_a_decode)

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 1,
            "max_static_window_sec": 1,
        }
    )

    generation = Path(response["markdown_path"]).parent
    assert generation.name == "g1"
    assert sorted(path.name for path in generation.rglob("audio.wav")) == []


def test_the_decode_survives_in_scratch_for_a_run_that_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The decode is kept, not unlinked: **resume** is what it exists for.

    Deleting `audio.wav` at publish would answer finding 3-opus by throwing
    away the reason the file is written where the run can find it again. It
    belongs to the **staging directory**, which survives an interrupted run and
    stops existing when the generation is published.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    root = tmp_path / "cache"
    args = {
        "path": str(video),
        "output_dir": str(root),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        "max_keyframes": 1,
        "max_static_window_sec": 1,
    }

    def fail_after_transcribing(
        video_path: Path,
        work_dir: Path,
        options: DistillOptions,
        progress: ProgressCounter,
        duration_sec: float,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        transcribe_leaving_a_decode(video_path, work_dir, options, progress, duration_sec)
        raise RuntimeError("interrupted after the decode")

    monkeypatch.setattr(distill_session, "transcribe_with_imports", fail_after_transcribing)
    with pytest.raises(RuntimeError):
        distill_session.process_local_video(args)

    staged = sorted((root).glob("*/.tmp.g1"))
    assert len(staged) == 1
    assert sorted(path.name for path in staged[0].rglob("audio.wav")) == ["audio.wav"]


def test_a_resume_rebuilds_the_frames_it_recorded_and_publishes_from_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A **resume** puts the recorded frames back into carriers and finishes the run.

    R-19/M4.4: the stages hand each other **frame artifacts**, but a **stage
    result** is JSON on disk, so a run that picks one up has to reconstitute
    them before the next stage can add to it. Nothing downstream would tolerate
    a mapping - the vision pass, the **render** and the **manifest** all read
    the carrier - so the rebuild is what makes a resumed run and a fresh one
    produce the same **generation**.

    The first run dies after OCR, which is what leaves an `_ocr.json` holding
    the frames. The second run must not re-select keyframes and must publish a
    **render** naming the image the first run extracted.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    def read_slide(
        frames: list[FrameArtifact],
        _language: str,
        _enabled: bool,
        _progress: object = None,
        _preprocess: bool = True,
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        return [frame.with_extracted_text("SLIDE TEXT")[0] for frame in frames], []

    monkeypatch.setattr(distill_session, "ocr_frames", read_slide)
    monkeypatch.setattr(
        distill_session,
        "render_markdown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom after ocr")),
    )
    args = {
        "path": str(video),
        "output_dir": str(tmp_path / "cache"),
        "ocr": True,
        "redact_secrets": True,
        "caption_frames": False,
        "max_keyframes": 1,
        "max_static_window_sec": 1,
    }
    with pytest.raises(RuntimeError):
        distill_session.process_local_video(args)

    def fail_select(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a resume must not select keyframes again")

    monkeypatch.undo()
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)
    monkeypatch.setattr(distill_session, "select_keyframes", fail_select)
    response = distill_session.process_local_video(args)

    assert response["cached"] is False
    frame = response["frames"][0]
    assert frame["ocr_text"] == "SLIDE TEXT"
    assert frame["relative_path"] == "frames/frame_0001.png"
    markdown = Path(response["markdown_path"]).read_text()
    assert "![Frame 1](frames/frame_0001.png)" in markdown
    assert "SLIDE TEXT" in markdown
