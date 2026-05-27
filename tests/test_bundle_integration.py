from __future__ import annotations

from pathlib import Path

from saccade.bundle import (
    active_paths,
    ensure_safe_directory,
    publish_generation,
    read_manifest,
    response_from_paths,
    stage_paths,
    write_bundle_files,
)
from saccade.errors import SaccadeError
from saccade.options import SaccadeOptions
from saccade.source import SourceInfo


def source(tmp_path: Path) -> SourceInfo:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return SourceInfo(
        source_type="local",
        resolved_path=video,
        duration_sec=1.0,
        source_fingerprint="fingerprint",
        source_hash="hash",
        warnings=[],
    )


def minimal_manifest(tmp_path: Path) -> dict:
    return {
        "pipeline_version": 1,
        "saccade_version": "0.1.0",
        "source_type": "local",
        "source_hash": "hash",
        "source_resolved_path": str(tmp_path / "video.mp4"),
        "duration_sec": 1.0,
        "options": {},
        "frame_count": 0,
        "transcript_present": True,
        "warning_count": 0,
        "frames": [],
        "warnings": [],
    }


def test_generation_publish_and_active_manifest(tmp_path: Path) -> None:
    root = tmp_path / "hash"
    root.mkdir()
    staged = stage_paths(root)
    frame = staged.frames / "frame_0001.png"
    frame.write_bytes(b"png")
    frames = [
        {
            "index": 1,
            "timestamp_sec": 0.0,
            "path": str(frame),
            "relative_path": "frames/frame_0001.png",
            "ocr_text": "text",
        }
    ]
    manifest = write_bundle_files(
        staged,
        source(tmp_path),
        SaccadeOptions(),
        {"segments": [{"start": 0, "end": 1, "text": "hi"}]},
        "# Video\n",
        frames,
        [],
    )
    final = publish_generation(staged, manifest)
    assert final.generation.name == "g1"
    assert final.markdown.exists()
    assert read_manifest(root)["active_generation"] == "g1"
    assert active_paths(root).generation == final.generation


def test_response_shape(tmp_path: Path) -> None:
    root = tmp_path / "hash"
    root.mkdir()
    staged = stage_paths(root)
    staged.markdown.write_text("# Video\n")
    staged.transcript.write_text("{}")
    final = publish_generation(staged, minimal_manifest(tmp_path))
    response = response_from_paths(final, source(tmp_path), [], True, [], cached=False)
    assert response["markdown_path"].endswith("video.md")
    assert response["transcript_path"].endswith("transcript.json")
    assert response["manifest_path"].endswith("_manifest.json")
    assert response["cached"] is False
    assert response["pipeline_version"] == 7


def test_response_can_include_progress_summary(tmp_path: Path) -> None:
    root = tmp_path / "hash"
    root.mkdir()
    staged = stage_paths(root)
    staged.markdown.write_text("# Video\n")
    staged.transcript.write_text("{}")
    final = publish_generation(staged, minimal_manifest(tmp_path))

    response = response_from_paths(
        final,
        source(tmp_path),
        [],
        True,
        [],
        cached=False,
        progress={"overall_percent": 100.0, "mechanisms": {}},
    )

    assert response["progress"] == {"overall_percent": 100.0, "mechanisms": {}}


def test_response_frames_keep_ocr_and_visual_interpretation_separate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hash"
    root.mkdir()
    staged = stage_paths(root)
    staged.markdown.write_text("# Video\n")
    final = publish_generation(staged, minimal_manifest(tmp_path))
    frame = {
        "index": 1,
        "timestamp_sec": 0.0,
        "path": str(staged.frames / "frame.png"),
        "relative_path": "frames/frame.png",
        "ocr_text": "raw text",
        "visual_interpretation": {
            "visual_summary": "A chart",
            "detected_elements": ["axis"],
            "interpretation": "Trend rises.",
            "uncertainty": "Low",
            "backend": "ollama",
            "model": "qwen3-vl:8b",
            "prompt_profile": "technical",
        },
    }

    response = response_from_paths(final, source(tmp_path), [frame], False, [], cached=False)

    assert response["frames"][0]["ocr_text"] == "raw text"
    assert response["frames"][0]["visual_interpretation"]["visual_summary"] == "A chart"
    assert response["summary"] == (
        "Processed 1.0s video with 1 keyframes and visual interpretation for 1 frames"
    )


def test_manifest_schema_validation_rejects_malformed_cache_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hash"
    root.mkdir()
    (root / "_manifest.json").write_text('{"active_generation": "g1"}')

    try:
        read_manifest(root)
    except SaccadeError as exc:
        assert exc.code == "E_BAD_MANIFEST"
        assert exc.stage == "bundle"
        assert exc.details["field"] == "pipeline_version"
    else:
        raise AssertionError("expected malformed manifest to fail validation")


def test_precreated_symlink_component_under_output_tree_fails_closed(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (output_root / "hash").symlink_to(target, target_is_directory=True)

    try:
        ensure_safe_directory(output_root / "hash" / "g1", output_root)
    except SaccadeError as exc:
        assert exc.code == "E_BAD_OUTPUT_DIR"
        assert exc.stage == "bundle"
    else:
        raise AssertionError("expected symlinked output component to fail closed")


def test_precreated_tmp_symlink_fails_before_cleanup(tmp_path: Path) -> None:
    bundle_root = tmp_path / "hash"
    bundle_root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (bundle_root / ".tmp.g1").symlink_to(target, target_is_directory=True)

    try:
        stage_paths(bundle_root)
    except SaccadeError as exc:
        assert exc.code == "E_BAD_OUTPUT_DIR"
        assert exc.stage == "bundle"
    else:
        raise AssertionError("expected symlinked staging path to fail closed")
