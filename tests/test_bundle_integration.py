from __future__ import annotations

import json
from pathlib import Path

from distill.bundle import (
    publish_generation,
    response_from_paths,
    write_bundle_files,
)
from distill.bundle_store import (
    BundleStore,
    active_paths,
    ensure_safe_directory,
    read_manifest,
    stage_paths,
)
from distill.errors import DistillError
from distill.options import DistillOptions
from distill.source import SourceInfo
from distill.version import PIPELINE_VERSION


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


def youtube_source(tmp_path: Path) -> SourceInfo:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return SourceInfo(
        source_type="youtube",
        resolved_path=video,
        duration_sec=1.0,
        source_fingerprint="fingerprint",
        source_hash="hash",
        warnings=[],
        related_links=[
            {
                "url": "https://github.com/example/catch-me-up",
                "label": "Skill repo",
                "source": "youtube_description",
                "reason": "code_or_reference_domain",
            }
        ],
    )


def minimal_manifest(tmp_path: Path) -> dict:
    return {
        "pipeline_version": 1,
        "distill_version": "0.1.0",
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
        DistillOptions(),
        {"segments": [{"start": 0, "end": 1, "text": "hi"}]},
        "# Video\n",
        frames,
        [],
    )
    final = publish_generation(staged, manifest)
    assert final.generation.name == "g1"
    assert final.markdown.exists()
    manifest_data = read_manifest(root)
    active = active_paths(root)
    assert manifest_data is not None
    assert active is not None
    assert manifest_data["active_generation"] == "g1"
    assert active.generation == final.generation


def test_publish_rewrites_staged_paths_in_partial_files(tmp_path: Path) -> None:
    root = tmp_path / "hash"
    root.mkdir()
    staged = stage_paths(root)
    frame = staged.frames / "frame_0001.png"
    frame.write_bytes(b"png")
    partial = {
        "frames": [
            {
                "index": 1,
                "timestamp_sec": 0.0,
                "path": str(frame),
                "relative_path": "frames/frame_0001.png",
            }
        ],
        "warnings": [{"detail": {"path": str(staged.generation / "log.txt")}}],
    }
    (staged.generation / "_ocr.json").write_text(json.dumps(partial, indent=2) + "\n")

    final = publish_generation(staged, minimal_manifest(tmp_path))
    rewritten = json.loads((final.generation / "_ocr.json").read_text())

    assert rewritten["frames"][0]["path"] == str(final.frames / "frame_0001.png")
    assert rewritten["warnings"][0]["detail"]["path"] == str(final.generation / "log.txt")
    assert ".tmp.g1" not in (final.generation / "_ocr.json").read_text()


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
    assert response["pipeline_version"] == PIPELINE_VERSION


def test_bundle_manifest_and_response_include_related_links(tmp_path: Path) -> None:
    root = tmp_path / "hash"
    root.mkdir()
    staged = stage_paths(root)
    source_info = youtube_source(tmp_path)

    manifest = write_bundle_files(
        staged,
        source_info,
        DistillOptions(),
        {"segments": [{"start": 0, "end": 1, "text": "hi"}]},
        "# Video\n",
        [],
        [],
    )
    final = publish_generation(staged, manifest)
    response = response_from_paths(final, source_info, [], True, [], cached=False)

    manifest_data = read_manifest(root)
    assert manifest_data is not None
    assert manifest_data["related_links"] == source_info.related_links
    assert response["related_links"] == source_info.related_links


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
            "backend": "rapid-mlx",
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


def test_a_cache_lookup_never_fails_the_run_over_a_manifest_it_cannot_use(
    tmp_path: Path,
) -> None:
    """R-04 (was `test_manifest_schema_validation_rejects_malformed_cache_manifest`).

    Classified a *defect*: the old test pinned a fatal `E_BAD_MANIFEST` keyed on
    the `pipeline_version` field for a manifest naming a generation that is not
    on disk. R-04 makes that a cache miss - the lookup answers "no bundle here",
    and the run produces one - rather than an error that ends the run.

    Both shapes are misses: the original malformed fixture, and the one finding 2
    actually produced - a well-formed manifest still naming the generation
    retention deleted out from under it.
    """
    output_root = tmp_path / "output"
    (output_root / "malformed").mkdir(parents=True)
    (output_root / "malformed" / "_manifest.json").write_text('{"active_generation": "g1"}')

    (output_root / "hash").mkdir()
    manifest = {**minimal_manifest(tmp_path), "active_generation": "g1"}
    (output_root / "hash" / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    store = BundleStore.open(output_root)

    assert store.load_active("malformed") is None
    assert store.load_active("hash") is None


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
    except DistillError as exc:
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
    except DistillError as exc:
        assert exc.code == "E_BAD_OUTPUT_DIR"
        assert exc.stage == "bundle"
    else:
        raise AssertionError("expected symlinked staging path to fail closed")
