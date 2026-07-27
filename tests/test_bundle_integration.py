"""What a run publishes, and what its caller is told about it.

The seam is the pair D-041 leaves after `bundle.py` is deleted: `BundleRun`,
which turns a **staging directory** into the **active generation**, and
`response.py`, which produces the **manifest** content and the response payload
from the same description of the run. These tests hold the behavior the old
`bundle.py` tests pinned - the publish, the response keys, the frame shape - now
stated against the surfaces that own them.
"""

from __future__ import annotations

import json
from pathlib import Path

from distill.bundle_store import (
    BundleRun,
    BundleStore,
    ensure_safe_directory,
    stage_paths,
)
from distill.errors import DistillError
from distill.options import DistillOptions
from distill.response import manifest_document, run_response
from distill.source import SourceInfo
from distill.version import PIPELINE_VERSION

BUNDLE_KEY = "hash"


def source(tmp_path: Path) -> SourceInfo:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return SourceInfo(
        source_type="local",
        resolved_path=video,
        duration_sec=1.0,
        source_fingerprint="fingerprint",
        source_hash=BUNDLE_KEY,
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
        source_hash=BUNDLE_KEY,
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


def begin(root: Path) -> tuple[BundleStore, BundleRun]:
    """Open a run over a fresh bundle key, the way a pipeline run does."""
    root.mkdir(parents=True, exist_ok=True)
    store = BundleStore.open(root)
    run = store.begin(BUNDLE_KEY)
    assert isinstance(run, BundleRun)
    return store, run


def minimal_manifest(tmp_path: Path) -> dict:
    return {
        "pipeline_version": 1,
        "distill_version": "0.1.0",
        "source_type": "local",
        "source_hash": BUNDLE_KEY,
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
    store, run = begin(tmp_path / "output")
    frame = run.frames_dir / "frame_0001.png"
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
    run.write_render("# Video\n")
    run.write_transcript({"segments": [{"start": 0, "end": 1, "text": "hi"}]})

    snapshot = run.commit(
        manifest_document(
            source(tmp_path),
            DistillOptions(),
            transcript_present=True,
            frames=frames,
            warnings=[],
        )
    )

    assert snapshot.generation.name == "g1"
    assert snapshot.markdown.exists()
    active = store.load_active(BUNDLE_KEY)
    assert active is not None
    assert active.manifest["active_generation"] == "g1"
    assert active.generation == snapshot.generation


# `test_publish_rewrites_staged_paths_in_partial_files` stood here. It required
# `_ocr.json` to survive publication with its staged paths rewritten - a
# **stage result** served as bundle content, which is finding 4's disk half
# exactly. classification.md records it as a *defect* against R-13 with action
# delete: R-13 forbids a stage result from existing in a **generation** at all,
# so there is no path left to rewrite. `tests/test_bundle_publish.py` asserts
# the invariant that replaces it.


def test_response_shape(tmp_path: Path) -> None:
    _, run = begin(tmp_path / "output")
    run.write_render("# Video\n")
    run.write_transcript({})
    snapshot = run.commit(minimal_manifest(tmp_path))

    response = run_response(snapshot, source(tmp_path), [], True, [], cached=False)

    assert response["markdown_path"].endswith("video.md")
    assert response["transcript_path"].endswith("transcript.json")
    assert response["manifest_path"].endswith("_manifest.json")
    assert response["cached"] is False
    assert response["pipeline_version"] == PIPELINE_VERSION


def test_bundle_manifest_and_response_include_related_links(tmp_path: Path) -> None:
    store, run = begin(tmp_path / "output")
    source_info = youtube_source(tmp_path)
    run.write_render("# Video\n")
    run.write_transcript({"segments": [{"start": 0, "end": 1, "text": "hi"}]})

    snapshot = run.commit(
        manifest_document(
            source_info,
            DistillOptions(),
            transcript_present=True,
            frames=[],
            warnings=[],
        )
    )
    response = run_response(snapshot, source_info, [], True, [], cached=False)

    active = store.load_active(BUNDLE_KEY)
    assert active is not None
    assert active.manifest["related_links"] == source_info.related_links
    assert response["related_links"] == source_info.related_links


def test_response_can_include_progress_summary(tmp_path: Path) -> None:
    _, run = begin(tmp_path / "output")
    run.write_render("# Video\n")
    run.write_transcript({})
    snapshot = run.commit(minimal_manifest(tmp_path))

    response = run_response(
        snapshot,
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
    _, run = begin(tmp_path / "output")
    run.write_render("# Video\n")
    snapshot = run.commit(minimal_manifest(tmp_path))
    frame = {
        "index": 1,
        "timestamp_sec": 0.0,
        "path": str(snapshot.frames / "frame.png"),
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

    response = run_response(snapshot, source(tmp_path), [frame], False, [], cached=False)

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
