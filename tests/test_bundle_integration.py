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

from distill.artifacts import FrameArtifact, Transcript
from distill.bundle_store import (
    BundleRun,
    BundleStore,
    ensure_safe_directory,
    stage_paths,
)
from distill.errors import DistillError
from distill.links import extract_relevant_links
from distill.options import DistillOptions
from distill.response import (
    manifest_document,
    response_frames,
    response_related_links,
    run_response,
)
from distill.source import SourceInfo
from distill.version import PIPELINE_VERSION

BUNDLE_KEY = "hash"


def spoken(*segments: dict[str, object]) -> Transcript:
    """A **transcript** carrier holding exactly these segments.

    `write_transcript` takes a carrier and not a document (R-20): `serialize`
    is the last check before **extracted text** becomes durable, so a caller
    that could hand in a mapping would be a caller that had already skipped it.
    """
    return Transcript(language="en", segments=tuple(segments))


# Secret-shaped by `redact_secrets.SECRET_PATTERNS`, placed in both halves of a
# **related link** so the manifest is asked about the label and the destination.
LINK_SECRET = "sk-live-0123456789abcdefghij"


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
        # Built the way a run builds them - `extract_relevant_links` is where
        # the carrier is constructed, and R-19 puts the **redaction** policy
        # there - rather than hand-written, which would describe a source no
        # run can produce.
        related_links=extract_relevant_links(
            f"Skill repo ({LINK_SECRET}): "
            f"https://github.com/example/catch-me-up?api_key={LINK_SECRET}",
            source="youtube_description",
        ),
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
    frames = response_frames(
        [
            FrameArtifact(
                index=1,
                timestamp_sec=0.0,
                path=str(frame),
                relative_path="frames/frame_0001.png",
                extracted_text="text",
            )
        ]
    )
    run.write_render("# Video\n")
    run.write_transcript(spoken({"start": 0, "end": 1, "text": "hi"}))

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
    run.write_transcript(spoken())
    snapshot = run.commit(minimal_manifest(tmp_path))

    response = run_response(snapshot, source(tmp_path), [], True, [], cached=False)

    assert response["markdown_path"].endswith("video.md")
    assert response["transcript_path"].endswith("transcript.json")
    assert response["manifest_path"].endswith("_manifest.json")
    assert response["cached"] is False
    assert response["pipeline_version"] == PIPELINE_VERSION


def test_bundle_manifest_and_response_include_related_links(tmp_path: Path) -> None:
    """Both documents carry the links, and neither carries what the policy removed.

    R-21: a **related link**'s label and destination are **extracted text**, so
    what reaches the durable **manifest** is what came out of the carrier. The
    old assertion was that the manifest matched the source byte for byte, which
    holds whether or not anything redacted them - so it is stated here as an
    absence of the secret as well as a presence of the links.

    Carried byte-for-byte is also no longer what either document does: a source
    holds carriers now, and each sink serializes them into the four fields that
    describe a link. `redaction` and `warnings` are the carrier's bookkeeping
    and stay out of both (finding 7) - the policy is recorded once, under
    `options`, and a link's warnings travel with the source's.
    """
    store, run = begin(tmp_path / "output")
    source_info = youtube_source(tmp_path)
    run.write_render("# Video\n")
    run.write_transcript(spoken({"start": 0, "end": 1, "text": "hi"}))

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
    documents = response_related_links(source_info.related_links)
    assert active.manifest["related_links"] == documents
    assert response["related_links"] == documents
    assert [sorted(link) for link in documents] == [["label", "reason", "source", "url"]]
    published = json.dumps(active.manifest) + json.dumps(response)
    assert LINK_SECRET not in published
    assert "github.com/example/catch-me-up" in published


def test_response_can_include_progress_summary(tmp_path: Path) -> None:
    _, run = begin(tmp_path / "output")
    run.write_render("# Video\n")
    run.write_transcript(spoken())
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


# --- A new manifest records its bundle key by that name (finding 5-opus) ----


def test_a_new_manifest_records_the_bundle_key_under_the_current_name(
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 5-opus, D-008): every new manifest used the old name.

    The value hashes a **source fingerprint** together with an **options hash**,
    so it identifies a **bundle** and not a source - which is why D-008 renamed
    it. `IDENTITY_FIELDS` says only `bundle_key` is written from here on and
    accepts `source_hash` for what is already on disk; the writer never caught
    up, so the legacy name was still the only one Distill produced and the
    vocabulary's `_Avoid_` entry described current behavior.
    """
    store, run = begin(tmp_path / "output")
    run.write_render("# Video\n")

    run.commit(
        manifest_document(
            source(tmp_path),
            DistillOptions(),
            transcript_present=False,
            frames=[],
            warnings=[],
        )
    )

    active = store.load_active(BUNDLE_KEY)
    assert active is not None
    assert active.manifest["bundle_key"] == BUNDLE_KEY
    assert "source_hash" not in active.manifest


def test_a_manifest_written_before_the_rename_is_still_a_bundle(tmp_path: Path) -> None:
    """D-017: the legacy name stays readable, so old bundles stay prunable.

    Writing the current name is not the same as refusing the old one. A bundle
    published before this rename records `source_hash` and nothing will ever
    rewrite it, so recognition has to keep accepting it - the alternative is
    disk no **prune** can reclaim because nothing recognizes it as Distill's.
    """
    store, run = begin(tmp_path / "output")
    run.write_render("# Video\n")
    legacy = {**minimal_manifest(tmp_path)}
    assert legacy["source_hash"] == BUNDLE_KEY

    run.commit(legacy)

    active = store.load_active(BUNDLE_KEY)
    assert active is not None
    assert active.bundle_key == BUNDLE_KEY
    assert active.manifest["source_hash"] == BUNDLE_KEY
