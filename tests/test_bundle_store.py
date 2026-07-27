"""Tests for bundle identity: what makes a directory a bundle, and what does not.

The seam under test is `BundleStore` - the single owner of the question "is this
directory a **bundle**?". Finding 1 is what happens when nothing owns it: a
directory holding `g1/video.md` was treated as a bundle and deleted, whether or
not Distill ever wrote it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.bundle_store import BundleStore, ensure_safe_directory
from distill.errors import DistillError


def published_manifest(*, identity_field: str, identity: str, active: str = "g1") -> dict:
    """A schema-valid manifest recording `identity` under the given field name.

    `bundle_key` is the field this plan writes; `source_hash` is the legacy field
    a bundle written before it carries. D-017 makes both a valid **bundle
    marker**, so a legacy bundle stays recognizable and prunable.
    """
    return {
        "pipeline_version": 1,
        "distill_version": "0.1.0",
        "source_type": "local",
        identity_field: identity,
        "source_resolved_path": "/tmp/video.mp4",
        "duration_sec": 1.0,
        "options": {},
        "frame_count": 0,
        "transcript_present": False,
        "warning_count": 0,
        "frames": [],
        "warnings": [],
        "active_generation": active,
    }


def write_bundle(
    root: Path,
    bundle_key: str,
    *,
    identity_field: str = "bundle_key",
    identity: str | None = None,
    active: str = "g1",
    generation: str | None = "g1",
    render: bool = True,
) -> Path:
    """Lay out a bundle directory on disk, marker included."""
    directory = root / bundle_key
    directory.mkdir(parents=True)
    if generation is not None:
        (directory / generation).mkdir()
        if render:
            (directory / generation / "video.md").write_text("# Video\n")
    manifest = published_manifest(
        identity_field=identity_field,
        identity=bundle_key if identity is None else identity,
        active=active,
    )
    (directory / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def test_a_directory_with_a_render_but_no_marker_is_not_a_bundle(tmp_path: Path) -> None:
    """Finding 1: `g1/video.md` alone made an unrelated directory a bundle.

    A user pointing `--output-dir` at a directory of their own can have a `g1`
    subdirectory holding a `video.md` for reasons that have nothing to do with
    Distill. Bundle-hood must rest on a **bundle marker** Distill wrote, not on
    a filename anyone can produce.
    """
    root = tmp_path / "output"
    (root / "notes" / "g1").mkdir(parents=True)
    (root / "notes" / "g1" / "video.md").write_text("# my own notes\n")

    store = BundleStore.open(root)

    verdict = store.marker("notes")
    assert verdict.is_bundle is False
    assert verdict.kind == "absent"
    assert store.load_active("notes") is None


def test_a_marker_recording_another_identity_is_not_a_bundle(tmp_path: Path) -> None:
    """D-017: the marker proves identity only if it names the directory it sits in.

    A copied or renamed directory carries a manifest whose recorded **bundle
    key** no longer matches where it lives; serving or pruning it under the new
    name would treat one bundle's manifest as another's.
    """
    root = tmp_path / "output"
    write_bundle(root, "hash", identity="a-different-bundle-key")

    store = BundleStore.open(root)

    verdict = store.marker("hash")
    assert verdict.is_bundle is False
    assert verdict.kind == "foreign"
    assert store.load_active("hash") is None


def test_a_legacy_manifest_carrying_source_hash_is_a_valid_marker(tmp_path: Path) -> None:
    """Migration Strategy: legacy bundles stay recognizable, and so prunable."""
    root = tmp_path / "output"
    write_bundle(root, "hash", identity_field="source_hash")

    store = BundleStore.open(root)

    verdict = store.marker("hash")
    assert verdict.is_bundle is True
    assert verdict.kind == "published"
    assert verdict.bundle_key == "hash"
    snapshot = store.load_active("hash")
    assert snapshot is not None
    assert snapshot.generation.name == "g1"


def test_a_manifest_carrying_bundle_key_is_a_valid_marker(tmp_path: Path) -> None:
    """The field this plan writes is accepted on the same terms as the legacy one."""
    root = tmp_path / "output"
    write_bundle(root, "hash", identity_field="bundle_key")

    store = BundleStore.open(root)

    assert store.marker("hash").kind == "published"
    snapshot = store.load_active("hash")
    assert snapshot is not None
    assert snapshot.bundle_key == "hash"
    assert snapshot.markdown.read_text() == "# Video\n"
    assert snapshot.transcript == snapshot.generation / "transcript.json"
    assert snapshot.frames == snapshot.generation / "frames"


def test_a_manifest_naming_a_missing_generation_is_a_cache_miss(tmp_path: Path) -> None:
    """R-04, finding 2: a manifest is a promise, not proof the generation exists.

    Retention deleted the **active generation** and left the manifest naming it.
    Serving that as a cache hit hands back paths to nothing.
    """
    root = tmp_path / "output"
    write_bundle(root, "hash", generation=None)

    store = BundleStore.open(root)

    assert store.marker("hash").kind == "published"
    assert store.load_active("hash") is None


def test_a_generation_without_a_render_is_a_cache_miss(tmp_path: Path) -> None:
    """R-04: the **render** is what a reader is entitled to, so it must exist."""
    root = tmp_path / "output"
    write_bundle(root, "hash", render=False)

    store = BundleStore.open(root)

    assert store.load_active("hash") is None


def test_a_malformed_manifest_is_a_cache_miss_not_a_fatal_error(tmp_path: Path) -> None:
    """A manifest that fails schema validation is not a marker, and not fatal.

    The directory may not be Distill's at all, which is exactly the case R-01
    exists for: refuse to claim it, rather than failing the run over it.
    """
    root = tmp_path / "output"
    (root / "hash").mkdir(parents=True)
    (root / "hash" / "_manifest.json").write_text('{"active_generation": "g1"}')

    store = BundleStore.open(root)

    verdict = store.marker("hash")
    assert verdict.is_bundle is False
    assert verdict.kind == "invalid"
    assert store.load_active("hash") is None


def test_an_unreadable_manifest_is_a_cache_miss_not_a_fatal_error(tmp_path: Path) -> None:
    """Truncated JSON from an interrupted write must not raise out of a read."""
    root = tmp_path / "output"
    (root / "hash").mkdir(parents=True)
    (root / "hash" / "_manifest.json").write_text('{"active_generation":')

    store = BundleStore.open(root)

    assert store.marker("hash").kind == "invalid"
    assert store.load_active("hash") is None


def test_an_ownership_marker_identifies_a_directory_distill_owns(tmp_path: Path) -> None:
    """RV-9, D-025: a crashed first run leaves no manifest, only its own marker.

    Such a directory is not a **bundle** - there is nothing to serve - but it is
    Distill's, so prune may reclaim it rather than having to skip it forever.
    """
    root = tmp_path / "output"
    (root / "hash").mkdir(parents=True)
    (root / "hash" / "_owner.json").write_text(json.dumps({"bundle_key": "hash"}) + "\n")

    store = BundleStore.open(root)

    verdict = store.marker("hash")
    assert verdict.is_bundle is False
    assert verdict.kind == "owned"
    assert verdict.is_distill_owned is True
    assert store.load_active("hash") is None


def test_a_bundle_key_may_not_escape_the_output_root(tmp_path: Path) -> None:
    """A bundle key is a directory name, so it can never traverse upwards."""
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    for escape in ("..", "../elsewhere", "/absolute", "nested/key"):
        with pytest.raises(DistillError) as failure:
            store.marker(escape)
        assert failure.value.code == "E_BAD_OUTPUT_DIR"


def test_a_symlinked_write_target_is_refused_rather_than_followed(tmp_path: Path) -> None:
    """R-16/S1: the target of a write is checked, not only the directories above it.

    `ensure_safe_directory` already walked path components; what it never saw was
    a file target, because file writes did not go through it. A pre-created
    symlink at `video.md` therefore redirected the write outside the bundle.

    Both directions are refused, and by different guards: a link out of the root
    fails confinement, while a link that stays inside it can only be caught by
    the symlink walk.
    """
    root = tmp_path / "output"
    (root / "hash").mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("original\n")
    neighbour = root / "hash" / "neighbour.md"
    neighbour.write_text("neighbour\n")
    (root / "hash" / "escapes.md").symlink_to(outside)
    (root / "hash" / "stays-inside.md").symlink_to(neighbour)

    for name in ("escapes.md", "stays-inside.md"):
        with pytest.raises(DistillError) as failure:
            ensure_safe_directory(root / "hash" / name, root, create_leaf=False)
        assert failure.value.code == "E_BAD_OUTPUT_DIR"

    assert outside.read_text() == "original\n"
    assert neighbour.read_text() == "neighbour\n"


def test_a_target_outside_the_output_root_is_refused(tmp_path: Path) -> None:
    """The other half of R-16: a path that never enters the root at all."""
    root = tmp_path / "output"
    root.mkdir()

    for escape in (tmp_path / "elsewhere", root / ".." / "elsewhere", Path("/etc/distill")):
        with pytest.raises(DistillError) as failure:
            ensure_safe_directory(escape, root)
        assert failure.value.code == "E_BAD_OUTPUT_DIR"

    assert not (tmp_path / "elsewhere").exists()


def test_validating_a_file_target_creates_its_parents_but_not_the_target(
    tmp_path: Path,
) -> None:
    """`create_leaf=False` is what lets one checker serve a file target."""
    root = tmp_path / "output"
    root.mkdir()

    target = ensure_safe_directory(root / "hash" / "g1" / "video.md", root, create_leaf=False)

    assert target.parent.is_dir()
    assert not target.exists()
