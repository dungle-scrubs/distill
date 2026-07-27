"""Bundle identity, layout and lifecycle for Distill.

This module owns the question a **bundle** is: which directory holds one, what
proves it (the **bundle marker**), which **generation** is active, and where
every file under a **bundle key** lives. It is the only place that names
`_manifest.json`, `g<N>`, `.tmp.g<N>`, `frames/`, `video.md` or `transcript.json`,
and the only place that decides whether a path may be written to.

The invariant it exists to hold: *a directory is a bundle only if it carries a
bundle marker*. Before this module, any directory containing `g1/video.md` was
one, so pointing `--output-dir` at a directory of a user's own made that
directory prunable - audit finding 1. Recognition is now positive and identity
bound: the marker is a schema-valid **manifest** whose recorded bundle identity
equals the directory name, accepting either the current `bundle_key` field or
the legacy `source_hash` field so bundles written before this plan stay
recognizable and prunable (D-017).

A marker proves the directory is Distill's; it does not prove the bundle is
servable. `load_active` additionally verifies the **active generation** and its
**render** exist on disk, because a **manifest** is a promise, not evidence
(R-04): retention that deleted a generation left the manifest still naming it.

What this module does not own: what a pipeline stage computes, what a
**generation** contains, job records (`job_store`), the **source fingerprint**
or **options hash** that combine into a bundle key (`source.py`, `options.py`),
and the output-root policy, which answers a different question - whether Distill
may write under a root at all - and stays with `source.validate_output_root`.

Milestone note: `begin`, `commit`, `patch_published` and the prune surface are
declared here as the interface later milestones of this plan fill in (M3.2
locking, M3.3 publish ordering, M3.4 prune). They raise `NotImplementedError`
rather than being omitted, so callers migrate onto one shape once.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import DistillError

MANIFEST_NAME = "_manifest.json"
"""The **published marker**: present only once a generation has been published."""

OWNERSHIP_MARKER_NAME = "_owner.json"
"""The **ownership marker** `begin` writes first, so a directory is identifiable
as Distill-owned from its first moment - before any manifest exists (R-11,
D-025). A run that crashes before publishing leaves this and nothing else."""

RENDER_NAME = "video.md"
TRANSCRIPT_NAME = "transcript.json"
FRAMES_DIR_NAME = "frames"
GENERATION_PREFIX = "g"
STAGING_PREFIX = ".tmp."

IDENTITY_FIELDS = ("bundle_key", "source_hash")
"""Manifest fields carrying the recorded **bundle key**, current name first.

`source_hash` is the legacy name for the same value (D-008): it hashes the
**source fingerprint** together with the **options hash**, so it identifies a
bundle rather than a source. Both are accepted as identity; only `bundle_key` is
written from here on.
"""

MarkerKind = Literal["published", "owned", "foreign", "invalid", "absent"]


@dataclass(frozen=True)
class BundlePaths:
    root: Path
    generation: Path
    frames: Path
    manifest: Path
    transcript: Path
    markdown: Path


@dataclass(frozen=True)
class MarkerVerdict:
    """Why a directory is, or is not, a **bundle**.

    Carries a reason for every verdict rather than a bare boolean, because the
    destructive operations that consume it must be able to report *why* a
    directory was skipped - "considered nothing" and "deleted nothing" are
    different answers (R-57).
    """

    kind: MarkerKind
    reason: str
    bundle_key: str | None = None
    manifest: dict[str, Any] | None = None

    @property
    def is_bundle(self) -> bool:
        """Whether the directory is a **bundle**: something may be served from it."""
        return self.kind == "published"

    @property
    def is_distill_owned(self) -> bool:
        """Whether Distill wrote the directory, published or not.

        A directory holding only an **ownership marker** is not a bundle - there
        is nothing to serve - but it is Distill's, so prune may reclaim it
        instead of skipping it forever (RV-9).
        """
        return self.kind in ("published", "owned")


@dataclass(frozen=True)
class BundleSnapshot:
    """A readable **active generation**, proven to exist when it was loaded."""

    root: Path
    bundle_key: str
    generation: Path
    manifest: dict[str, Any]

    @property
    def markdown(self) -> Path:
        return self.generation / RENDER_NAME

    @property
    def transcript(self) -> Path:
        return self.generation / TRANSCRIPT_NAME

    @property
    def frames(self) -> Path:
        return self.generation / FRAMES_DIR_NAME


@dataclass(frozen=True)
class BundleStore:
    """Every **bundle** under one output root."""

    root: Path

    @classmethod
    def open(cls, root: Path) -> BundleStore:
        """Open the store over an output root already accepted by the root policy.

        Resolving here means every path this store derives is compared against a
        root free of symlinks, so confinement checks cannot be defeated by one.
        """
        return cls(Path(root).resolve())

    def bundle_root(self, bundle_key: str) -> Path:
        """The directory a **bundle key** names, refusing anything that escapes.

        A bundle key is a single directory name. Treating it as a path fragment
        would let `..` or an absolute value name a directory outside the output
        root - and everything below reads, and later deletes, what it is handed.
        """
        parts = Path(bundle_key).parts
        if len(parts) != 1 or bundle_key in (".", ".."):
            raise DistillError(
                "E_BAD_OUTPUT_DIR",
                "bundle",
                "bundle key must be a single directory name under the output root",
                {"bundle_key": bundle_key, "output_root": str(self.root)},
            )
        return self.root / bundle_key

    def marker(self, bundle_key: str) -> MarkerVerdict:
        """The **bundle marker** verdict for one bundle key."""
        return read_marker(self.bundle_root(bundle_key))

    def load_active(self, bundle_key: str) -> BundleSnapshot | None:
        """The servable **active generation**, or `None` for a cache miss.

        `None` covers every way a directory can fail to be a servable bundle: no
        marker, a marker recording another identity, a malformed manifest, or a
        manifest naming a generation or **render** that is not on disk (R-04).
        None of these ends the run - the run simply produces the bundle.
        """
        directory = self.bundle_root(bundle_key)
        verdict = read_marker(directory)
        if not verdict.is_bundle or verdict.manifest is None:
            return None

        generation_name = verdict.manifest.get("active_generation")
        if not isinstance(generation_name, str) or not is_generation_name(generation_name):
            return None
        generation = directory / generation_name
        if not generation.is_dir() or not (generation / RENDER_NAME).is_file():
            return None

        return BundleSnapshot(
            root=directory,
            bundle_key=bundle_key,
            generation=generation,
            manifest=verdict.manifest,
        )

    def begin(self, bundle_key: str, *, resume: bool = True) -> Any:
        """Take the run lock and open a **staging directory**. Lands in M3.2."""
        raise NotImplementedError("BundleStore.begin lands with locking in M3.2")

    def patch_published(self, snapshot: BundleSnapshot, fields: dict[str, Any]) -> Any:
        """Amend a published **manifest** in place. Lands in M3.3."""
        raise NotImplementedError("BundleStore.patch_published lands in M3.3")

    def plan_prune(self, policy: Any) -> Any:
        """Propose **generations** and bundles to remove. Lands in M3.4."""
        raise NotImplementedError("BundleStore.plan_prune lands with prune in M3.4")

    def apply_prune(self, plan: Any) -> Any:
        """Revalidate a plan under lock and delete what survives. Lands in M3.4."""
        raise NotImplementedError("BundleStore.apply_prune lands with prune in M3.4")


class BundleRun:
    """One run's exclusive hold on a **bundle**, from `begin` to `commit`.

    Declared here so callers migrate onto one shape; M3.2 gives it a lock and a
    **staging directory**, and M3.3 gives it the publish ordering.
    """

    def read_stage(self, name: str) -> Any:
        """The recorded **stage result** for `name`, if it may be reused."""
        raise NotImplementedError("BundleRun.read_stage lands in M3.3")

    def write_stage(self, name: str, result: Any) -> None:
        """Record a completed stage so an interrupted run can **resume**."""
        raise NotImplementedError("BundleRun.write_stage lands in M3.3")

    def commit(self, manifest: dict[str, Any]) -> BundleSnapshot:
        """**Publish** the staging directory as the **active generation**."""
        raise NotImplementedError("BundleRun.commit lands in M3.3")

    def abandon(self, reason: str) -> None:
        """Give up the run, leaving the previous **active generation** intact."""
        raise NotImplementedError("BundleRun.abandon lands in M3.3")


def read_marker(directory: Path) -> MarkerVerdict:
    """Decide whether `directory` carries a **bundle marker**.

    Every failure is a verdict rather than an exception: an unmarked or
    unreadable directory may not be Distill's at all, and refusing to claim it is
    the whole point (R-01). A malformed manifest is likewise not a reason to end
    a run - it is a reason to rebuild the bundle.
    """
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        if (directory / OWNERSHIP_MARKER_NAME).is_file():
            return MarkerVerdict(
                kind="owned",
                reason="ownership marker present, nothing published yet",
                bundle_key=directory.name,
            )
        return MarkerVerdict(kind="absent", reason="no bundle marker")

    try:
        document = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return MarkerVerdict(kind="invalid", reason="manifest is not readable JSON")
    if not isinstance(document, dict):
        return MarkerVerdict(kind="invalid", reason="manifest is not a JSON object")

    try:
        validate_manifest_schema(document, require_active_generation=True)
    except DistillError as exc:
        return MarkerVerdict(
            kind="invalid",
            reason=f"manifest schema is invalid: {exc.details.get('field', 'unknown field')}",
        )

    identity = recorded_identity(document)
    if identity != directory.name:
        return MarkerVerdict(
            kind="foreign",
            reason=f"manifest records bundle key {identity!r}, not {directory.name!r}",
            bundle_key=identity,
        )
    return MarkerVerdict(
        kind="published",
        reason="manifest records this directory's bundle key",
        bundle_key=identity,
        manifest=document,
    )


def recorded_identity(manifest: dict[str, Any]) -> str | None:
    """The **bundle key** a manifest records, under either accepted field name."""
    for field in IDENTITY_FIELDS:
        value = manifest.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def is_generation_name(name: str) -> bool:
    """Whether `name` names a **generation** (`g1`, `g2`, ...)."""
    return name.startswith(GENERATION_PREFIX) and name[len(GENERATION_PREFIX) :].isdigit()


def validate_manifest_schema(
    manifest: dict[str, Any],
    *,
    require_active_generation: bool,
) -> None:
    """Raise `E_BAD_MANIFEST` unless the manifest carries the fields a bundle needs.

    Identity is checked separately from the typed fields because it has two
    accepted names: the current `bundle_key` and the legacy `source_hash`.
    """
    required_types: dict[str, type | tuple[type, ...]] = {
        "pipeline_version": int,
        "distill_version": str,
        "source_type": str,
        "source_resolved_path": str,
        "duration_sec": (int, float),
        "options": dict,
        "frame_count": int,
        "transcript_present": bool,
        "warning_count": int,
        "frames": list,
        "warnings": list,
    }
    if require_active_generation:
        required_types["active_generation"] = str
    for key, expected_type in required_types.items():
        value = manifest.get(key)
        if not isinstance(value, expected_type):
            expected_name = (
                " or ".join(item.__name__ for item in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise DistillError(
                "E_BAD_MANIFEST",
                "bundle",
                "cache manifest schema is invalid",
                {"field": key, "expected": expected_name},
            )
    if recorded_identity(manifest) is None:
        raise DistillError(
            "E_BAD_MANIFEST",
            "bundle",
            "cache manifest schema is invalid",
            {"field": " or ".join(IDENTITY_FIELDS), "expected": "str"},
        )


def read_manifest(bundle_root: Path) -> dict[str, Any] | None:
    """The published **manifest**, validated. Prefer `BundleStore.load_active`."""
    manifest = bundle_root / MANIFEST_NAME
    if not manifest.exists():
        return None
    with manifest.open() as handle:
        data = json.load(handle)
    validate_manifest_schema(data, require_active_generation=True)
    return data


def active_paths(bundle_root: Path) -> BundlePaths | None:
    """Paths to the **active generation** without proving it exists.

    Kept for callers this plan has not migrated yet; `BundleStore.load_active`
    is the surface that answers "is there a bundle to serve?" (R-04).
    """
    manifest = read_manifest(bundle_root)
    if not manifest:
        return None
    generation = bundle_root / str(manifest["active_generation"])
    return BundlePaths(
        root=bundle_root,
        generation=generation,
        frames=generation / FRAMES_DIR_NAME,
        manifest=bundle_root / MANIFEST_NAME,
        transcript=generation / TRANSCRIPT_NAME,
        markdown=generation / RENDER_NAME,
    )


def next_generation(bundle_root: Path) -> str:
    existing = [
        int(path.name[len(GENERATION_PREFIX) :])
        for path in bundle_root.glob(f"{GENERATION_PREFIX}*")
        if path.is_dir() and is_generation_name(path.name)
    ]
    return f"{GENERATION_PREFIX}{max(existing, default=0) + 1}"


def ensure_safe_directory(path: Path, root: Path, *, create_leaf: bool = True) -> Path:
    """Validate `path` as a write target under `root`, creating what is missing.

    The single symlink refusal in Distill (R-16, D-041). It walks every component
    below `root`, refusing any that is a symlink, and refuses a target that
    resolves outside `root` at all.

    `create_leaf=False` validates a target without creating it, which is what
    lets one checker serve a file write as well as a directory: the parents are
    created, the final component is checked and left alone. A file target went
    unchecked before, so a symlink pre-created at, say, `video.md` redirected the
    write out of the bundle (S1).

    Confinement is decided lexically and the walk is over the components as
    written, not as resolved. Resolving first and then walking inspects a path
    with every symlink already followed, which sees no symlink at all - so a link
    pointing back inside the root passed. Between them the two rules leave no
    third case: `..` cannot survive normalization, and a path that reaches
    outside the root can only do so through a component the walk refuses.
    """
    target = path if path.is_absolute() else root / path
    lexical_root = Path(os.path.normpath(root.absolute()))
    lexical_target = Path(os.path.normpath(target.absolute()))
    if not lexical_target.is_relative_to(lexical_root):
        raise DistillError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "bundle path must stay under output_dir",
            {"path": str(target), "output_root": str(lexical_root)},
        )

    relative_parts = lexical_target.relative_to(lexical_root).parts
    current = lexical_root
    leaf_index = len(relative_parts) - 1
    for index, part in enumerate(relative_parts):
        current = current / part
        if current.is_symlink():
            raise DistillError(
                "E_BAD_OUTPUT_DIR",
                "bundle",
                "output tree must not contain symlink components",
                {"path": str(current), "output_root": str(lexical_root)},
            )
        if create_leaf or index < leaf_index:
            current.mkdir(exist_ok=True)
    return target


def stage_paths(bundle_root: Path, *, reset: bool = True) -> BundlePaths:
    """Open the **staging directory** for the next **generation**."""
    generation_name = next_generation(bundle_root)
    tmp = bundle_root / f"{STAGING_PREFIX}{generation_name}"
    # Validated before the reset below, so a symlinked staging path is refused
    # rather than deleted through.
    ensure_safe_directory(tmp, bundle_root, create_leaf=False)
    if tmp.exists() and reset:
        shutil.rmtree(tmp)
    frames = tmp / FRAMES_DIR_NAME
    ensure_safe_directory(frames, bundle_root)
    return BundlePaths(
        root=bundle_root,
        generation=tmp,
        frames=frames,
        manifest=bundle_root / MANIFEST_NAME,
        transcript=tmp / TRANSCRIPT_NAME,
        markdown=tmp / RENDER_NAME,
    )
