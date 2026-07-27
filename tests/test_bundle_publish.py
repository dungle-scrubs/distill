"""Tests for **publish**: the ordered transition from staging to a generation.

The seams under test are `BundleRun.write_stage` and `BundleRun.commit` - the
only points at which a run records resume scratch and then turns a **staging
directory** into a **generation**.

Finding 4's disk half is what happens when publish is a bare rename: every
**stage result** the run wrote for **resume** was renamed along with the
generation and served as bundle content. R-13 makes that impossible - a
generation never contains a stage result - and R-12 fixes the order that keeps
the transition survivable: assemble, strip, rename, replace the manifest. A
crash in the one gap that order leaves must cost the previous **active
generation** nothing.

D-033 scopes atomicity rather than spreading it: manifests and job records are
read by other processes and are written by atomic replace; **stage results**,
written under the run lock into a directory nothing else may read, are ordinary
writes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from test_bundle_store import published_manifest

from distill import bundle_store
from distill.bundle_store import (
    BundlePaths,
    BundleRun,
    BundleSnapshot,
    BundleStore,
    atomic_write_text,
    orphan_generations,
    publish_staging,
    read_marker,
    stage_paths,
)
from distill.errors import DistillError
from distill.job_store import JobStore

BUNDLE_KEY = "b0a1c2d3"


class StepClock:
    """A monotonic clock a test advances by hand.

    A staging duration is a claim about the run, not about how long the test
    took. Driving the clock explicitly makes the reported duration exact and
    keeps the test instant.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def manifest_for(bundle_key: str = BUNDLE_KEY) -> dict[str, Any]:
    return published_manifest(identity_field="bundle_key", identity=bundle_key)


def bundle_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == bundle_store.LOGGER.name and record.message.startswith("{")
    ]


def begin_run(store: BundleStore, bundle_key: str = BUNDLE_KEY) -> BundleRun:
    run = store.begin(bundle_key)
    assert isinstance(run, BundleRun)
    return run


def assemble(paths: Any, text: str = "# Video\n") -> None:
    """Write the **render** a servable generation must carry."""
    paths.markdown.write_text(text)


def test_a_published_generation_holds_no_stage_result(tmp_path: Path) -> None:
    """Finding 4, disk half: publish renamed the resume scratch into the bundle.

    A **stage result** is scratch an interrupted run reads back to **resume**.
    It has not been through a **redaction sink**, and nothing about a
    **generation** claims it has. Renaming the staging directory published it
    anyway, so `g1/_ocr.json` sat inside the bundle carrying whatever the OCR
    pass read off the screen.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    staged = stage_paths(bundle_root)
    assemble(staged)
    (staged.generation / "_ocr.json").write_text(
        json.dumps({"frames": [{"ocr_text": "sk-live-not-for-the-bundle"}]}) + "\n"
    )

    final = publish_staging(staged, manifest_for())

    assert list(final.generation.rglob("_*.json")) == []
    assert not (final.generation / "_ocr.json").exists()
    assert "sk-live-not-for-the-bundle" not in "".join(
        path.read_text() for path in final.generation.rglob("*") if path.is_file()
    )


def test_commit_publishes_a_generation_holding_no_stage_result(tmp_path: Path) -> None:
    """R-13 at the seam a run actually uses: `write_stage` then `commit`."""
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    run = begin_run(store)
    run.write_stage("ocr", {"frames": [{"ocr_text": "scratch"}]})
    assert run.read_stage("ocr") == {"frames": [{"ocr_text": "scratch"}]}
    assemble(run.paths)
    snapshot = run.commit(manifest_for())

    assert isinstance(snapshot, BundleSnapshot)
    assert snapshot.generation.name == "g1"
    assert snapshot.markdown.read_text() == "# Video\n"
    assert list(snapshot.generation.rglob("_*.json")) == []


def test_stage_results_are_removed_before_the_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-12: the strip precedes the rename, so no ordering makes them visible.

    Stripping after the rename would leave a window in which the generation is
    on disk holding stage results. Failing the rename proves the strip already
    happened: the staging directory is left with none.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    staged = stage_paths(bundle_root)
    assemble(staged)
    (staged.generation / "_ocr.json").write_text("{}\n")

    def refuse_rename(self: Path, target: Any) -> Path:
        raise OSError("rename interrupted")

    monkeypatch.setattr(Path, "rename", refuse_rename)

    with pytest.raises(OSError):
        publish_staging(staged, manifest_for())

    assert staged.generation.is_dir()
    assert not (staged.generation / "_ocr.json").exists()
    assert not (bundle_root / "g1").exists()


def test_commit_order_is_assemble_strip_rename_replace_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-12: the whole order, observed as the filesystem operations it performs."""
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    staged = stage_paths(bundle_root)
    assemble(staged)
    (staged.generation / "_ocr.json").write_text("{}\n")

    trace: list[str] = []
    real_unlink = Path.unlink
    real_rename = Path.rename
    real_replace = Path.replace

    def unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name.startswith("_") and self.suffix == ".json":
            trace.append(f"strip:{self.name}")
        return real_unlink(self, *args, **kwargs)

    def rename(self: Path, target: Any) -> Path:
        trace.append(f"rename:{Path(target).name}")
        return real_rename(self, target)

    def replace(self: Path, target: Any) -> Path:
        trace.append(f"replace:{Path(target).name}")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(Path, "replace", replace)

    final = publish_staging(staged, manifest_for())

    assert trace == ["strip:_ocr.json", "rename:g1", "replace:_manifest.json"]
    # Assembly happened in staging, not after the rename: the render arrived
    # with the directory rather than being written into it afterwards.
    assert final.markdown.read_text() == "# Video\n"


def test_a_crash_between_rename_and_manifest_replace_keeps_the_active_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-12: the one gap the order leaves costs a reader nothing.

    Between the rename and the manifest replace there is a moment where `g2`
    exists and the manifest still names `g1`. That is the ordering's whole
    point: the **manifest** is what makes a generation active, so a run that
    dies in the gap leaves the previous **active generation** exactly as it
    was - servable, and named by a manifest that was never half-written.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    store = BundleStore.open(tmp_path)

    first = stage_paths(bundle_root)
    assemble(first, "# first\n")
    publish_staging(first, manifest_for())
    first_manifest_bytes = (bundle_root / "_manifest.json").read_bytes()

    second = stage_paths(bundle_root)
    assemble(second, "# second\n")

    def crash(*_args: Any, **_kwargs: Any) -> Path:
        raise RuntimeError("killed between the rename and the manifest replace")

    monkeypatch.setattr(bundle_store, "write_manifest", crash)

    with pytest.raises(RuntimeError):
        publish_staging(second, manifest_for())

    snapshot = store.load_active(BUNDLE_KEY)
    assert snapshot is not None
    assert snapshot.generation.name == "g1"
    assert snapshot.markdown.read_text() == "# first\n"
    assert (bundle_root / "_manifest.json").read_bytes() == first_manifest_bytes


def test_the_generation_left_by_such_a_crash_is_a_prunable_orphan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-12: what the crash leaves is an **orphan generation**, not a stray file.

    The directory is a complete, finished generation that no **manifest** names.
    Naming it as such is what lets prune reclaim it (M3.4) rather than leaving
    it to accumulate as unattributable disk.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()

    first = stage_paths(bundle_root)
    assemble(first, "# first\n")
    publish_staging(first, manifest_for())
    assert orphan_generations(bundle_root, read_marker(bundle_root)) == []

    second = stage_paths(bundle_root)
    assemble(second, "# second\n")

    def crash(*_args: Any, **_kwargs: Any) -> Path:
        raise RuntimeError("killed between the rename and the manifest replace")

    monkeypatch.setattr(bundle_store, "write_manifest", crash)
    with pytest.raises(RuntimeError):
        publish_staging(second, manifest_for())

    assert (bundle_root / "g2").is_dir()
    assert orphan_generations(bundle_root, read_marker(bundle_root)) == [
        bundle_root / "g2"
    ]


def test_the_manifest_is_written_by_atomic_replace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-14/D-033: another process may read the manifest at any moment.

    A manifest written in place is readable half-written, and a half-written
    manifest is an invalid marker - the bundle stops being recognizable while
    the bytes land. The replace is what makes the transition indivisible.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    staged = stage_paths(bundle_root)
    assemble(staged)

    written: list[Path] = []
    replaced: list[tuple[Path, Path]] = []
    real_write_text = Path.write_text
    real_replace = Path.replace

    def write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        written.append(self)
        return real_write_text(self, *args, **kwargs)

    def replace(self: Path, target: Any) -> Path:
        replaced.append((self, Path(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "replace", replace)

    publish_staging(staged, manifest_for())

    manifest = bundle_root / "_manifest.json"
    assert manifest not in written
    assert [target for _source, target in replaced] == [manifest]
    assert json.loads(manifest.read_text())["active_generation"] == "g1"


def test_job_records_are_written_by_atomic_replace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-14: a caller polls a job record while the run is rewriting it.

    A job record read half-written is unparseable JSON, which a poller cannot
    distinguish from a job that never started (finding 12's neighbourhood).
    """
    root = tmp_path / "cache"

    written: list[Path] = []
    replaced: list[tuple[Path, Path]] = []
    real_write_text = Path.write_text
    real_replace = Path.replace

    def write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        written.append(self)
        return real_write_text(self, *args, **kwargs)

    def replace(self: Path, target: Any) -> Path:
        replaced.append((self, Path(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "replace", replace)

    store = JobStore.open(root)
    store.start("job-1", "process_local_video")

    record = store.record_path("job-1")
    assert record not in written
    assert [target for _source, target in replaced] == [record]
    assert json.loads(record.read_text())["status"] == "running"


def test_a_stage_result_is_an_ordinary_write_not_an_atomic_replace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D-033 scopes atomicity rather than applying it everywhere.

    A **stage result** is written into a **staging directory** held under the
    run lock, which no other process may read and no reader is ever entitled
    to. Nothing observes a torn stage result, and the run that wrote it
    validates what it reads back. Making every write atomic would spend two
    filesystem operations per stage to protect a reader that does not exist,
    and would blur where the boundary actually is.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)

    written: list[Path] = []
    replaced: list[Path] = []
    real_write_text = Path.write_text
    real_replace = Path.replace

    def write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        written.append(self)
        return real_write_text(self, *args, **kwargs)

    def replace(self: Path, target: Any) -> Path:
        replaced.append(Path(target))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "replace", replace)

    run.write_stage("ocr", {"frames": []})

    stage_result = run.paths.generation / "_ocr.json"
    assert written == [stage_result]
    assert replaced == []
    assert json.loads(stage_result.read_text()) == {"frames": []}


def test_two_atomic_writers_of_one_path_do_not_share_a_temporary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-14: a shared temporary name is worse than no temporary at all.

    Two writers using one temporary name interleave into it: the second
    truncates the file while the first replaces it onto the target, so the
    replace publishes a half-written document - the torn read it exists to
    prevent, moved from the target to the temporary and back again.
    """
    root = tmp_path / "output"
    root.mkdir()

    temporaries: list[Path] = []
    real_replace = Path.replace

    def replace(self: Path, target: Any) -> Path:
        temporaries.append(self)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)

    atomic_write_text(root / "_manifest.json", "first\n", root=root)
    atomic_write_text(root / "_manifest.json", "second\n", root=root)

    assert len(set(temporaries)) == 2
    assert (root / "_manifest.json").read_text() == "second\n"
    assert sorted(path.name for path in root.iterdir()) == ["_manifest.json"]


def test_a_stage_name_outside_the_bounded_domain_is_refused(tmp_path: Path) -> None:
    """R-13 is about what is on disk, not about the discipline of callers.

    A stage name reaches the filesystem, so an unbounded one is a path.
    `ocr/sub` writes a file the strip's glob does not match - published scratch
    - and `x/../../_manifest` names the **manifest** itself, so recording a
    stage result would overwrite the **bundle marker**. Both are refused rather
    than sanitized: mapping two names onto one file is R-18's mistake.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    marker_before = (root / BUNDLE_KEY / "_owner.json").read_bytes()

    for name in ("ocr/sub", "x/../../_manifest", "..", "", "a" * 65, "with space"):
        with pytest.raises(DistillError) as failure:
            run.write_stage(name, {"frames": []})
        assert failure.value.code == "E_BAD_STAGE_NAME"

    assert (root / BUNDLE_KEY / "_owner.json").read_bytes() == marker_before
    assert list((root / BUNDLE_KEY).rglob("_*.json")) == [root / BUNDLE_KEY / "_owner.json"]


def test_an_amendment_may_not_set_identity_or_the_active_generation(
    tmp_path: Path,
) -> None:
    """Both are publish's to decide, so amending them is a publish without one.

    Setting `active_generation` from an amendment names a generation nothing
    renamed into place - a manifest promising a directory that need not exist,
    which is finding 2 from the other side.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    assemble(run.paths)
    snapshot = run.commit(manifest_for())
    on_disk = (root / BUNDLE_KEY / "_manifest.json").read_bytes()

    for fields in ({"active_generation": "g9"}, {"bundle_key": "somebody-elses"}):
        with pytest.raises(DistillError) as failure:
            store.patch_published(snapshot, fields)
        assert failure.value.code == "E_BAD_MANIFEST"

    assert (root / BUNDLE_KEY / "_manifest.json").read_bytes() == on_disk

    amended = store.patch_published(snapshot, {"progress": {"overall_percent": 100.0}})
    assert amended.manifest["progress"] == {"overall_percent": 100.0}
    assert amended.generation == snapshot.generation
    assert store.load_active(BUNDLE_KEY) is not None


def test_a_job_record_does_not_follow_a_symlink_at_the_jobs_directory(
    tmp_path: Path,
) -> None:
    """R-16: `_jobs` is a component below the root, so it is walked, not assumed.

    A confinement root is compared against, never inspected. Passing `_jobs`
    itself as the root would leave the one component that can be pre-created as
    a link the only one nothing checks.
    """
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "_jobs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DistillError) as failure:
        JobStore.open(root).start("job-1", "process_local_video")

    assert failure.value.code == "E_BAD_OUTPUT_DIR"
    assert list(outside.iterdir()) == []


def test_the_snapshot_a_commit_returns_describes_what_is_on_disk(tmp_path: Path) -> None:
    """A caller reads the returned manifest instead of re-reading the bundle.

    Frame paths are recorded absolute against the **staging directory**, which
    stops existing under that name at the rename. A snapshot carrying the
    pre-publish document hands back paths into `.tmp.g1` - a directory that is
    gone - while the manifest on disk carries the right ones.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    assemble(run.paths)
    frame = run.paths.frames / "frame_0001.png"
    frame.write_bytes(b"png")
    manifest = {
        **manifest_for(),
        "frame_count": 1,
        "frames": [{"index": 1, "timestamp_sec": 0.0, "path": str(frame)}],
    }

    snapshot = run.commit(manifest)

    on_disk = json.loads((root / BUNDLE_KEY / "_manifest.json").read_text())
    assert snapshot.manifest == on_disk
    assert snapshot.manifest["frames"][0]["path"] == str(snapshot.frames / "frame_0001.png")
    assert ".tmp." not in snapshot.manifest["frames"][0]["path"]


def test_commit_reports_the_generation_it_published_and_how_long_staging_took(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A published bundle is otherwise silent about which generation it became.

    The generation name is how a bundle on disk is tied back to the run that
    produced it, and the staging duration is the run's own cost - distinct from
    the lock wait `lock_acquired` already reports, so a slow run and a queued
    one stay distinguishable.
    """
    root = tmp_path / "output"
    root.mkdir()
    clock = StepClock()
    store = BundleStore.open(root, clock=clock.monotonic)
    caplog.set_level(logging.DEBUG, logger=bundle_store.LOGGER.name)

    run = begin_run(store)
    clock.advance(12.5)
    assemble(run.paths)
    run.commit(manifest_for())

    committed = [
        event for event in bundle_events(caplog) if event["event"] == "generation_committed"
    ]
    assert len(committed) == 1
    assert committed[0]["detail"]["bundle_key"] == BUNDLE_KEY
    assert committed[0]["detail"]["generation"] == "g1"
    assert committed[0]["detail"]["staging_duration_sec"] == pytest.approx(12.5)


def test_abandoning_a_run_reports_why_and_leaves_the_bundle_alone(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A run that produced nothing is the case worth explaining.

    Abandoning keeps the **staging directory**: its **stage results** are what
    a later run **resumes** from. What it must not do is disturb the previous
    **active generation**, and what it must record is the reason - a bundle
    that did not change is otherwise indistinguishable from a run that never
    happened.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    caplog.set_level(logging.DEBUG, logger=bundle_store.LOGGER.name)

    first = begin_run(store)
    assemble(first.paths, "# first\n")
    first.commit(manifest_for())

    # Assembled by hand because `begin` short-circuits to a `BundleSnapshot`
    # once a generation is active (R-08); the second run of a bundle that
    # already has one arrives through force-reprocess, which M3.6 migrates.
    bundle_root = root / BUNDLE_KEY
    staged = stage_paths(bundle_root)
    run = BundleRun(
        store=store,
        bundle_key=BUNDLE_KEY,
        paths=staged,
        lock=first.lock,
        staged_at=store.clock(),
    )
    run.write_stage("ocr", {"frames": []})
    run.abandon("vision backend unavailable")

    abandoned = [event for event in bundle_events(caplog) if event["event"] == "run_abandoned"]
    assert len(abandoned) == 1
    assert abandoned[0]["detail"]["reason"] == "vision backend unavailable"
    assert abandoned[0]["detail"]["bundle_key"] == BUNDLE_KEY

    snapshot = store.load_active(BUNDLE_KEY)
    assert snapshot is not None
    assert snapshot.generation.name == "g1"
    assert snapshot.markdown.read_text() == "# first\n"
    # The scratch survives, because resuming from it is what it is for.
    assert (staged.generation / "_ocr.json").is_file()


def test_an_amendment_cannot_reinstate_the_generation_it_read(tmp_path: Path) -> None:
    """FAILS FIRST: a stale snapshot merged back over a newer manifest.

    A cache hit reads the **manifest** and releases the lock - that is what
    makes a hit cheap. If the reader then backfills a field by merging the
    document it captured, everything else that manifest said travels with it,
    including `active_generation`. A run that published `g2` in between is
    undone by a *reader*: the marker names `g1` again and `g2` becomes an
    **orphan generation**, which is finding 2's shape arriving from the side
    nobody was watching.

    The amendment therefore happens under the run lock and against the manifest
    on disk, and declines rather than writes when what it reads back is no
    longer what it was handed.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    first = begin_run(store)
    assemble(first.paths)
    stale = first.commit(manifest_for())

    # Between the read and the amendment, another run publishes over g1.
    second = store.begin(BUNDLE_KEY, reuse_active=False)
    assert isinstance(second, BundleRun)
    assemble(second.paths)
    published = second.commit(manifest_for())
    assert published.generation.name == "g2"

    amended = store.patch_published(stale, {"progress": {"overall_percent": 100.0}})

    active = store.load_active(BUNDLE_KEY)
    assert active is not None
    assert active.manifest["active_generation"] == "g2"
    assert active.manifest.get("progress") is None
    assert amended.generation == stale.generation


def test_an_amendment_defers_to_a_run_that_holds_the_bundle(tmp_path: Path) -> None:
    """A live run is about to publish, so its manifest is the one that matters.

    Amending is a backfill, never the point of the call that makes it: waiting
    for a 40-minute run to finish, or racing it, would both cost more than the
    field is worth. The lock says somebody else owns this bundle's manifest now.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    first = begin_run(store)
    assemble(first.paths)
    snapshot = first.commit(manifest_for())
    on_disk = (root / BUNDLE_KEY / "_manifest.json").read_bytes()

    held = store.begin(BUNDLE_KEY, reuse_active=False)
    assert isinstance(held, BundleRun)
    try:
        amended = store.patch_published(snapshot, {"progress": {"overall_percent": 100.0}})
    finally:
        held.release()

    assert (root / BUNDLE_KEY / "_manifest.json").read_bytes() == on_disk
    assert amended.generation == snapshot.generation


# --- A generation with no render is not publishable (finding 4-codex) -------


def test_publish_refuses_a_generation_carrying_no_render(tmp_path: Path) -> None:
    """A manifest is a promise; the **render** on disk is the evidence (R-04).

    `publish_staging` validated the manifest and then renamed whatever was in
    staging, so a commit that never called `write_render` still made a
    generation active - and `load_active`, which does check the render exists,
    then reported a cache miss for a bundle whose manifest named a live
    generation. The previous active generation had already been superseded.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    staged = stage_paths(bundle_root)

    with pytest.raises(DistillError) as failure:
        publish_staging(staged, manifest_for())

    assert failure.value.code == "E_INCOMPLETE_GENERATION"
    assert staged.generation.is_dir(), "the staging directory must survive to be resumed"
    assert not (bundle_root / "g1").exists()
    assert not (bundle_root / "_manifest.json").exists()


def test_a_commit_without_a_render_leaves_the_active_generation_alone(
    tmp_path: Path,
) -> None:
    """The same refusal at the seam a run uses, with something to lose."""
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    first = begin_run(store)
    assemble(first.paths, "# published\n")
    first.commit(manifest_for())

    second = store.begin(BUNDLE_KEY, reuse_active=False)
    assert isinstance(second, BundleRun)
    try:
        with pytest.raises(DistillError) as failure:
            second.commit(manifest_for())
    finally:
        second.release()

    assert failure.value.code == "E_INCOMPLETE_GENERATION"
    active = store.load_active(BUNDLE_KEY)
    assert active is not None
    assert active.generation.name == "g1"
    assert active.markdown.read_text() == "# published\n"


def test_publish_refuses_a_render_that_is_a_symlink(tmp_path: Path) -> None:
    """R-16 at the last moment before the rename, not only at the write.

    `write_render` refuses to follow a link, but nothing stopped one being put
    there afterwards - and a published generation whose render is a link out of
    the bundle serves bytes prune does not own and cannot reclaim.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    outside = tmp_path / "elsewhere.md"
    outside.write_text("# not the run's output\n")
    staged = stage_paths(bundle_root)
    staged.markdown.symlink_to(outside)

    with pytest.raises(DistillError):
        publish_staging(staged, manifest_for())

    assert not (bundle_root / "g1").exists()
    assert outside.read_text() == "# not the run's output\n"


def test_publish_proves_the_render_inside_the_generation_it_is_renaming(
    tmp_path: Path,
) -> None:
    """The proof is derived from the staging directory, never taken on trust.

    Checking whatever `BundlePaths.markdown` happens to point at proves nothing
    about what is being renamed: a field aimed at the *previous* generation's
    render passes every check and publishes an empty directory over a servable
    bundle. Publish is destructive for what it supersedes, so it re-derives at
    the boundary rather than trusting its input - the same reason `apply_prune`
    re-derives under the lock (R-10).
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    first = begin_run(store)
    assemble(first.paths, "# published\n")
    live = first.commit(manifest_for())

    second = store.begin(BUNDLE_KEY, reuse_active=False)
    assert isinstance(second, BundleRun)
    # Staging holds no render; the field points at the live generation's.
    misdirected = BundlePaths(
        root=second.paths.root,
        generation=second.paths.generation,
        frames=second.paths.frames,
        manifest=second.paths.manifest,
        transcript=second.paths.transcript,
        markdown=live.markdown,
    )
    try:
        with pytest.raises(DistillError) as failure:
            publish_staging(misdirected, manifest_for())
    finally:
        second.release()

    assert failure.value.code == "E_INCOMPLETE_GENERATION"
    active = store.load_active(BUNDLE_KEY)
    assert active is not None
    assert active.generation.name == "g1"
    assert active.markdown.read_text() == "# published\n"


# --- Run scratch is stripped with the stage results (finding 3-opus) --------


def test_a_published_generation_holds_no_run_scratch(tmp_path: Path) -> None:
    """FAILS FIRST (finding 3-opus, R-13): scratch the strip cannot see.

    `scratch_dir` handed a stage the **staging directory** itself, so anything
    a stage wrote there that is not a **stage result** - a decoded audio track,
    a temporary the stage forgot - was renamed into the **generation** and
    published. The strip recognizes stage results by name because it must be
    exhaustive rather than informed, and that same exhaustiveness has to cover
    the scratch a stage names itself.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    run = begin_run(store)
    (run.scratch_dir / "audio.wav").write_bytes(b"RIFF-not-bundle-content")
    assemble(run.paths)
    snapshot = run.commit(manifest_for())

    assert [path.name for path in snapshot.generation.rglob("audio.wav")] == []
    assert sorted(path.name for path in snapshot.generation.iterdir()) == [
        "frames",
        "video.md",
    ]


def test_run_scratch_lives_inside_the_staging_directory_a_resume_reads(
    tmp_path: Path,
) -> None:
    """Scratch is kept for **resume**, so it is removed by publishing, not before.

    Unlinking what a stage wrote would answer finding 3-opus by discarding the
    reason the file exists: a run that is interrupted after a 40-minute decode
    resumes against it. Scratch therefore belongs to the **staging directory**,
    which outlives an interrupted run and stops existing at the publish.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    first = begin_run(store)
    (first.scratch_dir / "audio.wav").write_bytes(b"RIFF-expensive-to-produce")
    first.abandon("interrupted")

    second = begin_run(store)
    try:
        assert (second.scratch_dir / "audio.wav").read_bytes() == b"RIFF-expensive-to-produce"
        assert second.scratch_dir.is_relative_to(second.paths.generation)
    finally:
        second.release()


# --- The replace survives losing power, not only losing a race (9-opus) -----


def test_a_manifest_is_on_disk_before_the_replace_that_publishes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 9-opus, R-14): atomic against a reader, not against a crash.

    `rename` is atomic with respect to another process and says nothing about
    power: the directory entry can reach the platter while the bytes it names
    are still in cache, and what comes back is a zero-length `_manifest.json`.
    That file is the **bundle marker**, so the bundle reads `invalid` - not
    servable, and not something the walk descends into either, which makes the
    generations underneath it unreachable *and* unreclaimable.

    Both barriers are needed and they are different: the file's own data, then
    the directory entry the replace created.
    """
    bundle_root = tmp_path / BUNDLE_KEY
    bundle_root.mkdir()
    synced: list[int] = []
    real_fsync = os.fsync

    def record(fd: int) -> None:
        synced.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(bundle_store.os, "fsync", record)

    atomic_write_text(bundle_root / "_manifest.json", "{}\n", root=bundle_root)

    manifest = bundle_root / "_manifest.json"
    assert manifest.stat().st_ino in synced, "the manifest's own bytes were never flushed"
    assert bundle_root.stat().st_ino in synced, "the replace's directory entry was never flushed"
