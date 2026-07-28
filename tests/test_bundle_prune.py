"""Tests for **prune**: what retention may take, what expiry may take, and what
neither may take while a run is live.

The seams under test are `BundleStore.plan_prune` and `BundleStore.apply_prune`.
Two findings live here. Finding 2 is what happens when retention and expiry are
one operation: `keep_generations=0` proposed every **generation** in a bundle,
the **active generation** included, and deleted output a reader was entitled to.
Finding 17 is what happens when the walk only looks at the top level: a bundle
under a playlist root was never a candidate for anything. RV-1 is what happens
when a plan is trusted: it is computed without a lock, so a run may publish into
a bundle between the plan and its application.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from distill.bundle_store import (
    BundleStore,
    ExclusiveLock,
    PrunePlan,
    PrunePolicy,
    PruneTarget,
    prune_lock_path,
)
from distill.errors import DistillError

DAY = 86400.0


def manifest_document(bundle_key: str, active: str) -> dict:
    """A schema-valid **manifest**: the **bundle marker** prune requires (R-01)."""
    return {
        "pipeline_version": 1,
        "distill_version": "0.1.0",
        "source_type": "local",
        "bundle_key": bundle_key,
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
    parent: Path,
    bundle_key: str = "abc123",
    *,
    generations: tuple[str, ...] = ("g1",),
    active: str = "g1",
    staging: tuple[str, ...] = (),
) -> Path:
    """Lay out a published **bundle** on disk, marker included."""
    directory = parent / bundle_key
    directory.mkdir(parents=True)
    for name in generations:
        (directory / name).mkdir()
        (directory / name / "video.md").write_text(f"# {name}\n")
    for name in staging:
        (directory / name).mkdir()
    (directory / "_manifest.json").write_text(
        json.dumps(manifest_document(bundle_key, active), indent=2) + "\n"
    )
    return directory


def write_owned(parent: Path, bundle_key: str = "abc123", *, staging: str | None = None) -> Path:
    """A directory carrying only the **ownership marker**: Distill's, unpublished."""
    directory = parent / bundle_key
    directory.mkdir(parents=True)
    (directory / "_owner.json").write_text(json.dumps({"bundle_key": bundle_key}) + "\n")
    if staging is not None:
        (directory / staging).mkdir()
    return directory


def age_everything(directory: Path, days: float) -> None:
    """Backdate a directory and everything in it by `days`."""
    stamp = time.time() - days * DAY
    for path in sorted(directory.rglob("*"), reverse=True):
        os.utime(path, (stamp, stamp))
    os.utime(directory, (stamp, stamp))


def hold_lock(bundle_root: Path) -> ExclusiveLock:
    """Hold the run lock for `bundle_root` the way a live run holds it."""
    path = prune_lock_path(bundle_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = ExclusiveLock.take(bundle_root.name, path)
    assert lock is not None
    return lock


def target_paths(plan: PrunePlan) -> set[Path]:
    return {target.path for target in plan.targets}


# --- Retention never takes the active generation (R-02, finding 2) ----------


def test_keep_generations_zero_is_refused_rather_than_wiping_the_active_generation(
    tmp_path: Path,
) -> None:
    """Finding 2, at the input that caused it.

    The old retention kept `len(generations) - keep_generations` of the oldest
    as candidates, so zero proposed all of them and took the **active
    generation** with the rest. R-03 refuses the value outright: there is no
    reading of "keep zero generations" that a bundle survives, so it is a bad
    option rather than an aggressive one.
    """
    with pytest.raises(DistillError) as raised:
        PrunePolicy(keep_generations=0)

    assert raised.value.code == "E_BAD_OPTIONS"
    assert raised.value.stage == "prune"


@pytest.mark.parametrize("keep", [-5, -1, 0])
def test_keep_generations_below_one_raises_bad_options(keep: int) -> None:
    with pytest.raises(DistillError) as raised:
        PrunePolicy(keep_generations=keep)

    assert raised.value.code == "E_BAD_OPTIONS"


@pytest.mark.parametrize("keep", ["3", 2.5, None, True])
def test_keep_generations_must_be_an_integer(keep: object) -> None:
    """`True` is an `int` in Python; meaning 1 by that coincidence is not a policy."""
    with pytest.raises(DistillError) as raised:
        PrunePolicy(keep_generations=keep)  # ty: ignore[invalid-argument-type]

    assert raised.value.code == "E_BAD_OPTIONS"


@pytest.mark.parametrize("keep", [1, 2, 3, 4, 10])
@pytest.mark.parametrize("active", ["g1", "g2", "g4"])
def test_retention_never_proposes_the_active_generation_at_any_keep_value(
    tmp_path: Path, keep: int, active: str
) -> None:
    """R-02: the rule binds at every value, not only at the one that broke.

    The active generation is subtracted before the count is applied, so it
    survives even when it is the *oldest* generation on disk and every newer one
    is being reclaimed - which is the case a "keep the newest N" rule gets wrong.
    """
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2", "g3", "g4"), active=active)
    store = BundleStore.open(root)

    plan = store.plan_prune(PrunePolicy(keep_generations=keep))
    store.apply_prune(plan)

    assert bundle / active not in target_paths(plan)
    assert (bundle / active).is_dir()


def test_retention_proposes_only_generations_beyond_the_newest_kept(tmp_path: Path) -> None:
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2", "g3", "g4"), active="g4")
    store = BundleStore.open(root)

    outcome = store.apply_prune(store.plan_prune(PrunePolicy(keep_generations=2)))

    assert set(outcome.deleted) == {bundle / "g1", bundle / "g2"}
    assert (bundle / "g3").is_dir()
    assert (bundle / "g4").is_dir()


def test_a_plan_deletes_nothing_on_its_own(tmp_path: Path) -> None:
    """A `PrunePlan` is a proposal; producing one is what a dry run is (D-023)."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2"), active="g2")
    store = BundleStore.open(root)

    plan = store.plan_prune(PrunePolicy(keep_generations=1))

    assert target_paths(plan) == {bundle / "g1"}
    assert (bundle / "g1").is_dir()


def test_a_plan_over_an_absent_root_considers_nothing(tmp_path: Path) -> None:
    """The control for the refusal below: an absent root is a knowable answer."""
    plan = BundleStore.open(tmp_path / "never-created").plan_prune(
        PrunePolicy(keep_generations=1)
    )

    assert plan.targets == ()
    assert plan.considered == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not refused by directory permissions")
def test_a_plan_over_a_root_that_cannot_be_reached_is_refused(tmp_path: Path) -> None:
    """FAILS FIRST: an empty plan for a root nobody was able to ask about.

    Prune's guard asked `Path.is_dir()`, which answers `False` both for a root
    that is not there and for one whose parent this user may not search. The
    second answer is a guess, and an empty plan reported as "considered 0" tells
    an operator their cache holds nothing prunable when it may hold everything.

    Refused rather than skipped, because the skip machinery exists so that one
    unreadable directory does not cost the report on every other one - and when
    it is the root that cannot be reached there is no other one.
    """
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    unreachable = sealed / "output"
    sealed.chmod(0o000)
    try:
        with pytest.raises(DistillError) as failure:
            BundleStore.open(unreachable).plan_prune(PrunePolicy(keep_generations=1))
    finally:
        sealed.chmod(0o700)

    assert failure.value.code == "E_OUTPUT_ROOT_UNREADABLE"
    assert failure.value.stage == "bundle"
    assert failure.value.details == {"root": str(unreachable), "errno": "EACCES"}


# --- max_age_days validation (R-03) ----------------------------------------


@pytest.mark.parametrize("days", [0, -1, -0.5, float("inf"), float("nan"), "30"])
def test_max_age_days_outside_its_domain_raises_bad_options(days: object) -> None:
    with pytest.raises(DistillError) as raised:
        PrunePolicy(max_age_days=days)  # ty: ignore[invalid-argument-type]

    assert raised.value.code == "E_BAD_OPTIONS"


def test_max_age_days_none_means_no_expiry_at_all(tmp_path: Path) -> None:
    """`None` is "never expire", which is not the same as a horizon of zero days."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1",), active="g1")
    age_everything(bundle, days=400)
    store = BundleStore.open(root)

    outcome = store.apply_prune(store.plan_prune(PrunePolicy(max_age_days=None)))

    assert outcome.deleted == ()
    assert bundle.is_dir()


# --- Expiry takes the whole bundle (D-018) ---------------------------------


def test_expiry_removes_a_whole_aged_bundle_including_its_active_generation(
    tmp_path: Path,
) -> None:
    """D-018: expiry's purpose is to remove a bundle, active generation included."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2"), active="g2")
    age_everything(bundle, days=40)
    store = BundleStore.open(root)

    plan = store.plan_prune(PrunePolicy(max_age_days=30))
    outcome = store.apply_prune(plan)

    assert [target.rule for target in plan.targets] == ["expiry"]
    assert outcome.deleted == (bundle,)
    assert not bundle.exists()


def test_expiry_leaves_a_bundle_republished_inside_the_horizon(tmp_path: Path) -> None:
    """Age is the newest thing in the bundle, not the oldest generation in it."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2"), active="g2")
    age_everything(bundle, days=400)
    now = time.time()
    os.utime(bundle / "g2", (now, now))
    store = BundleStore.open(root)

    outcome = store.apply_prune(store.plan_prune(PrunePolicy(max_age_days=30)))

    assert outcome.deleted == ()
    assert bundle.is_dir()


def test_retention_and_expiry_are_distinct_operations(tmp_path: Path) -> None:
    """The same bundle, at the same age, under the two rules (D-018).

    Retention refuses to take the **active generation** at any
    `keep_generations`; expiry takes it along with everything else. That
    difference is the whole of D-018, and collapsing the two is what made
    finding 2 possible.
    """
    root = tmp_path / "output"
    retained = write_bundle(root, "retention", generations=("g1", "g2", "g3"), active="g3")
    expired = write_bundle(root, "expiry", generations=("g1", "g2", "g3"), active="g3")
    age_everything(retained, days=400)
    age_everything(expired, days=400)
    store = BundleStore.open(root)

    retention_plan = store.plan_prune(PrunePolicy(keep_generations=1))
    expiry_plan = store.plan_prune(PrunePolicy(keep_generations=1, max_age_days=30))

    retention_targets = {
        target.path for target in retention_plan.targets if target.bundle_root == retained
    }
    expiry_targets = {
        target.path for target in expiry_plan.targets if target.bundle_root == expired
    }
    assert retention_targets == {retained / "g1", retained / "g2"}
    assert expiry_targets == {expired}
    assert {target.rule for target in retention_plan.targets} == {"retention"}
    assert {target.rule for target in expiry_plan.targets} == {"expiry"}


# --- Nested roots (R-05, finding 17) ---------------------------------------


def test_bundles_under_a_playlist_root_are_pruned(tmp_path: Path) -> None:
    """Finding 17: a playlist gives each item its own output root, two levels down.

    Nothing under `playlists/` was ever a prune candidate, so a playlist's
    bundles grew without bound while the same policy reclaimed top-level ones.
    """
    root = tmp_path / "output"
    nested = write_bundle(
        root / "playlists" / "PL-demo", generations=("g1", "g2", "g3"), active="g3"
    )
    store = BundleStore.open(root)

    outcome = store.apply_prune(store.plan_prune(PrunePolicy(keep_generations=1)))

    assert set(outcome.deleted) == {nested / "g1", nested / "g2"}
    assert (nested / "g3").is_dir()


def test_nested_bundle_roots_are_traversed_and_bundles_are_not_descended_into(
    tmp_path: Path,
) -> None:
    """The walk descends through containers and stops at bundles.

    A **generation** is a directory inside a bundle, not a nested bundle root:
    descending into one would let a directory a bundle happens to contain be
    judged as though it were a bundle of its own.
    """
    root = tmp_path / "output"
    top = write_bundle(root, "top", generations=("g1",), active="g1")
    deep = write_bundle(root / "playlists" / "PL-demo" / "season-1", "deep", active="g1")
    store = BundleStore.open(root)

    plan = store.plan_prune(PrunePolicy(keep_generations=1))

    scanned = {skip.path for skip in plan.skipped}
    assert root / "playlists" in scanned
    assert root / "playlists" / "PL-demo" in scanned
    assert top / "g1" not in scanned
    assert deep / "g1" not in scanned


def test_a_markerless_directory_is_skipped_and_reported(tmp_path: Path) -> None:
    """R-01: `g1/video.md` is not a bundle marker, and the skip is the report."""
    root = tmp_path / "output"
    (root / "notes" / "g1").mkdir(parents=True)
    (root / "notes" / "g1" / "video.md").write_text("# my own notes\n")
    store = BundleStore.open(root)

    outcome = store.apply_prune(store.plan_prune(PrunePolicy(keep_generations=1)))

    assert outcome.deleted == ()
    assert (root / "notes" / "g1" / "video.md").is_file()
    reasons = {skip.path: skip.reason for skip in outcome.skipped}
    assert reasons[root / "notes"] == "no bundle marker"


def test_reserved_directories_are_skipped_with_a_reason(tmp_path: Path) -> None:
    root = tmp_path / "output"
    write_bundle(root, generations=("g1",), active="g1")
    (root / "_jobs").mkdir()
    store = BundleStore.open(root)

    plan = store.plan_prune(PrunePolicy(keep_generations=1))

    skipped = {skip.path: skip for skip in plan.skipped}
    assert skipped[root / "_jobs"].verdict == "reserved"


# --- Concurrency: the plan is advisory, the lock is authoritative (R-10) ----


def test_a_plan_applied_after_a_concurrent_publish_spares_the_new_active_generation(
    tmp_path: Path,
) -> None:
    """RV-1: the plan was right when it was made and wrong when it was applied.

    A directory holding only an **ownership marker** has published nothing, so
    prune proposes the whole thing. A run that publishes between the plan and its
    application turns it into a **bundle** with an **active generation** - and
    applying the stale plan would delete a generation a reader is entitled to.
    Re-deriving under the lock is what makes that impossible (D-023).
    """
    root = tmp_path / "output"
    bundle = write_owned(root)
    store = BundleStore.open(root)
    plan = store.plan_prune(PrunePolicy(keep_generations=1))
    assert target_paths(plan) == {bundle}

    (bundle / "g1").mkdir()
    (bundle / "g1" / "video.md").write_text("# published while the plan sat\n")
    (bundle / "_manifest.json").write_text(
        json.dumps(manifest_document("abc123", "g1"), indent=2) + "\n"
    )

    outcome = store.apply_prune(plan)

    assert outcome.deleted == ()
    assert (bundle / "g1" / "video.md").read_text() == "# published while the plan sat\n"
    assert outcome.retained[0].verdict == "skipped"
    assert "revalidation under lock" in outcome.retained[0].reason


def test_a_stale_expiry_plan_does_not_delete_a_bundle_that_was_republished(
    tmp_path: Path,
) -> None:
    """The same staleness on the expiry path: an aged bundle stops being aged."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1",), active="g1")
    age_everything(bundle, days=400)
    store = BundleStore.open(root)
    plan = store.plan_prune(PrunePolicy(max_age_days=30))
    assert target_paths(plan) == {bundle}

    (bundle / "g2").mkdir()
    (bundle / "g2" / "video.md").write_text("# republished\n")
    (bundle / "_manifest.json").write_text(
        json.dumps(manifest_document("abc123", "g2"), indent=2) + "\n"
    )

    outcome = store.apply_prune(plan)

    assert outcome.deleted == ()
    assert (bundle / "g2" / "video.md").is_file()


def test_apply_prune_skips_every_target_of_a_bundle_a_live_run_holds(tmp_path: Path) -> None:
    """R-10: the lock is acquired before the delete, not asserted after it."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2", "g3"), active="g3")
    store = BundleStore.open(root)
    plan = store.plan_prune(PrunePolicy(keep_generations=1))
    assert target_paths(plan) == {bundle / "g1", bundle / "g2"}

    lock = hold_lock(bundle)
    try:
        outcome = store.apply_prune(plan)
    finally:
        lock.release()

    assert outcome.deleted == ()
    assert {result.reason for result in outcome.retained} == {
        "another run holds this bundle key"
    }
    assert (bundle / "g1").is_dir()
    assert (bundle / "g2").is_dir()


def test_planning_skips_a_bundle_a_live_run_holds(tmp_path: Path) -> None:
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2"), active="g2")
    store = BundleStore.open(root)

    lock = hold_lock(bundle)
    try:
        plan = store.plan_prune(PrunePolicy(keep_generations=1))
    finally:
        lock.release()

    assert plan.targets == ()
    assert {skip.path: skip.verdict for skip in plan.skipped}[bundle] == "locked"


def test_apply_prune_revalidates_the_marker(tmp_path: Path) -> None:
    """A bundle that stopped being one between plan and apply is not pruned (R-01)."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2"), active="g2")
    store = BundleStore.open(root)
    plan = store.plan_prune(PrunePolicy(keep_generations=1))

    (bundle / "_manifest.json").unlink()
    outcome = store.apply_prune(plan)

    assert outcome.deleted == ()
    assert (bundle / "g1").is_dir()


def test_apply_prune_revalidates_root_confinement(tmp_path: Path) -> None:
    """A target outside the output root is refused however it got into the plan."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1",), active="g1")
    outside = tmp_path / "somebody-elses"
    outside.mkdir()
    (outside / "keep.txt").write_text("mine\n")
    store = BundleStore.open(root)

    plan = PrunePlan(
        root=store.root,
        policy=PrunePolicy(keep_generations=1),
        targets=(
            PruneTarget(
                path=outside,
                kind="bundle",
                rule="orphan",
                bundle_root=bundle,
                bundle_key="abc123",
                reason="hand-built plan",
            ),
        ),
    )
    outcome = store.apply_prune(plan)

    assert outcome.deleted == ()
    assert outcome.retained[0].reason == "target is not confined to the output root"
    assert (outside / "keep.txt").is_file()


def test_apply_prune_refuses_a_bundle_root_outside_the_output_root(tmp_path: Path) -> None:
    """Confinement is checked before the lock path is derived, not only before the delete.

    A bundle root outside the output root would put this store's `_locks`
    directory somewhere it has no business writing, which happens before any
    per-target check could refuse it.
    """
    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "somebody-elses"
    (outside / "g1").mkdir(parents=True)
    store = BundleStore.open(root)

    plan = PrunePlan(
        root=store.root,
        policy=PrunePolicy(keep_generations=1),
        targets=(
            PruneTarget(
                path=outside / "g1",
                kind="generation",
                rule="retention",
                bundle_root=outside,
                bundle_key="somebody-elses",
                reason="hand-built plan",
            ),
        ),
    )
    outcome = store.apply_prune(plan)

    assert outcome.deleted == ()
    assert outcome.retained[0].reason == "target is not confined to the output root"
    assert (outside / "g1").is_dir()
    assert not (tmp_path / "_locks").exists()


def test_apply_prune_spares_a_target_that_became_the_active_generation(
    tmp_path: Path,
) -> None:
    """The narrowest form of RV-1, asserted on its own reason."""
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2"), active="g2")
    store = BundleStore.open(root)
    plan = store.plan_prune(PrunePolicy(keep_generations=1))
    assert target_paths(plan) == {bundle / "g1"}

    (bundle / "_manifest.json").write_text(
        json.dumps(manifest_document("abc123", "g1"), indent=2) + "\n"
    )
    outcome = store.apply_prune(plan)

    assert outcome.deleted == ()
    assert outcome.retained[0].reason == "target is now the active generation"
    assert (bundle / "g1").is_dir()


# --- Orphans and staging (R-06) --------------------------------------------


def test_orphan_generations_are_removed(tmp_path: Path) -> None:
    """A **manifest** naming a generation that is not on disk serves nothing.

    R-04 already makes that bundle a cache miss. What is left beside it is
    **generations** no manifest names, which no reader can reach and retention
    would keep forever.
    """
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1", "g2"), active="g9")
    store = BundleStore.open(root)

    plan = store.plan_prune(PrunePolicy(keep_generations=3))
    outcome = store.apply_prune(plan)

    assert {target.rule for target in plan.targets} == {"orphan"}
    assert set(outcome.deleted) == {bundle / "g1", bundle / "g2"}


def test_a_directory_that_never_published_is_removed_whole(tmp_path: Path) -> None:
    """RV-9's leftovers: an **ownership marker** and nothing that can be served."""
    root = tmp_path / "output"
    bundle = write_owned(root, staging=".tmp.g1")
    store = BundleStore.open(root)

    outcome = store.apply_prune(store.plan_prune(PrunePolicy(keep_generations=3)))

    assert outcome.deleted == (bundle,)
    assert not bundle.exists()


def test_staging_liveness_is_decided_by_the_lock_and_not_by_a_timestamp(
    tmp_path: Path,
) -> None:
    """R-06, stated so a staleness window cannot pass it.

    The live run's **staging directory** is ancient and the dead run's is
    seconds old, so any rule reading mtimes gets both backwards. The lock is the
    same one `begin` takes, so holding it is proof rather than evidence.
    """
    root = tmp_path / "output"
    dead = write_bundle(root, "dead-run", generations=("g1",), active="g1", staging=(".tmp.g2",))
    live = write_bundle(root, "live-run", generations=("g1",), active="g1", staging=(".tmp.g2",))
    age_everything(live / ".tmp.g2", days=400)
    now = time.time()
    os.utime(dead / ".tmp.g2", (now, now))
    store = BundleStore.open(root)

    lock = hold_lock(live)
    try:
        outcome = store.apply_prune(store.plan_prune(PrunePolicy(keep_generations=3)))
    finally:
        lock.release()

    assert outcome.deleted == (dead / ".tmp.g2",)
    assert not (dead / ".tmp.g2").exists()
    assert (live / ".tmp.g2").is_dir()


def test_a_staging_directory_that_became_live_is_spared_at_apply_time(tmp_path: Path) -> None:
    root = tmp_path / "output"
    bundle = write_bundle(root, generations=("g1",), active="g1", staging=(".tmp.g2",))
    store = BundleStore.open(root)
    plan = store.plan_prune(PrunePolicy(keep_generations=3))
    assert target_paths(plan) == {bundle / ".tmp.g2"}

    lock = hold_lock(bundle)
    try:
        outcome = store.apply_prune(plan)
    finally:
        lock.release()

    assert outcome.deleted == ()
    assert (bundle / ".tmp.g2").is_dir()


# --- Per-target verdicts and reasons (R-57, checkbox 14/15) ----------------


def test_every_target_and_every_skip_carries_a_verdict_and_a_reason(tmp_path: Path) -> None:
    root = tmp_path / "output"
    write_bundle(root, "pruned", generations=("g1", "g2", "g3"), active="g3")
    write_owned(root, "unpublished")
    (root / "notes").mkdir()
    store = BundleStore.open(root)

    outcome = store.apply_prune(store.plan_prune(PrunePolicy(keep_generations=1)))

    assert outcome.results
    for result in outcome.results:
        assert result.verdict in ("deleted", "skipped")
        assert result.reason
        assert result.target.rule in ("retention", "expiry", "orphan", "staging")
    assert outcome.skipped
    for skip in outcome.skipped:
        assert skip.verdict
        assert skip.reason
    payload = outcome.to_dict()
    assert payload["deleted_count"] == len(outcome.deleted)
    assert payload["skipped_count"] == len(outcome.skipped)


def test_considering_nothing_is_distinguishable_from_deleting_nothing(
    tmp_path: Path,
) -> None:
    """Both prunes delete nothing; only one of them looked at anything."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    busy_root = tmp_path / "busy"
    for name in ("one", "two", "three"):
        (busy_root / name).mkdir(parents=True)
    policy = PrunePolicy(keep_generations=1)

    empty = BundleStore.open(empty_root).apply_prune(
        BundleStore.open(empty_root).plan_prune(policy)
    )
    busy = BundleStore.open(busy_root).apply_prune(
        BundleStore.open(busy_root).plan_prune(policy)
    )

    assert empty.deleted == () and busy.deleted == ()
    assert empty.considered == 0
    assert busy.considered == 3
    assert len(busy.skipped) == 3
