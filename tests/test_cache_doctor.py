"""`cache-doctor`: the read-only surface over the destructive operations.

R-57. Findings 1 and 2 were both invisible before they were irreversible - a
directory Distill never wrote proposed for deletion, and an **active
generation** deleted out from under a manifest that still named it. Neither had
a surface a user could ask first. This one reports, for an output root: the
**bundles** found, the **bundle marker** verdict for every directory looked at,
each bundle's **active generation** and **orphan generations**, which run locks
are live and which are leftovers, and what a **prune** would do.

The property that carries the requirement is the *skips*: a directory the walk
declined to treat as a bundle is reported with the reason, so "considered
nothing" and "deleted nothing" are different answers. A doctor that reported
only what it found would say the same empty thing about a healthy cache and
about a root it never descended into.

Read-only is asserted, not assumed: the command is given a root that does not
exist and must not create it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from distill import bundle_store
from distill.bundle_store import BundleStore, ExclusiveLock
from distill.cli import main
from distill.errors import DistillError
from distill.pipeline import call_registered_tool

ROOT_ARGS = {"keep_generations": 2}


def manifest(bundle_key: str, active: str) -> str:
    return (
        json.dumps(
            {
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
            },
            indent=2,
        )
        + "\n"
    )


def write_bundle(
    root: Path,
    bundle_key: str,
    *,
    generations: tuple[str, ...] = ("g1",),
    active: str = "g1",
    staging: tuple[str, ...] = (),
) -> Path:
    """A **bundle**: generations, staging scratch, and the marker proving it is one."""
    directory = root / bundle_key
    directory.mkdir(parents=True)
    for name in (*generations, *staging):
        (directory / name).mkdir()
        (directory / name / "video.md").write_text("# Video\n")
    (directory / "_manifest.json").write_text(manifest(bundle_key, active))
    return directory


def inspect(root: Path, **overrides: object) -> dict:
    """The doctor as a caller reaches it: through the registered tool."""
    return call_registered_tool("cache_doctor", {"output_dir": str(root), **ROOT_ARGS, **overrides})


def bundles_by_key(report: dict) -> dict[str, dict]:
    return {bundle["bundle_key"]: bundle for bundle in report["bundles"]}


def verdicts_by_path(report: dict) -> dict[str, str]:
    """Every directory the walk looked at, bundle or not, and its marker verdict."""
    return {
        entry["path"]: entry["verdict"]
        for entry in (*report["bundles"], *report["skipped"])
    }


def test_cache_doctor_reports_the_bundles_found_under_a_root(tmp_path: Path) -> None:
    """The first question: what is actually here?"""
    root = tmp_path / "cache"
    write_bundle(root, "aaa")
    write_bundle(root, "bbb")

    report = inspect(root)

    assert report["root"] == str(root.resolve())
    assert report["bundle_count"] == 2
    assert sorted(bundles_by_key(report)) == ["aaa", "bbb"]
    assert bundles_by_key(report)["aaa"]["path"] == str(root / "aaa")


def test_cache_doctor_reports_the_marker_verdict_for_every_directory(tmp_path: Path) -> None:
    """R-01: bundle-hood is a verdict with a reason, for the found and the skipped alike."""
    root = tmp_path / "cache"
    write_bundle(root, "published")
    (root / "owned").mkdir()
    (root / "owned" / "_owner.json").write_text(json.dumps({"bundle_key": "owned"}) + "\n")
    (root / "foreign").mkdir()
    (root / "foreign" / "_manifest.json").write_text(manifest("somebody-else", "g1"))
    (root / "invalid").mkdir()
    (root / "invalid" / "_manifest.json").write_text('{"active_generation": "g1"}')
    (root / "not-mine" / "g1").mkdir(parents=True)
    (root / "not-mine" / "g1" / "video.md").write_text("# my own notes\n")

    report = inspect(root)

    verdicts = verdicts_by_path(report)
    assert verdicts[str(root / "published")] == "published"
    assert verdicts[str(root / "owned")] == "owned"
    assert verdicts[str(root / "foreign")] == "foreign"
    assert verdicts[str(root / "invalid")] == "invalid"
    assert verdicts[str(root / "not-mine")] == "absent"
    reasons = {entry["path"]: entry["reason"] for entry in report["skipped"]}
    assert reasons[str(root / "not-mine")] == "no bundle marker"
    assert "somebody-else" in reasons[str(root / "foreign")]


def test_cache_doctor_reports_the_active_generation_of_each_bundle(tmp_path: Path) -> None:
    """Finding 2's blind spot: which generation is the one a reader is entitled to."""
    root = tmp_path / "cache"
    write_bundle(root, "aaa", generations=("g1", "g2", "g3"), active="g3")
    # An **ownership marker** records the **bundle key** it claims. `{}` claims
    # nothing, and pinning it as valid asserted the defect finding 1 is: the
    # marker must name this directory before anything treats the directory as
    # Distill's.
    (root / "owned").mkdir()
    (root / "owned" / "_owner.json").write_text(json.dumps({"bundle_key": "owned"}) + "\n")

    report = inspect(root)

    bundles = bundles_by_key(report)
    assert bundles["aaa"]["active_generation"] == "g3"
    assert bundles["aaa"]["generations"] == ["g1", "g2", "g3"]
    # Distill's, but nothing published: there is no generation to be entitled to.
    assert bundles["owned"]["active_generation"] is None


def test_cache_doctor_reports_orphan_generations(tmp_path: Path) -> None:
    """A generation no **manifest** names: superseded, or a crash before the replace."""
    root = tmp_path / "cache"
    write_bundle(root, "aaa", generations=("g1", "g2", "g3"), active="g2")

    report = inspect(root)

    assert bundles_by_key(report)["aaa"]["orphan_generations"] == ["g1", "g3"]


def test_cache_doctor_reports_a_live_lock_and_a_leftover_one(tmp_path: Path) -> None:
    """R-06: liveness is the `flock`, so a held lock and an abandoned file differ.

    The lock file outlives every holder by design - releasing closes a
    descriptor and leaves the file - so its presence on disk says nothing. Only
    asking the kernel separates a run in progress from a run that finished.
    """
    root = tmp_path / "cache"
    write_bundle(root, "live")
    write_bundle(root, "leftover")
    write_bundle(root, "never-run")
    locks = root / "_locks"
    locks.mkdir()
    (locks / "leftover.lock").touch()
    held = ExclusiveLock.take("live", locks / "live.lock")
    assert held is not None

    try:
        report = inspect(root)
    finally:
        held.release()

    bundles = bundles_by_key(report)
    assert bundles["live"]["lock"]["state"] == "live"
    assert bundles["leftover"]["lock"]["state"] == "stale"
    assert bundles["never-run"]["lock"]["state"] == "absent"
    states = {entry["path"]: entry["state"] for entry in report["locks"]}
    assert states[str(locks / "live.lock")] == "live"
    assert states[str(locks / "leftover.lock")] == "stale"


def test_cache_doctor_reports_a_prune_preview_and_deletes_nothing(tmp_path: Path) -> None:
    """R-57: what a prune *would* do, asked before it is asked to do it."""
    root = tmp_path / "cache"
    bundle = write_bundle(root, "aaa", generations=("g1", "g2", "g3"), active="g3")

    report = inspect(root, keep_generations=2)

    preview = report["prune_preview"]
    assert preview["keep_generations"] == 2
    assert [target["path"] for target in preview["candidates"]] == [str(bundle / "g1")]
    assert preview["candidates"][0]["rule"] == "retention"
    assert sorted(path.name for path in bundle.iterdir()) == [
        "_manifest.json",
        "g1",
        "g2",
        "g3",
    ]


def test_cache_doctor_reports_every_skipped_directory_with_a_reason(tmp_path: Path) -> None:
    """The requirement's point: "considered nothing" is not "deleted nothing"."""
    root = tmp_path / "cache"
    (root / "notes").mkdir(parents=True)
    (root / "_internal").mkdir()
    (root / "elsewhere").mkdir()
    (root / "link").symlink_to(root / "elsewhere", target_is_directory=True)

    report = inspect(root)

    assert report["bundle_count"] == 0
    assert report["considered"] == 4
    skipped = {entry["path"]: entry for entry in report["skipped"]}
    assert skipped[str(root / "notes")]["reason"] == "no bundle marker"
    assert skipped[str(root / "_internal")]["verdict"] == "reserved"
    assert skipped[str(root / "link")]["verdict"] == "symlink"
    assert report["skipped_count"] == len(report["skipped"])


def test_cache_doctor_never_asks_for_a_lock_file_to_be_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-57: the lock probe reads; it does not bring a lock into existence.

    Checking that a lock file exists and *then* taking it is two calls with a
    gap: the file can be unlinked in between, and `take` opens with `O_CREAT`,
    so the probe recreates what it was only supposed to look at. An inspection
    of a thousand bundle keys would leave a thousand lock files behind. The
    property is about the syscall, so that is what is asserted - a race cannot
    be observed reliably, but the flag that makes it possible can.
    """
    root = tmp_path / "cache"
    write_bundle(root, "aaa")
    (root / "_locks").mkdir()
    (root / "_locks" / "aaa.lock").touch()
    real_open = os.open
    creating: list[str] = []

    def record(path: str | Path, flags: int, mode: int = 0o777) -> int:
        if flags & os.O_CREAT:
            creating.append(str(path))
        return real_open(path, flags, mode)

    monkeypatch.setattr(bundle_store.os, "open", record)

    inspect(root)

    assert creating == []


def test_cache_doctor_creates_nothing_under_the_root_it_inspects(tmp_path: Path) -> None:
    """R-57: read-only. It reports; it never mutates - not even the root itself."""
    root = tmp_path / "never-created"

    report = inspect(root)

    assert root.exists() is False
    assert report["root_exists"] is False
    assert report["bundles"] == []
    assert report["considered"] == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not refused by directory permissions")
def test_a_root_that_cannot_be_reached_is_refused_rather_than_reported_absent(
    tmp_path: Path,
) -> None:
    """FAILS FIRST: `root_exists: false` for a root nobody was able to ask about.

    "It is not there" and "I may not look" are different answers, and only one
    of them was on offer. The guard asked `Path.is_dir()`, which answers `False`
    for a root whose parent this user cannot search - so the doctor reported an
    empty, absent root for a directory that may hold every bundle the user owns,
    and an operator reading `root_exists: false` would go and create one.

    `_file_state` already states the rule this restores: a path this process may
    not stat into is a different answer from one holding nothing, and guessing
    between them is how a permission problem became a crash or a deletion.
    """
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    unreachable = sealed / "cache"
    sealed.chmod(0o000)
    try:
        with pytest.raises(DistillError) as failure:
            inspect(unreachable)
    finally:
        sealed.chmod(0o700)

    assert failure.value.code == "E_OUTPUT_ROOT_UNREADABLE"
    assert failure.value.stage == "bundle"
    assert failure.value.details == {"root": str(unreachable), "errno": "EACCES"}


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not refused by directory permissions")
def test_the_survey_behind_the_report_refuses_the_root_the_prune_preview_does(
    tmp_path: Path,
) -> None:
    """The report is a survey *and* a prune preview, so one guard is not enough.

    Asserted at the seam because the tool cannot tell the two apart: `survey`
    runs first, so a guard present only in `plan_prune` still refuses the whole
    report, and the surface test above would pass over a survey that had gone
    back to answering "no root here" for a root it could not reach.
    """
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    unreachable = sealed / "cache"
    sealed.chmod(0o000)
    try:
        with pytest.raises(DistillError) as failure:
            BundleStore.open(unreachable).survey()
    finally:
        sealed.chmod(0o700)

    assert failure.value.code == "E_OUTPUT_ROOT_UNREADABLE"
    assert failure.value.details == {"root": str(unreachable), "errno": "EACCES"}


def test_a_root_that_is_a_regular_file_holds_no_bundles_and_is_not_created(
    tmp_path: Path,
) -> None:
    """The other control: `root_exists` asks whether an output root is there.

    A regular file at that path is not one, and `cache-doctor` cannot make it
    one - it creates nothing. So this is reported rather than refused, unlike
    the root above: the answer is known, and it is "no bundles here".
    """
    root = tmp_path / "not-a-directory"
    root.write_text("this is a file\n")

    report = inspect(root)

    assert report["root_exists"] is False
    assert report["bundles"] == []
    assert root.read_text() == "this is a file\n"


def test_cache_doctor_leaves_a_bundle_untouched(tmp_path: Path) -> None:
    """Read-only stated against a bundle, not only against an absent root."""
    root = tmp_path / "cache"
    bundle = write_bundle(root, "aaa", generations=("g1", "g2"), active="g2", staging=(".tmp.g3",))
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

    inspect(root, keep_generations=1)

    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert after == before
    assert bundles_by_key(inspect(root))["aaa"]["staging_directories"] == [".tmp.g3"]
    assert bundle.exists()


def test_cache_doctor_is_reachable_as_a_cli_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The surface is the command, not the function behind it."""
    root = tmp_path / "cache"
    write_bundle(root, "aaa", generations=("g1", "g2"), active="g2")

    main(["cache-doctor", "--output-dir", str(root), "--keep-generations", "1"])

    report = json.loads(capsys.readouterr().out)
    assert report["bundle_count"] == 1
    assert report["bundles"][0]["active_generation"] == "g2"
    assert report["prune_preview"]["candidate_count"] == 1


# --- One marker read per bundle, so a report cannot contradict itself --------


def test_a_bundle_is_described_from_the_marker_the_walk_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 6-opus): the survey read the marker a second time.

    `StoreSurvey` says what a **prune** would consider, which only holds if both
    are derived from one reading of one **bundle marker**. The orphan list went
    back to disk for its own copy, so a publish landing between the two reads
    made the report describe two different bundles at once: the generation named
    active by the first read appears in the orphan list produced by the second.

    A report that names the same generation active *and* orphaned is not a
    stale report, it is an incoherent one - and the orphan half is the half a
    user acts on.
    """
    root = tmp_path / "cache"
    write_bundle(root, "aaa", generations=("g1", "g2"), active="g1")
    real_read = bundle_store.read_marker
    reads = {"count": 0}

    def read_then_republish(directory: Path) -> object:
        verdict = real_read(directory)
        reads["count"] += 1
        if directory.name == "aaa" and reads["count"] == 1:
            # What a run publishing g2 does the instant after the walk looked.
            (directory / "_manifest.json").write_text(manifest("aaa", "g2"))
        return verdict

    monkeypatch.setattr(bundle_store, "read_marker", read_then_republish)

    report = inspect(root)

    bundle = bundles_by_key(report)["aaa"]
    assert bundle["active_generation"] == "g1"
    assert bundle["orphan_generations"] == ["g2"]
    assert bundle["active_generation"] not in bundle["orphan_generations"]


def test_a_generation_that_is_a_symlink_is_neither_reported_nor_proposed(
    tmp_path: Path,
) -> None:
    """The survey and the plan agree about what a **generation** even is.

    A symlink named `g2` is not a generation: nothing published it, and prune
    never follows one. Reported as an orphan it would name a cleanup
    `cleanup-cache` will not perform, which is the same disagreement from the
    other side.
    """
    root = tmp_path / "cache"
    bundle = write_bundle(root, "aaa", generations=("g1",), active="g1")
    outside = tmp_path / "not-a-generation"
    outside.mkdir()
    (bundle / "g2").symlink_to(outside, target_is_directory=True)

    report = inspect(root)

    described = bundles_by_key(report)["aaa"]
    assert described["generations"] == ["g1"]
    assert described["orphan_generations"] == []
    assert [target["path"] for target in report["prune_preview"]["candidates"]] == []


def test_a_directory_no_marker_can_identify_is_reported_as_needing_a_hand(
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 9-opus, R-57): unprunable, and silent about it.

    A truncated `_manifest.json` - what losing power between the entry and the
    bytes used to leave - is an `invalid` marker. Nothing can serve the bundle
    and nothing may reclaim it either: prune skips a directory whose identity it
    cannot establish, deliberately, because a file of that name in a user's own
    directory is exactly finding 1. **Expiry** does not reach it, so the disk is
    permanent.

    That is the right refusal and the wrong silence. The skip already carried
    the fact; it now carries the consequence, because "this will never be
    reclaimed unless you remove it" is the only part a user can act on.
    """
    root = tmp_path / "cache"
    damaged = root / "aaa"
    (damaged / "g1").mkdir(parents=True)
    (damaged / "g1" / "video.md").write_text("# Video\n")
    (damaged / "_manifest.json").write_text("")

    report = inspect(root, max_age_days=0.000001)

    skip = next(entry for entry in report["skipped"] if entry["path"] == str(damaged))
    assert skip["verdict"] == "invalid"
    assert "not readable JSON" in skip["reason"]
    assert "remove it" in skip["reason"]
    assert [target["path"] for target in report["prune_preview"]["candidates"]] == []
