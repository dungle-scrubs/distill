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
from distill.bundle_store import ExclusiveLock
from distill.cli import main
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
    (root / "owned").mkdir()
    (root / "owned" / "_owner.json").write_text("{}\n")

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
