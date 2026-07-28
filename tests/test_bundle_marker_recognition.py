"""Recognition must prove identity, and the walk must survive a directory it
cannot read.

Finding 1 was "any directory holding `g1/video.md` is a bundle". The **bundle
marker** closed it for the *published* marker: the **manifest** is parsed,
schema-checked, and its recorded **bundle key** compared to the directory name.
The **ownership marker** was left enforcing nothing at all - `read_marker` asked
only whether `_owner.json` existed. So finding 1 survived at a narrower
filename: any tool's `_owner.json` in a user's directory made that whole
directory a prune target, and `_bundle_targets` proposes the *bundle root* for a
directory that never published, which is the one target the active-generation
guard cannot spare. These tests reproduce that through the shipped CLI on a real
tree and assert the user's bytes survive.

`is_file()` follows symlinks, so both marker kinds also had to become
non-symlink regular files: a link named `_owner.json` or `_manifest.json`
pointing at a marker Distill did write recognized a directory Distill did not.

The second half is the walk itself. Every `iterdir()` and `is_file()` in it was
bare, and on Python 3.13 `Path.is_file()` did not swallow `EACCES` - so one
unreadable directory under the root ended `cache-doctor` with a
`PermissionError`. The read-only command that exists *because* the destructive
ones were unpreviewable could not report on that root at all. R-57 already
requires a reason per skip; an unreadable directory is a skip with a reason like
any other.

The walk asks `lstat` directly and catches, which is why these tests read the
same on both supported interpreters: Python 3.14 reimplemented `is_file()`,
`is_dir()` and `exists()` on top of the `os.path` predicates, which answer
`False` for *every* `OSError` rather than only for the not-there ones. Under
that behaviour the original defect would not have raised - it would have
reported the user's unreadable directory as not a directory at all.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from distill import bundle_store
from distill.bundle_store import (
    BundleStore,
    ExclusiveLock,
    PrunePolicy,
    bundle_mtime,
    lock_is_held,
    sorted_generations,
)
from distill.cli import main

USER_DATA = "# my notes\n\nnothing to do with distill\n"

OWNER_MARKER = "_owner.json"
MANIFEST_MARKER = "_manifest.json"


def manifest_document(bundle_key: str, active: str = "g1") -> dict:
    """A schema-valid **manifest** recording `bundle_key`."""
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


def user_tree(parent: Path, name: str = "my-notes") -> Path:
    """A directory of the user's own, with their data nested inside it."""
    directory = parent / name
    (directory / "deep").mkdir(parents=True)
    (directory / "deep" / "notes.md").write_text(USER_DATA)
    return directory


def doctor(root: Path, capsys: pytest.CaptureFixture[str]) -> dict:
    """`distill cache-doctor` as a user runs it."""
    main(["cache-doctor", "--output-dir", str(root)])
    return json.loads(capsys.readouterr().out)


def cleanup(root: Path, capsys: pytest.CaptureFixture[str]) -> dict:
    """`distill cleanup-cache --no-dry-run` as a user runs it: for real."""
    main(["cleanup-cache", "--output-dir", str(root), "--no-dry-run"])
    return json.loads(capsys.readouterr().out)


def skips_by_path(report: dict) -> dict[str, dict]:
    return {entry["path"]: entry for entry in report["skipped"]}


def assert_survived(directory: Path) -> None:
    """The user's data is still there, byte for byte."""
    notes = directory / "deep" / "notes.md"
    assert notes.is_file(), f"{notes} was deleted"
    assert notes.read_text() == USER_DATA


# --- The ownership marker must prove identity (finding 1) -------------------


def test_an_unparseable_owner_file_does_not_make_a_directory_prunable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding 1's exact shape at a narrower filename, through the real CLI.

    The file is not a marker Distill wrote and is not even JSON. Recognizing it
    made the whole directory - the user's data included - a prune target under
    the `orphan` rule, which proposes the bundle root itself.
    """
    root = tmp_path / "output"
    root.mkdir()
    notes = user_tree(root)
    (notes / OWNER_MARKER).write_text("not even json")

    report = doctor(root, capsys)

    assert report["bundle_count"] == 0
    assert report["prune_preview"]["candidate_count"] == 0
    assert str(notes) in skips_by_path(report)

    result = cleanup(root, capsys)

    assert result["deleted"] == []
    assert_survived(notes)


def test_an_owner_file_recording_another_bundle_key_is_foreign(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Identity, not shape: a well-formed marker still has to name this directory.

    The published marker has always refused a manifest recording somebody else's
    **bundle key**. The ownership marker enforced nothing, so a marker copied
    out of a real bundle claimed whatever directory it was dropped into.
    """
    root = tmp_path / "output"
    root.mkdir()
    notes = user_tree(root)
    (notes / OWNER_MARKER).write_text(json.dumps({"bundle_key": "somebody-else"}) + "\n")

    report = doctor(root, capsys)

    assert report["bundle_count"] == 0
    skip = skips_by_path(report)[str(notes)]
    assert skip["verdict"] == "foreign"
    assert "somebody-else" in skip["reason"]

    assert cleanup(root, capsys)["deleted"] == []
    assert_survived(notes)


def test_a_nested_users_directory_survives_at_the_deepest_walked_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The walk descends to `PRUNE_MAX_DEPTH`, so the hole was that deep too."""
    root = tmp_path / "output"
    nested = root / "a" / "b" / "c" / "d" / "e"
    nested.mkdir(parents=True)
    notes = user_tree(nested)
    (notes / OWNER_MARKER).write_text("not even json")

    report = doctor(root, capsys)

    assert report["bundle_count"] == 0
    assert report["prune_preview"]["candidate_count"] == 0

    assert cleanup(root, capsys)["deleted"] == []
    assert_survived(notes)


def test_an_owner_file_that_is_not_a_json_object_is_not_a_marker(tmp_path: Path) -> None:
    """Valid JSON is not the bar either: a marker records a **bundle key**."""
    root = tmp_path / "output"
    root.mkdir()
    (root / "my-notes").mkdir()
    (root / "my-notes" / OWNER_MARKER).write_text("[]\n")

    verdict = BundleStore.open(root).marker("my-notes")

    assert verdict.is_distill_owned is False
    assert verdict.kind == "invalid"


# --- Neither marker may be a symlink (finding 1b) ---------------------------


def test_a_symlinked_ownership_marker_is_not_a_marker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`is_file()` follows symlinks, so a link to a real marker was one."""
    root = tmp_path / "output"
    root.mkdir()
    real_marker = tmp_path / "elsewhere.json"
    real_marker.write_text(json.dumps({"bundle_key": "my-notes"}) + "\n")
    notes = user_tree(root)
    (notes / OWNER_MARKER).symlink_to(real_marker)

    report = doctor(root, capsys)

    assert report["bundle_count"] == 0
    assert report["prune_preview"]["candidate_count"] == 0

    assert cleanup(root, capsys)["deleted"] == []
    assert_survived(notes)


def test_a_symlinked_manifest_is_not_a_marker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The published marker has the same recognition problem through a link."""
    root = tmp_path / "output"
    root.mkdir()
    real_marker = tmp_path / "elsewhere.json"
    real_marker.write_text(json.dumps(manifest_document("my-notes"), indent=2) + "\n")
    notes = user_tree(root)
    (notes / MANIFEST_MARKER).symlink_to(real_marker)

    report = doctor(root, capsys)

    assert report["bundle_count"] == 0
    skip = skips_by_path(report)[str(notes)]
    assert skip["verdict"] == "invalid"
    assert MANIFEST_MARKER in skip["reason"]

    assert cleanup(root, capsys)["deleted"] == []
    assert_survived(notes)


def test_a_marker_that_is_a_directory_is_refused_as_a_marker_rather_than_read(
    tmp_path: Path,
) -> None:
    """Regular file or nothing, decided before anything tries to read it.

    A marker name occupied by something that is not a file is refused for that
    reason - not for whatever error reading it happens to produce. The rule is
    what the directory is skipped on, so the reason has to say so.
    """
    root = tmp_path / "output"
    root.mkdir()
    (root / "my-notes" / OWNER_MARKER).mkdir(parents=True)

    verdict = BundleStore.open(root).marker("my-notes")

    assert verdict.is_distill_owned is False
    assert verdict.kind == "invalid"
    assert verdict.reason == f"{OWNER_MARKER} is not a regular file"


# --- A directory the walk cannot read is a skip, not a crash (finding 2) ----


skip_as_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses the permission bits this asserts on"
)


@pytest.fixture
def unreadable(tmp_path: Path) -> Iterator[Path]:
    """A directory under the output root that this user may not stat into.

    Restored before teardown, so the temporary tree can still be removed.
    """
    root = tmp_path / "output"
    root.mkdir()
    directory = root / "sealed"
    directory.mkdir()
    (directory / "data.txt").write_text("something of the user's\n")
    os.chmod(directory, 0o000)
    yield root
    os.chmod(directory, 0o700)


@skip_as_root
def test_cache_doctor_reports_an_unreadable_directory_rather_than_failing(
    unreadable: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The read-only command must be able to report on the root it is given.

    On Python 3.13 `Path.is_file()` did not swallow `EACCES`, so `read_marker`
    raised out of the walk and the whole inspection ended as `E_INTERNAL` - on a
    root whose only problem was one directory this user cannot look inside. On
    3.14 it would have been reported as no directory at all. The assertion is
    the same either way, because the walk stopped asking a predicate that
    decides for itself what a refusal means.
    """
    report = doctor(unreadable, capsys)

    skip = skips_by_path(report)[str(unreadable / "sealed")]
    assert skip["verdict"] == "unreadable"
    assert "EACCES" in skip["reason"]


@skip_as_root
def test_cleanup_cache_reports_an_unreadable_directory_rather_than_failing(
    unreadable: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The destructive command has the same hole, and the same answer."""
    result = cleanup(unreadable, capsys)

    assert result["deleted"] == []
    assert skips_by_path(result)[str(unreadable / "sealed")]["verdict"] == "unreadable"


@skip_as_root
def test_a_directory_that_cannot_be_listed_is_reported_rather_than_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The walk's own `iterdir()` was bare too.

    A directory that can be stat'ed into but not listed carries no marker, so
    the walk descends - and `iterdir()` raises `EACCES` from inside the
    recursion rather than at the top of it.
    """
    root = tmp_path / "output"
    root.mkdir()
    opaque = root / "opaque"
    opaque.mkdir()
    os.chmod(opaque, 0o111)
    try:
        report = doctor(root, capsys)
    finally:
        os.chmod(opaque, 0o700)

    skip = skips_by_path(report)[str(opaque)]
    assert skip["verdict"] == "unlistable"
    assert "EACCES" in skip["reason"]


@skip_as_root
def test_an_entry_that_cannot_be_stat_ed_is_reported_rather_than_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A root that can be listed but not searched hands back names and nothing else.

    Listing needs read on a directory and stat'ing what is in it needs execute,
    so the two fail independently. The walk asked `is_symlink()` and `is_dir()`
    of every entry before any guard reached them.
    """
    root = tmp_path / "output"
    root.mkdir()
    (root / "something").mkdir()
    os.chmod(root, 0o444)
    try:
        report = doctor(root, capsys)
    finally:
        os.chmod(root, 0o700)

    skip = skips_by_path(report)[str(root / "something")]
    assert skip["verdict"] == "unreadable"
    assert "EACCES" in skip["reason"]


@skip_as_root
def test_a_lock_entry_that_cannot_be_stat_ed_is_left_out_rather_than_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lock nobody can ask about is a lock nobody can report.

    Unlike a marker, where "may not look" and "not there" are different
    verdicts, there is nothing to say about a lock entry that refuses to be
    stat'ed - and saying nothing must not cost the report on the rest of the
    root.
    """
    root = tmp_path / "output"
    (root / "_locks").mkdir(parents=True)
    (root / "_locks" / "aaa.lock").touch()
    os.chmod(root / "_locks", 0o444)
    try:
        report = doctor(root, capsys)
    finally:
        os.chmod(root / "_locks", 0o700)

    assert report["locks"] == []
    assert skips_by_path(report)[str(root / "_locks")]["verdict"] == "reserved"


@skip_as_root
def test_a_lock_nobody_can_open_is_unknown_and_therefore_treated_as_held(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`probe` opened the lock file and caught only "not there".

    A lock this process cannot open is not a lock it may declare free: prune's
    whole staging-liveness argument (R-06) is that the lock, and only the lock,
    separates a live run from a leftover. Unknown therefore counts as held - the
    bundle is skipped rather than reclaimed - and the inspection reports it
    instead of ending on it.
    """
    root = tmp_path / "output"
    bundle = root / "aaa"
    bundle.mkdir(parents=True)
    lock = root / "_locks" / "aaa.lock"
    lock.parent.mkdir()
    lock.touch()
    os.chmod(lock, 0o000)
    try:
        assert ExclusiveLock.probe(lock) == "unknown"
        assert lock_is_held(bundle) is True
        report = doctor(root, capsys)
    finally:
        os.chmod(lock, 0o600)

    assert [entry["state"] for entry in report["locks"]] == ["unknown"]


@skip_as_root
def test_a_generation_that_cannot_be_stat_ed_is_left_out_of_the_listing(
    tmp_path: Path,
) -> None:
    """Listing a bundle root needs read; asking what is in it needs execute.

    The layout helpers listed the root and then called `is_dir()` on what came
    back, so a bundle root that lost its execute bit raised `EACCES` from inside
    retention rather than reporting a bundle it could say nothing about.
    """
    bundle = tmp_path / "aaa"
    (bundle / "g1").mkdir(parents=True)
    os.chmod(bundle, 0o444)
    try:
        assert sorted_generations(bundle) == []
    finally:
        os.chmod(bundle, 0o700)


@skip_as_root
def test_a_target_that_cannot_be_removed_is_reported_rather_than_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removing an entry needs write on its parent, which prune never proved.

    A bundle root the user made read-only stopped `cleanup-cache` at the first
    target and left every later bundle unconsidered - the destructive command
    failing loudly at exactly the point it can no longer report what it did.
    """
    root = tmp_path / "output"
    bundle = root / "aaa"
    for name in ("g1", "g2"):
        (bundle / name).mkdir(parents=True)
        (bundle / name / "video.md").write_text("# Video\n")
    (bundle / MANIFEST_MARKER).write_text(
        json.dumps(manifest_document("aaa", active="g2"), indent=2) + "\n"
    )
    os.chmod(bundle, 0o555)
    try:
        main(["cleanup-cache", "--output-dir", str(root), "--keep-generations", "1", "--no-dry-run"])
        result = json.loads(capsys.readouterr().out)
    finally:
        os.chmod(bundle, 0o755)

    assert result["deleted"] == []
    retained = [entry for entry in result["results"] if entry["verdict"] == "skipped"]
    assert [entry["path"] for entry in retained] == [str(bundle / "g1")]
    assert "EACCES" in retained[0]["reason"]


def test_bundle_mtime_of_a_bundle_deleted_underneath_the_walk_is_not_fatal(
    tmp_path: Path,
) -> None:
    """Two prunes over one root: expiry asked the age of a bundle already gone.

    `bundle_mtime` guarded its own `stat` calls and then called
    `sorted_generations` outside that guard, so the `iterdir()` inside it raised
    `FileNotFoundError` straight out of planning.

    An age it could not learn is `None`, never `0.0`: the epoch is not "unknown",
    it is "older than everything", which is the input **expiry** deletes a whole
    bundle on.
    """
    assert bundle_mtime(tmp_path / "never-existed") is None


def test_an_unknowable_age_proposes_no_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safe answer to "how old is this?" is no target, not the oldest target.

    Every path `bundle_mtime` stats can stop answering between the marker read
    and the age question - the bundle can be removed by another prune, or a
    transient `EIO` can refuse one stat. Reporting that as the Unix epoch hands
    **expiry** an age of decades for a bundle published seconds ago.
    """
    root = tmp_path / "output"
    root.mkdir()
    bundle = root / "aaa"
    (bundle / "g1").mkdir(parents=True)
    (bundle / "g1" / "video.md").write_text("# Video\n")
    (bundle / MANIFEST_MARKER).write_text(json.dumps(manifest_document("aaa"), indent=2) + "\n")
    monkeypatch.setattr(bundle_store, "bundle_mtime", lambda _root: None)

    plan = BundleStore.open(root).plan_prune(PrunePolicy(keep_generations=3, max_age_days=1.0))

    assert [target.rule for target in plan.targets] == []
