"""Tests for the run lock: what stops two runs from producing one bundle.

The seam under test is `BundleStore.begin` - the single point at which a run
takes exclusive hold of a **bundle key**. Finding 6 is what happens when nothing
owns it: two runs of one bundle key both opened `.tmp.g1` and wrote **stage
results** over each other. Finding 11 is what happens when the lock is a file
whose content is refreshed: a heartbeat nobody refreshed made every live holder
look abandoned after 180 s.

Exclusion here is the kernel's, so the tests that matter use real processes -
`flock` is per open file description, and threads in one process share nothing
that would make a thread-level test say anything about two runs. The wait
budgets (5 s for a batch item, 300 s for a single-source run) are exercised
through an injected clock, so a budget is asserted exactly rather than waited
out.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_bundle_store import published_manifest, write_bundle

from distill import bundle_store
from distill.bundle_store import (
    BATCH_ITEM_LOCK_WAIT_SEC,
    LOCK_DIR_NAME,
    OWNERSHIP_MARKER_NAME,
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleRun,
    BundleSnapshot,
    BundleStore,
)
from distill.errors import DistillError

BUNDLE_KEY = "b0a1c2d3"


class FakeClock:
    """A monotonic clock that advances only when the waiter sleeps.

    A wait budget is a promise about how long a run waits, not about how long a
    test takes. Driving the clock from the sleep the poll loop performs makes
    the elapsed budget exact and the test instant.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def lock_path(root: Path, bundle_key: str = BUNDLE_KEY) -> Path:
    return root / LOCK_DIR_NAME / f"{bundle_key}.lock"


def lock_is_held(path: Path) -> bool:
    """Whether somebody else holds the lock at `path`.

    `flock` is per open file description, so a fresh descriptor opened here is
    refused even when the holder lives in this process - which is what lets a
    single-process test observe a real hold.
    """
    if not path.exists():
        return False
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def hold_the_lock(root: Path, bundle_key: str = BUNDLE_KEY) -> int:
    """Take the bundle lock the way another run holds it, returning its descriptor."""
    path = lock_path(root, bundle_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def begin_run(
    store: BundleStore,
    *,
    wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
    resume: bool = True,
) -> BundleRun:
    """`begin`, asserting it opened a run rather than answering with a cache hit.

    `begin` returns a **bundle snapshot** when an **active generation** appeared
    under the lock, so a test that means "this run staged" has to say so.
    """
    outcome = store.begin(BUNDLE_KEY, wait_sec=wait_sec, resume=resume)
    assert isinstance(outcome, BundleRun)
    return outcome


HOLDER = '''
"""A run that takes the bundle lock and then only holds it."""

import sys
import time
from pathlib import Path

from distill.bundle_store import BundleStore

root, bundle_key = sys.argv[1:]
run = BundleStore.open(Path(root)).begin(bundle_key)
print(run.paths.generation, flush=True)
# Nothing here ever releases: giving the lock up is what ending this process
# has to accomplish on its own.
while True:
    time.sleep(3600)
'''


def start_a_holding_run(tmp_path: Path, root: Path) -> subprocess.Popen[str]:
    """Start a real second run, returning once it holds the lock and has staged."""
    script = tmp_path / "holder.py"
    script.write_text(HOLDER)
    holder = subprocess.Popen(
        [sys.executable, str(script), str(root), BUNDLE_KEY],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    staged = holder.stdout.readline().strip()
    assert staged.endswith(".tmp.g1"), holder.stderr.read() if holder.stderr else staged
    return holder


def test_two_processes_on_one_bundle_key_do_not_both_stage(tmp_path: Path) -> None:
    """FAILS FIRST (finding 6): one **bundle key**, one staging directory, two runs.

    `.tmp.g1` is derived from what is on disk, so two runs of one bundle key
    both compute it, both open it, and both write **stage results** into it -
    each reading back the other's. The second run here is a real process, since
    that is the only arrangement in which the kernel's answer means anything.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    holder = start_a_holding_run(tmp_path, root)

    try:
        with pytest.raises(DistillError) as exc:
            store.begin(BUNDLE_KEY, wait_sec=0.0)
    finally:
        holder.kill()

    assert exc.value.code == "E_LOCKED"
    # One staging directory, and it is the holder's: the denied run staged
    # nothing rather than joining the holder in `.tmp.g1`.
    bundle = root / BUNDLE_KEY
    assert sorted(path.name for path in bundle.iterdir()) == [".tmp.g1", OWNERSHIP_MARKER_NAME]
    assert holder.wait() != 0


def test_the_lock_is_held_for_the_run_and_released_when_the_run_ends(
    tmp_path: Path,
) -> None:
    """The hold covers the run, not the moment of taking it.

    A lock released once staging exists would leave every later stage - the
    download, the vision pass, the publish - running against a bundle key a
    second run is free to take.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    run = begin_run(store)

    assert lock_is_held(lock_path(root)) is True
    run.release()
    assert lock_is_held(lock_path(root)) is False


def test_a_lock_whose_holder_was_killed_is_reacquirable_at_once(tmp_path: Path) -> None:
    """The kernel releases the lock; nothing in Distill has to notice the death.

    The holder is `SIGKILL`ed and deliberately not reaped, so it is a process
    that released nothing, cannot release anything, and whose pid the kernel
    still answers for. That is also what a holder whose pid was reused looks
    like. Asking the kernel for the lock is the only question with a true
    answer, and it answers immediately - no staleness window to wait out.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    holder = start_a_holding_run(tmp_path, root)
    holder.kill()
    # Wait for the death without collecting it: WNOWAIT leaves the pid in the
    # table, so this is a dead holder rather than a dying one, with no sleep.
    os.waitid(os.P_PID, holder.pid, os.WEXITED | os.WNOWAIT)

    run = begin_run(store, wait_sec=0.0)

    run.release()
    assert holder.wait() != 0


def test_a_lock_a_live_run_holds_is_not_stealable_however_old_it_is(
    tmp_path: Path,
) -> None:
    """Finding 11: age is not evidence, and every real run gets old.

    The heartbeat scheme this replaces read a lock file older than 180 s as
    abandoned, and a run that reaches the vision pass has held its lock for
    longer than that. The lock file is aged an hour on disk and the clock moved
    an hour forward: a held lock is held, and neither may matter.
    """
    root = tmp_path / "output"
    root.mkdir()
    clock = FakeClock()
    store = BundleStore.open(root, clock=clock.monotonic, sleep=clock.sleep)
    held = hold_the_lock(root)
    os.utime(lock_path(root), (0, 0))
    clock.now = 3600.0

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY, wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)

    assert exc.value.code == "E_LOCKED"
    assert lock_is_held(lock_path(root)) is True
    os.close(held)


def refuse_flock(monkeypatch: pytest.MonkeyPatch, code: int = errno.ENOLCK) -> None:
    """Make this filesystem one that cannot grant `flock`."""

    def refuse(_fd: int, _operation: int) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(bundle_store.fcntl, "flock", refuse)


def test_a_filesystem_that_cannot_lock_fails_before_touching_the_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (R-09): staging happened first and asked about locking never.

    `ENOLCK` is what a filesystem with no lock manager answers - some network
    mounts, and the reason R-09 exists. Distill cannot tell one run from two
    there, so it says so and stops. Stopping *after* creating the bundle
    directory and its staging directory would leave the same shared `.tmp.g1`
    finding 6 is about, with no lock to have prevented it.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    refuse_flock(monkeypatch)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY)

    assert exc.value.code == "E_LOCK_UNSUPPORTED"
    assert exc.value.details["errno"] == "ENOLCK"
    assert (root / BUNDLE_KEY).exists() is False


def test_a_probe_that_fails_mutates_nothing_in_an_existing_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No mutation before the probe succeeds, including into a bundle that exists.

    A bundle already on disk is the case where "mutates nothing" is observable
    rather than vacuous: its **active generation**, its manifest and the
    contents of its directory are exactly what they were.
    """
    root = tmp_path / "output"
    root.mkdir()
    bundle = write_bundle(root, BUNDLE_KEY)
    before = sorted(path.name for path in bundle.iterdir())
    manifest_before = (bundle / "_manifest.json").read_bytes()
    store = BundleStore.open(root)
    refuse_flock(monkeypatch)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY)

    assert exc.value.code == "E_LOCK_UNSUPPORTED"
    assert sorted(path.name for path in bundle.iterdir()) == before
    assert (bundle / "_manifest.json").read_bytes() == manifest_before


def test_a_waiter_that_arrives_after_the_winner_published_gets_the_snapshot(
    tmp_path: Path,
) -> None:
    """RV-1: the winner's result, not a second run of the same work.

    The bundle is published, and the lock released, from inside the waiter's
    own sleep - so the waiter's re-check happens strictly after a publish it
    could not have seen when it started waiting. A `begin` that decided cache
    hit before taking the lock would hand back a run and redo the work.
    """
    root = tmp_path / "output"
    root.mkdir()
    held = hold_the_lock(root)
    published: list[Path] = []

    def publish_then_release(_seconds: float) -> None:
        if published:
            return
        published.append(write_bundle(root, BUNDLE_KEY))
        os.close(held)

    store = BundleStore.open(root, sleep=publish_then_release)

    outcome = store.begin(BUNDLE_KEY, wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)

    assert isinstance(outcome, BundleSnapshot)
    assert outcome.generation == published[0] / "g1"
    # The waiter took the lock to answer the question and gave it straight back:
    # there is no run to release it later.
    assert lock_is_held(lock_path(root)) is False
    assert list((root / BUNDLE_KEY).glob(".tmp.*")) == []


def test_a_batch_item_waits_five_seconds_and_then_fails_locked(tmp_path: Path) -> None:
    """D-044: a playlist behind a 40-minute video fails fast and moves on.

    `continue_on_error` already defaults true, so the item's `E_LOCKED` ends the
    item and not the batch; re-running picks it up as a cache hit.
    """
    root = tmp_path / "output"
    root.mkdir()
    clock = FakeClock()
    store = BundleStore.open(root, clock=clock.monotonic, sleep=clock.sleep)
    held = hold_the_lock(root)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY, wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)

    assert exc.value.code == "E_LOCKED"
    assert BATCH_ITEM_LOCK_WAIT_SEC == 5.0
    assert clock.now == pytest.approx(5.0)
    assert exc.value.details["waited_sec"] == pytest.approx(5.0)
    assert exc.value.details["bundle_key"] == BUNDLE_KEY
    os.close(held)


def test_a_single_source_run_waits_five_minutes_and_then_fails_locked(
    tmp_path: Path,
) -> None:
    """D-044: one video is worth waiting for; it is the run the user is watching."""
    root = tmp_path / "output"
    root.mkdir()
    clock = FakeClock()
    store = BundleStore.open(root, clock=clock.monotonic, sleep=clock.sleep)
    held = hold_the_lock(root)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY)

    assert exc.value.code == "E_LOCKED"
    assert SINGLE_SOURCE_LOCK_WAIT_SEC == 300.0
    assert clock.now == pytest.approx(300.0)
    assert exc.value.details["waited_sec"] == pytest.approx(300.0)
    os.close(held)


def test_begin_writes_the_ownership_marker_before_any_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (RV-9): a first run that dies leaves a directory nobody owns.

    Staging is made to fail, which is what any crash between taking the lock and
    publishing looks like from the directory's point of view. What is left must
    already be identifiable as Distill's - otherwise the directory is
    simultaneously "not a bundle" (no manifest) and holding an abandoned staging
    directory, which **prune** may then never reclaim.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    def crash(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("staging died")

    monkeypatch.setattr(bundle_store, "stage_paths", crash)

    with pytest.raises(RuntimeError):
        store.begin(BUNDLE_KEY)

    verdict = store.marker(BUNDLE_KEY)
    assert verdict.kind == "owned"
    assert verdict.is_distill_owned is True
    assert verdict.is_bundle is False
    assert (root / BUNDLE_KEY / OWNERSHIP_MARKER_NAME).is_file()
    # The lock is not held by a run that never started.
    assert lock_is_held(lock_path(root)) is False


def lock_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == bundle_store.LOGGER.name and record.message.startswith("{")
    ]


def test_lock_events_report_acquired_waited_denied_and_unsupported(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A run that waited, a run that was denied, and a filesystem that cannot lock.

    Contention is invisible from the outside: a run that finished in twenty
    minutes because it spent ten of them waiting looks exactly like a slow run.
    The wait duration is what distinguishes them.
    """
    root = tmp_path / "output"
    root.mkdir()
    clock = FakeClock()
    store = BundleStore.open(root, clock=clock.monotonic, sleep=clock.sleep)
    caplog.set_level(logging.DEBUG, logger=bundle_store.LOGGER.name)

    run = begin_run(store)
    run.release()

    held = hold_the_lock(root)
    with pytest.raises(DistillError):
        store.begin(BUNDLE_KEY, wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)
    os.close(held)

    released_at = clock.now
    releasing = {"done": False}

    def release_after_one_poll(seconds: float) -> None:
        clock.sleep(seconds)
        if not releasing["done"]:
            releasing["done"] = True
            os.close(second_hold)

    second_hold = hold_the_lock(root)
    waiting_store = BundleStore.open(root, clock=clock.monotonic, sleep=release_after_one_poll)
    waited_run = begin_run(waiting_store, wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)
    waited_run.release()

    refuse_flock(monkeypatch)
    with pytest.raises(DistillError):
        store.begin(BUNDLE_KEY)

    events = lock_events(caplog)
    by_event = {event["event"]: event for event in events}
    assert set(by_event) >= {"lock_acquired", "lock_waited", "lock_denied", "lock_unsupported"}
    assert by_event["lock_denied"]["detail"]["waited_sec"] == pytest.approx(5.0)
    assert by_event["lock_waited"]["detail"]["waited_sec"] == pytest.approx(
        clock.now - released_at
    )
    assert by_event["lock_unsupported"]["detail"]["errno"] == "ENOLCK"
    assert by_event["lock_acquired"]["detail"]["bundle_key"] == BUNDLE_KEY


def test_a_published_bundle_short_circuits_begin_without_staging(tmp_path: Path) -> None:
    """R-08: the re-check under the lock is a cache hit, not a second generation."""
    root = tmp_path / "output"
    root.mkdir()
    write_bundle(root, BUNDLE_KEY)
    store = BundleStore.open(root)

    outcome = store.begin(BUNDLE_KEY)

    assert isinstance(outcome, BundleSnapshot)
    assert list((root / BUNDLE_KEY).glob(".tmp.*")) == []


def test_a_manifest_naming_a_missing_generation_is_not_a_snapshot(tmp_path: Path) -> None:
    """A manifest is a promise; `begin` needs evidence, so this run rebuilds.

    Coupled to R-04 deliberately: the re-check under the lock is `load_active`,
    so a bundle whose active generation was deleted stages instead of handing
    back a snapshot naming nothing.
    """
    root = tmp_path / "output"
    root.mkdir()
    bundle = root / BUNDLE_KEY
    bundle.mkdir()
    (bundle / "_manifest.json").write_text(
        json.dumps(published_manifest(identity_field="bundle_key", identity=BUNDLE_KEY))
    )
    store = BundleStore.open(root)

    run = begin_run(store)

    assert run.paths.generation.name == ".tmp.g1"
    run.release()


def test_a_bundle_key_that_escapes_the_output_root_takes_no_lock(tmp_path: Path) -> None:
    """The lock path is derived from the bundle key, so it is validated too.

    `../evil` would otherwise name a lock file outside the output root and, with
    it, a bundle directory outside the output root.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    with pytest.raises(DistillError) as exc:
        store.begin("../evil")

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    assert list(tmp_path.glob("**/*.lock")) == []


def test_releasing_a_run_twice_frees_nobody_a_second_time(tmp_path: Path) -> None:
    """Release is idempotent: the failure paths that release overlap with cleanup."""
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    run.release()

    other = begin_run(store, wait_sec=0.0)
    run.release()

    assert lock_is_held(lock_path(root)) is True
    other.release()


def test_the_lock_survives_a_run_that_was_never_released_by_this_process(
    tmp_path: Path,
) -> None:
    """A second `begin` in one process is refused, because `flock` is per description.

    This is what makes the single-process assertions above mean what they say:
    the kernel does not grant the same lock twice to one process either.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY, wait_sec=0.0)

    assert exc.value.code == "E_LOCKED"
    run.release()


def test_a_run_releases_its_lock_when_used_as_a_context_manager(tmp_path: Path) -> None:
    """The hold ends with the block, including when the block raises."""
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)

    run = begin_run(store)
    with pytest.raises(RuntimeError), run:
        assert lock_is_held(lock_path(root)) is True
        raise RuntimeError("run failed")

    assert lock_is_held(lock_path(root)) is False


def test_resume_keeps_the_staging_directory_a_stopped_run_left(tmp_path: Path) -> None:
    """Resume is a real feature (D-046): the stage results survive `begin`."""
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    first = begin_run(store)
    (first.paths.generation / "_ocr.json").write_text("{}")
    first.release()

    resumed = begin_run(store)

    assert (resumed.paths.generation / "_ocr.json").is_file()
    resumed.release()

    fresh = begin_run(store, resume=False)
    assert (fresh.paths.generation / "_ocr.json").exists() is False
    fresh.release()


def test_the_wait_is_not_spent_on_a_lock_that_was_never_contended(tmp_path: Path) -> None:
    """An uncontended `begin` sleeps not at all - the budget is a ceiling."""
    root = tmp_path / "output"
    root.mkdir()
    clock = FakeClock()
    store = BundleStore.open(root, clock=clock.monotonic, sleep=clock.sleep)

    run = begin_run(store)

    assert clock.now == 0.0
    assert run.waited_sec == 0.0
    run.release()


def test_a_run_takes_the_bundle_lock_for_the_budget_its_caller_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D-044 reaches a real caller: the pipeline stages under the lock, on a budget.

    Until M3.6 the pipeline staged without taking the bundle lock at all, so
    both budgets were unreachable from any command a user runs and the store's
    own tests were the only thing asserting them. A single-source run waits five
    minutes for the run a user is watching; one item of a batch waits five
    seconds and lets the batch move on.
    """
    from distill import pipeline
    from distill.options import DistillOptions
    from distill.progress import ProgressReporter

    budgets: list[float] = []

    def record(_self: object, _key: str, *, wait_sec: float, **_rest: object) -> None:
        budgets.append(wait_sec)
        raise DistillError("E_STOP", "bundle", "far enough")

    monkeypatch.setattr(pipeline.BundleStore, "begin", record)

    class StubSource:
        source_hash = BUNDLE_KEY
        warnings: list[dict[str, str]] = []

    root = tmp_path / "output"
    root.mkdir()
    options = DistillOptions.from_args({"output_dir": str(root), "job_id": "j"})
    progress = ProgressReporter(emitter=lambda _event: None)

    # The default is the single-source budget, so a caller that says nothing
    # waits for the run a user is watching.
    with pytest.raises(DistillError):
        pipeline.process_resolved_source(StubSource(), options, root, progress=progress)
    with pytest.raises(DistillError):
        pipeline.process_resolved_source(
            StubSource(),
            options,
            root,
            progress=progress,
            lock_wait_sec=BATCH_ITEM_LOCK_WAIT_SEC,
        )

    assert budgets == [SINGLE_SOURCE_LOCK_WAIT_SEC, BATCH_ITEM_LOCK_WAIT_SEC]


def test_a_run_that_fails_mid_stage_abandons_its_hold_and_says_why(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`BundleRun.abandon` had no caller until the pipeline entered through `begin`.

    A run that raises leaves a bundle that did not change, which is otherwise
    indistinguishable from a run that never happened - so the reason is the
    record. The previous **active generation** is untouched and the **staging
    directory** stays for the next run to **resume** from; only the lock goes.
    """
    from distill import pipeline
    from distill.options import DistillOptions
    from distill.progress import ProgressReporter

    def explode(_self: object, _run: object, _heartbeat: object) -> dict[str, Any]:
        raise DistillError("E_STAGE_BOOM", "keyframes", "keyframe selection failed")

    monkeypatch.setattr(pipeline.ProcessingRun, "_produce_generation", explode)

    class StubSource:
        source_hash = BUNDLE_KEY
        warnings: list[dict[str, str]] = []

    root = tmp_path / "output"
    root.mkdir()
    caplog.set_level(logging.DEBUG, logger=bundle_store.LOGGER.name)

    with pytest.raises(DistillError):
        pipeline.process_resolved_source(
            StubSource(),
            DistillOptions.from_args({"output_dir": str(root), "job_id": "j"}),
            root,
            progress=ProgressReporter(emitter=lambda _event: None),
        )

    abandoned = [event for event in lock_events(caplog) if event["event"] == "run_abandoned"]
    assert len(abandoned) == 1
    assert abandoned[0]["detail"]["bundle_key"] == BUNDLE_KEY
    assert "E_STAGE_BOOM" in abandoned[0]["detail"]["reason"]
    assert (root / BUNDLE_KEY / ".tmp.g1").is_dir()
    assert lock_is_held(lock_path(root)) is False


def test_the_same_budget_reaches_the_lock_a_run_takes_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 4-opus, D-044): the budget stopped at the bundle key.

    A run takes two locks: the **acquisition lease** on its **lock key** and the
    run lock on its **bundle key**. Acquisition is the one it reaches first, and
    the one two runs of *the same video* contend for - they share a lock key
    whatever their options. A budget that reached only the second lock was
    therefore never spent on the case D-044 was written for: the second run was
    denied at acquisition before `begin` was called at all.
    """
    from distill import pipeline
    from distill.options import DistillOptions
    from distill.progress import ProgressReporter

    budgets: list[float] = []

    def record(
        _source_type: str,
        _value: str,
        _options: object,
        *,
        progress: object = None,
        lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
        **_rest: object,
    ) -> None:
        budgets.append(lock_wait_sec)
        raise DistillError("E_STOP", "youtube", "far enough")

    monkeypatch.setattr(pipeline, "resolve_source_for_processing", record)

    root = tmp_path / "output"
    root.mkdir()
    options = DistillOptions.from_args({"output_dir": str(root), "job_id": "j"})
    progress = ProgressReporter(emitter=lambda _event: None)
    for budget in (SINGLE_SOURCE_LOCK_WAIT_SEC, BATCH_ITEM_LOCK_WAIT_SEC):
        with pytest.raises(DistillError):
            pipeline.acquire_and_process(
                "youtube",
                "https://youtu.be/abc123",
                options,
                root,
                progress=progress,
                tool="process_youtube_video",
                lock_wait_sec=budget,
            )

    assert budgets == [SINGLE_SOURCE_LOCK_WAIT_SEC, BATCH_ITEM_LOCK_WAIT_SEC]


# --- Every way the lock can be refused is R-09's answer (finding 8-opus) ----


def test_a_lock_file_that_cannot_be_opened_is_a_capability_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 8-opus, R-09): the refusal escaped as a raw `OSError`.

    Taking the lock is two syscalls, and only the second was inside the guard:
    a read-only mount or a directory this process may not write refuses the
    `open`, never reaching `flock` at all. That is the same fact R-09 is about -
    this filesystem cannot give Distill the exclusion it asked for - and a
    caller told `E_INTERNAL` learns nothing it can act on, where
    `E_LOCK_UNSUPPORTED` names the mount and the errno.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    real_open = os.open

    def refuse_lock_files(path: Any, flags: int, *rest: Any, **kwargs: Any) -> int:
        if str(path).endswith(".lock"):
            raise OSError(errno.EROFS, os.strerror(errno.EROFS))
        return real_open(path, flags, *rest, **kwargs)

    monkeypatch.setattr(bundle_store.os, "open", refuse_lock_files)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY)

    assert exc.value.code == "E_LOCK_UNSUPPORTED"
    assert exc.value.details["errno"] == "EROFS"
    assert (root / BUNDLE_KEY).exists() is False


def test_a_filesystem_answering_eacces_is_not_read_as_a_lock_somebody_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FAILS FIRST (finding 8-opus, R-09): a fatal answer was read as contention.

    `flock(2)` reports a lock somebody else holds as `EWOULDBLOCK`/`EAGAIN` and
    never as `EACCES`. Counting `EACCES` as "held" therefore misreads a
    filesystem that cannot lock as a busy one: the run spends its whole budget
    polling a lock nobody holds and is then told another run has the bundle,
    which is the one report that stops a user looking at the mount.
    """
    root = tmp_path / "output"
    root.mkdir()
    clock = FakeClock()
    store = BundleStore.open(root, clock=clock.monotonic, sleep=clock.sleep)
    refuse_flock(monkeypatch, code=errno.EACCES)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY)

    assert exc.value.code == "E_LOCK_UNSUPPORTED"
    assert exc.value.details["errno"] == "EACCES"
    assert clock.now == 0.0, "a capability failure is not something to wait out"


def test_a_filesystem_that_cannot_lock_leaves_no_lock_directory_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 6-codex, R-09): "before mutating anything" was not.

    The refusal was raised by the `flock` at the end of a sequence that had
    already created `_locks` and an empty `<bundle key>.lock` inside it. Both
    are Distill's files in a user's output root, written on a filesystem that
    just said it cannot support what they are for - and the existing test only
    watched the bundle directory, so it could not see them.

    The capability is therefore asked of a directory that already exists, with
    a shared lock taken and dropped at once: the question is whether this
    filesystem grants `flock` at all, and asking it creates nothing.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    refuse_flock(monkeypatch)

    with pytest.raises(DistillError) as exc:
        store.begin(BUNDLE_KEY)

    assert exc.value.code == "E_LOCK_UNSUPPORTED"
    assert sorted(entry.name for entry in root.iterdir()) == []
