"""Job records: the lifecycle a run's outcome is recorded through.

A **job record** is what a caller polls to learn what happened to a run it
started. These tests hold the two properties that made it useless before: that
an outcome is recorded whether the run succeeded or failed (finding 12), and
that the identifier naming the record is a bounded domain rather than something
sanitized into a shape two different identifiers can share.

The run-shaped tests here drive `pipeline.process_resolved_source`, which is the
seam every tool enters a run through, and read the record back through the
`get_job_status` tool - the surface a poller actually has. Nothing here needs
ffmpeg: the source is a stub and the generation-producing step is replaced, so
what is under test is the record, not the pipeline.

One test needs a run whose holder is gone rather than one that returned, so it
spawns a real process and has the kernel kill it. That is the only honest way to
ask the question `abandoned` answers: liveness is the `flock` a dead holder's
descriptors release, and a holder that tidied up after itself is not a dead one.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from distill import pipeline
from distill.errors import DistillError
from distill.job_store import JobOutcome, JobStore
from distill.options import DistillOptions
from distill.progress import ProgressReporter

STAGE_FAILURE = DistillError("E_STAGE_BOOM", "keyframes", "keyframe selection failed")

KILLED_MID_RUN = """
import os
import signal
import sys
from pathlib import Path

from distill.job_store import JobStore

JobStore.open(Path(sys.argv[1])).start(sys.argv[2], "process_local_video")
os.kill(os.getpid(), signal.SIGKILL)
"""
"""A run that starts a job and never gets to finish it.

Killed with `SIGKILL` from inside, so nothing in Distill and nothing in Python
runs afterwards: no `finally`, no atexit, no interpreter shutdown. What is left
on disk is exactly what a `kill -9` leaves.
"""


@dataclass
class StubSource:
    """The little a run touches before it produces a **generation**."""

    source_hash: str = "abc123"
    warnings: list[dict[str, str]] = field(default_factory=list)


def run(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    job_id: str,
    *,
    outcome: dict[str, Any] | BaseException,
    during: Callable[[], None] | None = None,
    source_hash: str = "abc123",
) -> dict[str, Any]:
    """Drive one run to `outcome`, recording whatever the run records.

    The stage that produces a **generation** is replaced rather than faked out
    with tools, because what the run computes is not what these tests are about:
    a run either reaches a result or raises, and both have to leave a record.

    `during` runs inside that stage, at the point a real run is midway through
    its work. It is the only moment from which "what does a poller see while
    this is still running?" can be asked at all.
    """

    def produce(_self: Any, _generation: Any, _heartbeat: Any) -> dict[str, Any]:
        if during is not None:
            during()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(pipeline.ProcessingRun, "_produce_generation", produce)
    root.mkdir(parents=True, exist_ok=True)
    options = DistillOptions.from_args({"output_dir": str(root), "job_id": job_id})
    return pipeline.process_resolved_source(
        StubSource(source_hash=source_hash),
        options,
        root,
        progress=ProgressReporter(emitter=lambda _event: None),
        tool="process_local_video",
    )


def status_of(root: Path, job_id: str) -> dict[str, Any]:
    """What a poller sees: the record read back through the `get_job_status` tool."""
    return pipeline.get_job_status({"output_dir": str(root), "job_id": job_id})


def test_a_failed_run_records_a_terminal_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 12): the record was only ever written on success.

    A run that raises left no record at all, so a poller could not tell a job
    that failed from a job that was never started - the two answers a caller
    most needs to distinguish.
    """
    root = tmp_path / "output"

    with pytest.raises(DistillError) as failure:
        run(monkeypatch, root, "distill-fails", outcome=STAGE_FAILURE)
    assert failure.value.code == "E_STAGE_BOOM"

    status = status_of(root, "distill-fails")
    assert status["status"] == "failed"
    # The payload shape the old direct-write test pinned: a terminal failure
    # carries the code and the message, and carries no result to be mistaken
    # for one.
    assert status["error"]["code"] == "E_STAGE_BOOM"
    assert status["error"]["message"] == "keyframe selection failed"
    assert status["error"]["stage"] == "keyframes"
    assert "result" not in status


def test_a_run_that_fails_unexpectedly_still_records_a_terminal_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-17 covers every way a run ends, not only the coded ones.

    A `DistillError` is a failure Distill anticipated. The failures that reach a
    poller with nothing to read are the ones nobody wrote a path for, so the
    record is written from whatever ended the run and an uncoded exception is
    reported as `E_INTERNAL` rather than as no record at all.
    """
    root = tmp_path / "output"

    with pytest.raises(RuntimeError):
        run(monkeypatch, root, "distill-surprise", outcome=RuntimeError("unmapped"))

    status = status_of(root, "distill-surprise")
    assert status["status"] == "failed"
    assert status["error"] == {
        "code": "E_INTERNAL",
        "stage": "internal",
        "message": "unmapped",
    }


def test_running_is_recorded_before_the_work_begins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-17: the record exists while the run is still working, not once it is done.

    Written where the result was, the record could only ever describe a run that
    got as far as producing one. A poller asking about a run genuinely in
    progress got `E_JOB_NOT_FOUND` - the same answer a mistyped identifier gets,
    so "still working" and "no such job" were one reply.
    """
    root = tmp_path / "output"
    observed: dict[str, Any] = {}

    def observe() -> None:
        observed.update(status_of(root, "distill-inflight"))

    run(
        monkeypatch,
        root,
        "distill-inflight",
        outcome={"manifest_path": "/x/manifest.json"},
        during=observe,
    )

    assert observed["status"] == "running"
    assert observed["tool"] == "process_local_video"
    assert "result" not in observed


def test_a_successful_run_records_a_terminal_completed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-17: `finish` replaces the `running` record on the success path.

    The lifecycle read end to end - start, finish, read - in place of the direct
    terminal write that let a `completed` record exist with no run behind it.
    """
    root = tmp_path / "output"
    run(monkeypatch, root, "distill-ok", outcome={"manifest_path": "/tmp/manifest.json"})

    status = status_of(root, "distill-ok")
    assert status["status"] == "completed"
    assert status["tool"] == "process_local_video"
    assert status["job_id"] == "distill-ok"
    assert status["updated_at"]
    assert status["result"]["manifest_path"] == "/tmp/manifest.json"
    assert "error" not in status


def test_a_record_left_running_is_reported_abandoned_not_completed(tmp_path: Path) -> None:
    """R-17: a `running` record whose holder is gone is not a job in progress.

    Nobody is left to write the terminal status for a run the kernel killed, so
    `running` is what stays on disk. Reporting that as `running` tells a poller
    to keep waiting for a process that no longer exists; reporting it as
    `completed` would be worse still. The verdict comes from the lock the dead
    holder's descriptors released, not from how old the record looks.
    """
    root = tmp_path / "output"
    root.mkdir(parents=True)

    killed = subprocess.run(
        [sys.executable, "-c", KILLED_MID_RUN, str(root), "distill-killed"],
        capture_output=True,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr.decode()

    store = JobStore.open(root)
    stored = json.loads(store.record_path("distill-killed").read_text())
    assert stored["status"] == "running", "the dead run had no chance to write anything else"

    record = store.read("distill-killed")
    assert record is not None
    assert record.status == "abandoned"


def test_a_second_live_run_cannot_start_under_a_job_id_already_held(tmp_path: Path) -> None:
    """Two live runs under one identifier are one record describing both.

    R-18 stops identifiers colliding by being mapped together; agreeing on the
    same identifier reaches the same place, and the second run would overwrite
    the first's `running` record and then its outcome. `flock` is per open file
    description, so the refusal holds between two stores in one process exactly
    as it does between two processes.
    """
    root = tmp_path / "output"
    JobStore.open(root).start("distill-held", "process_local_video")

    with pytest.raises(DistillError) as raised:
        JobStore.open(root).start("distill-held", "process_local_video")

    assert raised.value.code == "E_JOB_RUNNING"


def test_a_job_id_differing_only_in_case_is_out_of_the_domain(tmp_path: Path) -> None:
    """R-18: an identifier is a filename, and a filename is not always case-sensitive.

    macOS and Windows fold case, so `Distill-Job` and `distill-job` would name
    one record between them - the collision sanitizing produced, reached through
    the filesystem instead of through a character mapping. The domain cannot
    contain both, and the one outside it is refused rather than folded, because
    folding is the many-to-one mapping.
    """
    with pytest.raises(DistillError) as raised:
        JobStore.open(tmp_path / "output").start("Distill-Job", "process_local_video")

    assert raised.value.code == "E_BAD_JOB_ID"


def test_a_run_that_finishes_between_the_read_and_the_lock_probe_is_not_abandoned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A holder that is gone finished as often as it died, and the two differ.

    `read` sees `running`, then asks whether a holder is live. A run finishing
    in between releases its lock, and reporting the record read *before* that as
    `abandoned` tells a poller its completed job was killed. `finish` writes the
    terminal record before releasing, so the record itself settles the question -
    but only if it is read again.
    """
    root = tmp_path / "output"
    owner = JobStore.open(root)
    owner.start("distill-finishing", "process_local_video")

    def finish_then_report_the_holder_gone(_self: Any, job_id: str) -> bool:
        owner.finish(job_id, JobOutcome.success({"manifest_path": "/tmp/manifest.json"}))
        return False

    monkeypatch.setattr(JobStore, "_holder_is_live", finish_then_report_the_holder_gone)

    record = JobStore.open(root).read("distill-finishing")
    assert record is not None
    assert record.status == "completed"
    assert record.result == {"manifest_path": "/tmp/manifest.json"}


def test_a_job_restarted_between_the_read_and_the_probe_is_not_abandoned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-reading alone judges a new record by an old probe.

    Identifiers are reused, so the run that finished and the run that started
    after it are two runs under one name. Reading again after finding no holder
    picks up the *new* run's `running` record and hands it to a probe taken
    before that run existed - reporting a job that is live at this instant as
    killed. The verdict has to rest on a probe that came after the record.
    """
    root = tmp_path / "output"
    finishing = JobStore.open(root)
    finishing.start("distill-relay", "process_local_video")
    successor = JobStore.open(root)
    genuinely_live = JobStore._holder_is_live
    handed_over: list[str] = []

    def hand_the_job_over(self: Any, job_id: str) -> bool:
        if not handed_over:
            handed_over.append(job_id)
            finishing.finish(job_id, JobOutcome.success({"manifest_path": "/first.json"}))
            successor.start(job_id, "process_local_video")
            return False
        return bool(genuinely_live(self, job_id))

    monkeypatch.setattr(JobStore, "_holder_is_live", hand_the_job_over)

    record = JobStore.open(root).read("distill-relay")
    assert record is not None
    assert record.status == "running"


def test_a_failed_record_write_does_not_leave_the_job_id_locked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lock outlives the run that took it only for as long as the process does.

    Distill's tools run inside a long-lived session, so a hold kept past the run
    it covers is an identifier nothing can ever start again: one unwritable
    record, then `E_JOB_RUNNING` for that job for the life of the process. The
    hold is given up whether the record could be written or not, at `start` and
    at `finish` alike.
    """
    root = tmp_path / "output"
    unwritable = DistillError("E_DISK_FULL", "job", "cannot write the record")

    def refuse(_self: Any, _record: Any) -> Any:
        raise unwritable

    monkeypatch.setattr(JobStore, "_write", refuse)
    with pytest.raises(DistillError) as at_start:
        JobStore.open(root).start("distill-unwritable", "process_local_video")
    assert at_start.value.code == "E_DISK_FULL"
    monkeypatch.undo()

    started = JobStore.open(root)
    assert started.start("distill-unwritable", "process_local_video").status == "running"

    monkeypatch.setattr(JobStore, "_write", refuse)
    with pytest.raises(DistillError):
        started.finish("distill-unwritable", JobOutcome.success({"manifest_path": "/x.json"}))
    monkeypatch.undo()

    assert JobStore.open(root).start("distill-unwritable", "process_local_video").status == "running"


def test_a_job_outcome_must_be_terminal_and_carry_only_its_own_evidence() -> None:
    """The carrier's guarantee is checked at construction, not merely annotated.

    `finish` releases the lock as it writes, so a `running` outcome would leave a
    record no holder backs and nobody will replace - a run that succeeded,
    reported `abandoned`. A `completed` carrying an error, or a `failed`
    carrying a result, is the other half: a record that contradicts itself and
    leaves a reader to pick.
    """
    with pytest.raises(DistillError) as not_terminal:
        JobOutcome(status="running")  # ty: ignore[invalid-argument-type]
    assert not_terminal.value.code == "E_BAD_JOB_OUTCOME"

    with pytest.raises(DistillError):
        JobOutcome(status="completed", error={"code": "E_INTERNAL"})

    with pytest.raises(DistillError):
        JobOutcome(status="failed", result={"manifest_path": "/x.json"})


def test_reading_an_unknown_job_id_reports_no_record(tmp_path: Path) -> None:
    """An identifier with no record is `None`, not an error and not a guess."""
    assert JobStore.open(tmp_path / "output").read("absent") is None


def test_the_record_directory_is_created_by_writing_and_not_by_reading(tmp_path: Path) -> None:
    """The record directory appears on demand, and asking a question is not demand.

    Creating it in order to answer "is there a record?" has already answered
    "no" by creating it, and leaves a directory behind under a root a caller may
    only have been inspecting. Writing creates it, repeatedly and idempotently:
    the second job under a root finds the directory the first one made.
    """
    root = tmp_path / "output"
    store = JobStore.open(root)

    assert store.read("distill-one") is None
    assert not (root / "_jobs").exists()

    store.start("distill-one", "process_local_video")
    store.start("distill-two", "process_local_video")

    assert store.record_path("distill-one").is_file()
    assert store.record_path("distill-two").is_file()


def test_a_reused_job_id_does_not_keep_the_previous_completed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 12): a failure left the previous success standing.

    Job identifiers are supplied by the caller and are reused - `distill-job`
    twice, or the same batch item re-run. With nothing written on the failure
    path, the record from the earlier run stayed on disk saying `completed`,
    carrying the earlier run's result: a poller asking about the run that just
    failed is told it succeeded, and handed a manifest path to prove it.
    """
    root = tmp_path / "output"
    run(monkeypatch, root, "distill-reused", outcome={"manifest_path": "/first/manifest.json"})
    assert status_of(root, "distill-reused")["status"] == "completed"

    with pytest.raises(DistillError):
        run(monkeypatch, root, "distill-reused", outcome=STAGE_FAILURE)

    status = status_of(root, "distill-reused")
    assert status["status"] == "failed"
    assert "result" not in status


def test_an_out_of_domain_job_id_is_rejected_rather_than_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-18: an identifier outside the domain is refused, not mapped into it.

    `job/../escape` was mapped character by character onto `job____escape` and
    the run carried on under a name nobody asked for.
    """
    root = tmp_path / "output"

    with pytest.raises(DistillError) as failure:
        run(monkeypatch, root, "job/../escape", outcome={"manifest_path": "/x/manifest.json"})

    assert failure.value.code == "E_BAD_JOB_ID"


def test_two_distinct_job_ids_cannot_collide_onto_one_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-18 regression: sanitizing is what let two identifiers share one record.

    `job/../escape` and `job____escape` are different jobs. Sanitizing mapped
    both onto `job____escape.json`, so the second run overwrote the first run's
    outcome and the first job's caller was told about a run it never started.
    """
    root = tmp_path / "output"
    run(monkeypatch, root, "job____escape", outcome={"manifest_path": "/first/manifest.json"})

    with pytest.raises(DistillError):
        run(monkeypatch, root, "job/../escape", outcome={"manifest_path": "/second/manifest.json"})

    status = status_of(root, "job____escape")
    assert status["status"] == "completed"
    assert status["result"]["manifest_path"] == "/first/manifest.json"
