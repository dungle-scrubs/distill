"""Job records: the lifecycle a run's outcome is recorded through.

A **job record** is what a caller polls to learn what happened to a run it
started. These tests hold the two properties that made it useless before: that
an outcome is recorded whether the run succeeded or failed (finding 12), and
that the identifier naming the record is a bounded domain rather than something
sanitized into a shape two different identifiers can share.

The run-shaped tests here drive the public tool handlers - `process_local_video`
and its three siblings - and read the record back through the `get_job_status`
tool. Both ends are the surface a caller actually has, which matters: driving an
already-resolved source instead would skip the whole acquisition half of a run,
and that half is exactly where the record used to be missing. Nothing here needs
ffmpeg: source resolution is stubbed where a run has to get past it, and the
generation-producing step is replaced, so what is under test is the record.

One test needs a run whose holder is gone rather than one that returned, so it
spawns a real process and has the kernel kill it. That is the only honest way to
ask the question `abandoned` answers: liveness is the `flock` a dead holder's
descriptors release, and a holder that tidied up after itself is not a dead one.
"""

from __future__ import annotations

import json
import os
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
from distill.source import SourceResolution

STAGE_FAILURE = DistillError("E_STAGE_BOOM", "keyframes", "keyframe selection failed")
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLQHpFq3RA7fEJ0z3DABwTPvwre0Vu6OBH"
VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

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


def stub_acquisition(monkeypatch: pytest.MonkeyPatch, source_hash: str = "abc123") -> None:
    """Let a run past acquisition without a file, a probe or a network.

    Acquisition is what these tests need to *survive*, not what they are about:
    the tests that care about it fail it on purpose through the real path.
    """

    def resolve(
        _source_type: str,
        _value: str,
        _options: Any,
        *,
        progress: Any = None,
        downloader: Any = None,
        lock_wait_sec: float = 0.0,
    ) -> SourceResolution:
        # A stub stands in for the `SourceInfo` a real resolution returns: a run
        # touches only its **bundle key** and its warnings before the stage
        # these tests replace, and building a real one needs a real video.
        return SourceResolution(
            StubSource(source_hash=source_hash),  # ty: ignore[invalid-argument-type]
            output_root=None,
            progress=progress,
        )

    monkeypatch.setattr(pipeline, "resolve_source_for_processing", resolve)


def run(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    job_id: str,
    *,
    outcome: dict[str, Any] | BaseException,
    during: Callable[[], None] | None = None,
    source_hash: str = "abc123",
) -> dict[str, Any]:
    """Drive one `process_local_video` call to `outcome`, through the tool handler.

    The entry point is the tool a caller invokes, because the record has to
    cover everything that call does - acquisition included. The stage that
    produces a **generation** is replaced rather than faked out with tools,
    because what the run computes is not what these tests are about: a run
    either reaches a result or raises, and both have to leave a record.

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
    stub_acquisition(monkeypatch, source_hash)
    root.mkdir(parents=True, exist_ok=True)
    return pipeline.process_local_video(
        {"path": "/stub/video.mp4", "output_dir": str(root), "job_id": job_id}
    )


def status_of(root: Path, job_id: str) -> dict[str, Any]:
    """What a poller sees: the record read back through the `get_job_status` tool."""
    return pipeline.get_job_status({"output_dir": str(root), "job_id": job_id})


# --- The envelope covers the whole tool call, acquisition included (R-17) ----
#
# A run does not begin at a resolved source. It begins at a tool call, and
# everything between the two - a local probe, a **source fingerprint**, a
# YouTube download, a directory scan, a playlist listing - can fail. A record
# opened after all of that is finding 12 again for the entire acquisition half:
# no `running` while it happens, and no terminal failure when it does not
# finish. Job identifiers are reused, so what a poller reads in that window is
# whatever the *previous* run under the identifier left behind.


def test_a_local_run_that_never_reaches_a_source_records_a_failure(tmp_path: Path) -> None:
    """FAILS FIRST (finding 3-codex, R-17): the record started after acquisition.

    Nothing is stubbed here: the path does not exist, which is the commonest way
    a local run fails and the earliest. The record opened only once a resolved
    source existed, so this run wrote nothing at all and a poller was told
    `E_JOB_NOT_FOUND` - the answer a job that was never started gets.
    """
    root = tmp_path / "output"

    with pytest.raises(DistillError) as failure:
        pipeline.process_local_video(
            {
                "path": str(tmp_path / "no-such-video.mp4"),
                "output_dir": str(root),
                "job_id": "distill-no-source",
            }
        )
    assert failure.value.code == "E_BAD_SOURCE"

    status = status_of(root, "distill-no-source")
    assert status["status"] == "failed"
    assert status["tool"] == "process_local_video"
    assert status["error"]["code"] == "E_BAD_SOURCE"


def test_a_youtube_run_that_fails_while_acquiring_records_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 3-codex, R-17): the download is inside the run.

    A YouTube run spends most of its wall-clock time before it has a source at
    all - metadata, the lock wait, the download itself - so acquisition is where
    a poller is most likely to be asking and where a failure is most likely to
    land. It is stubbed at the resolution seam because the failure under test is
    that *any* of it raised, not which part.
    """
    root = tmp_path / "output"
    download_failed = DistillError("E_YTDLP", "youtube", "yt-dlp exited 1")

    def refuse(*_args: Any, **_kwargs: Any) -> SourceResolution:
        raise download_failed

    monkeypatch.setattr(pipeline, "resolve_source_for_processing", refuse)

    with pytest.raises(DistillError):
        pipeline.process_youtube_video(
            {"url": VIDEO_URL, "output_dir": str(root), "job_id": "distill-download"}
        )

    status = status_of(root, "distill-download")
    assert status["status"] == "failed"
    assert status["tool"] == "process_youtube_video"
    assert status["error"]["code"] == "E_YTDLP"


def test_a_playlist_that_cannot_be_listed_records_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 3-codex, R-17): enumeration precedes the batch.

    The playlist listing is the parent job's first real step and its own
    yt-dlp call. The record was opened around the per-item loop, so a playlist
    that could not be listed had no loop to wrap and left nothing behind.
    """
    root = tmp_path / "output"
    listing_failed = DistillError("E_YTDLP", "youtube", "yt-dlp could not list playlist videos")

    def refuse(*_args: Any, **_kwargs: Any) -> list[str]:
        raise listing_failed

    monkeypatch.setattr(pipeline, "youtube_playlist_urls", refuse)

    with pytest.raises(DistillError):
        pipeline.process_youtube_playlist(
            {"url": PLAYLIST_URL, "output_dir": str(root), "job_id": "distill-playlist"}
        )

    status = status_of(root, "distill-playlist")
    assert status["status"] == "failed"
    assert status["tool"] == "process_youtube_playlist"
    assert status["error"]["code"] == "E_YTDLP"


def test_a_directory_run_that_finds_no_directory_records_a_failure(tmp_path: Path) -> None:
    """FAILS FIRST (finding 3-codex, R-17): the scan precedes the batch too.

    The directory batch refused a path that is not a directory before opening
    its record, so the one failure every batch caller hits first was the one
    failure that left no record.
    """
    root = tmp_path / "output"

    with pytest.raises(DistillError) as failure:
        pipeline.process_video_directory(
            {
                "path": str(tmp_path / "no-such-directory"),
                "output_dir": str(root),
                "job_id": "distill-directory",
            }
        )
    assert failure.value.code == "E_BAD_SOURCE"

    status = status_of(root, "distill-directory")
    assert status["status"] == "failed"
    assert status["tool"] == "process_video_directory"
    assert status["error"]["code"] == "E_BAD_SOURCE"


def test_a_reused_job_id_does_not_survive_a_failed_acquisition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (finding 3-codex, R-17): the sharp end of the same gap.

    Writing no record is only half the damage. Identifiers are reused, so the
    record left standing is the *previous* run's - and a caller polling the run
    that just died in acquisition is handed `completed` and the earlier run's
    manifest path. `start` replaces the prior record, so covering acquisition is
    what makes that replacement happen before acquisition can fail.
    """
    root = tmp_path / "output"
    run(monkeypatch, root, "distill-again", outcome={"manifest_path": "/first/manifest.json"})
    assert status_of(root, "distill-again")["status"] == "completed"
    monkeypatch.undo()

    with pytest.raises(DistillError):
        pipeline.process_local_video(
            {
                "path": str(tmp_path / "gone.mp4"),
                "output_dir": str(root),
                "job_id": "distill-again",
            }
        )

    status = status_of(root, "distill-again")
    assert status["status"] == "failed"
    assert "result" not in status


def test_a_tool_call_opens_exactly_one_job_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One run, one record: the outer envelope replaces the inner one.

    Moving the record out to the tool handler without removing the one inside
    `process_resolved_source` would open two envelopes over one run. The second
    is refused outright now that a held identifier cannot be started twice, so
    the visible failure is a run that cannot run at all - but the property worth
    pinning is the count, because that is what stays true if the refusal ever
    softens.
    """
    root = tmp_path / "output"
    started: list[str] = []
    genuine_start = JobStore.start

    def counted(self: Any, job_id: str, tool: str) -> Any:
        started.append(job_id)
        return genuine_start(self, job_id, tool)

    monkeypatch.setattr(JobStore, "start", counted)
    run(monkeypatch, root, "distill-one-envelope", outcome={"manifest_path": "/x/manifest.json"})

    assert started == ["distill-one-envelope"]


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


def test_one_store_cannot_start_the_same_job_id_twice(tmp_path: Path) -> None:
    """FAILS FIRST (finding 5-codex): a held identifier was startable again.

    A store that already held the identifier returned from `start` without
    taking anything, and wrote a fresh `running` record over the live run's -
    two logical runs under one record, which is the collision R-18 exists to
    stop. The refusal has to be the same one a second process gets, because the
    two are the same mistake reached from different directions.
    """
    root = tmp_path / "output"
    store = JobStore.open(root)
    store.start("distill-twice", "process_local_video")

    with pytest.raises(DistillError) as raised:
        store.start("distill-twice", "process_video_directory")

    assert raised.value.code == "E_JOB_RUNNING"
    live = JobStore.open(root).read("distill-twice")
    assert live is not None
    assert live.tool == "process_local_video", "the live run's record was overwritten"


def test_a_store_holding_no_lock_cannot_close_a_live_runs_record(tmp_path: Path) -> None:
    """FAILS FIRST (finding 5-codex): `finish` never asked who was running.

    `finish` recovered the tool from whatever record was on disk, so a store
    that had started nothing could replace a live holder's `running` record with
    a terminal status - the run carries on, and the caller polling it is told it
    is over. Closing a record is a thing only its holder may do; a record whose
    holder is gone is answered by `abandoned`, which nobody has to write.
    """
    root = tmp_path / "output"
    holder = JobStore.open(root)
    holder.start("distill-live", "process_local_video")

    with pytest.raises(DistillError) as raised:
        JobStore.open(root).finish("distill-live", JobOutcome.success({"manifest_path": "/x.json"}))
    assert raised.value.code == "E_JOB_NOT_HELD"

    still_running = JobStore.open(root).read("distill-live")
    assert still_running is not None
    assert still_running.status == "running"

    holder.finish("distill-live", JobOutcome.success({"manifest_path": "/mine.json"}))
    closed = JobStore.open(root).read("distill-live")
    assert closed is not None
    assert closed.status == "completed"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can open a lock file whatever its mode")
def test_a_lock_file_that_cannot_be_opened_is_not_read_as_nobody_running(tmp_path: Path) -> None:
    """FAILS FIRST (finding 10-opus): the liveness question was asked with `take`.

    `take` opens with `O_CREAT` because a run that needs the lock needs the
    file; a reader must not, and a lock file it cannot open at all is not the
    same as one nobody holds. Asked through `take`, an unopenable lock file
    raised a bare `PermissionError` out of `get_job_status` - an inspection
    crashing on a root it was only asked about. `probe` is the read-only
    primitive for this, and it reports `unknown`, which is read as running:
    "may not ask" must never resolve to "nobody is running".
    """
    root = tmp_path / "output"
    store = JobStore.open(root)
    (root / "_jobs").mkdir(parents=True)
    store.record_path("distill-sealed").write_text(
        json.dumps(
            {
                "job_id": "distill-sealed",
                "tool": "process_local_video",
                "status": "running",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    sealed = store.lock_path("distill-sealed")
    sealed.write_bytes(b"")
    sealed.chmod(0o000)

    record = store.read("distill-sealed")

    assert record is not None
    assert record.status == "running"


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

    assert (
        JobStore.open(root).start("distill-unwritable", "process_local_video").status == "running"
    )


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
