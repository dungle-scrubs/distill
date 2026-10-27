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

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from runtime_fakes import configure_run

import distill.job_store as job_store
from distill import bundle_store, pipeline, run_command
from distill.artifacts import FrameArtifact, Interpretation, Provenance, RedactionState
from distill.bundle_store import ExclusiveLock
from distill.errors import DistillError
from distill.job_store import JobOutcome, JobProgress, JobStore
from distill.local_vision import LocalVisionConfig, LocalVisionProbe
from distill.progress import ProgressReporter, ProgressStatus
from distill.source import SourceInfo, SourceResolution

STAGE_FAILURE = DistillError("E_STAGE_BOOM", "keyframes", "keyframe selection failed")
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLQHpFq3RA7fEJ0z3DABwTPvwre0Vu6OBH"
VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

KILLED_MID_RUN = """
import os
import signal
import sys
from pathlib import Path

from distill.job_store import JobProgress, JobStore

store = JobStore.open(Path(sys.argv[1]))
store.start(sys.argv[2], "process_local_video")
store.note_progress(
    sys.argv[2],
    JobProgress(
        stage="local_vision",
        stage_status="running",
        stage_percent=50.0,
        overall_percent=42.0,
        observed_at="2026-10-27T09:00:00Z",
    ),
)
os.kill(os.getpid(), signal.SIGKILL)
"""
"""A run that starts a job, is seen partway through, and never gets to finish.

Killed with `SIGKILL` from inside, so nothing in Distill and nothing in Python
runs afterwards: no `finally`, no atexit, no interpreter shutdown. What is left
on disk is exactly what a `kill -9` leaves - a `running` record carrying the
last progress the run was seen making.
"""


def stub_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    source_hash: str = "abc123",
    *,
    provenance: Provenance | None = None,
) -> None:
    """Let a run past acquisition without a file, a probe or a network.

    Acquisition is what these tests need to *survive*, not what they are about:
    the tests that care about it fail it on purpose through the real path.

    `provenance` is passed through for the runs that go all the way to a
    **generation** through the real `ProcessingRun`: a self-contained render is
    refused without it, and the source stub's stand-in file never existed, so
    there is nowhere else it could have come from.
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
        return SourceResolution(
            SourceInfo(
                "local",
                Path("fixture.mp4"),
                1.0,
                "fingerprint",
                source_hash,
                [],
                provenance=provenance,
            ),
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
    during: Callable[[ProgressReporter | None], None] | None = None,
    source_hash: str = "abc123",
) -> dict[str, Any]:
    """Drive one `process_local_video` call to `outcome`, through the tool handler.

    The entry point is the tool a caller invokes, because the record has to
    cover everything that call does - acquisition included. The stage that
    produces a **generation** is replaced rather than faked out with tools,
    because what the run computes is not what these tests are about: a run
    either reaches a result or raises, and both have to leave a record.

    `during` runs inside that stage, at the point a real run is midway through
    its work, and is handed the **progress reporter** the run is actually
    carrying - the one the handler built, flowing through the same reporter a
    real stage emits through - so a test can ask what a poller would be shown
    for events a run really sent. It is the only moment from which "what does a
    poller see while this is still running?" can be asked at all.
    """
    captured: list[ProgressReporter | None] = []

    def fake_processing_run(*args: Any) -> Any:
        # The reporter is the fourth thing a run is handed, after the source,
        # the options and the output root; capturing it is what lets `during`
        # emit through the run's own reporter rather than a test's own.
        captured.append(args[3] if len(args) > 3 else None)
        return SimpleNamespace(execute=produce)

    def produce() -> dict[str, Any]:
        if during is not None:
            during(captured[-1] if captured else None)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(pipeline, "ProcessingRun", fake_processing_run)
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
    assert status["error"]["message"] == "directory does not exist"


def test_a_directory_run_pointed_at_a_file_finds_no_directory(tmp_path: Path) -> None:
    """The third answer: a path that is there and is not a directory.

    "No directory here" is true of a regular file, so this is reported like an
    absent path rather than refused like an unreachable one - the batch knows
    what it found. Pinned because it is the only assertion that separates "is a
    directory" from "could be stat'ed".
    """
    root = tmp_path / "output"
    not_a_directory = tmp_path / "videos.mp4"
    not_a_directory.write_bytes(b"\x00")

    with pytest.raises(DistillError) as failure:
        pipeline.process_video_directory(
            {
                "path": str(not_a_directory),
                "output_dir": str(root),
                "job_id": "distill-file-directory",
            }
        )

    assert failure.value.code == "E_BAD_SOURCE"
    assert failure.value.message == "directory does not exist"
    assert not_a_directory.read_bytes() == b"\x00"


def test_a_directory_run_pointed_at_an_impossible_path_finds_no_directory(
    tmp_path: Path,
) -> None:
    """A NUL in the path is refused as a path, not raised as a `ValueError`.

    The fourth answer, and the one an errno split does not cover: `stat` of a
    path holding a NUL raises `ValueError`, which is neither an `OSError` nor a
    diagnosis. Both `Path.exists()` implementations swallow it on purpose, so a
    guard that replaces `exists()` has to swallow it too or an operator's typo
    becomes an internal fault.
    """
    root = tmp_path / "output"

    with pytest.raises(DistillError) as failure:
        pipeline.process_video_directory(
            {
                "path": f"{tmp_path}/a\x00b",
                "output_dir": str(root),
                "job_id": "distill-impossible-directory",
            }
        )

    assert failure.value.code == "E_BAD_SOURCE"
    assert failure.value.message == "directory does not exist"


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not refused by directory permissions")
def test_a_directory_run_that_may_not_look_says_so_rather_than_no_such_directory(
    tmp_path: Path,
) -> None:
    """FAILS FIRST: "directory does not exist" about a directory that may be full.

    The scan opened with `Path.exists()`, which cannot tell the two refusals
    apart - and tells them apart differently on each interpreter. On 3.14 it
    answers `False` for a directory whose parent this user may not search, so
    the batch reported the operator's own videos as a path that is not there; on
    3.13 it raised `PermissionError` and the run ended as `E_INTERNAL`, a defect
    in Distill rather than a permission the operator can grant.

    The record is asserted for the same reason its neighbour asserts it: both
    refusals happen inside `work()`, so both are failures of a run that started.
    """
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    unreachable = sealed / "videos"
    unreachable.mkdir()
    root = tmp_path / "output"
    sealed.chmod(0o000)
    try:
        with pytest.raises(DistillError) as failure:
            pipeline.process_video_directory(
                {
                    "path": str(unreachable),
                    "output_dir": str(root),
                    "job_id": "distill-sealed-directory",
                }
            )
    finally:
        sealed.chmod(0o700)

    assert failure.value.code == "E_SOURCE_UNREADABLE"
    assert failure.value.stage == "source"
    assert failure.value.details == {"path": str(unreachable), "errno": "EACCES"}

    status = status_of(root, "distill-sealed-directory")
    assert status["status"] == "failed"
    assert status["error"]["code"] == "E_SOURCE_UNREADABLE"
    assert status["error"]["message"] == "source path could not be read"


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
        "message": "an unexpected RuntimeError ended the command",
        "details": {"exception": "RuntimeError", "message": "unmapped"},
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

    def observe(_progress: ProgressReporter | None) -> None:
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


def lock_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """The lock stream, read where a job lock's events actually land.

    A job's lock is an `ExclusiveLock`, and its own `lock_released` events go to
    the bundle store's logger under the job identifier as `subject`. A release
    that *failed* is the same fact about the same lock, so it is read from the
    same stream rather than from a second one invented for job records.
    """
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == bundle_store.LOGGER.name and record.message.startswith("{")
    ]


def refuse_to_release(_self: ExclusiveLock) -> None:
    """A release that cannot give the lock up - `os.close` on a bad descriptor."""
    raise OSError("the descriptor could not be closed")


@pytest.mark.parametrize(
    "in_flight",
    [
        KeyboardInterrupt(),
        DistillError("E_DISK_FULL", "job", "cannot write the record"),
    ],
    ids=["interrupt", "distill_error"],
)
def test_a_failure_travelling_through_finish_survives_a_release_that_fails(
    in_flight: BaseException,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST: the release's `OSError` replaces whatever `finish` was carrying.

    `finish` releases the hold in a bare `finally`, and cleanup that raises
    while an exception is travelling *substitutes* that exception for its own.
    An operator's `Ctrl-C` therefore left the CLI boundary as `E_INTERNAL`, exit
    2, blaming an `OSError` about a descriptor - the run was not ended by a
    defect, and the interrupt was not Distill's to relabel. The same
    substitution buries an ordinary `DistillError`: the diagnosis a caller can
    act on is replaced by one it cannot.

    Both halves are asserted, because declining to raise and dropping the
    failure look identical from the caller's side: the original exception
    reaches the caller, *and* the lock that was not given up is recorded with
    the failure that was in flight when it happened. A hold this process
    believes it released and did not is a job identifier nothing can start
    again for the life of the process.
    """
    root = tmp_path / "output"
    store = JobStore.open(root)
    store.start("distill-cleanup", "process_local_video")
    caplog.set_level(logging.DEBUG, logger=bundle_store.LOGGER.name)

    def refuse_to_write(_self: JobStore, _record: Any) -> Any:
        raise in_flight

    monkeypatch.setattr(JobStore, "_write", refuse_to_write)
    monkeypatch.setattr(ExclusiveLock, "release", refuse_to_release)

    with pytest.raises(type(in_flight)) as raised:
        store.finish("distill-cleanup", JobOutcome.success({"manifest_path": "/x.json"}))
    assert raised.value is in_flight, "the release replaced the exception it was cleaning up after"

    failures = [event for event in lock_events(caplog) if event["event"] == "lock_release_failed"]
    assert len(failures) == 1
    assert failures[0]["detail"]["subject"] == "distill-cleanup"
    assert "could not be closed" in failures[0]["detail"]["error"]
    assert failures[0]["detail"]["during"] == type(in_flight).__name__


def test_a_failure_travelling_through_start_survives_a_release_that_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same defect at the other end: `start` releases while a failure travels.

    `start` gives the hold up when the `running` record cannot be written, for
    the reason the record-write test covers. That release is fallible too, so an
    interrupt landing on the first write was relabelled exactly as one landing
    on the last.
    """
    root = tmp_path / "output"
    store = JobStore.open(root)
    interrupt = KeyboardInterrupt()

    def refuse_to_write(_self: JobStore, _record: Any) -> Any:
        raise interrupt

    monkeypatch.setattr(JobStore, "_write", refuse_to_write)
    monkeypatch.setattr(ExclusiveLock, "release", refuse_to_release)

    with pytest.raises(KeyboardInterrupt) as raised:
        store.start("distill-cleanup-start", "process_local_video")
    assert raised.value is interrupt


def test_a_release_that_fails_with_nothing_in_flight_is_still_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other half: with nothing travelling there is nothing to preserve.

    A hold this process believes it released and did not is a job identifier no
    later run can take, so swallowing the failure unconditionally would trade
    one wrong diagnosis for a silent one.
    """
    root = tmp_path / "output"
    store = JobStore.open(root)
    store.start("distill-clean-exit", "process_local_video")
    monkeypatch.setattr(ExclusiveLock, "release", refuse_to_release)

    with pytest.raises(OSError, match="could not be closed"):
        store.finish("distill-clean-exit", JobOutcome.success({"manifest_path": "/x.json"}))


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


# --- The record names the stage a run is in (issue #45) ----------------------
#
# `running` answered "is it still going" and nothing else: a run whose last
# event said `ocr` completed and whose vision stage then went silent had a
# record with no stage and no activity time in it, so a poller could not tell
# a slow run from a stuck one. These tests hold the projection of the run's
# own progress events into the record a poller reads - names and numbers only,
# written by the run that owns the job, while it owns it.


def fake_stage_producers(monkeypatch: pytest.MonkeyPatch, frame_count: int = 2) -> None:
    """Stages that produce real carriers without ffmpeg, whisper or tesseract.

    The stages a poller is not asking about run as themselves with their
    external dependency injected, which is the `runtime_fakes` seam. The vision
    stage is not injected at all - it runs for real, the one whose silence the
    issue describes - and a test blocks it through the `probe=`/`try_interpret=`
    injection `configure_run` already offers.
    """

    def transcribe(*_args: Any, **_kwargs: Any) -> tuple[None, list[dict[str, str]]]:
        return None, []

    def select_keyframes(
        _video: Path,
        frames_dir: Path,
        *_args: Any,
        redaction: RedactionState,
        **_kwargs: Any,
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        artifacts: list[FrameArtifact] = []
        for index in range(1, frame_count + 1):
            image = frames_dir / f"frame_{index:04d}.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            artifacts.append(
                FrameArtifact(
                    index=index,
                    timestamp_sec=float(index),
                    path=str(image),
                    relative_path=f"frames/frame_{index:04d}.png",
                    extracted_text="a frame this test never reads",
                    redaction=redaction,
                )
            )
        return artifacts, []

    configure_run(monkeypatch, transcribe=transcribe, select_keyframes=select_keyframes)


def an_interpretation(config: LocalVisionConfig, image_path: Path, prompt_profile: str) -> Any:
    """A reading the vision stage accepts as one, from no model at all."""
    return Interpretation(
        visual_summary=image_path.name,
        detected_elements=(),
        interpretation="a reading this test supplied",
        uncertainty="Low",
        backend=config.backend,
        model=config.model,
        prompt_profile=prompt_profile,
    )


def an_available_probe(config: LocalVisionConfig) -> LocalVisionProbe:
    return LocalVisionProbe(
        available=True,
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        code="local_vision_available",
        message="available",
        detail={},
    )


def a_fake_interpretation(
    config: LocalVisionConfig, image_path: Path, _prompt: str, *, prompt_profile: str = "technical"
) -> tuple[Interpretation, None]:
    return an_interpretation(config, image_path, prompt_profile), None


def a_provenance() -> Provenance:
    return Provenance(title="fixture.mp4", duration_sec=1.0, processed_at="2026-10-27T09:00:00Z")


def test_a_blocked_vision_request_names_the_stage_in_job_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (issue #45): a silent run answered `running` with no stage.

    A vision request that never returns is the exact wait the issue could not
    see: the run's last event said `ocr` completed, the record said nothing but
    `running`, and a caller timed out without knowing what it was waiting for.
    The run is driven through the real handler with the real `ProcessingRun`
    and the real `FrameInterpreter` - only the probe is injected, and it blocks
    - so the stage a poller is shown is announced by the code a real run runs,
    not by a test re-implementing its events.
    """
    root = tmp_path / "output"
    job_id = "distill-blocked-vision"
    entered = threading.Event()
    release = threading.Event()

    def blocked_probe(config: LocalVisionConfig) -> LocalVisionProbe:
        entered.set()
        if not release.wait(timeout=10.0):
            raise AssertionError("the poller never released the probe")
        return LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_available",
            message="available",
            detail={},
        )

    def interpret(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        return an_interpretation(config, image_path, prompt_profile), None

    fake_stage_producers(monkeypatch)
    configure_run(monkeypatch, probe=blocked_probe, try_interpret=interpret)
    stub_acquisition(monkeypatch, provenance=a_provenance())

    result: dict[str, Any] = {}
    failures: list[BaseException] = []

    def work() -> None:
        try:
            result["response"] = pipeline.process_local_video(
                {
                    "path": "/stub/video.mp4",
                    "output_dir": str(root),
                    "job_id": job_id,
                    "ocr": False,
                }
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert entered.wait(timeout=10.0), "the run never reached the vision stage"
        status = status_of(root, job_id)
        written_status = json.loads((root / "_jobs" / f"{job_id}.json").read_text())
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive(), "the run did not finish after the probe was released"

    # What a poller sees while the vision request is still in flight: the
    # stage by name, that it is running, and the two times that separate
    # "when this job began" from "when it was last seen to do anything".
    assert status["status"] == "running"
    assert status["stage"] == "local_vision"
    assert status["stage_status"] == "running"
    assert status["stage_percent"] == 0.0
    assert "overall_percent" in status
    assert status["started_at"]
    assert status["last_progress_at"]
    assert status["updated_at"] >= status["started_at"]
    # The allowlist, pinned (round 2, tests lens 2): a record carries the
    # projection's names and numbers only - the write-side sequence and the
    # event detail never reach one, on the running record or the terminal one.
    assert "sequence" not in status
    assert "detail" not in status
    assert not {"sequence", "detail"} & written_status.keys()

    final = status_of(root, job_id)
    assert final["status"] == "completed"
    assert final["result"]["manifest_path"]
    # The start is one fact, set when the record was opened, and is not
    # rewritten by anything the run does afterwards - including its ending.
    assert final["started_at"] == status["started_at"]
    # A terminal record preserves the last stage the run was seen in. A run
    # that finished moves on after vision - rendering, publish - so for a
    # completed run the last observed stage is the publish; what the assertion
    # holds is that it is the run's own last event and not a blank.
    assert final["stage"] == "bundle_publish"
    assert final["stage_status"] == "completed"
    assert final["last_progress_at"]
    assert "sequence" not in final
    assert "detail" not in final
    written_final = json.loads((root / "_jobs" / f"{job_id}.json").read_text())
    assert not {"sequence", "detail"} & written_final.keys()


def test_a_blocked_frame_leaves_the_completed_frames_counted_in_job_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (issue #45): per-frame vision events arrived only in one burst.

    The vision stage interpreted every frame and *then* reported all of them at
    once, so a poller watching a run whose second frame hung saw a stage that
    had entered and nothing more - the whole vision pass silent until its last
    frame returned. The frame events are now reported as each frame completes,
    in the caller thread, so the record names the position: frame 1 of 3 done,
    no further activity.
    """
    root = tmp_path / "output"
    job_id = "distill-blocked-frame"
    entered = threading.Event()
    release = threading.Event()
    # A deterministic monotonic clock, advanced by the frame fakes rather than
    # by sleeping: the note throttle is a second of monotonic time, and a test
    # that waited out real seconds to pass it would be slow for no evidence.
    clock = {"now": 0.0}
    monkeypatch.setattr(pipeline, "_monotonic", lambda: clock["now"])

    def available_probe(config: LocalVisionConfig) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_available",
            message="available",
            detail={},
        )

    def interpret(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        if image_path.name == "frame_0002.png":
            entered.set()
            if not release.wait(timeout=10.0):
                raise AssertionError("the poller never released the frame")
        else:
            # A frame that lands fast - well inside the note throttle's window,
            # which is what a real run's early frames do. Frame milestones are
            # persisted the moment they are consumed, so no clock advance is
            # needed to make one visible; the clock still moves so the run's
            # later events are not mistaken for a frozen stream.
            clock["now"] += 0.25
        return an_interpretation(config, image_path, prompt_profile), None

    fake_stage_producers(monkeypatch, frame_count=3)
    configure_run(monkeypatch, probe=available_probe, try_interpret=interpret)
    stub_acquisition(monkeypatch, provenance=a_provenance())

    failures: list[BaseException] = []

    def work() -> None:
        try:
            pipeline.process_local_video(
                {
                    "path": "/stub/video.mp4",
                    "output_dir": str(root),
                    "job_id": job_id,
                    "ocr": False,
                }
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert entered.wait(timeout=10.0), "the run never reached the second frame"
        status = status_of(root, job_id)
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive(), "the run did not finish after the frame was released"

    # One frame of three has completed; the second is in flight. A poller that
    # cannot see the event stream still knows both facts: the position, and
    # that the last activity was that frame's completion. The percent carries
    # the reporter's 3-decimal rounding (`clamp_percent`) - one frame of three
    # is 33.333 however the code computes it.
    assert status["status"] == "running"
    assert status["stage"] == "local_vision"
    assert status["stage_status"] == "running"
    assert status["stage_percent"] == pytest.approx(33.333)
    assert status["last_progress_at"]

    final = status_of(root, job_id)
    assert final["status"] == "completed"


def test_a_record_written_before_progress_fields_reads_as_it_always_did(
    tmp_path: Path,
) -> None:
    """Every record Distill wrote before this change is still a record.

    The progress fields are read leniently because their absence was the truth
    for every job ever run: refusing an old record now would be telling every
    poller that their history is unreadable, in exchange for nothing - the
    fields are a projection, and a run that never noted one was a run whose
    stages had nothing to say.
    """
    root = tmp_path / "output"
    (root / "_jobs").mkdir(parents=True)
    (root / "_jobs" / "distill-old.json").write_text(
        json.dumps(
            {
                "job_id": "distill-old",
                "tool": "process_youtube_video",
                "status": "completed",
                "updated_at": "2026-01-01T00:00:00Z",
                "result": {"manifest_path": "/old/manifest.json"},
            }
        )
    )

    status = status_of(root, "distill-old")

    assert status["status"] == "completed"
    assert status["result"]["manifest_path"] == "/old/manifest.json"
    assert "stage" not in status
    assert "stage_status" not in status
    assert "stage_percent" not in status
    assert "overall_percent" not in status
    assert "started_at" not in status
    assert "last_progress_at" not in status


def a_foreign_record(root: Path, job_id: str, **fields: Any) -> None:
    """Write a record this run did not write, as a poller might find one.

    `get_job_status` answers for any record under the root, and a hand-edited
    or corrupt one is exactly the case lenient reading was written for, so the
    parse's answers are pinned from the same surface a caller reads.
    """
    (root / "_jobs").mkdir(parents=True, exist_ok=True)
    (root / "_jobs" / f"{job_id}.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "tool": "process_local_video",
                # Terminal, so the read is a parse and not a liveness question:
                # no holder exists for a record nobody's run is behind.
                "status": "completed",
                "updated_at": "2026-10-27T09:00:00Z",
                **fields,
            }
        )
    )


def test_a_half_projected_progress_is_absent_rather_than_halved(tmp_path: Path) -> None:
    """FAILS FIRST (review round 1): a stage with no status read back as one.

    `JobProgress` exists so a projection cannot be handed a stage with no
    status; the parse undid that by reading five independent optionals. A
    record that carries `stage` and not `stage_status` is either corrupt or
    foreign, and the honest answer is no projection at all - not the half of
    one, which reads as a stage that is somehow in no state.
    """
    root = tmp_path / "output"
    a_foreign_record(
        root, "distill-half", stage="local_vision", last_progress_at="2026-10-27T09:00:00Z"
    )

    status = status_of(root, "distill-half")

    assert status["status"] == "completed"
    assert "stage" not in status
    assert "stage_status" not in status
    assert "stage_percent" not in status
    assert "overall_percent" not in status
    assert "last_progress_at" not in status


def test_percents_a_percent_cannot_be_do_not_reach_the_status_json(
    tmp_path: Path,
) -> None:
    """FAILS FIRST (review round 1): `NaN` and `1e9` read back as percents.

    The writer can only write `clamp_percent`'s answers, but the reader answers
    for any record under the root, and Python's JSON accepts `NaN` and
    `Infinity` on the way in and re-emits them on the way out - so one corrupt
    field made the whole `get-job-status` document one a strict parser rejects.
    A percent that is not a finite number inside 0..100 is absent; the record
    still reads, and the stage it names still answers.
    """
    root = tmp_path / "output"
    a_foreign_record(
        root,
        "distill-nan",
        stage="local_vision",
        stage_status="running",
        last_progress_at="2026-10-27T09:00:00Z",
        stage_percent=float("nan"),
        overall_percent=1e9,
    )

    status = status_of(root, "distill-nan")

    assert status["stage"] == "local_vision"
    assert status["stage_status"] == "running"
    assert "stage_percent" not in status
    assert "overall_percent" not in status
    # The proof of the leak: the answer a caller is handed re-serializes as
    # strict JSON, `NaN` and all.
    assert "NaN" not in json.dumps(status)
    assert "Infinity" not in json.dumps(status)

    a_foreign_record(
        root,
        "distill-out-of-range",
        stage="local_vision",
        stage_status="running",
        last_progress_at="2026-10-27T09:00:00Z",
        stage_percent=150.0,
        overall_percent=-5.0,
    )
    bounded = status_of(root, "distill-out-of-range")
    assert "stage_percent" not in bounded
    assert "overall_percent" not in bounded


# --- A batch's parent answers for its children (review round 1) ------------
#
# The ticket's promise is written for "a running job", and a playlist or
# directory run is exactly the long job whose caller polls. The parent record
# used to carry lifecycle fields only: the stage lived in each child's own
# record, so a caller polling the parent it started still saw `running` with
# no stage - the original silence, one level up. The parent now observes its
# children: it mirrors the child's own stage and status, maps the child's
# progress onto the batch's fraction, and counts items as they complete,
# while each child keeps its own record under its own job id.


def stub_each_source_uniquely(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a batch's children past acquisition as *different* sources.

    One shared stub hash would name one **bundle key**, and the second child
    would be served the first child's generation as a cache hit - never
    reaching the vision stage the batch test blocks in. Each value the batch
    hands the handler gets its own hash, so every child runs for real.
    """

    def resolve(
        _source_type: str,
        value: str,
        _options: Any,
        *,
        progress: Any = None,
        downloader: Any = None,
        lock_wait_sec: float = 0.0,
    ) -> SourceResolution:
        digest = hashlib.sha256(str(value).encode()).hexdigest()[:32]
        return SourceResolution(
            SourceInfo(
                "local",
                Path(value),
                1.0,
                "fingerprint",
                digest,
                [],
                provenance=a_provenance(),
            ),
            output_root=None,
            progress=progress,
        )

    monkeypatch.setattr(pipeline, "resolve_source_for_processing", resolve)


def a_blocking_probe_for_the_second_child(
    entered: threading.Event, release: threading.Event
) -> Callable[[LocalVisionConfig], LocalVisionProbe]:
    """An available probe that blocks the second child's vision request.

    Children run one at a time, so the second probe call this store sees is
    the second child's - the run a caller is left waiting on.
    """
    calls: list[int] = []

    def probe(config: LocalVisionConfig) -> LocalVisionProbe:
        calls.append(len(calls) + 1)
        if len(calls) == 2:
            entered.set()
            if not release.wait(timeout=10.0):
                raise AssertionError("the poller never released the second child")
        return LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_available",
            message="available",
            detail={},
        )

    return probe


@pytest.mark.parametrize(
    ("handler", "args"),
    [
        ("process_video_directory", {"path": "__REPLACED__"}),
        ("process_youtube_playlist", {"url": PLAYLIST_URL}),
    ],
    ids=["directory", "playlist"],
)
def test_a_blocked_child_names_the_stage_in_the_parent_record_too(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    handler: str,
    args: dict[str, str],
) -> None:
    """FAILS FIRST (review round 1): the parent answered `running` with no stage.

    One item completes; the second blocks inside its vision request. What a
    poller reads then, from the parent's own record and the child's, is the
    whole of the parent's coverage: the parent mirrors the child's stage and
    carries the batch's fraction - one of two items done, the second partway -
    while the child's record says the same thing about itself under its own
    job id, in its own record root for a playlist.
    """
    root = tmp_path / "output"
    job_id = "distill-batch-parent"
    entered = threading.Event()
    release = threading.Event()

    if handler == "process_video_directory":
        videos = tmp_path / "videos"
        videos.mkdir()
        (videos / "one.mp4").write_bytes(b"one")
        (videos / "two.mp4").write_bytes(b"two")
        call_args = {**args, "path": str(videos)}
        parent_root, child_root = root, root
    else:
        monkeypatch.setattr(
            pipeline, "youtube_playlist_urls", lambda *_a, **_k: [VIDEO_URL, PLAYLIST_URL]
        )
        call_args = dict(args)
        parent_root = root
        child_root = root / "playlists" / pipeline.playlist_folder_name(PLAYLIST_URL)

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        probe=a_blocking_probe_for_the_second_child(entered, release),
        try_interpret=a_fake_interpretation,
    )
    stub_each_source_uniquely(monkeypatch)

    failures: list[BaseException] = []
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["response"] = getattr(pipeline, handler)(
                {**call_args, "output_dir": str(root), "job_id": job_id, "ocr": False}
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert entered.wait(timeout=10.0), "the batch never reached its second child"
        parent = status_of(parent_root, job_id)
        child = status_of(child_root, f"{job_id}-2")
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive(), "the batch did not finish after the child was released"

    # The parent mirrors the blocked child's own answer: the stage by name,
    # running, with real activity time and the start preserved - plus the one
    # fact only the parent knows, that one item of two is already done.
    assert parent["status"] == "running"
    assert parent["stage"] == "local_vision"
    assert parent["stage_status"] == "running"
    assert parent["started_at"]
    assert parent["last_progress_at"]
    assert 50.0 <= parent["overall_percent"] < 100.0

    # The child answers for itself under its own job id, in its own root.
    assert child["status"] == "running"
    assert child["stage"] == "local_vision"
    assert child["stage_status"] == "running"

    final_parent = status_of(parent_root, job_id)
    assert final_parent["status"] == "completed"
    assert final_parent["result"]["processed_count"] == 2
    assert status_of(child_root, f"{job_id}-1")["status"] == "completed"
    assert status_of(child_root, f"{job_id}-2")["status"] == "completed"


def test_a_note_from_a_finished_item_cannot_regress_the_next_items_parent_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (review round 1): nothing observed children, or anything could.

    A dispatcher callback can outlive the invocation that queued it, so a
    batch item's own emitter may still be holding events when the *next* item
    is running. Forwarding is therefore armed per child call and disarmed the
    moment the child returns: a late event from item one lands nowhere, and
    the parent keeps describing item two - the item that is actually live.
    """
    root = tmp_path / "output"
    job_id = "distill-stale-item"
    entered = threading.Event()
    release = threading.Event()
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "one.mp4").write_bytes(b"one")
    (videos / "two.mp4").write_bytes(b"two")
    stale_emitters: list[Any] = []
    stale_events: list[Any] = []

    def ocr_capturing_the_first_childs_emitter(
        frames: list[FrameArtifact],
        _language: str,
        _enabled: bool,
        progress: ProgressReporter,
        _preprocess: bool = True,
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        # The first child's sink, held past its run the way a slow dispatcher
        # callback would hold it.
        if not stale_emitters and progress.emitter is not None:
            stale_emitters.append(progress.emitter)
            stale_events.append(progress.update("ocr", percent=50.0))
        return frames, []

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        ocr_frames=ocr_capturing_the_first_childs_emitter,
        probe=a_blocking_probe_for_the_second_child(entered, release),
        try_interpret=a_fake_interpretation,
    )
    stub_each_source_uniquely(monkeypatch)

    failures: list[BaseException] = []

    def work() -> None:
        try:
            pipeline.process_video_directory(
                {
                    "path": str(videos),
                    "output_dir": str(root),
                    "job_id": job_id,
                    "ocr": False,
                }
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert entered.wait(timeout=10.0), "the batch never reached its second child"
        # Item one is finished and its sink is still held: replay its event, as
        # a delayed delivery would.
        stale_emitters[0](stale_events[0])
        parent = status_of(root, job_id)
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive()

    assert parent["stage"] == "local_vision", parent
    assert parent["stage_status"] == "running", parent

    final = status_of(root, job_id)
    assert final["status"] == "completed"


def test_disarm_waits_for_an_admitted_observation_from_a_failing_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The admission lock's own proof (round 2, tests lens 1): disarm must wait.

    The surviving mutant dropped the lock from `disarm` alone: every existing
    proof stayed green because a live child's own later events queue on the
    admission lock, so disarm was never contended. Here the admitted
    observation is the *failed* item's last progress event - the child returns
    while its observation is in flight, exactly the case disarm's lock exists
    for - and the next item must not start until that observation lands.
    """

    def transcribe_that_fails_the_first_item_after_its_last_event(
        resolved_path: Path,
        _scratch_dir: Path,
        _options: Any,
        progress: ProgressReporter,
        *,
        duration_sec: float | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if "one" not in str(resolved_path):
            return {"language": "en", "language_probability": 1.0, "segments": ()}, []

        def on_stderr_line(_line: str) -> None:
            progress.update("transcription", percent=25.0)

        run_command.stream(
            [sys.executable, "-c", 'import sys; print("fixture progress", file=sys.stderr)'],
            stage="transcript",
            total_timeout_sec=30.0,
            idle_timeout_sec=30.0,
            on_stderr_line=on_stderr_line,
            check=False,
        )
        # The report above was the item's last progress event; nothing else it
        # does can queue behind the observation that event earned.
        raise DistillError(
            "E_FAILS_AFTER_LAST_EVENT", "transcript", "the item fails after its last progress event"
        )

    root = tmp_path / "output"
    job_id = "distill-disarm-failing-item"
    child_one = f"{job_id}-1"
    child_two = f"{job_id}-2"
    parked = threading.Event()
    release_observer = threading.Event()
    committed = threading.Event()
    child_one_failed = threading.Event()
    child_two_started = threading.Event()
    genuine_write = JobStore._write
    genuine_reporter = pipeline.ProgressReporter
    parked_once: list[bool] = []
    parent_built: list[bool] = []

    def reporter_that_parks_the_admitted_observation(*args: Any, **kwargs: Any) -> Any:
        reporter = genuine_reporter(*args, **kwargs)
        if parked_once or parent_built:
            return reporter
        parent_built.append(True)
        genuine_update = reporter.update

        def parking_update(
            mechanism: str, *, status: ProgressStatus = "running", **update: Any
        ) -> Any:
            if mechanism == "transcription" and not parked_once:
                # Past observe's admission (the lock is held), inside the
                # parent update it admitted: where a slow parent write sits.
                parked_once.append(True)
                parked.set()
                assert release_observer.wait(timeout=10.0), "the test never released the observer"
                try:
                    return genuine_update(mechanism, status=status, **update)
                finally:
                    committed.set()
            return genuine_update(mechanism, status=status, **update)

        # Shadowing the bound method is the injection; the wrapper forwards
        # every keyword untouched.
        reporter.update = parking_update  # ty: ignore[invalid-assignment]
        return reporter

    def signalling_write(store: JobStore, record: Any) -> Any:
        payload = record.to_dict()
        written = genuine_write(store, record)
        if payload.get("job_id") == child_one and payload.get("status") == "failed":
            child_one_failed.set()
        if payload.get("job_id") == child_two and payload.get("status") == "running":
            child_two_started.set()
        return written

    monkeypatch.setattr(pipeline, "ProgressReporter", reporter_that_parks_the_admitted_observation)
    monkeypatch.setattr(JobStore, "_write", signalling_write)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "one.mp4").write_bytes(b"one")
    (videos / "two.mp4").write_bytes(b"two")

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        transcribe=transcribe_that_fails_the_first_item_after_its_last_event,
        probe=an_available_probe,
        try_interpret=a_fake_interpretation,
    )
    stub_each_source_uniquely(monkeypatch)

    failures: list[BaseException] = []
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["response"] = pipeline.process_video_directory(
                {"path": str(videos), "output_dir": str(root), "job_id": job_id, "ocr": False}
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert parked.wait(timeout=10.0), "the failed item's observation never went in flight"
        assert child_one_failed.wait(timeout=10.0), (
            "item one never failed around the in-flight observation"
        )
        # While the admitted observation is still in flight, the next item
        # must not start. The bound is generous to the unlocked disarm, which
        # starts item two almost at once; the locked one waits it out.
        next_item_started_early = child_two_started.wait(timeout=2.0)
        release_observer.set()
        assert committed.wait(timeout=10.0), "the admitted observation never landed"
    finally:
        release_observer.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive()
    assert not next_item_started_early, "item two began before the failed item's observation landed"

    final = status_of(root, job_id)
    assert final["status"] == "completed"
    assert final["result"]["error_count"] == 1
    assert status_of(root, child_one)["error"]["code"] == "E_FAILS_AFTER_LAST_EVENT"
    assert status_of(root, child_two)["status"] == "completed"


def test_an_admitted_observer_write_lands_before_the_next_item_takes_over(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (lead's finalization): disarm did not wait for an admitted write.

    A child event delivered on a dispatcher thread can be mid-observation when
    the child call returns, because the supervisor's shutdown is bounded rather
    than waited for. The observation is admitted the moment its forwarding
    check passes; disarming the item used to clear forwarding independently,
    so the batch moved on to the next item while an admitted event from item
    one was still in flight - the same ordering hole the terminal-outcome
    regression closed for `finish`, one level up. The pause is injected
    between admission and the parent's write, exactly where a slow thread
    leaves one: unserialized, item two's record opens before the observation
    lands; serialized, item one's remaining events wait their turn and item
    two cannot start.
    """
    root = tmp_path / "output"
    job_id = "distill-admitted-observer"
    child_two = f"{job_id}-2"
    entered = threading.Event()
    release_probe = threading.Event()
    parked = threading.Event()
    release_observer = threading.Event()
    committed = threading.Event()
    child_two_started = threading.Event()
    genuine_write = JobStore._write
    genuine_reporter = pipeline.ProgressReporter
    parked_once: list[bool] = []
    parent_built: list[bool] = []

    def reporter_that_parks_the_first_child_mirror(*args: Any, **kwargs: Any) -> Any:
        reporter = genuine_reporter(*args, **kwargs)
        if parked_once or parent_built:
            return reporter
        parent_built.append(True)
        genuine_update = reporter.update

        def parking_update(
            mechanism: str, *, status: ProgressStatus = "running", **update: Any
        ) -> Any:
            if mechanism == "transcription" and not parked_once:
                # Item one's mirrored event, admitted by the forwarding check
                # and paused before the parent's reporter sees it: the window
                # disarm has to close.
                parked_once.append(True)
                parked.set()
                assert release_observer.wait(timeout=10.0), "the test never released the observer"
                try:
                    return genuine_update(mechanism, status=status, **update)
                finally:
                    committed.set()
            return genuine_update(mechanism, status=status, **update)

        # Shadowing the bound method is the injection; the wrapper forwards
        # every keyword untouched.
        reporter.update = parking_update  # ty: ignore[invalid-assignment]
        return reporter

    def signalling_write(store: JobStore, record: Any) -> Any:
        payload = record.to_dict()
        written = genuine_write(store, record)
        if payload.get("job_id") == child_two and payload.get("status") == "running":
            child_two_started.set()
        return written

    # The batch's parent is the first reporter built under the handler; the
    # children built after it are untouched.
    monkeypatch.setattr(pipeline, "ProgressReporter", reporter_that_parks_the_first_child_mirror)
    monkeypatch.setattr(JobStore, "_write", signalling_write)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "one.mp4").write_bytes(b"one")
    (videos / "two.mp4").write_bytes(b"two")

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        transcribe=a_real_process_reporting_progress,
        probe=a_blocking_probe_for_the_second_child(entered, release_probe),
        try_interpret=a_fake_interpretation,
    )
    stub_each_source_uniquely(monkeypatch)

    failures: list[BaseException] = []
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["response"] = pipeline.process_video_directory(
                {"path": str(videos), "output_dir": str(root), "job_id": job_id, "ocr": False}
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert parked.wait(timeout=10.0), "item one's observation never went in flight"
        # While the admitted observation is still in flight, the next item
        # must not start. The bound is generous to the unserialized code,
        # which starts item two almost at once; the serialized one holds item
        # one's remaining events at the admission lock instead.
        next_item_started_early = child_two_started.wait(timeout=2.0)
        release_observer.set()
        assert committed.wait(timeout=10.0), "the admitted observation never landed"
        assert entered.wait(timeout=10.0), "the batch never reached its second child"
        parent = status_of(root, job_id)
    finally:
        release_observer.set()
        release_probe.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive()
    assert not next_item_started_early, (
        "item two began before item one's admitted observation landed"
    )

    # The parent describes the child that is live, not the item whose
    # observation was in flight when that item ended.
    assert parent["stage"] == "local_vision", parent
    assert parent["stage_status"] == "running", parent

    final = status_of(root, job_id)
    assert final["status"] == "completed"
    assert final["result"]["processed_count"] == 2


def test_a_failed_item_still_counts_toward_the_batch_fraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (lead's finalization): a failed item's slot never advanced.

    With `continue_on_error` a failed item is a slot the batch has spent: it
    will never emit again. Counting only successful items left its fraction
    pinned at zero, so the second child's whole run mapped onto
    `(0 + child_fraction) / 2` and a poller watched the last item capped below
    half the batch. The slot moves in the item's `finally`, after forwarding
    stops, so the next item's events map onto the batch as it actually stands.
    """
    root = tmp_path / "output"
    job_id = "distill-failed-item"
    entered = threading.Event()
    release = threading.Event()
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "one.mp4").write_bytes(b"one")
    (videos / "two.mp4").write_bytes(b"two")

    def transcribe_that_fails_the_first_item(
        resolved_path: Path,
        _scratch_dir: Path,
        _options: Any,
        _progress: ProgressReporter,
        *,
        duration_sec: float | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if "one" in str(resolved_path):
            raise DistillError("E_CHILD_ONE", "transcript", "the first item fails on purpose")
        return {"language": "en", "language_probability": 1.0, "segments": ()}, []

    def blocking_first_probe(config: LocalVisionConfig) -> LocalVisionProbe:
        # The first item fails before vision, so this call is the second
        # child's - the run the batch is left waiting on.
        entered.set()
        if not release.wait(timeout=10.0):
            raise AssertionError("the poller never released the second child")
        return LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_available",
            message="available",
            detail={},
        )

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        transcribe=transcribe_that_fails_the_first_item,
        probe=blocking_first_probe,
        try_interpret=a_fake_interpretation,
    )
    stub_each_source_uniquely(monkeypatch)

    failures: list[BaseException] = []
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["response"] = pipeline.process_video_directory(
                {
                    "path": str(videos),
                    "output_dir": str(root),
                    "job_id": job_id,
                    "ocr": False,
                }
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert entered.wait(timeout=10.0), "the batch never reached its second child"
        parent = status_of(root, job_id)
        child_one = status_of(root, f"{job_id}-1")
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive()

    # The first item is spent either way: its slot is gone, its record says
    # failed, and the batch's fraction says one item of two is behind us.
    assert child_one["status"] == "failed"
    assert child_one["error"]["code"] == "E_CHILD_ONE"
    assert parent["status"] == "running"
    assert parent["stage"] == "local_vision"
    assert 50.0 <= parent["overall_percent"] < 100.0, parent

    final = status_of(root, job_id)
    assert final["status"] == "completed"
    assert final["result"]["processed_count"] == 1
    assert final["result"]["error_count"] == 1
    assert status_of(root, f"{job_id}-2")["status"] == "completed"


@pytest.mark.parametrize(
    ("handler", "args"),
    [
        ("process_video_directory", {}),
        ("process_youtube_playlist", {"url": PLAYLIST_URL}),
    ],
    ids=["directory", "playlist"],
)
def test_a_delayed_event_from_the_current_child_cannot_regress_the_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    handler: str,
    args: dict[str, str],
) -> None:
    """FAILS FIRST (round 2, panel B 1): the parent mirrored a refused event.

    The child's own store refuses a note sequenced at or below the newest one
    it committed, but the observer used to forward that refused event anyway,
    and the parent's reporter handed it a fresh parent sequence - so while the
    child sat blocked in vision, a transcription callback delayed on the
    dispatcher thread pulled the parent's record back to an earlier stage and
    left it there for the rest of the wait. The observer now rejects the
    child's stale sequences under the admission lock, before the parent's
    sequence is spent on them. The delay parks before the child's print,
    outside every store lock, exactly where a slow callback sits; the child's
    transcription event rides a real `run_command` dispatcher thread.
    """
    root = tmp_path / "output"
    job_id = "distill-delayed-current-child"
    child_id = f"{job_id}-1"
    entered = threading.Event()
    release_probe = threading.Event()
    parked = threading.Event()
    release_delayed = threading.Event()
    delayed_done = threading.Event()
    regression_noted = threading.Event()
    genuine_print = pipeline.progress_emitter
    genuine_write = JobStore._write
    built: list[bool] = []
    parked_once: list[bool] = []

    def print_that_delays_the_childs_first_event(job: str) -> Any:
        print_event = genuine_print(job)
        built.append(True)
        if len(built) != 2:
            # The parent's emitter is built first, the child's second; the
            # delay lands between the event's construction (its sequence and
            # its own overall are already fixed) and its print, outside every
            # store lock - exactly where a slow callback sits.
            return print_event

        def delaying_print(event: Any) -> None:
            if event.mechanism == "transcription" and not parked_once:
                parked_once.append(True)
                parked.set()
                assert release_delayed.wait(timeout=10.0), "the test never released the callback"
                try:
                    print_event(event)
                finally:
                    delayed_done.set()
                return
            print_event(event)

        return delaying_print

    def regression_signalling_write(store: JobStore, record: Any) -> Any:
        written = genuine_write(store, record)
        payload = record.to_dict()
        if (
            payload.get("job_id") == job_id
            and payload.get("stage") == "transcription"
            and payload.get("status") == "running"
        ):
            # The mirror note panel B's counterexample lands: signalled after
            # the write committed, so a read that follows observes it.
            regression_noted.set()
        return written

    monkeypatch.setattr(pipeline, "progress_emitter", print_that_delays_the_childs_first_event)
    monkeypatch.setattr(JobStore, "_write", regression_signalling_write)

    def blocking_probe(config: LocalVisionConfig) -> LocalVisionProbe:
        entered.set()
        if not release_probe.wait(timeout=10.0):
            raise AssertionError("the poller never released the vision probe")
        return LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_available",
            message="available",
            detail={},
        )

    if handler == "process_video_directory":
        videos = tmp_path / "videos"
        videos.mkdir()
        (videos / "one.mp4").write_bytes(b"one")
        call_args: dict[str, Any] = {**args, "path": str(videos)}
        parent_root, child_root = root, root
    else:
        monkeypatch.setattr(pipeline, "youtube_playlist_urls", lambda *_a, **_k: [VIDEO_URL])
        call_args = dict(args)
        parent_root = root
        child_root = root / "playlists" / pipeline.playlist_folder_name(PLAYLIST_URL)

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        transcribe=a_real_process_reporting_progress,
        probe=blocking_probe,
        try_interpret=a_fake_interpretation,
    )
    stub_each_source_uniquely(monkeypatch)

    failures: list[BaseException] = []
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["response"] = getattr(pipeline, handler)(
                {**call_args, "output_dir": str(root), "job_id": job_id, "ocr": False}
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert parked.wait(timeout=10.0), "the child's first event never went in flight"
        assert entered.wait(timeout=10.0), "the child never reached its vision probe"
        parent_before = status_of(parent_root, job_id)
        child_before = status_of(child_root, child_id)
        release_delayed.set()
        assert delayed_done.wait(timeout=10.0), "the delayed print never returned"
        # Give the unserialized forward every chance to land its mirror note -
        # serialized, the observer rejects the stale sequence and this wait
        # simply expires.
        regression_noted.wait(timeout=2.0)
        parent_after = status_of(parent_root, job_id)
        child_after = status_of(child_root, child_id)
    finally:
        release_delayed.set()
        release_probe.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive(), "the batch did not finish after the probe was released"

    # Before the delayed event lands, both records name the blocked stage.
    assert parent_before["stage"] == "local_vision", parent_before
    assert child_before["stage"] == "local_vision", child_before

    # After it lands: the child's store already refused the stale sequence, so
    # its record never moved; the parent must not have moved either.
    assert parent_after["stage"] == "local_vision", parent_after
    assert parent_after["stage_status"] == "running", parent_after
    assert parent_after["overall_percent"] >= parent_before["overall_percent"], parent_after
    assert child_after["stage"] == "local_vision", child_after

    final = status_of(parent_root, job_id)
    assert final["status"] == "completed"
    assert final["result"]["processed_count"] == 1


def test_only_the_run_holding_a_job_may_note_its_progress(tmp_path: Path) -> None:
    """A note is a write to a record, so it answers to the same rule as `finish`.

    Without the refusal, a second store could fold its own progress into a live
    holder's record while that holder carried on working - two runs' stages
    alternating in one record, which describes neither run. The refusal is the
    one `finish` already makes; what is new is only the thing being refused.
    """
    root = tmp_path / "output"
    holder = JobStore.open(root)
    holder.start("distill-noted", "process_local_video")
    progress = JobProgress(
        stage="local_vision",
        stage_status="running",
        stage_percent=50.0,
        overall_percent=42.0,
        observed_at="2026-10-27T09:00:00Z",
    )

    with pytest.raises(DistillError) as raised:
        JobStore.open(root).note_progress("distill-noted", progress)
    assert raised.value.code == "E_JOB_NOT_HELD"

    holder.note_progress("distill-noted", progress)
    status = status_of(root, "distill-noted")
    assert status["status"] == "running"
    assert status["stage"] == "local_vision"
    assert status["stage_percent"] == 50.0
    assert status["last_progress_at"] == "2026-10-27T09:00:00Z"


def test_a_killed_run_leaves_its_last_stage_in_the_abandoned_answer(
    tmp_path: Path,
) -> None:
    """`abandoned` is reported, not written - so it reports all that was read.

    The record the dead holder left is the only evidence of what it was doing,
    and `read`'s abandoned answer is the last thing anybody will ever say about
    that run. Rebuilding the record field by field here would have dropped the
    stage the run died in - the one fact the issue's caller was missing.
    """
    root = tmp_path / "output"
    root.mkdir(parents=True)

    killed = subprocess.run(
        [sys.executable, "-c", KILLED_MID_RUN, str(root), "distill-killed-mid-stage"],
        capture_output=True,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr.decode()

    status = status_of(root, "distill-killed-mid-stage")
    assert status["status"] == "abandoned"
    assert status["stage"] == "local_vision"
    assert status["stage_status"] == "running"
    assert status["stage_percent"] == 50.0
    assert status["last_progress_at"] == "2026-10-27T09:00:00Z"
    assert status["started_at"]


def test_a_reused_job_id_starts_over_rather_than_inheriting_the_previous_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reused identifier is a new run, and its record says so from the start.

    Identifiers are reused, and the previous run's record - stage and all - was
    still standing when this one started. A start that carried the old stage
    over would have a poller told the new run was in the old run's stage, with
    the old run's last-activity time, before it had done anything at all.
    """
    root = tmp_path / "output"

    def first_run(progress: ProgressReporter | None) -> None:
        assert progress is not None
        progress.update("local_vision", percent=50.0)

    run(
        monkeypatch,
        root,
        "distill-recycled",
        outcome={"manifest_path": "/first/manifest.json"},
        during=first_run,
    )
    assert status_of(root, "distill-recycled")["stage"] == "local_vision"

    observed: dict[str, Any] = {}

    def second_run(progress: ProgressReporter | None) -> None:
        assert progress is not None
        observed.update(status_of(root, "distill-recycled"))
        progress.update("transcription", status="running", percent=25.0)
        observed["after"] = status_of(root, "distill-recycled")

    with pytest.raises(DistillError):
        run(
            monkeypatch,
            root,
            "distill-recycled",
            outcome=STAGE_FAILURE,
            during=second_run,
        )

    # The fresh start wrote a record with nothing inherited: no stage until the
    # new run emitted one, and then the new run's own stage.
    assert "stage" not in observed
    assert observed["after"]["stage"] == "transcription"
    assert observed["after"]["stage_percent"] == 25.0

    final = status_of(root, "distill-recycled")
    assert final["status"] == "failed"
    assert final["error"]["code"] == "E_STAGE_BOOM"
    assert final["stage"] == "transcription"
    assert "result" not in final


def test_a_note_that_cannot_be_written_warns_once_and_never_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Observation is not load-bearing: a run survives its own progress notes.

    A record store that cannot take a note is degraded, not broken - the run's
    result and its terminal record are the things a caller is owed. The first
    failure says so once with the cause attached; the next nine hundred events
    say nothing, because a store that cannot write one note cannot write the
    rest either, and one warning per event is the flood the notes are throttled
    to avoid.
    """
    root = tmp_path / "output"

    def refuse_notes(_store: Any, _job_id: str, _progress: Any) -> None:
        raise OSError("the record directory vanished")

    monkeypatch.setattr(JobStore, "note_progress", refuse_notes)
    caplog.set_level(logging.WARNING, logger=pipeline.LOGGER.name)

    def emit_a_burst(progress: ProgressReporter | None) -> None:
        assert progress is not None
        for percent in (10.0, 20.0, 30.0, 40.0, 50.0):
            progress.update("transcription", percent=percent)

    result = run(
        monkeypatch,
        root,
        "distill-silent-notes",
        outcome={"manifest_path": "/x/manifest.json"},
        during=emit_a_burst,
    )

    assert result["manifest_path"] == "/x/manifest.json"
    warnings = [
        record
        for record in caplog.records
        if record.name == pipeline.LOGGER.name and "job progress note" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "distill-silent-notes" in warnings[0].getMessage()

    status = status_of(root, "distill-silent-notes")
    assert status["status"] == "completed"
    assert status["result"]["manifest_path"] == "/x/manifest.json"


def test_same_stage_events_note_the_first_and_a_transition_writes_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The throttle, with a clock that does not move: burst, then transition.

    A frozen monotonic clock makes the throttle's behaviour deterministic - no
    wall-clock assertion, no sleeping - so what this pins is the rule itself:
    within one stage, one note stands for the burst; the stage's completion is
    a transition, and a transition is written the moment it happens, whatever
    the clock says.
    """
    root = tmp_path / "output"
    monkeypatch.setattr(pipeline, "_monotonic", lambda: 0.0)
    observed: dict[str, Any] = {}

    def burst_and_finish(progress: ProgressReporter | None) -> None:
        assert progress is not None
        for percent in (10.0, 20.0, 30.0):
            progress.update("transcription", percent=percent)
        observed["burst"] = status_of(root, "distill-throttled")
        progress.complete("transcription")
        observed["complete"] = status_of(root, "distill-throttled")

    run(
        monkeypatch,
        root,
        "distill-throttled",
        outcome={"manifest_path": "/x/manifest.json"},
        during=burst_and_finish,
    )

    # One note stood for the whole same-stage burst: the first event's percent,
    # not the last's, and `running` - the events between were throttled away.
    assert observed["burst"]["status"] == "running"
    assert observed["burst"]["stage"] == "transcription"
    assert observed["burst"]["stage_percent"] == 10.0
    # The completion did not wait out the interval: a stage that ended is
    # reported ended at once, or a poller would watch a finished stage "still
    # running" for up to the throttle window.
    assert observed["complete"]["stage"] == "transcription"
    assert observed["complete"]["stage_status"] == "completed"
    assert observed["complete"]["stage_percent"] == 100.0


@pytest.mark.parametrize(
    ("handler", "args"),
    [
        ("process_youtube_video", {"url": VIDEO_URL}),
        ("process_x_video", {"url": "https://x.com/example/status/1234567890"}),
    ],
    ids=["youtube", "x"],
)
def test_remote_source_handlers_persist_their_progress_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, handler: str, args: dict[str, str]
) -> None:
    """The download half is where a remote run spends most of its wall clock.

    A YouTube or X run is acquisition for most of its life, so a record that
    only learned about stages after acquisition would be silent for exactly the
    window a poller is most likely to be asking in. Both handlers run through
    their own real stages here - only the external tools are injected - and the
    completed record carries the projection their own runs emitted.
    """
    root = tmp_path / "output"
    job_id = f"distill-{handler}"

    fake_stage_producers(monkeypatch)
    configure_run(monkeypatch, probe=an_available_probe, try_interpret=a_fake_interpretation)
    stub_acquisition(monkeypatch, provenance=a_provenance())

    getattr(pipeline, handler)({**args, "output_dir": str(root), "job_id": job_id, "ocr": False})

    status = status_of(root, job_id)
    assert status["status"] == "completed"
    assert status["stage"] == "bundle_publish"
    assert status["stage_status"] == "completed"
    assert status["started_at"]
    assert status["last_progress_at"]


# --- A note in flight cannot take the outcome away (review round 1) ---------
#
# Progress callbacks run on the dispatcher threads of the tools a run drives,
# and the supervisor that owns those threads shuts them down on a bounded
# window rather than waiting for the callback: a note whose filesystem write is
# slow is still "in flight" when the run moves on to `finish`. Without
# serialization, that note's atomic replace lands after the terminal record and
# erases it - a run that succeeded reads `abandoned`, result and all gone. The
# ownership check has already passed by then, so checking again cannot fix it:
# the whole check/write/release lifecycle of a record has to be one serialized
# thing, and a note sequenced before the newest one already committed must not
# overwrite it.


def a_real_process_reporting_progress(
    _resolved_path: Path,
    _scratch_dir: Path,
    _options: Any,
    progress: ProgressReporter,
    *,
    duration_sec: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """A transcription stage that reports through a real `run_command` child.

    The child is a Python one-liner, not ffmpeg: what is under test is the
    delivery path - a real subprocess, a real dispatcher thread, a callback on
    that thread folding an event into the job record while the run's own thread
    carries on. That is the thread the production callbacks run on, reached
    without a network, a model or an installed tool.
    """

    def on_stderr_line(_line: str) -> None:
        progress.update("transcription", percent=25.0)

    run_command.stream(
        [sys.executable, "-c", 'import sys; print("fixture progress", file=sys.stderr)'],
        stage="transcript",
        total_timeout_sec=30.0,
        idle_timeout_sec=30.0,
        on_stderr_line=on_stderr_line,
        check=False,
    )
    return {"language": "en", "language_probability": 1.0, "segments": ()}, []


def test_a_note_in_flight_cannot_erase_the_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (review round 1): a slow note write beat the terminal record.

    The record write is paused after the note's ownership check has passed -
    exactly where a slow filesystem leaves one - while the run finishes around
    it: the supervisor's dispatcher shutdown is bounded and does not wait for
    the callback, by design. Unserialized, the paused note commits last and
    replaces the `completed` record with a stale `running` one, so a poller is
    told `abandoned` and the result is gone. Serialized, the terminal write
    waits for the in-flight note (one authoritative owner), and the outcome
    persists whichever write began first.
    """
    root = tmp_path / "output"
    job_id = "distill-delayed-note"
    genuine_write = JobStore._write
    parked = threading.Event()
    release = threading.Event()
    committed = threading.Event()
    parked_once: list[bool] = []

    def pausing_write(store: JobStore, record: Any) -> Any:
        payload = record.to_dict()
        if (
            not parked_once
            and payload.get("status") == "running"
            and payload.get("stage") == "transcription"
        ):
            # The one admitted note, held between its check and its commit.
            parked_once.append(True)
            parked.set()
            assert release.wait(timeout=10.0), "the test never released the parked note"
            try:
                return genuine_write(store, record)
            finally:
                committed.set()
        return genuine_write(store, record)

    monkeypatch.setattr(JobStore, "_write", pausing_write)
    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        transcribe=a_real_process_reporting_progress,
        probe=an_available_probe,
        try_interpret=a_fake_interpretation,
    )
    stub_acquisition(monkeypatch, provenance=a_provenance())

    result: dict[str, Any] = {}
    failures: list[BaseException] = []
    handler_returned = threading.Event()

    def work() -> None:
        try:
            result["response"] = pipeline.process_local_video(
                {
                    "path": "/stub/video.mp4",
                    "output_dir": str(root),
                    "job_id": job_id,
                    "ocr": False,
                }
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            handler_returned.set()

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert parked.wait(timeout=10.0), "the run never delivered a progress note"
        # The whole claim under test is that the run finished while the note was
        # still in flight. Bounded, because under serialization it never can:
        # the wait expires and the release below is what lets the run proceed.
        # Unserialized the run reaches this in about the dispatcher's one-second
        # shutdown grace, so the bound is generous to it on purpose.
        finished_around_the_note = handler_returned.wait(timeout=3.0)
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive(), "the run did not finish after the note was released"
    assert committed.wait(timeout=10.0), "the parked note never committed"

    assert not finished_around_the_note, "the run closed its record around a note in flight"
    final = status_of(root, job_id)
    assert final["status"] == "completed", final
    assert final["result"]["manifest_path"] == result["response"]["manifest_path"]
    assert final["result"]["manifest_path"]


def test_a_stale_sequenced_note_cannot_regress_the_newer_observed_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (review round 1): a delayed callback answered last, and won.

    Two events from one run can be delivered out of order when the earlier one
    rides a slower thread. The note carrier now carries the event's sequence,
    and the store ignores a note sequenced at or before the newest one it has
    committed, so the older event cannot overwrite the newer stage a poller has
    already been shown.
    """
    root = tmp_path / "output"
    job_id = "distill-stale-sequence"
    observed: dict[str, Any] = {}

    def ocr_with_a_delayed_callback(
        frames: list[FrameArtifact],
        _language: str,
        _enabled: bool,
        progress: ProgressReporter,
        _preprocess: bool = True,
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        assert progress is not None
        early = progress.update("ocr", percent=10.0)
        progress.complete("ocr")
        # The sink is the handler's own composed emitter: this is the exact call
        # a delayed dispatcher callback would make with the older event.
        assert progress.emitter is not None
        progress.emitter(early)
        observed.update(status_of(root, job_id))
        return frames, []

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        ocr_frames=ocr_with_a_delayed_callback,
        probe=an_available_probe,
        try_interpret=a_fake_interpretation,
    )
    stub_acquisition(monkeypatch, provenance=a_provenance())

    pipeline.process_local_video(
        {"path": "/stub/video.mp4", "output_dir": str(root), "job_id": job_id, "ocr": False}
    )

    assert observed["stage"] == "ocr"
    assert observed["stage_status"] == "completed", observed
    assert observed["stage_percent"] == 100.0, observed

    final = status_of(root, job_id)
    assert final["status"] == "completed"


def test_a_fast_failing_frame_does_not_silence_the_frames_that_succeeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (round 2, panel A 2): one fast failure withheld every success.

    `future.result()` re-raises a worker's exception, so the first failing
    frame to complete used to abort the consumption loop before any other
    future was yielded: the frames that had already answered were left
    unreported, and the record stood at the stage-entry note for the rest of
    the wait - the same silence the issue describes, reached from the failure
    side. The pool is now drained on failure: successes still count, and the
    first failure is re-raised once the executor has finished, so the run ends
    in the same fatal error rather than silently degrading.
    """
    root = tmp_path / "output"
    job_id = "distill-fast-failing-frame"
    entered = threading.Event()
    release = threading.Event()
    frame_two_failed = threading.Event()
    milestone_noted = threading.Event()
    genuine_write = JobStore._write

    def milestone_signalling_write(store: JobStore, record: Any) -> Any:
        written = genuine_write(store, record)
        payload = record.to_dict()
        if (
            payload.get("status") == "running"
            and payload.get("stage") == "local_vision"
            and payload.get("stage_percent") == 33.333
        ):
            milestone_noted.set()
        return written

    monkeypatch.setattr(JobStore, "_write", milestone_signalling_write)

    def interpret(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        if image_path.name == "frame_0002.png":
            frame_two_failed.set()
            raise DistillError("E_FRAME_TWO", "local_vision", "the second frame fails immediately")
        if image_path.name == "frame_0001.png":
            # Succeeds only once the failure has happened, so the failing
            # future is already complete when this answer lands.
            assert frame_two_failed.wait(timeout=10.0), "frame two never failed"
            return an_interpretation(config, image_path, prompt_profile), None
        entered.set()
        assert release.wait(timeout=10.0), "the poller never released the third frame"
        return an_interpretation(config, image_path, prompt_profile), None

    fake_stage_producers(monkeypatch, frame_count=3)
    configure_run(
        monkeypatch,
        probe=an_available_probe,
        try_interpret=interpret,
        max_parallel=3,
    )
    stub_acquisition(monkeypatch, provenance=a_provenance())

    failures: list[BaseException] = []

    def work() -> None:
        try:
            pipeline.process_local_video(
                {
                    "path": "/stub/video.mp4",
                    "output_dir": str(root),
                    "job_id": job_id,
                    "ocr": False,
                }
            )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert entered.wait(timeout=10.0), "the run never reached the third frame"
        # The successful frame's milestone must reach the record while the
        # third frame is still in flight. The bound is generous to the
        # aborting loop, which never writes it at all.
        assert milestone_noted.wait(timeout=2.0), "the successful frame's milestone never landed"
        status = status_of(root, job_id)
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not worker.is_alive(), "the run did not finish after the third frame was released"

    assert status["status"] == "running"
    assert status["stage"] == "local_vision"
    assert status["stage_percent"] == pytest.approx(33.333)
    assert status["last_progress_at"]

    # The run ends in the worker's own failure, not a degradation: the first
    # completion-order exception is re-raised once the executor has drained.
    assert len(failures) == 1
    assert isinstance(failures[0], DistillError)
    assert failures[0].code == "E_FRAME_TWO"

    final = status_of(root, job_id)
    assert final["status"] == "failed"
    assert final["error"]["code"] == "E_FRAME_TWO"


def test_a_blocked_first_frame_does_not_silence_finished_frames_in_parallel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (review round 1): the pool reported in frame order, not completion order.

    With `max_parallel > 1`, the pool was consumed through `map`, which yields
    in submission order: frame 1's event waited for frame 1 itself, so a hung
    first frame withheld the milestones of every frame that had already
    answered - the record stuck at the stage-entry note, which is the silence
    the issue describes, in the parallel path. Consumed in completion order,
    the percent a poller sees counts answered frames - 2 of 3 here - and the
    merged output stays in keyframe order when the block clears.
    """
    root = tmp_path / "output"
    job_id = "distill-parallel-blocked-frame"
    entered = threading.Event()
    release = threading.Event()
    second_answer_noted = threading.Event()
    genuine_write = JobStore._write

    def signposting_write(store: JobStore, record: Any) -> Any:
        written = genuine_write(store, record)
        payload = record.to_dict()
        if (
            payload.get("status") == "running"
            and payload.get("stage") == "local_vision"
            and isinstance(payload.get("stage_percent"), float)
            and payload["stage_percent"] >= 60.0
        ):
            # Two of three frames answered and the count reached the record -
            # signalled after the write committed, so a read that follows it
            # cannot race the pool, only observe it.
            second_answer_noted.set()
        return written

    monkeypatch.setattr(JobStore, "_write", signposting_write)

    def available_probe(config: LocalVisionConfig) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_available",
            message="available",
            detail={},
        )

    def interpret(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        if image_path.name == "frame_0001.png":
            entered.set()
            if not release.wait(timeout=10.0):
                raise AssertionError("the poller never released the first frame")
        return an_interpretation(config, image_path, prompt_profile), None

    fake_stage_producers(monkeypatch, frame_count=3)
    configure_run(
        monkeypatch,
        probe=available_probe,
        try_interpret=interpret,
        max_parallel=3,
    )
    stub_acquisition(monkeypatch, provenance=a_provenance())

    result: dict[str, Any] = {}
    failures: list[BaseException] = []

    def work() -> None:
        try:
            result["response"] = pipeline.process_local_video(
                {
                    "path": "/stub/video.mp4",
                    "output_dir": str(root),
                    "job_id": job_id,
                    "ocr": False,
                }
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert entered.wait(timeout=10.0), "the run never reached the first frame"
        assert second_answer_noted.wait(timeout=10.0), (
            "the finished frames never reached the record"
        )
        status = status_of(root, job_id)
    finally:
        release.set()
        worker.join(timeout=10.0)
    assert not failures
    assert not worker.is_alive(), "the run did not finish after the frame was released"

    # Frames 2 and 3 of 3 have answered while frame 1 is still in flight: a
    # poller sees the count, not the position of the hung frame.
    assert status["status"] == "running"
    assert status["stage"] == "local_vision"
    assert status["stage_percent"] == pytest.approx(66.667)
    assert status["last_progress_at"]

    # The merged output is unchanged by completion order: frames in keyframe
    # order, each carrying its own reading.
    final = status_of(root, job_id)
    assert final["status"] == "completed"
    frames = result["response"]["frames"]
    assert [frame["index"] for frame in frames] == [1, 2, 3]
    assert all(frame.get("visual_interpretation") for frame in frames)


# --- One run, one start time, observed at literal instants (review round 1) ---
#
# Timestamps are second-granular, so a test that reads two of them inside one
# second cannot tell a rewrite from a preservation: the mutant that stamps
# `started_at` with the finish clock survives it. These tests pin both clocks
# the record reads - the store's write clock and the reporter's event clock -
# to literals a full minute apart, so every assertion names an instant rather
# than a coincidence of timing.

PINNED_START = 1793029200.0
PINNED_EVENT = 1793029260.0
PINNED_FAILED_NOTE_EVENT = 1793029290.0
PINNED_FINISH = 1793029320.0


def pinned_at(epoch: dict[str, float]) -> str:
    """The fake store clock's reading, through the format a record uses."""
    return job_store.format_timestamp(epoch["epoch"])


def test_a_failed_note_leaves_the_carry_at_the_last_written_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The unpinned clause (round 2, tests lens 3): a failed note changes nothing.

    The hold is the write-side carry for the record's progress, and it moves
    only after the write succeeded - so a note whose write fails leaves the
    carry describing exactly what the record on disk says. A stage that emits
    a committed 10 percent, a 20 percent whose note write fails, and then
    nothing but its own fatal error, must leave a failed record still carrying
    the 10 percent and *that* event's time. The mutant that moves the hold
    update ahead of the write would publish the unwritten 20 percent and its
    later time, and dies here. Injection is at the write boundary only; both
    clocks are pinned so every expected value is a literal.
    """
    root = tmp_path / "output"
    job_id = "distill-failed-note"
    store_clock = {"epoch": PINNED_START}
    event_clock = {"epoch": PINNED_START}
    monotonic = {"now": 0.0}
    monkeypatch.setattr(job_store, "_timestamp", lambda: pinned_at(store_clock))
    monkeypatch.setattr(
        pipeline, "ProgressReporter", partial(ProgressReporter, clock=lambda: event_clock["epoch"])
    )
    monkeypatch.setattr(pipeline, "_monotonic", lambda: monotonic["now"])
    genuine_write = JobStore._write

    def refusing_write(store: JobStore, record: Any) -> Any:
        payload = record.to_dict()
        if (
            payload.get("status") == "running"
            and payload.get("stage") == "transcription"
            and payload.get("stage_percent") == 20.0
        ):
            raise OSError("the record directory vanished mid-note")
        return genuine_write(store, record)

    monkeypatch.setattr(JobStore, "_write", refusing_write)

    def transcribe_that_loses_its_second_note(
        _resolved_path: Path,
        _scratch_dir: Path,
        _options: Any,
        progress: ProgressReporter,
        *,
        duration_sec: float | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        event_clock["epoch"] = PINNED_EVENT
        progress.update("transcription", percent=10.0)
        # Past the throttle floor, so the second event's note is genuinely
        # attempted and its write is the one that fails.
        monotonic["now"] += 2.0
        event_clock["epoch"] = PINNED_FAILED_NOTE_EVENT
        progress.update("transcription", percent=20.0)
        # Fatal, immediately: no later event can repaint the record.
        raise DistillError("E_FATAL_AFTER_NOTE", "transcript", "the stage fails right after")

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        transcribe=transcribe_that_loses_its_second_note,
        probe=an_available_probe,
        try_interpret=a_fake_interpretation,
    )
    stub_acquisition(monkeypatch, provenance=a_provenance())

    with pytest.raises(DistillError) as failure:
        pipeline.process_local_video(
            {"path": "/stub/video.mp4", "output_dir": str(root), "job_id": job_id, "ocr": False}
        )
    assert failure.value.code == "E_FATAL_AFTER_NOTE"

    final = status_of(root, job_id)
    assert final["status"] == "failed"
    assert final["error"]["code"] == "E_FATAL_AFTER_NOTE"
    # The committed event, whole: its percent and its own minute, not the
    # failed note's later time.
    assert final["stage"] == "transcription"
    assert final["stage_percent"] == 10.0
    assert final["last_progress_at"] == "2026-10-26T15:41:00Z"
    assert final["started_at"] == "2026-10-26T15:40:00Z"


def test_start_time_is_immutable_and_progress_carries_the_events_own_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILS FIRST (review round 1): nothing pinned `started_at` against a rewrite.

    The start is one fact, set when the record opens; the last progress event
    is another, carrying the event's own time rather than the write's. A finish
    that stamps `started_at` from its own clock - the surviving mutant - reads
    as a run that began when it ended, and a note written in the same second as
    the start makes `updated_at == started_at` true twice, so agreement proves
    nothing. Both clocks are pinned, every expected time is a literal, and the
    finish lands two minutes after the start.
    """
    root = tmp_path / "output"
    job_id = "distill-pinned-clocks"
    store_clock = {"epoch": PINNED_START}
    event_clock = {"epoch": PINNED_START}
    monkeypatch.setattr(job_store, "_timestamp", lambda: pinned_at(store_clock))
    monkeypatch.setattr(
        pipeline, "ProgressReporter", partial(ProgressReporter, clock=lambda: event_clock["epoch"])
    )
    observed: dict[str, Any] = {}

    def ocr_at_pinned_times(
        frames: list[FrameArtifact],
        _language: str,
        _enabled: bool,
        progress: ProgressReporter,
        _preprocess: bool = True,
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        assert progress is not None
        # The event happens a minute after the start; its note is written in
        # the start's own second, so `updated_at` agrees with `started_at`
        # without being the same reading.
        event_clock["epoch"] = PINNED_EVENT
        progress.update("ocr", percent=40.0)
        observed["running"] = status_of(root, job_id)
        # The finish stamp comes later; every write from here on carries it.
        store_clock["epoch"] = PINNED_FINISH
        return frames, []

    fake_stage_producers(monkeypatch)
    configure_run(
        monkeypatch,
        ocr_frames=ocr_at_pinned_times,
        probe=an_available_probe,
        try_interpret=a_fake_interpretation,
    )
    stub_acquisition(monkeypatch, provenance=a_provenance())

    pipeline.process_local_video(
        {"path": "/stub/video.mp4", "output_dir": str(root), "job_id": job_id, "ocr": False}
    )

    # While running: the start is the pinned start, the last progress time is
    # the event's own minute (not the write's second), and `updated_at` agrees
    # with `started_at` only because both were pinned to the same second.
    assert observed["running"]["started_at"] == "2026-10-26T15:40:00Z"
    assert observed["running"]["updated_at"] == "2026-10-26T15:40:00Z"
    assert observed["running"]["last_progress_at"] == "2026-10-26T15:41:00Z"

    # After a finish stamped two minutes later, the start is still the start.
    final = status_of(root, job_id)
    assert final["status"] == "completed"
    assert final["started_at"] == "2026-10-26T15:40:00Z"
    assert final["updated_at"] == "2026-10-26T15:42:00Z"
