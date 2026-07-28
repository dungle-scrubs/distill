"""Job records: the durable answer to "what happened to the run I started?".

This module owns the job record - its identifier domain, its lifecycle, where it
lives on disk, and what a reader is told about one. A **job** is one invocation
of a Distill tool. It is not keyed by **bundle key** and never can be: a batch or
playlist job spans many bundles, and a bundle outlives the job that produced it
(D-027). That is why job records live beside the bundles under `_jobs/` rather
than inside any of them, and why this store is separate from `BundleStore`.

The invariant it exists to hold: *a run's outcome is recorded whether it
succeeded or failed*. Before this module the record was written in one place -
after the work returned - so a run that raised left nothing at all, and a caller
polling could not tell a failed job from a job that never started (finding 12).
Worse, identifiers get reused, so a failure silently left the previous run's
`completed` record standing, result and all. The lifecycle is now two steps:
`start` records `running` before any work begins, and `finish` replaces it with a
terminal status on success *or* failure (R-17).

`running` is therefore a state a hard kill can leave behind, and a leftover
`running` record must not read as a job still in progress. Liveness is answered
by the same mechanism `BundleStore` uses for a run - an `ExclusiveLock` the
kernel releases when the holder's descriptors close - so a record whose holder is
gone is reported as `abandoned`, and a record whose holder is alive is reported
as `running`. Neither answer is a guess about elapsed time, and `abandoned` is
reported rather than stored: nobody is left to write it.

Identifiers are validated against a bounded domain and rejected outside it,
never sanitized (R-18). Sanitizing mapped every unusable character onto `_`, so
`job/../escape` and `job____escape` named one file: two distinct jobs, one
record, and the second run's outcome reported as the first's.

Records are written by atomic replace, because they are state another process
reads while the run that owns them is still writing (R-14, D-033). A record
caught half-written is unparseable, which a poller cannot tell apart from a job
that never started.

Writing a record is a thing only the run that owns the identifier may do, and
owning it means holding its lock. `start` refuses an identifier any holder is
live on - including this store, which used to be waved through - and `finish`
refuses one this store does not hold. Without both, two runs can share one
record from opposite ends: two `start`s writing over each other's `running`, or
a stranger's `finish` replacing a live holder's record with a terminal status
while that holder is still working. There is no way for a process to close out a
record it did not open, deliberately: `abandoned` already answers "the holder is
gone" and is reported rather than stored, so nothing is lost by refusing.

What this module does not own: the durable-write mechanism and the lock
(`bundle_store`), anything keyed by a **bundle key** - including cache-hit
manifest patching, which is `BundleStore.patch_published` on a published
snapshot - and what a run computes. It stores the result a run reports; it does
not interpret it. It does not own where a run's envelope goes either: `pipeline`
opens one per tool call, and this module only enforces that a second one under
the same identifier cannot.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .bundle_store import (
    ExclusiveLock,
    LockState,
    _bundle_log,
    atomic_write_text,
    confined_path,
    ensure_safe_directory,
)
from .errors import DistillError

JOB_DIR_NAME = "_jobs"
"""Where job records live: beside the bundles, under the output root.

Reserved by prune, which skips underscore-prefixed directories, so a record is
never mistaken for a **bundle** and a bundle's retention never reclaims one.
"""

RECORD_SUFFIX = ".json"
LOCK_SUFFIX = ".lock"

JOB_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
"""The bounded domain a job identifier must fall in, matched in full (R-18).

A job identifier becomes a filename, so an unbounded one is a path:
`job/../escape` reaches out of the record directory entirely. It is rejected
rather than repaired, on the same grounds as `STAGE_NAME_RE`: a repair is a
many-to-one mapping, and the two identifiers it maps together get one record
between them. The first character is alphanumeric so no identifier can be `.`,
`..`, or a name the record directory's own conventions reserve.

Lower case for the same reason, not for tidiness. macOS and Windows fold case in
filenames, so `Job-1` and `job-1` are two identifiers naming one record - the
collision sanitizing used to produce, reached through the filesystem instead. A
domain that cannot contain both cannot collide on any filesystem, and the
uppercase form is refused rather than folded, because folding *is* the
many-to-one mapping.

128 characters covers what Distill itself generates - `distill-<32 hex>`, plus a
`-<index>` suffix per batch item - with room for a caller's own scheme.
"""

RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
ABANDONED = "abandoned"

StoredStatus = Literal["running", "completed", "failed"]
"""What a record can say on disk. `abandoned` is not here: it is a verdict about
a record nobody is left to write."""

ReportedStatus = Literal["running", "completed", "failed", "abandoned"]

STORED_STATUSES = frozenset({RUNNING, COMPLETED, FAILED})
TERMINAL_STATUSES = frozenset({COMPLETED, FAILED})

JOB_STAGE = "job"

LIVENESS_ROUNDS = 2
"""How many times `read` asks before it calls a record abandoned.

Each round reads the record and *then* probes for a holder, never the other way
round: the verdict has to rest on a record read before the probe that found
nobody, or a run started in between is reported as the dead one it replaced.

One round cannot answer, because a holder found gone may have been finishing
rather than dying - `finish` writes the terminal record before releasing - and
only a second read sees what it wrote. Two rounds end on a probe that came after
the record it judges, which is the only order the verdict is sound in. A third
would answer a different question: whether the identifier was started again
after we looked, which is not something a snapshot owes anyone.
"""

HOLDER_PRESENT_STATES: frozenset[LockState] = frozenset({"live", "unknown"})
"""The lock states `read` reports as a run still in progress.

`live` is a holder. `unknown` is a lock file this process could not open at all,
which is not the same as one nobody holds - prune reads it the same way, because
"may not ask" must never resolve to "nobody is running". Reported as `running`
it costs a poller another look; reported as `abandoned` it tells a caller its
live run is dead.
"""


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_job_id(job_id: str) -> str:
    """Return `job_id` if it is in the domain, and refuse it if it is not."""
    if not JOB_ID_RE.fullmatch(job_id):
        raise DistillError(
            "E_BAD_JOB_ID",
            JOB_STAGE,
            "job_id must be lower-case alphanumeric with - or _, 128 characters or fewer",
            {"job_id": job_id},
        )
    return job_id


@dataclass(frozen=True)
class JobOutcome:
    """How a run ended: one terminal status and the evidence for it.

    A carrier rather than two arguments, so `finish` cannot be handed a status
    that says `failed` with a result attached, or `completed` with an error.
    """

    status: Literal["completed", "failed"]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Hold that guarantee at runtime, because a type is not a check.

        The annotation constrains what a checker will accept, not what arrives.
        `finish` releases the lock as it writes, so an outcome carrying
        `running` would leave a record no live holder backs and nobody is coming
        back to replace - reported `abandoned` for a run that in fact succeeded.
        """
        if self.status not in TERMINAL_STATUSES:
            raise DistillError(
                "E_BAD_JOB_OUTCOME",
                JOB_STAGE,
                "a job outcome must be terminal",
                {"status": str(self.status)},
            )
        contradiction = self.error if self.status == COMPLETED else self.result
        if contradiction is not None:
            raise DistillError(
                "E_BAD_JOB_OUTCOME",
                JOB_STAGE,
                "a job outcome carries the evidence its status calls for and no other",
                {"status": self.status},
            )

    @classmethod
    def success(cls, result: dict[str, Any]) -> JobOutcome:
        return cls(status=COMPLETED, result=result)

    @classmethod
    def failure(cls, error: BaseException | dict[str, Any]) -> JobOutcome:
        """A failure, from whatever ended the run.

        Takes the exception itself so no call site has to remember the payload
        shape - the failure path is the one a caller is least likely to have
        exercised, and the whole point of R-17 is that it is not the path that
        gets skipped.
        """
        if isinstance(error, DistillError):
            payload: dict[str, Any] = {
                "code": error.code,
                "stage": error.stage,
                "message": error.message,
            }
        elif isinstance(error, BaseException):
            payload = {"code": "E_INTERNAL", "stage": "internal", "message": str(error)}
        else:
            payload = dict(error)
        return cls(status=FAILED, error=payload)


@dataclass(frozen=True)
class JobRecord:
    """One job's state as a reader sees it.

    `status` is what the reader is told, which is not always what is on disk: a
    `running` record whose holder is gone is reported `abandoned`.
    """

    job_id: str
    tool: str
    status: ReportedStatus
    updated_at: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @property
    def is_running(self) -> bool:
        return self.status == RUNNING

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "tool": self.tool,
            "updated_at": self.updated_at,
        }
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class JobHold:
    """One live claim this store has on an identifier: the lock, and the tool.

    The two are one fact, not two, so they are kept as one. Held separately they
    can disagree, and the disagreement is a record closed out under a tool no
    run ever started: `finish` is told the outcome, never the tool, because the
    tool is not something an ending run gets to change.
    """

    lock: ExclusiveLock
    tool: str


@dataclass
class JobStore:
    """The job records under one output root, and the locks proving who is live.

    An instance holds the locks for the jobs *it* started, and those are the
    only jobs it may finish. A store that never started a job can still read
    every record under the root: reading is a question about the filesystem, not
    about this process, and it takes nothing to ask.
    """

    root: Path
    _held: dict[str, JobHold] = field(default_factory=dict)

    @classmethod
    def open(cls, root: Path) -> JobStore:
        return cls(root=root)

    def start(self, job_id: str, tool: str) -> JobRecord:
        """Record `running` and take the liveness lock, before any work begins.

        Writing the record before the work is the whole of R-17: written after,
        it can only ever describe a run that got that far. The prior record for
        this identifier is replaced here rather than left to `finish`, so a run
        that is killed outright cannot leave the *previous* run's `completed`
        standing as though it described this one.
        """
        validate_job_id(job_id)
        self._take_hold(job_id, tool)
        try:
            return self._write(
                JobRecord(job_id=job_id, tool=tool, status=RUNNING, updated_at=_timestamp())
            )
        except BaseException as failure:
            # A hold whose record was never written names a run that is not
            # happening. Distill's tools run inside a long-lived session, so a
            # descriptor kept past that point is an identifier nothing can ever
            # start again: one failed write, then `E_JOB_RUNNING` forever.
            self._release_without_replacing(job_id, during=failure)
            raise

    def finish(self, job_id: str, outcome: JobOutcome) -> JobRecord:
        """Replace this store's `running` record with a terminal one, releasing the lock.

        Called on the success path and the failure path alike; a run that ends
        without reaching here is what `abandoned` describes.

        Only the holder may call it. A store that holds nothing here is not the
        run whose record this is: it is either a process that never started the
        job - whose terminal status would overwrite a live holder's record while
        that holder carries on working - or this store finishing twice. Both
        write an outcome for a run that did not end, so both are refused.
        """
        validate_job_id(job_id)
        hold = self._held.get(job_id)
        if hold is None:
            raise DistillError(
                "E_JOB_NOT_HELD",
                JOB_STAGE,
                "only the run holding a job_id may record its outcome",
                {"job_id": job_id},
            )
        try:
            record = self._write(
                JobRecord(
                    job_id=job_id,
                    tool=hold.tool,
                    status=outcome.status,
                    updated_at=_timestamp(),
                    result=outcome.result,
                    error=outcome.error,
                )
            )
        except BaseException as failure:
            # Released whether or not the record could be written: the run is
            # over either way, and a hold outliving it is a lock nothing will
            # come back to give up. Never in a bare `finally`, because a release
            # that raises there takes the failing write's place.
            self._release_without_replacing(job_id, during=failure)
            raise
        # Nothing in flight, so a release failure is raised normally: it is the
        # only report that a hold was not given up.
        self._release(job_id)
        return record

    def read(self, job_id: str) -> JobRecord | None:
        """The record for `job_id`, or `None` if there is none.

        Creates nothing: a read that has to make a directory to answer "is there
        a record?" has already answered "no" by making it.

        `running` is the only answer that costs anything to give, because it is
        the one a dead holder leaves behind. It is settled over `LIVENESS_ROUNDS`
        of read-then-probe rather than in one look; the constant carries why.
        """
        validate_job_id(job_id)
        path = self.record_path(job_id)
        record: JobRecord | None = None
        for _round in range(LIVENESS_ROUNDS):
            if not path.is_file():
                return None
            record = self._parse(path.read_text(), job_id)
            if not record.is_running or self._holder_is_live(job_id):
                return record
        if record is None:
            return None
        return JobRecord(
            job_id=record.job_id,
            tool=record.tool,
            status=ABANDONED,
            updated_at=record.updated_at,
            result=record.result,
            error=record.error,
        )

    def record_path(self, job_id: str) -> Path:
        """Where `job_id`'s record lives. Validates and confines; creates nothing."""
        validate_job_id(job_id)
        return confined_path(
            self.root / JOB_DIR_NAME / f"{job_id}{RECORD_SUFFIX}",
            self.root,
        )

    def lock_path(self, job_id: str) -> Path:
        """Where `job_id`'s liveness lock lives. Validates and confines; creates nothing."""
        validate_job_id(job_id)
        return confined_path(self.root / JOB_DIR_NAME / f"{job_id}{LOCK_SUFFIX}", self.root)

    def _record_directory(self) -> Path:
        """The record directory, created on demand and refusing a symlink at `_jobs`.

        `_jobs` is a component below the root the writes are confined to, not the
        confinement root itself: a root is only ever compared against, never
        inspected, so passing `_jobs` as the root would leave the one component an
        attacker can pre-create as a link unchecked (R-16).
        """
        # The confinement root is created rather than validated: `ensure_safe_directory`
        # compares against a root, so somebody has to make it, and the record store's
        # own root is the caller's to choose.
        self.root.mkdir(parents=True, exist_ok=True)
        return ensure_safe_directory(self.root / JOB_DIR_NAME, self.root)

    def _write(self, record: JobRecord) -> JobRecord:
        self._record_directory()
        atomic_write_text(
            self.record_path(record.job_id),
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
            root=self.root,
        )
        return record

    def _take_hold(self, job_id: str, tool: str) -> None:
        """Claim liveness for `job_id`, refusing a job any run is live on.

        Two runs sharing one identifier would share one record, which is the
        collision R-18 exists to stop - reached by agreement rather than by
        sanitizing, but the same single record describing two runs.

        *Any* run includes this one. A store that already holds the identifier
        is not a special case to wave through: waved through, it wrote a second
        `running` record over the first run's and the two ran on under one
        record, which is the whole of what this refuses. The kernel gives the
        same answer either way - `flock` is per open file description, so a
        second `take` against a path is refused inside the holding process
        exactly as it is outside - so there is nothing here to special-case
        with.
        """
        self._record_directory()
        path = self.lock_path(job_id)
        ensure_safe_directory(path, self.root, create_leaf=False)
        lock = ExclusiveLock.take(job_id, path, stage=JOB_STAGE)
        if lock is None:
            raise DistillError(
                "E_JOB_RUNNING",
                JOB_STAGE,
                "another run is live under this job_id",
                {"job_id": job_id},
            )
        self._held[job_id] = JobHold(lock=lock, tool=tool)

    def _release(self, job_id: str) -> None:
        hold = self._held.pop(job_id, None)
        if hold is not None:
            hold.lock.release()

    def _release_without_replacing(self, job_id: str, *, during: BaseException) -> None:
        """Release while `during` is travelling, and never take its place.

        The same rule `BundleRun._release_without_replacing` holds, for the same
        reason: cleanup that raises during an exception *substitutes* that
        exception for its own. Here the exception being replaced was the run's
        real diagnosis and the replacement is a note about a descriptor - an
        operator's `Ctrl-C` reached the CLI boundary as `E_INTERNAL`, exit 2,
        saying an `OSError` ended the command, which is wrong twice: the run was
        not ended by a defect, and the interrupt was not Distill's to relabel.

        The release failure is not dropped, because a hold this process believes
        it gave up and did not is a job identifier no later run can start - for
        the life of the process, since `flock` is refused inside the holding
        process exactly as outside it. It is recorded beside the failure it
        declined to replace, on the lock stream the lock's own `lock_released`
        event goes to, under the same `subject`.

        `Exception`, so a `BaseException` raised *by the release* still
        propagates - a second `Ctrl-C` landing inside cleanup, say. Swallowing
        an interrupt in order to preserve an earlier one is not an improvement
        on losing the earlier one. And with nothing in flight there is nothing
        to preserve: callers use `_release` there, and a release that fails then
        still raises.
        """
        try:
            self._release(job_id)
        except Exception as release_failure:
            _bundle_log(
                "lock_release_failed",
                subject=job_id,
                error=repr(release_failure),
                during=type(during).__name__,
            )

    def _holder_is_live(self, job_id: str) -> bool:
        """Whether a run is still live on `job_id`, asked of the lock, not a clock.

        `flock` is per open file description, so the lock this process holds is
        refused here exactly as another process's would be - one answer, whoever
        the holder is.

        Asked through `probe`, which is the read-only way to ask it: `take`
        opens with `O_CREAT` and would have a poller leave a lock file behind
        for every identifier it asked about, and its existence pre-check was a
        second syscall the answer could race - the file can be unlinked in
        between, and the `O_CREAT` puts it back. `probe` also reports a lock
        file it cannot open at all as `unknown` rather than raising, so a
        `get_job_status` on a root whose permissions this process cannot satisfy
        answers instead of crashing.

        What no reader can avoid is holding the lock for the instant it takes to
        find out: the only way to learn whether an `flock` is held is to ask for
        a conflicting one. So a `start` that lands in that instant can still be
        told `E_JOB_RUNNING` by a poller rather than by a run. That window is
        the mechanism's, not this module's, and it is narrower than the wrong
        answer any of the alternatives give.
        """
        return ExclusiveLock.probe(self.lock_path(job_id)) in HOLDER_PRESENT_STATES

    def _parse(self, text: str, job_id: str) -> JobRecord:
        """Read one record, refusing anything that is not one.

        A record is written by atomic replace, so a half-written one is not a
        state this can see. What it can see is a file somebody else put here, and
        reporting that as a job's outcome would be reporting a stranger's word as
        Distill's.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DistillError(
                "E_BAD_JOB_RECORD",
                JOB_STAGE,
                "job record is not readable JSON",
                {"job_id": job_id},
            ) from exc
        status = payload.get("status") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or status not in STORED_STATUSES:
            raise DistillError(
                "E_BAD_JOB_RECORD",
                JOB_STAGE,
                "job record does not carry a known status",
                {"job_id": job_id, "status": str(status)},
            )
        result = payload.get("result")
        error = payload.get("error")
        return JobRecord(
            job_id=str(payload.get("job_id", job_id)),
            tool=str(payload.get("tool", "")),
            status=status,
            updated_at=str(payload.get("updated_at", "")),
            result=result if isinstance(result, dict) else None,
            error=error if isinstance(error, dict) else None,
        )
