"""Bundle identity, layout and lifecycle for Distill.

This module owns the question a **bundle** is: which directory holds one, what
proves it (the **bundle marker**), which **generation** is active, and where
every file under a **bundle key** lives. It is the only place that names
`_manifest.json`, `g<N>`, `.tmp.g<N>`, `frames/`, `video.md` or `transcript.json`,
and the only place that decides whether a path may be written to.

The invariant it exists to hold: *a directory is a bundle only if it carries a
bundle marker*. Before this module, any directory containing `g1/video.md` was
one, so pointing `--output-dir` at a directory of a user's own made that
directory prunable - audit finding 1. Recognition is now positive and identity
bound, and *both* markers are held to it: a marker is a non-symlink regular file
holding a JSON object whose recorded bundle identity equals the directory name,
accepting either the current `bundle_key` field or the legacy `source_hash`
field so bundles written before this plan stay recognizable and prunable
(D-017). The **published marker** is additionally schema-valid, because it also
has to name an **active generation**.

Both, because finding 1 survived the first fix at a narrower filename: the
**ownership marker** was recognized by the presence of `_owner.json` alone -
never opened, never parsed, never compared - and a directory that carries only
an ownership marker is proposed for deletion *whole*, which is the one target
the active-generation guard cannot spare. Any tool's file of that name in a
user's directory deleted it. `is_file()` follows symlinks, so a link named like
either marker was one too.

A marker proves the directory is Distill's; it does not prove the bundle is
servable. `load_active` additionally verifies the **active generation** and its
**render** exist on disk, because a **manifest** is a promise, not evidence
(R-04): retention that deleted a generation left the manifest still naming it.
`publish_staging` proves the same **render** before it renames anything, so the
promise is not made in the first place unless there is something to serve.

Every question this module asks of the filesystem can be refused, and none of
them may end a walk. A directory prune cannot read is a **skip with a reason**
like any other (R-57), not a `PermissionError` out of `cache-doctor` - the
read-only command that exists *because* the destructive ones were unpreviewable.
The output root itself is the one refusal that ends the command rather than
becoming a skip, because a root that cannot be reached leaves no walk to save
and no report to salvage - see `_root_directory_exists`, which is also where the
rule that a root nobody could stat is not a root that is absent is stated.

What this module does not own: what a pipeline stage computes, what a
**generation** contains, job records (`job_store`), the **source fingerprint**
or **options hash** that combine into a bundle key (`source.py`, `options.py`),
and the output-root policy, which answers a different question - whether Distill
may write under a root at all - and stays with `source.validate_output_root`.

Exclusion between runs is the kernel's, via `flock` on a descriptor `begin`
holds for the run's duration. There is no heartbeat and no staleness window: a
lock file whose *content* describes its holder has to be created before it can
be written, and a contender reading it in between finds a lock nobody owns and
takes it - two runs, one **bundle key** (finding 11). Here the lock exists for
exactly as long as the descriptor does, so a holder that is killed outright
releases it the moment the kernel closes its descriptors, and a live holder is
never stealable however old its lock file is.

**Publish** is ordered rather than atomic, because no filesystem offers one
operation that both renames a directory and rewrites a file: assemble in
staging, strip every **stage result**, rename to `g<N>`, then atomically replace
the **manifest** (R-12). Only the last step makes a generation active, so the
one gap the order leaves - a renamed directory no manifest names yet - costs a
reader nothing: the previous **active generation** stays servable and the new
directory is an **orphan generation** prune can reclaim.

Atomicity is scoped, not universal (D-033): the **manifest** and job records are
read by other processes and are written by atomic replace, while **stage
results** are written into a **staging directory** held under the run lock,
which nothing else may read, and are ordinary writes. Which discipline a target
gets is decided here; performing either is `emit.TextEmitter`'s, so that "how
many places make text durable?" has an answer other than "however many modules
call `write_text`" (R-22).

Nothing durable is written from a document a caller assembled: a **transcript**
and a **stage result** are handed in as carriers and serialized here, because
`artifacts.serialize` is the last point at which a carrier whose **redaction**
policy never ran can be refused (R-20). A writer that accepted the document
instead would accept text that had already been past that check without going
through it.

Written cheaply, read suspiciously. A **stage result** is durable scratch a
*later* run reads back, so this module also owns what one must prove before a
**resume** believes it: the schema version it was written under, the **bundle
key** it belongs to, and that no path in it leaves the bundle root (R-23). It
does not own what the payload means - a stage's own output stays the stage's -
and it never fails a run over one. A document that does not hold up is
discarded and the stage recomputed (D-030), because a stage result is by
definition something that can be produced again, and a resume that could end a
run would be worse than the trust it replaced (RV-8).

**Prune** is two operations, not one (D-018). *Generation retention* keeps the
newest `keep_generations` **generations** of a bundle and never proposes the
**active generation** at any value - the old code kept `len - keep_generations`
oldest as candidates, so `keep_generations=0` proposed every generation
including the active one and wiped a bundle a reader was entitled to (finding
2). *Bundle expiry* removes an entire aged bundle, active generation included.
Collapsing them is what made the critical finding possible: "never delete the
active generation" is retention's rule, and expiry's whole purpose is to delete
a bundle outright.

A `PrunePlan` is advisory and the revalidation `apply_prune` performs under each
target's lock is authoritative (D-023, R-10). Nothing else can make prune safe
against a concurrent run: a plan is computed by walking a tree no lock covers,
so by the time it is applied a run may have published into a bundle the plan
proposed. Applying re-derives the same decision under the lock and deletes only
what the re-derivation still proposes, which is why a **generation** that became
active between the two steps survives (RV-1).

What prune does not own: whether Distill may write under a root at all
(`source.validate_output_root`), and what the CLI calls this - the public
command stays `cleanup-cache` (D-042) while the vocabulary here is **prune**.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import math
import os
import re
import shutil
import stat
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .artifacts import (
    Carrier,
    RedactionState,
    StageResult,
    Transcript,
    is_path_field,
    serialize,
)
from .emit import EMITTER
from .errors import DistillError, errno_name

LOGGER = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"
"""The **published marker**: present only once a generation has been published."""

OWNERSHIP_MARKER_NAME = "_owner.json"
"""The **ownership marker** `begin` writes first, so a directory is identifiable
as Distill-owned from its first moment - before any manifest exists (R-11,
D-025). A run that crashes before publishing leaves this and nothing else.

The filename is where to look, never what is proved: what makes the directory
Distill's is the **bundle key** recorded inside, matching the directory's name."""

RENDER_NAME = "video.md"
TRANSCRIPT_NAME = "transcript.json"
FRAMES_DIR_NAME = "frames"
GENERATION_PREFIX = "g"
STAGING_PREFIX = ".tmp."

SCRATCH_DIR_NAME = "_scratch"
"""Where a stage puts working files that are not bundle content.

A **stage result** is scratch Distill writes and can therefore recognize by
name; this is scratch a *stage* writes, whose names Distill does not know - the
decoded audio track transcription keeps so an interrupted run does not decode
the source twice, and whatever a stage added later keeps beside it. Handing a
stage the **staging directory** itself gave it nowhere to put those that the
rename would not publish: the decode was renamed into `g<N>` and served as
bundle content, so every bundle carried a full decode of its source's audio for
as long as it existed (finding 3-opus).

Inside the staging directory rather than outside it, because scratch is exactly
as valuable as a **resume** is: a temporary directory nothing owns would be
reclaimed by the system between the crash and the resume, and one beside the
bundle would be disk **prune** has no rule for. Removed by the publish, with the
stage results and for the same reason (R-13).
"""

STAGE_RESULT_PREFIX = "_"
STAGE_RESULT_SUFFIX = ".json"
STAGE_RESULT_GLOB = f"{STAGE_RESULT_PREFIX}*{STAGE_RESULT_SUFFIX}"
"""How a **stage result** is recognized inside a **staging directory**.

Recognition is by name because the strip has to be exhaustive rather than
informed: R-13 forbids a stage result in a **generation**, and a strip driven by
a list of stage names the run happens to remember would publish the one nobody
added to the list. Everything matching this shape goes, whoever wrote it.

The name is scoped to the generation directory. The **manifest** and the
**ownership marker** carry the same underscore convention but live at the bundle
root, which is not renamed and is never walked by the strip.
"""

STAGE_RESULT_SCHEMA_VERSION = 1
"""The schema this Distill writes a **stage result** under, and the only one it reads.

A single known version rather than a set: D-015 says pre-1.0 there are no
dependents and on-disk breakage is acceptable, and a stage result is scratch, so
the cost of not understanding an older one is a recomputation. A set of accepted
versions would be the beginning of a migration path that nothing needs and that
nothing could test against a schema that does not exist yet.
"""


STAGE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
"""The bounded domain a stage name must fall in, matched in full.

A **stage result**'s name becomes a filename inside the **staging directory**,
so an unbounded name is a path: `x/../../_manifest` names the **manifest**, and
`ocr/sub` names a file the strip's `_*.json` glob does not match, which
publishes it. Neither is reachable from Distill's own call sites today, and
neither may be reachable from a call site added later - R-13 is an invariant
about what is on disk, not about the discipline of callers. Rejected rather
than sanitized, on the same grounds as R-18: sanitizing maps two names onto one
file.
"""

IDENTITY_FIELDS = ("bundle_key", "source_hash")
"""Manifest fields carrying the recorded **bundle key**, current name first.

`source_hash` is the legacy name for the same value (D-008): it hashes the
**source fingerprint** together with the **options hash**, so it identifies a
bundle rather than a source. Both are accepted as identity; only `bundle_key` is
written from here on.
"""

LOCK_DIR_NAME = "_locks"
"""Where the run locks live: beside the bundles, not inside them.

A lock file inside the bundle it protects is a lock **prune** deletes along with
the bundle - and a waiter holding a descriptor on an unlinked inode locks that
inode while the next run creates a fresh file at the same path and locks that,
which is two runs holding one bundle key. Keeping the lock outside the bundle
makes the path itself the identity of the lock, for as long as the output root
exists.
"""

LOCK_SUFFIX = ".lock"
"""How a run lock is named inside `_locks`, and how one is recognized there."""

SINGLE_SOURCE_LOCK_WAIT_SEC = 300.0
"""How long one video's run waits for a contended **bundle key** (D-044).

The run a user is watching is worth waiting for: the likely holder is another
run of the same video that is nearly done, and waiting for it costs less than
failing and re-running.
"""

BATCH_ITEM_LOCK_WAIT_SEC = 5.0
"""How long a batch or playlist item waits before failing `E_LOCKED` (D-044).

Serializing a 25-item playlist behind another run's 40-minute video would cost
hours of blocked waiting. `continue_on_error` already defaults true, so the item
fails and the batch proceeds; re-running picks the item up as a cache hit.
"""

LOCK_POLL_SEC = 0.05
"""How often a waiter re-asks the kernel for the lock.

`flock` can block until the lock is free, but a blocking wait cannot be bounded
by a budget without a signal, and a signal-based deadline would be a second
concurrency mechanism to get right. Polling costs one cheap syscall per interval
and makes the budget exact.
"""

_LOCK_HELD_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN})
"""`flock` refusing a lock somebody else holds, which is an answer rather than a
failure. Every other errno says this filesystem cannot give Distill the
exclusion it asked for, and is fatal (R-09).

`EACCES` is deliberately not here. `flock(2)` reports contention as
`EWOULDBLOCK` - `EAGAIN` is the same number - and never as `EACCES`, so a
filesystem answering `EACCES` is one that cannot lock. Counted as "held", it
made a run poll out its whole budget and then report `E_LOCKED`: a user told
another run holds the bundle does not go and look at the mount (finding
8-opus)."""

DEFAULT_KEEP_GENERATIONS = 3
"""How many **generations** of a bundle retention keeps when nobody says."""

SECONDS_PER_DAY = 86400.0

PRUNE_MAX_DEPTH = 6
"""How deep prune walks looking for nested bundle roots (R-05).

Bundles nest because a playlist run gives each item its own output root under
`playlists/<playlist>/`, which is two levels below the root the user names - a
walk of the top level alone never sees them, and they were unprunable
(finding 17). The walk is bounded because the root is a directory a user chose:
prune stops descending rather than walking an arbitrarily deep tree that has
nothing to do with Distill, and reports the directories it stopped at.
"""

BUNDLE_EVENT_TYPE = "distill.bundle"

MarkerKind = Literal["published", "owned", "foreign", "invalid", "absent", "unreadable"]
"""Why a directory is or is not a **bundle**.

`unreadable` is the directory this process may not look inside, which is neither
"no marker" nor "a bad marker": it is Distill declining to answer. Kept distinct
because the answer a user needs for it - a permission to fix - is not the answer
either of the others calls for, and because folding it into `absent` would make
a directory prune descends into out of one it cannot see (finding 2-opus).
"""

FileState = Literal["regular", "irregular", "absent"]
"""What is at a path Distill will only accept as an ordinary file.

`irregular` is anything present under that name which is not a non-symlink
regular file - a symlink, a directory, a device. It is a third answer rather than
a second spelling of `absent` because it means somebody put something there:
recognition refuses it, and reports it (finding 1b).
"""

EntryKind = Literal["symlink", "directory", "other"]
"""What one entry of a directory listing is, before anything follows it."""

PruneKind = Literal["bundle", "generation", "staging"]
"""What a prune target is on disk: a whole **bundle**, one **generation**, or one
**staging directory**."""

PruneRule = Literal["retention", "expiry", "orphan", "staging"]
"""Which rule proposed a target.

`retention` and `expiry` are the two operations of D-018 and are never the same
one: retention is governed by `keep_generations` and never proposes the **active
generation**; expiry is governed by `max_age_days` and removes a whole bundle
including its active generation. `orphan` is a **generation** nothing can serve
because no **manifest** names it, and `staging` is scratch left by a run that is
not live.
"""

PruneVerdict = Literal["deleted", "skipped"]

LockState = Literal["live", "stale", "absent", "unknown"]
"""What a run lock is right now, asked of the kernel rather than of the file.

`live` is a lock another run holds. `stale` is a lock file nobody holds - a
release closes a descriptor and deliberately leaves the file, so its presence
alone means nothing. `absent` is a **bundle key** no run has ever reached
staging under. `unknown` is a lock file this process could not open at all,
which is not the same as free: prune counts it as held, because "may not ask"
must never resolve to "nobody is running".
"""


def _bundle_log(event: str, **detail: Any) -> None:
    """Emit one bundle-store boundary event: lock taken, waited for, denied, refused.

    Metadata only, in the shape `run_command` and `source` use, so one log stream
    answers "what did this run do with this bundle key". Contention is otherwise
    invisible: a run that took twenty minutes because it waited ten of them for a
    lock looks exactly like a run that was slow.
    """
    LOGGER.debug(
        json.dumps(
            {
                "type": BUNDLE_EVENT_TYPE,
                "event": event,
                "detail": {"pid": os.getpid(), **detail},
            },
            sort_keys=True,
        )
    )


@dataclass(frozen=True)
class BundlePaths:
    root: Path
    generation: Path
    frames: Path
    manifest: Path
    transcript: Path
    markdown: Path


@dataclass(frozen=True)
class MarkerVerdict:
    """Why a directory is, or is not, a **bundle**.

    Carries a reason for every verdict rather than a bare boolean, because the
    destructive operations that consume it must be able to report *why* a
    directory was skipped - "considered nothing" and "deleted nothing" are
    different answers (R-57).
    """

    kind: MarkerKind
    reason: str
    bundle_key: str | None = None
    manifest: dict[str, Any] | None = None

    @property
    def is_bundle(self) -> bool:
        """Whether the directory is a **bundle**: something may be served from it."""
        return self.kind == "published"

    @property
    def is_distill_owned(self) -> bool:
        """Whether Distill wrote the directory, published or not.

        A directory holding only an **ownership marker** is not a bundle - there
        is nothing to serve - but it is Distill's, so prune may reclaim it
        instead of skipping it forever (RV-9).
        """
        return self.kind in ("published", "owned")


@dataclass(frozen=True)
class BundleSnapshot:
    """A readable **active generation**, proven to exist when it was loaded."""

    root: Path
    bundle_key: str
    generation: Path
    manifest: dict[str, Any]

    @property
    def markdown(self) -> Path:
        return self.generation / RENDER_NAME

    @property
    def transcript(self) -> Path:
        return self.generation / TRANSCRIPT_NAME

    @property
    def frames(self) -> Path:
        return self.generation / FRAMES_DIR_NAME

    @property
    def manifest_path(self) -> Path:
        """Where the **manifest** this snapshot was read from lives.

        A caller that reports paths back to its own caller needs the marker's
        path as well as its content, and deriving it from `root` outside this
        module is how the filename escapes into a second module (D-041).
        """
        return self.root / MANIFEST_NAME


@dataclass
class ExclusiveLock:
    """One holder's exclusive claim on a **subject**, granted by the kernel.

    The subject is whatever the lock path stands for - a **bundle key** here, a
    job identifier in `job_store`. The type is stated about the mechanism rather
    than about bundles because there is exactly one right way to ask "is another
    process live on this?", and a second copy of it is a second chance to get
    finding 11 wrong.

    The lock *is* an open descriptor holding a `flock`, not a file whose contents
    name a holder. That is what makes it exclusive and what makes it
    self-releasing: there is nothing to publish, no staleness to guess at, and no
    pid that could be reused. A holder killed outright releases it the moment the
    kernel closes its descriptors (finding 11).

    Held until `release`, which the holder calls when it is finished - not when
    staging ends. `release` is idempotent, because the failure paths that release
    a lock early overlap with the caller's own cleanup.

    Does not own: what the subject contains, or which path a subject locks on.
    It does not own the *meaning* of a lock either - the **acquisition lease**
    in `source.py` is this mechanism keyed by **lock key**, answering "is
    another run fetching this source?" where the bundle lock answers "is another
    run producing this bundle?". Two questions, two lock paths, one primitive:
    the exclusion argument is subtle enough that a second copy of it is a second
    chance to get finding 11 wrong.
    """

    subject: str
    path: Path
    fd: int
    released: bool = False

    @classmethod
    def take(
        cls,
        subject: str,
        path: Path,
        *,
        stage: str = "bundle",
        message: str = "filesystem cannot grant an exclusive lock here",
    ) -> ExclusiveLock | None:
        """Take the lock, or report `None` if another holder has it.

        The descriptor stays open inside the returned lock: closing it is what
        releasing means, so nothing else may close it. `flock` is per open file
        description rather than per process, so a second lock taken against the
        same path is refused even inside the process that holds the first.

        A filesystem that cannot grant the lock is fatal (R-09): an errno other
        than "held" means Distill cannot tell one run from two here, and the only
        answer that does not silently reintroduce finding 6 is to stop. `stage`
        and `message` name the caller's question in that error, because a
        failure to lock a source directory and a failure to lock a bundle are
        the same defect reported to a user doing different things.

        Taking a lock is two syscalls and both are inside the guard. A read-only
        mount, or a lock directory this process may not write, refuses the
        `open` and never reaches `flock` - which is the same fact about the same
        filesystem, reported as `E_INTERNAL` while only the second call was
        guarded (finding 8-opus).
        """
        fd: int | None = None
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            if exc.errno in _LOCK_HELD_ERRNOS:
                return None
            raise DistillError(
                "E_LOCK_UNSUPPORTED",
                stage,
                message,
                {
                    "subject": subject,
                    "lock_path": str(path),
                    "errno": errno_name(exc),
                },
            ) from exc
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        return cls(subject=subject, path=path, fd=fd)

    @staticmethod
    def require_capability(
        directory: Path,
        *,
        subject: str,
        stage: str = "bundle",
        message: str = "filesystem cannot grant an exclusive lock here",
    ) -> None:
        """Prove this filesystem grants `flock`, creating nothing (R-09).

        R-09 says Distill fails `E_LOCK_UNSUPPORTED` *before mutating anything*,
        and taking the real lock cannot answer it that way: `_locks` and an
        empty `<subject>.lock` are both on disk by the time `flock` is asked,
        so an unsupported filesystem was left carrying two files written for a
        mechanism it had just refused (finding 6-codex).

        The question is about the filesystem, not about this subject, so it is
        asked of a directory that is already there - a shared lock taken on its
        descriptor and dropped in the same breath. Shared, because two runs
        probing at once must both get an answer; dropped at once, because
        holding it would exclude something this is not entitled to exclude.

        Contention is an answer: a refusal saying somebody holds a conflicting
        lock proves the mechanism works. Any other errno is the fatal one.

        A directory this process cannot open at all proves nothing either way
        and is left to `take`, which reports whatever the real attempt hits.
        The probe is a claim about `directory`'s filesystem, so a lock path on a
        different mount below it is still guarded by `take` and only by `take`.
        """
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in _LOCK_HELD_ERRNOS:
                return
            raise DistillError(
                "E_LOCK_UNSUPPORTED",
                stage,
                message,
                {
                    "subject": subject,
                    "lock_path": str(directory),
                    "errno": errno_name(exc),
                },
            ) from exc
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @classmethod
    def probe(cls, path: Path) -> LockState:
        """Report whether anybody holds this lock, creating nothing.

        `take` opens with `O_CREAT`, because a run that needs the lock needs the
        file to exist. A probe must not: a caller that only wants to *know*
        would otherwise leave a lock file behind for every key it asked about,
        and `cache-doctor` is read-only (R-57). Checking existence first and
        then calling `take` is not the same thing - between the two calls the
        file can be unlinked, and the `O_CREAT` recreates it.

        The lock is released immediately: holding it would answer the question
        by making the answer true.

        A lock file that will not open is `unknown` rather than an exception:
        the read-only inspection must be able to report on a root whatever its
        permissions, and `lock_is_held` reads `unknown` as held so the
        destructive side errs toward leaving the bundle alone.
        """
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unknown"
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in _LOCK_HELD_ERRNOS:
                return "live"
            raise
        finally:
            os.close(fd)
        return "stale"

    def release(self) -> None:
        """Give the lock up by closing the descriptor the kernel locked.

        The lock file itself stays on disk, deliberately: unlinking it is a
        second way to lose exclusivity, because a waiter that already opened the
        path holds a descriptor on an inode that now has no name and can lock it
        while the next run locks a fresh file at the same path.

        Nothing reclaims those empty files, and that is the intended end state
        rather than a gap: the path *is* the identity of the lock, so a file
        deleted while any run might still be about to open it reintroduces the
        problem above. **Prune** treats `_locks` as a reserved name and never
        descends into it - a claim that it would reclaim them stood here and was
        never true of any prune rule (finding 7-opus). One empty inode per
        **bundle key** is the price of the mechanism, and `cache-doctor` reports
        every one of them so a lock whose bundle is long gone is at least
        visible.
        """
        if self.released:
            return
        self.released = True
        os.close(self.fd)
        _bundle_log("lock_released", subject=self.subject, lock_path=str(self.path))


@dataclass(frozen=True)
class PrunePolicy:
    """How much of a **bundle** prune is allowed to reclaim.

    Validated on construction rather than at the call site (R-03), so a policy
    that exists is a policy that means something: `keep_generations=0` is the
    input that wiped the **active generation** (finding 2), and it is now
    refused where it is written rather than reinterpreted where it is used.
    `max_age_days=None` means no **bundle expiry** at all, which is different
    from an expiry horizon of zero days - that would expire every bundle the
    moment it was published, and is refused.
    """

    keep_generations: int = DEFAULT_KEEP_GENERATIONS
    max_age_days: float | None = None

    def __post_init__(self) -> None:
        keep = self.keep_generations
        # bool is an int in Python, and `keep_generations=True` meaning 1 is a
        # coincidence rather than an intention.
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise DistillError(
                "E_BAD_OPTIONS",
                "prune",
                "keep_generations must be an integer of at least 1",
                {"keep_generations": repr(keep)},
            )
        age = self.max_age_days
        if age is None:
            return
        if isinstance(age, bool) or not isinstance(age, int | float):
            raise DistillError(
                "E_BAD_OPTIONS",
                "prune",
                "max_age_days must be a finite number greater than 0",
                {"max_age_days": repr(age)},
            )
        if not math.isfinite(float(age)) or float(age) <= 0:
            raise DistillError(
                "E_BAD_OPTIONS",
                "prune",
                "max_age_days must be a finite number greater than 0",
                {"max_age_days": repr(age)},
            )


@dataclass(frozen=True)
class PruneTarget:
    """One thing prune proposes to remove, and the rule that proposed it."""

    path: Path
    kind: PruneKind
    rule: PruneRule
    bundle_root: Path
    bundle_key: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "rule": self.rule,
            "bundle_key": self.bundle_key,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DirectorySkip:
    """One directory the walk looked at and did not treat as a **bundle**, and why.

    Reported rather than silently passed over, because "considered nothing" and
    "deleted nothing" are different answers, and a caller that cannot tell them
    apart cannot tell a healthy cache from a prune that skipped every bundle it
    found (R-01, R-57).

    Shared by **prune** and by the survey `cache-doctor` reports, because they
    are the same walk asking the same question: a skip a preview does not
    mention is a skip the destructive operation will make silently.
    """

    path: Path
    verdict: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "verdict": self.verdict, "reason": self.reason}


@dataclass(frozen=True)
class LockReport:
    """One run lock and whether anybody holds it (R-57)."""

    path: Path
    subject: str
    state: LockState

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "subject": self.subject, "state": self.state}


@dataclass(frozen=True)
class BundleReport:
    """What one directory the walk recognized as Distill's looks like right now.

    Descriptive, not advisory: it says what is on disk, and says nothing about
    what should happen to any of it. What a **prune** would remove is a
    `PrunePlan`, derived separately from the same walk so the two answers cannot
    drift into disagreeing about the same bundle.
    """

    path: Path
    bundle_key: str
    verdict: str
    reason: str
    active_generation: str | None
    generations: tuple[str, ...]
    orphan_generations: tuple[str, ...]
    staging_directories: tuple[str, ...]
    lock: LockReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bundle_key": self.bundle_key,
            "verdict": self.verdict,
            "reason": self.reason,
            "active_generation": self.active_generation,
            "generations": list(self.generations),
            "orphan_generations": list(self.orphan_generations),
            "staging_directories": list(self.staging_directories),
            "lock": self.lock.to_dict(),
        }


@dataclass(frozen=True)
class StoreSurvey:
    """Every **bundle** under one output root, and everything that is not one.

    The skips are half the answer, not a footnote: a survey listing only the
    bundles it found reports the same empty result for a healthy root and for a
    root the walk declined to descend into (R-57).
    """

    root: Path
    root_exists: bool
    bundles: tuple[BundleReport, ...] = ()
    skipped: tuple[DirectorySkip, ...] = ()
    locks: tuple[LockReport, ...] = ()
    considered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "root_exists": self.root_exists,
            "considered": self.considered,
            "bundle_count": len(self.bundles),
            "bundles": [bundle.to_dict() for bundle in self.bundles],
            "skipped_count": len(self.skipped),
            "skipped": [skip.to_dict() for skip in self.skipped],
            "locks": [lock.to_dict() for lock in self.locks],
        }


@dataclass(frozen=True)
class PrunePlan:
    """What prune proposes to remove. Advisory: `apply_prune` decides (D-023).

    `considered` counts the directories the walk looked at, so an empty plan over
    a root full of bundles and an empty plan over an empty root are
    distinguishable.
    """

    root: Path
    policy: PrunePolicy
    targets: tuple[PruneTarget, ...] = ()
    skipped: tuple[DirectorySkip, ...] = ()
    considered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "keep_generations": self.policy.keep_generations,
            "max_age_days": self.policy.max_age_days,
            "considered": self.considered,
            "candidate_count": len(self.targets),
            "candidates": [target.to_dict() for target in self.targets],
            "skipped_count": len(self.skipped),
            "skipped": [skip.to_dict() for skip in self.skipped],
        }


@dataclass(frozen=True)
class PruneResult:
    """What happened to one proposed target once the lock was held."""

    target: PruneTarget
    verdict: PruneVerdict
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.target.to_dict(), "verdict": self.verdict, "reason": self.reason}


@dataclass(frozen=True)
class PruneOutcome:
    """What prune actually did, per target, including everything it did not do."""

    root: Path
    results: tuple[PruneResult, ...] = ()
    skipped: tuple[DirectorySkip, ...] = ()
    considered: int = 0

    @property
    def deleted(self) -> tuple[Path, ...]:
        return tuple(result.target.path for result in self.results if result.verdict == "deleted")

    @property
    def retained(self) -> tuple[PruneResult, ...]:
        """Targets the revalidation under lock refused to delete after all."""
        return tuple(result for result in self.results if result.verdict == "skipped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "considered": self.considered,
            "deleted_count": len(self.deleted),
            "deleted": [str(path) for path in self.deleted],
            "results": [result.to_dict() for result in self.results],
            "skipped_count": len(self.skipped),
            "skipped": [skip.to_dict() for skip in self.skipped],
        }


@dataclass
class _Scan:
    """Mutable accumulator for the walk, so recursion has one place to add to.

    Shared by **prune** and by the survey: what each of them does when the walk
    reaches a Distill-owned directory differs, and nothing else about the walk
    does. Two copies of "which directories are bundles, which are skipped and
    why, and how deep to descend" is two answers that can disagree about the
    same root - which is the shape of finding 1.
    """

    skipped: list[DirectorySkip] = field(default_factory=list)
    locks: list[LockReport] = field(default_factory=list)
    considered: int = 0


@dataclass(frozen=True)
class BundleStore:
    """Every **bundle** under one output root."""

    root: Path
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> BundleStore:
        """Open the store over an output root already accepted by the root policy.

        Resolving here means every path this store derives is compared against a
        root free of symlinks, so confinement checks cannot be defeated by one.

        `clock` and `sleep` exist so a wait budget can be asserted exactly rather
        than waited out; production passes neither.
        """
        return cls(Path(root).resolve(), clock, sleep)

    def bundle_root(self, bundle_key: str) -> Path:
        """The directory a **bundle key** names, refusing anything that escapes.

        A bundle key is a single directory name. Treating it as a path fragment
        would let `..` or an absolute value name a directory outside the output
        root - and everything below reads, and later deletes, what it is handed.
        """
        parts = Path(bundle_key).parts
        if len(parts) != 1 or bundle_key in (".", ".."):
            raise DistillError(
                "E_BAD_OUTPUT_DIR",
                "bundle",
                "bundle key must be a single directory name under the output root",
                {"bundle_key": bundle_key, "output_root": str(self.root)},
            )
        return self.root / bundle_key

    def marker(self, bundle_key: str) -> MarkerVerdict:
        """The **bundle marker** verdict for one bundle key."""
        return read_marker(self.bundle_root(bundle_key))

    def load_active(self, bundle_key: str) -> BundleSnapshot | None:
        """The servable **active generation**, or `None` for a cache miss.

        `None` covers every way a directory can fail to be a servable bundle: no
        marker, a marker recording another identity, a malformed manifest, or a
        manifest naming a generation or **render** that is not on disk (R-04).
        None of these ends the run - the run simply produces the bundle.
        """
        directory = self.bundle_root(bundle_key)
        verdict = read_marker(directory)
        if not verdict.is_bundle or verdict.manifest is None:
            return None

        generation_name = verdict.manifest.get("active_generation")
        if not isinstance(generation_name, str) or not is_generation_name(generation_name):
            return None
        generation = directory / generation_name
        if not generation.is_dir() or not (generation / RENDER_NAME).is_file():
            return None

        return BundleSnapshot(
            root=directory,
            bundle_key=bundle_key,
            generation=generation,
            manifest=verdict.manifest,
        )

    def lock_path(self, bundle_key: str) -> Path:
        """Where the run lock for one **bundle key** lives. Derives, creates nothing.

        Validated through `bundle_root`, so a key that would escape the output
        root is refused here too: a lock file outside the root protects nothing
        that is under it.
        """
        return prune_lock_path(self.bundle_root(bundle_key))

    def begin(
        self,
        bundle_key: str,
        *,
        wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
        resume: bool = True,
        reuse_active: bool = True,
    ) -> BundleRun | BundleSnapshot:
        """Take the run lock for `bundle_key` and open a **staging directory**.

        Returns a `BundleSnapshot` instead when an **active generation** is on
        disk *once the lock is held* (R-08). The re-check has to happen under the
        lock to be worth anything: a waiter that decided "cache miss" before
        queueing would come out of a wait it spent behind the winner and redo
        the work the winner just published (RV-1).

        `reuse_active=False` is `force_reprocess`: the caller wants the work
        done again whatever is on disk. It suppresses the hand-back, not the
        re-check - the lock is still taken first and the existing **active
        generation** is left servable until this run publishes over it, so a
        forced run that fails costs a reader nothing.

        Ordering is the requirement here, not just the outcome:

        1. the lock, which is also the capability probe - nothing is created
           under the bundle key until the kernel has granted exclusion, so a
           filesystem that cannot lock fails having mutated nothing (R-09);
        2. the **ownership marker**, before any content, so a run that dies
           anywhere after this leaves a directory identifiable as Distill's
           rather than one that is neither a bundle nor anybody's (R-11, RV-9);
        3. the staging directory.

        The lock is held for the run's duration, released by `BundleRun.release`
        (or by the kernel, if the holder dies). `wait_sec` is the caller's
        budget: `SINGLE_SOURCE_LOCK_WAIT_SEC` for the run a user is watching,
        `BATCH_ITEM_LOCK_WAIT_SEC` for one item of a batch or playlist, and
        `E_LOCKED` when it runs out.
        """
        lock, waited_sec = self._take_lock(bundle_key, wait_sec)
        try:
            snapshot = self.load_active(bundle_key) if reuse_active else None
            if snapshot is not None:
                # A cache hit holds nothing: the question the lock was taken to
                # answer has been answered, so the next run may have it.
                lock.release()
                return snapshot
            directory = self.bundle_root(bundle_key)
            ensure_safe_directory(directory, self.root)
            write_ownership_marker(directory, bundle_key)
            paths = stage_paths(directory, reset=not resume)
        except BaseException:
            lock.release()
            raise
        return BundleRun(
            store=self,
            bundle_key=bundle_key,
            paths=paths,
            lock=lock,
            # Started once the lock is held, so the staging duration a commit
            # reports is the run's own cost. The wait that preceded it is
            # already reported, separately, by `lock_acquired`.
            staged_at=self.clock(),
            waited_sec=waited_sec,
            resumed=resume,
        )

    def _take_lock(self, bundle_key: str, wait_sec: float) -> tuple[ExclusiveLock, float]:
        """Poll for the run lock within this caller's budget.

        The budget is the only thing that ends the wait: a lock is held until its
        holder gives it up or dies, and neither is something to time out on a
        holder's behalf.
        """
        path = self.lock_path(bundle_key)

        def reported(exc: DistillError) -> DistillError:
            """One report of the capability failure, whichever attempt found it.

            The probe and the real take ask the same filesystem the same
            question at two moments, and an operator needs the answer either
            way - the lock path is logged rather than the probe's directory,
            because the path is what the run was trying to take.
            """
            if exc.code == "E_LOCK_UNSUPPORTED":
                _bundle_log(
                    "lock_unsupported",
                    bundle_key=bundle_key,
                    lock_path=str(path),
                    errno=exc.details.get("errno"),
                )
            return exc

        # R-09 in the order it is written: the capability is proved against the
        # output root, which is already there, before `_locks` or a lock file is
        # created under it. A filesystem that cannot lock is then a filesystem
        # Distill has written nothing to (finding 6-codex).
        try:
            ExclusiveLock.require_capability(self.root, subject=bundle_key)
        except DistillError as exc:
            raise reported(exc) from None
        path.parent.mkdir(parents=True, exist_ok=True)
        started = self.clock()
        contended = False
        while True:
            try:
                lock = ExclusiveLock.take(bundle_key, path)
            except DistillError as exc:
                raise reported(exc) from None
            waited = self.clock() - started
            if lock is not None:
                if contended:
                    _bundle_log(
                        "lock_waited",
                        bundle_key=bundle_key,
                        lock_path=str(path),
                        waited_sec=waited,
                    )
                _bundle_log(
                    "lock_acquired",
                    bundle_key=bundle_key,
                    lock_path=str(path),
                    waited_sec=waited,
                )
                return lock, waited
            contended = True
            if waited >= wait_sec:
                _bundle_log(
                    "lock_denied",
                    bundle_key=bundle_key,
                    lock_path=str(path),
                    waited_sec=waited,
                    wait_budget_sec=wait_sec,
                )
                raise DistillError(
                    "E_LOCKED",
                    "bundle",
                    "another run holds this bundle",
                    {
                        "bundle_key": bundle_key,
                        "lock_path": str(path),
                        "waited_sec": waited,
                        "wait_budget_sec": wait_sec,
                    },
                )
            self.sleep(min(LOCK_POLL_SEC, wait_sec - waited))

    def patch_published(
        self, snapshot: BundleSnapshot, fields: dict[str, Any]
    ) -> BundleSnapshot:
        """Amend a published **manifest**, keeping it a valid marker throughout.

        The amendment goes through the same atomic replace as the original
        write (R-14): a manifest is the **bundle marker**, so a reader that
        catches it half-rewritten sees a directory that is briefly not a bundle
        at all. The merged document is validated before it is written, so an
        amendment cannot turn a servable bundle into an unrecognizable one.

        Identity and the **active generation** are not amendable. Both are
        publish's to decide - the first is what makes the manifest a marker for
        *this* directory, and the second is what makes a generation active, so
        an amendment that could set either would be a publish that skipped the
        rename, naming a generation that need not exist (finding 2's shape).

        Refusing to *set* them is not enough, because a merge carries them.
        `snapshot` is a document read at some earlier moment, typically by a
        cache lookup that released the lock as soon as it had an answer - that
        is what makes a hit cheap. Merging fields into that captured document
        and writing the result brings its `active_generation` along, so a run
        that published in the meantime is undone by a *reader*: the marker names
        the older generation again and the newer one becomes an **orphan
        generation**. The amendment therefore happens under the run lock and
        against the manifest on disk, and declines when what it reads back is no
        longer the generation it was handed.

        Declining rather than waiting or failing is the right answer for what
        this is used for. An amendment is a backfill, never the purpose of the
        call that makes it, so it is not worth queueing behind a 40-minute run
        and not worth ending a caller over: the caller is handed back what it
        already had.
        """
        fixed = {
            field: value
            for field, value in fields.items()
            if field in ("active_generation", *IDENTITY_FIELDS)
            and value != snapshot.manifest.get(field)
        }
        if fixed:
            raise DistillError(
                "E_BAD_MANIFEST",
                "bundle",
                "identity and the active generation are set by publish, not by an amendment",
                {"bundle_key": snapshot.bundle_key, "fields": sorted(fixed)},
            )

        path = self.lock_path(snapshot.bundle_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = ExclusiveLock.take(snapshot.bundle_key, path)
        if lock is None:
            _bundle_log(
                "amend_declined",
                bundle_key=snapshot.bundle_key,
                reason="another run holds this bundle",
            )
            return snapshot
        try:
            current = self.load_active(snapshot.bundle_key)
            if current is None or current.generation != snapshot.generation:
                _bundle_log(
                    "amend_declined",
                    bundle_key=snapshot.bundle_key,
                    reason="the active generation is no longer the one this amendment read",
                    read=snapshot.generation.name,
                    active=None if current is None else current.generation.name,
                )
                return snapshot
            manifest = {**current.manifest, **fields}
            validate_manifest_schema(manifest, require_active_generation=True)
            write_manifest(current.root, manifest)
        finally:
            lock.release()
        return BundleSnapshot(
            root=current.root,
            bundle_key=current.bundle_key,
            generation=current.generation,
            manifest=manifest,
        )

    def plan_prune(self, policy: PrunePolicy) -> PrunePlan:
        """Propose what may be reclaimed under `policy`, deleting nothing.

        The walk descends through directories that are not bundles, so a bundle
        under a playlist root is subject to the same policy as one at the top
        level (R-05), and stops at every directory that is one - a **generation**
        is not a nested bundle. Directories that are neither are reported as
        skips with the reason they were not treated as bundles (R-01), which is
        what keeps "considered nothing" distinguishable from "deleted nothing".

        Advisory by construction (D-023): nothing here holds a lock for longer
        than the liveness probe, so every decision in the returned plan is a
        decision about a moment that has already passed. `apply_prune` makes it
        again under the lock.

        The root guard is `survey`'s, for the same reason the walk is: an empty
        plan over a root nobody could reach says "nothing to reclaim" about a
        cache that may be full (see `_root_directory_exists`).
        """
        if not _root_directory_exists(self.root):
            return PrunePlan(root=self.root, policy=policy)
        scan = _Scan()
        targets: list[PruneTarget] = []
        now = time.time()

        def propose(directory: Path, verdict: MarkerVerdict) -> None:
            if lock_is_held(directory):
                scan.skipped.append(
                    DirectorySkip(directory, "locked", "another run holds this bundle key")
                )
                return
            targets.extend(self._bundle_targets(directory, verdict, policy=policy, now=now))

        self._walk(self.root, depth=0, scan=scan, on_bundle=propose)
        plan = PrunePlan(
            root=self.root,
            policy=policy,
            targets=tuple(targets),
            skipped=tuple(scan.skipped),
            considered=scan.considered,
        )
        _bundle_log(
            "prune_planned",
            root=str(self.root),
            considered=plan.considered,
            candidate_count=len(plan.targets),
            skipped_count=len(plan.skipped),
        )
        return plan

    def apply_prune(self, plan: PrunePlan) -> PruneOutcome:
        """Delete what the plan proposes and the lock still agrees with (R-10).

        Every target is revalidated while its bundle's lock is held, and the
        revalidation is a full re-derivation rather than a spot check: marker,
        **active generation**, staging liveness and root confinement all come
        back out of the same code that produced the plan, so a target the
        current state would not propose is not deleted whatever the plan says.

        The lock is what makes staging liveness knowable at all (R-06): holding
        it *is* the proof that the run which wrote a **staging directory** is
        gone, where a timestamp is only a guess about it. A bundle whose lock
        another run holds is skipped and reported, never waited for - prune is
        maintenance, and blocking it behind a 40-minute run buys nothing.
        """
        results: list[PruneResult] = []
        now = time.time()
        for bundle_root, targets in _grouped_by_bundle(plan.targets):
            try:
                # Before the lock path is derived from it, let alone created:
                # a bundle root outside the output root would put a `_locks`
                # directory somewhere this store has no business writing, and
                # every target under it is outside the root by construction.
                confined_path(bundle_root, self.root)
            except DistillError:
                results.extend(
                    PruneResult(target, "skipped", "target is not confined to the output root")
                    for target in targets
                )
                continue
            lock_path = prune_lock_path(bundle_root)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = ExclusiveLock.take(bundle_root.name, lock_path)
            if lock is None:
                results.extend(
                    PruneResult(target, "skipped", "another run holds this bundle key")
                    for target in targets
                )
                continue
            try:
                verdict = read_marker(bundle_root)
                authoritative = self._bundle_targets(
                    bundle_root, verdict, policy=plan.policy, now=now
                )
                allowed = {target.path for target in authoritative}
                for target in targets:
                    results.append(self._delete_target(target, verdict, allowed))
            finally:
                lock.release()

        for result in results:
            _bundle_log(
                f"prune_{result.verdict}",
                path=str(result.target.path),
                kind=result.target.kind,
                rule=result.target.rule,
                bundle_key=result.target.bundle_key,
                reason=result.reason,
            )
        outcome = PruneOutcome(
            root=self.root,
            results=tuple(results),
            skipped=plan.skipped,
            considered=plan.considered,
        )
        _bundle_log(
            "prune_applied",
            root=str(self.root),
            considered=outcome.considered,
            deleted_count=len(outcome.deleted),
            retained_count=len(outcome.retained),
            skipped_count=len(outcome.skipped),
        )
        return outcome

    def _delete_target(
        self, target: PruneTarget, verdict: MarkerVerdict, allowed: set[Path]
    ) -> PruneResult:
        """Remove one target, or say why the state under the lock forbade it."""
        try:
            confined_path(target.path, self.root)
        except DistillError:
            return PruneResult(target, "skipped", "target is not confined to the output root")
        if not target.path.exists():
            return PruneResult(target, "skipped", "target no longer exists")
        active = active_generation_name(verdict)
        if active is not None and target.path == target.bundle_root / active:
            return PruneResult(target, "skipped", "target is now the active generation")
        if target.path not in allowed:
            return PruneResult(
                target,
                "skipped",
                f"revalidation under lock no longer proposes this target ({verdict.reason})",
            )
        try:
            shutil.rmtree(target.path)
        except OSError as exc:
            # Removing an entry needs write on its parent, which nothing above
            # proved and a user may have taken away. One target Distill cannot
            # remove must not cost the report on every target it did - the
            # command that deletes is the one that most has to say what it did.
            return PruneResult(target, "skipped", f"removal failed: {errno_name(exc)}")
        return PruneResult(target, "deleted", target.reason)

    def survey(self) -> StoreSurvey:
        """Describe every **bundle** under the root, mutating nothing (R-57).

        The read-only half of the same walk `plan_prune` uses, so what an
        inspection reports and what a prune would consider are the same set of
        directories by construction. It creates nothing - not the root, not a
        `_locks` directory, not a lock file - because a command whose whole
        purpose is to be safe to run first must be safe to run against a path
        the user mistyped.

        Liveness is asked of the kernel, never of a timestamp: a lock file
        outlives its holder by design, so only `flock` separates a run in
        progress from a leftover (R-06).

        A root that cannot be reached at all is refused rather than reported as
        absent - see `_root_directory_exists`.
        """
        if not _root_directory_exists(self.root):
            return StoreSurvey(root=self.root, root_exists=False)
        scan = _Scan()
        bundles: list[BundleReport] = []

        def describe(directory: Path, verdict: MarkerVerdict) -> None:
            bundles.append(self._bundle_report(directory, verdict))

        self._walk(self.root, depth=0, scan=scan, on_bundle=describe)
        return StoreSurvey(
            root=self.root,
            root_exists=True,
            bundles=tuple(bundles),
            skipped=tuple(scan.skipped),
            locks=tuple(scan.locks),
            considered=scan.considered,
        )

    def _bundle_report(self, bundle_root: Path, verdict: MarkerVerdict) -> BundleReport:
        """Describe one bundle: its marker, its generations, its lock."""
        active = active_generation_name(verdict)
        generations = sorted_generations(bundle_root)
        return BundleReport(
            path=bundle_root,
            bundle_key=verdict.bundle_key or bundle_root.name,
            verdict=verdict.kind,
            reason=verdict.reason,
            active_generation=active,
            generations=tuple(path.name for path in generations),
            orphan_generations=tuple(
                path.name for path in orphan_generations(bundle_root, verdict)
            ),
            staging_directories=tuple(path.name for path in staging_directories(bundle_root)),
            lock=lock_report(bundle_root.name, prune_lock_path(bundle_root)),
        )

    def _walk(
        self,
        directory: Path,
        *,
        depth: int,
        scan: _Scan,
        on_bundle: Callable[[Path, MarkerVerdict], None],
    ) -> None:
        """Walk one level, handing each **bundle** to `on_bundle` and skipping the rest.

        The descent is the policy R-05 and R-01 state together: descend through
        directories that are not bundles so a bundle under a playlist root is
        reachable, stop at every directory that is one because a **generation**
        is not a nested bundle, and stop at every directory whose marker says
        Distill is not free to reason about what is underneath it.

        A directory that cannot be listed stops the descent there and is
        reported, rather than ending the walk: one unreadable directory under a
        root must not cost a user the report on everything else in it.
        """
        entries, unlistable = _listing(directory)
        if unlistable is not None:
            scan.skipped.append(DirectorySkip(directory, "unlistable", unlistable))
            return
        for child in entries:
            try:
                child_kind = _entry_kind(child)
            except OSError as exc:
                # Listing a directory needs read and stat'ing what is in it
                # needs execute, so a listing that succeeded proves nothing
                # about the entries it returned.
                scan.considered += 1
                scan.skipped.append(
                    DirectorySkip(
                        child, "unreadable", f"entry could not be read: {errno_name(exc)}"
                    )
                )
                continue
            if child_kind == "symlink":
                scan.considered += 1
                scan.skipped.append(
                    DirectorySkip(child, "symlink", "a symlinked directory is never followed")
                )
                continue
            if child_kind != "directory":
                continue
            scan.considered += 1
            if child.name.startswith(("_", ".")):
                if child.name == LOCK_DIR_NAME:
                    scan.locks.extend(lock_reports(child))
                scan.skipped.append(
                    DirectorySkip(child, "reserved", "reserved name, not a bundle directory")
                )
                continue
            verdict = read_marker(child)
            if verdict.is_distill_owned:
                on_bundle(child, verdict)
                continue
            scan.skipped.append(DirectorySkip(child, verdict.kind, _skip_reason(verdict)))
            if verdict.kind != "absent":
                # A manifest that is unreadable or records another directory's
                # identity still says Distill is not free to reason about what is
                # underneath it (R-01).
                continue
            if depth + 1 >= PRUNE_MAX_DEPTH:
                scan.skipped.append(
                    DirectorySkip(child, "too-deep", f"deeper than {PRUNE_MAX_DEPTH} levels")
                )
                continue
            self._walk(child, depth=depth + 1, scan=scan, on_bundle=on_bundle)

    def _bundle_targets(
        self,
        bundle_root: Path,
        verdict: MarkerVerdict,
        *,
        policy: PrunePolicy,
        now: float,
    ) -> list[PruneTarget]:
        """Everything `policy` proposes to remove from one bundle.

        The single derivation both `plan_prune` and `apply_prune` run, which is
        what makes a stale plan harmless: applying re-derives rather than trusts.

        Liveness is the caller's question, not this one's: the plan probes the
        lock before asking, and `apply_prune` already holds it - asking again
        from the process that holds it would be refused by its own hold, and
        every target would be dropped for the wrong reason.
        """
        if not verdict.is_distill_owned:
            return []

        key = bundle_root.name
        targets: list[PruneTarget] = []
        if verdict.kind == "owned":
            # An ownership marker and no manifest: nothing here has ever been
            # published, so there is no generation a reader is entitled to and
            # no manifest that could name one later (RV-9).
            return [
                PruneTarget(
                    path=bundle_root,
                    kind="bundle",
                    rule="orphan",
                    bundle_root=bundle_root,
                    bundle_key=key,
                    reason="distill-owned directory has never published a generation",
                )
            ]

        changed_at = bundle_mtime(bundle_root)
        if policy.max_age_days is not None and changed_at is not None:
            # An age nobody could learn proposes nothing: expiry is the one rule
            # that deletes an **active generation**, so it acts on a measured
            # age or not at all.
            age_sec = now - changed_at
            if age_sec > policy.max_age_days * SECONDS_PER_DAY:
                # Expiry takes the bundle whole, active generation included
                # (D-018). Retention has nothing left to say about it.
                return [
                    PruneTarget(
                        path=bundle_root,
                        kind="bundle",
                        rule="expiry",
                        bundle_root=bundle_root,
                        bundle_key=key,
                        reason=(
                            f"whole bundle unchanged for {age_sec / SECONDS_PER_DAY:.1f} days, "
                            f"past the {policy.max_age_days} day horizon"
                        ),
                    )
                ]

        generations = sorted_generations(bundle_root)
        active = active_generation_name(verdict)
        if active is not None and (bundle_root / active).is_dir():
            # Retention: the newest `keep_generations`, and the active
            # generation whether or not it is among them (R-02). The active
            # generation is subtracted from the candidates before the count is
            # applied, so no value of `keep_generations` can propose it.
            retained = set(generations[-policy.keep_generations :])
            retained.add(bundle_root / active)
            targets.extend(
                PruneTarget(
                    path=generation,
                    kind="generation",
                    rule="retention",
                    bundle_root=bundle_root,
                    bundle_key=key,
                    reason=(
                        "superseded generation beyond the newest "
                        f"{policy.keep_generations} kept"
                    ),
                )
                for generation in generations
                if generation not in retained
            )
        else:
            targets.extend(
                PruneTarget(
                    path=generation,
                    kind="generation",
                    rule="orphan",
                    bundle_root=bundle_root,
                    bundle_key=key,
                    reason="no manifest names this generation, so nothing can serve it",
                )
                for generation in generations
            )

        targets.extend(
            PruneTarget(
                path=staging,
                kind="staging",
                rule="staging",
                bundle_root=bundle_root,
                bundle_key=key,
                reason="staging directory whose run no longer holds the bundle lock",
            )
            for staging in staging_directories(bundle_root)
        )
        return targets


@dataclass
class BundleRun:
    """One run's exclusive hold on a **bundle**, from `begin` to `commit`.

    The hold lasts as long as this object is in use, not as long as staging
    takes: every stage after staging - the download, the vision pass, the
    publish - runs against a **bundle key** no other run may take. It ends one
    of two ways: `commit`, which publishes what was staged, or `abandon`, which
    gives the run up and says why.

    Does not own what a stage computes, or what goes into the **manifest** it is
    handed: the run assembles the generation's files, and this object decides
    when they stop being scratch.
    """

    store: BundleStore
    bundle_key: str
    paths: BundlePaths
    lock: ExclusiveLock
    staged_at: float = 0.0
    waited_sec: float = 0.0
    resumed: bool = True

    def release(self) -> None:
        """End the hold. Idempotent, and safe to call from a failure path."""
        self.lock.release()

    def _release_without_replacing(self, failure: BaseException) -> None:
        """Release while `failure` is travelling, and never take its place.

        Cleanup that raises during an exception *substitutes* that exception for
        its own. Here the exception being replaced was the run's real diagnosis
        and the replacement is a note about a descriptor: an operator's `Ctrl-C`
        reached the CLI boundary as `E_INTERNAL` saying an unexpected
        `RuntimeError` ended the command, which is wrong twice - the run was not
        interrupted by a defect, and the interrupt was not Distill's to relabel.

        The release failure is not dropped, because a lock this process believes
        it released and did not is a **bundle key** no later run can take. It is
        recorded, beside the failure it declined to replace.

        `Exception`, so a `BaseException` raised *by the release* still
        propagates - a second `Ctrl-C` landing inside cleanup, say. Swallowing an
        interrupt in order to preserve an earlier one is not an improvement on
        losing the earlier one, and an operator interrupting twice is asking
        harder rather than asking again.
        """
        try:
            self.release()
        except Exception as release_failure:
            _bundle_log(
                "lock_release_failed",
                bundle_key=self.bundle_key,
                error=repr(release_failure),
                during=type(failure).__name__,
            )

    def __enter__(self) -> BundleRun:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        """End the hold, and let whatever is travelling keep travelling.

        A release failure with nothing in flight is raised normally: it is the
        only report that a lock was not given up.
        """
        if exc is None:
            self.release()
        else:
            self._release_without_replacing(exc)

    @property
    def staging_duration_sec(self) -> float:
        """How long this run has spent assembling its **staging directory**."""
        return self.store.clock() - self.staged_at

    def read_stage(self, name: str) -> dict[str, Any] | None:
        """The recorded **stage result** for `name`, or `None` to recompute it.

        `None` for every reason a stage result cannot be used: the run is not
        resuming, nothing was recorded, what was recorded is unreadable, or it
        fails the checks R-23 puts a resumed document through. A stage result is
        scratch - recomputing it is always available, so nothing about it is
        worth ending a run over (D-030).

        The **bundle key** the document is checked against is this run's, and
        the root paths inside it are confined to is this bundle's: the run
        asking the question is the only thing entitled to answer what "belongs
        to this bundle" means.
        """
        if not self.resumed:
            return None
        return read_stage_result(
            self.paths.generation, name, bundle_key=self.bundle_key, root=self.paths.root
        )

    def discard_stage(self, name: str, reason: str) -> None:
        """Record that this run read a **stage result** it cannot use.

        The other half of `read_stage`, for the rejections this module is not
        entitled to make. What a payload *means* is the stage's - R-23 checks
        the envelope and hands the payload back unexamined - so a consumer that
        finds the shape inside it is not the shape it produces is the only one
        that can say so. Where that is *recorded* is still here, in the one
        stream that answers what a run did with a **bundle key**: a discard the
        store logs and a discard the pipeline logs are the same event, and a
        second implementation of it in the caller would be a second answer.

        Recording only. Discarding a stage result *is* recomputing its stage,
        which the caller then does - nothing here removes the file, because the
        recomputation overwrites it and a run that dies before then is better
        off leaving the next one something to check than nothing at all.
        """
        _reject_stage_result(name, self.bundle_key, reason)

    @property
    def generation_name(self) -> str:
        """What this run's **staging directory** will be called once published.

        Known before the publish because `next_generation` picked it when
        staging opened, and answerable from here so a caller reporting progress
        does not peel the staging prefix off a directory name itself (D-041).
        """
        return published_paths(self.paths).generation.name

    @property
    def scratch_dir(self) -> Path:
        """Where a stage may put working files that are not bundle content.

        A directory *inside* the **staging directory**, created on demand: the
        run holds the lock on it, an interrupted run finds what it left there
        when it **resumes**, and the publish removes it rather than renaming it
        into the **generation**.

        The staging directory itself was handed out before, which made every
        file a stage wrote bundle content unless it happened to be named like a
        **stage result** - and the one file the transcription stage writes, a
        decode of the whole source's audio, is not (finding 3-opus).
        """
        return ensure_safe_directory(self.paths.generation / SCRATCH_DIR_NAME, self.paths.root)

    @property
    def frames_dir(self) -> Path:
        """Where this run's **keyframe** images go.

        Handed out rather than derived by the caller: `frames/` is layout, and
        the stage that fills it has no business knowing the name (D-041).
        """
        return self.paths.frames

    def write_render(self, markdown: str) -> Path:
        """Write the **render** into the staging directory.

        Checked before it is written, not after: R-16 refuses to follow a
        symlink at the target, so a link pre-created at the render's path cannot
        redirect the write out of the bundle (S1).
        """
        ensure_safe_directory(self.paths.markdown, self.paths.root, create_leaf=False)
        return EMITTER.emit(self.paths.markdown, markdown)

    def write_transcript(self, transcript: Transcript) -> Path:
        """Write the **transcript** into the staging directory, on the same terms.

        A carrier and not a document, because `serialize` is where R-20 is
        enforced: a transcript whose **redaction** policy never ran is refused
        here rather than written and refused later by nobody. A caller holding
        a document has bypassed the check already, so it cannot be allowed to
        hand one in (finding 15).
        """
        ensure_safe_directory(self.paths.transcript, self.paths.root, create_leaf=False)
        document = serialize(transcript)
        return EMITTER.emit(
            self.paths.transcript, json.dumps(document, indent=2, sort_keys=True) + "\n"
        )

    def write_stage(
        self,
        name: str,
        result: Any,
        *,
        redaction: RedactionState = RedactionState.NOT_APPLIED,
    ) -> None:
        """Record a completed stage so an interrupted run can **resume**.

        An ordinary write, deliberately (D-033). The target is inside a
        **staging directory** held under this run's lock: no other process may
        read it, and no reader is ever entitled to it, so there is nobody an
        atomic replace would protect. A torn stage result costs the one run that
        wrote it a recomputation.

        The **bundle key** stamped into the document is this run's, which is
        what `read_stage` later checks it against (R-23). `redaction` is the
        caller's policy, because which policy a run is under is the pipeline's
        knowledge and not the store's; the default runs it.
        """
        write_stage_result(
            self.paths.generation,
            name,
            result,
            root=self.paths.root,
            bundle_key=self.bundle_key,
            redaction=redaction,
        )

    def commit(self, manifest: dict[str, Any]) -> BundleSnapshot:
        """**Publish** the staging directory as the **active generation**.

        Ends the run: the lock is released once the manifest names the new
        generation, because the question it was taken to answer has been
        answered. A failure leaves the lock held for the caller to release,
        along with the staging directory the next run resumes from.
        """
        final_paths = publish_staging(self.paths, manifest)
        _bundle_log(
            "generation_committed",
            bundle_key=self.bundle_key,
            generation=final_paths.generation.name,
            staging_duration_sec=self.staging_duration_sec,
        )
        self.release()
        return BundleSnapshot(
            root=final_paths.root,
            bundle_key=self.bundle_key,
            generation=final_paths.generation,
            manifest=published_manifest(manifest, final_paths),
        )

    def abandon(self, reason: str, *, during: BaseException | None = None) -> None:
        """Give up the run, leaving the previous **active generation** intact.

        The **staging directory** stays: its **stage results** are what a later
        run **resumes** from, and reclaiming them is prune's decision rather
        than a failing run's. Nothing under the bundle root is touched, so a
        bundle that already had an **active generation** still has it.

        The reason is the record: a bundle that did not change is otherwise
        indistinguishable from a run that never happened.

        `during` is the failure this abandonment is cleaning up after, when
        there is one. Named, because a release that raises while it travels
        would replace it - see `_release_without_replacing`.
        """
        _bundle_log(
            "run_abandoned",
            bundle_key=self.bundle_key,
            reason=reason,
            staging=str(self.paths.generation),
            staging_duration_sec=self.staging_duration_sec,
        )
        if during is None:
            self.release()
        else:
            self._release_without_replacing(during)


UNRECLAIMABLE_NOTE = (
    "no rule can reclaim this directory - remove it by hand once you have "
    "checked it is Distill's"
)
"""What a user has to be told about a directory prune will never propose.

An `invalid` marker is a directory whose identity cannot be established, and
nothing here reclaims one: **retention** and **expiry** both act on a bundle
this walk recognized, and the walk does not even descend past it. That refusal
is deliberate - a file named `_manifest.json` in a user's own directory would
otherwise make it deletable, which is finding 1 - so the answer is not to widen
what prune deletes but to say plainly that this one is permanent until somebody
looks at it (finding 9-opus, R-57).
"""


def _skip_reason(verdict: MarkerVerdict) -> str:
    """The reason a skip carries: what was found, and what follows from it.

    Only the verdicts no rule can act on gain the consequence. `absent` is an
    ordinary directory the walk descends through and `foreign` names the
    identity that was recorded, which is already actionable.
    """
    if verdict.kind in ("invalid", "unreadable"):
        return f"{verdict.reason}; {UNRECLAIMABLE_NOTE}"
    return verdict.reason


def _file_state(path: Path) -> FileState:
    """What is at `path`: an ordinary file, something else, or nothing.

    `lstat` rather than `Path.is_file()`, which follows symlinks - so a link
    named `_owner.json` pointing at a marker Distill really did write made a
    directory Distill never wrote recognizable as its own, and a link at
    `video.md` let a publish serve bytes from outside the bundle (finding 1b).
    The question recognition asks is about the name, not about what it points at.

    Raises `OSError` for anything but "not there", because a directory this
    process may not stat into is a different answer from one holding no such
    file, and guessing between them is how a permission problem became a
    deletion or a crash.
    """
    try:
        info = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return "absent"
    return "regular" if stat.S_ISREG(info.st_mode) else "irregular"


def _root_directory_exists(root: Path) -> bool:
    """Whether there is a directory at `root`, refusing to guess when it cannot be asked.

    `Path.is_dir()` cannot answer this, and answers it differently on different
    interpreters, which is how the difference stayed hidden. Reaching a path
    needs execute on every directory above it, so a root whose parent this
    process may not search cannot be stat'ed at all - and `is_dir()` reports
    that as `False` on Python 3.14 (where it delegates to `os.path.isdir`, which
    swallows every `OSError`) and raises `PermissionError` on 3.13 (where only
    the not-there errnos are ignored). One interpreter answered "no root here"
    about a directory that may hold every bundle the user owns; the other ended
    the read-only command with an internal fault.

    Neither is the answer, because neither is known. This is `_file_state`'s rule
    applied to the root: "not there" and "may not be asked about" are different
    facts, and guessing between them is how a permission problem became a crash
    or a deletion. The first is reported (`root_exists: false`, an empty plan);
    the second is refused, because a report about a root nobody could reach is
    a claim with nothing behind it (D-022).

    Refused rather than recorded as a **skip with a reason**, which is what a
    directory *under* the root gets: a skip exists so one unreadable directory
    does not cost the report on all the others, and when the root itself cannot
    be reached there are no others - there is no report left to save.
    """
    try:
        info = root.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise DistillError(
            "E_OUTPUT_ROOT_UNREADABLE",
            "bundle",
            "output root could not be read",
            {"root": str(root), "errno": errno_name(exc)},
        ) from exc
    return stat.S_ISDIR(info.st_mode)


def _entry_kind(directory_entry: Path) -> EntryKind:
    """What one entry of a listing is, from a single `lstat` that follows nothing.

    One question rather than the `is_symlink()`-then-`is_dir()` pair it replaces:
    those are two stats of a path that may change between them, and both refuse
    an entry this process may not stat by raising rather than by answering.
    Raising is the right shape here - the walk has one place to put a reason -
    but only if somebody catches it.
    """
    info = directory_entry.lstat()
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    return "directory" if stat.S_ISDIR(info.st_mode) else "other"


def _listing(directory: Path) -> tuple[list[Path], str | None]:
    """Everything directly under `directory`, or nothing and why.

    The one place that knows a directory listing can fail, and the reason every
    layout question below is total rather than throwing. A failed listing is not
    exceptional here: prune walks a tree a *user* chose, so a directory this
    process cannot read is ordinary, and two prunes over one root routinely
    descend into a bundle the other already removed. Left bare, one such
    directory ended `cleanup-cache` and `cache-doctor` alike - and `cache-doctor`
    is the read-only command that exists *because* the destructive ones were
    unpreviewable (finding 2-opus).

    An empty listing with a reason attached is the shape R-57 already requires of
    every skip, so the walk has somewhere to put it.
    """
    try:
        return sorted(directory.iterdir()), None
    except OSError as exc:
        return [], f"directory could not be listed: {errno_name(exc)}"


def prune_lock_path(bundle_root: Path) -> Path:
    """The run lock for the bundle at `bundle_root`. Derives, creates nothing.

    Stated against the directory rather than the **bundle key** because prune
    finds bundles by walking, including bundles under a playlist root that a
    different `BundleStore` produced. The rule is the same one `begin` uses -
    `_locks/<name>.lock` beside the bundle - so both reach the same file, which
    is the only reason prune's lock excludes a live run at all.
    """
    return bundle_root.parent / LOCK_DIR_NAME / f"{bundle_root.name}{LOCK_SUFFIX}"


def lock_is_held(bundle_root: Path) -> bool:
    """Whether a run is live on this bundle, asked of the lock rather than a clock.

    R-06: liveness is the same question `begin` answers, so the answer is the
    same lock. A timestamp cannot answer it - a **staging directory** that has
    not been written to for an hour belongs either to a dead run or to a live one
    doing a long download, and a staleness window picks between them by guessing
    (finding 11's shape).

    A lock file that does not exist means no run has ever reached staging here,
    so nothing is created to find that out: a plan mutates nothing, and the
    probe is what guarantees it rather than an existence check the answer could
    race.

    A lock the probe could not open counts as held. Prune is maintenance and
    skipping costs a user disk; deciding a bundle is free because the question
    was refused costs them the bundle.
    """
    return ExclusiveLock.probe(prune_lock_path(bundle_root)) in ("live", "unknown")


def lock_report(subject: str, path: Path) -> LockReport:
    """Whether `path` is a lock somebody holds, one nobody holds, or no lock at all.

    Asking is itself read-only: a lock file that is not there is reported absent
    rather than created, so an inspection of a root no run has ever touched
    leaves it exactly as it was (R-57).
    """
    return LockReport(path=path, subject=subject, state=ExclusiveLock.probe(path))


def lock_reports(lock_dir: Path) -> list[LockReport]:
    """Every run lock in one `_locks` directory, held or leftover.

    Reported alongside the per-bundle state because the two answer different
    questions: a bundle's own lock says whether a run is live on it, while a
    lock here whose bundle is gone is disk nothing will ever reclaim - prune
    treats `_locks` as a reserved name and never descends into it.
    """
    return sorted(
        (
            lock_report(path.name.removesuffix(LOCK_SUFFIX), path)
            for path in _listing(lock_dir)[0]
            if _is_lock_file(path)
        ),
        key=lambda report: report.path,
    )


def _is_lock_file(path: Path) -> bool:
    """Whether `path` is a run lock this walk can report on.

    An entry that cannot be stat'ed is not one: unlike a marker, where "may not
    look" and "not there" lead to different verdicts, a lock nobody can ask
    about is a lock nobody can report, and refusing to answer must not end an
    inspection of everything else under the root.
    """
    if not path.name.endswith(LOCK_SUFFIX):
        return False
    try:
        return _file_state(path) == "regular"
    except OSError:
        return False


def _listed_kind(directory_entry: Path) -> EntryKind | None:
    """`_entry_kind` for a listing walked without a place to put a reason.

    `None` where the walk would record a skip: an entry a listing returned can
    still refuse to be stat'ed, because listing needs read on the directory and
    stat'ing what is in it needs execute. Every layout question below wants the
    same answer for that - leave it out - since a generation nobody can stat is
    one nothing may propose deleting, and a bundle nobody can describe is
    reported empty rather than not at all.
    """
    try:
        return _entry_kind(directory_entry)
    except OSError:
        return None


def sorted_generations(bundle_root: Path) -> list[Path]:
    """Every **generation** directory under `bundle_root`, oldest first.

    A bundle root that cannot be listed - gone, or unreadable - has no
    generations rather than raising, so a prune racing another prune over the
    same root reports a bundle it can no longer see instead of ending.
    """
    return sorted(
        (
            path
            for path in _listing(bundle_root)[0]
            if _listed_kind(path) == "directory" and is_generation_name(path.name)
        ),
        key=lambda path: int(path.name[len(GENERATION_PREFIX) :]),
    )


def staging_directories(bundle_root: Path) -> list[Path]:
    """Every **staging directory** left under `bundle_root`. Total, as above."""
    return sorted(
        path
        for path in _listing(bundle_root)[0]
        if _listed_kind(path) == "directory" and path.name.startswith(STAGING_PREFIX)
    )


def active_generation_name(verdict: MarkerVerdict) -> str | None:
    """The **active generation** a marker names, or `None` if it names none."""
    if not verdict.is_bundle or verdict.manifest is None:
        return None
    name = verdict.manifest.get("active_generation")
    if isinstance(name, str) and is_generation_name(name):
        return name
    return None


def bundle_mtime(bundle_root: Path) -> float | None:
    """When this **bundle** last changed, or `None` if that cannot be learned.

    Expiry is about a bundle nobody has produced into for a while, so the
    youngest of the **manifest** and the **generations** is the age that matters:
    a bundle whose oldest generation is a year old but which was republished
    yesterday is not aged. The old code read the bundle directory's own mtime,
    which changes when anything at all is created beside it.

    Every path this asks about may stop answering between the marker read and
    this question - another prune removing the bundle, or one refused `stat` -
    so each is guarded and the listing behind `sorted_generations` is total. The
    answer for "nothing would say" is `None` and not `0.0`: the epoch is not
    "unknown", it is *older than everything*, and an age is the sole input on
    which **expiry** deletes an entire bundle including its **active
    generation**. Unknown must propose nothing.
    """
    candidates = [bundle_root, bundle_root / MANIFEST_NAME, *sorted_generations(bundle_root)]
    times = []
    for path in candidates:
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def _grouped_by_bundle(
    targets: Iterable[PruneTarget],
) -> list[tuple[Path, list[PruneTarget]]]:
    """Targets grouped by bundle, in first-appearance order.

    One lock per bundle rather than one per target: the lock is the bundle's, so
    taking it again per generation would only widen the window in which a run
    could slip between two deletions in the same bundle.
    """
    grouped: dict[Path, list[PruneTarget]] = {}
    for target in targets:
        grouped.setdefault(target.bundle_root, []).append(target)
    return list(grouped.items())


def write_ownership_marker(directory: Path, bundle_key: str) -> Path:
    """Claim `directory` as Distill's before anything is written into it (R-11).

    The recorded **bundle key** is the claim, not the filename: `read_marker`
    parses this file and compares what it records to the directory's name, so a
    payload naming anything else claims nothing. That is what keeps a file some
    other tool happens to call `_owner.json` from making a user's directory
    prunable.

    Written in place rather than by atomic replace. An atomic replace would keep
    the claim in a temporary file until the rename completed, which is the window
    RV-9 is about, and the failure it would avoid is not a failure here: a
    torn marker records no bundle key, so it reads as invalid and the directory
    is skipped rather than reclaimed. Unreclaimable is the safe side of that
    trade, and it is reported with its reason (R-57).

    Rewritten on every `begin`, so the recorded pid is the run that holds the
    bundle now rather than the first one that ever did.
    """
    marker = directory / OWNERSHIP_MARKER_NAME
    ensure_safe_directory(marker, directory, create_leaf=False)
    return EMITTER.emit(
        marker,
        json.dumps(
            {
                "bundle_key": bundle_key,
                "pid": os.getpid(),
                "claimed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            sort_keys=True,
        )
        + "\n",
    )


def read_marker(directory: Path) -> MarkerVerdict:
    """Decide whether `directory` carries a **bundle marker**, and which one.

    Every failure is a verdict rather than an exception: an unmarked or
    unreadable directory may not be Distill's at all, and refusing to claim it is
    the whole point (R-01). A malformed manifest is likewise not a reason to end
    a run - it is a reason to rebuild the bundle.

    Both marker kinds prove the same thing on the same terms: a non-symlink
    regular file, parseable as a JSON object, recording a **bundle key** equal to
    this directory's name. The **published marker** always did. The **ownership
    marker** proved only that a file of that name existed - never opened, never
    parsed, never compared - so any tool's `_owner.json` sitting in a user's
    directory made that directory Distill-owned, and `_bundle_targets` proposes
    the *bundle root* for a directory that never published. That is finding 1
    surviving at a narrower filename, inside the operation that closed it, and
    the active-generation guard cannot help when the target is the root itself.

    Identity is what is being proved, so it is proved once here for both files
    rather than trusted from the one whose contents nobody read.
    """
    try:
        manifest_state = _file_state(directory / MANIFEST_NAME)
        owner_state = _file_state(directory / OWNERSHIP_MARKER_NAME)
    except OSError as exc:
        return MarkerVerdict(
            kind="unreadable",
            reason=f"directory could not be read: {errno_name(exc)}",
        )

    for name, state in (
        (MANIFEST_NAME, manifest_state),
        (OWNERSHIP_MARKER_NAME, owner_state),
    ):
        if state == "irregular":
            # A link named like a marker is refused rather than ignored: it is
            # not "no marker", and descending past it would be treating a
            # directory somebody planted a marker name in as ordinary (1b).
            return MarkerVerdict(kind="invalid", reason=f"{name} is not a regular file")

    if manifest_state == "regular":
        return _published_marker(directory)
    if owner_state == "regular":
        return _ownership_marker(directory)
    return MarkerVerdict(kind="absent", reason="no bundle marker")


def _published_marker(directory: Path) -> MarkerVerdict:
    """Read the **manifest** as a marker: schema-valid, and this directory's."""
    document, why = _marker_document(directory / MANIFEST_NAME)
    if document is None:
        return MarkerVerdict(kind="invalid", reason=why)

    try:
        validate_manifest_schema(document, require_active_generation=True)
    except DistillError as exc:
        return MarkerVerdict(
            kind="invalid",
            reason=f"manifest schema is invalid: {exc.details.get('field', 'unknown field')}",
        )

    identity = recorded_identity(document)
    if identity != directory.name:
        return MarkerVerdict(
            kind="foreign",
            reason=f"manifest records bundle key {identity!r}, not {directory.name!r}",
            bundle_key=identity,
        )
    return MarkerVerdict(
        kind="published",
        reason="manifest records this directory's bundle key",
        bundle_key=identity,
        manifest=document,
    )


def _ownership_marker(directory: Path) -> MarkerVerdict:
    """Read the **ownership marker**: a **bundle key**, and this directory's.

    No schema beyond identity, because that is all the file claims - a run that
    crashed before publishing wrote nothing else worth trusting. Identity is not
    optional though: it is the whole difference between "Distill made this
    directory" and "a file with this name is in it".
    """
    document, why = _marker_document(directory / OWNERSHIP_MARKER_NAME)
    if document is None:
        return MarkerVerdict(kind="invalid", reason=why)

    identity = recorded_identity(document)
    if identity is None:
        return MarkerVerdict(
            kind="invalid",
            reason=f"{OWNERSHIP_MARKER_NAME} records no bundle key",
        )
    if identity != directory.name:
        return MarkerVerdict(
            kind="foreign",
            reason=(
                f"{OWNERSHIP_MARKER_NAME} records bundle key {identity!r}, "
                f"not {directory.name!r}"
            ),
            bundle_key=identity,
        )
    return MarkerVerdict(
        kind="owned",
        reason="ownership marker present, nothing published yet",
        bundle_key=identity,
    )


def _marker_document(path: Path) -> tuple[dict[str, Any] | None, str]:
    """The JSON object a marker holds, or `None` and why it is not one."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, f"{path.name} is not readable JSON"
    if not isinstance(document, dict):
        return None, f"{path.name} is not a JSON object"
    return document, ""


def recorded_identity(manifest: dict[str, Any]) -> str | None:
    """The **bundle key** a manifest records, under either accepted field name."""
    for name in IDENTITY_FIELDS:
        value = manifest.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def is_generation_name(name: str) -> bool:
    """Whether `name` names a **generation** (`g1`, `g2`, ...)."""
    return name.startswith(GENERATION_PREFIX) and name[len(GENERATION_PREFIX) :].isdigit()


def validate_manifest_schema(
    manifest: dict[str, Any],
    *,
    require_active_generation: bool,
) -> None:
    """Raise `E_BAD_MANIFEST` unless the manifest carries the fields a bundle needs.

    Identity is checked separately from the typed fields because it has two
    accepted names: the current `bundle_key` and the legacy `source_hash`.
    """
    required_types: dict[str, type | tuple[type, ...]] = {
        "pipeline_version": int,
        "distill_version": str,
        "source_type": str,
        "source_resolved_path": str,
        "duration_sec": (int, float),
        "options": dict,
        "frame_count": int,
        "transcript_present": bool,
        "warning_count": int,
        "frames": list,
        "warnings": list,
    }
    # `provenance` is optional here so bundles published before Phase 2 remain
    # servable. Every newly resolved source supplies the carrier and
    # `manifest_document` writes it, but an old bundle cannot be made to contain
    # provenance without reprocessing its source.
    if require_active_generation:
        required_types["active_generation"] = str
    for key, expected_type in required_types.items():
        value = manifest.get(key)
        if not isinstance(value, expected_type):
            expected_name = (
                " or ".join(item.__name__ for item in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise DistillError(
                "E_BAD_MANIFEST",
                "bundle",
                "cache manifest schema is invalid",
                {"field": key, "expected": expected_name},
            )
    if "provenance" in manifest and not isinstance(manifest["provenance"], dict):
        raise DistillError(
            "E_BAD_MANIFEST",
            "bundle",
            "cache manifest schema is invalid",
            {"field": "provenance", "expected": "dict"},
        )
    if recorded_identity(manifest) is None:
        raise DistillError(
            "E_BAD_MANIFEST",
            "bundle",
            "cache manifest schema is invalid",
            {"field": " or ".join(IDENTITY_FIELDS), "expected": "str"},
        )


# `read_manifest` and `active_paths` stood here. They answered "where would the
# active generation be?" from the manifest alone, without proving the generation
# or its **render** is on disk - which is finding 2's cache hit, handing back
# paths to a directory retention had deleted. `BundleStore.load_active` is the
# only surface that answers "is there a bundle to serve?" (R-04), and D-041
# leaves no caller needing the weaker one.


def next_generation(bundle_root: Path) -> str:
    existing = [
        int(path.name[len(GENERATION_PREFIX) :])
        for path in bundle_root.glob(f"{GENERATION_PREFIX}*")
        if path.is_dir() and is_generation_name(path.name)
    ]
    return f"{GENERATION_PREFIX}{max(existing, default=0) + 1}"


def confined_path(path: Path, root: Path) -> Path:
    """Refuse `path` unless it is under `root` and reached through no symlink.

    The single symlink refusal in Distill (R-16, D-041), and the only thing that
    decides whether a path may be *touched* at all - which is a question a
    deletion asks as sharply as a write does. It creates nothing and follows
    nothing, so prune can ask it about a path it is about to remove (R-10).

    Confinement is decided lexically and the walk is over the components as
    written, not as resolved. Resolving first and then walking inspects a path
    with every symlink already followed, which sees no symlink at all - so a link
    pointing back inside the root passed. Between them the two rules leave no
    third case: `..` cannot survive normalization, and a path that reaches
    outside the root can only do so through a component the walk refuses.
    """
    target = path if path.is_absolute() else root / path
    lexical_root = Path(os.path.normpath(root.absolute()))
    lexical_target = Path(os.path.normpath(target.absolute()))
    if not lexical_target.is_relative_to(lexical_root):
        raise DistillError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "bundle path must stay under output_dir",
            {"path": str(target), "output_root": str(lexical_root)},
        )
    current = lexical_root
    for part in lexical_target.relative_to(lexical_root).parts:
        current = current / part
        if current.is_symlink():
            raise DistillError(
                "E_BAD_OUTPUT_DIR",
                "bundle",
                "output tree must not contain symlink components",
                {"path": str(current), "output_root": str(lexical_root)},
            )
    return target


def ensure_safe_directory(path: Path, root: Path, *, create_leaf: bool = True) -> Path:
    """Validate `path` as a write target under `root`, creating what is missing.

    Confinement is `confined_path`'s answer, checked in full before anything is
    created, so a refused path leaves no half-built tree behind.

    `create_leaf=False` validates a target without creating it, which is what
    lets one checker serve a file write as well as a directory: the parents are
    created, the final component is checked and left alone. A file target went
    unchecked before, so a symlink pre-created at, say, `video.md` redirected the
    write out of the bundle (S1).
    """
    target = confined_path(path, root)
    lexical_root = Path(os.path.normpath(root.absolute()))
    lexical_target = Path(os.path.normpath(target.absolute()))
    relative_parts = lexical_target.relative_to(lexical_root).parts
    current = lexical_root
    leaf_index = len(relative_parts) - 1
    for index, part in enumerate(relative_parts):
        current = current / part
        if create_leaf or index < leaf_index:
            current.mkdir(exist_ok=True)
    return target


def atomic_write_text(path: Path, text: str, *, root: Path) -> None:
    """Confine `path` under `root`, then emit `text` atomically into it.

    The pairing every caller of durable shared state wants: the manifest, the
    job records, the playlist summary, and anything later joining them. Two
    questions, answered by their owners - whether Distill may write here at all,
    which is `ensure_safe_directory`'s and this module's, and how the bytes get
    there without a reader ever seeing half of them, which is the emitter's
    (R-14, D-033).

    The temporary is the emitter's to name, and the check above already covers
    it: it is created in the target's own directory, so its ancestors are the
    ancestors just walked, and its final component is opened `O_NOFOLLOW` - a
    link pre-created at the name is refused by the kernel at the moment of use,
    which is stronger than the lexical check that used to be asked of it and
    then go stale before the open (R-16). It is never mistaken for a marker,
    which is matched by name, and it is reclaimed only when the whole bundle is
    - **expiry** takes a bundle entire, while **retention** acts on
    **generations** and never on a loose file at the bundle root (finding
    7-opus).
    """
    ensure_safe_directory(path, root, create_leaf=False)
    EMITTER.emit_atomically(path, text)


def write_manifest(bundle_root: Path, manifest: dict[str, Any]) -> Path:
    """Replace a bundle's **manifest**, the step that makes a generation active.

    Named rather than inlined into `publish_staging` because it is the last step
    of the publish order and therefore the boundary the crash-survival property
    is stated about (R-12).
    """
    path = bundle_root / MANIFEST_NAME
    atomic_write_text(
        path, json.dumps(manifest, indent=2, sort_keys=True) + "\n", root=bundle_root
    )
    return path


def stage_result_path(generation: Path, name: str) -> Path:
    """Where the **stage result** for `name` lives inside a generation directory.

    Refuses a name outside `STAGE_NAME_RE`, so the path is always one file
    directly inside the generation directory - the only shape the strip that
    holds R-13 can see.
    """
    if not STAGE_NAME_RE.fullmatch(name):
        raise DistillError(
            "E_BAD_STAGE_NAME",
            "bundle",
            "stage name must be alphanumeric with - or _, 64 characters or fewer",
            {"stage_name": name},
        )
    return generation / f"{STAGE_RESULT_PREFIX}{name}{STAGE_RESULT_SUFFIX}"


def read_stage_result(
    generation: Path, name: str, *, bundle_key: str, root: Path
) -> dict[str, Any] | None:
    """The usable **stage result** for `name`, or `None` if there is not one.

    Unusable scratch is a miss, not a failure (D-030): the stage that produced
    it can always produce it again, and a run that ends over its own scratch is
    a run that cannot recover from the interruption the scratch exists for.
    That covers an absent file, an unreadable one, one that is not a file at
    all, and - since R-23 - one whose envelope does not hold up.

    A stage result is what is at the name, not what the name points at, so the
    state is `lstat`'s (`_file_state`), the open is `O_NOFOLLOW`'s, and a
    symlink is refused rather than followed. Reading through one would have let
    a document planted outside the **output root** be resumed from on the
    strength of carrying the right **bundle key**, and `write_stage_result`
    refuses the same link, so nothing this run did would ever have replaced it
    (R-16, boundary review finding 1).

    The guard is over the parse *and* the validation, and it is deliberately
    broad. What is being read is a document some other process wrote, so
    everything about it is input: a nesting depth that exhausts the parser's
    stack, a name too long for the filesystem to answer questions about. There
    is no way for validating an untrusted document to fail that is worth ending
    a run over, and the fallback - do the work - is always available. `except
    Exception` and not `BaseException`, so an interrupt still ends the run.
    """
    path = stage_result_path(generation, name)
    try:
        state = _file_state(path)
    except OSError as exc:
        _reject_stage_result(name, bundle_key, f"unstattable: {errno_name(exc)}")
        return None
    if state == "absent":
        return None
    if state != "regular":
        _reject_stage_result(name, bundle_key, f"not_a_regular_file: {state}")
        return None
    try:
        document = json.loads(_read_regular_file(path))
        return validated_stage_payload(document, stage=name, bundle_key=bundle_key, root=root)
    except Exception as exc:
        _reject_stage_result(name, bundle_key, f"unreadable: {type(exc).__name__}")
        return None


def _reject_stage_result(stage: str, bundle_key: str, reason: str) -> None:
    """Record that a **stage result** was discarded, and why.

    A rejection is invisible from the outside - the run simply does the work -
    so without this a bundle key that can never resume looks like a pipeline
    that is merely slow.
    """
    _bundle_log("stage_result_rejected", bundle_key=bundle_key, stage=stage, reason=reason)


def validated_stage_payload(
    document: Any, *, stage: str, bundle_key: str, root: Path
) -> dict[str, Any] | None:
    """The payload `document` may be resumed from, or `None` to recompute the stage.

    R-23, and the answer to RV-8: a **stage result** is read back off disk by a
    process that did not write it, so everything it claims is a claim. Four of
    them are checked, and any failure discards the whole document:

    - the schema version is the one this code writes. An unknown version is
      refused outright rather than parsed for the fields that happen to be
      recognizable - two schemas can spell different things the same way, so a
      best-effort parse is how a schema change becomes silent corruption;
    - the **bundle key** is this run's. Scratch carrying another key describes
      a different **source fingerprint**, different **options**, or both, and
      reusing it would publish a **generation** for a source that never
      produced it;
    - the stage is the one being read, so a document cannot answer for a stage
      it is not the output of;
    - every path in the payload stays under the bundle root.

    A rejection is never fatal. The caller recomputes (D-030), which is what
    keeps the validation cheaper than the trust it replaced.
    """
    if not isinstance(document, dict):
        _reject_stage_result(stage, bundle_key, "not_a_document")
        return None
    version = document.get("schema_version")
    # `type(...) is not int` rather than `isinstance`, because `True` is an
    # instance of `int` and equals 1: a document whose version field is a
    # boolean would otherwise read as version 1 and be resumed from.
    if type(version) is not int or version != STAGE_RESULT_SCHEMA_VERSION:
        _reject_stage_result(stage, bundle_key, "unknown_schema_version")
        return None
    if document.get("bundle_key") != bundle_key:
        _reject_stage_result(stage, bundle_key, "bundle_key_mismatch")
        return None
    if document.get("stage") != stage:
        _reject_stage_result(stage, bundle_key, "stage_mismatch")
        return None
    payload = document.get("payload")
    if not isinstance(payload, dict):
        _reject_stage_result(stage, bundle_key, "payload_not_a_document")
        return None
    if not _paths_are_confined(payload, root):
        _reject_stage_result(stage, bundle_key, "path_outside_bundle_root")
        return None
    return _with_recorded_warnings(payload, document.get("warnings"))


def _with_recorded_warnings(payload: dict[str, Any], recorded: Any) -> dict[str, Any]:
    """Fold the envelope's **warnings** into the payload a **resume** carries forward.

    The carrier that wrote the document can have changed what it holds - R-58
    caps an extracted-text field at 256 KiB - and the record of that is on the
    envelope, not in the payload. A run resuming from the truncated text is the
    run the warning describes, so it is the run that has to carry it: dropping
    it here would publish a **generation** built from shortened text with
    nothing saying so.

    Only folded when both sides are lists, so a payload using `warnings` for
    something else is passed through rather than rewritten.
    """
    if not isinstance(recorded, list) or not recorded:
        return payload
    existing = payload.get("warnings", [])
    if not isinstance(existing, list):
        return payload
    return {**payload, "warnings": [*existing, *recorded]}


def _paths_are_confined(value: Any, root: Path) -> bool:
    """Whether every path field under `value` stays inside the bundle root.

    Which keys name a path is `artifacts.is_path_field`'s, and deliberately the
    same answer the carrier uses to decide what the **redaction** policy may
    rewrite: a field the policy treats as extracted text and this treats as a
    path is a path the policy is free to mangle into one that fails here.

    `confined_path` decides confinement - the same refusal every write and every
    deletion is put through (R-16, D-041), rather than a second rule that could
    disagree with it. Not `ensure_safe_directory`, because that one *creates* what is
    missing: validating a resumed payload must not leave directories behind, and
    a relative path field would otherwise mkdir its way into the bundle root
    just for being read.

    A relative path is confined against the bundle root, which is where
    `confined_path` joins it - so `frames/frame_0001.png` is inside and
    `../../etc/passwd` is not.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if is_path_field(str(key)) and isinstance(item, str):
                try:
                    confined_path(Path(item), root)
                except DistillError:
                    return False
            elif not _paths_are_confined(item, root):
                return False
        return True
    if isinstance(value, list):
        return all(_paths_are_confined(item, root) for item in value)
    return True


def write_stage_result(
    generation: Path,
    name: str,
    payload: Any,
    *,
    root: Path,
    bundle_key: str,
    redaction: RedactionState = RedactionState.NOT_APPLIED,
) -> None:
    """Record a completed stage as resume scratch. An ordinary write (D-033).

    What lands on disk is a serialized `StageResult`: the payload the stage
    produced, wrapped in the schema version and the **bundle key** that make it
    checkable when some later run reads it back (R-23). The envelope is the
    store's and not the stage's - a stage owns what its payload says, never
    which bundle it belongs to.

    Going through the carrier is also what makes this a covered **redaction
    sink**: a stage result is durable, and finding 4 is what happened when that
    was treated as a detail. `redaction` is the run's policy, so
    `--no-redact-secrets` reaches the scratch it reaches the **generation**
    with, and a resumed run is not handed text a first run would not have had.

    A payload holding carriers is serialized carrier by carrier first, which is
    what puts each of them through R-20's check: a **frame artifact** whose
    redaction policy never ran is refused here, at the write, rather than
    flattened into JSON by a `default=str` that would have made its text durable
    and unreadable at once.

    What is left after that is turned into plain JSON, which is what the write
    did anyway. It matters here because the carrier refuses a type it cannot
    freeze: normalizing first means the stage result caps and redacts exactly
    the strings that are about to be on disk, instead of `write_stage` becoming
    a new way for a run to die on its own scratch.

    Neither is the target. A stage result that cannot be recorded is recorded
    nowhere and costs the *next* **resume** one recomputation, which is what a
    stage result is for; the run that could not write it carries on and
    publishes (D-046, boundary review finding 1). The alternative was a bundle
    key that could never complete again: the write refused a symlink at its
    target with `E_BAD_OUTPUT_DIR`, and since a failing run keeps its **staging
    directory** and `resume_partial` defaults on, the next run reached the same
    link and died the same way.

    What is *not* relaxed is the confinement (R-16). The link is refused, the
    directory is refused, and neither is followed or removed - the target is a
    regular file this process may write, or there is nowhere to record. The
    refusal is asked twice on purpose: once of what is on disk, for a reason an
    operator can read, and once by the kernel at the open, because the path
    could have changed in between.

    Serialization happens first so that R-20 is unconditional: a carrier whose
    redaction policy never ran is refused whatever is at the path, rather than
    slipping through on the one run whose scratch could not be written.
    """
    path = stage_result_path(generation, name)
    document = serialize(
        StageResult(
            schema_version=STAGE_RESULT_SCHEMA_VERSION,
            bundle_key=bundle_key,
            stage=name,
            payload=json.loads(json.dumps(_serialized(payload), default=str)),
            redaction=redaction,
        )
    )
    if not _recordable_stage_result(path, root=root, stage=name, bundle_key=bundle_key):
        return
    try:
        EMITTER.emit(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        # A full disk, a read-only mount, a file this process may not write:
        # scratch that cannot be recorded, on the same terms as a target that
        # was never usable. The run has the payload in hand either way.
        _unrecordable_stage_result(name, bundle_key, f"write_refused: {errno_name(exc)}")


def _recordable_stage_result(
    path: Path, *, root: Path, stage: str, bundle_key: str
) -> bool:
    """Whether a **stage result** may be written at `path`, logging why not.

    Two questions, and a refusal to either is a stage result that goes
    unrecorded rather than a run that ends:

    - confinement, which is `ensure_safe_directory`'s and unchanged - the path
      stays under the bundle root and no component of it is a symlink (R-16);
    - what is already there, which is `lstat`'s. A regular file is overwritten
      and an absent one is created; a directory, a socket or a fifo is not a
      stage result and never becomes one by being written to.

    Asked here rather than left to the emitter, which refuses all three anyway,
    because a refusal has to say *why* to be worth anything: the run carries on
    without its scratch either way, and an operator looking at a bundle key that
    never resumes needs `not_a_regular_file` in the log and not an errno the
    write happened to produce (R-57).
    """
    try:
        ensure_safe_directory(path, root, create_leaf=False)
        state = _file_state(path)
    except DistillError as exc:
        _unrecordable_stage_result(stage, bundle_key, f"path_refused: {exc.code}")
        return False
    except OSError as exc:
        _unrecordable_stage_result(stage, bundle_key, f"unstattable: {errno_name(exc)}")
        return False
    if state == "irregular":
        _unrecordable_stage_result(stage, bundle_key, "not_a_regular_file")
        return False
    return True


def _read_regular_file(path: Path) -> str:
    """Read `path` refusing to be redirected at the open, as the emitter writes it.

    The read half of the window `TextEmitter.emit` covers on the way out: a
    **stage result** is what is at the name, and `O_NOFOLLOW` is what makes that
    true of the bytes rather than of a check that ran first, which is exactly
    what `_recordable_stage_result` cannot be - between an `lstat` and an `open`
    a path can be replaced. `O_NONBLOCK` so a fifo swapped in cannot make the
    read wait for a writer.

    Here rather than in `emit`, because reading a stage result is this module's
    question about its own scratch, and the emitter owns emission.
    """
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)) as handle:
        return handle.read()


def _unrecordable_stage_result(stage: str, bundle_key: str, reason: str) -> None:
    """Record that a **stage result** could not be written, and why.

    The counterpart of `_reject_stage_result`, and needed for the same reason:
    the run carries on, so from the outside a bundle key whose scratch can never
    be recorded is indistinguishable from one that simply never resumes - a
    pipeline that looks slow rather than one whose staging directory somebody
    has to look at.
    """
    _bundle_log("stage_result_not_recorded", bundle_key=bundle_key, stage=stage, reason=reason)


def _serialized(value: Any) -> Any:
    """Replace every carrier under `value` with the document `serialize` allows out.

    A stage hands its result along as carriers, because that is what the stages
    speak to each other (R-19). Turning them into documents is this module's
    job and not theirs: `serialize` is the last check before **extracted text**
    becomes durable (R-20), and a stage that flattened its own carriers would
    be a stage that could skip it.
    """
    if isinstance(value, Carrier):
        return serialize(value)
    if isinstance(value, Mapping):
        return {key: _serialized(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serialized(item) for item in value]
    return value


def strip_stage_results(generation: Path) -> list[str]:
    """Remove every **stage result** under `generation`, returning what went.

    R-13: a **generation** never contains a stage result. The removal walks the
    directory rather than a list of the stages this run ran, because the
    invariant is about what is on disk - a stage result written by an earlier,
    interrupted run being resumed is exactly the one a list would miss.

    This is defense in depth rather than the only protection: R-19 (D-019)
    redacts **extracted text** at carrier construction, so a stage result is not
    unredacted on disk in the first place. Both are needed - one keeps secrets
    out of scratch, this one keeps scratch out of bundles.

    R-13 is about the name, so the strip is too: a directory, a link or a fifo
    called `_ocr.json` is not a stage result and is still an entry a published
    **generation** must not carry. Only regular files went before, which was
    unreachable while `write_stage_result` ended the run over one - and stopped
    being unreachable the moment it stopped doing that. A link is asked about
    before anything else and then removed, never followed or descended: what it
    points at is outside the bundle and none of this module's business (R-16).

    An entry that refuses to be removed fails the **publish**, deliberately.
    R-13 is a promise about what a **generation** contains, and the way to keep
    it when the strip cannot is to not publish - the previous **active
    generation** stays servable and the **staging directory** stays for somebody
    to look at. That is the one thing here that is not scratch's tolerance: a
    stage result that cannot be *written* costs a resume (D-030), while one that
    cannot be *removed* would cost the bundle its invariant.
    """
    removed = []
    for path in sorted(generation.rglob(STAGE_RESULT_GLOB)):
        try:
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
        except FileNotFoundError:
            # Named inside a directory this loop already removed. `rglob` lists
            # a parent before its children and the walk is not re-read.
            continue
        removed.append(path.name)
    return removed


def strip_run_scratch(generation: Path) -> bool:
    """Remove the run's scratch directory, reporting whether there was one.

    The other half of R-13, and the half a name-matched strip cannot reach: a
    **stage result** is scratch Distill named, while what a stage writes into
    `scratch_dir` it names itself. Both are removed by the publish and neither
    survives into a **generation**.

    Removed rather than kept and ignored, because a generation is what a reader
    is served: a decode of the source's audio published beside the **render** is
    disk nobody asked for, and a second copy of that audio inside a bundle whose
    purpose is to make the source unnecessary.
    """
    scratch = generation / SCRATCH_DIR_NAME
    if not scratch.is_dir() or scratch.is_symlink():
        return False
    shutil.rmtree(scratch)
    return True


def publish_staging(paths: BundlePaths, manifest: dict[str, Any]) -> BundlePaths:
    """Turn a **staging directory** into the **active generation**, in order.

    Assemble (the caller's work, already done) -> strip **stage results** and
    run scratch -> rename to `g<N>` -> atomically replace the **manifest**
    (R-12).

    The order is the survivability argument. The strip precedes the rename, so
    there is no instant at which a generation on disk holds scratch. It has two
    halves because scratch does: Distill names a stage result and can match it,
    while a stage names its own working files and only the directory it was
    given to write them in is known (finding 3-opus). The
    manifest replace comes last, so the only gap is a finished `g<N>` that no
    manifest names yet: the previous **active generation** is untouched and
    still servable, and the new directory is an **orphan generation** prune
    reclaims. Reversing the last two would instead point the manifest at a
    directory that does not exist - finding 2's shape, from the other side.

    The **render** is proved on disk first, because everything after it is
    irreversible for the generation being superseded: the rename and the
    manifest replace both happen whatever staging holds, so a commit that never
    called `write_render` published an empty generation over a servable one and
    turned every later read into a cache miss (finding 4-codex). A manifest is a
    promise and the file is the evidence, which is R-04 stated at the moment the
    promise is made rather than at the moment it is read.
    """
    _require_render(paths)
    final_paths = published_paths(paths)
    published = published_manifest(manifest, final_paths)

    strip_stage_results(paths.generation)
    strip_run_scratch(paths.generation)
    ensure_safe_directory(final_paths.generation, paths.root, create_leaf=False)
    paths.generation.rename(final_paths.generation)
    write_manifest(paths.root, published)
    return final_paths


def _require_render(paths: BundlePaths) -> None:
    """Refuse to publish a **staging directory** that holds no **render**.

    The path is *derived* from the directory about to be renamed rather than
    read off `paths.markdown`, because that field proves nothing about what the
    rename will publish: aimed at the previous **generation**'s render it passes
    every check while staging stays empty, and the empty directory becomes the
    **active generation** over a servable one.

    Confined first, so a symlink pre-created at the render's path is refused
    rather than followed (R-16): `write_render` checks its own target, but
    nothing stopped a link being put there afterwards, and a generation whose
    render points out of the bundle serves bytes prune neither owns nor can
    reclaim. Then a regular file, because that is what "there is something to
    serve" means on disk.
    """
    render = paths.generation / RENDER_NAME
    confined_path(render, paths.root)
    try:
        state = _file_state(render)
    except OSError as exc:
        state = "absent"
        LOGGER.debug("render state unreadable: %s", errno_name(exc))
    if state != "regular":
        raise DistillError(
            "E_INCOMPLETE_GENERATION",
            "bundle",
            "a generation must carry a render before it can be published",
            {"render": str(render), "render_state": state},
        )


def published_paths(paths: BundlePaths) -> BundlePaths:
    """Where a **staging directory**'s contents will live once published."""
    generation = paths.root / paths.generation.name.removeprefix(STAGING_PREFIX)
    return BundlePaths(
        root=paths.root,
        generation=generation,
        frames=generation / FRAMES_DIR_NAME,
        manifest=paths.root / MANIFEST_NAME,
        transcript=generation / TRANSCRIPT_NAME,
        markdown=generation / RENDER_NAME,
    )


def published_manifest(manifest: dict[str, Any], final_paths: BundlePaths) -> dict[str, Any]:
    """The **manifest** a publish writes: the run's, addressed to the generation.

    One function rather than a step inside `publish_staging`, so that what a
    commit hands its caller and what is on disk cannot drift into two different
    documents - the frame paths below are rewritten, and a caller reading the
    pre-publish manifest would be reading paths to a directory that no longer
    exists under that name.
    """
    published = dict(manifest)
    published["active_generation"] = final_paths.generation.name
    validate_manifest_schema(published, require_active_generation=True)
    # Frame paths are recorded absolute and were recorded against the staging
    # directory, which is about to stop existing under that name.
    published["frames"] = [
        {**frame, "path": str(final_paths.frames / Path(str(frame["path"])).name)}
        if isinstance(frame, dict) and "path" in frame
        else frame
        for frame in published.get("frames", [])
    ]
    return published


def orphan_generations(bundle_root: Path, verdict: MarkerVerdict) -> list[Path]:
    """Every **generation** on disk that `verdict`'s **manifest** does not name.

    What a crash between the rename and the manifest replace leaves, and what a
    superseded generation becomes once a newer one is published. Naming them is
    what lets prune reclaim them (M3.4) instead of leaving unattributable disk;
    a directory with no valid marker has no **active generation**, so every
    generation under it is an orphan.

    The verdict is the caller's rather than a second `read_marker` of its own.
    Two reads are two moments, and a publish landing between them made the
    survey describe a bundle whose **active generation** was also in its orphan
    list - the one thing `StoreSurvey` promises cannot happen, since the survey
    and the **prune** plan are meant to be one walk's answer (finding 6-opus).

    What counts as a generation is `sorted_generations`' rule, so a symlink
    named `g2` is neither reported here nor proposed there: a report naming a
    cleanup prune will not perform is the same disagreement from the other side.
    """
    active = active_generation_name(verdict)
    return [path for path in sorted_generations(bundle_root) if path.name != active]


def stage_paths(bundle_root: Path, *, reset: bool = True) -> BundlePaths:
    """Open the **staging directory** for the next **generation**."""
    generation_name = next_generation(bundle_root)
    tmp = bundle_root / f"{STAGING_PREFIX}{generation_name}"
    # Validated before the reset below, so a symlinked staging path is refused
    # rather than deleted through.
    ensure_safe_directory(tmp, bundle_root, create_leaf=False)
    if tmp.exists() and reset:
        shutil.rmtree(tmp)
    frames = tmp / FRAMES_DIR_NAME
    ensure_safe_directory(frames, bundle_root)
    return BundlePaths(
        root=bundle_root,
        generation=tmp,
        frames=frames,
        manifest=bundle_root / MANIFEST_NAME,
        transcript=tmp / TRANSCRIPT_NAME,
        markdown=tmp / RENDER_NAME,
    )
