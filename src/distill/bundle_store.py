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
bound: the marker is a schema-valid **manifest** whose recorded bundle identity
equals the directory name, accepting either the current `bundle_key` field or
the legacy `source_hash` field so bundles written before this plan stay
recognizable and prunable (D-017).

A marker proves the directory is Distill's; it does not prove the bundle is
servable. `load_active` additionally verifies the **active generation** and its
**render** exist on disk, because a **manifest** is a promise, not evidence
(R-04): retention that deleted a generation left the manifest still naming it.

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
which nothing else may read, and are ordinary writes.

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
import itertools
import json
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .errors import DistillError

LOGGER = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"
"""The **published marker**: present only once a generation has been published."""

OWNERSHIP_MARKER_NAME = "_owner.json"
"""The **ownership marker** `begin` writes first, so a directory is identifiable
as Distill-owned from its first moment - before any manifest exists (R-11,
D-025). A run that crashes before publishing leaves this and nothing else."""

RENDER_NAME = "video.md"
TRANSCRIPT_NAME = "transcript.json"
FRAMES_DIR_NAME = "frames"
GENERATION_PREFIX = "g"
STAGING_PREFIX = ".tmp."

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

ATOMIC_TEMP_SUFFIX = ".tmp"

_TEMP_COUNTER = itertools.count()
"""Distinguishes two atomic writes to one path from inside a single process,
which a pid alone does not."""

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

_LOCK_HELD_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})
"""`flock` refusing a lock somebody else holds, which is an answer rather than a
failure. Every other errno says this filesystem cannot give Distill the
exclusion it asked for, and is fatal (R-09)."""

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

MarkerKind = Literal["published", "owned", "foreign", "invalid", "absent"]

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


@dataclass
class BundleLock:
    """One run's exclusive hold on a **bundle key**, granted by the kernel.

    The lock *is* an open descriptor holding a `flock`, not a file whose contents
    name a holder. That is what makes it exclusive and what makes it
    self-releasing: there is nothing to publish, no staleness to guess at, and no
    pid that could be reused. A holder killed outright releases it the moment the
    kernel closes its descriptors (finding 11).

    Held until `release`, which the run calls when it is finished - not when
    staging ends. `release` is idempotent, because the failure paths that release
    a lock early overlap with the caller's own cleanup.

    Does not own: what the bundle contains, or the acquisition lease in
    `source.py`, which is keyed by **lock key** and answers a different question -
    "is another run fetching this source?" rather than "is another run producing
    this bundle?".
    """

    bundle_key: str
    path: Path
    fd: int
    released: bool = False

    @classmethod
    def take(cls, bundle_key: str, path: Path) -> BundleLock | None:
        """Take the lock, or report `None` if another run holds it.

        The descriptor stays open inside the returned lock: closing it is what
        releasing means, so nothing else may close it. `flock` is per open file
        description rather than per process, so a second lock taken against the
        same path is refused even inside the process that holds the first.

        A filesystem that cannot grant the lock is fatal (R-09): an errno other
        than "held" means Distill cannot tell one run from two here, and the only
        answer that does not silently reintroduce finding 6 is to stop.
        """
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in _LOCK_HELD_ERRNOS:
                return None
            reported = errno.errorcode.get(exc.errno, "") if exc.errno is not None else ""
            raise DistillError(
                "E_LOCK_UNSUPPORTED",
                "bundle",
                "filesystem cannot lock this bundle key",
                {
                    "bundle_key": bundle_key,
                    "lock_path": str(path),
                    "errno": reported or str(exc.errno),
                },
            ) from exc
        except BaseException:
            os.close(fd)
            raise
        return cls(bundle_key=bundle_key, path=path, fd=fd)

    def release(self) -> None:
        """Give the lock up by closing the descriptor the kernel locked.

        The lock file itself stays on disk, deliberately: unlinking it is a
        second way to lose exclusivity, because a waiter that already opened the
        path holds a descriptor on an inode that now has no name and can lock it
        while the next run locks a fresh file at the same path. Reclaiming those
        empty files is **prune**'s business, not a release's.
        """
        if self.released:
            return
        self.released = True
        os.close(self.fd)
        _bundle_log("lock_released", bundle_key=self.bundle_key, lock_path=str(self.path))


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
class PruneSkip:
    """One directory prune looked at and did not treat as prunable, and why.

    Reported rather than silently passed over, because "prune considered nothing"
    and "prune deleted nothing" are different answers and a caller that cannot
    tell them apart cannot tell a healthy cache from a prune that skipped every
    bundle it found (R-01, R-57).
    """

    path: Path
    verdict: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "verdict": self.verdict, "reason": self.reason}


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
    skipped: tuple[PruneSkip, ...] = ()
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
    skipped: tuple[PruneSkip, ...] = ()
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
class _PruneScan:
    """Mutable accumulator for the walk, so recursion has one place to add to."""

    targets: list[PruneTarget] = field(default_factory=list)
    skipped: list[PruneSkip] = field(default_factory=list)
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
    ) -> BundleRun | BundleSnapshot:
        """Take the run lock for `bundle_key` and open a **staging directory**.

        Returns a `BundleSnapshot` instead when an **active generation** is on
        disk *once the lock is held* (R-08). The re-check has to happen under the
        lock to be worth anything: a waiter that decided "cache miss" before
        queueing would come out of a wait it spent behind the winner and redo
        the work the winner just published (RV-1).

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
            snapshot = self.load_active(bundle_key)
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

    def _take_lock(self, bundle_key: str, wait_sec: float) -> tuple[BundleLock, float]:
        """Poll for the run lock within this caller's budget.

        The budget is the only thing that ends the wait: a lock is held until its
        holder gives it up or dies, and neither is something to time out on a
        holder's behalf.
        """
        path = self.lock_path(bundle_key)
        # The lock directory is the one thing that has to exist before the probe
        # can run at all. It holds no bundle content: an empty file per bundle
        # key, outside every bundle, is not a mutation of anything R-09 protects.
        path.parent.mkdir(parents=True, exist_ok=True)
        started = self.clock()
        contended = False
        while True:
            try:
                lock = BundleLock.take(bundle_key, path)
            except DistillError as exc:
                if exc.code == "E_LOCK_UNSUPPORTED":
                    _bundle_log(
                        "lock_unsupported",
                        bundle_key=bundle_key,
                        lock_path=str(path),
                        errno=exc.details.get("errno"),
                    )
                raise
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
        manifest = {**snapshot.manifest, **fields}
        validate_manifest_schema(manifest, require_active_generation=True)
        write_manifest(snapshot.root, manifest)
        return BundleSnapshot(
            root=snapshot.root,
            bundle_key=snapshot.bundle_key,
            generation=snapshot.generation,
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
        """
        if not self.root.is_dir():
            return PrunePlan(root=self.root, policy=policy)
        scan = _PruneScan()
        self._scan_directory(self.root, depth=0, policy=policy, now=time.time(), scan=scan)
        plan = PrunePlan(
            root=self.root,
            policy=policy,
            targets=tuple(scan.targets),
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
            lock = BundleLock.take(bundle_root.name, lock_path)
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
        shutil.rmtree(target.path)
        return PruneResult(target, "deleted", target.reason)

    def _scan_directory(
        self, directory: Path, *, depth: int, policy: PrunePolicy, now: float, scan: _PruneScan
    ) -> None:
        """Walk one level, recording bundles as targets and everything else as skips."""
        for child in sorted(directory.iterdir()):
            if child.is_symlink():
                scan.considered += 1
                scan.skipped.append(
                    PruneSkip(child, "symlink", "prune never follows a symlinked directory")
                )
                continue
            if not child.is_dir():
                continue
            scan.considered += 1
            if child.name.startswith(("_", ".")):
                scan.skipped.append(
                    PruneSkip(child, "reserved", "reserved name, not a bundle directory")
                )
                continue
            verdict = read_marker(child)
            if verdict.is_distill_owned:
                if lock_is_held(child):
                    scan.skipped.append(
                        PruneSkip(child, "locked", "another run holds this bundle key")
                    )
                    continue
                scan.targets.extend(self._bundle_targets(child, verdict, policy=policy, now=now))
                continue
            scan.skipped.append(PruneSkip(child, verdict.kind, verdict.reason))
            if verdict.kind != "absent":
                # A manifest that is unreadable or records another directory's
                # identity still says Distill is not free to reason about what is
                # underneath it (R-01).
                continue
            if depth + 1 >= PRUNE_MAX_DEPTH:
                scan.skipped.append(
                    PruneSkip(child, "too-deep", f"deeper than {PRUNE_MAX_DEPTH} levels")
                )
                continue
            self._scan_directory(child, depth=depth + 1, policy=policy, now=now, scan=scan)

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

        if policy.max_age_days is not None:
            age_sec = now - bundle_mtime(bundle_root)
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
    lock: BundleLock
    staged_at: float = 0.0
    waited_sec: float = 0.0
    resumed: bool = True

    def release(self) -> None:
        """End the hold. Idempotent, and safe to call from a failure path."""
        self.lock.release()

    def __enter__(self) -> BundleRun:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    @property
    def staging_duration_sec(self) -> float:
        """How long this run has spent assembling its **staging directory**."""
        return self.store.clock() - self.staged_at

    def read_stage(self, name: str) -> Any | None:
        """The recorded **stage result** for `name`, or `None` to recompute it.

        `None` for every reason a stage result cannot be used: the run is not
        resuming, nothing was recorded, or what was recorded is unreadable. A
        stage result is scratch - recomputing it is always available, so nothing
        about it is worth ending a run over.
        """
        if not self.resumed:
            return None
        return read_stage_result(self.paths.generation, name)

    def write_stage(self, name: str, result: Any) -> None:
        """Record a completed stage so an interrupted run can **resume**.

        An ordinary write, deliberately (D-033). The target is inside a
        **staging directory** held under this run's lock: no other process may
        read it, and no reader is ever entitled to it, so there is nobody an
        atomic replace would protect. A torn stage result costs the one run that
        wrote it a recomputation.
        """
        write_stage_result(self.paths.generation, name, result, root=self.paths.root)

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

    def abandon(self, reason: str) -> None:
        """Give up the run, leaving the previous **active generation** intact.

        The **staging directory** stays: its **stage results** are what a later
        run **resumes** from, and reclaiming them is prune's decision rather
        than a failing run's. Nothing under the bundle root is touched, so a
        bundle that already had an **active generation** still has it.

        The reason is the record: a bundle that did not change is otherwise
        indistinguishable from a run that never happened.
        """
        _bundle_log(
            "run_abandoned",
            bundle_key=self.bundle_key,
            reason=reason,
            staging=str(self.paths.generation),
            staging_duration_sec=self.staging_duration_sec,
        )
        self.release()


def prune_lock_path(bundle_root: Path) -> Path:
    """The run lock for the bundle at `bundle_root`. Derives, creates nothing.

    Stated against the directory rather than the **bundle key** because prune
    finds bundles by walking, including bundles under a playlist root that a
    different `BundleStore` produced. The rule is the same one `begin` uses -
    `_locks/<name>.lock` beside the bundle - so both reach the same file, which
    is the only reason prune's lock excludes a live run at all.
    """
    return bundle_root.parent / LOCK_DIR_NAME / f"{bundle_root.name}.lock"


def lock_is_held(bundle_root: Path) -> bool:
    """Whether a run is live on this bundle, asked of the lock rather than a clock.

    R-06: liveness is the same question `begin` answers, so the answer is the
    same lock. A timestamp cannot answer it - a **staging directory** that has
    not been written to for an hour belongs either to a dead run or to a live one
    doing a long download, and a staleness window picks between them by guessing
    (finding 11's shape).

    A lock file that does not exist means no run has ever reached staging here,
    so nothing is created to find that out: a plan mutates nothing.
    """
    path = prune_lock_path(bundle_root)
    if not path.is_file():
        return False
    lock = BundleLock.take(bundle_root.name, path)
    if lock is None:
        return True
    lock.release()
    return False


def sorted_generations(bundle_root: Path) -> list[Path]:
    """Every **generation** directory under `bundle_root`, oldest first."""
    return sorted(
        (
            path
            for path in bundle_root.iterdir()
            if path.is_dir() and not path.is_symlink() and is_generation_name(path.name)
        ),
        key=lambda path: int(path.name[len(GENERATION_PREFIX) :]),
    )


def staging_directories(bundle_root: Path) -> list[Path]:
    """Every **staging directory** left under `bundle_root`."""
    return sorted(
        path
        for path in bundle_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.startswith(STAGING_PREFIX)
    )


def active_generation_name(verdict: MarkerVerdict) -> str | None:
    """The **active generation** a marker names, or `None` if it names none."""
    if not verdict.is_bundle or verdict.manifest is None:
        return None
    name = verdict.manifest.get("active_generation")
    if isinstance(name, str) and is_generation_name(name):
        return name
    return None


def bundle_mtime(bundle_root: Path) -> float:
    """When this **bundle** last changed, as the newest thing in it.

    Expiry is about a bundle nobody has produced into for a while, so the
    youngest of the **manifest** and the **generations** is the age that matters:
    a bundle whose oldest generation is a year old but which was republished
    yesterday is not aged. The old code read the bundle directory's own mtime,
    which changes when anything at all is created beside it.
    """
    candidates = [bundle_root, bundle_root / MANIFEST_NAME, *sorted_generations(bundle_root)]
    times = []
    for path in candidates:
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(times, default=0.0)


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

    Written in place rather than by atomic replace, because recognition is by
    presence: `read_marker` treats the file's existence as the claim and does not
    parse it, so the directory is Distill's from the moment the file is created.
    An atomic replace would keep the claim in a temp file until it completed,
    which is the window RV-9 is about. The payload is for a human reading the
    directory, and a truncated one costs nothing.

    Rewritten on every `begin`, so the recorded pid is the run that holds the
    bundle now rather than the first one that ever did.
    """
    marker = directory / OWNERSHIP_MARKER_NAME
    ensure_safe_directory(marker, directory, create_leaf=False)
    marker.write_text(
        json.dumps(
            {
                "bundle_key": bundle_key,
                "pid": os.getpid(),
                "claimed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return marker


def read_marker(directory: Path) -> MarkerVerdict:
    """Decide whether `directory` carries a **bundle marker**.

    Every failure is a verdict rather than an exception: an unmarked or
    unreadable directory may not be Distill's at all, and refusing to claim it is
    the whole point (R-01). A malformed manifest is likewise not a reason to end
    a run - it is a reason to rebuild the bundle.
    """
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        if (directory / OWNERSHIP_MARKER_NAME).is_file():
            return MarkerVerdict(
                kind="owned",
                reason="ownership marker present, nothing published yet",
                bundle_key=directory.name,
            )
        return MarkerVerdict(kind="absent", reason="no bundle marker")

    try:
        document = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return MarkerVerdict(kind="invalid", reason="manifest is not readable JSON")
    if not isinstance(document, dict):
        return MarkerVerdict(kind="invalid", reason="manifest is not a JSON object")

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
    if recorded_identity(manifest) is None:
        raise DistillError(
            "E_BAD_MANIFEST",
            "bundle",
            "cache manifest schema is invalid",
            {"field": " or ".join(IDENTITY_FIELDS), "expected": "str"},
        )


def read_manifest(bundle_root: Path) -> dict[str, Any] | None:
    """The published **manifest**, validated. Prefer `BundleStore.load_active`."""
    manifest = bundle_root / MANIFEST_NAME
    if not manifest.exists():
        return None
    with manifest.open() as handle:
        data = json.load(handle)
    validate_manifest_schema(data, require_active_generation=True)
    return data


def active_paths(bundle_root: Path) -> BundlePaths | None:
    """Paths to the **active generation** without proving it exists.

    Kept for callers this plan has not migrated yet; `BundleStore.load_active`
    is the surface that answers "is there a bundle to serve?" (R-04).
    """
    manifest = read_manifest(bundle_root)
    if not manifest:
        return None
    generation = bundle_root / str(manifest["active_generation"])
    return BundlePaths(
        root=bundle_root,
        generation=generation,
        frames=generation / FRAMES_DIR_NAME,
        manifest=bundle_root / MANIFEST_NAME,
        transcript=generation / TRANSCRIPT_NAME,
        markdown=generation / RENDER_NAME,
    )


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
    """Write `text` so a concurrent reader sees the old bytes or the new ones.

    The one durable-write helper for state another process may read (R-14,
    D-033): the manifest, the job records, and anything later joining them.
    Writing in place is readable half-written, and for a **manifest** that means
    a directory that is briefly not a **bundle** - the marker is the file.

    Both the target and the temporary file are checked against `root`, so
    neither can be redirected by a symlink pre-created at either name (R-16).
    The temporary sits beside the target because a replace has to stay on one
    filesystem to be atomic at all.

    The temporary name is unique per writer, which is what makes the property
    hold under concurrency rather than only in a single process. A shared name
    is worse than no temporary at all: a second writer truncating it while the
    first replaces it publishes a half-written file *onto the target*, which is
    exactly the torn read the replace exists to prevent. A writer that dies
    mid-write therefore leaves its own temporary behind; that is prune's to
    reclaim, and it is never mistaken for a marker, which is matched by name.
    """
    ensure_safe_directory(path, root, create_leaf=False)
    unique = f"{os.getpid()}.{next(_TEMP_COUNTER)}"
    temporary = path.with_name(f"{path.name}.{unique}{ATOMIC_TEMP_SUFFIX}")
    ensure_safe_directory(temporary, root, create_leaf=False)
    try:
        temporary.write_text(text)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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


def read_stage_result(generation: Path, name: str) -> Any | None:
    """The recorded **stage result** for `name`, or `None` if it cannot be used.

    Unreadable scratch is a miss, not a failure: the stage that produced it can
    always produce it again, and a run that ends over its own scratch is a run
    that cannot recover from an interruption it was interrupted by.
    """
    path = stage_result_path(generation, name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_stage_result(generation: Path, name: str, payload: Any, *, root: Path) -> None:
    """Record a completed stage as resume scratch. An ordinary write (D-033)."""
    path = stage_result_path(generation, name)
    ensure_safe_directory(path, root, create_leaf=False)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


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
    """
    removed = []
    for path in sorted(generation.rglob(STAGE_RESULT_GLOB)):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def publish_staging(paths: BundlePaths, manifest: dict[str, Any]) -> BundlePaths:
    """Turn a **staging directory** into the **active generation**, in order.

    Assemble (the caller's work, already done) -> strip **stage results** ->
    rename to `g<N>` -> atomically replace the **manifest** (R-12).

    The order is the survivability argument. The strip precedes the rename, so
    there is no instant at which a generation on disk holds scratch. The
    manifest replace comes last, so the only gap is a finished `g<N>` that no
    manifest names yet: the previous **active generation** is untouched and
    still servable, and the new directory is an **orphan generation** prune
    reclaims. Reversing the last two would instead point the manifest at a
    directory that does not exist - finding 2's shape, from the other side.
    """
    final_paths = published_paths(paths)
    published = published_manifest(manifest, final_paths)

    strip_stage_results(paths.generation)
    ensure_safe_directory(final_paths.generation, paths.root, create_leaf=False)
    paths.generation.rename(final_paths.generation)
    write_manifest(paths.root, published)
    return final_paths


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


def orphan_generations(bundle_root: Path) -> list[Path]:
    """Every **generation** on disk that the **manifest** does not name.

    What a crash between the rename and the manifest replace leaves, and what a
    superseded generation becomes once a newer one is published. Naming them is
    what lets prune reclaim them (M3.4) instead of leaving unattributable disk;
    a directory with no valid marker has no **active generation**, so every
    generation under it is an orphan.
    """
    verdict = read_marker(bundle_root)
    active = None
    if verdict.is_bundle and verdict.manifest is not None:
        name = verdict.manifest.get("active_generation")
        if isinstance(name, str):
            active = name
    return sorted(
        path
        for path in bundle_root.iterdir()
        if path.is_dir() and is_generation_name(path.name) and path.name != active
    )


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
