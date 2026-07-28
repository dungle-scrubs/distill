"""Structured **warning** and **fatal error** records, and how warnings fold.

This module owns the shape of both records - the snake_case code domain, the
`E_CODE` fatal domain, the JSON text a fatal error is written as - and R-41's
fold: warnings sharing a stage and a code become one record carrying how many
times it happened.

It does not own *what* is worth warning about, does not decide which stage a
warning belongs to, and does not know when a run has collected all of them. A
caller folds its own list when it is done producing warnings; this module only
says what folding means.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FATAL_CODE_RE = re.compile(r"^E_[A-Z0-9_]+$")

WarningRecord = dict[str, Any]
"""One **warning** as it travels: a stage, a code, a message, and a count.

A mapping and not a dataclass because it is written into a **manifest** and read
back out of one, and every stage builds it. `Any` and not `str` because R-41's
occurrence count is a number: declared as a string, every reader that wanted to
compare or add counts would be doing arithmetic the checker had already been
told was text.
"""

FrozenWarningRecord = Mapping[str, Any]
"""The same record once a carrier holds it: read-only, and still counted."""


@dataclass(frozen=True)
class WarningPayload:
    """One **warning**, and how many times the thing it describes happened.

    `occurrences` is carried from the start rather than added when something
    repeats (R-41). A field that appears only on repeats would make every
    reader of a **manifest** spell the default itself, and a run that warned
    once would be indistinguishable in shape from one that never folded.
    """

    stage: str
    code: str
    message: str
    occurrences: int = 1

    def to_dict(self) -> WarningRecord:
        if not CODE_RE.match(self.code):
            raise ValueError(f"warning code must be snake_case: {self.code}")
        # Exactly `int`, not `isinstance`. `bool` is an `int` subtype, so a
        # JSON `true` where a count belongs satisfied `>= 1` and published as
        # the number of times something happened.
        if type(self.occurrences) is not int:
            raise ValueError(f"a warning count must be a whole number: {self.occurrences!r}")
        if self.occurrences < 1:
            raise ValueError(f"a warning happened at least once: {self.occurrences}")
        return asdict(self)


INTERNAL_CODE = "E_INTERNAL"
INTERNAL_STAGE = "internal"
"""What an exception Distill never coded becomes, named once (R-46).

Two surfaces convert one: the CLI's error boundary, and a batch reporting an
item that failed. Spelling the pair out at each of them is how the CLI came to
report `stage: "internal"` for a failure a batch reported with no stage at all.
"""


class DistillError(Exception):
    """Fatal error serialized as JSON text in Distill's error channel."""

    @classmethod
    def from_unexpected(cls, exc: BaseException) -> DistillError:
        """The **fatal error** an uncoded exception becomes at a boundary.

        Enough to diagnose and no more: the exception's type and its message,
        which together identify the failure, and never its traceback. A stack is
        what leaks Distill's internals to whoever ran the command, and it is not
        a thing an operator can act on - the code and the stage are. The stack is
        still reachable, by opting in: `DISTILL_TRACEBACK=1` at the CLI boundary
        re-raises instead of converting.

        The message says *unexpected*, deliberately. Every other **fatal error**
        in Distill is a diagnosis somebody wrote; this one means nobody did, and
        a reader who cannot tell the two apart cannot tell a bad argument from a
        defect.
        """
        return cls(
            INTERNAL_CODE,
            INTERNAL_STAGE,
            f"an unexpected {type(exc).__name__} ended the command",
            {"exception": type(exc).__name__, "message": str(exc)},
        )

    def __init__(
        self,
        code: str,
        stage: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not FATAL_CODE_RE.match(code):
            raise ValueError(f"fatal code must look like E_CODE: {code}")
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "details": self.details,
        }

    def to_json_text(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def warning(stage: str, code: str, message: str, occurrences: int = 1) -> WarningRecord:
    return WarningPayload(
        stage=stage,
        code=code,
        message=message,
        occurrences=occurrences,
    ).to_dict()


def occurrences_of(record: FrozenWarningRecord) -> int:
    """How many times the thing one **warning** describes happened.

    One reading of the field, because every caller that counts warnings wants
    the same answer and each one spelling `record.get("occurrences", 1)` itself
    is how a **manifest** and a stage's own tally came to disagree about the
    same run. A record without the field counts as one, so a warning built as a
    bare mapping (a probe's, a capability's) is counted rather than skipped.
    """
    occurrences = record.get("occurrences", 1)
    return occurrences if type(occurrences) is int and occurrences >= 1 else 1


def total_occurrences(warnings: Iterable[FrozenWarningRecord]) -> int:
    """How many **warning** events a list stands for, folded or not.

    Not `len`. After R-41's fold a list holds one record per (stage, code), so
    its length counts *kinds*; a run that timed out eighty times publishes two
    records and raised eighty-one warnings. Both numbers are worth having and
    only one of them is what a field named for a count of warnings means.
    """
    return sum(occurrences_of(record) for record in warnings)


def aggregate_warnings(warnings: Iterable[FrozenWarningRecord]) -> list[WarningRecord]:
    """The same warnings, folded on (stage, code), each carrying its count (R-41).

    First appearance decides both order and message. Order because a reader
    scans a **manifest** in run order; message because the first failure is the
    one carrying the context that explains the ones after it - a later message
    naming a later **keyframe** would describe the last symptom rather than the
    first.

    The fold composes: a record already carrying a count adds that count, not
    one. That matters because it runs more than once on the way out - a stage
    folds what it produced, the run folds every stage's - and a second pass
    that reset each record to one would discard what the first one counted.

    A record without the field counts as one, so a **warning** built as a bare
    mapping (a probe's, a capability's) folds with the rest rather than
    needing to be rebuilt first.

    What the fold costs, stated because a count is not a summary: everything
    that distinguished the later records from the first one is gone. Keeping
    the first message keeps the *first* keyframe number, the *first* field
    name, the *first* path - so a reader of a folded record learns that eighty
    keyframes timed out and which one timed out first, and nothing about the
    other seventy-nine. That is deliberate and it is not the only copy. A
    per-frame failure is also recorded on the frame itself, as the
    **grounding** its **frame artifact** carries, and the **render** shows the
    count beside the message so nobody reads one record as one event. What is
    genuinely unrecoverable afterwards is the set of identifiers: nothing
    rebuilds the list of which keyframes failed from the folded record.
    """
    folded: dict[tuple[str, str], WarningRecord] = {}
    for record in warnings:
        key = (str(record.get("stage", "")), str(record.get("code", "")))
        count = occurrences_of(record)
        first = folded.get(key)
        if first is None:
            folded[key] = {**record, "occurrences": count}
            continue
        first["occurrences"] += count
    return list(folded.values())
