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
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FATAL_CODE_RE = re.compile(r"^E_[A-Z0-9_]+$")

DETAIL_TEXT_LIMIT = 2048
DETAIL_TRUNCATION_SUFFIX = "...<truncated>"
DETAIL_DEPTH_LIMIT = 6
"""What a **fatal error**'s `details` may carry once it is published as JSON.

A record is a diagnosis, not a transcript. The text limit bounds one value, the
depth limit bounds a nested structure, and both are stated here because a reader
that finds `...<truncated>` should be able to find out what did the truncating.
"""


def _capped(text: str) -> str:
    if len(text) <= DETAIL_TEXT_LIMIT:
        return text
    return text[:DETAIL_TEXT_LIMIT] + DETAIL_TRUNCATION_SUFFIX


def _described(value: Any) -> str:
    """What a value JSON has no word for is published as: its `repr`, bounded.

    `repr` and not `str`, because the two disagree exactly where it matters -
    `str(b"\\xff")` is a lossy `b'\\xff'`-shaped guess in some types and `repr`
    is the form a Python reader can act on. A `repr` that itself raises answers
    with the type name alone: the one thing a boundary may not do is fail while
    reporting a failure.
    """
    try:
        return _capped(repr(value))
    except Exception:
        return f"<{type(value).__name__}>"


def _json_safe(value: Any, seen: tuple[int, ...] = (), depth: int = 0) -> Any:
    """`value` as something `json.dumps` can write, and a strict reader can parse.

    One coercion, in one place, because both surfaces that publish a **fatal
    error** - the CLI boundary and a batch item's report - serialize `details`
    the stage that raised put there, and a stage reaches for the `Path` it was
    working on, the bytes it could not decode, or a ratio that came out `inf`
    without thinking about JSON. Uncoerced, each of those ends the serialization
    rather than the record: `TypeError` for the first two, and for the third a
    literal `Infinity`, which `json.dumps` writes happily and a strict reader
    (`jq`, every non-Python parser) refuses.

    That failure is worst exactly where it is likeliest. `cli._fail` runs from
    inside an `except` clause, so nothing catches what the serialization raises:
    the operator gets no error object at all, only the stack the boundary exists
    to replace.

    Bounded in three ways, so no `details` can make a record unpublishable:
    long text is capped, a structure deeper than `DETAIL_DEPTH_LIMIT` is
    described rather than walked, and a container that contains itself reports
    `<recursive>` instead of a `ValueError: Circular reference detected`.
    """
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        # `inf` and `nan` are not JSON. `repr` names them the way the option
        # validators already do, which keeps one spelling across the records.
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, str):
        return _capped(value)
    if depth >= DETAIL_DEPTH_LIMIT:
        return _described(value)
    if id(value) in seen:
        return "<recursive>"
    if isinstance(value, Mapping):
        nested = (*seen, id(value))
        return {
            (key if isinstance(key, str) else _described(key)): _json_safe(item, nested, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        nested = (*seen, id(value))
        return [_json_safe(item, nested, depth + 1) for item in value]
    return _described(value)


def json_safe_details(details: Mapping[str, Any]) -> dict[str, Any]:
    """A `details` mapping every JSON reader can parse. See `_json_safe`."""
    coerced = _json_safe(details)
    return coerced if isinstance(coerced, dict) else {"details": coerced}

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

        An exception whose `__str__` raises is still converted. Both callers of
        this are already handling a failure, so a conversion that raised would
        replace the failure it was converting with one that names neither - and
        `__str__` running arbitrary code is ordinary, not contrived: a lazily
        formatted message is a `repr` of whatever the exception was built with.
        """
        try:
            message = str(exc)
        except Exception as unreadable:
            message = f"<unreadable: {type(unreadable).__name__}>"
        return cls(
            INTERNAL_CODE,
            INTERNAL_STAGE,
            f"an unexpected {type(exc).__name__} ended the command",
            {"exception": type(exc).__name__, "message": _capped(message)},
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
        """The record as every surface publishes it, JSON-safe by construction.

        `details` is coerced here rather than at each surface: the CLI boundary
        and a batch item's report both serialize whatever the raising stage put
        there, and a coercion written at one of them is the one missing at the
        other. `self.details` itself is left as the stage wrote it, so a caller
        holding the exception still has the original objects.
        """
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "details": json_safe_details(self.details),
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
