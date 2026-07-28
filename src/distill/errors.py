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
        if self.occurrences < 1:
            raise ValueError(f"a warning happened at least once: {self.occurrences}")
        return asdict(self)


class DistillError(Exception):
    """Fatal error serialized as JSON text in Distill's error channel."""

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
    """
    folded: dict[tuple[str, str], WarningRecord] = {}
    for record in warnings:
        key = (str(record.get("stage", "")), str(record.get("code", "")))
        occurrences = record.get("occurrences", 1)
        count = occurrences if isinstance(occurrences, int) and occurrences >= 1 else 1
        first = folded.get(key)
        if first is None:
            folded[key] = {**record, "occurrences": count}
            continue
        first["occurrences"] += count
    return list(folded.values())
