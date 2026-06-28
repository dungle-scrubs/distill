"""Structured warning and fatal error helpers for Distill."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FATAL_CODE_RE = re.compile(r"^E_[A-Z0-9_]+$")


@dataclass(frozen=True)
class WarningPayload:
    stage: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        if not CODE_RE.match(self.code):
            raise ValueError(f"warning code must be snake_case: {self.code}")
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


def warning(stage: str, code: str, message: str) -> dict[str, str]:
    return WarningPayload(stage=stage, code=code, message=message).to_dict()
