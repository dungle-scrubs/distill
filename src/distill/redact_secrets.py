"""OCR secret redaction for Distill.

This module owns secret-pattern matching and targeted OCR-confusable handling.
It is independent of Tesseract and bundle rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import warning

CONFUSABLES = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "３": "3",
        "５": "5",
        "８": "8",
        "Ｏ": "O",
        "ｏ": "o",
        "Ｉ": "I",
        "ｌ": "l",
        "Ｓ": "S",
        "Ａ": "A",
        "ｓ": "s",
        "ｋ": "k",
        "Ｋ": "K",
    }
)
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]
ENV_ASSIGNMENT_RE = re.compile(
    r"(?P<name>[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD))\s*=\s*(?P<value>[^\s#]+)"
)
TUTORIAL_PLACEHOLDERS = {
    "your_key_here",
    "your_api_key",
    "example",
    "changeme",
    "<your-api-key>",
}


@dataclass(frozen=True)
class RedactionResult:
    text: str
    warnings: list[dict[str, str]]
    redaction_count: int


def normalize_confusables(text: str) -> str:
    return text.translate(CONFUSABLES)


def redact_text(text: str, max_possible_secret_warnings: int = 10) -> RedactionResult:
    warnings: list[dict[str, str]] = []
    redacted = text
    count = 0

    def replace_env(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group("value")
        if value.lower() in TUTORIAL_PLACEHOLDERS or value.startswith("<"):
            return match.group(0)
        count += 1
        return f"{match.group('name')}=[REDACTED]"

    redacted = ENV_ASSIGNMENT_RE.sub(replace_env, redacted)

    for pattern in SECRET_PATTERNS:
        matches = pattern.findall(redacted)
        if matches:
            count += len(matches)
            redacted = pattern.sub("[REDACTED]", redacted)

    normalized = normalize_confusables(text)
    if normalized != text:
        possible_matches: list[str] = []
        for pattern in SECRET_PATTERNS:
            possible_matches.extend(pattern.findall(normalized))
        capped = possible_matches[:max_possible_secret_warnings]
        for _match in capped:
            warnings.append(
                warning(
                    "redaction",
                    "possible_confusable_secret",
                    "OCR text contained a secret-like value after confusable normalization",
                )
            )
        if len(possible_matches) > len(capped):
            truncated = len(possible_matches) - len(capped)
            warnings.append(
                warning(
                    "redaction",
                    "possible_secret_warnings_truncated",
                    f"{truncated} additional possible-secret warnings were truncated",
                )
            )

    return RedactionResult(text=redacted, warnings=warnings, redaction_count=count)
