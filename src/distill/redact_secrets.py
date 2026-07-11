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
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),  # OpenAI-style
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{10,}\b"),  # Stripe
    re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),  # GitHub PAT (classic)
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),  # GitHub PAT (fine-grained)
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),  # Google API key
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),  # Slack token
    re.compile(  # JWT (header.payload.signature)
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),  # generic base64 blob
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
]
# Config-style assignments: `API_KEY=...`, `api_key: ...`, `access-token = ...`,
# `apiKey: ...`, and bare `password: ...` / `secret: ...`. The name must be a
# compound identifier (a `_`/`-` separator or a camelCase boundary right before
# the suffix) or one of the two strong bare words, so ordinary words that merely
# end in a suffix ("monkey", "Turkey", a chart legend "Key:") are NOT treated as
# assignments and their text is left intact.
_SECRET_NAME = (
    r"[A-Za-z0-9][A-Za-z0-9_-]*[_-](?i:key|token|secret|password)"  # separator: API_KEY, access-token
    r"|[a-z0-9]+(?:Key|Token|Secret|Password)"  # camelCase: apiKey
    r"|(?i:password|secret)"  # bare strong word
)
ENV_ASSIGNMENT_RE = re.compile(
    rf"(?P<name>\b(?:{_SECRET_NAME}))\s*[:=]\s*(?P<value>[^\s#]+)"
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
        # CONFUSABLES maps single char -> single char, so match offsets in the
        # normalized string line up 1:1 with the raw text. Redact the raw slice
        # (not just warn) so obfuscated secrets do not survive into the bundle.
        possible_matches: list[str] = []
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(normalized):
                possible_matches.append(match.group(0))
                original_slice = text[match.start() : match.end()]
                if original_slice and original_slice in redacted:
                    count += redacted.count(original_slice)
                    redacted = redacted.replace(original_slice, "[REDACTED]")
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
