"""OCR secret redaction for Distill.

This module owns secret-pattern matching and targeted OCR-confusable handling.
It is independent of Tesseract and bundle rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import WarningRecord, aggregate_warnings, warning

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
    # GitHub tokens: `ghp_` classic PAT, `gho_` OAuth, `ghs_` server-to-server,
    # `ghu_` user-to-server. All four are what `gh auth status` or a `git
    # config` on screen shows, and all four are credentials.
    re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),  # GitHub PAT (fine-grained)
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),  # Google API key
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),  # Slack token
    re.compile(  # JWT (header.payload.signature)
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),  # generic base64 blob
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),  # GitLab personal access token
    # `npm_` and `hf_` are also ordinary identifier prefixes (`npm_config_*`,
    # `hf_hub_download`), so these take the issued length exactly rather than a
    # floor: alphanumeric only, so an underscore ends the run, and 36 or 34
    # characters of it, so a long camelCase identifier is not a token.
    re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),  # npm automation/publish token
    re.compile(r"\bhf_[A-Za-z0-9]{34}\b"),  # HuggingFace user access token
    # PEM private-key headers - `RSA`, `EC`, `OPENSSH`, `ENCRYPTED`, `PGP ...
    # BLOCK` and the bare form. The armour is what makes a screenful of base64
    # identifiable as a key; the body itself is caught by the base64 rule above.
    # `PUBLIC KEY` and `CERTIFICATE` are not secrets and are left alone.
    re.compile(r"-----(?:BEGIN|END) [A-Z0-9 ]{0,20}PRIVATE KEY(?: BLOCK)?-----"),
]
# Config-style assignments: `API_KEY=...`, `api_key: ...`, `access-token = ...`,
# `apiKey: ...`, and bare `password: ...` / `secret: ...`. The name must be a
# compound identifier (a `_`/`-` separator or a camelCase boundary right before
# the suffix) or one of the two strong bare words, so ordinary words that merely
# end in a suffix ("monkey", "Turkey", a chart legend "Key:") are NOT treated as
# assignments and their text is left intact.
#
# The identifier runs are length-capped. A name can begin at every word boundary
# in the text, so an unbounded run means each of those starts scans to the end of
# the frame's text: 180 KB of `ab-` took 45 seconds before the cap, on text an
# attacker chooses by putting it on screen. 64 characters is longer than any
# configuration key a screen shows.
_SECRET_NAME = (
    # separator: API_KEY, access-token
    r"[A-Za-z0-9][A-Za-z0-9_-]{0,64}[_-](?i:key|token|secret|password)"
    r"|[a-z0-9]{1,64}(?:Key|Token|Secret|Password)"  # camelCase: apiKey
    r"|(?i:password|secret)"  # bare strong word
)
ENV_ASSIGNMENT_RE = re.compile(
    rf"(?P<name>\b(?:{_SECRET_NAME}))(?P<sep>\s*[:=]\s*)(?P<value>[^\s#]+)"
)
# A value that looks like a credential rather than like prose: the run of
# letters it opens with has to end in a digit or in `.`/`+`/`/`/`=`. Neither
# `-` nor `_` counts, because those are how identifiers and hyphenated English
# are spelled - `sign-in-token-value` and `not_configured_here` are prose, while
# `9f8c...`, a JWT and a base64 blob all qualify. The lookahead is a plain class
# followed by one required class, so it cannot backtrack super-linearly (see the
# nested-quantifier guard).
_CREDENTIAL_VALUE = r"(?=[A-Za-z]*[0-9.+/=])[A-Za-z0-9._~+/=-]"
# `Authorization: Bearer <token>`. Only the token is replaced: the header is not
# a secret, and a **render** that says a request was authenticated is more use
# than one where the whole line vanished.
BEARER_RE = re.compile(
    rf"(?P<name>\b(?i:bearer))(?P<sep>[ \t]+)(?P<value>{_CREDENTIAL_VALUE}{{16,}})"
)
# Bare `token:` / `apikey:`, which `_SECRET_NAME` deliberately excludes: they are
# ordinary English words and a chart legend or a UI label writes them the same
# way an .npmrc does. The name alone is therefore not enough here - the value
# must also be credential-shaped and long - which is why these are a separate
# rule rather than two more alternatives in `_SECRET_NAME`.
WEAK_ASSIGNMENT_RE = re.compile(
    rf"(?P<name>\b(?i:token|apikey))(?P<sep>\s*[:=]\s*)(?P<value>{_CREDENTIAL_VALUE}{{12,}})"
)
# Assignment-shaped rules, each replaced as name + separator + [REDACTED].
ASSIGNMENT_PATTERNS = (
    ENV_ASSIGNMENT_RE,
    BEARER_RE,
    WEAK_ASSIGNMENT_RE,
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
    warnings: list[WarningRecord]
    redaction_count: int


def normalize_confusables(text: str) -> str:
    return text.translate(CONFUSABLES)


def redact_text(text: str) -> RedactionResult:
    warnings: list[WarningRecord] = []
    redacted = text
    count = 0

    def replace_assignment(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group("value")
        if value.lower() in TUTORIAL_PLACEHOLDERS or value.startswith("<"):
            return match.group(0)
        count += 1
        # The separator is reproduced as it was written: rewriting `api_key: x`
        # as `api_key=x` turns YAML into an env file, which is a small lie about
        # what the screen showed.
        return f"{match.group('name')}{match.group('sep')}[REDACTED]"

    for assignment in ASSIGNMENT_PATTERNS:
        redacted = assignment.sub(replace_assignment, redacted)

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
        for _match in possible_matches:
            warnings.append(
                warning(
                    "redaction",
                    "possible_confusable_secret",
                    "OCR text contained a secret-like value after confusable normalization",
                )
            )

    # R-41: one record per match would bury a **manifest**, which is what the
    # old cap was for - it emitted ten and then a second **warning** saying how
    # many it had suppressed, so a reader who wanted the real number had to add
    # two records together. The fold says it once and says it exactly, so there
    # is nothing left to cap.
    return RedactionResult(
        text=redacted,
        warnings=aggregate_warnings(warnings),
        redaction_count=count,
    )
