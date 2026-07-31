"""OCR secret redaction for Distill.

This module owns secret-pattern matching and targeted OCR-confusable handling.
It is independent of Tesseract and bundle rendering.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .errors import WarningRecord, warning

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
MARK_RE = re.compile(r"\[REDACTED:[a-z0-9-]{1,32}\]")
"""What a **redaction mark** looks like, so a rule can recognize its own work.

<!-- D-008 --> Every replacement checks this before substituting. Redaction runs
twice over the same text by design - `redact_for_prompt` is documented as an
idempotent second pass - and without recognition the second pass re-marks an
already-marked value under whichever rule matched it that time.
"""


def _mark(kind: str) -> str:
    """The text written where a **credential-shaped value** stood.

    Contains no whitespace, deliberately. `ENV_ASSIGNMENT_RE` captures its value
    as `[^\\s#]+`, so a mark with a space in it is captured only up to that
    space and re-substituted, corrupting the text and leaving a stray bracket.
    Today's idempotence is accidental: it holds only because `[REDACTED]`
    happens to have no space.
    """
    return f"[REDACTED:{kind}]"


@dataclass(frozen=True)
class SecretRule:
    """One pattern, what it removes, and an optional say in whether to.

    A record rather than a bare pattern because two things need per-rule
    knowledge: the **redaction mark** has to name a kind, and a rule has to be
    able to match and then decline. Both passes iterate these same records, so
    neither can drift from the other <!-- D-017 -->.
    """

    pattern: re.Pattern[str]
    kind: str


SECRET_RULES = [
    SecretRule(re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "api-key"),  # OpenAI-style
    SecretRule(re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{10,}\b"), "stripe-key"),
    # GitHub tokens: `ghp_` classic PAT, `gho_` OAuth, `ghs_` server-to-server,
    # `ghu_` user-to-server. All four are what `gh auth status` or a `git
    # config` on screen shows, and all four are credentials.
    SecretRule(re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b"), "github-token"),
    SecretRule(re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "github-token"),
    SecretRule(re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "google-api-key"),
    SecretRule(re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "slack-token"),
    SecretRule(  # JWT (header.payload.signature)
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "jwt",
    ),
    SecretRule(re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"), "base64-blob"),
    SecretRule(re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws-key"),
    SecretRule(re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"), "gitlab-token"),
    # `npm_` and `hf_` are also ordinary identifier prefixes (`npm_config_*`,
    # `hf_hub_download`), so these take the issued length exactly rather than a
    # floor: alphanumeric only, so an underscore ends the run, and 36 or 34
    # characters of it, so a long camelCase identifier is not a token.
    SecretRule(re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"), "npm-token"),
    SecretRule(re.compile(r"\bhf_[A-Za-z0-9]{34}\b"), "hf-token"),
    # PEM private-key headers - `RSA`, `EC`, `OPENSSH`, `ENCRYPTED`, `PGP ...
    # BLOCK` and the bare form. The armour is what makes a screenful of base64
    # identifiable as a key; the body itself is caught by the base64 rule above.
    # `PUBLIC KEY` and `CERTIFICATE` are not secrets and are left alone.
    SecretRule(
        re.compile(r"-----(?:BEGIN|END) [A-Z0-9 ]{0,20}PRIVATE KEY(?: BLOCK)?-----"),
        "private-key",
    ),
]

KINDS = frozenset({rule.kind for rule in SECRET_RULES} | {"assigned-secret", "url-password"})
"""Every kind a **redaction mark** may name, as a closed set.

Closed so that a rule added without a kind fails a test rather than shipping a
mark nobody defined. Fixed in source rather than configurable: a caller choosing
its own kinds would make the mark mean different things in different bundles.
"""
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
ASSIGNMENT_RULES = (
    SecretRule(ENV_ASSIGNMENT_RE, "assigned-secret"),
    # A bearer token is an issued credential rather than a configured value, so
    # it keeps its own kind: a reader seeing `oauth-token` next to a header
    # learns something a generic `assigned-secret` would not tell them.
    SecretRule(BEARER_RE, "oauth-token"),
    SecretRule(WEAK_ASSIGNMENT_RE, "assigned-secret"),
)
# A URL whose authority embeds a credential (`scheme://user:secret@host`).
# The password is the secret; the username can be a key id and the host is
# what makes the line diagnosable, so only the password is replaced. Every
# run is bounded and the scheme is anchored by a lookbehind, for the same
# reason the entropy rules above are: unanchored unbounded runs went
# quadratic on attacker-chosen screen text (measured 25.7s on 256 KiB of
# 'a's before the bounds, 14ms after).
URL_USERINFO_RE = re.compile(
    r"(?P<prefix>(?<![a-zA-Z0-9+.-])(?i:[a-z][a-z0-9+.-]{0,31})://[^/@\s:]{1,256}:)"
    r"(?P<value>[^@/\s]{1,512})(?=@)"
)
# The escape hatches the bounded rule above leaves open, each replaced
# whole. User-only opaque tokens (`https://tok_abc123@host`): the 16-char
# floor keeps `ssh://git@github.com` and friends intact - a short service
# username is not a credential shape. And passwords past the primary rule's
# 512 cap: still bounded, still linear.
URL_USERINFO_TOKEN_RE = re.compile(
    r"(?P<prefix>(?<![a-zA-Z0-9+.-])(?i:[a-z][a-z0-9+.-]{0,31})://)"
    r"(?P<value>[^@/:\s]{16,2048})(?=@)"
)
URL_USERINFO_OVERLONG_RE = re.compile(
    r"(?P<prefix>(?<![a-zA-Z0-9+.-])(?i:[a-z][a-z0-9+.-]{0,31})://)"
    r"(?P<value>[^/@\s:]{1,256}:[^@/\s]{513,2048})(?=@)"
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


def cut_without_splitting_a_mark(text: str) -> str:
    """`text`, minus a trailing **redaction mark** that a cut left half-written.

    Truncation runs *after* redaction, so no secret can be split by a cap - only
    the mark standing where one used to be. A half-written mark is worse than
    either keeping it or dropping it: `[REDACTED:a` is not a mark, so a second
    pass cannot recognize it, and a reader sees it as truncated content rather
    than as a redaction. The cap is a byte limit, so the whole mark cannot be
    kept; what is left is to drop it.

    Called by every site that cuts already-redacted text. The alternative -
    giving each cap its own headroom for the longest kind - puts a fact about
    the mark in four places, and the mark's shape lives here.
    """
    opened = text.rfind("[REDACTED:")
    if opened == -1:
        return text
    # `match`, not `fullmatch`: the question is whether a *complete* mark starts
    # at the last opener, not whether the remainder is only that mark. Ordinary
    # text follows a mark all the time, and treating that as a partial cut every
    # line after the last redaction.
    if MARK_RE.match(text, opened):
        return text
    return text[:opened]


def normalize_confusables(text: str) -> str:
    return text.translate(CONFUSABLES)


def redact_for_prompt(text: str) -> str:
    """The prompt-side sink (D-010): extracted text headed into a vision
    prompt, redacted unconditionally.

    No flag consults this path - `--no-redact-secrets` governs what the
    operator sees in their own render, never what leaves for a model. Local
    and remote take the same text so their outputs stay cache-coherent. The
    per-secret warnings are dropped here: on the default path the carrier's
    response-side sink already recorded them for the same text, and under
    `--no-redact-secrets` the operator asked to see the raw text they are
    already looking at - a warning that their own screen contains a secret
    adds noise, not protection. On the default path this is a second,
    idempotent pass over already-redacted text; its whole value is the
    flag-off and eval paths.
    """
    return redact_text(text).text


def redact_text(text: str) -> RedactionResult:
    warnings: list[WarningRecord] = []
    redacted = text
    count = 0

    def replace_assignment(kind: str) -> Callable[[re.Match[str]], str]:
        def replace(match: re.Match[str]) -> str:
            nonlocal count
            value = match.group("value")
            if value.lower() in TUTORIAL_PLACEHOLDERS or value.startswith("<"):
                return match.group(0)
            # <!-- D-008 --> Already marked: leave it exactly as it is. Without
            # this the second pass re-marks the mark under whichever rule
            # matched it this time, which is a different kind and a changed
            # count for text that lost nothing new.
            if MARK_RE.fullmatch(value):
                return match.group(0)
            count += 1
            # The separator is reproduced as it was written: rewriting
            # `api_key: x` as `api_key=x` turns YAML into an env file, which is
            # a small lie about what the screen showed.
            return f"{match.group('name')}{match.group('sep')}{_mark(kind)}"

        return replace

    for rule in ASSIGNMENT_RULES:
        redacted = rule.pattern.sub(replace_assignment(rule.kind), redacted)

    def replace_userinfo(match: re.Match[str]) -> str:
        nonlocal count
        if MARK_RE.fullmatch(match.group("value")):
            return match.group(0)
        count += 1
        return f"{match.group('prefix')}{_mark('url-password')}"

    redacted = URL_USERINFO_RE.sub(replace_userinfo, redacted)
    redacted = URL_USERINFO_TOKEN_RE.sub(replace_userinfo, redacted)
    redacted = URL_USERINFO_OVERLONG_RE.sub(replace_userinfo, redacted)

    for rule in SECRET_RULES:
        matches = rule.pattern.findall(redacted)
        if matches:
            count += len(matches)
            redacted = rule.pattern.sub(_mark(rule.kind), redacted)

    normalized = normalize_confusables(text)
    if normalized != text:
        # CONFUSABLES maps single char -> single char, so match offsets in the
        # normalized string line up 1:1 with the raw text. Redact the raw slice
        # (not just warn) so obfuscated secrets do not survive into the bundle.
        possible_matches: list[str] = []
        for rule in SECRET_RULES:
            for match in rule.pattern.finditer(normalized):
                possible_matches.append(match.group(0))
                original_slice = text[match.start() : match.end()]
                if original_slice and original_slice in redacted:
                    count += redacted.count(original_slice)
                    redacted = redacted.replace(original_slice, _mark(rule.kind))
        if possible_matches:
            # R-41: one record per match would bury a **manifest**, which is
            # what the old cap was for - it emitted ten and then a second
            # **warning** saying how many it had suppressed, so a reader who
            # wanted the real number had to add two records together. One
            # record says it once and says it exactly, so there is nothing left
            # to cap. Counted rather than accumulated and folded afterwards:
            # how many matches there are is decided by text Distill did not
            # write, and a record per match is a list held in memory whether or
            # not it is the list anybody sees.
            warnings.append(
                warning(
                    "redaction",
                    "possible_confusable_secret",
                    "OCR text contained a secret-like value after confusable normalization",
                    occurrences=len(possible_matches),
                )
            )

    return RedactionResult(text=redacted, warnings=warnings, redaction_count=count)
