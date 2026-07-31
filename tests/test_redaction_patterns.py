"""Tests for the **redaction** pattern set - R-50, finding 19.

The question this file asks is coverage: does the pattern set match the
credential formats that are actually on screen while somebody records their
terminal and their browser? Finding 19 is that eight of them were not matched at
all, so a run that redacted nothing looked exactly like a run with nothing to
redact.

Every pattern here is stated twice: once with a value it MUST match, and once
with prose it MUST leave alone. The second half is not decoration. Extracted
text is mostly prose, and a pattern that eats prose destroys the transcript the
**bundle** exists to carry, so an over-matching pattern is worse than a missing
one. `test_ordinary_words_ending_in_a_suffix_are_not_redacted` in
`tests/test_redaction_render.py` guards the same edge for the assignment rules
that already existed.
"""

from __future__ import annotations

import re
from importlib import import_module
from typing import Any

import pytest

from distill import redact_secrets
from distill.redact_secrets import redact_text

# An opaque 32-character credential: long enough to be a real token, and matched
# by none of the pre-R-50 rules, so each test below is about the rule it names
# rather than about the generic base64 rule catching the fixture by accident.
OPAQUE = "9f8c2b1a4d7e6f0c3b5a8d92e1f4c7b0"


def assert_intact(text: str) -> None:
    """The false-positive half: prose survives byte for byte."""
    result = redact_text(text)
    assert result.text == text, text
    assert result.redaction_count == 0, text


# 1. Authorization: Bearer <token>


def test_an_authorization_bearer_header_value_is_redacted() -> None:
    """R-50: the header a developer has open in their network tab all day."""
    result = redact_text(f"Authorization: Bearer {OPAQUE}")

    assert OPAQUE not in result.text
    assert result.redaction_count >= 1
    # The header itself is not a secret. It stays, so a reader can still see
    # that the frame showed an authenticated request.
    assert "Authorization: Bearer [REDACTED:oauth-token]" in result.text


def test_prose_about_bearer_tokens_is_not_redacted() -> None:
    assert_intact("The bearer token scheme is described in RFC 6750")
    assert_intact("Bearer of bad news")
    # What the API docs on the other half of the screen show.
    assert_intact("Authorization: Bearer <token>")
    # Past the 16-character floor, so the length alone no longer decides: a run
    # of letters and hyphens is how English and identifiers are spelled, and
    # only the credential-shape lookahead keeps this one whole. Without that
    # half of the rule the floor passes this and the line is eaten.
    assert_intact("Authorization: Bearer authentication-scheme-docs")


# 2. A bare `token:` assignment


def test_a_bare_token_assignment_is_redacted() -> None:
    """R-50: `token:` with no compound name in front of it is still a token."""
    result = redact_text(f"token: {OPAQUE}")

    assert OPAQUE not in result.text
    assert result.redaction_count >= 1
    assert "token: [REDACTED:assigned-secret]" in result.text


def test_prose_after_a_bare_token_label_is_not_redacted() -> None:
    # The existing "ordinary words" guard owns "Token: a single sign-in label".
    assert_intact("token: see the authentication docs")
    assert_intact("Token: expired")
    # A single long word is the harder half: an identifier is not a credential,
    # so neither `_` nor `-` makes a value look like one.
    assert_intact("token: documentation_v2")
    assert_intact("token: sign-in-token-value")


# 3. A bare `apikey:` assignment


def test_a_bare_apikey_assignment_is_redacted() -> None:
    """R-50: `apikey` is one word, so no separator and no camelCase boundary."""
    result = redact_text(f"apikey: {OPAQUE}")

    assert OPAQUE not in result.text
    assert result.redaction_count >= 1
    assert "apikey: [REDACTED:assigned-secret]" in result.text


def test_prose_after_a_bare_apikey_label_is_not_redacted() -> None:
    assert_intact("apikey: not configured")
    assert_intact("apikey: refer to the internal onboarding page")
    assert_intact("apikey: not_configured_here")


# 4. GitLab personal access tokens


def test_a_gitlab_personal_access_token_is_redacted() -> None:
    # Assembled rather than written as one literal, like the npm and HuggingFace
    # fixtures below: a whole token-shaped string in a tracked file is what
    # GitHub's push protection blocks, and a fixture that cannot be pushed is a
    # fixture nobody can run.
    token = "glpat-" + "ABCdef1234567890abcd"
    result = redact_text(f"remote token {token} is live")

    assert token not in result.text
    assert result.redaction_count >= 1


def test_the_glpat_prefix_named_in_prose_is_not_redacted() -> None:
    assert_intact("Tokens issued by GitLab start with the glpat- prefix.")
    assert_intact("glpat-short")


# 5. npm tokens


def test_an_npm_token_is_redacted() -> None:
    # 36 alphanumeric characters after the prefix, which is what npm issues.
    token = "npm_" + "AbCd1234EfGh5678IjKl9012MnOp3456QrSt"
    result = redact_text(f"//registry.npmjs.org/:_authToken={token}")

    assert token not in result.text
    assert result.redaction_count >= 1


def test_npm_config_variables_are_not_redacted() -> None:
    assert_intact("npm_config_registry is not a credential")
    assert_intact("npm_lifecycle_event")
    # A long identifier is not a token: the issued length is exact.
    assert_intact("npm_ordinaryRegistryIdentifier1234567")


# 6. HuggingFace tokens


def test_a_huggingface_token_is_redacted() -> None:
    # 34 alphanumeric characters after the prefix, which is what HuggingFace
    # issues.
    token = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz01234567"
    result = redact_text(f"login({token})")

    assert token not in result.text
    assert result.redaction_count >= 1


def test_huggingface_helper_names_are_not_redacted() -> None:
    assert_intact("hf_hub_download() fetches the weights")
    assert_intact("hf_api")
    # Any length but the issued one. An alphanumeric identifier of exactly 34
    # characters is indistinguishable from a HuggingFace token by shape alone,
    # which is why the length here is exact rather than a floor: it is the only
    # thing separating the two.
    assert_intact("hf_datasetsCacheDirectoryOverride12345")


# 7. The remaining GitHub token prefixes


@pytest.mark.parametrize("prefix", ["gho_", "ghs_", "ghu_"])
def test_the_remaining_github_token_prefixes_are_redacted(prefix: str) -> None:
    """R-50: only `ghp_` and `github_pat_` were covered.

    `gho_` is an OAuth access token, `ghs_` a server-to-server token and `ghu_`
    a user-to-server token. All three are credentials a `gh auth status` or a
    `git config` on screen will show.
    """
    token = prefix + "AbCd1234EfGh5678IjKl9012MnOp3456QrSt"
    result = redact_text(f"the checkout used {token} today")

    assert token not in result.text
    assert result.redaction_count >= 1


def test_the_github_token_prefixes_named_in_prose_are_not_redacted() -> None:
    assert_intact("The gho_ prefix marks an OAuth access token.")
    assert_intact("ghs_short")


# 8. PEM private-key headers


@pytest.mark.parametrize(
    "header",
    [
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----",
    ],
)
def test_a_pem_private_key_header_is_redacted(header: str) -> None:
    result = redact_text(f"$ cat id_rsa\n{header}\nMIIEvQIBADANBg")

    assert header not in result.text
    assert result.redaction_count >= 1


def test_pem_headers_that_are_not_private_keys_are_not_redacted() -> None:
    assert_intact("-----BEGIN PUBLIC KEY-----")
    assert_intact("-----BEGIN CERTIFICATE-----")


# 11. `replace_env` preserves the separator it found


def test_an_assignment_is_redacted_with_the_separator_it_was_written_with() -> None:
    """The **redaction** output is still the screen the recording showed.

    Rewriting `api_key: x` as `api_key=[REDACTED:assigned-secret]` changes YAML into an env
    file, which is a small lie about what was on screen and a confusing one in
    a **render** that carries both.
    """
    colon = redact_text("api_key: sk-abcdefghijklmnopqrstuvwxyz")
    assert "api_key: [REDACTED:assigned-secret]" in colon.text
    assert "api_key=" not in colon.text

    equals = redact_text("API_KEY=sk-abcdefghijklmnopqrstuvwxyz")
    assert "API_KEY=[REDACTED:assigned-secret]" in equals.text

    spaced = redact_text("password = hunter2length")
    assert "password = [REDACTED:assigned-secret]" in spaced.text


# 12. No pattern uses a nested quantifier


def compiled_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Every compiled pattern the module holds, wherever it is stored.

    Read off the module namespace rather than off a named list, so a pattern
    added to a new collection tomorrow is covered by the guard below without
    anyone remembering to enrol it.
    """
    found: list[tuple[str, re.Pattern[str]]] = []

    def collect(label: str, value: Any) -> None:
        if isinstance(value, re.Pattern):
            found.append((label, value))
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                collect(f"{label}[{index}]", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(f"{label}[{key!r}]", item)

    for name, value in vars(redact_secrets).items():
        collect(name, value)
    return found


def regex_internals() -> tuple[Any, Any]:
    """`re`'s own parser and opcode constants.

    Private stdlib modules, reached deliberately and imported dynamically
    because the type checker cannot see into them. The guard below is about the
    structure the regex engine will execute, and the parse tree is the only
    place that structure exists - reading the pattern string is exactly the
    eyeballing this test replaces.
    """
    return import_module("re._parser"), import_module("re._constants")


def repeat_opcodes() -> tuple[Any, ...]:
    """Every opcode `re` compiles a quantifier to.

    `POSSESSIVE_REPEAT` is `*+` / `{m,n}+`, which the parser gives its own
    opcode: a guard that only knew the greedy and lazy opcodes would report a
    possessive unbounded run as no run at all.
    """
    _parser, constants = regex_internals()
    names = ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
    opcodes = tuple(
        getattr(constants, name) for name in names if getattr(constants, name, None) is not None
    )
    assert len(opcodes) == len(names), f"an opcode this guard relies on is gone: {opcodes}"
    return opcodes


def repeat_nodes(node: Any) -> list[Any]:
    """Every repeat (`*`, `+`, `{m,n}`) in a parsed pattern, with its body."""
    parser, _constants = regex_internals()
    repeat_ops = repeat_opcodes()

    repeats: list[Any] = []

    def walk(subpattern: Any) -> None:
        for opcode, args in subpattern:
            if opcode in repeat_ops:
                repeats.append((args[1], args[2]))
                walk(args[2])
            else:
                for arg in args if isinstance(args, tuple) else (args,):
                    if isinstance(arg, parser.SubPattern):
                        walk(arg)
                    elif isinstance(arg, list):
                        for branch in arg:
                            if isinstance(branch, parser.SubPattern):
                                walk(branch)

    walk(node)
    return repeats


def test_no_redaction_pattern_uses_a_nested_quantifier() -> None:
    """Extracted text is attacker-chosen, so a pattern is a runtime cost.

    A quantifier inside a quantifier - `(a+)+`, `(?:[a-z]*)+`, `(?:a?){23}` -
    gives the backtracking engine an exponential number of ways to split a
    subject, so a frame carrying a long run of one character stops being a frame
    and becomes a denial of service on the whole run. Any repeat inside a repeat
    counts, including an inner `?`: `(?:a?){23}a{23}` is the textbook blowup and
    its inner quantifier tops out at one. This is asserted against the compiled
    patterns' parse trees rather than by reading them, because "I looked at it"
    does not survive the next pattern somebody adds.
    """
    parser, _constants = regex_internals()

    assert compiled_patterns(), "no compiled patterns were found to check"

    offenders: list[str] = []
    for name, pattern in compiled_patterns():
        parsed = parser.parse(pattern.pattern, pattern.flags)
        for _max_count, body in repeat_nodes(parsed):
            if repeat_nodes(body):
                offenders.append(f"{name}: {pattern.pattern}")
                break

    assert offenders == [], f"nested quantifiers found: {offenders}"


def group_body(pattern: re.Pattern[str], group: str) -> Any:
    """The parsed body of `pattern`'s named group, or None if it has none."""
    parser, constants = regex_internals()
    wanted = pattern.groupindex.get(group)
    if wanted is None:
        return None

    def walk(subpattern: Any) -> Any:
        for opcode, args in subpattern:
            if opcode is constants.SUBPATTERN and args[0] == wanted:
                return args[3]
            for arg in args if isinstance(args, tuple) else (args,):
                if isinstance(arg, parser.SubPattern):
                    found = walk(arg)
                    if found is not None:
                        return found
                elif isinstance(arg, list):
                    for branch in arg:
                        if isinstance(branch, parser.SubPattern):
                            found = walk(branch)
                            if found is not None:
                                return found
        return None

    return walk(parser.parse(pattern.pattern, pattern.flags))


def test_an_assignment_name_cannot_scan_the_whole_frame() -> None:
    """The same denial of service without a nested quantifier anywhere.

    An assignment's name can begin at every word boundary in the text, so an
    unbounded run inside it is not one scan, it is one scan per start position.
    `\\b[A-Za-z0-9][A-Za-z0-9_-]*[_-](?i:key|token|...)` did exactly that: 180 KB
    of `ab-` - text an attacker chooses simply by putting it on screen - took 45
    seconds, and each of those bytes was somebody's frame. The value runs are not
    covered by this: they are reached only once a name has already matched.

    Length-capping the runs is the fix, and this is the assertion that keeps it,
    without a wall-clock threshold that would be slow when it passed and flaky
    when it did not.
    """
    _parser, constants = regex_internals()

    checked = 0
    offenders: list[str] = []
    for pattern in redact_secrets.ASSIGNMENT_PATTERNS:
        body = group_body(pattern, "name")
        assert body is not None, pattern.pattern
        checked += 1
        for max_count, _body in repeat_nodes(body):
            if max_count is constants.MAXREPEAT:
                offenders.append(pattern.pattern)
                break

    assert checked == len(redact_secrets.ASSIGNMENT_PATTERNS)
    assert offenders == [], f"unbounded repeat in an assignment name: {offenders}"


def test_url_userinfo_redacts_the_password_only() -> None:
    result = redact_text("curl https://admin:sk-live-abcdef123456@api.example.com/v1")

    assert "sk-live-abcdef123456" not in result.text
    assert "admin:" in result.text
    assert "api.example.com" in result.text


def test_url_userinfo_leaves_benign_authority_shapes_intact() -> None:
    benign = [
        "git clone ssh://git@github.com/org/repo.git",
        "mailto:someone@example.com",
        "http://[::1]:8000/v1",
        "meet at 12:30@cafe",
        "image ghcr.io/org/app:1.2@sha256:abcdef",
        "HTTPS://EXAMPLE.COM/PLAIN/PATH",
    ]
    for text in benign:
        assert redact_text(text).text == text, text


def test_url_userinfo_matches_uppercase_schemes() -> None:
    result = redact_text("HTTPS://USER:SECRET-VALUE-123456@HOST.EXAMPLE.COM")

    assert "SECRET-VALUE-123456" not in result.text


def test_url_userinfo_escape_hatches_are_covered() -> None:
    opaque = redact_text("https://tok_abcdefghijklmnop@api.example.com/v1")
    assert "tok_abcdefghijklmnop" not in opaque.text

    overlong = redact_text(f"https://user:{'p' * 540}@host.example.com")
    assert "p" * 540 not in overlong.text

    # Short service usernames stay: a username is not a credential shape.
    assert_intact("git clone ssh://git@github.com/org/repo.git")


# --- M1.2: kinds and re-entrancy -------------------------------------------


def test_the_mark_names_the_kind_of_value_it_replaced() -> None:
    """P2: a reader cannot act on `[REDACTED]` because it says nothing.

    A frame that lost an API key and a frame that lost a content digest read
    identically today, so a downstream agent cannot tell whether the frame still
    carries what it needed. The kind comes from the rule that matched, never
    from the value, so it discloses a category and nothing narrower.
    """
    assert "[REDACTED:api-key]" in redact_text("sk-abcdefghijklmnopqrstuvwx").text
    assert "[REDACTED:jwt]" in redact_text(
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    ).text
    assert "[REDACTED:aws-key]" in redact_text("AKIAIOSFODNN7EXAMPLE").text


def test_the_mark_carries_no_whitespace() -> None:
    """<!-- D-008 --> Whitespace in the mark breaks the second pass.

    `ENV_ASSIGNMENT_RE` captures its value as `[^\\s#]+`, so a mark containing a
    space is captured only up to that space and re-substituted, corrupting the
    text and leaving a stray bracket. Today's idempotence is accidental - it
    holds only because `[REDACTED]` happens to have no space in it.
    """
    marked = redact_text("api_key: sk-abcdefghijklmnopqrstuvwx").text

    assert " " not in marked.split("api_key:")[1].strip()


def test_redaction_is_re_entrant_over_every_fixture() -> None:
    """<!-- D-018 --> A second pass changes nothing - text, count, or warnings.

    `redact_for_prompt` is documented as an idempotent second pass over
    already-redacted text, so this is a live path rather than a hypothetical.
    Asserted as a property over a table rather than on one hand-picked example,
    because the cases that break it are the ones nobody thinks to pick: an
    assignment wrapping a mark, and the confusable path.
    """
    fixtures = [
        "api_key: sk-abcdefghijklmnopqrstuvwx",
        "password: hunter2hunter2hunter2",
        "Authorization: Bearer abcdefghijklmnop1234",
        "curl https://admin:s3cr3t-value-here@api.example.com/v1",
        "AKIAIOSFODNN7EXAMPLE",
        "-----BEGIN RSA PRIVATE KEY-----",
        # The confusable path: a fullwidth digit normalizes into a secret shape.
        "ｓk-abcdefghijklmnopqrstuvwx",
    ]
    for text in fixtures:
        once = redact_text(text)
        twice = redact_text(once.text)
        assert twice.text == once.text, text
        assert twice.redaction_count == 0, text
        assert twice.warnings == [], text
