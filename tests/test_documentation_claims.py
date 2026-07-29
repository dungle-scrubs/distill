"""What the documentation claims, held against the code that would keep it.

R-51, and D-022's rule that an overstated claim in a README is a defect. Nothing
renders `README.md` or `AGENTS.md` from the package - both are written by hand -
so the only thing that can keep a documented default, class, flag or bound in
step with the code is an assertion. Each test here names one claim and derives
the other side of it from the package rather than restating it, so a claim that
stops being true fails the suite instead of aging quietly.

The scope is claims a program can check: a default value, a classification, a
registered flag, a derived bound. Prose that describes a property (what a
sampled fingerprint costs an attacker, what redaction does not promise) is
traced to the test that proves the property in the plan's Phase 9 appendix, not
here - a test that asserted the sentence would pin the wording rather than the
fact.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from distill.capabilities import EXTERNAL_TOOLS
from distill.cli import PROCESSING_KEYS, build_parser
from distill.source import (
    CONTENT_HASH_LIMIT_BYTES,
    FINGERPRINT_INTERIOR_ANCHORS,
    FINGERPRINT_SAMPLE_BYTES,
    fingerprint_anchor_offsets,
    validate_output_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
AGENTS = REPO_ROOT / "AGENTS.md"
CONFIG_SOURCE = REPO_ROOT / "src" / "distill" / "config.py"
VERSION_SOURCE = REPO_ROOT / "src" / "distill" / "version.py"

# Indentation allowed on both fences: AGENTS.md fences a command inside a list
# item, and a fence this misses leaves its backticks to pair with prose, which
# swallows every claim after it.
FENCED_BLOCK = re.compile(r"^[ \t]*```.*?^[ \t]*```", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`([^`]+)`")
FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")
# Numbers small enough to be written as words in prose, which is how this
# repository writes them. The map is here rather than in the documents so a
# count derived from the package can be compared with the word a reader sees.
NUMBER_WORDS = {
    2: "two",
    3: "three",
    7: "seven",
    9: "nine",
}


def prose(path: Path) -> str:
    """The document with fenced code blocks removed.

    A fenced block is an example of something else being run - `pytest --cov`,
    `score.py --with-vision` - and its flags are not Distill's to register. The
    claims these tests are about are made in prose and in inline code.
    """
    return FENCED_BLOCK.sub("", path.read_text(encoding="utf-8"))


def flowed(text: str) -> str:
    """`text` with every run of whitespace collapsed to one space.

    A claim is a phrase, and a phrase in a hard-wrapped document has a newline
    wherever the wrap fell. Searching the raw text for one means the assertion
    holds the *wrapping* as well as the claim, so re-flowing a paragraph breaks
    a test about something the paragraph still says.
    """
    return " ".join(text.split())


def sentences(text: str) -> list[str]:
    """`text` as sentences, close enough that a claim is found whole in one.

    Split after a full stop, a colon or a semicolon, over the re-flowed text, so
    a claim that spans two source lines is one sentence here. Approximate on
    purpose: what it is used for is scoping a search to the clause that carries
    a claim, and a clause boundary in the wrong place widens or narrows that
    scope rather than inventing or losing a claim.
    """
    return re.split(r"(?<=[.:;])\s+", flowed(text))


def subparser_options(command: str) -> set[str]:
    """Every option string one subcommand registers.

    Distinct from `registered_options`, which is the union over every
    subcommand: a claim that some *set* of commands shares a flag is not kept by
    the flag existing on one other command, and the union cannot tell those
    apart.
    """
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return {
                string
                for subaction in action.choices[command]._actions
                for string in subaction.option_strings
            }
    raise AssertionError("the parser registers no subcommands")


def table_rows(text: str, columns: int) -> dict[str, tuple[str, ...]]:
    """Markdown table rows of exactly `columns` cells, keyed by the first cell.

    The key is stripped of its backticks, because every table here names a tool
    or a command in code style. Rows of another width belong to another table
    and are skipped, which is what lets one document carry more than one.
    """
    rows: dict[str, tuple[str, ...]] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != columns or not cells[0].startswith("`"):
            continue
        rows[cells[0].strip("`")] = tuple(cells[1:])
    return rows


def registered_options() -> set[str]:
    """Every option string the parser registers, on any subcommand."""
    parser = build_parser()
    options = {string for action in parser._actions for string in action.option_strings}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                options.update(
                    string
                    for subaction in subparser._actions
                    for string in subaction.option_strings
                )
    return options


def registered_subcommands() -> set[str]:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("the parser registers no subcommands")


def declared_default_output_roots(text: str) -> list[str]:
    """Every path `text` declares as where output lands by default.

    Found by what carries the claim - a sentence about `output` that says
    `default`, and a path written in it - rather than by one exact phrasing.
    Matching a phrase does two things wrong at once: a *true* rewording fails
    the test, with a message saying the README states no default at all; and a
    second sentence naming a different default passes, because a search stops at
    its first hit.

    The limit: a sentence about output and defaults that names a path
    incidentally is a claim as far as this is concerned. That way round on
    purpose - a spurious one is answered by rewording a sentence, and a missed
    one is a documented default nothing keeps.
    """
    return [
        span
        for sentence in sentences(text)
        if "output" in sentence.lower() and "default" in sentence.lower()
        for span in INLINE_CODE.findall(sentence)
        if span.startswith("~/")
    ]


def test_the_readme_states_the_default_output_root_the_code_uses() -> None:
    """The output directory finding 13 was about: documented, once, and true.

    `~/.distill` was documented as the default for the whole of 0.1.0 and never
    was one - it is the config directory. The path is read back out of the
    README and expanded, so a documented default that drifts from
    `validate_output_root`'s fails here rather than sending a reader to an empty
    directory.

    Exactly one declaration, because two are worse than a wrong one: a reader
    who finds the right sentence has no way to know the other exists, and a
    reader who finds the other has no reason to doubt it.
    """
    declared = declared_default_output_roots(prose(README))

    assert len(declared) == 1, (
        f"the README declares {len(declared)} default output roots: {declared}"
    )
    assert Path(declared[0]).expanduser() == validate_output_root(None, create=False)


def test_phase_one_configuration_claims_match_their_mechanisms() -> None:
    readme = README.read_text(encoding="utf-8")
    configuration = readme.split("## Configuration", 1)[1].split("## Commands", 1)[0]
    flowed_configuration = flowed(configuration)
    lower_configuration = flowed_configuration.lower()
    config_source = CONFIG_SOURCE.read_text(encoding="utf-8")
    version_source = VERSION_SOURCE.read_text(encoding="utf-8")

    assert "endpoint policy" in lower_configuration
    assert "fatal" in lower_configuration
    assert "`cache_mode`" in configuration
    assert "tool argument" in flowed_configuration
    assert "option table" in flowed_configuration
    assert "argparse" in flowed_configuration
    assert "2.5" in flowed_configuration
    assert "E_BAD_OPTIONS" in flowed_configuration
    assert "Configuring one does not configure the other" in flowed(readme)
    assert "explicit override is authoritative even when it is absent" in flowed(config_source)
    assert "every processing option there is part of the options" not in version_source


def test_the_readme_states_the_class_and_cost_the_capability_table_records() -> None:
    """ADR-0002 in the README, per tool rather than as a blanket promise.

    The three-column table states what an absence costs; the four-column
    dependency table above it states the class, and `test_capabilities.py` holds
    that one. Both are checked here against `EXTERNAL_TOOLS`, because a cost
    stated for the wrong class is the reading of ADR-0002 that finding 13 was.
    """
    rows = table_rows(README.read_text(encoding="utf-8"), columns=3)

    for name, tool in EXTERNAL_TOOLS.items():
        assert name in rows, f"{name} is classified but the README states no absence cost"
        requirement, cost = rows[name]
        assert requirement == str(tool.requirement), name
        assert cost == tool.absence_cost, name


def test_agents_carries_the_same_capability_table() -> None:
    """R-51's other half: the same table, for the agent that reads AGENTS.md."""
    rows = table_rows(AGENTS.read_text(encoding="utf-8"), columns=4)

    for name, tool in EXTERNAL_TOOLS.items():
        assert name in rows, f"{name} is classified but AGENTS.md does not list it"
        capability, requirement, cost = rows[name]
        assert capability == tool.capability, name
        assert requirement == str(tool.requirement), name
        assert cost == tool.absence_cost, name


def test_the_readme_commands_table_lists_every_registered_subcommand() -> None:
    """ "The following subcommands" has to be all of them, and only them.

    Read off `build_parser`'s own registry, so a command added without a row -
    which is how `cache-doctor` came to be documented only in an upgrade note -
    fails here.

    Equality and not containment. Containment holds the direction where a
    command exists and is undocumented, and lets the opposite through
    unremarked: a row for `frobnicate` documents a command nobody can run, which
    is a reader following instructions into a usage error.
    """
    documented = {command.split()[0] for command in table_rows(prose(README), columns=2)}

    assert documented == registered_subcommands()


def test_the_readme_names_every_shared_processing_option() -> None:
    """The flag list the processing commands share, against the list they take.

    Scoped to the paragraph that claims to be the list, not to the README as a
    whole: a flag documented only in the section about what it is for leaves the
    list itself stale, which is how `--local-vision-allow-remote-endpoint` came
    to be missing from it.

    Both directions, and the second one is scoped per command. A flag in this
    paragraph is claimed to be taken by *each* of the commands the paragraph
    names, and the union of every subcommand's options cannot judge that: it is
    satisfied by `--dry-run`, which `cleanup-cache` registers and none of these
    three do, so the paragraph could claim a flag that every one of the commands
    it is about would reject.
    """
    listing = re.search(
        r"The processing commands \((?P<commands>.*?)\).*?for the full list\.",
        flowed(prose(README)),
        re.DOTALL,
    )

    assert listing is not None, "the README lists no shared processing options"
    named = {flag for span in INLINE_CODE.findall(listing.group(0)) for flag in FLAG.findall(span)}
    for key in PROCESSING_KEYS:
        flag = f"--{key.replace('_', '-')}"
        assert flag in named, f"{flag} is accepted but the list omits it"

    commands = INLINE_CODE.findall(listing.group("commands"))
    assert commands, "the paragraph names no commands the options are shared by"
    for command in commands:
        registered = subparser_options(command)
        assert named <= registered, f"{command} does not take {sorted(named - registered)}"


def distill_flags(text: str) -> set[str]:
    """The flags a document names as Distill's, read off its inline code spans.

    A span holding a whole command line belongs to whichever program it starts
    with, so `uv run python tests/evals/score.py --with-vision` contributes
    nothing and `distill cleanup-cache --dry-run` contributes `--dry-run`. A
    span that is a flag on its own is always Distill's - that is the only reason
    these documents write one.

    The limit this leaves: a flag inside a fenced block, or inside a command
    line some other program owns, is not checked here.
    """
    return {
        flag
        for span in INLINE_CODE.findall(text)
        if not span.split()[1:] or span.startswith("distill ")
        for flag in FLAG.findall(span)
    }


@pytest.mark.parametrize("document", [README, AGENTS], ids=["README", "AGENTS"])
def test_no_document_names_a_flag_the_parser_does_not_register(document: Path) -> None:
    """A flag in prose that no command takes is a claim nothing keeps."""
    registered = registered_options()
    named = distill_flags(prose(document))

    assert named <= registered, sorted(named - registered)


def test_the_readme_states_the_sampling_bounds_the_fingerprint_actually_reads() -> None:
    """R-51's collision property is stated in terms of what is read.

    The numbers a reader is asked to reason about - how many anchors, how much
    at each, how much in total, and where `content` stops - are derived here
    from the code that does the reading, so a widened anchor set leaves the
    README's arithmetic failing rather than quietly wrong.

    Each number is asserted *inside the phrase that carries its claim*, not as a
    free-floating substring. Loose, the two byte counts were interchangeable:
    swapping them in the README - 576 KiB at each of nine anchors, bounding a
    lookup at 64 KiB, wrong by a factor of 81 in both directions - left both
    strings present and the test green.
    """
    documented = flowed(prose(README))
    anchors = len(fingerprint_anchor_offsets(FINGERPRINT_SAMPLE_BYTES * 1000))
    total_kib = anchors * FINGERPRINT_SAMPLE_BYTES // 1024

    assert (
        f"{FINGERPRINT_SAMPLE_BYTES // 1024} KiB read at each of "
        f"{NUMBER_WORDS[anchors]} anchors" in documented
    )
    assert f"{NUMBER_WORDS[FINGERPRINT_INTERIOR_ANCHORS]} evenly spread interior" in documented
    assert f"bounds a cache lookup at {total_kib} KiB" in documented
    assert f"refuses files over {CONTENT_HASH_LIMIT_BYTES // 1024**3} GB" in documented
