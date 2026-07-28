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


def test_the_readme_states_the_default_output_root_the_code_uses() -> None:
    """The output directory finding 13 was about: documented, and true.

    `~/.distill` was documented as the default for the whole of 0.1.0 and never
    was one - it is the config directory. The sentence is read back out of the
    README and expanded, so a documented default that drifts from
    `validate_output_root`'s fails here rather than sending a reader to an empty
    directory.
    """
    stated = re.search(r"Output lands under `([^`]+)` by default", README.read_text())

    assert stated is not None, "the README does not state a default output root"
    assert Path(stated.group(1)).expanduser() == validate_output_root(None, create=False)


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
    """"The following subcommands" has to be all of them.

    Read off `build_parser`'s own registry, so a command added without a row -
    which is how `cache-doctor` came to be documented only in an upgrade note -
    fails here.
    """
    documented = {
        command.split()[0] for command in table_rows(prose(README), columns=2)
    }

    assert registered_subcommands() <= documented


def test_the_readme_names_every_shared_processing_option() -> None:
    """The flag list the processing commands share, against the list they take.

    Scoped to the paragraph that claims to be the list, not to the README as a
    whole: a flag documented only in the section about what it is for leaves the
    list itself stale, which is how `--local-vision-allow-remote-endpoint` came
    to be missing from it.
    """
    listing = re.search(
        r"The processing commands \(.*?for the full list\.",
        prose(README),
        re.DOTALL,
    )

    assert listing is not None, "the README lists no shared processing options"
    for key in PROCESSING_KEYS:
        flag = f"--{key.replace('_', '-')}"
        assert f"`{flag}`" in listing.group(0), f"{flag} is accepted but the list omits it"


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
    """
    documented = prose(README)
    anchors = len(fingerprint_anchor_offsets(FINGERPRINT_SAMPLE_BYTES * 1000))
    total_kib = anchors * FINGERPRINT_SAMPLE_BYTES // 1024

    assert f"{FINGERPRINT_SAMPLE_BYTES // 1024} KiB" in documented
    assert f"{NUMBER_WORDS[anchors]} anchors" in documented
    assert f"{NUMBER_WORDS[FINGERPRINT_INTERIOR_ANCHORS]} evenly spread interior" in documented
    assert f"{total_kib} KiB" in documented
    assert f"{CONTENT_HASH_LIMIT_BYTES // 1024**3} GB" in documented
