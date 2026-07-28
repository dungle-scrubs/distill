"""What `untrusted_blocks` is, checked against a reader that is one.

Every assertion about the **untrusted-data boundary** - in a **render** and in
the vision prompt alike - is built on `untrusted_blocks._scan`, so what that
scanner is has to be stated correctly or every test above it is trusted for the
wrong reason. It is not a CommonMark parser. It is a conservative validator of
*explicit closure*: a block counts only where Distill opened it, tagged it and
wrote a closing fence, and where CommonMark is more forgiving than that the
scanner declines to follow (D-022 - the claim has to match what the code does).

This file is what keeps that sentence honest. The one divergence is exercised
against `commonmark_ast`, which parses with `markdown-it-py`, so "stricter than
a reader, and stricter in the direction that reports an escape rather than
hiding one" is a comparison rather than a docstring's word for itself. Where
the two agree, they are asserted to agree, because a validator that diverged
anywhere else would be measuring something other than the boundary.

It does not own what a render or a prompt should say - `test_render_delimiting`
and `test_vision_prompts` own those - only what the shared scanner reports.
"""

from __future__ import annotations

import re

import pytest
from commonmark_ast import code_blocks
from untrusted_blocks import SENTINEL, assert_delimited, fenced_blocks, outside_fences

from distill.emit import EMITTER, UNTRUSTED_TEXT_LABEL

MARKER = "SCANNER-BOUNDARY-MARKER"


def document(*block: str) -> str:
    """`block` set in a document that has structure on both sides of it."""
    return "\n".join(["# Heading", "", *block, "", "Trailing line."])


def closed_block(body: str) -> str:
    """A document holding one block Distill opened, tagged and closed."""
    return document(*EMITTER.delimit(body))


def unclosed_block(body: str) -> str:
    """The same document with the closing fence missing.

    The failure the scanner exists to catch: a **render** that opened a block
    and never ended it. Nothing Distill ships emits this - `emit.delimit`
    always writes the closer - so it is built here by dropping that line.
    """
    return document(*EMITTER.delimit(body)[:-1])


def test_a_closed_block_holds_the_text_a_commonmark_reader_reads() -> None:
    """Where the scanner and a real reader agree, they agree exactly.

    A validator that reported a different body than the reader does would be
    answering a different question than "did the payload stay quoted?".
    """
    document = closed_block(f"{SENTINEL} - {MARKER}")

    scanned = fenced_blocks(document)
    parsed = code_blocks(document)

    assert [block.body for block in scanned] == [
        block.content.rstrip("\n") for block in parsed
    ]
    assert [block.info for block in scanned] == [block.info for block in parsed]
    assert scanned[0].info == UNTRUSTED_TEXT_LABEL


def test_a_block_nothing_closed_is_code_to_a_reader_and_structure_here() -> None:
    """The one divergence, stated as the comparison it is.

    CommonMark ends an unclosed fence at the end of the containing block or the
    document, so everything after the opener is code to a reader. The scanner
    refuses that: a block that was never closed is not a block, and its
    contents go back to the structure they are in fact part of.
    """
    document = unclosed_block(f"{SENTINEL} - {MARKER}")

    assert [block.content.rstrip("\n") for block in code_blocks(document)] == [
        f"{SENTINEL} - {MARKER}\n\nTrailing line."
    ]
    assert fenced_blocks(document) == []
    assert MARKER in outside_fences(document)


def test_the_divergence_reports_the_escape_a_reader_would_hide() -> None:
    """Strictness is only worth having if it points the same way every time.

    Were the scanner to close a block the way a reader does, a **render** that
    opened a block and never closed it would swallow the rest of the document
    and every later payload would read as quoted - the assertion would pass
    *because* the delimiter failed. It reports the payload undelimited instead,
    which is a false alarm a maintainer can look at rather than a failure that
    hides itself.
    """
    document = unclosed_block(f"{SENTINEL} - {MARKER}")

    with pytest.raises(AssertionError, match="is not inside an untrusted-text block"):
        assert_delimited(document, SENTINEL, MARKER)


def test_the_docstrings_say_which_of_the_two_this_scanner_is() -> None:
    """D-022: the scanner may not be described as the reader it is stricter than.

    The words a maintainer trusts are the ones in the module, so the claim is
    read back out of it: it names itself a validator of explicit closure, and
    it names the divergence rather than leaving a reader to find it.
    """
    import untrusted_blocks

    prose = f"{untrusted_blocks.__doc__}\n{untrusted_blocks._scan.__doc__}"

    assert "explicit clos" in prose
    assert "not a CommonMark parser" in prose
    assert re.search(r"strict(er|ly)?\b", prose)
    assert "the way a CommonMark reader does" not in prose
