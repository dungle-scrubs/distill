"""A **render** read by a real CommonMark parser, not by a scanner of our own.

This module owns the answer to "what does a reader actually get?" - where a link
points and what its label says, which is what `emit.link_label` and
`emit.link_destination` have to be true of, and which regions a reader treats as
code rather than as document structure. It answers by parsing the render with
`markdown-it-py` in CommonMark mode and reporting the resulting token stream -
destinations as the parser resolved them, label text as the parser decoded it,
code blocks with the reader's own closing rule rather than a stricter one.

Why not the scanner this replaced. `tests/test_render_delimiting.py` used to
find links by walking the text and undoing backslash escapes, and it could only
see the escapes its author had thought of. CommonMark also decodes **entity and
numeric character references**, in a label *and* in a destination, so a label of
`&copy;` was a copyright sign to a reader and `?q=&copy;` resolved to
`?q=%C2%A9` - a destination changed by its own content, which is the retarget
R-27 exists to stop, reached through entities instead of through brackets. The
scanner reported the raw bytes and called it equal. A test that cannot see the
defect is not evidence, so the parser used here is one this suite does not own.

Destinations come back percent-decoded (`mdurl`, the decoder markdown-it encodes
with), because a parser normalizes a destination on the way in: a space becomes
`%20`, a line ending `%0A`. Decoding is what makes "the reader resolved exactly
the URL the carrier held" a comparison that can be made at all. `%2F` and its
neighbours are left encoded, which is `mdurl`'s own default and the reason a
destination that *carried* `%2F` still round-trips.

What this module does not own. It does not know what a render should say, which
regions are **extracted text**, or which fences Distill was entitled to open: it
reports the blocks a reader resolved, whoever opened them, and `untrusted_blocks`
is what asks whether the *render* opened, tagged and explicitly closed one. The
tests own the claims; this module reports what a reader saw and nothing about
whether that was right.

One property worth naming because assertions here rely on it: content inside a
fenced block is not parsed as inline content, so a payload that wrote `[a](b)`
inside a block produces no link here. A link this module reports is a link a
reader has, which is the question R-27 asks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdurl import decode as percent_decode

READER = MarkdownIt("commonmark")
"""CommonMark mode: no tables, no strikethrough, no linkify - the dialect the
render is written in and the smallest one a downstream reader is likely to use."""

LinkKind = Literal["inline", "autolink", "image"]


@dataclass(frozen=True)
class CodeBlock:
    """One code block as a reader resolved it: its info string and its content.

    `content` is the text the reader treats as code rather than as document
    structure, which is the question a delimiter exists to settle.
    """

    info: str
    content: str


@dataclass(frozen=True)
class Link:
    """One link construct as a reader resolved it.

    `text` is the label after the reader decoded it - backslash escapes undone,
    entity and numeric character references resolved - which is the string a
    human or a model actually reads. `destination` is percent-decoded for the
    same reason.
    """

    text: str
    destination: str
    kind: LinkKind


def _inline_tokens(markdown: str) -> list[Token]:
    """Every inline token stream in the document, fenced blocks excluded by the parser."""
    return [token for token in READER.parse(markdown) if token.type == "inline"]


def _resolved(token: Token, attribute: str) -> str:
    """`attribute` off `token`, percent-decoded, as the string a reader resolved.

    `attrGet` is typed loosely enough to return a number, which no destination
    ever is; the coercion is what keeps that a test-helper detail rather than a
    type error at every call site.
    """
    return percent_decode(str(token.attrGet(attribute) or ""))


def _text_of(children: list[Token] | None) -> str:
    """The literal text a reader sees for `children`.

    A soft or hard break is a line ending to a reader, so it is reported as one.
    Raw HTML is reported as the text it is written as, because a test asking
    whether a payload became a tag asks `raw_html`, not this.
    """
    if not children:
        return ""
    parts: list[str] = []
    for child in children:
        if child.type in ("softbreak", "hardbreak"):
            parts.append("\n")
        elif child.children:
            parts.append(_text_of(child.children))
        else:
            parts.append(child.content)
    return "".join(parts)


def links(markdown: str) -> list[Link]:
    """Every link and image a CommonMark reader resolves, in document order."""
    found: list[Link] = []
    for inline in _inline_tokens(markdown):
        children = inline.children or []
        index = 0
        while index < len(children):
            token = children[index]
            if token.type == "image":
                found.append(
                    Link(
                        text=_text_of(token.children),
                        destination=_resolved(token, "src"),
                        kind="image",
                    )
                )
                index += 1
                continue
            if token.type != "link_open":
                index += 1
                continue
            depth = 1
            cursor = index + 1
            while cursor < len(children) and depth:
                if children[cursor].type == "link_open":
                    depth += 1
                elif children[cursor].type == "link_close":
                    depth -= 1
                cursor += 1
            found.append(
                Link(
                    text=_text_of(children[index + 1 : cursor - 1]),
                    destination=_resolved(token, "href"),
                    kind="autolink" if token.markup == "autolink" else "inline",
                )
            )
            index = cursor
    return found


def destinations(markdown: str) -> list[str]:
    """Where every link and image in the document points, as the reader resolved it."""
    return [link.destination for link in links(markdown)]


def autolinks(markdown: str) -> list[str]:
    """The destinations of the `<scheme:...>` links a reader makes on its own.

    Separate from the rest because a link may not hold another link: an autolink
    inside a label wins, and the destination the render chose is dropped.
    """
    return [link.destination for link in links(markdown) if link.kind == "autolink"]


def code_blocks(markdown: str) -> list[CodeBlock]:
    """Every code block a CommonMark reader resolves, in document order.

    The reader's own closing rule, including the one `untrusted_blocks` declines
    to follow: a fence nothing ever closes runs to the end of the document, and
    everything it swallowed is code. That is what makes "the helper is stricter
    than a reader, and stricter in the direction that reports an escape rather
    than hiding one" a claim a test can check instead of a sentence in a
    docstring.
    """
    return [
        CodeBlock(info=token.info.strip(), content=token.content)
        for token in READER.parse(markdown)
        if token.type in ("fence", "code_block")
    ]


def raw_html(markdown: str) -> list[str]:
    """Every fragment the reader treats as HTML rather than as text."""
    found: list[str] = []
    for token in READER.parse(markdown):
        if token.type == "html_block":
            found.append(token.content)
        for child in token.children or []:
            if child.type == "html_inline":
                found.append(child.content)
    return found


def inline_text(markdown: str) -> str:
    """The document's inline content as a reader reads it, one line per block.

    What a payload *said* after every escape is undone, which is how a test
    asserts that escaping a construct did not also destroy the words in it.
    """
    return "\n".join(_text_of(inline.children) for inline in _inline_tokens(markdown))
