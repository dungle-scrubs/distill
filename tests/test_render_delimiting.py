"""The **untrusted-data boundary** a **render** puts around **extracted text**.

A render is written to be fed to an LLM agent, and every word of extracted text
in it was chosen by whoever produced the **source**. It is therefore
attacker-controlled input to a downstream model, and the boundary is what marks
it as quoted data and reduces the chance of it being read as Distill's own
instructions (R-24 through R-27, finding 5, RV-5).

The boundary is a mitigation and not a guarantee. A sufficiently persuasive
payload may still influence a model that reads the render; what these tests
check is that a payload did not *stop being quoted* - that it did not close the
fence it is in, retarget the link it labels, or emit a heading, a bullet or an
instruction line the document structure vouches for. Raising the cost of a
payload and making its provenance legible is all this buys, and no test here
claims more (D-022).

The assertions are structural rather than textual on purpose. `untrusted_blocks`
is a conservative validator of explicit closure rather than a CommonMark parser,
so a test passes because the payload is inside a block the render opened, tagged
and closed, not because the render happened to contain a string; `outside_fences`
is everything it will not vouch for as quoted. Where a test needs the answer a
reader actually resolves - which is the question R-27 asks of a link - it asks
`commonmark_ast`, which parses with `markdown-it-py`.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from commonmark_ast import autolinks, destinations, inline_text, links, raw_html
from untrusted_blocks import (
    SENTINEL,
    assert_delimited,
    attack,
    outside_fences,
    untrusted_bodies,
)

from distill.artifacts import FrameArtifact, Interpretation, Provenance, Transcript
from distill.emit import EMITTER, UNTRUSTED_TEXT_LABEL
from distill.links import RelatedLink
from distill.render import render_markdown


def keyframe(**overrides: Any) -> FrameArtifact:
    """One **frame artifact** as the pipeline hands it to the **render**."""
    fields: dict[str, Any] = {
        "index": 1,
        "timestamp_sec": 2.0,
        "path": "/tmp/frames/frame_0001.png",
        "relative_path": "frames/frame_0001.png",
    }
    fields.update(overrides)
    return FrameArtifact(**fields)


def spoken(*segments: dict[str, Any]) -> Transcript:
    """A **transcript** carrier holding exactly these segments."""
    return Transcript(language="en", segments=tuple(segments))


def read(**overrides: Any) -> Interpretation:
    """An **interpretation** with only the field under test filled in."""
    return Interpretation(**overrides)


def frame_reading(reading: Interpretation, **frame_fields: Any) -> FrameArtifact:
    frame, _warnings = keyframe(**frame_fields).with_interpretation(reading)
    return frame


def render(**overrides: Any) -> str:
    """A render of the smallest bundle that has anything to say."""
    arguments: dict[str, Any] = {
        "source_label": "demo.mp4",
        "duration_sec": 10.0,
        "transcript": None,
        "frames": [keyframe(extracted_text="slide text")],
        "warnings": [],
        "related_links": None,
        "provenance": None,
        "include_frame_links": True,
    }
    arguments.update(overrides)
    return render_markdown(
        arguments["source_label"],
        arguments["duration_sec"],
        arguments["transcript"],
        arguments["frames"],
        arguments["warnings"],
        arguments["related_links"],
        provenance=arguments["provenance"],
        include_frame_links=arguments["include_frame_links"],
    )


# --- R-25: the fence a payload cannot terminate -----------------------------


def test_extracted_image_text_holding_a_fence_cannot_terminate_its_block() -> None:
    """Finding 5: OCR text carrying ``` closed the block it was quoted in.

    The render fenced image text with exactly three backticks, so a slide
    showing a code sample - or a payload written to look like one - ended the
    quotation and continued as document structure. Everything after it was then
    the render's own voice as far as a reading model is concerned.
    """
    markdown = render(frames=[keyframe(extracted_text=attack("ocr"))])

    assert_delimited(markdown, SENTINEL, "# ocr")


@pytest.mark.parametrize("run", [1, 2, 3, 4, 7])
def test_the_fence_is_longer_than_the_longest_backtick_run_in_the_content(run: int) -> None:
    """R-25's rule is mechanical, not a fixed fence length picked to be roomy.

    Whatever the longest run of backticks in the content is, the fence is
    longer, and it is never shorter than the three a fenced block needs.
    """
    text = f"before {'`' * run} after"

    fence = EMITTER.fence_for(text)

    assert set(fence) == {"`"}
    assert len(fence) > run
    assert len(fence) >= 3
    assert len(fence) == max(3, run + 1)


def test_a_payload_of_nested_fences_is_still_one_block_to_a_reader() -> None:
    """The property the rule buys, checked through the render rather than stated."""
    text = "````\n```\nstill inside\n```\n````"
    markdown = render(frames=[keyframe(extracted_text=text)])

    assert any("still inside" in body for body in untrusted_bodies(markdown))
    assert "still inside" not in outside_fences(markdown)


# --- R-26: every extracted-text region of a frame ---------------------------


def test_a_vision_caption_containing_newlines_does_not_become_document_structure() -> None:
    """Finding 5: an **interpretation** field was emitted as a bare bullet.

    `- Summary: {text}` ends at the model's first newline, so everything after
    it is a line of the document at the document's own indentation - a heading,
    a bullet, an instruction - and nothing marks it as the model's report of
    what a **keyframe** showed.
    """
    markdown = render(frames=[frame_reading(read(visual_summary=attack("summary")))])

    assert "- Summary:" not in markdown
    assert_delimited(markdown, SENTINEL, "# summary")


def test_the_vision_summary_is_delimited_rather_than_inlined() -> None:
    markdown = render(frames=[frame_reading(read(visual_summary="A settings form"))])

    assert "A settings form" in untrusted_bodies(markdown)


def test_the_interpretation_field_is_delimited() -> None:
    markdown = render(frames=[frame_reading(read(interpretation=attack("interpretation")))])

    assert_delimited(markdown, SENTINEL, "# interpretation")


def test_detected_elements_are_delimited() -> None:
    """One element per line inside the block, so a comma in one is not a boundary."""
    markdown = render(
        frames=[
            frame_reading(read(detected_elements=("save button", attack("element"), "title")))
        ]
    )

    assert_delimited(markdown, SENTINEL, "# element")
    body = next(body for body in untrusted_bodies(markdown) if "save button" in body)
    assert body.splitlines()[0] == "save button"
    assert "title" in body.splitlines()


def test_uncertainty_text_is_delimited() -> None:
    markdown = render(frames=[frame_reading(read(uncertainty=attack("uncertainty")))])

    assert_delimited(markdown, SENTINEL, "# uncertainty")


def test_text_confidence_is_delimited() -> None:
    """The model's word about its own reading is the model's word, not Distill's."""
    markdown = render(frames=[frame_reading(read(text_confidence=attack("confidence")))])

    assert_delimited(markdown, SENTINEL, "# confidence")


def test_verbatim_text_is_delimited() -> None:
    markdown = render(frames=[frame_reading(read(verbatim_text=attack("verbatim")))])

    assert_delimited(markdown, SENTINEL, "# verbatim")


def test_the_low_confidence_banner_is_bounded_to_one_line() -> None:
    """The banner is Distill's own voice, and it is held to being one line.

    A **grounding** is Distill's assessment and its level and reason are
    literals in `grounding.py`, so the banner is not delimited - the render is
    speaking. But a `GroundingAssessment` rebuilt from a document takes what the
    document held, and "another module only ever writes literals here" is a
    claim about that module rather than a property of this text.

    Two lines hold the banner to one line and they are not the same claim. The
    escape is what makes a line ending unable to end the *document* line, and it
    is what this asserts first. Collapsing the whitespace is what keeps the
    result a sentence, and that is asserted through a reader - the escape
    preserves the line ending rather than destroying it (R-27), so a banner that
    stopped collapsing would still occupy one line of the render while reading
    as several. A reason nobody can read is a banner that has stopped reporting.
    """
    frame, _warnings = keyframe().with_interpretation(
        read(visual_summary="A dark slide"),
        grounding={
            "level": "ungrounded",
            "text_overlap": None,
            "reason": f"no readable text\n\n# grounding\n\n{SENTINEL}",
        },
    )
    markdown = render(frames=[frame])
    banner = next(
        line for line in outside_fences(markdown).split("\n") if "Low-confidence frame" in line
    )

    assert SENTINEL in banner
    assert not any(
        line.lstrip().startswith("# grounding") for line in outside_fences(markdown).split("\n")
    )
    assert "\n" not in inline_text(banner)


def test_the_banner_names_a_level_only_when_it_is_one() -> None:
    """A `level` this codebase does not define is not repeated as though it were one.

    `GroundingAssessment.from_document` passes an unrecognized level through
    deliberately - anything outside `NOT_LOW_CONFIDENCE` reads as low
    confidence - so the banner says low confidence for it without quoting the
    string back.
    """
    frame, _warnings = keyframe().with_interpretation(
        read(visual_summary="A dark slide"),
        grounding={
            "level": f"weak): {SENTINEL} (",
            "text_overlap": None,
            "reason": "no readable text",
        },
    )
    markdown = render(frames=[frame])

    assert "Low-confidence frame (low):" in markdown
    assert SENTINEL not in markdown


def test_the_banner_cannot_open_a_construct_from_its_reason() -> None:
    """The banner is Distill's sentence, so what lands in it must not be able to act.

    Folding the reason onto one line closes the block-level escape and nothing
    else: a link, an autolink and a raw tag are all inline, so a reason rebuilt
    from a **stage result** on **resume** put a live link and raw HTML in the
    sentence the document presents as its own assessment.

    Escaping rather than delimiting, because the banner is Distill speaking:
    the same rule a link label runs under neutralizes every opener without
    labelling Distill's words as the source's, and it is lossless, so the
    reason still reads as what it said.
    """
    reason = f"unreadable: see [docs]({ATTACKER_URL}), <{ATTACKER_URL}>, <b>read this &amp; that</b>"
    frame, _warnings = keyframe().with_interpretation(
        read(visual_summary="A dark slide"),
        grounding={"level": "ungrounded", "text_overlap": None, "reason": reason},
    )
    markdown = render(frames=[frame])
    banner = next(
        line for line in outside_fences(markdown).split("\n") if "Low-confidence frame" in line
    )

    assert destinations(markdown) == [FRAME_IMAGE]
    assert autolinks(markdown) == []
    assert raw_html(markdown) == []
    assert reason in inline_text(banner)


# --- R-26: the transcript ---------------------------------------------------


def test_transcript_segment_text_is_delimited() -> None:
    """Speech is extracted text: the speaker chose the words, Distill only timed them."""
    markdown = render(
        transcript=spoken({"start": 0.0, "end": 5.0, "text": attack("segment"), "words": []}),
        frames=[],
    )

    assert_delimited(markdown, SENTINEL, "# segment")


def test_transcript_words_split_around_a_keyframe_stay_delimited() -> None:
    """A segment split at a **keyframe** emits its words twice, so it has two sinks.

    The words before the frame and the words after it are separate emissions,
    and a payload spoken on either side of a slide change has to land inside a
    block on both.
    """
    markdown = render(
        transcript=spoken(
            {
                "start": 0.0,
                "end": 3.0,
                "text": "first second",
                "words": [
                    {"word": "```", "start": 0.0, "end": 0.5},
                    {"word": f"#-before-{SENTINEL}", "start": 0.5, "end": 1.0},
                    {"word": "```", "start": 2.0, "end": 2.2},
                    {"word": f"#-after-{SENTINEL}", "start": 2.2, "end": 2.5},
                ],
            }
        ),
        frames=[keyframe(timestamp_sec=1.5)],
    )
    bodies = untrusted_bodies(markdown)

    assert SENTINEL not in outside_fences(markdown)
    assert any(f"#-before-{SENTINEL}" in body for body in bodies)
    assert any(f"#-after-{SENTINEL}" in body for body in bodies)


# --- R-26 with D-028: the regions that do not look like extracted text ------


def test_the_source_label_is_delimited() -> None:
    """RV-5: a filename is extracted text, and it was rendered inside backticks.

    Whoever produced the **source** named the file, and a YouTube title reaches
    the same field. One backtick in it ended the code span; a newline in it
    ended the bullet.
    """
    markdown = render(source_label=attack("source"))

    assert_delimited(markdown, SENTINEL, "# source")
    assert "- Source: `" not in markdown


def test_warning_messages_are_delimited() -> None:
    """A **warning** carries text Distill did not write: a path, a tool's complaint."""
    markdown = render(
        warnings=[{"stage": "ocr", "code": "tesseract_failed", "message": attack("warning")}]
    )

    assert_delimited(markdown, SENTINEL, "# warning")


def test_exception_text_carried_in_a_warning_is_delimited() -> None:
    """The text of a failure is the failing thing's text, not Distill's.

    A tool's stderr, a filename, a URL - all of it reaches a **warning** message
    through `str(exception)`, and all of it was chosen somewhere other than
    here.
    """
    failure = OSError(f"cannot read `{attack('exception')}`")
    markdown = render(
        warnings=[{"stage": "source", "code": "probe_failed", "message": str(failure)}]
    )

    assert_delimited(markdown, SENTINEL, "# exception")


def test_a_render_with_no_warnings_has_no_warning_section() -> None:
    markdown = render()

    assert "## Warnings" not in markdown
    assert "- Warnings: 0" in markdown


# --- R-27: link constructs --------------------------------------------------


def link(**overrides: Any) -> RelatedLink:
    fields: dict[str, Any] = {
        "url": "https://github.com/example/repo",
        "label": "Example repo",
        "source": "youtube_description",
        "reason": "code_or_reference",
    }
    fields.update(overrides)
    return RelatedLink(**fields)


ATTACKER_URL = "https://attacker.example/evil"
FRAME_IMAGE = "frames/frame_0001.png"


def link_halves(markdown: str) -> list[tuple[str, str]]:
    """Every link a reader resolves, as `(what it says, where it points)`."""
    return [(found.text, found.destination) for found in links(markdown)]


def outside_links(text: str) -> str:
    """The lines of `text` no link sits on.

    A link label and a destination are escaped rather than fenced (R-27), so a
    payload in either stays visible as prose - held to the one line the link
    occupies, which is the whole of what escaping buys there. Everything else
    is where the render speaks in its own voice, and a payload reaching it is
    the finding.
    """
    return "\n".join(line for line in text.split("\n") if "](" not in line)


@pytest.mark.parametrize(
    "hostile",
    [
        f"docs]({ATTACKER_URL}) [rest",
        f"<{ATTACKER_URL}>",
    ],
    ids=["closes-the-label", "autolink"],
)
def test_a_link_label_cannot_retarget_the_link(hostile: str) -> None:
    """RV-5: a label can name the destination unless every opener in it is escaped.

    The label is extracted text - a description's author wrote it - and two
    shapes of ordinary text take the link over. `docs](https://…) [rest` closes
    the label and opens a destination the label's author chose. `<https://…>`
    is an autolink, and since a link may not hold another link, a reader drops
    the render's own link and leaves the label's URL as the live one. Either
    way the render showed a plausible name over an attacker's URL.

    A code span in a label is not a third shape: a backtick swallows the `]`
    that would close the label rather than retargeting anything, so escaping it
    buys legibility and not this.
    """
    markdown = render(related_links=[link(label=hostile)])

    assert destinations(markdown) == ["https://github.com/example/repo", FRAME_IMAGE]
    assert autolinks(markdown) == []
    assert (hostile, "https://github.com/example/repo") in link_halves(markdown)


@pytest.mark.parametrize(
    ("label", "url"),
    [
        ("&copy;", "https://ok.example/?q=&copy;"),
        ("a&amp;b", "https://ok.example/a&amp;b"),
        ("&#35; not a heading", "https://ok.example/a&#41;b"),
        ("&notanentity;", "https://ok.example/?a=1&b=2"),
    ],
    ids=["named", "amp", "numeric", "unterminated"],
)
def test_an_entity_reference_in_a_link_cannot_change_what_it_says_or_where_it_goes(
    label: str, url: str
) -> None:
    """CommonMark decodes entity references, in a label and in a destination alike.

    The backslash escapes cover every character that could *close* a construct
    and none that can be *rewritten* inside one. `&copy;` is six characters a
    reader turns into one, so a label read as something the source never wrote
    and, worse, a destination read as somewhere the source never pointed:
    `?q=&copy;` resolves to `?q=%C2%A9`, and `a&amp;b` to `a&b`. That is the
    same retarget R-27 exists to stop, reached through an entity rather than
    through a bracket - the destination is changed by what it carries.
    """
    markdown = render(related_links=[link(label=label, url=url)])

    assert link_halves(markdown) == [(label, url), ("Frame 1", FRAME_IMAGE)]


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_a_link_label_holding_a_line_ending_is_read_back_as_that_line_ending(
    ending: str,
) -> None:
    """The label is escaped rather than fenced, and R-27 calls that lossless.

    Writing a line ending as the two visible characters `\\n` is not lossless:
    a reader gets a backslash and a letter, and a label that carried a literal
    backslash followed by `n` is written the same way, so two different sources
    become one string nobody can tell apart. A numeric character reference is
    the way out - `&#10;` is not a line ending in the source, so the label is
    still one line and cannot open a heading, and it *is* a line ending to the
    reader, so the text comes back whole.
    """
    label = f"first{ending}second"
    markdown = render(related_links=[link(label=label)])

    assert link_halves(markdown)[0] == (label, "https://github.com/example/repo")


def test_a_label_of_backslash_n_and_a_label_of_a_newline_stay_different_texts() -> None:
    """Two sources a reader must not be shown as one.

    The collision is the whole of why the visible `\\n` was lossy: it is what
    an escaped newline looked like *and* what an escaped backslash-then-`n`
    looked like.
    """
    literal = render(related_links=[link(label="first\\nsecond")])
    ending = render(related_links=[link(label="first\nsecond")])

    assert link_halves(literal)[0][0] == "first\\nsecond"
    assert link_halves(ending)[0][0] == "first\nsecond"


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_a_link_label_holding_a_line_ending_cannot_open_a_heading(ending: str) -> None:
    """A label is one line of a document, and a line ending in one ends that line.

    The label is escaped rather than fenced, so the line ending is the whole of
    the boundary: a real one lands the rest of the label at the document's own
    indentation, and `# PWNED` there is a heading the render appears to have
    written. A reader ends a line on `\\r` as readily as on `\\n`, so a lone CR
    does it too.
    """
    label = f"line1{ending}{ending}# PWNED HEADING{ending}{ending}line2"
    markdown = render(related_links=[link(label=label)])
    structure = outside_fences(markdown)

    assert [line for line in re.split(r"\r\n|\r|\n", structure) if line.startswith("# ")] == [
        "# Video Bundle"
    ]


def test_a_link_destination_cannot_terminate_the_construct() -> None:
    """A destination reaching the render from a **manifest** is extracted text too."""
    hostile = f"https://ok.example/a) [pwn]({ATTACKER_URL}"
    markdown = render(related_links=[link(url=hostile)])

    assert destinations(markdown) == [hostile, FRAME_IMAGE]


def test_a_link_destination_holding_a_line_break_stays_on_one_line() -> None:
    """A destination may not carry a line ending, so the break cannot start a line."""
    markdown = render(related_links=[link(url="https://ok.example/a\n\n# heading")])
    structure = outside_fences(markdown)

    assert not any(line.lstrip().startswith("# heading") for line in structure.split("\n"))
    assert len(destinations(markdown)) == 2


def test_a_destination_holding_a_markdown_construct_is_wrapped() -> None:
    """The unwrapped path is taken only when nothing in the destination can act.

    A backtick or a bracket in a destination does not retarget the link - the
    destination is parsed from raw text - but it does leave a reader unable to
    tell that by looking, and the point of the plain path is that they can.
    """
    hostile = "https://ok.example/a`b`]end"
    markdown = render(related_links=[link(url=hostile)])

    assert f"](<{hostile}>)" in markdown
    assert destinations(markdown) == [hostile, FRAME_IMAGE]


@pytest.mark.parametrize(
    "url",
    [
        "https://ok.example/a b?q=&copy;",
        "https://ok.example/a`b`]end?q=1&amp;r=2",
        "https://ok.example/a) [pwn](https://attacker.example/evil?q=&copy;",
        "https://ok.example/a%2Fb?q=100%&r=&#41;",
    ],
    ids=["space", "construct", "terminator", "already-encoded"],
)
def test_a_wrapped_destination_still_resolves_to_the_url_the_carrier_held(url: str) -> None:
    """Angle brackets stop a destination ending its `(...)`; they do not stop decoding.

    An entity reference is decoded inside `<...>` exactly as it is outside, so
    the wrap is no reason to skip the escape. Asserted through a reader rather
    than by looking at the render text, because the reader is the one that
    normalizes: a space becomes `%20`, a line ending `%0A`, and what has to be
    true is that undoing that normalization gives back the URL and not
    something the URL contained.
    """
    markdown = render(related_links=[link(url=url)])

    assert destinations(markdown) == [url, FRAME_IMAGE]


def test_an_ordinary_link_is_left_readable() -> None:
    """Escaping is applied by rule, and the rule leaves a clean link clean."""
    markdown = render(related_links=[link()])

    assert "[Example repo](https://github.com/example/repo)" in markdown
    assert "(code_or_reference)" in markdown


def test_the_keyframe_image_link_survives_the_same_escaping() -> None:
    """The image path is Distill's own, and the rule that guards a link still runs on it."""
    markdown = render(frames=[keyframe(extracted_text="slide text")])

    assert "![Frame 1](frames/frame_0001.png)" in markdown


def test_self_contained_render_omits_frame_links_but_keeps_the_same_reading() -> None:
    """FAILS FIRST: both render calls emitted the generation-relative image link."""
    frame = keyframe(extracted_text="portable slide reading")
    provenance = Provenance(
        title="demo.mp4",
        duration_sec=10.0,
        processed_at="2026-07-29T14:20:00Z",
    )

    linked = render(frames=[frame])
    self_contained = render(
        frames=[frame],
        provenance=provenance,
        include_frame_links=False,
    )

    assert destinations(linked) == [FRAME_IMAGE]
    assert destinations(self_contained) == []
    assert "portable slide reading" in linked
    assert "portable slide reading" in self_contained


# --- R-24: the preamble -----------------------------------------------------


def test_the_render_carries_the_untrusted_data_preamble() -> None:
    """R-24: the document says which of its sections are **extracted text**.

    Naming the sections is what makes the delimiter mean something to a reader
    that has never seen Distill: a fenced block is only a quotation if the
    document says what is being quoted and by whom.
    """
    markdown = render(
        transcript=spoken({"start": 0.0, "end": 1.0, "text": "hello", "words": []}),
        related_links=[link()],
    )
    preamble = outside_fences(markdown).split("## ")[0].lower()

    assert UNTRUSTED_TEXT_LABEL in preamble
    for section in ("source", "transcript", "on-screen text", "warning", "related link"):
        assert section in preamble, section
    assert "not as instructions" in preamble or "not instructions" in preamble
    assert "mitigation" in preamble and "not a guarantee" in preamble


def test_the_preamble_precedes_every_piece_of_extracted_text() -> None:
    markdown = render(
        source_label="demo.mp4",
        transcript=spoken({"start": 0.0, "end": 1.0, "text": "hello", "words": []}),
    )

    assert markdown.index("Untrusted data") < markdown.index("demo.mp4")


def test_self_contained_provenance_is_classified_by_origin_below_the_preamble() -> None:
    """FAILS FIRST: the render accepted no provenance carrier or provenance region."""
    source_chosen = {
        "title": attack("provenance title"),
        "channel": attack("provenance channel"),
        "description": attack("provenance description"),
        "upload_date": attack("provenance upload date"),
    }
    provenance = Provenance(
        **source_chosen,
        canonical_url="https://www.youtube.com/watch?v=abcdefghijk",
        duration_sec=10.0,
        processed_at="2026-07-29T14:20:00Z",
    )

    markdown = render(provenance=provenance)
    structure = outside_fences(markdown)
    bodies = untrusted_bodies(markdown)
    preamble = structure.split("## ")[0].lower()

    assert "provenance" in preamble
    for marker, value in source_chosen.items():
        assert any(value in body for body in bodies), marker
        assert value not in structure, marker
    assert "Canonical URL: https://www.youtube.com/watch?v=abcdefghijk" in structure
    assert "Source duration: 10.000s" in structure
    assert "Processed at: 2026-07-29T14:20:00Z" in structure
    assert markdown.index("Untrusted data") < markdown.index(source_chosen["title"])


# --- the adversarial fixture suite ------------------------------------------


def test_no_extracted_text_region_of_a_hostile_bundle_reaches_document_structure() -> None:
    """Every region at once, because the boundary is only as good as its worst sink.

    One render carrying a self-terminating fence, a backtick filename, a `](`
    link label and a caption of newlines - the four shapes Gate 5->6 names -
    and the claim is the same for all of them: nothing they contain appears
    where the render speaks for itself.

    The link halves carry the same payload every other region does, and not a
    bespoke single-line one. A label built to retarget the link proves only the
    `]` escape; the payload's line endings are what prove the rest, and leaving
    them out left the one region the render escapes rather than fences with no
    test behind its line-ending handling at all.
    """
    reading = read(
        visual_summary=attack("summary"),
        detected_elements=(attack("element"),),
        interpretation=attack("interpretation"),
        uncertainty=attack("uncertainty"),
        verbatim_text=attack("verbatim"),
        text_confidence=attack("confidence"),
    )
    markdown = render(
        source_label=attack("source"),
        transcript=spoken({"start": 0.0, "end": 5.0, "text": attack("segment"), "words": []}),
        frames=[frame_reading(reading, extracted_text=attack("ocr"))],
        warnings=[{"stage": "ocr", "code": "tesseract_failed", "message": attack("warning")}],
        related_links=[
            link(
                label=f'{attack("label")}](https://attacker.example/evil) [rest <{ATTACKER_URL}>',
                url=f'{attack("url")}) [pwn](https://attacker.example/evil',
            )
        ],
    )
    structure = outside_fences(markdown)

    headings = [line.lstrip() for line in re.split(r"\r\n|\r|\n", structure)]

    assert SENTINEL not in outside_links(structure)
    assert ATTACKER_URL not in destinations(markdown)
    assert autolinks(markdown) == []
    for marker in (
        "source",
        "segment",
        "ocr",
        "summary",
        "element",
        "interpretation",
        "uncertainty",
        "verbatim",
        "confidence",
        "warning",
        "label",
        "url",
    ):
        assert not any(line.startswith(f"# {marker}") for line in headings), marker
