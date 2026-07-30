from __future__ import annotations

from distill.artifacts import Interpretation
from distill.grounding import CORROBORATED, SELF_REPORT, UNGROUNDED, WEAK, assess_grounding


def _graded(*, ocr_text: str, reading: Interpretation):
    """The assessment the pipeline makes for a reading, asked the way it asks.

    The same four values `local_vision._interpreted` passes, read off the
    reading rather than written out here: what these tests are about is which
    readings get which level, and a hand-written `has_interpretation` makes
    that a question about the boolean the test chose.
    """
    return assess_grounding(
        ocr_text=ocr_text,
        verbatim_text=reading.verbatim_text,
        text_confidence=reading.text_confidence,
        has_interpretation=reading.has_interpretation,
        carries_a_reading=reading.carries_a_reading,
    )


def test_matching_ocr_and_vision_text_is_corroborated() -> None:
    assessment = assess_grounding(
        ocr_text="We're closer than you think state machines durable execution",
        verbatim_text="We're closer than you think state machines durable execution gates",
        text_confidence="high",
        has_interpretation=True,
        carries_a_reading=True,
    )

    assert assessment.level == CORROBORATED
    assert assessment.text_overlap is not None and assessment.text_overlap >= 0.5


def test_confident_interpretation_with_no_readable_text_is_ungrounded() -> None:
    # The frame-16 failure: the model emitted a full interpretation while neither
    # OCR nor its own transcription recovered any on-screen text.
    assessment = assess_grounding(
        ocr_text="",
        verbatim_text="",
        text_confidence="none",
        has_interpretation=True,
        carries_a_reading=True,
    )

    assert assessment.level == UNGROUNDED
    assert assessment.is_corroborated is False


def test_a_textless_frame_without_an_interpretation_is_not_marked() -> None:
    # Both readers were asked and both recovered nothing, which is agreement,
    # and no interpretation was made over it - there is no claim here to mark.
    # This is the one case the no-on-screen-text reason is reserved for, so the
    # reason is asserted too: a frame the model described is not this frame.
    assessment = assess_grounding(
        ocr_text="",
        verbatim_text="",
        text_confidence="none",
        has_interpretation=False,
        carries_a_reading=False,
    )

    assert assessment.level == CORROBORATED
    assert assessment.is_corroborated is True
    assert assessment.reason == "no on-screen text present"


def test_disagreeing_readers_are_weak() -> None:
    assessment = assess_grounding(
        ocr_text="alpha beta gamma delta",
        verbatim_text="generative machine learning models",
        text_confidence="high",
        has_interpretation=True,
        carries_a_reading=True,
    )

    assert assessment.level == WEAK
    assert assessment.text_overlap == 0.0


def test_short_vision_text_uncorroborated_by_empty_ocr_is_weak() -> None:
    assessment = assess_grounding(
        ocr_text="",
        verbatim_text="some heading",  # too short to trust on its own
        text_confidence="medium",
        has_interpretation=True,
        carries_a_reading=True,
    )

    assert assessment.level == WEAK
    assert assessment.text_overlap is None


def test_a_confident_lone_reader_is_not_given_the_level_two_agreeing_readers_get() -> None:
    """R-42: one reader, however confident, has corroborated nothing.

    The model read a dark slide well and OCR (which struggles on dark
    backgrounds) returned nothing. That reading may well be right, and it is
    still the model's own word for the model's own work: the only evidence that
    the text is there is the sentence claiming it is. Stated against the level a
    genuinely corroborated frame receives rather than against a literal, because
    the claim is that these two situations are not the same answer - a rename of
    the level would not make them the same.
    """
    corroborated = assess_grounding(
        ocr_text="What a software factory needs Agent Runtimes Orchestration",
        verbatim_text="What a software factory needs Agent Runtimes Orchestration",
        text_confidence="high",
        has_interpretation=True,
        carries_a_reading=True,
    )

    lone_reader = assess_grounding(
        ocr_text="",
        verbatim_text="What a software factory needs Agent Runtimes Orchestration",
        text_confidence="high",
        has_interpretation=True,
        carries_a_reading=True,
    )

    assert corroborated.level == CORROBORATED
    assert corroborated.is_corroborated is True
    assert lone_reader.level != corroborated.level
    assert lone_reader.level == SELF_REPORT
    assert lone_reader.is_corroborated is False


def test_a_lone_readers_description_of_an_unreadable_frame_is_marked() -> None:
    """The shape the prompt itself asks for when the text cannot be read.

    Told to leave `verbatim_text` empty and set `text_confidence` to none when
    a frame is unreadable, a model that describes the slide anyway comes back
    with nothing but `visual_summary`. Neither reader recovered a word, and the
    frame is emphatically not textless - a claim was made about it - so the one
    answer this must not give is the unmarked one that says no text is present.
    """
    reading = Interpretation(
        visual_summary="A slide titled 'Q3 revenue: $4.2M, up 340% YoY' with a bar chart",
        text_confidence="none",
        backend="rapid-mlx",
        model="m",
    )

    assessment = _graded(ocr_text="", reading=reading)

    assert reading.carries_a_reading is True
    assert assessment.is_corroborated is False
    assert assessment.level != CORROBORATED
    assert "no on-screen text present" not in assessment.reason


def test_a_reading_admitting_it_cannot_read_the_frame_is_not_called_textless() -> None:
    """A reading whose own `uncertainty` says the frame is unreadable.

    The metadata fields describe a reading rather than being one, so this
    payload's substance is its `visual_summary` alone - the same shape as
    above, arriving with the model's own admission attached. The reason must
    not tell an operator no text is present: the model just said it could not
    read the text that is.
    """
    reading = Interpretation(
        visual_summary="Chart of revenue by quarter",
        uncertainty="cannot read the axis labels",
        text_confidence="none",
        backend="rapid-mlx",
        model="m",
    )

    assessment = _graded(ocr_text="", reading=reading)

    assert assessment.is_corroborated is False
    assert "no on-screen text present" not in assessment.reason


def test_a_reading_is_only_a_description_when_visual_summary_carries_it() -> None:
    """`has_interpretation` and `carries_a_reading` cannot answer differently.

    Both are "is there a reading here" asked of the same object, and they
    disagreed: `has_interpretation` looked at two fields where
    `SUBSTANTIVE_FIELDS` names four. A description of the frame is something
    the model said about it, whichever field it landed in.
    """
    summary_only = Interpretation(visual_summary="A dark slide with a bar chart")
    elements_only = Interpretation(detected_elements=("axis", "legend"))
    blank_elements = Interpretation(detected_elements=("", "  "))
    transcription_only = Interpretation(verbatim_text="Q3 revenue")

    assert summary_only.has_interpretation is True
    assert elements_only.has_interpretation is True
    # Whitespace is as empty as absent, the way a reading judges it.
    assert blank_elements.has_interpretation is False
    assert blank_elements.carries_a_reading is False
    # Transcribing text is reading the frame, not describing it - and it is
    # already the signal `verbatim_text` carries into the assessment.
    assert transcription_only.has_interpretation is False
    assert transcription_only.carries_a_reading is True


def test_a_transcription_that_tokenizes_to_nothing_is_still_something_the_model_said() -> None:
    """The other route to a reading no token survives: symbols, not words.

    A frame showing `€ $ ¥` transcribes to text the tokenizer discards, so both
    readers come back empty and nothing was described on top of it - the same
    inputs as a blank slide, from a model that did read something. It is one
    reader's word and nothing else, which is what SELF_REPORT names, and it is
    not the frame the no-on-screen-text reason is reserved for.
    """
    reading = Interpretation(verbatim_text="€ $ ¥", text_confidence="low")

    assessment = _graded(ocr_text="", reading=reading)

    assert reading.carries_a_reading is True
    assert reading.has_interpretation is False
    assert assessment.level == SELF_REPORT
    assert assessment.is_corroborated is False


def test_a_reading_that_said_nothing_at_all_still_reaches_the_unmarked_answer() -> None:
    """The over-correction guard: an empty reading is still a textless frame.

    `test_a_textless_frame_without_an_interpretation_is_not_marked` states the
    same claim against the booleans; this states it from the reading they are
    read off, so "mark whatever has no verbatim text" cannot pass.
    """
    reading = Interpretation(text_confidence="none")

    assert reading.carries_a_reading is False
    assert _graded(ocr_text="", reading=reading).level == CORROBORATED


def test_a_reading_rebuilt_from_a_drifted_document_shows_no_elements_it_invented() -> None:
    """R-39's shape rule holds on the way back in, not only on the way in.

    A **resume** and a **cache** hit rebuild a reading from a mapping that was
    written by some other run, so `detected_elements` can arrive as anything.
    `document_carries_a_reading` refuses a string there - it is not the shape
    the field is declared with - while `from_document` iterated it into
    characters, so a frame that did NOT count as a reading could still DISPLAY
    `a`, `x`, `i`, `s` as four elements the model never detected.
    """
    from distill.artifacts import document_carries_a_reading

    for wrong_shape in ("axis", 42, {"label": "axis"}, None):
        document = {"detected_elements": wrong_shape}

        assert document_carries_a_reading(document) is False
        rebuilt = Interpretation.from_document(document)
        assert rebuilt is not None
        assert rebuilt.detected_elements == ()
        assert rebuilt.carries_a_reading is False

    # A well-shaped list keeps its strings and drops only what is not one.
    mixed = Interpretation.from_document({"detected_elements": ["axis", 7, {"a": 1}, "legend"]})
    assert mixed is not None
    assert mixed.detected_elements == ("axis", "legend")


def test_grounding_no_longer_carries_a_confidence_verdict() -> None:
    """M3.1 (D-002): the vision reading is the authoritative reader, so
    grounding records whether a second reader agreed - it does not grade
    confidence. The `is_low_confidence` verdict is gone; `is_corroborated`
    states the fact instead, and `corroborated` keeps its strict two-readers
    meaning."""
    corroborated = assess_grounding(
        ocr_text="Deploy pipeline overview",
        verbatim_text="Deploy pipeline overview",
        text_confidence="high",
        has_interpretation=True,
        carries_a_reading=True,
    )
    lone_reader = assess_grounding(
        ocr_text="",
        verbatim_text="Deploy pipeline overview",
        text_confidence="high",
        has_interpretation=True,
        carries_a_reading=True,
    )

    assert corroborated.level == CORROBORATED
    assert corroborated.is_corroborated is True
    assert lone_reader.is_corroborated is False
    # The vocabulary that graded a reading is gone, not merely unused.
    assert not hasattr(corroborated, "is_low_confidence")


def test_the_render_note_is_neutral_and_collapses_the_uncorroborated_levels() -> None:
    """M3.2 (D-002/D-015): the three non-corroborated levels collapse to one
    neutral note. No warning framing, no low-confidence verdict, and the
    corroborated note claims agreement - not independence, since the vision
    reader is shown the image-text reader's output."""
    from pathlib import Path

    from distill.artifacts import FrameArtifact, RedactionState
    from distill.render import render_markdown

    def render_at(level: str) -> str:
        frame = FrameArtifact(
            index=1,
            timestamp_sec=1.0,
            path=str(Path("frames/frame1.png")),
            relative_path="frames/frame1.png",
            extracted_text="Deploy pipeline overview",
            redaction=RedactionState.APPLIED,
        )
        carried, _w = frame.with_interpretation(
            Interpretation(
                visual_summary="A deployment diagram",
                verbatim_text="Deploy pipeline overview",
                text_confidence="high",
            ),
            grounding={"level": level, "reason": "reason text"},
        )
        return render_markdown("demo.mp4", 10.0, None, [carried], [])

    corroborated = render_at(CORROBORATED)
    assert "matches the on-screen-text reader" in corroborated
    assert "Low-confidence" not in corroborated
    assert "⚠" not in corroborated

    notes = set()
    for level in (SELF_REPORT, WEAK, UNGROUNDED):
        rendered = render_at(level)
        assert "Low-confidence" not in rendered
        assert "⚠" not in rendered
        assert "unverified" not in rendered.lower()
        notes.add(next(line for line in rendered.split("\n") if "on-screen-text reader" in line))

    # One note for all three, not three differently-worded ones.
    assert len(notes) == 1
    assert "did not" in notes.pop()
