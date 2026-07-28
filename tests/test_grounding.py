from __future__ import annotations

from distill.grounding import CORROBORATED, SELF_REPORT, UNGROUNDED, WEAK, assess_grounding


def test_matching_ocr_and_vision_text_is_corroborated() -> None:
    assessment = assess_grounding(
        ocr_text="We're closer than you think state machines durable execution",
        verbatim_text="We're closer than you think state machines durable execution gates",
        text_confidence="high",
        has_interpretation=True,
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
    )

    assert assessment.level == UNGROUNDED
    assert assessment.is_low_confidence is True


def test_a_textless_frame_without_an_interpretation_is_not_marked() -> None:
    # Both readers were asked and both recovered nothing, which is agreement,
    # and no interpretation was made over it - there is no claim here to mark.
    assessment = assess_grounding(
        ocr_text="",
        verbatim_text="",
        text_confidence="none",
        has_interpretation=False,
    )

    assert assessment.level == CORROBORATED
    assert assessment.is_low_confidence is False


def test_disagreeing_readers_are_weak() -> None:
    assessment = assess_grounding(
        ocr_text="alpha beta gamma delta",
        verbatim_text="generative machine learning models",
        text_confidence="high",
        has_interpretation=True,
    )

    assert assessment.level == WEAK
    assert assessment.text_overlap == 0.0


def test_short_vision_text_uncorroborated_by_empty_ocr_is_weak() -> None:
    assessment = assess_grounding(
        ocr_text="",
        verbatim_text="some heading",  # too short to trust on its own
        text_confidence="medium",
        has_interpretation=True,
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
    )

    lone_reader = assess_grounding(
        ocr_text="",
        verbatim_text="What a software factory needs Agent Runtimes Orchestration",
        text_confidence="high",
        has_interpretation=True,
    )

    assert corroborated.level == CORROBORATED
    assert corroborated.is_low_confidence is False
    assert lone_reader.level != corroborated.level
    assert lone_reader.level == SELF_REPORT
    assert lone_reader.is_low_confidence is True
