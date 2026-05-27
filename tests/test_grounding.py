from __future__ import annotations

from saccade.grounding import GROUNDED, UNGROUNDED, WEAK, assess_grounding


def test_matching_ocr_and_vision_text_is_grounded() -> None:
    assessment = assess_grounding(
        ocr_text="We're closer than you think state machines durable execution",
        verbatim_text="We're closer than you think state machines durable execution gates",
        text_confidence="high",
        has_interpretation=True,
    )

    assert assessment.level == GROUNDED
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


def test_textless_frame_without_interpretation_stays_grounded() -> None:
    assessment = assess_grounding(
        ocr_text="",
        verbatim_text="",
        text_confidence="none",
        has_interpretation=False,
    )

    assert assessment.level == GROUNDED
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


def test_confident_substantive_vision_with_empty_ocr_is_grounded() -> None:
    # The false-positive fix: the model read a dark slide well, but OCR (which
    # struggles on dark backgrounds) returned nothing. Trust the model.
    assessment = assess_grounding(
        ocr_text="",
        verbatim_text="What a software factory needs Agent Runtimes Orchestration",
        text_confidence="high",
        has_interpretation=True,
    )

    assert assessment.level == GROUNDED
