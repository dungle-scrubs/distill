"""Deterministic grounding checks for local vision interpretations.

This module owns the cross-check between OCR text and a vision model's
transcription, so a confident interpretation that no readable text supports can
be flagged instead of silently trusted. It owns the **grounding** levels and
which of them is not low confidence. It does not call a model, read images, or
render markdown, and it does not decide what a marked frame looks like to a
reader - that is `render.py`'s.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

CORROBORATED = "corroborated"
"""Two readers independently recovered the same text from the **keyframe**.

The only level that is not low confidence, because it is the only one holding
evidence a second reader produced. It replaces `grounded`, which named this and
a lone reader's own say-so with one word (R-42) - and a level a reader cannot
distinguish is a level that vouches for the wrong thing.
"""

SELF_REPORT = "self_report"
"""One reader recovered the text and nothing else confirms it.

Distinct from `WEAK` rather than folded into it: the reading is substantive and
the model was confident, so the frame is worth reporting - it is the
*corroboration* that is missing, not the reading. Distinct from `CORROBORATED`
because a model grading its own homework is not a second opinion, however
confident the grade (D-007). Low confidence, so the render marks it.
"""

WEAK = "weak"
"""The readers were asked and what came back does not support the reading.

Partial or no token agreement, a lone reading too thin or too unsure to report
as one, or an interpretation over text neither reader recovered.
"""

UNGROUNDED = "ungrounded"
"""An interpretation exists and no reader recovered text that could support it."""

CORROBORATED_LEVELS = frozenset({CORROBORATED})
"""The levels that mean a second reader recovered the same text.

A set of one, and stated as a set anyway: `is_corroborated` is a question
about the level rather than an inequality against a particular string, so a
level added later is a decision made here instead of an operator quietly
inverting it somewhere else.
"""

# Overlap of vision-transcribed tokens that also appear in OCR. Above STRONG the
# two readers corroborate each other; below WEAK they effectively disagree.
STRONG_OVERLAP = 0.5
WEAK_OVERLAP = 0.2

# What makes a lone vision transcription substantive enough to report as
# self-report rather than dismiss as weak. It is not what makes it trustworthy:
# no length and no confidence turns one reader into two. OCR returns nothing on
# plenty of legible slides (dark backgrounds especially), so a thin or unsure
# reading and a full one are still worth telling apart.
MIN_SELF_REPORT_TOKENS = 4
SELF_REPORT_CONFIDENCE = frozenset({"high", "medium"})

_TOKEN = re.compile(r"[a-z0-9]{2,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@dataclass(frozen=True)
class GroundingAssessment:
    """How well a vision interpretation is supported by readable text."""

    level: str
    text_overlap: float | None
    reason: str

    @property
    def is_corroborated(self) -> bool:
        """Whether a second reader recovered the same text (D-002).

        A fact about agreement, not a verdict on the reading. Since the vision
        model became the authoritative reader, an uncorroborated frame is not
        a doubted one - it is a frame the image-text reader did not confirm,
        which is what the render says in plain words.
        """
        return self.level in CORROBORATED_LEVELS

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_document(cls, document: Mapping[str, Any] | None) -> GroundingAssessment | None:
        """Rebuild an assessment from the mapping a **frame artifact** carries.

        A **frame artifact** holds its grounding as a mapping, because that is
        what the carrier freezes; this is how a reader gets the assessment back
        without restating these three field names somewhere else. `None` in and
        `None` out: a frame the vision pass never reached has no assessment, and
        inventing a corroborated one for it would say two readers agreed when
        neither was asked.

        A `level` this module does not define is passed through rather than
        refused. `is_corroborated` is stated against `CORROBORATED_LEVELS`, so
        an unrecognized level reads as uncorroborated, which is the answer that
        does not claim agreement nobody established.
        """
        if document is None:
            return None
        overlap = document.get("text_overlap")
        return cls(
            level=str(document.get("level", "")),
            text_overlap=None if overlap is None else float(overlap),
            reason=str(document.get("reason", "")),
        )


def assess_grounding(
    *,
    ocr_text: str,
    verbatim_text: str,
    text_confidence: str,
    has_interpretation: bool,
    carries_a_reading: bool,
) -> GroundingAssessment:
    """Compare OCR and vision-transcribed text to grade interpretation support.

    The dangerous case this targets: the model emits a confident interpretation
    while no readable on-screen text (OCR or its own transcription) backs it —
    the "plausible fiction on an unreadable slide" failure. That returns
    ``UNGROUNDED``. Genuine textless frames (e.g. a photo) with no interpretation
    are nothing to flag; partial or one-sided agreement is ``WEAK``.

    Only agreement between the two readers returns ``CORROBORATED``. A lone
    reading, substantive and confident, returns ``SELF_REPORT``: this function
    grades support, and the support for a lone reading is the reading (R-42).

    Three signals about the reading, not one, because "was anything said about
    this frame" and "was anything said *beyond* transcribing it" are different
    questions and only one of them was being asked. `verbatim_text` is what the
    model transcribed, `has_interpretation` whether it described the frame on
    top of that, and `carries_a_reading` whether it said anything at all - the
    last is what tells a genuinely textless frame apart from one the model
    described without being able to read it, and grading those the same was how
    a description of an unreadable slide came back corroborated.
    """
    ocr_tokens = _tokens(ocr_text)
    vision_tokens = _tokens(verbatim_text)

    if vision_tokens and ocr_tokens:
        overlap = len(vision_tokens & ocr_tokens) / len(vision_tokens)
        if overlap >= STRONG_OVERLAP:
            return GroundingAssessment(
                CORROBORATED, overlap, "OCR corroborates the transcribed text"
            )
        if overlap >= WEAK_OVERLAP:
            return GroundingAssessment(
                WEAK, overlap, "OCR only partially corroborates the transcribed text"
            )
        return GroundingAssessment(
            WEAK, overlap, "transcribed text and OCR disagree on most tokens"
        )

    if vision_tokens and not ocr_tokens:
        # OCR returned nothing (common on dark slides), so whatever comes back
        # here has one reader behind it. A substantive, confident transcription
        # is reported as what it is - the model's own account of the slide -
        # rather than promoted for being emphatic; a thin or unsure one is not
        # even that.
        if (
            len(vision_tokens) >= MIN_SELF_REPORT_TOKENS
            and text_confidence in SELF_REPORT_CONFIDENCE
        ):
            return GroundingAssessment(
                SELF_REPORT,
                None,
                "vision model's own transcription only; OCR returned nothing to corroborate it",
            )
        return GroundingAssessment(
            WEAK, None, "vision transcribed text that OCR did not corroborate"
        )

    if ocr_tokens and not vision_tokens:
        return GroundingAssessment(WEAK, None, "OCR found text the vision model did not transcribe")

    # Neither reader recovered any text from the frame.
    if has_interpretation and text_confidence in {"none", "low"}:
        return GroundingAssessment(
            UNGROUNDED, None, "interpretation present but no readable on-screen text supports it"
        )
    if has_interpretation:
        return GroundingAssessment(
            WEAK, None, "no readable text available to corroborate the interpretation"
        )
    if carries_a_reading:
        # Nothing was described and no token survived tokenizing, yet the model
        # did put something in a substantive field. One reader spoke and
        # nothing supports it, which is what SELF_REPORT names.
        return GroundingAssessment(
            SELF_REPORT,
            None,
            "vision model's own reading only; OCR returned nothing to corroborate it",
        )
    # Both readers were asked, both recovered nothing, and nothing claims
    # otherwise: a photo, a face, a blank slide. That is the two readers
    # agreeing on what the frame holds, and there is no interpretation over it
    # for a marker to warn about - so it is not low confidence.
    #
    # Reserved for that case alone. A model that described the frame while
    # reading no text off it has claimed something, and this sentence would be
    # false about the frame as well as unmarked - the level no render marks
    # given to the reading most in need of marking.
    return GroundingAssessment(CORROBORATED, None, "no on-screen text present")
