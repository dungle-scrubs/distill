"""Deterministic grounding checks for local vision interpretations.

This module owns the cross-check between OCR text and a vision model's
transcription, so a confident interpretation that no readable text supports can
be flagged instead of silently trusted. It does not call a model, read images,
or render markdown.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

GROUNDED = "grounded"
WEAK = "weak"
UNGROUNDED = "ungrounded"

# Overlap of vision-transcribed tokens that also appear in OCR. Above STRONG the
# two readers corroborate each other; below WEAK they effectively disagree.
STRONG_OVERLAP = 0.5
WEAK_OVERLAP = 0.2

# When OCR gives no corroboration, a vision transcription this long and this
# confident is trusted as grounded rather than flagged — otherwise the model gets
# penalized for reading slides that OCR (e.g. on dark backgrounds) returns nothing for.
MIN_TRUSTED_VISION_TOKENS = 4
TRUSTED_CONFIDENCE = frozenset({"high", "medium"})

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
    def is_low_confidence(self) -> bool:
        return self.level != GROUNDED

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_grounding(
    *,
    ocr_text: str,
    verbatim_text: str,
    text_confidence: str,
    has_interpretation: bool,
) -> GroundingAssessment:
    """Compare OCR and vision-transcribed text to grade interpretation support.

    The dangerous case this targets: the model emits a confident interpretation
    while no readable on-screen text (OCR or its own transcription) backs it —
    the "plausible fiction on an unreadable slide" failure. That returns
    ``UNGROUNDED``. Genuine textless frames (e.g. a photo) with no interpretation
    stay ``GROUNDED``; partial or one-sided agreement is ``WEAK``.
    """
    ocr_tokens = _tokens(ocr_text)
    vision_tokens = _tokens(verbatim_text)

    if vision_tokens and ocr_tokens:
        overlap = len(vision_tokens & ocr_tokens) / len(vision_tokens)
        if overlap >= STRONG_OVERLAP:
            return GroundingAssessment(GROUNDED, overlap, "OCR corroborates the transcribed text")
        if overlap >= WEAK_OVERLAP:
            return GroundingAssessment(
                WEAK, overlap, "OCR only partially corroborates the transcribed text"
            )
        return GroundingAssessment(
            WEAK, overlap, "transcribed text and OCR disagree on most tokens"
        )

    if vision_tokens and not ocr_tokens:
        # OCR returned nothing (common on dark slides). Trust a substantive,
        # confident transcription instead of flagging the model for reading what
        # OCR missed; flag only thin or low-confidence transcriptions.
        if (
            len(vision_tokens) >= MIN_TRUSTED_VISION_TOKENS
            and text_confidence in TRUSTED_CONFIDENCE
        ):
            return GroundingAssessment(
                GROUNDED, None, "confident substantive transcription; OCR returned nothing"
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
    return GroundingAssessment(GROUNDED, None, "no on-screen text present")
