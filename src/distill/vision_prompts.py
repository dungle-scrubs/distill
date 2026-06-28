"""Prompt construction for local visual frame interpretation.

This module owns deterministic prompt text only. It does not call a model,
inspect images, or mutate frame metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

TECHNICAL_PROMPT_PROFILE = "technical"
FRAME_KINDS = (
    "ui_interface",
    "chart_graph",
    "diagram",
    "code_screenshot",
    "terminal",
    "slide",
)

TEXT_CONFIDENCE_LEVELS = ("high", "medium", "low", "none")

KIND_FOCUS = {
    "ui_interface": "UI state, selected controls, layout hierarchy, visible errors, and user workflow implications.",
    "chart_graph": "axes, legends, series, trends, outliers, comparisons, and what the chart is communicating.",
    "diagram": "nodes, arrows, grouping, sequence, dependencies, and the system or process relationship shown.",
    "code_screenshot": "language or framework clues, code structure, errors, diffs, identifiers, and likely intent.",
    "terminal": "commands, prompts, statuses, errors, progress, file paths, and operational state.",
    "slide": "title, main claim, supporting visual evidence, emphasized terms, and presentation context.",
}

GENERIC_FOCUS = "the title, main claim, emphasized terms, and what the frame is communicating."


@dataclass(frozen=True)
class VisionPrompt:
    profile: str
    frame_kind: str
    prompt: str


def build_technical_frame_prompt(
    frame_kind: str | None = None,
    *,
    ocr_text: str | None = None,
) -> VisionPrompt:
    """Build a grounding-first interpretation prompt.

    When ``frame_kind`` is omitted or unknown the model is asked to classify the
    frame itself, so callers need not guess a kind up front. The prompt forbids
    inventing content that is not visibly present and requires legible on-screen
    text to be transcribed separately from interpretation, so an unreadable frame
    yields an empty transcription rather than plausible fiction.
    """
    resolved_kind = frame_kind if frame_kind in KIND_FOCUS else None
    kinds = ", ".join(FRAME_KINDS)
    parts = [
        "Interpret this technical video frame for a developer or analyst.",
        f"First classify the frame as one of: {kinds}.",
    ]
    if resolved_kind is not None:
        parts.append(
            f"This frame is most likely a {resolved_kind}; focus on {KIND_FOCUS[resolved_kind]}"
        )
    else:
        parts.append(f"Focus on {GENERIC_FOCUS}")
    parts.extend(
        [
            "Transcribe only text you can actually read into verbatim_text. If the frame is too "
            "low-resolution, low-contrast, blurry, or cropped to read, leave verbatim_text empty "
            'and set text_confidence to "none".',
            "Transcribe the slide or screen content only. Ignore recurring presentation chrome: "
            "logos, sponsor or venue banners, watermarks, speaker-camera insets, and page numbers.",
            "Do not invent, infer, or guess any text, topic, technology, label, metric, or UI "
            "element that is not visibly present. Returning little is correct when the frame is unreadable.",
            "Base detected_elements and interpretation strictly on what is visible; never add "
            "plausible-sounding domain content that you cannot actually see.",
            "Beyond transcription, explain the visual meaning, relationships, and state that the "
            "image genuinely supports.",
            "If the image is unclear, low-confidence, or ambiguous, state that explicitly in "
            "uncertainty and lower text_confidence accordingly.",
        ]
    )
    if ocr_text:
        parts.append(
            "OCR text extracted from this frame is provided as supporting evidence; prefer it "
            "where on-screen text is hard to read, but ignore OCR that conflicts with what you see."
        )
        parts.append(f"OCR context: {ocr_text[:1200]}")
    parts.append(
        "Return compact JSON with: frame_kind (one of the listed kinds), verbatim_text (legible "
        "on-screen text only), text_confidence (one of high, medium, low, none), visual_summary, "
        "detected_elements (array of strings), interpretation, and uncertainty."
    )
    return VisionPrompt(
        profile=TECHNICAL_PROMPT_PROFILE,
        frame_kind=resolved_kind or "auto",
        prompt="\n".join(parts),
    )
