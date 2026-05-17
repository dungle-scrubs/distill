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

KIND_FOCUS = {
    "ui_interface": "UI state, selected controls, layout hierarchy, visible errors, and user workflow implications.",
    "chart_graph": "axes, legends, series, trends, outliers, comparisons, and what the chart is communicating.",
    "diagram": "nodes, arrows, grouping, sequence, dependencies, and the system or process relationship shown.",
    "code_screenshot": "language or framework clues, code structure, errors, diffs, identifiers, and likely intent.",
    "terminal": "commands, prompts, statuses, errors, progress, file paths, and operational state.",
    "slide": "title, main claim, supporting visual evidence, emphasized terms, and presentation context.",
}


@dataclass(frozen=True)
class VisionPrompt:
    profile: str
    frame_kind: str
    prompt: str


def build_technical_frame_prompt(
    frame_kind: str,
    *,
    ocr_text: str | None = None,
) -> VisionPrompt:
    if frame_kind not in KIND_FOCUS:
        frame_kind = "ui_interface"
    parts = [
        "Interpret this technical video frame for a developer or analyst.",
        f"Frame type: {frame_kind}.",
        f"Focus on: {KIND_FOCUS[frame_kind]}",
        "Explain visual meaning, relationships, state, and likely implication. Do not only transcribe text.",
        "If the image is unclear, cropped, low-confidence, or ambiguous, state that explicitly in uncertainty.",
        "Return compact JSON with: visual_summary, detected_elements, interpretation, uncertainty.",
    ]
    if ocr_text:
        parts.append(
            "OCR context is provided only as auxiliary evidence; use it to interpret the image and do not re-copy it verbatim."
        )
        parts.append(f"OCR context: {ocr_text[:1200]}")
    return VisionPrompt(
        profile=TECHNICAL_PROMPT_PROFILE,
        frame_kind=frame_kind,
        prompt="\n".join(parts),
    )
