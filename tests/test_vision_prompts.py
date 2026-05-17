from __future__ import annotations

from saccade.vision_prompts import (
    FRAME_KINDS,
    TECHNICAL_PROMPT_PROFILE,
    build_technical_frame_prompt,
)


def test_prompt_snapshots_cover_technical_frame_kinds() -> None:
    prompts = {
        frame_kind: build_technical_frame_prompt(frame_kind).prompt for frame_kind in FRAME_KINDS
    }

    assert set(prompts) == {
        "ui_interface",
        "chart_graph",
        "diagram",
        "code_screenshot",
        "terminal",
        "slide",
    }
    assert "UI state" in prompts["ui_interface"]
    assert "axes, legends, series" in prompts["chart_graph"]
    assert "nodes, arrows" in prompts["diagram"]
    assert "language or framework" in prompts["code_screenshot"]
    assert "commands, prompts" in prompts["terminal"]
    assert "main claim" in prompts["slide"]


def test_technical_prompt_asks_for_interpretation_not_ocr_only() -> None:
    prompt = build_technical_frame_prompt("chart_graph").prompt

    assert "visual meaning" in prompt
    assert "relationships" in prompt
    assert "state" in prompt
    assert "Do not only transcribe text" in prompt
    assert "uncertainty" in prompt


def test_ocr_context_is_auxiliary_and_not_recopy_instruction() -> None:
    prompt = build_technical_frame_prompt(
        "terminal",
        ocr_text="ERROR: failed to connect\nRetrying in 5s",
    ).prompt

    assert "OCR context is provided only as auxiliary evidence" in prompt
    assert "do not re-copy it verbatim" in prompt
    assert "ERROR: failed to connect" in prompt


def test_unclear_images_must_report_uncertainty() -> None:
    prompt = build_technical_frame_prompt("diagram").prompt

    assert "unclear" in prompt
    assert "low-confidence" in prompt
    assert "ambiguous" in prompt
    assert "state that explicitly in uncertainty" in prompt


def test_prompt_is_deterministic_and_concise() -> None:
    first = build_technical_frame_prompt("ui_interface", ocr_text="Save changes").prompt
    second = build_technical_frame_prompt("ui_interface", ocr_text="Save changes").prompt

    assert first == second
    assert len(first.split()) < 140
    assert build_technical_frame_prompt("ui_interface").profile == TECHNICAL_PROMPT_PROFILE
