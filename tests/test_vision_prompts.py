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


def test_technical_prompt_asks_for_grounded_interpretation() -> None:
    prompt = build_technical_frame_prompt("chart_graph").prompt

    assert "visual meaning" in prompt
    assert "relationships" in prompt
    assert "strictly on what is visible" in prompt
    assert "uncertainty" in prompt


def test_prompt_forbids_inventing_content_and_requires_verbatim_split() -> None:
    prompt = build_technical_frame_prompt("slide").prompt

    assert "Do not invent, infer, or guess" in prompt
    assert "verbatim_text" in prompt
    assert "text_confidence" in prompt
    # An unreadable frame must be allowed to return nothing rather than fiction.
    assert "Returning little is correct when the frame is unreadable" in prompt


def test_prompt_auto_classifies_when_no_kind_is_given() -> None:
    prompt = build_technical_frame_prompt().prompt

    assert "First classify the frame as one of:" in prompt
    for kind in FRAME_KINDS:
        assert kind in prompt
    assert build_technical_frame_prompt().frame_kind == "auto"


def test_ocr_context_is_supporting_evidence() -> None:
    prompt = build_technical_frame_prompt(
        "terminal",
        ocr_text="ERROR: failed to connect\nRetrying in 5s",
    ).prompt

    assert "supporting evidence" in prompt
    assert "ignore OCR that conflicts with what you see" in prompt
    assert "ERROR: failed to connect" in prompt


def test_unclear_images_must_report_uncertainty() -> None:
    prompt = build_technical_frame_prompt("diagram").prompt

    assert "unclear" in prompt
    assert "low-confidence" in prompt
    assert "ambiguous" in prompt
    assert "state that explicitly in\nuncertainty" in prompt or "uncertainty" in prompt


def test_prompt_is_deterministic() -> None:
    first = build_technical_frame_prompt("ui_interface", ocr_text="Save changes").prompt
    second = build_technical_frame_prompt("ui_interface", ocr_text="Save changes").prompt

    assert first == second
    assert build_technical_frame_prompt("ui_interface").profile == TECHNICAL_PROMPT_PROFILE
