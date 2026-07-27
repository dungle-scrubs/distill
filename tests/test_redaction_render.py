from __future__ import annotations

import pytest

from distill import ocr
from distill.errors import DistillError
from distill.local_vision import LocalVisionResult, _redact_result_fields
from distill.progress import ProgressReporter
from distill.redact_secrets import redact_text
from distill.render import frames_are_useless, render_markdown, transcript_is_empty


def test_all_vision_fields_are_redacted_not_only_verbatim_text() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    result = LocalVisionResult(
        visual_summary=f"summary {secret}",
        detected_elements=[f"element {secret}"],
        interpretation=f"interpretation {secret}",
        uncertainty=f"uncertainty {secret}",
        backend="rapid-mlx",
        model="test-model",
        prompt_profile="technical",
        verbatim_text=f"verbatim {secret}",
        text_confidence="high",
    )

    redacted, _warnings = _redact_result_fields(result)

    assert secret not in redacted.visual_summary
    assert secret not in redacted.interpretation
    assert secret not in redacted.uncertainty
    assert secret not in redacted.verbatim_text
    assert all(secret not in element for element in redacted.detected_elements)
    # Non-text metadata is preserved untouched.
    assert redacted.backend == "rapid-mlx"
    assert redacted.text_confidence == "high"


def test_redacts_env_value_but_not_tutorial_placeholder() -> None:
    result = redact_text("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\nDEMO_API_KEY=your_api_key")
    assert "OPENAI_API_KEY=[REDACTED]" in result.text
    assert "DEMO_API_KEY=your_api_key" in result.text


def test_env_tutorial_variable_names_are_preserved() -> None:
    result = redact_text(
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\n"
        "GITHUB_TOKEN=<your-api-key>\n"
        "DATABASE_PASSWORD=changeme"
    )
    assert "OPENAI_API_KEY=[REDACTED]" in result.text
    assert "GITHUB_TOKEN=<your-api-key>" in result.text
    assert "DATABASE_PASSWORD=changeme" in result.text
    assert "OPENAI_API_KEY" in result.text


def test_confusable_secret_produces_warning() -> None:
    result = redact_text("ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef")
    assert any(item["code"] == "possible_confusable_secret" for item in result.warnings)


def test_confusable_secret_is_redacted_not_just_warned() -> None:
    result = redact_text("ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef")
    assert "[REDACTED]" in result.text
    assert "ｓｋ" not in result.text
    assert result.redaction_count >= 1


def test_additional_secret_formats_are_redacted() -> None:
    google = "AIza" + "a" * 35
    slack = "xoxb-1234567890abcdef"
    stripe = "sk_live_0123456789abcdef"
    jwt = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12
    result = redact_text(f"g={google} s={slack} p={stripe} j={jwt}")
    for secret in (google, slack, stripe, jwt):
        assert secret not in result.text
    assert result.redaction_count >= 4


def test_lowercase_colon_config_assignment_is_redacted() -> None:
    result = redact_text("api_key: sk-abcdefghijklmnopqrstuvwxyz")
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in result.text
    assert "[REDACTED]" in result.text


def test_bare_and_camelcase_secret_assignments_are_redacted() -> None:
    for text in (
        "password: hunter2length",
        "secret: topsecretvalue",
        "apiKey: abcdefghijklmnop",
        "access-token: abcdefghijklmnop",
    ):
        result = redact_text(text)
        assert "[REDACTED]" in result.text, text
        assert result.redaction_count >= 1, text


def test_ordinary_words_ending_in_a_suffix_are_not_redacted() -> None:
    # These end in "key" / look assignment-ish but are plain prose, not secrets.
    for text in (
        "Turkey: 81 million people",
        "monkey: a small primate",
        "Whiskey: 40 percent",
        "Key: Legend for the chart",
        "Token: a single sign-in label",
    ):
        result = redact_text(text)
        assert result.text == text, text
        assert result.redaction_count == 0, text


def test_possible_secret_warnings_are_capped_with_aggregate_warning() -> None:
    text = " ".join(["ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef"] * 4)
    result = redact_text(text, max_possible_secret_warnings=2)

    assert [item["code"] for item in result.warnings] == [
        "possible_confusable_secret",
        "possible_confusable_secret",
        "possible_secret_warnings_truncated",
    ]
    assert result.warnings[-1]["message"] == (
        "2 additional possible-secret warnings were truncated"
    )


def test_render_interleaves_frame_and_transcript() -> None:
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        {"segments": [{"start": 0.0, "end": 5.0, "text": "hello", "words": []}]},
        [
            {
                "index": 1,
                "timestamp_sec": 2.0,
                "relative_path": "frames/frame_0001.png",
                "ocr_text": "slide text",
            }
        ],
        [],
    )
    assert "![Frame 1](frames/frame_0001.png)" in markdown
    assert "hello" in markdown
    assert "slide text" in markdown


def test_render_shows_visual_interpretation_adjacent_to_ocr() -> None:
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        None,
        [
            {
                "index": 1,
                "timestamp_sec": 2.0,
                "relative_path": "frames/frame_0001.png",
                "ocr_text": "Save",
                "visual_interpretation": {
                    "visual_summary": "A settings form",
                    "detected_elements": ["save button"],
                    "interpretation": "The form is ready to save.",
                    "uncertainty": "Low",
                },
            }
        ],
        [],
    )

    assert "Visual interpretation:" in markdown
    assert "- Summary: A settings form" in markdown
    assert "- Detected elements: save button" in markdown
    assert markdown.index("Visual interpretation:") < markdown.index("OCR:")
    assert "```text\nSave\n```" in markdown


def test_render_interleaves_frame_at_transcript_word_boundary() -> None:
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "first second third",
                    "words": [
                        {"word": "first", "start": 0.0, "end": 0.5},
                        {"word": "second", "start": 0.5, "end": 1.5},
                        {"word": "third", "start": 1.5, "end": 2.5},
                    ],
                }
            ]
        },
        [
            {
                "index": 1,
                "timestamp_sec": 1.6,
                "relative_path": "frames/frame_0001.png",
                "ocr_text": "",
            }
        ],
        [],
    )

    assert markdown.index("first second") < markdown.index("![Frame 1]")
    assert markdown.index("![Frame 1]") < markdown.index("third")


def test_vad_gap_frame_renders_as_standalone_chronological_section() -> None:
    markdown = render_markdown(
        "demo.mp4",
        20.0,
        {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "intro", "words": []},
                {"start": 10.0, "end": 12.0, "text": "outro", "words": []},
            ]
        },
        [
            {
                "index": 1,
                "timestamp_sec": 6.0,
                "relative_path": "frames/frame_0001.png",
                "ocr_text": "",
            }
        ],
        [],
    )

    assert markdown.index("intro") < markdown.index("## Frame 1")
    assert markdown.index("## Frame 1") < markdown.index("outro")


def test_content_threshold_helpers_are_documented_behavior() -> None:
    assert transcript_is_empty({"segments": [{"text": "ok"}]}) is True
    assert transcript_is_empty({"segments": [{"text": "okay"}]}) is False
    assert frames_are_useless([]) is True
    assert frames_are_useless([{"relative_path": "frames/a.png"}]) is False
    assert frames_are_useless([{"relative_path": "frames/a.png", "blank": True}]) is True


def test_ocr_reports_frame_index_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [
        {"path": "frame_1.png", "timestamp_sec": 0.0},
        {"path": "frame_2.png", "timestamp_sec": 1.0},
    ]
    monkeypatch.setattr(ocr, "ensure_tesseract_available", lambda: None)
    monkeypatch.setattr(ocr, "find_tesseract_command", lambda: "/opt/homebrew/bin/tesseract")
    monkeypatch.setattr(
        ocr,
        "ocr_frame",
        lambda _path, _language, _cmd=None, _preprocess=True: ("detected text", []),
    )
    progress = ProgressReporter()

    updated, warnings = ocr.ocr_frames(frames, "eng", True, progress)

    assert warnings == []
    assert [frame["ocr_text"] for frame in updated] == [
        "detected text",
        "detected text",
    ]
    ocr_events = [event for event in progress.events if event.mechanism == "ocr"]
    frame_events = [event for event in ocr_events if event.percent is not None]
    assert [event.percent for event in frame_events[:2]] == [50.0, 100.0]
    assert ocr_events[-1].status == "completed"


def test_no_content_escalates() -> None:
    with pytest.raises(DistillError) as exc:
        render_markdown("x", 1.0, None, [], [])
    assert exc.value.code == "E_NO_CONTENT"


def test_low_confidence_frame_renders_warning_marker_and_verbatim_block() -> None:
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        None,
        [
            {
                "index": 1,
                "timestamp_sec": 2.0,
                "relative_path": "frames/frame_0001.png",
                "ocr_text": "",
                "visual_interpretation": {
                    "visual_summary": "A dark slide",
                    "detected_elements": ["title"],
                    "interpretation": "The slide discusses something.",
                    "uncertainty": "High",
                    "verbatim_text": "",
                    "text_confidence": "none",
                },
                "visual_confidence": {
                    "level": "ungrounded",
                    "text_overlap": None,
                    "reason": "interpretation present but no readable on-screen text supports it",
                },
            }
        ],
        [],
    )

    assert "⚠ Low-confidence frame (ungrounded)" in markdown
    assert "treat the interpretation below as unverified" in markdown
    assert "- Text confidence: none" in markdown


def test_grounded_frame_omits_warning_marker_and_shows_verbatim() -> None:
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        None,
        [
            {
                "index": 1,
                "timestamp_sec": 2.0,
                "relative_path": "frames/frame_0001.png",
                "ocr_text": "We're closer than you think",
                "visual_interpretation": {
                    "visual_summary": "A title slide",
                    "detected_elements": ["title"],
                    "interpretation": "Closing slide.",
                    "uncertainty": "Low",
                    "verbatim_text": "We're closer than you think",
                    "text_confidence": "high",
                },
                "visual_confidence": {
                    "level": "grounded",
                    "text_overlap": 1.0,
                    "reason": "OCR corroborates the transcribed text",
                },
            }
        ],
        [],
    )

    assert "Low-confidence frame" not in markdown
    assert "Verbatim slide text:" in markdown
    assert "```text\nWe're closer than you think\n```" in markdown
