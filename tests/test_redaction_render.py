from __future__ import annotations

from typing import Any

import pytest

from distill import ocr
from distill.artifacts import FrameArtifact, Interpretation, RedactionState, Transcript
from distill.errors import DistillError
from distill.progress import ProgressReporter
from distill.redact_secrets import redact_text
from distill.render import frames_are_useless, render_markdown, transcript_is_empty


def keyframe(**overrides: Any) -> FrameArtifact:
    """One **frame artifact** as the pipeline hands it to the **render**."""
    fields: dict[str, Any] = {
        "index": 1,
        "timestamp_sec": 2.0,
        "path": "/tmp/frames/frame_0001.png",
        "relative_path": "frames/frame_0001.png",
    }
    fields.update(overrides)
    return FrameArtifact(**fields)


def spoken(*segments: dict[str, Any]) -> Transcript:
    """A **transcript** carrier holding exactly these segments."""
    return Transcript(language="en", segments=tuple(segments))


def test_every_field_of_an_interpretation_is_redacted_when_it_enters_the_frame() -> None:
    """R-19: the vision model's words are redacted where they enter the carrier.

    Not by a helper called afterwards. `local_vision._redact_result_fields` was
    that helper, and it is gone: while it existed there was a window in which an
    **interpretation** holding a secret was a value anything could write, and
    the only thing keeping it out of a **stage result** was the order two calls
    happened to be in.

    The model echoes the screen back into every field it has, so every field is
    checked - `verbatim_text` alone was the original defect.
    """
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    reading = Interpretation(
        visual_summary=f"summary {secret}",
        detected_elements=(f"element {secret}",),
        interpretation=f"interpretation {secret}",
        uncertainty=f"uncertainty {secret}",
        backend="rapid-mlx",
        model="test-model",
        prompt_profile="technical",
        verbatim_text=f"verbatim {secret}",
        text_confidence="high",
    )

    frame, _warnings = keyframe().with_interpretation(reading)

    redacted = frame.reading
    assert redacted is not None
    assert secret not in redacted.visual_summary
    assert secret not in redacted.interpretation
    assert secret not in redacted.uncertainty
    assert secret not in redacted.verbatim_text
    assert all(secret not in element for element in redacted.detected_elements)
    # Non-text metadata is preserved untouched.
    assert redacted.backend == "rapid-mlx"
    assert redacted.text_confidence == "high"


def test_an_interpretation_survives_intact_under_an_explicitly_disabled_policy() -> None:
    """R-20 with D-020: `--no-redact-secrets` reaches the vision pass too.

    The policy is the frame's, so a frame created under the opt-out carries the
    model's words verbatim without the vision pass being told anything.
    """
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    opted_out = keyframe(redaction=RedactionState.DISABLED)

    frame, _warnings = opted_out.with_interpretation(
        Interpretation(verbatim_text=f"verbatim {secret}")
    )

    reading = frame.reading
    assert reading is not None
    assert reading.verbatim_text == f"verbatim {secret}"
    assert frame.redaction is RedactionState.DISABLED


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


def test_four_confusable_secrets_are_one_warning_that_says_four() -> None:
    """R-41 replaces the cap this test was named for.

    Classified *defect*: the old contract emitted one record per match up to a
    cap, then a second record saying how many records it had suppressed - a
    **warning** about Distill's own bookkeeping, and a reader who wanted the
    real number had to add the two together. Folding on stage and code says it
    once and says it exactly, so the cap and its truncation record are gone.
    """
    text = " ".join(["ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef"] * 4)
    result = redact_text(text)

    assert result.warnings == [
        {
            "stage": "redaction",
            "code": "possible_confusable_secret",
            "message": (
                "OCR text contained a secret-like value after confusable normalization"
            ),
            "occurrences": 4,
        }
    ]


def test_render_interleaves_frame_and_transcript() -> None:
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        spoken({"start": 0.0, "end": 5.0, "text": "hello", "words": []}),
        [keyframe(extracted_text="slide text")],
        [],
    )
    assert "![Frame 1](frames/frame_0001.png)" in markdown
    assert "hello" in markdown
    assert "slide text" in markdown


def test_render_shows_visual_interpretation_adjacent_to_ocr() -> None:
    """The reading sits above the image text it is a reading of, both delimited.

    Adjacency is what this test is about and it is unchanged. What changed is
    the form: R-26 puts every **interpretation** field inside a block it cannot
    terminate, so the inline `- Summary: ...` bullets this used to pin are
    gone - a bullet ended at the model's first newline, which is finding 5.
    `tests/test_render_delimiting.py` is where that boundary is checked; here
    the fields only have to still be there and still be in order.
    """
    frame, _warnings = keyframe(extracted_text="Save").with_interpretation(
        Interpretation(
            visual_summary="A settings form",
            detected_elements=("save button",),
            interpretation="The form is ready to save.",
            uncertainty="Low",
        )
    )
    markdown = render_markdown("demo.mp4", 10.0, None, [frame], [])

    assert "Visual interpretation:" in markdown
    assert "Summary:\n\n```untrusted-text\nA settings form\n```" in markdown
    assert "Detected elements:\n\n```untrusted-text\nsave button\n```" in markdown
    assert markdown.index("Visual interpretation:") < markdown.index("OCR:")
    assert "```untrusted-text\nSave\n```" in markdown


def test_render_interleaves_frame_at_transcript_word_boundary() -> None:
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        spoken(
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
        ),
        [keyframe(timestamp_sec=1.6)],
        [],
    )

    assert markdown.index("first second") < markdown.index("![Frame 1]")
    assert markdown.index("![Frame 1]") < markdown.index("third")


def test_vad_gap_frame_renders_as_standalone_chronological_section() -> None:
    markdown = render_markdown(
        "demo.mp4",
        20.0,
        spoken(
            {"start": 0.0, "end": 2.0, "text": "intro", "words": []},
            {"start": 10.0, "end": 12.0, "text": "outro", "words": []},
        ),
        [keyframe(timestamp_sec=6.0)],
        [],
    )

    assert markdown.index("intro") < markdown.index("## Frame 1")
    assert markdown.index("## Frame 1") < markdown.index("outro")


def test_content_threshold_helpers_are_documented_behavior() -> None:
    """The two thresholds `ensure_content` reads, stated against carriers.

    The `blank` flag the third case used to assert is gone with the bare frame
    dict: no producer ever set it, so it described a frame nothing could make.
    A frame with no image is now the only useless one, and `select_keyframes`
    drops a **keyframe** it could not extract rather than recording it.
    """
    assert transcript_is_empty(spoken({"text": "ok"})) is True
    assert transcript_is_empty(spoken({"text": "okay"})) is False
    assert frames_are_useless([]) is True
    assert frames_are_useless([keyframe()]) is False
    assert frames_are_useless([keyframe(relative_path="")]) is True


def test_ocr_reports_frame_index_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-keyframe progress, and the reading landing on each carrier.

    The fixture is carriers because `ocr_frames` takes carriers (R-19); what is
    asserted about the text is `extracted_text` on the artifact rather than a
    key on a mapping, which is the same claim about the same reading made where
    the reading now lives.
    """
    frames = [
        keyframe(index=1, timestamp_sec=0.0, path="frame_1.png"),
        keyframe(index=2, timestamp_sec=1.0, path="frame_2.png"),
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
    assert [frame.extracted_text for frame in updated] == [
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
    frame, _warnings = keyframe().with_interpretation(
        Interpretation(
            visual_summary="A dark slide",
            detected_elements=("title",),
            interpretation="The slide discusses something.",
            uncertainty="High",
        ),
        grounding={
            "level": "ungrounded",
            "text_overlap": None,
            "reason": "interpretation present but no readable on-screen text supports it",
        },
    )
    markdown = render_markdown("demo.mp4", 10.0, None, [frame], [])

    assert "⚠ Low-confidence frame (ungrounded)" in markdown
    assert "treat the interpretation below as unverified" in markdown
    # The model's word about its own reading is the model's word: R-26 moved it
    # out of the inline bullet this used to pin and into a delimited block. The
    # marker above it is Distill's own assessment and stays document structure.
    assert "Text confidence:\n\n```untrusted-text\nnone\n```" in markdown


def test_grounded_frame_omits_warning_marker_and_shows_verbatim() -> None:
    frame, _warnings = keyframe(
        extracted_text="We're closer than you think"
    ).with_interpretation(
        Interpretation(
            visual_summary="A title slide",
            detected_elements=("title",),
            interpretation="Closing slide.",
            uncertainty="Low",
            verbatim_text="We're closer than you think",
            text_confidence="high",
        ),
        grounding={
            "level": "grounded",
            "text_overlap": 1.0,
            "reason": "OCR corroborates the transcribed text",
        },
    )
    markdown = render_markdown("demo.mp4", 10.0, None, [frame], [])

    assert "Low-confidence frame" not in markdown
    assert "Verbatim slide text:" in markdown
    # Still fenced, and now tagged with the trust class the text has: verbatim
    # text is the model's report of what a **keyframe** showed (R-26).
    assert "```untrusted-text\nWe're closer than you think\n```" in markdown
