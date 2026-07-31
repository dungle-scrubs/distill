from __future__ import annotations

from typing import Any

import pytest

from distill import ocr
from distill.artifacts import (
    FrameArtifact,
    Interpretation,
    Provenance,
    RedactionState,
    Transcript,
)
from distill.errors import DistillError
from distill.grounding import assess_grounding
from distill.progress import ProgressReporter
from distill.redact_secrets import redact_text
from distill.render import (
    UNCORROBORATED_NOTE,
    frames_are_useless,
    render_markdown,
    transcript_is_empty,
)


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
    assert "OPENAI_API_KEY=[REDACTED:assigned-secret]" in result.text
    assert "DEMO_API_KEY=your_api_key" in result.text


def test_env_tutorial_variable_names_are_preserved() -> None:
    result = redact_text(
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\n"
        "GITHUB_TOKEN=<your-api-key>\n"
        "DATABASE_PASSWORD=changeme"
    )
    assert "OPENAI_API_KEY=[REDACTED:assigned-secret]" in result.text
    assert "GITHUB_TOKEN=<your-api-key>" in result.text
    assert "DATABASE_PASSWORD=changeme" in result.text
    assert "OPENAI_API_KEY" in result.text


def test_confusable_secret_produces_warning() -> None:
    result = redact_text("ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef")
    assert any(item["code"] == "possible_confusable_secret" for item in result.warnings)


def test_confusable_secret_is_redacted_not_just_warned() -> None:
    result = redact_text("ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef")
    assert "[REDACTED:api-key]" in result.text
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
    assert "[REDACTED:assigned-secret]" in result.text


def test_bare_and_camelcase_secret_assignments_are_redacted() -> None:
    for text in (
        "password: hunter2length",
        "secret: topsecretvalue",
        "apiKey: abcdefghijklmnop",
        "access-token: abcdefghijklmnop",
    ):
        result = redact_text(text)
        assert "[REDACTED:assigned-secret]" in result.text, text
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
            "message": ("OCR text contained a secret-like value after confusable normalization"),
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
                    {"word": " second", "start": 0.5, "end": 1.5},
                    {"word": " third", "start": 1.5, "end": 2.5},
                ],
            }
        ),
        [keyframe(timestamp_sec=1.6)],
        [],
    )

    assert markdown.index("first second") < markdown.index("![Frame 1]")
    assert markdown.index("![Frame 1]") < markdown.index("third")


def test_transcript_tokens_keep_attached_punctuation_with_and_without_a_keyframe() -> None:
    """FAILS FIRST: stripping and joining tokens injected spaces at punctuation."""
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        spoken(
            {
                "start": 0.0,
                "end": 2.0,
                "text": "co-founder",
                "words": [
                    {"word": "co", "start": 0.0, "end": 0.5},
                    {"word": "-founder", "start": 0.5, "end": 1.5},
                ],
            },
            {
                "start": 3.0,
                "end": 6.0,
                "text": "40,000 raised",
                "words": [
                    {"word": "40", "start": 3.0, "end": 3.3},
                    {"word": ",000", "start": 3.3, "end": 3.8},
                    {"word": " raised", "start": 4.1, "end": 5.0},
                ],
            },
        ),
        [keyframe(timestamp_sec=4.0)],
        [],
    )

    assert "```untrusted-text\nco-founder\n```" in markdown
    assert "```untrusted-text\n40,000\n```" in markdown
    assert "co -founder" not in markdown
    assert "40 ,000" not in markdown


def test_a_segment_with_only_empty_word_records_uses_its_recorded_text() -> None:
    """FAILS FIRST: a nonempty `words` list suppressed the segment fallback."""
    markdown = render_markdown(
        "demo.mp4",
        10.0,
        spoken(
            {
                "start": 0.0,
                "end": 2.0,
                "text": "fallback transcript text",
                "words": [
                    {"word": "", "start": 0.0, "end": 0.5},
                    {"word": "   ", "start": 0.5, "end": 1.5},
                ],
            }
        ),
        [],
        [],
    )

    assert "```untrusted-text\nfallback transcript text\n```" in markdown


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

    A **self-contained render** deliberately omits image links, so a frame is
    useful only when it carries OCR text or an interpretation a reader can use.
    """
    assert transcript_is_empty(spoken({"text": "ok"})) is True
    assert transcript_is_empty(spoken({"text": "okay"})) is False
    assert frames_are_useless([]) is True
    assert frames_are_useless([keyframe()]) is True
    assert frames_are_useless([keyframe(extracted_text="slide")]) is False
    interpreted, _warnings = keyframe().with_interpretation(
        Interpretation(visual_summary="A readable slide")
    )
    assert frames_are_useless([interpreted]) is False


def test_a_slides_only_reading_is_usable_without_a_generation_path() -> None:
    """FAILS FIRST: usability was based on `relative_path`, not a frame reading."""
    frame = keyframe(relative_path="", extracted_text="slides-only source")

    assert frames_are_useless([frame]) is False
    markdown = render_markdown(
        "ignored-machine-path.mp4",
        10.0,
        None,
        [frame],
        [],
        provenance=Provenance(
            title="slides.mp4",
            duration_sec=10.0,
            processed_at="2026-07-29T14:20:00Z",
        ),
        include_frame_links=False,
    )
    assert "slides-only source" in markdown


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


def test_an_uncorroborated_frame_renders_the_neutral_note_and_verbatim_block() -> None:
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

    # M3.1/D-002: grounding is a note, not a verdict - the reading stands and
    # the render says only that the other reader did not confirm it.
    assert UNCORROBORATED_NOTE in markdown
    assert "Low-confidence" not in markdown
    assert "unverified" not in markdown
    # The model's word about its own reading is the model's word: R-26 moved it
    # out of the inline bullet this used to pin and into a delimited block. The
    # marker above it is Distill's own assessment and stays document structure.
    assert "Text confidence:\n\n```untrusted-text\nnone\n```" in markdown


def graded(*, ocr_text: str, reading: Interpretation) -> FrameArtifact:
    """A **frame artifact** carrying the **grounding** the pipeline would assess for it.

    The two marker tests below go through `assess_grounding` rather than
    hand-writing a level, because what they are about is which readings the
    render marks - and a hand-written level makes that a question about the
    string the test chose instead of about the reading. It is the same call
    `local_vision._interpreted` makes, in the same order: grounding is assessed
    against what the model returned, and the carrier redacts on the way in.
    """
    frame, _warnings = keyframe(extracted_text=ocr_text).with_interpretation(
        reading,
        grounding=assess_grounding(
            ocr_text=ocr_text,
            verbatim_text=reading.verbatim_text,
            text_confidence=reading.text_confidence,
            has_interpretation=reading.has_interpretation,
            carries_a_reading=reading.carries_a_reading,
        ).public_dict(),
    )
    return frame


def test_a_lone_confident_reader_is_still_reported_uncorroborated() -> None:
    """R-42: the render does not suppress the marker on a reader's own say-so.

    OCR recovered nothing, so the only evidence for this slide's text is the
    model's report of having read it. The reading is still shown - it may well
    be right - but it is shown under the banner, because a reader given it
    unmarked would have no way to tell it from text two readers agreed on.
    """
    frame = graded(
        ocr_text="",
        reading=Interpretation(
            visual_summary="A dark slide",
            detected_elements=("title",),
            interpretation="The slide lists what a software factory needs.",
            uncertainty="Low",
            verbatim_text="What a software factory needs Agent Runtimes Orchestration",
            text_confidence="high",
        ),
    )

    markdown = render_markdown("demo.mp4", 10.0, None, [frame], [])

    assert UNCORROBORATED_NOTE in markdown
    assert "Low-confidence" not in markdown
    # The reading is marked, not withheld.
    assert "```untrusted-text\nWhat a software factory needs Agent Runtimes Orchestration\n```" in (
        markdown
    )


def test_a_description_of_an_unreadable_frame_reaches_the_reader_with_its_note() -> None:
    """The render seam of the same finding: a summary-only reading is banner-ed.

    The model was told to leave `verbatim_text` empty when it cannot read the
    frame; it described the slide anyway, quoting figures nothing corroborates.
    Rendered unmarked, that paragraph reads exactly like text two readers
    agreed on - which is the one thing the banner exists to prevent.
    """
    frame = graded(
        ocr_text="",
        reading=Interpretation(
            visual_summary="A slide titled 'Q3 revenue: $4.2M, up 340% YoY' with a bar chart",
            text_confidence="none",
        ),
    )

    markdown = render_markdown("demo.mp4", 10.0, None, [frame], [])

    assert UNCORROBORATED_NOTE in markdown
    assert "Low-confidence" not in markdown
    # The reading is marked, not withheld.
    assert "Q3 revenue" in markdown


def test_a_frame_nobody_said_anything_about_is_rendered_without_a_banner() -> None:
    """The other direction: nothing was claimed, so there is nothing to warn about.

    A frame carrying no reading and no OCR text is a photo or a blank slide.
    Marking it would put the banner on every such frame and make it mean
    nothing, so the fix cannot be "mark whatever has no verbatim text".
    """
    frame = graded(ocr_text="", reading=Interpretation(text_confidence="none"))

    markdown = render_markdown("demo.mp4", 10.0, None, [frame], [])

    assert "Low-confidence" not in markdown


def test_a_corroborated_frame_omits_the_marker_and_shows_verbatim() -> None:
    """The other direction of R-42: genuine agreement is still not marked.

    Two readers independently recovered the same text, which is what
    **corroborated** means and the one thing that earns an unmarked reading.
    Marking this one too would answer the finding by making the banner say
    nothing.
    """
    frame = graded(
        ocr_text="We're closer than you think",
        reading=Interpretation(
            visual_summary="A title slide",
            detected_elements=("title",),
            interpretation="Closing slide.",
            uncertainty="Low",
            verbatim_text="We're closer than you think",
            text_confidence="high",
        ),
    )
    markdown = render_markdown("demo.mp4", 10.0, None, [frame], [])

    assert "Low-confidence" not in markdown
    assert "Verbatim slide text:" in markdown
    # Still fenced, and now tagged with the trust class the text has: verbatim
    # text is the model's report of what a **keyframe** showed (R-26).
    assert "```untrusted-text\nWe're closer than you think\n```" in markdown
