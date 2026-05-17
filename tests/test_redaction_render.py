from __future__ import annotations

import subprocess

import pytest

from saccade import ocr
from saccade.errors import SaccadeError
from saccade.progress import ProgressReporter
from saccade.redact_secrets import redact_text
from saccade.render import frames_are_useless, render_markdown, transcript_is_empty


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
        lambda _path, _language, _cmd=None: ("detected text", None),
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


def test_ocr_auto_installs_tesseract_with_brew_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups = {"tesseract": 0}
    commands: list[list[str]] = []

    def fake_which(command: str) -> str | None:
        if command == "brew":
            return "/opt/homebrew/bin/brew"
        if command == "tesseract":
            lookups["tesseract"] += 1
            return "/opt/homebrew/bin/tesseract" if lookups["tesseract"] > 1 else None
        return None

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ocr.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ocr.shutil, "which", fake_which)
    monkeypatch.setattr(ocr.Path, "exists", lambda _self: False)
    monkeypatch.setattr(ocr.subprocess, "run", fake_run)

    assert ocr.ensure_tesseract_available() is None
    assert commands == [["/opt/homebrew/bin/brew", "install", "tesseract"]]


def test_ocr_reports_install_warning_when_brew_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocr.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ocr.shutil, "which", lambda _command: None)
    monkeypatch.setattr(ocr.Path, "exists", lambda _self: False)

    warning = ocr.ensure_tesseract_available()

    assert warning == {
        "stage": "ocr",
        "code": "tesseract_brew_missing",
        "message": "tesseract is missing and Homebrew is not available to install it",
    }


def test_no_content_escalates() -> None:
    with pytest.raises(SaccadeError) as exc:
        render_markdown("x", 1.0, None, [], [])
    assert exc.value.code == "E_NO_CONTENT"
