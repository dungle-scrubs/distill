"""Unit tests for OCR helpers: tesseract discovery, frame OCR, and batch OCR."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from ast import literal_eval
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from distill import ocr
from distill.capabilities import EXTERNAL_TOOLS
from distill.ocr import (
    find_tesseract_command,
    ocr_frame,
    ocr_frames,
)


def _absent_tesseract_with_brew(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tesseract look absent on a macOS machine that does have Homebrew."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        ocr.shutil,
        "which",
        lambda command: "/opt/homebrew/bin/brew" if command == "brew" else None,
    )
    monkeypatch.setattr(ocr.Path, "exists", lambda _self: False)


def test_missing_tesseract_spawns_no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """R-34: Distill must not install software, so nothing may be spawned.

    The assertion is on the absence of a spawn, not on the warning: every
    subprocess entry point raises, so any install attempt fails the test.
    """
    spawned: list[Any] = []

    def forbid_spawn(*args: Any, **kwargs: Any) -> Any:
        spawned.append(args[0] if args else kwargs)
        raise AssertionError(f"ocr spawned a subprocess: {spawned[-1]!r}")

    _absent_tesseract_with_brew(monkeypatch)
    monkeypatch.setattr(subprocess, "run", forbid_spawn)
    monkeypatch.setattr(subprocess, "Popen", forbid_spawn)

    result = ocr.ensure_tesseract_available()

    assert spawned == []
    assert result is not None
    assert result["code"] == "tesseract_not_found"


def test_missing_tesseract_warns_and_the_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0002: image-text extraction is optional, so its absence degrades."""
    _absent_tesseract_with_brew(monkeypatch)
    frames = [
        {"index": 1, "path": str(tmp_path / "f1.png"), "relative_path": "frames/f1.png"},
        {"index": 2, "path": str(tmp_path / "f2.png"), "relative_path": "frames/f2.png"},
    ]

    updated, warnings = ocr_frames(frames, "eng", enabled=True)

    assert [frame["ocr_text"] for frame in updated] == ["", ""]
    assert [warning["code"] for warning in warnings] == ["tesseract_not_found"]
    # The warning states the cost the capability table records, not a bespoke
    # message that could drift from it.
    assert EXTERNAL_TOOLS["tesseract"].absence_cost in warnings[0]["message"]


def test_missing_tesseract_warning_does_not_depend_on_homebrew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-34 removed the install branch, so brew no longer changes the warning."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ocr.Path, "exists", lambda _self: False)

    monkeypatch.setattr(
        ocr.shutil,
        "which",
        lambda command: "/opt/homebrew/bin/brew" if command == "brew" else None,
    )
    with_brew = ocr.ensure_tesseract_available()

    monkeypatch.setattr(ocr.shutil, "which", lambda _command: None)
    without_brew = ocr.ensure_tesseract_available()

    assert with_brew == without_brew
    assert without_brew is not None
    assert without_brew["code"] == "tesseract_not_found"


def test_find_tesseract_command_returns_path_when_available() -> None:
    with patch("distill.ocr.shutil.which", return_value="/usr/bin/tesseract"):
        result = find_tesseract_command()
    assert result == "/usr/bin/tesseract"


def test_find_tesseract_command_returns_none_when_missing() -> None:
    with (
        patch("distill.ocr.shutil.which", return_value=None),
        patch("distill.ocr.Path.exists", return_value=False),
    ):
        result = find_tesseract_command()
    assert result is None


# A fake tesseract, invoked the way the CLI really is: `tesseract IMAGE stdout
# -l LANG`, with the recognized text on stdout. It records its argv so a test
# can assert on the invocation that happened rather than on a patched call.
FAKE_TESSERACT = """
import pathlib, sys

argv = sys.argv[1:]
image = pathlib.Path(argv[0])
(image.parent / "tesseract-argv.txt").write_text(repr(argv))
sys.stdout.write("SLIDE TEXT\\n\\n")
"""

FAKE_TESSERACT_FAILING = """
import sys

sys.stderr.write("Error in pixReadStream: Pix not read\\n")
sys.exit(1)
"""


def test_ocr_frame_reads_text_tesseract_printed_to_stdout(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """The adapter runs the tesseract binary itself and keeps what it read."""
    command = fake_tool("tesseract", FAKE_TESSERACT)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"not really a png")

    text, frame_warning = ocr_frame(frame, "eng", str(command))

    assert (text, frame_warning) == ("SLIDE TEXT", None)
    argv = literal_eval((tmp_path / "tesseract-argv.txt").read_text())
    # `stdout` as the output base is what makes tesseract print rather than
    # write a sibling .txt file, and `-l` selects the language pack.
    assert argv[1:] == ["stdout", "-l", "eng"]


def test_ocr_frame_degrades_when_tesseract_is_absent(tmp_path: Path) -> None:
    """ADR-0002: image-text extraction is an **optional capability**.

    Its absence yields the one warning the capability table defines, stating the
    cost, rather than a bespoke message or a raise.
    """
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake")

    with patch("distill.ocr.find_tesseract_command", return_value=None):
        text, frame_warning = ocr_frame(frame, "eng")

    assert text == ""
    assert frame_warning is not None
    assert frame_warning["code"] == "tesseract_not_found"
    assert EXTERNAL_TOOLS["tesseract"].absence_cost in frame_warning["message"]


def test_ocr_frame_reports_a_failing_tesseract_as_a_warning(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """One unreadable frame costs its extracted text, not the run."""
    command = fake_tool("tesseract", FAKE_TESSERACT_FAILING)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake")

    text, frame_warning = ocr_frame(frame, "eng", str(command))

    assert text == ""
    assert frame_warning is not None
    assert frame_warning["code"] == "ocr_failed"
    assert "frame.png" in frame_warning["message"]


def test_ocr_frame_emits_a_boundary_event(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """R-29: the invocation is visible at the boundary, named by its stage."""
    command = fake_tool("tesseract", FAKE_TESSERACT)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake")

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        ocr_frame(frame, "eng", str(command))

    events = [json.loads(record.message) for record in caplog.records]
    assert [event["detail"]["stage"] for event in events] == ["ocr"]
    assert events[0]["detail"]["tool"] == str(command)


def test_ocr_frames_sets_empty_ocr_text_when_disabled(tmp_path: Path) -> None:
    frames = [
        {"index": 1, "path": str(tmp_path / "f1.png"), "relative_path": "frames/f1.png"},
        {"index": 2, "path": str(tmp_path / "f2.png"), "relative_path": "frames/f2.png"},
    ]

    updated, warnings = ocr_frames(frames, "eng", enabled=False)

    assert len(updated) == 2
    assert all(frame["ocr_text"] == "" for frame in updated)
    assert warnings == []


def test_ocr_frames_skips_ocr_when_tesseract_unavailable(tmp_path: Path) -> None:
    frames = [
        {"index": 1, "path": str(tmp_path / "f1.png"), "relative_path": "frames/f1.png"},
    ]

    with (
        patch("distill.ocr.find_tesseract_command", return_value=None),
        patch("distill.ocr.ensure_tesseract_available") as mock_ensure,
    ):
        mock_ensure.return_value = {
            "stage": "ocr",
            "code": "tesseract_not_found",
            "message": "tesseract is not installed",
        }
        updated, warnings = ocr_frames(frames, "eng", enabled=True)

    assert updated[0]["ocr_text"] == ""
    assert any(w["code"] == "tesseract_not_found" for w in warnings)


def test_ocr_frames_processes_frames_when_tesseract_available(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "f1.png"
    frame_path.write_bytes(b"fake")
    frames = [
        {"index": 1, "path": str(frame_path), "relative_path": "frames/f1.png"},
    ]

    with (
        patch("distill.ocr.find_tesseract_command", return_value="/usr/bin/tesseract"),
        patch("distill.ocr.ensure_tesseract_available", return_value=None),
        patch("distill.ocr.ocr_frame", return_value=("hello world", None)),
    ):
        updated, warnings = ocr_frames(frames, "eng", enabled=True)

    assert updated[0]["ocr_text"] == "hello world"
    assert warnings == []
