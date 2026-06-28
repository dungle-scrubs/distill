"""Unit tests for OCR helpers: tesseract discovery, frame OCR, and batch OCR."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from distill.ocr import (
    find_tesseract_command,
    ocr_frame,
    ocr_frames,
)


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


def test_ocr_frame_returns_empty_and_warning_when_pytesseract_missing(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake")

    with patch.dict("sys.modules", {"pytesseract": None}):
        text, warning_result = ocr_frame(frame, "eng")

    assert text == ""
    assert warning_result is not None
    assert warning_result["code"] == "tesseract_python_missing"


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
