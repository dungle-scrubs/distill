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
from distill.artifacts import FrameArtifact, RedactionState
from distill.capabilities import EXTERNAL_TOOLS
from distill.ocr import (
    find_tesseract_command,
    ocr_frame,
    ocr_frames,
)


def keyframe(index: int, image: Path, **overrides: Any) -> FrameArtifact:
    """One **frame artifact** as `select_keyframes` hands it to the OCR pass.

    A carrier and not a mapping: R-19 moved the frame schema onto
    `FrameArtifact`, so a test that built a bare frame dict here would be
    asserting against a shape `ocr_frames` is no longer given and would keep
    passing while the real caller broke.
    """
    fields: dict[str, Any] = {
        "index": index,
        "timestamp_sec": float(index),
        "path": str(image),
        "relative_path": f"frames/{image.name}",
    }
    fields.update(overrides)
    return FrameArtifact(**fields)


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
    frames = [keyframe(1, tmp_path / "f1.png"), keyframe(2, tmp_path / "f2.png")]

    updated, warnings = ocr_frames(frames, "eng", enabled=True)

    assert [frame.extracted_text for frame in updated] == ["", ""]
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

    text, frame_warnings = ocr_frame(frame, "eng", str(command))

    assert (text, frame_warnings) == ("SLIDE TEXT", [])
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
        text, frame_warnings = ocr_frame(frame, "eng")

    assert text == ""
    assert [w["code"] for w in frame_warnings] == ["tesseract_not_found"]
    assert EXTERNAL_TOOLS["tesseract"].absence_cost in frame_warnings[0]["message"]


def test_ocr_frame_reports_a_failing_tesseract_as_a_warning(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """One unreadable frame costs its extracted text, not the run."""
    command = fake_tool("tesseract", FAKE_TESSERACT_FAILING)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake")

    text, frame_warnings = ocr_frame(frame, "eng", str(command))

    assert text == ""
    assert [w["code"] for w in frame_warnings] == ["ocr_failed"]
    assert "frame.png" in frame_warnings[0]["message"]
    # Finding 9: the generic "command failed: <path>" says nothing a reader can
    # act on, so tesseract's own reason travels with it.
    assert "Pix not read" in frame_warnings[0]["message"]


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


def test_ocr_frames_records_an_empty_reading_when_disabled(tmp_path: Path) -> None:
    """A disabled pass still produces the carrier, holding nothing.

    "Nobody looked" and "nothing was there" have to look the same downstream,
    because the **render** and the **manifest** are written from the artifact
    either way; what distinguishes them is the **warning**, not a missing field.
    """
    frames = [keyframe(1, tmp_path / "f1.png"), keyframe(2, tmp_path / "f2.png")]

    updated, warnings = ocr_frames(frames, "eng", enabled=False)

    assert len(updated) == 2
    assert all(frame.extracted_text == "" for frame in updated)
    assert warnings == []


def test_ocr_frames_skips_ocr_when_tesseract_unavailable(tmp_path: Path) -> None:
    frames = [keyframe(1, tmp_path / "f1.png")]

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

    assert updated[0].extracted_text == ""
    assert any(w["code"] == "tesseract_not_found" for w in warnings)


def test_what_tesseract_read_lands_on_the_carrier_redacted(tmp_path: Path) -> None:
    """The reading reaches the **frame artifact**, through the redaction policy.

    R-19: this is the seam finding 4 was hiding in. What `ocr_frame` returns is
    raw, and the next thing that happens to a frame is a `write_stage`, so the
    only arrangement in which a secret cannot become durable is one where the
    text is already inside a carrier when this function returns.
    """
    frame_path = tmp_path / "f1.png"
    frame_path.write_bytes(b"fake")
    reading = "hello world OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"

    with (
        patch("distill.ocr.find_tesseract_command", return_value="/usr/bin/tesseract"),
        patch("distill.ocr.ensure_tesseract_available", return_value=None),
        patch("distill.ocr.ocr_frame", return_value=(reading, [])),
    ):
        updated, warnings = ocr_frames([keyframe(1, frame_path)], "eng", enabled=True)

    assert updated[0].extracted_text == "hello world OPENAI_API_KEY=[REDACTED]"
    assert updated[0].redaction is RedactionState.APPLIED
    assert warnings == []


def test_the_opt_out_travels_on_the_frame_rather_than_being_passed_again(
    tmp_path: Path,
) -> None:
    """`--no-redact-secrets` is the frame's policy, and OCR never asks about it.

    R-20 keeps the opt-out working, and D-020 makes it a recorded state rather
    than an inference. `ocr_frames` takes no redaction argument at all: the
    policy entered the artifact at `select_keyframes` and every later stage
    inherits it, so there is no stage that can be given the wrong one.
    """
    frame_path = tmp_path / "f1.png"
    frame_path.write_bytes(b"fake")
    reading = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
    opted_out = keyframe(1, frame_path, redaction=RedactionState.DISABLED)

    with (
        patch("distill.ocr.find_tesseract_command", return_value="/usr/bin/tesseract"),
        patch("distill.ocr.ensure_tesseract_available", return_value=None),
        patch("distill.ocr.ocr_frame", return_value=(reading, [])),
    ):
        updated, _warnings = ocr_frames([opted_out], "eng", enabled=True)

    assert updated[0].extracted_text == reading
    assert updated[0].redaction is RedactionState.DISABLED


def test_a_tesseract_that_vanished_mid_run_still_only_degrades(tmp_path: Path) -> None:
    """The absent-binary branch inside the invocation, not the discovery check.

    Discovery found a path and the exec then failed, so run_command's real
    E_MISSING_TOOL reaches `_read_text`. That call site does not decide what an
    absent tool costs either - it asks the capability table, which classifies
    image-text extraction optional, so the frame is lost and the run is not.
    """
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake")

    text, frame_warnings = ocr_frame(
        frame, "eng", str(tmp_path / "tesseract-that-is-gone"), preprocess=False
    )

    assert text == ""
    assert [item["code"] for item in frame_warnings] == ["tesseract_not_found"]
    assert EXTERNAL_TOOLS["tesseract"].absence_cost in frame_warnings[0]["message"]
