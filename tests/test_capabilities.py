"""Unit tests for the stated optional/required capability table."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from distill.capabilities import EXTERNAL_TOOLS, Requirement, missing_tool_warning

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "distill"


def test_every_external_tool_distill_invokes_is_classified() -> None:
    assert set(EXTERNAL_TOOLS) == {"ffmpeg", "ffprobe", "tesseract", "yt-dlp"}
    assert EXTERNAL_TOOLS["tesseract"].requirement is Requirement.OPTIONAL
    assert all(
        EXTERNAL_TOOLS[name].requirement is Requirement.REQUIRED
        for name in ("ffmpeg", "ffprobe", "yt-dlp")
    )


def test_every_classification_states_what_its_absence_costs() -> None:
    for name, tool in EXTERNAL_TOOLS.items():
        assert tool.name == name
        assert tool.capability
        assert tool.invoked_when
        assert len(tool.absence_cost.split()) >= 5, name


def test_no_source_module_names_a_tool_the_table_omits() -> None:
    """A tool invoked but unclassified is the gap D-010 forbids."""
    known = set(EXTERNAL_TOOLS)
    invoked: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
                if candidate in {"ffmpeg", "ffprobe", "tesseract", "yt-dlp", "brew"}:
                    invoked.add(candidate)
    assert invoked <= known, f"invoked but unclassified: {sorted(invoked - known)}"


def test_optional_tool_absence_degrades_with_a_warning() -> None:
    result = missing_tool_warning("ocr", "tesseract")

    assert result["stage"] == "ocr"
    assert result["code"] == "tesseract_not_found"
    assert result["message"].startswith("tesseract is not installed or not on PATH; ")


def test_required_tool_absence_is_not_allowed_to_degrade() -> None:
    """ADR-0002 cuts both ways: a required capability must not warn and continue."""
    with pytest.raises(ValueError, match="required capability"):
        missing_tool_warning("source", "ffprobe")


def test_warning_code_is_snake_case_for_a_hyphenated_tool() -> None:
    assert EXTERNAL_TOOLS["yt-dlp"].warning_code == "yt_dlp_not_found"
