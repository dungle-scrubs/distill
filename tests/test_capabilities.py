"""Unit tests for the stated optional/required capability table."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from distill.capabilities import (
    EXTERNAL_TOOLS,
    Requirement,
    missing_tool_consequence,
    missing_tool_error,
    missing_tool_warning,
)
from distill.errors import DistillError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "distill"
README = REPO_ROOT / "README.md"


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


def test_the_consequence_of_an_absent_optional_tool_is_its_warning() -> None:
    """The one entry point a call site uses, answering for an optional tool."""
    result = missing_tool_consequence("ocr", "tesseract")

    assert result["code"] == "tesseract_not_found"
    assert EXTERNAL_TOOLS["tesseract"].absence_cost in result["message"]


def test_the_consequence_of_an_absent_required_tool_is_a_fatal_error() -> None:
    """ADR-0002 / R-34: a required capability's absence never returns a warning.

    It raises under the missing-tool code, naming the tool and stating what its
    absence costs, so a run that cannot produce a **bundle** stops at the tool
    rather than at the render that finds nothing to write.
    """
    with pytest.raises(DistillError) as failure:
        missing_tool_consequence("frames", "ffmpeg")

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.stage == "frames"
    assert "ffmpeg" in failure.value.message
    assert EXTERNAL_TOOLS["ffmpeg"].absence_cost in failure.value.message
    assert failure.value.details["requirement"] == "required"


def test_the_fatal_error_carries_the_invocation_that_failed() -> None:
    """The failing invocation's payload survives, so the run stays traceable."""
    cause = DistillError(
        "E_MISSING_TOOL",
        "frames",
        "required tool is not installed: ffmpeg",
        {"argv": ["ffmpeg", "-y"], "tool": "ffmpeg"},
    )

    error = missing_tool_error("frames", "ffmpeg", cause=cause)

    assert error.details["argv"] == ["ffmpeg", "-y"]
    assert error.details["tool"] == "ffmpeg"


def test_an_optional_tool_absence_is_not_allowed_to_end_a_run() -> None:
    """ADR-0002 cuts both ways: an optional capability must not raise."""
    with pytest.raises(ValueError, match="optional capability"):
        missing_tool_error("ocr", "tesseract")


def readme_dependency_rows() -> dict[str, tuple[str, str]]:
    """Tool -> (capability, class), read off the README's four-column table."""
    rows: dict[str, tuple[str, str]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not cells[0].startswith("`"):
            continue
        rows[cells[0].strip("`")] = (cells[1], cells[2])
    return rows


def test_the_readme_states_the_class_the_table_records() -> None:
    """D-022: the docstring's claim about the README is backed by this test.

    Nothing renders the README's system-dependency table from `EXTERNAL_TOOLS` -
    it is prose, written by hand - so the only thing keeping the promise it
    makes about degradation aligned with the code is this assertion. A tool
    reclassified here without the README changing with it fails the suite.
    """
    rows = readme_dependency_rows()

    for name, tool in EXTERNAL_TOOLS.items():
        assert name in rows, f"{name} is classified but the README does not list it"
        capability, requirement = rows[name]
        assert capability == tool.capability, name
        assert requirement == str(tool.requirement), name
