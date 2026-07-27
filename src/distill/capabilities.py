"""The stated optional/required classification of every external tool Distill runs.

This module owns one table: for each external tool Distill invokes, whether it
carries an **optional capability** or a **required capability**, when it is
invoked, and what its absence costs a **bundle**. Per ADR-0002 the two classes
have different consequences - an absent optional capability is a **degradation**
that records a **warning** and continues, an absent required capability is a
**fatal error** - so the classification must be stated here rather than left to
emerge from whichever call site happens to raise. It is also the single source
of truth the README and AGENTS.md render, so the promise those documents make
about degradation cannot drift from the code.

It does not own tool discovery (each adapter finds its own binary: `ocr.py` for
tesseract, `source.py` for ffprobe and yt-dlp), invocation or its failure
taxonomy (`run_command.py`, whose error table raises `E_MISSING_TOOL` for a
required tool that is not installed), or the shape of a warning record
(`errors.py`). This module states which class a tool is in and supplies the
degradation warning for the optional ones; it never installs anything, and
nothing here decides whether a *particular* run needed the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import warning


class Requirement(StrEnum):
    """Which class of capability a tool carries, in the ADR-0002 sense."""

    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True)
class ExternalTool:
    """One external tool and the consequence of its absence.

    `absence_cost` is written as a sentence fragment so it reads correctly both
    in a rendered table and appended to a degradation warning.
    """

    name: str
    capability: str
    requirement: Requirement
    invoked_when: str
    absence_cost: str

    @property
    def is_optional(self) -> bool:
        return self.requirement is Requirement.OPTIONAL

    @property
    def warning_code(self) -> str:
        """The snake_case warning code for this tool being absent."""
        return f"{self.name.replace('-', '_')}_not_found"


EXTERNAL_TOOLS: dict[str, ExternalTool] = {
    "ffmpeg": ExternalTool(
        name="ffmpeg",
        capability="audio extraction and keyframe extraction",
        requirement=Requirement.REQUIRED,
        invoked_when="every run",
        absence_cost=(
            "no audio can be extracted and no keyframe can be captured, which "
            "leaves a generation with neither a transcript nor frame artifacts "
            "and no usable bundle to publish"
        ),
    ),
    "ffprobe": ExternalTool(
        name="ffprobe",
        capability="source duration probing",
        requirement=Requirement.REQUIRED,
        invoked_when="every run",
        absence_cost=(
            "the source's duration cannot be read, so keyframe timestamps and "
            "the duration cap have nothing to work from and the run ends before "
            "any stage produces output"
        ),
    ),
    "yt-dlp": ExternalTool(
        name="yt-dlp",
        capability="YouTube source acquisition and metadata",
        requirement=Requirement.REQUIRED,
        invoked_when="a YouTube source; never for a local file",
        absence_cost=(
            "the source cannot be acquired at all, so a YouTube run has nothing "
            "to process"
        ),
    ),
    "tesseract": ExternalTool(
        name="tesseract",
        capability="image-text extraction from keyframes",
        requirement=Requirement.OPTIONAL,
        invoked_when="every run with OCR enabled",
        absence_cost=(
            "keyframes contribute no extracted text, so interpretations cannot "
            "be corroborated and grounding falls back to the vision model alone; "
            "the transcript, keyframes and render are unaffected"
        ),
    ),
}


def missing_tool_warning(stage: str, tool_name: str) -> dict[str, str]:
    """The degradation **warning** for an absent **optional capability**.

    Raises `ValueError` for a tool classified required: a required tool's
    absence is a **fatal error**, and letting it degrade here would be the
    silent-thin-bundle failure ADR-0002 warns about.
    """
    tool = EXTERNAL_TOOLS[tool_name]
    if not tool.is_optional:
        raise ValueError(
            f"{tool.name} is a required capability; its absence is a fatal error, "
            "not a degradation"
        )
    return warning(
        stage,
        tool.warning_code,
        f"{tool.name} is not installed or not on PATH; {tool.absence_cost}",
    )
