"""The stated optional/required classification of every external tool Distill runs.

This module owns one table: for each external tool Distill invokes, whether it
carries an **optional capability** or a **required capability**, when it is
invoked, and what its absence costs a **bundle**. Per ADR-0002 the two classes
have different consequences - an absent optional capability is a **degradation**
that records a **warning** and continues, an absent required capability is a
**fatal error** - so the classification must be stated here rather than left to
emerge from whichever call site happens to raise.

A call site that caught a missing tool therefore asks `missing_tool_consequence`
what that absence costs instead of deciding for itself: the table is what
answers, so no site can quietly degrade a required capability and leave the run
to die later under an unrelated code.

The README's system-dependency table is the same statement in prose. Nothing
renders it - it is written by hand - so
`tests/test_capabilities.py::test_the_readme_states_the_class_the_table_records`
pins it against this table instead, and a classification changed here without
the README changing with it fails the suite.

It does not own tool discovery (each adapter finds its own binary: `ocr.py` for
tesseract, `source.py` for ffprobe and yt-dlp), invocation or its failure
taxonomy (`run_command.py`, whose error table raises `E_MISSING_TOOL` for a
required tool that is not installed), or the shape of a warning record
(`errors.py`). This module states which class a tool is in and supplies the
consequence of its absence; it never installs anything, and nothing here decides
whether a *particular* run needed the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import DistillError, WarningRecord, warning
from .run_command import ERROR_CODES, FailureKind

# The code `run_command` raises when a tool is not installed. Named here so a
# call site can recognize that failure and hand it to the table, rather than
# matching the string and then hand-coding what to do about it.
MISSING_TOOL_CODE = ERROR_CODES[FailureKind.MISSING_TOOL]


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


def _absence_message(tool: ExternalTool) -> str:
    """One sentence for an absent tool, whichever class it is in.

    The wording is shared so a **degradation** and a **fatal error** describe the
    same absence the same way: what is missing, and what its absence costs.
    """
    return f"{tool.name} is not installed or not on PATH; {tool.absence_cost}"


def missing_tool_warning(stage: str, tool_name: str) -> WarningRecord:
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
    return warning(stage, tool.warning_code, _absence_message(tool))


def missing_tool_error(
    stage: str, tool_name: str, *, cause: DistillError | None = None
) -> DistillError:
    """The **fatal error** for an absent **required capability** (ADR-0002, R-34).

    It names the tool and states what its absence costs, so a run that cannot
    produce a **bundle** stops here saying which tool to install - rather than
    degrading, doing the rest of the work, and failing at render under a code
    about missing content.

    Raises `ValueError` for a tool classified optional: that is ADR-0002 read
    backwards, and an optional absence must not be able to end a run.
    """
    tool = EXTERNAL_TOOLS[tool_name]
    if tool.is_optional:
        raise ValueError(
            f"{tool.name} is an optional capability; its absence is a degradation, "
            "not a fatal error"
        )
    details: dict[str, Any] = dict(cause.details) if cause is not None else {}
    details["tool"] = tool.name
    details["requirement"] = str(tool.requirement)
    return DistillError(MISSING_TOOL_CODE, stage, _absence_message(tool), details)


def missing_tool_consequence(
    stage: str, tool_name: str, *, cause: DistillError | None = None
) -> WarningRecord:
    """What an absent `tool_name` costs this run, decided by the table.

    The single entry point for a call site that has just caught
    `MISSING_TOOL_CODE`. An **optional capability** returns its **degradation**
    warning for the caller to record; a **required capability** never returns -
    it raises, because ADR-0002 makes its absence a **fatal error** and R-34
    admits no third answer.

    Call sites route through here rather than deciding per site, so which class
    a tool is in is read off the table at the moment it matters instead of being
    re-derived - correctly or not - at each place the tool is run.
    """
    if EXTERNAL_TOOLS[tool_name].is_optional:
        return missing_tool_warning(stage, tool_name)
    raise missing_tool_error(stage, tool_name, cause=cause) from cause
