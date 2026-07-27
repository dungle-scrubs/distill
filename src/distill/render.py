"""Markdown rendering and no-content checks for Distill bundles.

This module owns deterministic `video.md` assembly. It does not write manifests
or run extraction stages.

It reads carriers rather than dicts (R-19, M4.4). What a **frame artifact**
holds is `artifacts.FrameArtifact`'s to say, what an **interpretation** holds is
`artifacts.Interpretation`'s, and what a **grounding** holds is
`grounding.GroundingAssessment`'s - so this module asks each of them rather than
restating their field names, and a field renamed at its source is a type error
here instead of a section that silently stops rendering.

What it still spells out itself: the shape of a **transcript** segment, which is
`transcript.py`'s, and every heading, bullet and fence in the document, which is
this module's alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .artifacts import FrameArtifact, Interpretation, Transcript, serialize
from .errors import DistillError
from .grounding import GroundingAssessment

MIN_TRANSCRIPT_CHARS = 3

UNVERIFIED_CAVEAT = (
    "On-screen text may be unreadable; treat the interpretation below as unverified."
)
NO_OUTPUT_CAVEAT = "The vision model returned no usable output for this frame."


def transcript_is_empty(transcript: Transcript | None) -> bool:
    """Return true when transcript text has fewer than 3 non-space characters."""
    if transcript is None:
        return True
    text = "".join(str(segment.get("text", "")).strip() for segment in transcript.segments)
    return len(text) < MIN_TRANSCRIPT_CHARS


def frames_are_useless(frames: list[FrameArtifact]) -> bool:
    """Return true when no frame names an image a reader could be shown.

    A **keyframe** whose extraction failed never becomes a **frame artifact** -
    `select_keyframes` drops it - so the only way a frame is useless here is
    having no path into the **generation** to point at.
    """
    if not frames:
        return True
    return all(not frame.relative_path.strip() for frame in frames)


def ensure_content(transcript: Transcript | None, frames: list[FrameArtifact]) -> None:
    if transcript_is_empty(transcript) and frames_are_useless(frames):
        raise DistillError(
            "E_NO_CONTENT",
            "render",
            "video produced no transcript text or usable frames",
        )


def _require_redaction_policy(transcript: Transcript | None, frames: list[FrameArtifact]) -> None:
    """Refuse to render a carrier whose **redaction** policy has not been applied.

    R-20 names a **render** as a sink beside a **generation**, so the check that
    guards the one has to guard the other. `serialize` is where it is written,
    and it is asked here rather than reimplemented - a second copy of "has the
    policy run" is a second answer that can drift from the first.

    Nothing else is taken from the serialized documents. The render reads the
    carriers, because what it needs is their typed fields; this is the check and
    only the check.
    """
    for carrier in (*frames, *(() if transcript is None else (transcript,))):
        serialize(carrier)


def render_markdown(
    source_label: str,
    duration_sec: float,
    transcript: Transcript | None,
    frames: list[FrameArtifact],
    warnings: list[dict[str, str]],
    related_links: list[dict[str, str]] | None = None,
) -> str:
    _require_redaction_policy(transcript, frames)
    ensure_content(transcript, frames)
    lines = [
        "# Video Bundle",
        "",
        f"- Source: `{source_label}`",
        f"- Duration: {duration_sec:.3f}s",
        f"- Frames: {len(frames)}",
        f"- Transcript: {'yes' if not transcript_is_empty(transcript) else 'no'}",
        f"- Warnings: {len(warnings)}",
        "",
    ]
    if related_links:
        lines.extend(["## Related links", ""])
        for link in related_links:
            label = str(link.get("label", "")).strip() or str(link.get("url", "")).strip()
            url = str(link.get("url", "")).strip()
            reason = str(link.get("reason", "")).strip()
            if not url:
                continue
            suffix = f" ({reason})" if reason else ""
            lines.append(f"- [{label}]({url}){suffix}")
        lines.append("")
    segments = list(transcript.segments) if transcript else []
    frame_index = 0
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        while frame_index < len(frames) and frames[frame_index].timestamp_sec < start:
            lines.extend(_frame_lines(frames[frame_index]))
            frame_index += 1
        segment_frames: list[FrameArtifact] = []
        while frame_index < len(frames) and frames[frame_index].timestamp_sec <= end:
            segment_frames.append(frames[frame_index])
            frame_index += 1
        lines.extend(_segment_lines(segment, segment_frames))
    while frame_index < len(frames):
        lines.extend(_frame_lines(frames[frame_index]))
        frame_index += 1
    return "\n".join(lines).rstrip() + "\n"


def _segment_lines(segment: Mapping[str, Any], frames: list[FrameArtifact]) -> list[str]:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    lines = [f"## {format_timestamp(start)} - {format_timestamp(end)}", ""]
    words = segment.get("words", [])
    if not words:
        lines.extend([str(segment.get("text", "")).strip(), ""])
        for frame in frames:
            lines.extend(_frame_lines(frame))
        return lines

    word_index = 0
    for frame in frames:
        chunk: list[str] = []
        while word_index < len(words) and float(words[word_index]["end"]) <= frame.timestamp_sec:
            chunk.append(str(words[word_index].get("word", "")).strip())
            word_index += 1
        if chunk:
            lines.extend([" ".join(chunk), ""])
        lines.extend(_frame_lines(frame))
    remaining = [
        str(word.get("word", "")).strip()
        for word in words[word_index:]
        if str(word.get("word", "")).strip()
    ]
    if remaining:
        lines.extend([" ".join(remaining), ""])
    return lines


def _frame_lines(frame: FrameArtifact) -> list[str]:
    timestamp = format_timestamp(frame.timestamp_sec)
    lines = [
        f"## Frame {frame.index} - {timestamp}",
        "",
        f"![Frame {frame.index}]({frame.relative_path})",
        "",
    ]
    reading = frame.reading
    assessment = GroundingAssessment.from_document(frame.grounding)
    if reading is not None:
        lines.extend(["Visual interpretation:", ""])
        lines.extend(_low_confidence_lines(assessment, UNVERIFIED_CAVEAT))
        lines.extend(_reading_lines(reading))
        lines.append("")
        if reading.verbatim_text.strip():
            lines.extend(
                ["Verbatim slide text:", "", "```text", reading.verbatim_text.strip(), "```", ""]
            )
    elif assessment is not None and assessment.is_low_confidence:
        lines.extend(["Visual interpretation:", ""])
        lines.extend(_low_confidence_lines(assessment, NO_OUTPUT_CAVEAT))
    if frame.extracted_text.strip():
        lines.extend(["OCR:", "", "```text", frame.extracted_text.strip(), "```", ""])
    return lines


def _low_confidence_lines(assessment: GroundingAssessment | None, caveat: str) -> list[str]:
    """The banner a **grounding** that is not grounded puts above a reading.

    Absent for a grounded frame, and absent for a frame nobody assessed: a
    banner that appeared whenever the assessment was missing would report low
    confidence for every frame produced before the vision pass ran.
    """
    if assessment is None or not assessment.is_low_confidence:
        return []
    level = assessment.level.strip() or "low"
    return [
        f"> ⚠ Low-confidence frame ({level}): {assessment.reason.strip()}. {caveat}",
        "",
    ]


def _reading_lines(reading: Interpretation) -> list[str]:
    """One bullet per field of an **interpretation** the model filled in."""
    bullets: list[tuple[str, str]] = [
        ("Summary", reading.visual_summary.strip()),
        ("Detected elements", ", ".join(reading.detected_elements)),
        ("Interpretation", reading.interpretation.strip()),
        ("Text confidence", reading.text_confidence.strip()),
        ("Uncertainty", reading.uncertainty.strip()),
    ]
    return [f"- {label}: {value}" for label, value in bullets if value]


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    millis = int(round((seconds - total) * 1000))
    minutes, sec = divmod(total, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"


__all__ = [
    "MIN_TRANSCRIPT_CHARS",
    "NO_OUTPUT_CAVEAT",
    "UNVERIFIED_CAVEAT",
    "ensure_content",
    "format_timestamp",
    "frames_are_useless",
    "render_markdown",
    "transcript_is_empty",
]
