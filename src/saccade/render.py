"""Markdown rendering and no-content checks for Saccade bundles.

This module owns deterministic `video.md` assembly. It does not write manifests
or run extraction stages.
"""

from __future__ import annotations

from typing import Any

from .errors import SaccadeError

MIN_TRANSCRIPT_CHARS = 3


def transcript_is_empty(transcript: dict[str, Any] | None) -> bool:
    """Return true when transcript text has fewer than 3 non-space characters."""
    if not transcript:
        return True
    text = "".join(
        str(segment.get("text", "")).strip() for segment in transcript.get("segments", [])
    )
    return len(text) < MIN_TRANSCRIPT_CHARS


def frames_are_useless(frames: list[dict]) -> bool:
    """Return true when every frame is explicitly blank or lacks an image path."""
    if not frames:
        return True
    return all(
        bool(frame.get("blank")) or not str(frame.get("relative_path", "")).strip()
        for frame in frames
    )


def ensure_content(transcript: dict[str, Any] | None, frames: list[dict]) -> None:
    if transcript_is_empty(transcript) and frames_are_useless(frames):
        raise SaccadeError(
            "E_NO_CONTENT",
            "render",
            "video produced no transcript text or usable frames",
        )


def render_markdown(
    source_label: str,
    duration_sec: float,
    transcript: dict[str, Any] | None,
    frames: list[dict],
    warnings: list[dict[str, str]],
) -> str:
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
    segments = transcript.get("segments", []) if transcript else []
    frame_index = 0
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        while frame_index < len(frames) and float(frames[frame_index]["timestamp_sec"]) < start:
            lines.extend(_frame_lines(frames[frame_index]))
            frame_index += 1
        segment_frames: list[dict] = []
        while frame_index < len(frames) and float(frames[frame_index]["timestamp_sec"]) <= end:
            segment_frames.append(frames[frame_index])
            frame_index += 1
        lines.extend(_segment_lines(segment, segment_frames))
    while frame_index < len(frames):
        lines.extend(_frame_lines(frames[frame_index]))
        frame_index += 1
    return "\n".join(lines).rstrip() + "\n"


def _segment_lines(segment: dict[str, Any], frames: list[dict]) -> list[str]:
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
        frame_timestamp = float(frame["timestamp_sec"])
        chunk: list[str] = []
        while word_index < len(words) and float(words[word_index]["end"]) <= frame_timestamp:
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


def _frame_lines(frame: dict) -> list[str]:
    timestamp = format_timestamp(float(frame["timestamp_sec"]))
    lines = [
        f"## Frame {frame['index']} - {timestamp}",
        "",
        f"![Frame {frame['index']}]({frame['relative_path']})",
        "",
    ]
    ocr_text = str(frame.get("ocr_text", "")).strip()
    visual_interpretation = frame.get("visual_interpretation")
    if isinstance(visual_interpretation, dict):
        lines.extend(["Visual interpretation:", ""])
        summary = str(visual_interpretation.get("visual_summary", "")).strip()
        interpretation = str(visual_interpretation.get("interpretation", "")).strip()
        uncertainty = str(visual_interpretation.get("uncertainty", "")).strip()
        elements = visual_interpretation.get("detected_elements", [])
        if summary:
            lines.extend([f"- Summary: {summary}"])
        if isinstance(elements, list) and elements:
            lines.extend([f"- Detected elements: {', '.join(str(item) for item in elements)}"])
        if interpretation:
            lines.extend([f"- Interpretation: {interpretation}"])
        if uncertainty:
            lines.extend([f"- Uncertainty: {uncertainty}"])
        lines.append("")
    if ocr_text:
        lines.extend(["OCR:", "", "```text", ocr_text, "```", ""])
    return lines


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    millis = int(round((seconds - total) * 1000))
    minutes, sec = divmod(total, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"
