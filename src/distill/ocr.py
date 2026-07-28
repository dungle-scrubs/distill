"""Tesseract OCR adapter for Distill keyframes.

This module owns OCR invocation, tesseract discovery, and per-frame OCR
warnings. It does not own redaction policy, frame selection, or the decision
that image-text extraction is an **optional capability** - that classification
is stated once in `capabilities.py`. It never installs tesseract (R-34), and it
does not own how a subprocess is run: tesseract is invoked through
`run_command`, like every other external tool (R-29).

It does not own the shape of a **frame artifact**. It takes the artifacts
`frame_selection` produced, reads the image each one names, and hands back the
same artifacts carrying what tesseract read. Putting the text on the carrier is
what applies the **redaction** policy to it (R-19, D-019): before M4.4 this
module wrote the reading onto a bare dict, and the raw text was durable in a
**stage result** before any redaction ran (finding 4).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import FrameArtifact
from .capabilities import MISSING_TOOL_CODE, missing_tool_consequence
from .errors import DistillError, WarningRecord, warning
from .progress import ProgressCounter, ProgressReporter
from .run_command import run, silent_tool_timeouts

# Below this mean luminance (0-255) a frame is treated as dark-background and
# inverted before OCR, since Tesseract is tuned for dark text on light.
_DARK_BACKGROUND_THRESHOLD = 110.0
# Upscaling small frames gives Tesseract more pixels per glyph on slide text.
_OCR_UPSCALE = 2
# Tesseract is silent by construction: it prints its reading when it is done and
# says nothing while it works, so the idle clock never resets - see
# `silent_tool_timeouts`, which is why one number governs here. 120 s is the
# budget one frame gets; a lower idle value would not catch a stall, it would
# just cut that budget and lose the frame's **extracted text** on a loaded
# machine, behind an `ocr_failed` warning that says nothing about the reason.
TESSERACT_TIMEOUTS = silent_tool_timeouts(120.0)


def preprocess_for_ocr(path: Path) -> Any:
    """Return a contrast-normalized grayscale image tuned for slide OCR.

    Tesseract struggles with the low-contrast, white-on-dark slides common in
    conference recordings. Converting to grayscale, stretching contrast,
    inverting dark backgrounds, and upscaling recovers text that the raw frame
    loses. Returns ``None`` (so the caller falls back to the path) if Pillow is
    unavailable or the image cannot be processed.
    """
    try:
        from PIL import Image, ImageOps, ImageStat
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        if ImageStat.Stat(gray).mean[0] < _DARK_BACKGROUND_THRESHOLD:
            gray = ImageOps.invert(gray)
        if _OCR_UPSCALE > 1:
            gray = gray.resize(
                (gray.width * _OCR_UPSCALE, gray.height * _OCR_UPSCALE),
                Image.Resampling.LANCZOS,
            )
        return gray
    except (OSError, ValueError):
        return None


def find_tesseract_command() -> str | None:
    command = shutil.which("tesseract")
    if command:
        return command
    for candidate in (
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def ensure_tesseract_available() -> dict[str, str] | None:
    """Report whether tesseract is present, without ever installing it.

    Per R-34 Distill installs nothing on a user's machine: no package manager
    is consulted and no process is spawned here. Image-text extraction is an
    **optional capability** (see `capabilities.EXTERNAL_TOOLS`), so an absent
    binary yields a **warning** and the run continues with the rest of the
    bundle intact.
    """
    if find_tesseract_command():
        return None
    return missing_tool_consequence("ocr", "tesseract")


def ocr_frame(
    path: Path,
    language: str,
    tesseract_cmd: str | None = None,
    preprocess: bool = True,
) -> tuple[str, list[dict[str, str]]]:
    """Read one keyframe's **extracted text**, with every **warning** it cost.

    Returns the text and the warnings together rather than a single warning,
    because an invocation can both succeed and lose output: `run_command`
    records truncated capture as a warning (R-33), and a reading that was
    truncated is still a reading. Dropping those here would put text in the
    **bundle** that is quietly less than what tesseract read.
    """
    command = tesseract_cmd or find_tesseract_command()
    if command is None:
        return "", [missing_tool_consequence("ocr", "tesseract")]

    # Preprocessing recovers low-contrast slide text. The processed frame is
    # written alongside the source - a directory we just read from, so it is
    # readable - rather than through ``$TMPDIR``, which a sandboxed tesseract
    # may not be able to open. Fall back to the original frame when
    # preprocessing is unavailable or its OCR failed.
    #
    # A failed first attempt's warnings are deliberately dropped on that
    # fallback: the frame is about to be read again, and the second reading is
    # the one that describes what the **bundle** actually holds. Reporting the
    # discarded attempt would put an `ocr_failed` warning on a frame whose text
    # arrived. A *successful* first attempt's warnings are kept, because that
    # reading is the one the bundle keeps.
    if preprocess:
        processed = preprocess_for_ocr(path)
        if processed is not None:
            text, prep_warnings = _ocr_processed_image(command, processed, path, language)
            if text is not None:
                return text, prep_warnings

    text, warnings = _read_text(command, path, language, source=path)
    return text or "", warnings


def _read_text(
    command: str,
    image: Path,
    language: str,
    *,
    source: Path,
) -> tuple[str | None, list[dict[str, str]]]:
    """Run tesseract over one image and return the text it printed.

    ``stdout`` as the output base is what makes tesseract print its reading
    instead of writing a sibling ``.txt``. Every failure degrades: one frame
    without **extracted text** reduces a **bundle**, it does not end the run
    (ADR-0002), and an absent binary is handed to the capability table, which
    states that image-text extraction is optional and returns its warning.

    ``None`` for the text means the attempt failed, which is what lets the
    caller fall back to the unprocessed frame; an empty string is a successful
    reading of a frame with no text on it.
    """
    try:
        result = run(
            [command, str(image), "stdout", "-l", language],
            stage="ocr",
            total_timeout_sec=TESSERACT_TIMEOUTS.total_sec,
            idle_timeout_sec=TESSERACT_TIMEOUTS.idle_sec,
        )
    except DistillError as exc:
        if exc.code == MISSING_TOOL_CODE:
            return None, [missing_tool_consequence("ocr", "tesseract", cause=exc)]
        return None, [warning("ocr", "ocr_failed", _failure_reason(source, exc))]
    return result.stdout.strip(), list(result.warnings)


def _failure_reason(source: Path, exc: DistillError) -> str:
    """Why this frame has no **extracted text**, in tesseract's own words.

    `exc.message` alone is the generic "command failed: <path>", which tells a
    reader of a degraded run nothing they can act on. The tool's stderr tail
    carries the reason it actually gave - "Failed loading language 'zzz'" - so
    it is appended when there is one.
    """
    reason = f"OCR failed for {source.name}: {exc.message}"
    tail = " ".join(str(exc.details.get("stderr_tail", "")).split())
    return f"{reason}: {tail}" if tail else reason


def _ocr_processed_image(
    command: str, processed: Any, source: Path, language: str
) -> tuple[str | None, list[dict[str, str]]]:
    """OCR a preprocessed PIL image via a temp file next to ``source``."""
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        suffix=".png", prefix=f".{source.stem}.ocr.", dir=str(source.parent), delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        try:
            processed.save(temp_path)
        except (OSError, ValueError) as exc:
            return None, [
                warning("ocr", "ocr_failed", f"OCR failed for {source.name}: {exc}")
            ]
        return _read_text(command, temp_path, language, source=source)
    finally:
        temp_path.unlink(missing_ok=True)


def ocr_frames(
    frames: list[FrameArtifact],
    language: str,
    enabled: bool,
    progress: ProgressCounter | ProgressReporter | None = None,
    preprocess: bool = True,
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    """Read every **keyframe**'s image text onto the artifact that names it.

    The text goes onto the carrier rather than beside it, which is what puts it
    through the run's **redaction** policy before anything can write it (R-19).
    A disabled pass records the empty reading the same way, so a frame's
    `extracted_text` means "nothing was read" rather than "nobody looked".
    """
    warnings: list[WarningRecord] = []
    updated: list[FrameArtifact] = []
    tesseract_cmd = None
    if enabled:
        if not find_tesseract_command() and isinstance(progress, ProgressReporter):
            progress.update(
                "ocr",
                status="running",
                detail={"dependency": "tesseract", "action": "ensure_available"},
            )
        tesseract_warning = ensure_tesseract_available()
        if tesseract_warning:
            warnings.append(tesseract_warning)
        else:
            tesseract_cmd = find_tesseract_command()
            if tesseract_cmd and isinstance(progress, ProgressReporter):
                progress.update(
                    "ocr",
                    status="running",
                    detail={"dependency": "tesseract", "path": tesseract_cmd},
                )
    for frame in frames:
        text = ""
        if enabled and tesseract_cmd:
            text, frame_warnings = ocr_frame(
                Path(frame.path),
                language,
                tesseract_cmd,
                preprocess,
            )
            warnings.extend(frame_warnings)
        read, carrier_warnings = frame.with_extracted_text(text)
        warnings.extend(carrier_warnings)
        if isinstance(progress, ProgressReporter):
            progress.update(
                "ocr",
                percent=(len(updated) + 1) / max(1, len(frames)) * 100,
                detail={"frame": len(updated) + 1, "frames": len(frames)},
            )
        elif progress:
            progress.increment()
        updated.append(read)
    if isinstance(progress, ProgressReporter):
        progress.complete("ocr", detail={"frames": len(frames)})
    return updated, warnings
