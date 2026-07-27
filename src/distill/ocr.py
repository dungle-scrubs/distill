"""Tesseract OCR adapter for Distill keyframes.

This module owns OCR invocation, tesseract discovery, and per-frame OCR
warnings. It does not own redaction policy, frame selection, or the decision
that image-text extraction is an **optional capability** - that classification
is stated once in `capabilities.py`. It never installs tesseract (R-34), and it
does not own how a subprocess is run: tesseract is invoked through
`run_command`, like every other external tool (R-29).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .capabilities import missing_tool_warning
from .errors import DistillError, warning
from .progress import ProgressCounter, ProgressReporter
from .run_command import CommandTimeouts, run

# Below this mean luminance (0-255) a frame is treated as dark-background and
# inverted before OCR, since Tesseract is tuned for dark text on light.
_DARK_BACKGROUND_THRESHOLD = 110.0
# Upscaling small frames gives Tesseract more pixels per glyph on slide text.
_OCR_UPSCALE = 2
# One frame at a time, so both limits are short: tesseract that has said nothing
# for half a minute on a single image is stuck, not slow.
TESSERACT_TIMEOUTS = CommandTimeouts(total_sec=120.0, idle_sec=30.0)


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
    return missing_tool_warning("ocr", "tesseract")


def ocr_frame(
    path: Path,
    language: str,
    tesseract_cmd: str | None = None,
    preprocess: bool = True,
) -> tuple[str, dict[str, str] | None]:
    command = tesseract_cmd or find_tesseract_command()
    if command is None:
        return "", missing_tool_warning("ocr", "tesseract")

    # Preprocessing recovers low-contrast slide text. The processed frame is
    # written alongside the source - a directory we just read from, so it is
    # readable - rather than through ``$TMPDIR``, which a sandboxed tesseract
    # may not be able to open. Fall back to the original frame when
    # preprocessing is unavailable or its OCR failed.
    if preprocess:
        processed = preprocess_for_ocr(path)
        if processed is not None:
            text, prep_warning = _ocr_processed_image(command, processed, path, language)
            if prep_warning is None:
                return text, None

    return _read_text(command, path, language, source=path)


def _read_text(
    command: str,
    image: Path,
    language: str,
    *,
    source: Path,
) -> tuple[str, dict[str, str] | None]:
    """Run tesseract over one image and return the text it printed.

    ``stdout`` as the output base is what makes tesseract print its reading
    instead of writing a sibling ``.txt``. Every failure degrades: one frame
    without **extracted text** reduces a **bundle**, it does not end the run
    (ADR-0002), and an absent binary reports the capability table's warning
    rather than a bespoke one.
    """
    try:
        result = run(
            [command, str(image), "stdout", "-l", language],
            stage="ocr",
            total_timeout_sec=TESSERACT_TIMEOUTS.total_sec,
            idle_timeout_sec=TESSERACT_TIMEOUTS.idle_sec,
        )
    except DistillError as exc:
        if exc.code == "E_MISSING_TOOL":
            return "", missing_tool_warning("ocr", "tesseract")
        return "", warning("ocr", "ocr_failed", f"OCR failed for {source.name}: {exc.message}")
    return result.stdout.strip(), None


def _ocr_processed_image(
    command: str, processed: Any, source: Path, language: str
) -> tuple[str, dict[str, str] | None]:
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
            return "", warning("ocr", "ocr_failed", f"OCR failed for {source.name}: {exc}")
        return _read_text(command, temp_path, language, source=source)
    finally:
        temp_path.unlink(missing_ok=True)


def ocr_frames(
    frames: list[dict],
    language: str,
    enabled: bool,
    progress: ProgressCounter | ProgressReporter | None = None,
    preprocess: bool = True,
) -> tuple[list[dict], list[dict[str, str]]]:
    warnings: list[dict[str, str]] = []
    updated: list[dict] = []
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
        copied = dict(frame)
        if enabled and tesseract_cmd:
            text, frame_warning = ocr_frame(
                Path(str(copied["path"])),
                language,
                tesseract_cmd,
                preprocess,
            )
            copied["ocr_text"] = text
            if frame_warning:
                warnings.append(frame_warning)
        else:
            copied["ocr_text"] = ""
        if isinstance(progress, ProgressReporter):
            progress.update(
                "ocr",
                percent=(len(updated) + 1) / max(1, len(frames)) * 100,
                detail={"frame": len(updated) + 1, "frames": len(frames)},
            )
        elif progress:
            progress.increment()
        updated.append(copied)
    if isinstance(progress, ProgressReporter):
        progress.complete("ocr", detail={"frames": len(frames)})
    return updated, warnings
