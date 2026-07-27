"""Tesseract OCR adapter for Distill keyframes.

This module owns OCR invocation, tesseract discovery, and per-frame OCR
warnings. It does not own redaction policy, frame selection, or the decision
that image-text extraction is an **optional capability** - that classification
is stated once in `capabilities.py`. It never installs tesseract (R-34).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .capabilities import missing_tool_warning
from .errors import warning
from .progress import ProgressCounter, ProgressReporter

# Below this mean luminance (0-255) a frame is treated as dark-background and
# inverted before OCR, since Tesseract is tuned for dark text on light.
_DARK_BACKGROUND_THRESHOLD = 110.0
# Upscaling small frames gives Tesseract more pixels per glyph on slide text.
_OCR_UPSCALE = 2


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
    try:
        import pytesseract
    except ImportError:
        return "", warning("ocr", "tesseract_python_missing", "pytesseract is not installed")

    if tesseract_cmd:
        setattr(pytesseract.pytesseract, "tesseract_cmd", tesseract_cmd)  # noqa: B010

    # Preprocessing recovers low-contrast slide text, but pytesseract spools a
    # PIL image through ``$TMPDIR``; in sandboxed/restricted environments the
    # tesseract subprocess cannot read that temp path and OCR silently fails.
    # Write the processed frame alongside the source (a directory we just read
    # from, so it is readable) and OCR by path instead of handing over a PIL
    # object. Fall back to the original frame if preprocessing is unavailable.
    if preprocess:
        processed = preprocess_for_ocr(path)
        if processed is not None:
            text, prep_warning = _ocr_processed_image(pytesseract, processed, path, language)
            if prep_warning is None:
                return text, None
            # Preprocessed OCR failed; recover what we can from the raw frame.

    try:
        return pytesseract.image_to_string(str(path), lang=language).strip(), None
    except Exception as exc:
        return "", warning("ocr", "ocr_failed", f"OCR failed for {path.name}: {exc}")


def _ocr_processed_image(
    pytesseract: Any, processed: Any, source: Path, language: str
) -> tuple[str, dict[str, str] | None]:
    """OCR a preprocessed PIL image via a temp file next to ``source``."""
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        suffix=".png", prefix=f".{source.stem}.ocr.", dir=str(source.parent), delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        processed.save(temp_path)
        return pytesseract.image_to_string(str(temp_path), lang=language).strip(), None
    except Exception as exc:
        return "", warning("ocr", "ocr_failed", f"OCR failed for {source.name}: {exc}")
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
