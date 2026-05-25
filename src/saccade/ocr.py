"""Tesseract OCR adapter for Saccade keyframes.

This module owns OCR invocation and per-frame OCR warnings. It does not own
redaction policy or frame selection.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
    if find_tesseract_command():
        return None
    if platform.system() != "Darwin":
        return warning(
            "ocr",
            "tesseract_not_found",
            "tesseract is not installed or not on PATH",
        )
    brew = shutil.which("brew")
    if not brew:
        return warning(
            "ocr",
            "tesseract_brew_missing",
            "tesseract is missing and Homebrew is not available to install it",
        )
    try:
        result = subprocess.run(
            [brew, "install", "tesseract"],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return warning(
            "ocr",
            "tesseract_install_failed",
            f"failed to install tesseract with Homebrew: {exc}",
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        return warning(
            "ocr",
            "tesseract_install_failed",
            f"failed to install tesseract with Homebrew{suffix}",
        )
    if find_tesseract_command():
        return None
    return warning(
        "ocr",
        "tesseract_not_found",
        "Homebrew installed tesseract, but the tesseract binary was not found",
    )


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

    target: Any = str(path)
    if preprocess:
        processed = preprocess_for_ocr(path)
        if processed is not None:
            target = processed
    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return pytesseract.image_to_string(target, lang=language).strip(), None
    except Exception as exc:
        return "", warning("ocr", "ocr_failed", f"OCR failed for {path.name}: {exc}")


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
