"""Tesseract OCR adapter for Saccade keyframes.

This module owns OCR invocation and per-frame OCR warnings. It does not own
redaction policy or frame selection.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from .errors import warning
from .progress import ProgressCounter, ProgressReporter


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
) -> tuple[str, dict[str, str] | None]:
    try:
        import pytesseract
    except ImportError:
        return "", warning("ocr", "tesseract_python_missing", "pytesseract is not installed")

    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return pytesseract.image_to_string(str(path), lang=language).strip(), None
    except Exception as exc:
        return "", warning("ocr", "ocr_failed", f"OCR failed for {path.name}: {exc}")


def ocr_frames(
    frames: list[dict],
    language: str,
    enabled: bool,
    progress: ProgressCounter | ProgressReporter | None = None,
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
