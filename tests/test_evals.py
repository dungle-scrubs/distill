"""Eval-set regression guard.

These are not unit tests — they score Saccade against the human-verified eval
frames in tests/evals/ and assert quality floors. They are marked ``eval`` and
skip when no case is verified yet or when Tesseract is missing, so the default
suite stays green until a human has confirmed ground truth.

Run explicitly:  uv run pytest -m eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_ROOT = Path(__file__).resolve().parent / "evals"
sys.path.insert(0, str(EVAL_ROOT))

pytestmark = pytest.mark.eval


def _scorer():
    from saccade.ocr import find_tesseract_command

    if not find_tesseract_command():
        pytest.skip("tesseract not installed")
    import score  # type: ignore[import-not-found]

    return score


def test_ocr_recovers_clean_frames() -> None:
    """Preprocessed OCR should read cleanly-legible frames with low error."""
    score = _scorer()
    results = [r for r in score.evaluate(with_vision=False) if r.legibility == "clean"]
    if not results:
        pytest.skip("no verified clean cases yet — confirm ground truth in tests/evals/")
    wers = [r.ocr_wer for r in results if r.ocr_wer is not None]
    mean_wer = sum(wers) / len(wers)
    # Floor, not a target: catches a regression that breaks OCR on readable text.
    assert mean_wer < 0.6, f"OCR word-error-rate regressed on clean frames: {mean_wer:.2f}"
