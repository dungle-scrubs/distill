"""Eval-set regression guard.

These are not unit tests — they score Distill against the human-verified eval
frames in tests/evals/ and assert quality floors. They are marked ``eval`` and
skip when no case is verified yet or when Tesseract is missing, so the default
suite stays green until a human has confirmed ground truth.

Run explicitly:  uv run pytest -m eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from eval_helpers import load_score_module

EVAL_ROOT = Path(__file__).resolve().parent / "evals"
sys.path.insert(0, str(EVAL_ROOT))

pytestmark = pytest.mark.eval


def _scorer():
    from distill.ocr import find_tesseract_command

    if not find_tesseract_command():
        pytest.skip("tesseract not installed")
    return load_score_module()


def test_ocr_recovers_clean_frames() -> None:
    """The OCR floor covers frames both a human and OCR should read.

    Partial-legibility frames are covered by the vision metrics instead.
    """
    score = _scorer()
    results = [
        result
        for result in score.evaluate(with_vision=False)
        if result.category == "clean_text" and result.legibility == "clean"
    ]
    if not results:
        pytest.skip("no verified clean_text cases yet; confirm ground truth in tests/evals/")
    wers = [r.ocr_wer for r in results if r.ocr_wer is not None]
    mean_wer = sum(wers) / len(wers)
    # Floor, not a target: catches a regression that breaks OCR on readable text.
    assert mean_wer < 0.6, f"OCR word-error-rate regressed on clean frames: {mean_wer:.2f}"
