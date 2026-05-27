from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "evals"))

from score import token_prf, word_error_rate  # noqa: E402


def test_token_prf_is_order_insensitive() -> None:
    a = "GitHub is not a coordination layer"
    b = "coordination layer GitHub is not a"

    precision, recall, f1 = token_prf(a, b)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)
    # The exact failure this metric exists to avoid: WER punishes the same words reordered.
    assert word_error_rate(a, b) > 0.5


def test_token_recall_ignores_extra_chrome() -> None:
    truth = "the coordination problem github is not a coordination layer"
    hypothesis = truth + " ona.com engineering the future of ai 18 37"

    precision, recall, _ = token_prf(truth, hypothesis)
    assert recall == 1.0  # all real content captured
    assert precision < 1.0  # but chrome the model added drags precision down


def test_token_prf_partial_capture() -> None:
    precision, recall, _ = token_prf("alpha beta gamma delta", "alpha beta")

    assert recall == 0.5
    assert precision == 1.0
