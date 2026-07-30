from __future__ import annotations

import importlib
from pathlib import Path

from eval_helpers import load_score_module

score = load_score_module()
generate_negatives = importlib.import_module("generate_negatives")
SYNTHETIC_CASE_TEXT = generate_negatives.SYNTHETIC_CASE_TEXT
token_prf = score.token_prf
word_error_rate = score.word_error_rate


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


def test_labelled_corpus_has_only_valid_categories() -> None:
    cases = score.load_labelled_cases()

    assert cases
    assert all(case.category in score.VALID_CATEGORIES for case in cases)


def test_labelled_corpus_represents_every_category() -> None:
    categories = {case.category for case in score.load_labelled_cases()}

    assert categories == score.VALID_CATEGORIES


def test_every_labelled_case_has_image_and_truth_fixtures() -> None:
    for case in score.load_labelled_cases():
        assert (score.FRAMES_DIR / f"{case.id}.png").is_file()
        assert (score.FRAMES_DIR / f"{case.id}.gt.txt").is_file()


def test_synthetic_truth_matches_the_generator_text() -> None:
    for case_id, expected in SYNTHETIC_CASE_TEXT.items():
        truth_path = Path(score.FRAMES_DIR) / f"{case_id}.gt.txt"
        body = "\n".join(
            line
            for line in truth_path.read_text().splitlines()
            if not line.lstrip().startswith("#")
        ).strip()

        assert body == expected
