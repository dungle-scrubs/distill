from __future__ import annotations

import pytest
from eval_helpers import load_eval_module, load_score_module

score = load_score_module()
generate_negatives = load_eval_module("generate_negatives")
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


def test_labelled_corpus_represents_every_category() -> None:
    cases = score.load_labelled_cases()
    categories = {case.category for case in cases}

    assert cases
    assert categories == score.VALID_CATEGORIES


def test_synthetic_truth_matches_the_generator_text() -> None:
    for case_id, expected in SYNTHETIC_CASE_TEXT.items():
        assert score.truth_text(case_id) == expected


def test_labelled_corpus_rejects_unknown_category(tmp_path, monkeypatch) -> None:
    (tmp_path / "cases.toml").write_text(
        '[[case]]\nid = "bad-category"\ncategory = "unknown"\nlegibility = "clean"\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", tmp_path / "frames")

    with pytest.raises(ValueError, match="bad-category"):
        score.load_labelled_cases()


@pytest.mark.parametrize("present_suffix", [".png", ".gt.txt"])
def test_labelled_corpus_rejects_missing_fixture(tmp_path, monkeypatch, present_suffix) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / f"missing-fixture{present_suffix}").write_bytes(b"fixture")
    (tmp_path / "cases.toml").write_text(
        '[[case]]\nid = "missing-fixture"\ncategory = "clean_text"\nlegibility = "clean"\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", frames_dir)

    with pytest.raises(ValueError, match="missing-fixture"):
        score.load_labelled_cases()


def test_labelled_corpus_rejects_missing_id(tmp_path, monkeypatch) -> None:
    (tmp_path / "cases.toml").write_text(
        '[[case]]\ncategory = "clean_text"\nlegibility = "clean"\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", tmp_path / "frames")

    with pytest.raises(ValueError, match="invalid id"):
        score.load_labelled_cases()


def test_labelled_corpus_rejects_invalid_legibility(tmp_path, monkeypatch) -> None:
    (tmp_path / "cases.toml").write_text(
        '[[case]]\nid = "bad-legibility"\ncategory = "clean_text"\nlegibility = "typo"\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", tmp_path / "frames")

    with pytest.raises(ValueError, match="bad-legibility"):
        score.load_labelled_cases()
