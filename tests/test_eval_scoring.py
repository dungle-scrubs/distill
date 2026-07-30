from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from eval_helpers import load_eval_module, load_score_module

score = load_score_module()
generate_negatives = load_eval_module("generate_negatives")
SYNTHETIC_CASE_TEXT = generate_negatives.SYNTHETIC_CASE_TEXT
token_prf = score.token_prf
word_error_rate = score.word_error_rate


def case_result(
    *,
    case_id: str = "case",
    has_text: bool,
    vision_recall: float | None = None,
    claimed_text: bool | None = None,
) -> score.CaseResult:
    return score.CaseResult(
        id=case_id,
        category="clean_text" if has_text else "textless",
        legibility="clean" if has_text else "unreadable",
        has_text=has_text,
        ocr_wer=None,
        vision_wer=None,
        vision_recall=vision_recall,
        vision_f1=None,
        flagged=None,
        should_flag=not has_text,
        claimed_text=claimed_text,
    )


def configure_textless_evaluation(
    monkeypatch: pytest.MonkeyPatch, interpretation: object | None
) -> None:
    case = score.LabelledCase(
        id="textless",
        category="textless",
        legibility="unreadable",
        has_text=False,
        verified=True,
    )
    monkeypatch.setattr(score, "find_tesseract_command", lambda: "tesseract")
    monkeypatch.setattr(score, "load_labelled_cases", lambda: [case])
    monkeypatch.setattr(score, "truth_text", lambda _case_id: "")
    monkeypatch.setattr(score, "ocr_frame", lambda *_args, **_kwargs: ("", None))
    monkeypatch.setattr(
        score,
        "_vision_interpreter",
        lambda _model, _backend, _base_url: lambda _image, _ocr_text: interpretation,
    )


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


def test_hallucination_rate_is_fraction_of_textless_cases_claiming_text() -> None:
    results = [
        case_result(case_id="claimed", has_text=False, claimed_text=True),
        case_result(case_id="not-claimed", has_text=False, claimed_text=False),
    ]

    assert score.hallucination_rate(results) == 0.5


def test_evaluate_does_not_count_punctuation_only_text_as_a_claim(monkeypatch) -> None:
    interpretation = SimpleNamespace(
        verbatim_text="...!!!",
        text_confidence="none",
        has_interpretation=False,
        carries_a_reading=True,
    )
    configure_textless_evaluation(monkeypatch, interpretation)

    [result] = score.evaluate(with_vision=True)

    assert result.claimed_text is False


def test_evaluate_counts_no_usable_textless_result_as_no_claim(monkeypatch) -> None:
    configure_textless_evaluation(monkeypatch, None)

    [result] = score.evaluate(with_vision=True)

    assert result.claimed_text is False


def test_text_recovery_accuracy_means_scored_text_bearing_recalls() -> None:
    results = [
        case_result(has_text=True, vision_recall=0.8),
        case_result(has_text=True, vision_recall=0.6),
        case_result(has_text=True),
    ]

    assert score.text_recovery_accuracy(results) == pytest.approx(0.7)
    assert score.text_recovery_accuracy([]) is None


def test_acceptance_rule_boundaries_are_inclusive() -> None:
    rule = score.AcceptanceRule(accuracy_floor=0.9, hallucination_ceiling=0.1)

    verdict = rule.evaluate(accuracy=0.9, hallucination_rate=0.1)

    assert verdict == score.AcceptanceVerdict(
        passed=True,
        accuracy_ok=True,
        hallucination_ok=True,
    )


@pytest.mark.parametrize(
    ("accuracy", "hallucination_rate", "accuracy_ok", "hallucination_ok"),
    [
        (0.89, 0.1, False, True),
        (0.9, 0.11, True, False),
        (None, 0.1, False, True),
        (0.9, None, True, False),
    ],
)
def test_acceptance_rule_fails_out_of_bounds_or_unmeasured_metrics(
    accuracy, hallucination_rate, accuracy_ok, hallucination_ok
) -> None:
    rule = score.AcceptanceRule(accuracy_floor=0.9, hallucination_ceiling=0.1)

    verdict = rule.evaluate(accuracy=accuracy, hallucination_rate=hallucination_rate)

    assert verdict == score.AcceptanceVerdict(
        passed=False,
        accuracy_ok=accuracy_ok,
        hallucination_ok=hallucination_ok,
    )


def test_local_baseline_pins_thresholds_from_its_own_measured_run() -> None:
    """The committed baseline is internally consistent: the pinned rule was
    derived from the stored run (not typed by hand), the corpus it was measured
    on is the corpus on disk, and the inclusive bounds do not exclude the
    reference run itself."""
    baseline = json.loads((score.EVAL_ROOT / "baseline_local.json").read_text())
    rule = score.AcceptanceRule(**baseline["acceptance_rule"])

    assert rule.accuracy_floor == baseline["summary"]["text_recovery_accuracy"]
    assert rule.hallucination_ceiling == baseline["summary"]["hallucination_rate"]

    # A corpus edit silently changes what the pinned floor means; fail loudly
    # so the baseline gets re-recorded instead.
    verified = [c.id for c in score.load_labelled_cases() if c.verified]
    assert baseline["summary"]["cases_scored"] == len(verified)
    assert [c["id"] for c in baseline["cases"]] == verified

    verdict = rule.evaluate(
        accuracy=baseline["summary"]["text_recovery_accuracy"],
        hallucination_rate=baseline["summary"]["hallucination_rate"],
    )

    assert verdict.passed  # inclusive bounds: the reference run is not excluded by its own rule


def test_summarize_includes_accuracy_and_hallucination_rate() -> None:
    results = [
        case_result(has_text=True, vision_recall=0.8),
        case_result(has_text=True, vision_recall=0.6),
        case_result(has_text=False, claimed_text=True),
        case_result(has_text=False, claimed_text=False),
    ]

    summary = score.summarize(results)

    assert summary["text_recovery_accuracy"] == pytest.approx(0.7)
    assert summary["vision_token_recall_mean"] == summary["text_recovery_accuracy"]
    assert summary["hallucination_rate"] == 0.5
