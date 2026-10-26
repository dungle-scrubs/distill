from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from eval_helpers import load_eval_module, load_score_module

if TYPE_CHECKING:
    from distill.local_vision import LocalVisionConfig

score = load_score_module()
generate_negatives = load_eval_module("generate_negatives")
SYNTHETIC_CASE_TEXT = generate_negatives.SYNTHETIC_CASE_TEXT
token_prf = score.token_prf
word_error_rate = score.word_error_rate


def case_result(
    *,
    case_id: str = "case",
    has_text: bool,
    category: str | None = None,
    vision_wer: float | None = None,
    vision_recall: float | None = None,
    vision_f1: float | None = None,
    claimed_text: bool | None = None,
    unusable: bool = False,
) -> score.CaseResult:
    return score.CaseResult(
        id=case_id,
        category=category or ("clean_text" if has_text else "textless"),
        legibility="clean" if has_text else "unreadable",
        has_text=has_text,
        ocr_wer=None,
        vision_wer=vision_wer,
        vision_recall=vision_recall,
        vision_f1=vision_f1,
        flagged=None,
        should_flag=not has_text,
        claimed_text=claimed_text,
        unusable=unusable,
    )


def configure_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    interpretation: object | None,
    *,
    has_text: bool,
) -> None:
    case = score.LabelledCase(
        id="text-bearing" if has_text else "textless",
        category="clean_text" if has_text else "textless",
        legibility="clean" if has_text else "unreadable",
        has_text=has_text,
        verified=True,
    )
    monkeypatch.setattr(score, "find_tesseract_command", lambda: "tesseract")
    monkeypatch.setattr(score, "load_labelled_cases", lambda: [case])
    monkeypatch.setattr(score, "truth_text", lambda _case_id: "alpha" if has_text else "")
    monkeypatch.setattr(score, "ocr_frame", lambda *_args, **_kwargs: ("", None))
    monkeypatch.setattr(
        score,
        "_vision_interpreter",
        lambda *_args, **_kwargs: lambda _image, _ocr_text: interpretation,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("…", []),
        ("• alpha • beta", ["alpha", "beta"]),
        ("100%", ["100"]),
    ],
)
def test_normalize_treats_unicode_punctuation_as_typography(text, expected) -> None:
    assert score.normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("…", False),
        ("•••", False),
        ("a…", True),
    ],
)
def test_claims_text_requires_an_alphanumeric_token(text, expected) -> None:
    assert score.claims_text(text) is expected


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
    assert "chrome_only" in categories
    assert categories == score.VALID_CATEGORIES


def test_labelled_corpus_has_at_least_six_true_negative_cases() -> None:
    cases = score.load_labelled_cases()

    assert sum(case.category == "textless" for case in cases) >= 6


def test_synthetic_truth_matches_the_generator_text() -> None:
    for case_id, expected in SYNTHETIC_CASE_TEXT.items():
        assert score.truth_text(case_id) == expected


def test_the_identifiers_frame_is_registered_and_scored() -> None:
    """The identifiers frame is in the corpus, not merely on disk.

    Its PNG and ground truth shipped with the redaction work that needed the
    frame's *content*; registering it is what makes the corpus assert that
    recovering a commit sha, a sha256 digest and a lockfile integrity hash is
    the correct reading rather than something redaction may delete. The
    generator-truth test above passes on an unregistered case, because it reads
    the fixture and never the corpus - so it cannot notice the registration
    going missing, and this can.
    """
    cases = {case.id: case for case in score.load_labelled_cases()}

    case = cases["30_synth_identifiers_f01"]
    assert case.verified, "an unverified case is skipped by the scorer"
    assert case.has_text
    assert case.category == "clean_text"
    assert score.truth_text(case.id)


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("has_text", '"false"'),
        ("verified", '"true"'),
    ],
)
def test_labelled_corpus_rejects_non_boolean_flags(tmp_path, monkeypatch, field, value) -> None:
    (tmp_path / "cases.toml").write_text(
        f'[[case]]\nid = "bad-boolean"\ncategory = "clean_text"\n'
        f'legibility = "clean"\n{field} = {value}\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", tmp_path / "frames")

    with pytest.raises(ValueError, match="bad-boolean"):
        score.load_labelled_cases()


def test_labelled_corpus_rejects_duplicate_ids(tmp_path, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "duplicate.png").write_bytes(b"png")
    (frames_dir / "duplicate.gt.txt").write_text("text")
    case = (
        '[[case]]\nid = "duplicate"\ncategory = "clean_text"\n'
        'legibility = "clean"\nhas_text = true\n'
    )
    (tmp_path / "cases.toml").write_text(case + case)
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", frames_dir)

    with pytest.raises(ValueError, match="duplicate"):
        score.load_labelled_cases()


@pytest.mark.parametrize(
    ("category", "has_text"),
    [
        ("textless", True),
        ("clean_text", False),
    ],
)
def test_labelled_corpus_rejects_category_has_text_mismatch(
    tmp_path, monkeypatch, category, has_text
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "mismatch.png").write_bytes(b"png")
    (frames_dir / "mismatch.gt.txt").write_text("")
    (tmp_path / "cases.toml").write_text(
        f'[[case]]\nid = "mismatch"\ncategory = "{category}"\n'
        f'legibility = "clean"\nhas_text = {str(has_text).lower()}\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", frames_dir)

    with pytest.raises(ValueError, match="mismatch"):
        score.load_labelled_cases()


def test_labelled_corpus_accepts_chrome_only_without_content_text(tmp_path, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "chrome.png").write_bytes(b"png")
    (frames_dir / "chrome.gt.txt").write_text("")
    (tmp_path / "cases.toml").write_text(
        '[[case]]\nid = "chrome"\ncategory = "chrome_only"\n'
        'legibility = "unreadable"\nhas_text = false\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", frames_dir)

    [case] = score.load_labelled_cases()

    assert case.category == "chrome_only"
    assert case.has_text is False


@pytest.mark.parametrize(
    ("category", "has_text", "truth"),
    [
        ("textless", False, "invented text"),
        ("clean_text", True, ""),
    ],
)
def test_labelled_corpus_rejects_verified_truth_incoherent_with_has_text(
    tmp_path, monkeypatch, category, has_text, truth
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "bad-truth.png").write_bytes(b"png")
    (frames_dir / "bad-truth.gt.txt").write_text(truth)
    (tmp_path / "cases.toml").write_text(
        f'[[case]]\nid = "bad-truth"\ncategory = "{category}"\n'
        f'legibility = "clean"\nhas_text = {str(has_text).lower()}\nverified = true\n'
    )
    monkeypatch.setattr(score, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(score, "FRAMES_DIR", frames_dir)

    with pytest.raises(ValueError, match="bad-truth"):
        score.load_labelled_cases()


def test_hallucination_rate_is_fraction_of_textless_cases_claiming_text() -> None:
    results = [
        case_result(case_id="claimed", has_text=False, claimed_text=True),
        case_result(case_id="not-claimed", has_text=False, claimed_text=False),
    ]

    assert score.hallucination_rate(results) == 0.5


def test_hallucination_rate_ignores_chrome_only_claims() -> None:
    results = [
        case_result(
            case_id="chrome-claim",
            category="chrome_only",
            has_text=False,
            claimed_text=True,
        ),
        case_result(case_id="true-negative", has_text=False, claimed_text=False),
    ]

    assert score.hallucination_rate(results) == 0.0


def test_chrome_transcription_rate_counts_only_chrome_only_claims() -> None:
    results = [
        case_result(
            case_id="chrome-claim",
            category="chrome_only",
            has_text=False,
            claimed_text=True,
        ),
        case_result(
            case_id="chrome-no-claim",
            category="chrome_only",
            has_text=False,
            claimed_text=False,
        ),
        case_result(case_id="true-negative-claim", has_text=False, claimed_text=True),
    ]

    assert score.chrome_transcription_rate(results) == 0.5
    assert score.chrome_transcription_rate(results[2:]) is None


def test_evaluate_does_not_count_punctuation_only_text_as_a_claim(monkeypatch) -> None:
    interpretation = SimpleNamespace(
        verbatim_text="...!!!",
        text_confidence="none",
        has_interpretation=False,
        carries_a_reading=True,
    )
    configure_evaluation(monkeypatch, interpretation, has_text=False)

    [result] = score.evaluate(with_vision=True)

    assert result.claimed_text is False


def test_evaluate_excludes_no_usable_textless_result_from_claims(monkeypatch) -> None:
    configure_evaluation(monkeypatch, None, has_text=False)

    [result] = score.evaluate(with_vision=True)

    assert result.claimed_text is None
    assert result.unusable is True


def test_evaluate_counts_no_usable_text_bearing_result_as_total_miss(monkeypatch) -> None:
    configure_evaluation(monkeypatch, None, has_text=True)

    [result] = score.evaluate(with_vision=True)

    assert result.vision_recall == 0.0
    assert result.vision_wer == 1.0
    assert result.vision_f1 == 0.0
    assert result.unusable is True


def test_text_recovery_accuracy_includes_total_misses_in_denominator() -> None:
    results = [
        case_result(has_text=True, vision_recall=1.0),
        case_result(has_text=True, vision_recall=0.0),
        case_result(has_text=True, vision_recall=0.0),
    ]

    assert score.text_recovery_accuracy(results) == pytest.approx(1 / 3)
    assert score.text_recovery_accuracy([]) is None


def test_acceptance_rule_boundaries_are_inclusive() -> None:
    rule = score.AcceptanceRule(accuracy_floor=0.9, hallucination_ceiling=0.1, f1_floor=0.9)

    verdict = rule.evaluate(accuracy=0.9, hallucination_rate=0.1, f1=0.9)

    assert verdict == score.AcceptanceVerdict(
        passed=True,
        accuracy_ok=True,
        hallucination_ok=True,
        f1_ok=True,
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
    rule = score.AcceptanceRule(accuracy_floor=0.9, hallucination_ceiling=0.1, f1_floor=0.9)

    verdict = rule.evaluate(accuracy=accuracy, hallucination_rate=hallucination_rate, f1=0.9)

    assert verdict == score.AcceptanceVerdict(
        passed=False,
        accuracy_ok=accuracy_ok,
        hallucination_ok=hallucination_ok,
        f1_ok=True,
    )


def test_local_baseline_pins_thresholds_from_its_own_measured_run() -> None:
    """The committed baseline is internally consistent: the pinned rule was
    derived from the stored run (not typed by hand), the corpus it was measured
    on is the corpus on disk, and the inclusive bounds do not exclude the
    reference run itself."""
    baseline = json.loads((score.EVAL_ROOT / "baseline_local.json").read_text())
    acceptance_rule = baseline["acceptance_rule"]
    rule = score.AcceptanceRule(
        **{key: value for key, value in acceptance_rule.items() if key != "noise_band"}
    )
    summary = baseline["summary"]

    assert rule.accuracy_floor == pytest.approx(
        summary["text_recovery_accuracy"] - acceptance_rule["noise_band"]
    )
    assert rule.hallucination_ceiling == summary["hallucination_rate"]

    # A corpus edit silently changes what the pinned floor means; fail loudly
    # so the baseline gets re-recorded instead.
    cases = score.load_labelled_cases()
    scored_ids = [
        case.id for case in cases if case.verified and score.truth_text(case.id) is not None
    ]
    assert score.corpus_digest(cases) == baseline["corpus_sha256"]
    assert summary["cases_scored"] == len(scored_ids)
    assert [case["id"] for case in baseline["cases"]] == scored_ids

    stored_results = [score.CaseResult(**case) for case in baseline["cases"]]
    assert score.text_recovery_accuracy(stored_results) == pytest.approx(
        summary["text_recovery_accuracy"]
    )
    assert score.hallucination_rate(stored_results) == pytest.approx(summary["hallucination_rate"])

    verdict = rule.evaluate(
        accuracy=summary["text_recovery_accuracy"],
        hallucination_rate=summary["hallucination_rate"],
        f1=summary["vision_token_f1_mean"],
    )

    assert verdict.passed


def test_summarize_includes_accuracy_and_claim_rates() -> None:
    results = [
        case_result(has_text=True, vision_recall=0.8),
        case_result(has_text=True, vision_recall=0.6),
        case_result(has_text=False, claimed_text=True),
        case_result(has_text=False, claimed_text=False),
        case_result(category="chrome_only", has_text=False, claimed_text=True),
    ]

    summary = score.summarize(results)

    assert summary["text_recovery_accuracy"] == pytest.approx(0.7)
    assert summary["vision_token_recall_mean"] == summary["text_recovery_accuracy"]
    assert summary["hallucination_rate"] == 0.5
    assert summary["chrome_transcription_rate"] == 1.0
    assert summary["unusable_readings"] == 0


def test_summarize_counts_only_readings_marked_unusable() -> None:
    results = [
        case_result(
            case_id="genuine-total-miss",
            has_text=True,
            vision_wer=1.0,
            vision_recall=0.0,
            vision_f1=0.0,
        ),
        case_result(
            case_id="unusable",
            has_text=True,
            vision_wer=1.0,
            vision_recall=0.0,
            vision_f1=0.0,
            unusable=True,
        ),
    ]

    assert score.summarize(results)["unusable_readings"] == 1


def test_the_f1_floor_catches_stray_text_the_other_bounds_cannot() -> None:
    """Accuracy is recall-only and invention looks only at textless frames, so
    without an F1 floor a reader could transcribe every banner and logo on real
    slides and still pass."""
    rule = score.AcceptanceRule(accuracy_floor=0.8, hallucination_ceiling=0.0, f1_floor=0.85)

    polluted = rule.evaluate(accuracy=0.99, hallucination_rate=0.0, f1=0.60)
    assert polluted.passed is False
    assert polluted.accuracy_ok is True and polluted.hallucination_ok is True
    assert polluted.f1_ok is False

    at_floor = rule.evaluate(accuracy=0.99, hallucination_rate=0.0, f1=0.85)
    assert at_floor.passed is True

    unmeasured = rule.evaluate(accuracy=0.99, hallucination_rate=0.0, f1=None)
    assert unmeasured.passed is False


def test_chrome_only_frames_stay_out_of_the_invention_metric() -> None:
    """Text read off a chrome frame was really there, so it is not invention -
    but it is still reported, because the prompt told the reader to ignore it."""
    results = [
        case_result(case_id="chrome", has_text=False, claimed_text=True, category="chrome_only"),
        case_result(case_id="true-neg", has_text=False, claimed_text=False, category="textless"),
    ]

    assert score.hallucination_rate(results) == 0.0
    assert score.chrome_transcription_rate(results) == 1.0


def test_the_corpus_keeps_a_wide_true_negative_denominator() -> None:
    """A zero invention ceiling only means something over enough true
    negatives, and only if they are frames with no text at all."""
    cases = score.load_labelled_cases()
    textless = [c for c in cases if c.category == "textless"]

    assert len(textless) >= 6
    for case in textless:
        assert case.has_text is False


def test_the_gate_evidence_file_is_internally_consistent() -> None:
    """The evidence must not be silently rewritten: its verdict has to follow
    from its own summary under the pinned rule, and the superseded run has to
    still record its failure."""
    evidence = json.loads((score.EVAL_ROOT / "gate_2_to_3_cloud.json").read_text())
    baseline = json.loads((score.EVAL_ROOT / "baseline_local.json").read_text())
    rule = score.AcceptanceRule(
        **{k: v for k, v in baseline["acceptance_rule"].items() if k != "noise_band"}
    )

    recomputed = rule.evaluate(
        accuracy=evidence["summary"]["text_recovery_accuracy"],
        hallucination_rate=evidence["summary"]["hallucination_rate"],
        f1=evidence["summary"]["vision_token_f1_mean"],
    )
    assert recomputed.passed is evidence["verdict"]["passed"]
    assert evidence["superseded_run"]["verdict"]["passed"] is False

    # The cloud file is not digest-pinned the way the baseline is, so without
    # this it can describe a corpus nobody scored: register a case, re-record
    # the baseline because line 438 forces it, leave this file alone, and the
    # suite stays green while the two evidence files disagree about what was
    # measured. README says drift between them and the corpus fails the suite;
    # this is what makes that true.
    baseline_digest = baseline["corpus_sha256"]
    assert evidence["corpus_sha256"] == baseline_digest
    scored_ids = [
        case.id
        for case in score.load_labelled_cases()
        if case.verified and score.truth_text(case.id) is not None
    ]
    assert evidence["summary"]["cases_scored"] == len(scored_ids)
    assert [case["id"] for case in evidence["cases"]] == scored_ids


def a_chain_config() -> dict[str, object]:
    """A configuration file's worth of a two-entry **endpoint chain**.

    The chain is what a run walks; the scorer's flags are resolved against it.
    """
    return {
        "endpoints": [
            {"model": "model-a", "base_url": "http://127.0.0.1:8001/v1"},
            {"model": "model-b", "base_url": "http://127.0.0.1:8002/v1"},
        ]
    }


def use_a_chain_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Scope Distill's configuration to a directory naming a two-entry chain.

    The scorer reads the same configuration layers a run reads, so the chain
    is written where a run would find it: a configuration directory, named by
    DISTILL_CONFIG_DIR, holding `distill.local-vision.json`.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "distill.local-vision.json").write_text(json.dumps(a_chain_config()))
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(config_dir))
    return config_dir


def a_probe(available: bool, model: str, base_url: str) -> object:
    from distill.local_vision import LocalVisionProbe

    return LocalVisionProbe(
        available=available,
        backend="rapid-mlx",
        model=model,
        base_url=base_url,
        code="local_vision_unreachable",
        message="no answer",
        detail={},
    )


def record_probes(
    monkeypatch: pytest.MonkeyPatch,
    available: Callable[[LocalVisionConfig], bool] = lambda entry: True,
) -> list[tuple[str, str]]:
    """Fake the endpoint probe, recording every endpoint that was asked.

    The endpoints are asked in the order the walk tries them, so the record
    says which one the scorer settled on and which it never reached.
    """
    from distill import local_vision

    asked: list[tuple[str, str]] = []

    def probe(entry: LocalVisionConfig) -> object:
        asked.append((entry.model, entry.base_url))
        return a_probe(bool(available(entry)), entry.model, entry.base_url)

    monkeypatch.setattr(local_vision, "probe_local_vision", probe)
    return asked


def test_the_scored_endpoint_is_the_one_the_chain_walk_selects(tmp_path, monkeypatch) -> None:
    """A config that names a chain is scored against the endpoint a run would use.

    The scorer used to read the config's own top-level address, so a config
    naming a chain was scored against the default endpoint rather than the one
    the chain selects. The score was then evidence about a reader the run would
    not have used.
    """
    use_a_chain_config(monkeypatch, tmp_path)
    asked = record_probes(monkeypatch, available=lambda entry: entry.model == "model-b")

    endpoint = score._scored_endpoint()

    assert endpoint.model == "model-b"
    assert endpoint.base_url == "http://127.0.0.1:8002/v1"
    assert asked == [
        ("model-a", "http://127.0.0.1:8001/v1"),
        ("model-b", "http://127.0.0.1:8002/v1"),
    ]


def test_a_model_override_over_a_chain_is_the_runs_own_refusal(tmp_path, monkeypatch) -> None:
    """`--model` over a chain of more than one entry is refused, as a run refuses it.

    The model decides which bundle a run publishes under, so a run asked for
    one model against a chain of two refuses rather than guessing which entry
    was meant. The scorer used to score the named model at the built-in
    default address - a reader no run would ever call.
    """
    from distill.configuration import local_vision_config_from_args
    from distill.errors import DistillError

    use_a_chain_config(monkeypatch, tmp_path)
    asked = record_probes(monkeypatch)

    with pytest.raises(DistillError) as run_refusal:
        local_vision_config_from_args({"local_vision_model": "model-x"})
    with pytest.raises(DistillError) as scorer_refusal:
        score._scored_endpoint(model="model-x")

    assert scorer_refusal.value.code == "E_BAD_OPTIONS"
    # The scorer's refusal is the run's refusal, word for word: the scorer
    # does not get to write a better one.
    assert scorer_refusal.value.message == run_refusal.value.message
    assert asked == [], "the endpoint was probed even though the options were refused"


def test_a_base_url_override_moves_the_chains_first_entry(tmp_path, monkeypatch) -> None:
    """`--base-url` re-addresses the chain's first entry and leaves the rest.

    A run asked for an address asks the chain's first entry there first and
    falls through to the others. The scorer used to score the built-in default
    model at the named address - a reader no run would call.
    """
    from distill.configuration import local_vision_config_from_args

    use_a_chain_config(monkeypatch, tmp_path)
    asked = record_probes(monkeypatch)

    endpoint = score._scored_endpoint(base_url="http://127.0.0.1:9000/v1")

    assert endpoint.model == "model-a"
    assert endpoint.base_url == "http://127.0.0.1:9000/v1"
    # The first entry is the one a run would try first, so it is the one asked.
    assert asked == [("model-a", "http://127.0.0.1:9000/v1")]

    run_config = local_vision_config_from_args(
        {"local_vision_base_url": "http://127.0.0.1:9000/v1"}
    )
    assert [(entry.model, entry.base_url) for entry in run_config.endpoints or ()] == [
        ("model-a", "http://127.0.0.1:9000/v1"),
        ("model-b", "http://127.0.0.1:8002/v1"),
    ]


def test_a_model_and_a_base_url_together_name_one_endpoint(tmp_path, monkeypatch) -> None:
    """Both flags name one endpoint completely, and that one is scored.

    A run told to use that endpoint carries no other, so there is no fallback
    to fall through to - the chain is replaced by the endpoint named.
    """
    use_a_chain_config(monkeypatch, tmp_path)
    asked = record_probes(monkeypatch)

    endpoint = score._scored_endpoint(model="model-x", base_url="http://127.0.0.1:9000/v1")

    assert endpoint.model == "model-x"
    assert endpoint.base_url == "http://127.0.0.1:9000/v1"
    assert asked == [("model-x", "http://127.0.0.1:9000/v1")]
    # The chain is replaced, not kept beside the endpoint named: a run told to
    # use one endpoint carries no other to fall through to.
    config = score._vision_config(model="model-x", base_url="http://127.0.0.1:9000/v1")
    assert [(entry.model, entry.base_url) for entry in config.endpoints] == [
        ("model-x", "http://127.0.0.1:9000/v1")
    ]


def test_the_backend_flag_still_names_the_backend(tmp_path, monkeypatch) -> None:
    """`--backend` lands on the resolved config, and does not disturb the chain.

    The backend is an endpoint field, not a chain-wide one: naming it says
    which server speaks HTTP, not which entry of the chain is meant.
    """
    use_a_chain_config(monkeypatch, tmp_path)

    config = score._vision_config(backend="rapid-mlx")

    assert config.backend == "rapid-mlx"
    assert [(entry.model, entry.base_url) for entry in config.endpoints] == [
        ("model-a", "http://127.0.0.1:8001/v1"),
        ("model-b", "http://127.0.0.1:8002/v1"),
    ]


def test_the_scorer_stops_with_the_runs_refusal_instead_of_scoring(tmp_path, monkeypatch) -> None:
    """`score.py --model model-x` stops with the run's refusal, and scores nothing.

    The scorer is a tool an operator runs by hand, so the refusal has to read
    as the run's refusal - the same code, the same message - rather than as a
    traceback out of the eval tooling.
    """
    from distill.configuration import local_vision_config_from_args
    from distill.errors import DistillError

    use_a_chain_config(monkeypatch, tmp_path)
    record_probes(monkeypatch)

    def refuse_to_score(*_args, **_kwargs):
        raise AssertionError("the corpus was scored after the options were refused")

    monkeypatch.setattr(score, "evaluate", refuse_to_score)

    with pytest.raises(DistillError) as run_refusal:
        local_vision_config_from_args({"local_vision_model": "model-x"})
    with pytest.raises(SystemExit) as stopped:
        score.main(["--with-vision", "--model", "model-x"])

    assert str(stopped.value) == f"{run_refusal.value.code}: {run_refusal.value.message}"
    assert "E_BAD_OPTIONS" in str(stopped.value)


def test_the_json_run_block_records_the_resolved_chain(tmp_path, monkeypatch) -> None:
    """The run block keeps every configured address, sanitized.

    The block is committed with a baseline, so it records the **effective**
    config - the chain a run would walk - and no address in it keeps a key.
    """
    use_a_chain_config(monkeypatch, tmp_path)

    public = score._public_config(score._vision_config())

    assert [entry["model"] for entry in public["endpoints"]] == ["model-a", "model-b"]
    assert [entry["base_url"] for entry in public["endpoints"]] == [
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
    ]


def test_the_scored_endpoint_is_named_with_its_model_and_sanitized_address() -> None:
    """The output says which reader produced the score, without the secret parts.

    A URL can carry a key in its userinfo or its path, and this text is pasted
    into bug reports, so the address is the authority only - the same form
    Distill's own messages use.
    """
    from distill.local_vision import LocalVisionConfig

    endpoint = LocalVisionConfig(
        model="model-a",
        base_url="http://user:key@127.0.0.1:8001/v1?token=secret",
    )

    assert score._endpoint_label(endpoint) == "model-a at http://127.0.0.1:8001"


def test_the_json_run_block_shows_every_configured_address_sanitized() -> None:
    """The run block is committed with a baseline, so no address in it keeps a key.

    `public_dict` reports `base_url` as configured, at the top level and in each
    chain entry.
    """
    from distill.local_vision import LocalVisionConfig

    secret = "http://user:key@127.0.0.1:8001/v1?token=secret"
    config = LocalVisionConfig(
        model="model-a",
        base_url=secret,
        endpoints=(
            LocalVisionConfig(model="model-a", base_url=secret),
            LocalVisionConfig(model="model-b", base_url="http://127.0.0.1:8002/v1"),
        ),
    )

    public = score._public_config(config)

    text = json.dumps(public)
    assert "key" not in text and "secret" not in text
    assert public["base_url"] == "http://127.0.0.1:8001"
