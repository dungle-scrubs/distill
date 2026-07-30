"""Score Distill's text recovery against the hand-written eval ground truth.

Reads ``cases.toml`` and each frame's ``<id>.gt.txt`` and reports two things:

1. Word-error-rate (WER) of recovered text vs. truth, for text-bearing frames.
   Preprocessed OCR is always scored; the vision model's ``verbatim_text`` is
   scored too when ``--with-vision`` is set and Rapid-MLX is running.
2. Grounding-flag accuracy: whether ``grounding.assess_grounding`` flags the
   frames a human marked hard/unreadable (``legibility != "clean"``) and stays
   quiet on clean ones — precision/recall over that label.

Unverified or empty cases are skipped, so the score reflects only frames a human
has actually confirmed. Labels and fixtures are validated up front, and a
malformed case aborts the run rather than being skipped. Run:

    uv run python tests/evals/score.py [--with-vision] [--json]
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from distill.grounding import assess_grounding
from distill.ocr import find_tesseract_command, ocr_frame

EVAL_ROOT = Path(__file__).resolve().parent
FRAMES_DIR = EVAL_ROOT / "frames"
VALID_CATEGORIES = frozenset(
    {
        "clean_text",
        "textless",
        "injection",
        "safety_blocked",
        "ocr_vision_disagreement",
    }
)
VALID_LEGIBILITY = frozenset({"clean", "partial", "unreadable"})

_PUNCT = str.maketrans(dict.fromkeys("\"'`.,:;!?()[]{}<>|/\\-–—_=+*#", " "))


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation, collapse whitespace into comparable tokens."""
    return text.lower().translate(_PUNCT).split()


def word_error_rate(truth: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by truth length (0.0 is perfect)."""
    ref = normalize(truth)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1] / len(ref)


def token_prf(truth: str, hypothesis: str) -> tuple[float, float, float]:
    """Order-insensitive token precision/recall/F1 over word multisets.

    WER is sequence-based, so it over-penalizes multi-region slides where the
    model reads the same words in a different order, and chrome the model adds
    that truth omits. Recall ("did it capture the content?") ignores both; this
    is the fairer headline for transcription quality. Returns (precision, recall, f1).
    """
    ref = Counter(normalize(truth))
    hyp = Counter(normalize(hypothesis))
    if not ref:
        return (0.0, 1.0, 0.0) if hyp else (1.0, 1.0, 1.0)
    overlap = sum((ref & hyp).values())
    recall = overlap / sum(ref.values())
    precision = overlap / sum(hyp.values()) if hyp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass
class CaseResult:
    id: str
    category: str
    legibility: str
    has_text: bool
    ocr_wer: float | None
    vision_wer: float | None
    vision_recall: float | None  # order-insensitive token recall (content captured)
    vision_f1: float | None
    flagged: bool | None  # grounding marked it low-confidence
    should_flag: bool  # human says it is not cleanly legible
    claimed_text: bool | None = None  # textless frame: reader claimed text? None when unscored


@dataclass(frozen=True)
class LabelledCase:
    id: str
    category: str
    legibility: str
    has_text: bool
    verified: bool


@dataclass(frozen=True)
class AcceptanceVerdict:
    """Per-condition outcome of an `AcceptanceRule`; `passed` is the single pass/fail."""

    passed: bool
    accuracy_ok: bool
    hallucination_ok: bool


@dataclass(frozen=True)
class AcceptanceRule:
    """The eval's pass/fail bar: an accuracy floor AND a hallucination ceiling."""

    accuracy_floor: float
    hallucination_ceiling: float

    def evaluate(
        self, accuracy: float | None, hallucination_rate: float | None
    ) -> AcceptanceVerdict:
        """Pass only when both metrics were measured and both stay inside their bounds.

        A missing metric (``None`` - a run without ``--with-vision``, or a corpus
        with no textless cases) fails its condition: an unmeasured gate is not a
        passed gate. Both bounds are inclusive.
        """
        accuracy_ok = accuracy is not None and accuracy >= self.accuracy_floor
        hallucination_ok = (
            hallucination_rate is not None and hallucination_rate <= self.hallucination_ceiling
        )
        return AcceptanceVerdict(
            passed=accuracy_ok and hallucination_ok,
            accuracy_ok=accuracy_ok,
            hallucination_ok=hallucination_ok,
        )


def load_labelled_cases() -> list[LabelledCase]:
    """Load labels, rejecting invalid ids, categories, legibility, or fixtures."""
    data = tomllib.loads((EVAL_ROOT / "cases.toml").read_text())
    cases: list[LabelledCase] = []
    for raw_case in data.get("case", []):
        case_id = raw_case.get("id")
        if not isinstance(case_id, str):
            raise ValueError(f"case has invalid id: {case_id!r}")
        category = raw_case.get("category")
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            raise ValueError(f"case {case_id} has invalid category: {category!r}")
        legibility = raw_case.get("legibility")
        if not isinstance(legibility, str) or legibility not in VALID_LEGIBILITY:
            raise ValueError(f"case {case_id} has invalid legibility: {legibility!r}")
        for suffix in (".png", ".gt.txt"):
            path = FRAMES_DIR / f"{case_id}{suffix}"
            if not path.exists():
                raise ValueError(f"case {case_id} is missing fixture: {path.name}")
        cases.append(
            LabelledCase(
                id=case_id,
                category=category,
                legibility=legibility,
                has_text=bool(raw_case.get("has_text", True)),
                verified=bool(raw_case.get("verified", False)),
            )
        )
    return cases


def truth_text(case_id: str) -> str | None:
    """Return confirmed truth text, or None if the file is missing/unverified.

    Lines starting with ``#`` are notes (e.g. provenance) and are stripped before
    scoring. The ``UNVERIFIED`` sentinel and ``verified = false`` in cases.toml
    both keep a case out of the score until a human has confirmed it.
    """
    path = FRAMES_DIR / f"{case_id}.gt.txt"
    if not path.exists():
        return None
    contents = path.read_text()
    if "UNVERIFIED" in contents:
        return None
    body = "\n".join(line for line in contents.splitlines() if not line.lstrip().startswith("#"))
    return body.strip()


def evaluate(
    with_vision: bool,
    model: str | None = None,
    backend: str | None = None,
    base_url: str | None = None,
) -> list[CaseResult]:
    # The same lookup that gates this function must also drive the OCR call.
    # find_tesseract_command() falls back to well-known install paths that are
    # not necessarily on PATH; if the guard used it but ocr_frame resolved
    # "tesseract" from PATH on its own, OCR would silently return nothing and
    # the eval would hard-fail at WER 1.00 instead of skipping.
    tesseract_cmd = find_tesseract_command()
    if not tesseract_cmd:
        raise SystemExit("tesseract not found; install it before scoring OCR")
    interpret = _vision_interpreter(model, backend, base_url) if with_vision else None
    results: list[CaseResult] = []
    for case in load_labelled_cases():
        case_id = case.id
        if not case.verified:
            continue
        truth = truth_text(case_id)
        if truth is None:
            continue
        image = FRAMES_DIR / f"{case_id}.png"
        ocr_text, _ = ocr_frame(image, "eng", tesseract_cmd=tesseract_cmd)
        has_text = case.has_text
        ocr_wer = word_error_rate(truth, ocr_text) if has_text else None

        vision_wer: float | None = None
        vision_recall: float | None = None
        vision_f1: float | None = None
        flagged: bool | None = None
        claimed_text: bool | None = None
        if interpret is not None:
            result = interpret(image, ocr_text)
            if result is not None:
                if has_text:
                    vision_wer = word_error_rate(truth, result.verbatim_text)
                    _, vision_recall, vision_f1 = token_prf(truth, result.verbatim_text)
                else:
                    claimed_text = bool(normalize(result.verbatim_text))
                assessment = assess_grounding(
                    ocr_text=ocr_text,
                    verbatim_text=result.verbatim_text,
                    text_confidence=result.text_confidence,
                    has_interpretation=result.has_interpretation,
                    carries_a_reading=result.carries_a_reading,
                )
                flagged = assessment.is_low_confidence
            else:
                # The model produced nothing usable: the pipeline now treats this
                # as a low-confidence (ungrounded) frame, so the eval does too.
                flagged = True
                if not has_text:
                    claimed_text = False

        results.append(
            CaseResult(
                id=case_id,
                category=case.category,
                legibility=case.legibility,
                has_text=has_text,
                ocr_wer=ocr_wer,
                vision_wer=vision_wer,
                vision_recall=vision_recall,
                vision_f1=vision_f1,
                flagged=flagged,
                should_flag=case.legibility != "clean",
                claimed_text=claimed_text,
            )
        )
    return results


def _vision_config(
    model: str | None = None, backend: str | None = None, base_url: str | None = None
):
    from distill.local_vision import LocalVisionConfig

    overrides: dict[str, str] = {}
    if model:
        overrides["model"] = model
    if backend:
        overrides["backend"] = backend
    if base_url:
        overrides["base_url"] = base_url.rstrip("/")
    return replace(LocalVisionConfig(), **overrides) if overrides else LocalVisionConfig()


def _vision_interpreter(
    model: str | None = None, backend: str | None = None, base_url: str | None = None
):
    from distill.local_vision import probe_local_vision, try_interpret_image
    from distill.vision_prompts import build_technical_frame_prompt

    config = _vision_config(model, backend, base_url)
    probe = probe_local_vision(config)
    if not probe.available:
        raise SystemExit(f"vision backend unavailable ({probe.code}): {probe.message}")

    def interpret(image: Path, ocr_text: str):
        prompt = build_technical_frame_prompt(ocr_text=ocr_text or None)
        result, _ = try_interpret_image(config, image, prompt.prompt, prompt_profile=prompt.profile)
        return result

    return interpret


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def text_recovery_accuracy(results: list[CaseResult]) -> float | None:
    recalls = [r.vision_recall for r in results if r.has_text and r.vision_recall is not None]
    return _mean(recalls)


def hallucination_rate(results: list[CaseResult]) -> float | None:
    claims = [
        float(r.claimed_text) for r in results if not r.has_text and r.claimed_text is not None
    ]
    return _mean(claims)


def summarize(results: list[CaseResult]) -> dict:
    ocr_wers = [r.ocr_wer for r in results if r.ocr_wer is not None]
    vision_wers = [r.vision_wer for r in results if r.vision_wer is not None]
    vision_f1s = [r.vision_f1 for r in results if r.vision_f1 is not None]
    flag_known = [r for r in results if r.flagged is not None]
    true_pos = sum(1 for r in flag_known if r.flagged and r.should_flag)
    flagged_total = sum(1 for r in flag_known if r.flagged)
    should_total = sum(1 for r in flag_known if r.should_flag)
    accuracy = text_recovery_accuracy(results)
    return {
        "cases_scored": len(results),
        "ocr_wer_mean": _mean(ocr_wers),
        "vision_wer_mean": _mean(vision_wers),
        # Same value as text_recovery_accuracy; kept for continuity with recorded
        # runs (the README Findings tables are keyed on token recall).
        "vision_token_recall_mean": accuracy,
        "text_recovery_accuracy": accuracy,
        "hallucination_rate": hallucination_rate(results),
        "vision_token_f1_mean": _mean(vision_f1s),
        "grounding_precision": (true_pos / flagged_total) if flagged_total else None,
        "grounding_recall": (true_pos / should_total) if should_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-vision", action="store_true", help="also score the vision model")
    parser.add_argument(
        "--backend",
        default=None,
        help="backend hint (default rapid-mlx); Distill talks to the Rapid-MLX server directly",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="optional vision model override; default is the eval-chosen Qwen3-VL-8B-8bit",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="vision endpoint override (e.g. http://127.0.0.1:17439/v1); default is the config's",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON for programmatic use")
    args = parser.parse_args()

    results = evaluate(args.with_vision, args.model, args.backend, args.base_url)
    summary = summarize(results)
    if args.json:
        # The run block makes a stored score self-describing (what model/endpoint
        # produced it), so a committed baseline stays reproducible. It records
        # the EFFECTIVE config (defaults resolved), not the flags as typed.
        run: dict[str, object] = {"with_vision": args.with_vision}
        if args.with_vision:
            run["vision_config"] = _vision_config(
                args.model, args.backend, args.base_url
            ).public_dict()
        print(
            json.dumps(
                {"run": run, "summary": summary, "cases": [vars(r) for r in results]}, indent=2
            )
        )
        return
    if not results:
        print("No verified cases yet. Fill in <id>.gt.txt and set verified = true in cases.toml.")
        return
    category_width = max(len(category) for category in VALID_CATEGORIES)
    for r in results:
        vis = f"{r.vision_wer:.2f}" if r.vision_wer is not None else " -  "
        rec = f"{r.vision_recall:.2f}" if r.vision_recall is not None else " -  "
        f1 = f"{r.vision_f1:.2f}" if r.vision_f1 is not None else " -  "
        flag = "?" if r.flagged is None else ("flag" if r.flagged else "ok")
        print(
            f"  {r.id:30s} category={r.category:{category_width}s} "
            f"legib={r.legibility:10s} "
            f"vis_wer={vis} recall={rec} f1={f1} grounding={flag}"
        )
    print()
    for key, value in summary.items():
        shown = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {key}: {shown}")


if __name__ == "__main__":
    main()
