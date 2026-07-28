"""Score Distill's text recovery against the hand-written eval ground truth.

Reads ``cases.toml`` and each frame's ``<id>.gt.txt`` and reports two things:

1. Word-error-rate (WER) of recovered text vs. truth, for text-bearing frames.
   Preprocessed OCR is always scored; the vision model's ``verbatim_text`` is
   scored too when ``--with-vision`` is set and Rapid-MLX is running.
2. Grounding-flag accuracy: whether ``grounding.assess_grounding`` flags the
   frames a human marked hard/unreadable (``legibility != "clean"``) and stays
   quiet on clean ones — precision/recall over that label.

Unverified or empty cases are skipped, so the score reflects only frames a human
has actually confirmed. Run:

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
    legibility: str
    has_text: bool
    ocr_wer: float | None
    vision_wer: float | None
    vision_recall: float | None  # order-insensitive token recall (content captured)
    vision_f1: float | None
    flagged: bool | None  # grounding marked it low-confidence
    should_flag: bool  # human says it is not cleanly legible


def _load_cases() -> list[dict]:
    data = tomllib.loads((EVAL_ROOT / "cases.toml").read_text())
    return data.get("case", [])


def _truth(case_id: str) -> str | None:
    """Return confirmed truth text, or None if the file is missing/unverified.

    Lines starting with ``#`` are notes (e.g. provenance) and are stripped before
    scoring. The ``UNVERIFIED`` sentinel and ``verified = false`` in cases.toml
    both keep a case out of the score until a human has confirmed it.
    """
    path = FRAMES_DIR / f"{case_id}.gt.txt"
    if not path.exists():
        return None
    if "UNVERIFIED" in path.read_text():
        return None
    body = "\n".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    return body.strip()


def evaluate(
    with_vision: bool, model: str | None = None, backend: str | None = None
) -> list[CaseResult]:
    # The same lookup that gates this function must also drive the OCR call.
    # find_tesseract_command() falls back to well-known install paths that are
    # not necessarily on PATH; if the guard used it but ocr_frame resolved
    # "tesseract" from PATH on its own, OCR would silently return nothing and
    # the eval would hard-fail at WER 1.00 instead of skipping.
    tesseract_cmd = find_tesseract_command()
    if not tesseract_cmd:
        raise SystemExit("tesseract not found; install it before scoring OCR")
    interpret = _vision_interpreter(model, backend) if with_vision else None
    results: list[CaseResult] = []
    for case in _load_cases():
        case_id = str(case["id"])
        if not bool(case.get("verified", False)):
            continue
        truth = _truth(case_id)
        if truth is None:
            continue
        image = FRAMES_DIR / f"{case_id}.png"
        ocr_text, _ = ocr_frame(image, "eng", tesseract_cmd=tesseract_cmd)
        has_text = bool(case.get("has_text", True))
        ocr_wer = word_error_rate(truth, ocr_text) if has_text else None

        vision_wer: float | None = None
        vision_recall: float | None = None
        vision_f1: float | None = None
        flagged: bool | None = None
        if interpret is not None:
            result = interpret(image, ocr_text)
            if result is not None:
                if has_text:
                    vision_wer = word_error_rate(truth, result.verbatim_text)
                    _, vision_recall, vision_f1 = token_prf(truth, result.verbatim_text)
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

        results.append(
            CaseResult(
                id=case_id,
                legibility=str(case.get("legibility", "")),
                has_text=has_text,
                ocr_wer=ocr_wer,
                vision_wer=vision_wer,
                vision_recall=vision_recall,
                vision_f1=vision_f1,
                flagged=flagged,
                should_flag=str(case.get("legibility", "")) != "clean",
            )
        )
    return results


def _vision_interpreter(model: str | None = None, backend: str | None = None):
    from distill.local_vision import LocalVisionConfig, probe_local_vision, try_interpret_image
    from distill.vision_prompts import build_technical_frame_prompt

    overrides: dict[str, str] = {}
    if model:
        overrides["model"] = model
    if backend:
        overrides["backend"] = backend
    config = replace(LocalVisionConfig(), **overrides) if overrides else LocalVisionConfig()
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


def summarize(results: list[CaseResult]) -> dict:
    ocr_wers = [r.ocr_wer for r in results if r.ocr_wer is not None]
    vision_wers = [r.vision_wer for r in results if r.vision_wer is not None]
    vision_recalls = [r.vision_recall for r in results if r.vision_recall is not None]
    vision_f1s = [r.vision_f1 for r in results if r.vision_f1 is not None]
    flag_known = [r for r in results if r.flagged is not None]
    true_pos = sum(1 for r in flag_known if r.flagged and r.should_flag)
    flagged_total = sum(1 for r in flag_known if r.flagged)
    should_total = sum(1 for r in flag_known if r.should_flag)
    return {
        "cases_scored": len(results),
        "ocr_wer_mean": _mean(ocr_wers),
        "vision_wer_mean": _mean(vision_wers),
        "vision_token_recall_mean": _mean(vision_recalls),
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
    parser.add_argument("--json", action="store_true", help="emit JSON for programmatic use")
    args = parser.parse_args()

    results = evaluate(args.with_vision, args.model, args.backend)
    summary = summarize(results)
    if args.json:
        print(json.dumps({"summary": summary, "cases": [vars(r) for r in results]}, indent=2))
        return
    if not results:
        print("No verified cases yet. Fill in <id>.gt.txt and set verified = true in cases.toml.")
        return
    for r in results:
        vis = f"{r.vision_wer:.2f}" if r.vision_wer is not None else " -  "
        rec = f"{r.vision_recall:.2f}" if r.vision_recall is not None else " -  "
        f1 = f"{r.vision_f1:.2f}" if r.vision_f1 is not None else " -  "
        flag = "?" if r.flagged is None else ("flag" if r.flagged else "ok")
        print(
            f"  {r.id:30s} legib={r.legibility:10s} vis_wer={vis} recall={rec} f1={f1} grounding={flag}"
        )
    print()
    for key, value in summary.items():
        shown = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {key}: {shown}")


if __name__ == "__main__":
    main()
