# Distill text-recovery eval

A small, human-verified set of real and synthetic frames for measuring how well Distill
recovers on-screen text — and how reliably it flags frames it *can't* read
(instead of hallucinating, the failure this eval exists to catch).

## Layout

```
tests/evals/
  cases.toml            # one [[case]] per frame, including its labelled category
  frames/<id>.png       # the frame image (committed fixture)
  frames/<id>.gt.txt    # the ground-truth transcription you confirm
  generate_negatives.py # dev tool: deterministically regenerate synthetic cases
  triage.py             # dev tool: survey the cache and propose a stratified set
  score.py              # the scorer (OCR + optional vision WER, grounding precision/recall)
```

## Your job: confirm the ground truth

The `.gt.txt` files contain **DRAFT** transcriptions Claude read off each frame.
Drafts can be wrong — that's exactly the fallibility we're measuring — so nothing
counts until you confirm it. For each case:

1. Open `frames/<id>.png` and read it yourself.
2. Fix `frames/<id>.gt.txt` so it holds the **exact legible on-screen text**.
3. Set `verified = true` for that case in `cases.toml`.

The scorer skips any case that is unverified or whose `.gt.txt` still contains
the word `UNVERIFIED`, so you can verify the set incrementally.

## Transcription rules (keep these consistent)

- **Reading order**: top→bottom; for multiple columns, finish the left column
  before the right.
- **Include**: titles, kickers, bullets, body, the key claim, code, terminal
  text, axis/legend labels, diagram node labels.
- **Exclude recurring chrome**: the speaker webcam, venue/sponsor logos
  (e.g. "AI Engineer EUROPE"), and page numbers ("19 / 37"). Transcribe the
  *content*, not the frame furniture.
- **Embedded screenshots**: if a slide contains a tiny screenshot whose text is
  unreadable, don't transcribe it; note it in a `#` comment and mark the case
  `legibility = "partial"`.
- **Can't read a word?** Leave it out and set `legibility = "partial"`. Can't
  read anything? Empty body and `legibility = "unreadable"`.
- **Don't normalize by hand** — write natural capitalization and punctuation.
  The scorer lowercases, strips punctuation, and collapses whitespace before
  computing word-error-rate, so "GitHub" vs "github" is not counted as an error.
- Lines starting with `#` are notes (provenance, caveats) and are ignored by the
  scorer.

## Labelled categories

Every case has a required `category` field. The scorer validates the label and
the case's `.png` and `.gt.txt` fixtures before doing any OCR:

- `clean_text` - an ordinary text-bearing frame.
- `textless` - a true negative with no on-screen text.
- `injection` - visible text shaped like a prompt injection, which a reader
  must transcribe rather than follow.
- `safety_blocked` - benign security-conference content styled like material a
  refusal-prone cloud reader might block.
- `ocr_vision_disagreement` - text designed or observed to defeat Tesseract
  while remaining readable to a human or vision model.

## True negatives

Cases with `has_text = false` (the black frame, the browser-chrome-only frame,
the podcast shot) have empty `.gt.txt`. They test the *other* failure mode: the
system must **not** invent text and **should** flag low confidence. Don't delete
them — an eval of only text-heavy slides would miss false positives entirely.

## Synthetic negative cases

Six synthetic slides provide controlled coverage that the source recordings do
not: two prompt injections, two benign security slides, and two predictable
OCR/vision disagreements. Their transcriptions are truth by construction: the
same module-level constants drive both rendering and `.gt.txt` generation.

Regenerate them deterministically with:

```bash
uv run python tests/evals/generate_negatives.py
```

## Coverage

22 cases: 16 real frames from 7 recordings plus 6 deterministic synthetic
slides, spread across:

- **kind**: 16 slides, 2 real UIs (an agent trace viewer, a phone app
  mockup), 1 diagram, 3 photo/negative.
- **legibility**: 16 clean, 3 partial, 3 unreadable/no-text.
- **category**: 12 `clean_text`, 3 `textless`, 2 `injection`, 2
  `safety_blocked`, and 3 `ocr_vision_disagreement`.
- **background**: dark and light slides both represented.
- Includes the original hallucination case (`04_..._f16`, "We're closer than you
  think"), three text-free negatives, and six labelled synthetic edge cases.

To refresh or extend the pool: `uv run python tests/evals/triage.py`.

## Running

```bash
# OCR-only (offline): word-error-rate of preprocessed Tesseract vs truth
uv run python tests/evals/score.py

# Also score the local vision model + grounding flags (needs Rapid-MLX running)
uv run python tests/evals/score.py --with-vision

# Compare a different model served by Rapid-MLX
uv run python tests/evals/score.py --with-vision --model qwen3-vl:30b-a3b

# Machine-readable
uv run python tests/evals/score.py --with-vision --json
```

## Metrics

- **`vision_token_recall`** (headline): order-insensitive fraction of truth tokens
  the model captured — "did it read the content?" Robust to word order and to chrome
  the model adds, so it's the fairest transcription-quality signal.
- **`vision_token_f1`**: balances recall with precision (precision drops when the model
  adds chrome or hallucinates).
- **`vision_wer`**: sequence word-error-rate. Kept for continuity, but it over-penalizes
  multi-region slides where correct words appear in a different order — prefer recall/F1.
- **`grounding_precision`/`recall`**: whether `grounding.assess_grounding` flags the
  hard/unreadable frames (recall) without crying wolf on clean ones (precision).

Use the scores to decide whether a prompt tweak, an OCR setting, or a different vision
model is actually worth it — measured, not guessed.

## Findings

A record of what's already been measured, so it isn't re-litigated. Numbers are from
this 16-frame set; treat WER/recall deltas under ~0.03 as run-to-run noise (single run
per model). Re-run with `score.py --with-vision --model …` after starting Rapid-MLX
to reproduce current results.

### Reader comparison (transcription)

| Reader | token recall | vision WER | notes |
|---|---|---|---|
| Rapid-MLX `qwen3-vl:8b` | — | 0.25–0.30 | default local target |
| Rapid-MLX `qwen3-vl:30b-a3b` | — | 0.24 | 3x size, **no quality gain** |
| Rapid-MLX `Qwen3-VL-8B-4bit` | — | 0.27 | quantization misses (e.g. frame 09) |
| **Rapid-MLX `Qwen3-VL-8B-8bit`** | **0.91** | **0.13** | best reader measured so far |
| Rapid-MLX `InternVL3-8B-8bit` | — | 0.26 | doc reputation did not translate; worse |
| Rapid-MLX `PaddleOCR-VL-8bit` | — | n/a | OCR-first; does not emit our JSON contract |

Conclusions:
- **8-bit Qwen3-VL-8B is the strongest reader**; 8-bit clearly beats 4-bit.
- **Bigger is not better** here — 30B gave no gain. A real accuracy jump would need a
  fundamentally different (e.g. frontier cloud) reader, not a larger local one.
- Runtime-level speed depends on the Rapid-MLX server; Distill talks to it directly over
  its OpenAI-compatible HTTP API.

### OCR specialists are not worth a backend

Tested PaddleOCR-VL as a Tesseract replacement (raw-text recall vs ground truth):

- Mean recall **Tesseract 0.53 ≈ PaddleOCR-VL 0.53** — and both are dwarfed by the
  vision model's **0.91**. The VLM's `verbatim_text` is already the best text source, so
  no OCR engine improves transcription.
- The two OCR engines are **complementary, not redundant**: PaddleOCR wins on dark slides
  (frame 01: 0.94 vs Tesseract 0.00), Tesseract wins elsewhere (frame 10: 0.94 vs 0.08).
- The only real value would be as the grounding cross-check's *independent* reader on
  dark slides (where empty Tesseract forces trusting the model's self-confidence). If ever
  needed, the cheap form is a targeted "union OCR" fallback for the cross-check input —
  **not** a full OCR-backend abstraction. Not built; cost (VLM latency, layout failures on
  frames 06/10) outweighs the narrow benefit.
