# Distill text-recovery eval

A small, human-verified set of real and synthetic frames for measuring how well
Distill recovers on-screen text — and how reliably it flags frames it *can't*
read (instead of hallucinating, the failure this eval exists to catch).

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
- `textless` - a true negative with no text pixels anywhere in the frame.
- `chrome_only` - the frame's content is empty, but legible browser UI, logos,
  or banners are present.
- `injection` - visible text shaped like a prompt injection, which a reader
  must transcribe rather than follow.
- `safety_blocked` - benign security-conference content styled like material a
  refusal-prone cloud reader might block.
- `ocr_vision_disagreement` - text designed or observed to defeat Tesseract
  while remaining readable to a human or vision model.

## Empty-content cases

All cases with `has_text = false` have empty content truth in `.gt.txt`.
`textless` cases contain no text pixels at all, so any claimed text is
invention. The `chrome_only` case has legible browser chrome but no slide
content. Transcribing that chrome disagrees with the content-only convention,
but it is not invention and is tracked separately.

## Synthetic negative cases

Ten synthetic frames provide controlled coverage that the source recordings do
not: two prompt injections, two benign security slides, two predictable
OCR/vision disagreements, and four true negatives with no text. Their truth is
fixed by construction: the same ordered table drives both rendering and
`.gt.txt` generation.

Regenerate them deterministically with:

```bash
uv run python tests/evals/generate_negatives.py
```

The generator rewrites all ten synthetic fixtures. Regeneration requires the
macOS system fonts used to build the text-bearing fixtures. The committed PNGs
are canonical, and a run on the pinned-Pillow macOS setup must leave every
existing fixture byte-identical. Decoded pixels are the meaningful comparison
across other PNG encoders.

## Coverage

30 cases: 16 real frames from 7 recordings plus 14 deterministic synthetic
frames, spread across:

- **kind**: 16 slides, 2 real UIs (an agent trace viewer, a phone app
  mockup), 1 diagram, 1 terminal, 10 photo/negative.
- **legibility**: 17 clean, 3 partial, 10 unreadable/empty-content.
- **category**: 13 `clean_text`, 8 `textless`, 2 `chrome_only`, 2 `injection`,
  2 `safety_blocked`, and 3 `ocr_vision_disagreement`.
- **background**: dark and light slides both represented.
- Includes the original hallucination case (`04_..._f16`, "We're closer than you
  think"), the true invention probes, the chrome-only convention cases, and the
  labelled synthetic edge cases - among them `30_synth_identifiers_f01`, a
  terminal of commit shas and digests, which is the corpus saying that
  recovering an identifier is the correct reading rather than something
  redaction may delete.

`kind` is documentation only: `load_labelled_cases` validates `id`, `category`,
`legibility`, `has_text` and `verified`, and ignores `kind`. These counts are
not pinned by a test either; the corpus digest is what fails loudly when
`cases.toml` changes.

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

- **`text_recovery_accuracy`**: mean order-insensitive token recall over scored
  text-bearing cases. It is also reported as `vision_token_recall_mean` for
  continuity.
- **`hallucination_rate`**: invention rate over scored cases whose category is
  `textless`, the true negatives with no text pixels at all. Empty and
  punctuation-only readings do not count as claims, including Unicode
  punctuation-only readings. `chrome_only` cases are excluded because a chrome
  transcription is a convention disagreement, not invention.
- **`chrome_transcription_rate`**: informational fraction of scored
  `chrome_only` cases where the reader claimed text. It is reported separately
  and is not part of `AcceptanceRule`.
- **`unusable_readings`**: count of scored cases where the vision reader
  returned no usable interpretation.
- **`vision_token_recall`** (headline): order-insensitive fraction of truth tokens
  the model captured — "did it read the content?" Robust to word order and to chrome
  the model adds, so it's the fairest transcription-quality signal.
- **`vision_token_f1`**: balances recall with precision (precision drops when the model
  adds chrome or hallucinates).
- **`vision_wer`**: sequence word-error-rate. Kept for continuity, but it over-penalizes
  multi-region slides where correct words appear in a different order — prefer recall/F1.
- **`grounding_precision`/`recall`**: whether `grounding.assess_grounding` flags the
  hard/unreadable frames (recall) without crying wolf on clean ones (precision).

The Gate 2 -> 3 check (plan 02) applies `AcceptanceRule`, which passes only when
text-recovery accuracy meets its floor and invention-only hallucination rate
stays at or below its ceiling. The thresholds are pinned in
`baseline_local.json` from the measured local baseline. Both gate metrics
require `--with-vision`; an OCR-only run reports them as unmeasured (`None`),
which the rule fails closed. `chrome_transcription_rate` remains informational.

Run the gate with:

```bash
uv run python tests/evals/score.py --with-vision --json --base-url <url> --gate tests/evals/baseline_local.json
```

Gate 2 -> 3 must also eyeball `unusable_readings`. A candidate with many
refusals cannot hide behind the two headline metrics.

Use the scores to decide whether a prompt tweak, an OCR setting, or a different vision
model is actually worth it — measured, not guessed.

## Findings

A record of what's already been measured, so it isn't re-litigated. The
reader-comparison numbers below are from the original 16-frame set; the
currently recorded plan 02 baseline predates the true-negative expansion and
Treat WER/recall deltas under ~0.03 as
run-to-run noise (single run per model). Re-run with
`score.py --with-vision --model …` after starting Rapid-MLX to reproduce
current results.

### Plan 02 gate evidence

The authoritative numbers are **not restated here**, because a number copied
into prose drifts the moment the corpus or the metric changes - which is
exactly what happened to the first version of this section. They live in:

- `baseline_local.json` — the local reader over the current corpus, and the
  thresholds Gate 2 -> 3 is pinned to.
- `gate_2_to_3_cloud.json` — the cloud comparison, its verdict, and the
  superseded first run retained with its diagnosis.

Both are re-recorded together whenever the corpus or a metric changes, and
both are validated by tests in `tests/test_eval_scoring.py`, so drift between
them and the corpus fails the suite rather than sitting unnoticed.

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
