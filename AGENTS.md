# AGENTS.md

Guidance for agents (and humans) working in this repository.

## Local vision: Rapid-MLX, direct

Distill's frame interpretation talks to a local **Rapid-MLX** server directly
over its OpenAI-compatible HTTP API. This is the only supported local vision
backend.

- **Default model**: `mlx-community/Qwen3-VL-8B-Instruct-8bit` (Qwen3-VL-8B at
  8-bit). This is the eval-chosen reader.
- **Default endpoint**: `http://127.0.0.1:8000/v1` (Rapid-MLX's default port).
- **Start the server yourself** — Distill assumes it is already running:
  ```bash
  rapid-mlx serve mlx-community/Qwen3-VL-8B-Instruct-8bit
  ```
- Distill probes availability with `GET <base_url>/models` and posts
  chat-completion requests to `POST <base_url>/chat/completions` using only the
  stdlib `urllib` (no new runtime dependency).
- If the server is down or the configured model is not loaded, Distill degrades
  to OCR-only output rather than failing the run.

### Single vision backend

- **Rapid-MLX is the only local vision path.** Distill owns the HTTP call
  directly via the stdlib `urllib` — no lifecycle/leasing/proxy layer, no
  alternative runtime shims. Do not add backend branches for other providers.
- **No `local_vision_provider` option.** `backend` is fixed to `rapid-mlx` and
  `model` selects the served model. Overrides are `--local-vision-model` and
  `--local-vision-base-url`.

### Why this model (eval rationale)

A human-verified 16-frame text-recovery eval (`tests/evals/`) selected the
8-bit reader. Do not change the default without re-running the eval:

- **8-bit over 4-bit** — quantization directly costs readable text.
- **Bigger is not better** — Qwen3-VL-30B gave no accuracy gain; a real jump
  would require a frontier *cloud* reader, not a larger local one.
- **MLX over Ollama** — ~28% faster on Apple Silicon at equal quality.
- **OCR specialists aren't worth a backend** — Tesseract ≈ PaddleOCR-VL (~0.53
  raw recall), both dwarfed by the VLM's 0.91.

Reproduce with `uv run python tests/evals/score.py --with-vision`.

## Branching

The goal is a single branch, `main`. Do not create long-lived feature branches.
Land changes directly on `main`; keep the working tree green (`uv run pytest`).

## Pipeline signature

`src/distill/version.py` carries `PIPELINE_VERSION`, `PIPELINE_SIGNATURE`
(hash of output-affecting modules), `SIGNED_MODULES`, and `EXEMPT_MODULES` (the
exemption table: module name -> the recorded reason it cannot change bundle
content). Together those two tables record a disposition for **every** module
under `src/distill/`, each keyed by its path *relative to* `src/distill/` in
posix form (`vision/prompts.py`, not `prompts.py`).

The released package version lives apart, in `src/distill/release.py`, and is
**signed**: `DISTILL_VERSION` is stamped into every manifest, so changing it
changes bundle content. `version.py` cannot be signed, because it holds the
signature and cannot hash itself.

Signed-ness is a property, not a list: if editing a module can change bundle
content, it is a signed module. `test_pipeline_signature` enumerates
`src/distill/**/*.py` from disk - recursively, so a module in a future
subpackage is covered - and fails when a module is neither signed nor exempted,
so a new module forces an explicit decision: sign it, or exempt it with a
specific reason. A generic reason is not a reason, and a reason that is no
longer true is a defect.

Any change to `local_vision.py`, `options.py`, `pipeline.py`, or the other
signed modules requires **both** recomputing `PIPELINE_SIGNATURE` **and**
bumping `PIPELINE_VERSION` (then appending the new `signature: version` pair to
`tests/pipeline_signature_history.json`) so `test_pipeline_signature` stays
green. To check the tables and recompute the signature:

```bash
uv run python -c "
import hashlib
from pathlib import Path
from distill.version import EXEMPT_MODULES, SIGNED_MODULES
package = Path('src/distill')
classified = set(SIGNED_MODULES) | set(EXEMPT_MODULES)
on_disk = {
    p.relative_to(package).as_posix()
    for p in package.rglob('*.py')
    if '__pycache__' not in p.relative_to(package).parts
}
unclassified = sorted(on_disk - classified)
assert not unclassified, f'neither signed nor exempted: {unclassified}'
digest = hashlib.sha256()
for name in SIGNED_MODULES:
    digest.update((package / name).read_bytes())
print(digest.hexdigest())
"
```

The mechanism detects unversioned change to Distill's own source. It does not
detect output changes from package data, a dependency upgrade, or an external
tool, and it is not a security control.

## Tests

- Unit tests run offline and hermetic: `uv run pytest`.
- Vision tests fake Rapid-MLX by monkeypatching `distill.local_vision._urlopen_json`
  (for the `try_interpret_image` path) or by passing a `requestor=` callable to
  `probe_rapid_mlx_availability` / `_interpret_with_rapid_mlx`. Never hit a real
  server in the default suite.
- The live smoke test is gated behind `DISTILL_RUN_RAPID_MLX_SMOKE=1`.
