# Distill

Distill converts local videos and personal-use YouTube videos into structured,
LLM-readable bundles: transcript JSON, keyframes, OCR text, local vision
captions, and a markdown render. Designed for feeding recorded talks, demos, and
screen recordings into LLM agents.

> **Platform:** Apple Silicon (macOS). Local vision uses [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), an MLX inference server.

## Install

> **Package name:** the distribution is published as **`distill-video`** (PyPI / git
> install), but it installs the **`distill`** Python package and a **`distill`**
> console script. The CLI you type is `distill`, not `distill-video`.

Distill needs Python 3.13+ and a few system tools. With [uv](https://docs.astral.sh/uv/) installed:

```bash
# clone
git clone https://github.com/dungle-scrubs/distill.git
cd distill
uv sync
uv run distill list-tools
```

Or add it to an existing uv project:

```bash
uv add "distill-video @ git+https://github.com/dungle-scrubs/distill.git"
```

### System dependencies

Install these with your package manager (e.g. [Homebrew](https://brew.sh) on macOS):

| Tool | Purpose | Install |
| --- | --- | --- |
| `ffmpeg` / `ffprobe` | frame extraction, duration probing | `brew install ffmpeg` |
| `tesseract` | OCR fallback / grounding cross-check | `brew install tesseract` |
| `yt-dlp` | YouTube download | `brew install yt-dlp` |
| `rapid-mlx[vision]` | local vision server (see below) | `brew install raullenchai/rapid-mlx/rapid-mlx` then `pip install 'rapid-mlx[vision]'` |

If any are missing, Distill degrades gracefully: no vision server means
OCR-only captions, no tesseract means vision-only captions, etc.

## Quickstart

```bash
# local file
distill process-local-video ./demo.mp4

# personal-use YouTube video
distill process-youtube-video "https://youtu.be/..."

# a whole directory (recursive)
distill process-video-directory ./recordings --recursive --max-items 10

# prune old cache bundles
distill cleanup-cache --keep-generations 3 --dry-run
```

Each run writes a bundle: `transcript.json`, `frames/*.png`, a `video.md`
render, and a `_manifest.json`. Output lands under `~/.distill` by default;
override with `--output-dir`.

## Commands

`distill` exposes the following subcommands. All print JSON to stdout on
success; fatal errors print a JSON error object to stderr and exit with code 2.

| Command | Purpose |
| --- | --- |
| `process-local-video PATH` | Process one local video file into a bundle. |
| `process-youtube-video URL` | Download and process one personal-use YouTube video. |
| `process-video-directory PATH` | Process every video in a directory. Flags: `--recursive`, `--max-items`, `--continue-on-error/--no-continue-on-error`, `--output-dir`, `--job-id`. |
| `process-youtube-playlist URL` | Process videos from a YouTube playlist or channel URL. Flags: `--max-items`, `--continue-on-error/--no-continue-on-error`, plus the processing options. |
| `cleanup-cache` | Prune old cache bundles. Flags: `--output-dir`, `--max-age-days`, `--keep-generations`, `--dry-run/--no-dry-run`. |
| `get-job-status JOB_ID` | Read a Distill job status record. Flags: `--output-dir`. |
| `list-tools` | Print the registered tool names as JSON. |
| `timeout-diagnostics` | Show the configured vs. effective timeout (assumption A-004). |
| `timeout-probe PROBE_MS` | Sleep for a bounded timeout probe; long probes require `DISTILL_ENABLE_LONG_TIMEOUT_PROBE=1`. |
| `local-vision-diagnostics` | Probe the local Rapid-MLX vision server and print the resolved config. Flags: `--caption-frames/--no-caption-frames`, `--local-vision-backend`, `--local-vision-model`, `--local-vision-base-url`, `--local-vision-timeout-sec`. |
| `call-tool TOOL [--args JSON]` | Call any registered tool by its MCP-style name with a JSON arguments object. |

The processing commands (`process-local-video`, `process-youtube-video`,
`process-youtube-playlist`) share these options: `--whisper-model`,
`--whisper-language`, `--ocr`/`--no-ocr`, `--ocr-language`, `--ocr-preprocess`,
`--redact-secrets`/`--no-redact-secrets`, `--max-keyframes`, `--min-interval-sec`,
`--max-duration-sec`, `--vad-filter`/`--no-vad-filter`, `--max-static-window-sec`,
`--output-dir`, `--force-reprocess`, `--job-id`, `--resume-partial`,
`--caption-frames`/`--no-caption-frames`, `--local-vision-backend`,
`--local-vision-model`, `--local-vision-base-url`, and
`--local-vision-timeout-sec`. Run `distill <command> --help` for the full list.

## Local Vision

Frame interpretation talks to a local **Rapid-MLX** server directly over its
OpenAI-compatible HTTP API. Distill does not manage the server lifecycle — start
it yourself and Distill will probe it and post image prompts:

```bash
# Install the vision extra (required for VLM models)
pip install 'rapid-mlx[vision]'

# Serve the model
rapid-mlx serve mlx-community/Qwen3-VL-8B-Instruct-8bit
```

By default Distill targets `http://127.0.0.1:8000/v1` (Rapid-MLX's default
port). Override the endpoint with `--local-vision-base-url`. If the server is
down or the configured model is not loaded, Distill degrades to OCR-only output
rather than failing the run.

### Model

**`Qwen3-VL-8B-Instruct` at 8-bit (`mlx-community/Qwen3-VL-8B-Instruct-8bit`) is
the eval-chosen reader.** A human-verified 16-frame text-recovery eval
(`tests/evals/README.md`) found it the strongest reader at token recall 0.91 /
WER 0.13. Key findings:

- **8-bit clearly beats 4-bit** — lower quantization loses readable text.
- **Bigger is not better** — Qwen3-VL-30B gave no accuracy gain over 8B; a real
  jump would require a frontier cloud reader, not a larger local one.
- **MLX over Ollama** — ~28% faster on Apple Silicon at equal quality.
- **OCR specialists are not worth it** — Tesseract ≈ PaddleOCR-VL (~0.53 raw-text
  recall), both dwarfed by the VLM's 0.91.

Per-call `--local-vision-model` and `--local-vision-base-url` overrides are
honored. Distill talks only to Rapid-MLX — no other runtime shims.

## Testing

```bash
# full offline suite (hermetic, no real servers)
uv run pytest

# with coverage
uv run pytest --cov

# the eval (needs Rapid-MLX running)
uv run python tests/evals/score.py --with-vision
```

The default suite never hits a real server: vision tests fake Rapid-MLX by
monkeypatching `distill.local_vision._urlopen_json`. The live vision smoke test
is gated behind `DISTILL_RUN_RAPID_MLX_SMOKE=1`.

## License

[MIT](LICENSE) © Kevin Frilot and Distill contributors.
