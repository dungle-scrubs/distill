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

| Tool | Capability | Class | Install |
| --- | --- | --- | --- |
| `ffmpeg` | audio extraction and keyframe extraction | required | `brew install ffmpeg` |
| `ffprobe` | source duration probing | required | `brew install ffmpeg` |
| `yt-dlp` | YouTube source acquisition and metadata | required | `brew install yt-dlp` |
| `tesseract` | image-text extraction from keyframes | optional | `brew install tesseract` |
| `rapid-mlx[vision]` | local vision server (see below) | optional | `pip install 'rapid-mlx[vision]'` |

The class decides what an absent tool costs. An **optional** one degrades: the
run records a warning naming what is missing and continues, so no vision server
means OCR-only captions and no tesseract means vision-only captions. A
**required** one is fatal: the run stops immediately, naming the tool and what
its absence costs, rather than doing the work and producing a bundle with
nothing in it. Per tool, an absence costs:

| Tool | Class | What its absence costs |
| --- | --- | --- |
| `ffmpeg` | required | no audio can be extracted and no keyframe can be captured, which leaves a generation with neither a transcript nor frame artifacts and no usable bundle to publish |
| `ffprobe` | required | the source's duration cannot be read, so keyframe timestamps and the duration cap have nothing to work from and the run ends before any stage produces output |
| `yt-dlp` | required | the source cannot be acquired at all, so a YouTube run has nothing to process |
| `tesseract` | optional | keyframes contribute no extracted text, so interpretations cannot be corroborated and grounding falls back to the vision model alone; the transcript, keyframes and render are unaffected |

Distill never installs any of them - an absent optional tool is a warning, not a
`brew install` Distill runs on your behalf. `src/distill/capabilities.py` holds
the table both tables above state in prose: every external tool Distill runs is
listed with the class and the absence cost that table records, and the test
suite fails if they disagree. The vision server is not a tool Distill runs, so
it is described here only; a run without it degrades the same way, on the
strength of the local-vision tests rather than that table.

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

## Output layout

Output lands under `~/.cache/distill` by default; override the root with
`--output-dir`. Each run publishes one **generation** into the **bundle** that
its **bundle key** names:

```
~/.cache/distill/
├── _jobs/                   # job status records
├── _locks/                  # one lock per bundle key, held for a run
├── _youtube_locks/          # one lock per source being acquired (YouTube runs)
├── _youtube_sources/        # downloaded media (YouTube runs)
└── <bundle key>/            # one bundle
    ├── _owner.json          # bundle marker: this directory is Distill's
    ├── _manifest.json       # names the active generation
    ├── g1/                  # an older generation, kept until pruned
    └── g2/                  # the active generation
        ├── video.md         # the render
        ├── transcript.json
        └── frames/          # keyframe images
```

Read the generation `_manifest.json` names as `active_generation`; the others
are older renderings kept until pruned, and a directory carrying no marker is
not a bundle and is never pruned. `_manifest.json` also carries the run's
warnings, folded so that one record with an `occurrences` count stands for
repeated events, alongside `warning_count`, which counts the events rather than
the records.

An output root must be a subdirectory of `$HOME` or of the system temp
directory - neither directory itself - and Distill refuses one inside a
sensitive path such as `~/.ssh` or `~/.aws`. `~/.distill` is the *config*
directory (override with `DISTILL_CONFIG_DIR`), not an output root.

### How a source is matched to its bundle

A **bundle key** combines a **source fingerprint** (which source this is) with an
**options hash** (how it will be processed), so changing an option that can
change output produces a different bundle rather than overwriting one. The
pipeline version participates in the options hash, so a release that changes
what the pipeline produces re-keys every bundle instead of serving stale output.

A YouTube source is fingerprinted by its video id. For a local file, the default
`fingerprint` cache mode samples: it hashes the file's size, its mtime in
nanoseconds, and 64 KiB read at each of nine anchors - the head, the tail, and
seven evenly spread interior offsets - which bounds a cache lookup at 576 KiB of
reading regardless of how large the source is. The anchors are a function of the
size alone, so they are deterministic and public; a file too small to span them
reports fewer, because coinciding offsets collapse into one.

What that buys, and what it costs, is worth stating plainly: two *distinct*
files collide iff they share a size, an mtime to the nanosecond, and the bytes
at every anchor. Independently produced videos do not - differing content gives
differing sizes, and mtimes are not byte-identical by accident - but such a
pair can be constructed deliberately by anyone who can write both files,
leaving every anchor untouched and differing everywhere else. A colliding pair
shares a bundle key, so one source would be served the other's bundle.

Where sources are untrusted or adversarial, ask for the `content` cache mode,
which hashes every byte and so distinguishes any two files that differ at all;
it refuses files over 5 GB, because reading one is not a cost a cache lookup may
impose. It has no CLI flag - it is a tool argument:

```bash
distill call-tool process_local_video \
  --args '{"path": "./demo.mp4", "cache_mode": "content"}'
```

## Commands

`distill` exposes the following subcommands. All print JSON to stdout on
success; every fatal error prints a JSON error object to stderr - carrying its
`code`, `stage`, `message` and `details` - and exits with code 2, with nothing
on stdout. An argument the parser rejects is argparse's usage message, also exit
2. Set `DISTILL_TRACEBACK=1` to have an unexpected failure re-raised with its
Python traceback instead of being converted; it is a debugging aid, not a
contract.

| Command | Purpose |
| --- | --- |
| `process-local-video PATH` | Process one local video file into a bundle. |
| `process-youtube-video URL` | Download and process one personal-use YouTube video. |
| `process-video-directory PATH` | Process every video in a directory. Flags: `--recursive`, `--max-items`, `--continue-on-error/--no-continue-on-error`, `--output-dir`, `--job-id`. |
| `process-youtube-playlist URL` | Process videos from a YouTube playlist or channel URL. Flags: `--max-items`, `--continue-on-error/--no-continue-on-error`, plus the processing options. |
| `cleanup-cache` | Prune old cache bundles. Flags: `--output-dir`, `--max-age-days`, `--keep-generations`, `--dry-run/--no-dry-run`. |
| `cache-doctor` | Report what is under an output root - bundles, active generations, orphan generations, locks, a prune preview - and change nothing. Flags: `--output-dir`, `--max-age-days`, `--keep-generations`. |
| `get-job-status JOB_ID` | Read a Distill job status record. Flags: `--output-dir`. |
| `list-tools` | Print the registered tool names as JSON. |
| `timeout-diagnostics` | Show the configured vs. effective timeout (assumption A-004). |
| `timeout-probe PROBE_MS` | Sleep for a bounded timeout probe; long probes require `DISTILL_ENABLE_LONG_TIMEOUT_PROBE=1`. |
| `local-vision-diagnostics` | Probe the local Rapid-MLX vision server and print the resolved config. Flags: `--caption-frames/--no-caption-frames`, `--local-vision-backend`, `--local-vision-model`, `--local-vision-base-url`, `--local-vision-timeout-sec`, `--local-vision-allow-remote-endpoint/--no-local-vision-allow-remote-endpoint`. |
| `call-tool TOOL [--args JSON]` | Call any registered tool by its MCP-style name with a JSON arguments object. |

The processing commands (`process-local-video`, `process-youtube-video`,
`process-youtube-playlist`) share these options: `--whisper-model`,
`--whisper-language`, `--ocr`/`--no-ocr`, `--ocr-language`, `--ocr-preprocess`,
`--redact-secrets`/`--no-redact-secrets`, `--max-keyframes`, `--min-interval-sec`,
`--max-duration-sec`, `--vad-filter`/`--no-vad-filter`, `--max-static-window-sec`,
`--output-dir`, `--force-reprocess`, `--job-id`, `--resume-partial`,
`--caption-frames`/`--no-caption-frames`, `--local-vision-backend`,
`--local-vision-model`, `--local-vision-base-url`,
`--local-vision-timeout-sec`, and `--local-vision-allow-remote-endpoint`. Run
`distill <command> --help` for the full list.

Each numeric option has a domain, and a value outside it is refused with
`E_BAD_OPTIONS` naming the option rather than taken as written. The domain is a
finite quantity above zero, with three exceptions: `--min-interval-sec` also
accepts `0`, which asks for no gap between keyframes; `--max-static-window-sec`
has a floor of `0.001`, the millisecond a keyframe timestamp is rounded to, so a
narrower window is refused rather than naming a schedule that cannot be
expressed; and `--local-vision-timeout-sec` has a ceiling of what a socket
timeout can hold. `--max-keyframes` and `--max-items` are counts, so a negative
one is refused rather than read as a slice from the end.

A batch run (`process-video-directory`, `process-youtube-playlist`) reports each
failed item as the whole error record - `code`, `stage`, `message`, `details` -
plus the `batch_index` it failed at, so an item that a re-run would pick up can
be told from one that will fail the same way forever.

### Upgrading a cache written by 0.1.0

Run `cache-doctor` before the first `cleanup-cache` of an existing cache
directory. 0.1.0 had a retention bug that could delete the generation its own
manifest named as active, and a bundle left in that state - a manifest naming a
generation that is not on disk - is not a bundle anything can serve. Cleanup now
treats every generation under such a bundle as an orphan and proposes all of
them, so readable generations that the manifest does not name can be removed by
that first run. `cache-doctor` lists them, and `cleanup-cache --dry-run` shows
exactly what would go; copy anything you still want out first. Reprocessing the
source rebuilds the bundle either way.

## Local Vision

Frame interpretation talks to a local **Rapid-MLX** server directly over its
OpenAI-compatible HTTP API. Distill does not manage the server lifecycle - start
it yourself and Distill will probe it and post image prompts:

```bash
# Install the vision extra (required for VLM models)
pip install 'rapid-mlx[vision]'

# Serve the model
rapid-mlx serve mlx-community/Qwen3-VL-8B-Instruct-8bit
```

By default Distill targets `http://127.0.0.1:8000/v1` (Rapid-MLX's default
port). Override the endpoint with `--local-vision-base-url`. If the server is
down, the configured model is not loaded, or the endpoint is refused, Distill
degrades to OCR-only output rather than failing the run, and records a warning
naming which of those it was.

### Which endpoint Distill will speak to

A vision endpoint is handed **extracted text** and images from your sources, so
where it points is a decision rather than a default:

- **Loopback only, unless you say otherwise.** Every address the host resolves
  to must be loopback, checked per request rather than once at configuration
  time. Pass `--local-vision-allow-remote-endpoint` (or set
  `"allow_remote_endpoint": true` in the local-vision config) to reach a server
  elsewhere deliberately.
- **`http` or `https` only.** No other scheme is accepted, with or without the
  opt-out.
- **Redirects are not followed**, and a response body beyond 32 MiB is refused.
  Both hold wherever the endpoint is: they are about what the client does with
  an answer, not about whose answer it is.

A refused endpoint is a warning and OCR-only output, not a failed run.

### Model

**`Qwen3-VL-8B-Instruct` at 8-bit (`mlx-community/Qwen3-VL-8B-Instruct-8bit`) is
the eval-chosen reader.** A human-verified 16-frame text-recovery eval
(`tests/evals/README.md`) found it the strongest reader at token recall 0.91 /
WER 0.13. Key findings:

- **8-bit clearly beats 4-bit** - lower quantization loses readable text.
- **Bigger is not better** - Qwen3-VL-30B gave no accuracy gain over 8B; a real
  jump would require a frontier cloud reader, not a larger local one.
- **MLX over Ollama** - ~28% faster on Apple Silicon at equal quality.
- **OCR specialists are not worth it** - Tesseract ≈ PaddleOCR-VL (~0.53 raw-text
  recall), both dwarfed by the VLM's 0.91.

Per-call `--local-vision-model` and `--local-vision-base-url` overrides are
honored. Distill talks only to Rapid-MLX - no other runtime shims.

## Extracted text is data, not instruction

A render is written to be fed to an LLM agent, and everything recovered from a
source - transcript text, image text, vision interpretations, link labels - was
chosen by whoever produced that source. Distill therefore treats it as untrusted
data at both places it becomes durable:

- **The render marks it.** Every piece of extracted text sits inside a delimiter
  it cannot terminate, under a preamble telling the reader that what follows is
  data and not instructions. The preamble says so in those terms: it is a
  mitigation, not a guarantee, and a downstream model can still be talked into
  ignoring it.
- **Secret-shaped values are redacted** before the text is written to disk or
  placed into a render - bearer tokens, `token:`/`apikey:` assignments, GitHub,
  GitLab, npm and HuggingFace token formats, and private-key headers. This is
  pattern matching over recovered text and nothing more: a secret in a shape no
  pattern names survives it, and `--no-redact-secrets` turns it off entirely. It
  reduces what leaks into a bundle; it does not make a bundle safe to publish.

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
