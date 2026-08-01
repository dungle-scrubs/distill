# Distill

Distill converts local videos and personal-use YouTube videos into structured,
LLM-readable bundles: transcript JSON, keyframes, OCR text, local vision
captions, and a markdown render. Designed for feeding recorded talks, demos, and
screen recordings into LLM agents.

> **Platform:** tested on macOS and Linux. Transcript, image-text extraction,
> keyframes and render need only Python 3.13+, `ffmpeg` and `ffprobe`.
> The **default** vision server, [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX),
> requires Apple Silicon — but vision is optional, and any OpenAI-compatible
> endpoint can serve it instead.

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

### Several vision endpoints

`endpoints` names the **vision endpoints** a run may use, in preference order:

```json
{
  "endpoints": [
    { "model": "gemini-3.6-flash", "base_url": "https://...", "api_key_env": "MY_KEY",
      "allow_remote_endpoint": true },
    { "model": "mlx-community/Qwen3-VL-8B-Instruct-8bit", "base_url": "http://127.0.0.1:8000/v1" }
  ]
}
```

A run uses **exactly one** of them. Order is preference, not fallback in the
degradation sense: moving from a cloud endpoint to a local one produces a full
reading from a different reader, and only losing every endpoint is degradation.

At most four entries, and no two may share a model *and* remoteness — such
entries derive the same **bundle key** and there would be nothing left to tell
their bundles apart. Each entry is read from a fresh default, so entry 2 never
inherits entry 1's credential or its remote opt-in.

**The cache is asked before any endpoint is.** If a bundle for any of the keys
the chain could publish under is already on disk, it is served and no endpoint
is contacted at all — including when the cached bundle came from a
less-preferred entry. An offline machine serves what it already has.

Configuring `endpoints` alongside a top-level `model` or `base_url` is refused:
that names an endpoint twice, in two ways, and no rule for picking a winner
would be anything but a guess.

**One-time reprocess.** Which endpoint produced a reading is part of a bundle's
identity, so upgrading to a version that resolves a chain re-keys existing
bundles and the first run afterwards rebuilds them. Adding a fallback that never
gets used does *not* re-key anything: three chains selecting the same endpoint
produce byte-identical bundles.

`distill local-vision-diagnostics` reports every entry — its outcome, the env
var its credential comes from, and whether that variable is set (never the
value).

### System dependencies

Install these with your platform's package manager. Distill names what it needs
and never installs it for you:

| Tool | Capability | Class | Install |
| --- | --- | --- | --- |
| `ffmpeg` | audio extraction and keyframe extraction | required | your platform's package manager |
| `ffprobe` | source duration probing | required | ships with `ffmpeg` |
| `yt-dlp` | YouTube source acquisition and metadata | required | `uv tool install yt-dlp` |
| `tesseract` | image-text extraction from keyframes | optional | your platform's package manager |
| `rapid-mlx[vision]` | local vision server (see below) | optional | `pip install 'rapid-mlx[vision]'` |

The class decides what an absent tool costs. An **optional** one degrades: the
run records a warning naming what is missing and continues, so no vision server
means OCR-only captions and no tesseract means vision-only captions. A
**required** one is fatal: a run that has to *produce* a **generation** stops
immediately, naming the tool and what its absence costs, rather than doing the
work and producing a bundle with nothing in it.

Both classes are about a run that does the work. A run that hits the cache does
none of it: it serves the **generation** already on disk, so a cached local
bundle is servable with `ffprobe` absent, and a cached YouTube bundle with
`yt-dlp` absent - as long as the video id can be read from the URL itself. A URL
that also names a playlist, or whose id is not exactly eleven id characters, or
that names more than one video, has to be resolved by `yt-dlp` before its
**bundle key** is known, so a run of one needs the tool whether or not the
bundle is already there. Per tool, an absence costs:

| Tool | Class | What its absence costs |
| --- | --- | --- |
| `ffmpeg` | required | no audio can be extracted and no keyframe can be captured, which leaves a generation with neither a transcript nor frame artifacts and no usable bundle to publish |
| `ffprobe` | required | the source's duration cannot be read, so keyframe timestamps and the duration cap have nothing to work from and the run ends before any stage produces output |
| `yt-dlp` | required | the source cannot be acquired at all, so a YouTube run has nothing to process |
| `tesseract` | optional | keyframes contribute no extracted text, so interpretations cannot be corroborated and grounding falls back to the vision model alone; the transcript, keyframes and render are unaffected |

Distill never installs any of them - an absent optional tool is a warning, not a
package manager Distill runs on your behalf. `src/distill/capabilities.py` holds
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

Distill writes two different things, with two different lifetimes.

The **bundle** is derived state - generations, keyframes, manifests, locks -
and it is disposable: everything in it can be rebuilt from the source. By
default the bundle output root is `~/.cache/distill`, or `$XDG_CACHE_HOME/distill`
when that variable is set to an absolute path. Override it with `--output-dir`,
`DISTILL_OUTPUT_DIR`, or `output_dir` in a config file (see
[Configuration](#configuration)).

The **artifact** is the deliverable: one self-contained markdown file per
source, carrying the whole account and no machine-local path. It is *not* put in
the cache, because a deliverable a cache cleaner may delete is not a
deliverable. By default it lands in the project you are standing in -
`<git work tree root>/.distill/<entry>.md` - so a run inside a repository leaves
its output where the work is. Resolution, highest precedence first:
`--artifact-dir`, `DISTILL_ARTIFACT_DIR`, a `.distill` directory that already
exists at or above the working directory, `artifact_dir` in a config file, the
git work tree root, and finally `$XDG_DATA_HOME/distill/artifacts` outside any
repository. The run's JSON result names it as `artifact_path`.

Distill never edits your `.gitignore`. Whether `.distill/` is committed is
your decision.

A run that does the work publishes one **generation** into the
**bundle** that its **bundle key** names; a run that hits the cache publishes
nothing and serves the generation already there:

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
sensitive path such as `~/.ssh` or `~/.aws`. Configuration and output roots are
separate settings. Configuring one does not configure the other.

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

## Configuration

Configuration is read from one directory. An absolute `DISTILL_CONFIG_DIR` is
authoritative even when the directory does not exist; in that case there is no
config file and no lower directory is consulted. Without an absolute explicit
override, `$XDG_CONFIG_HOME/distill`, then `~/.config/distill`, then
`~/.distill` are considered. The first existing directory is read and the rest
are not. Relative `DISTILL_CONFIG_DIR` and `XDG_CONFIG_HOME` values are ignored
so configuration never depends on the process working directory.

`distill.json` in that directory configures a run. Recognized top-level keys
come from the same option table used by processing tools. `job_id` is excluded
because it identifies one invocation. `cache_mode` is the exception to the CLI
flag surface: it is configurable here and remains a tool argument accepted
through `call-tool --args`. The nested `local_vision` object, and
`distill.local-vision.json` beside it, configure the vision client:

```json
{
  "output_dir": "~/media/distill",
  "max_keyframes": 40,
  "local_vision": { "model": "mlx-community/Qwen3-VL-8B-Instruct-8bit" }
}
```

A `distill.json` that cannot be parsed, cannot be read, or does not hold a JSON
object ends the command with `E_BAD_OPTIONS` naming the file and what was wrong
with it. It is not treated as an empty file: a run that quietly fell back to the
defaults for every option you meant to set would produce a bundle you did not
ask for. Unknown top-level keys are refused for the same reason. A file that is
simply not there is the ordinary case and configures nothing.

General values use the same option table and numeric domains regardless of
source, but the outer refusal surface can differ. For example,
`--max-keyframes 2.5` is rejected by argparse with a usage message, while
`"max_keyframes": 2.5` in `distill.json` reaches the option table and is
refused as `E_BAD_OPTIONS` naming the file.

`distill.local-vision.json` and the nested `local_vision` object keep their own
forgiving reader. A malformed file or unusable timeout falls back to defaults,
which can cost captions without ending the run. Endpoint policy is different:
a forbidden scheme or host is a fatal `E_BAD_OPTIONS`, because silently
ignoring a configured endpoint would hide that the operator's configuration
was not used.

A value can come from four places, and the first of these that names it wins:
the command line, then the environment, then the config file, then Distill's
own defaults. `DISTILL_OUTPUT_DIR` is the only option the environment sets - it
answers where this machine keeps its bundles, which is a property of the machine
rather than of the processing being asked for. The other `DISTILL_*` variables
(`DISTILL_TRACEBACK`, `DISTILL_LOCAL_VISION_DEBUG`,
`DISTILL_EFFECTIVE_TIMEOUT_MS`, `DISTILL_ENABLE_LONG_TIMEOUT_PROBE`) steer
diagnosis and are not options: none of them changes what a run produces.

## Commands

`distill` exposes the following subcommands. All print JSON to stdout on
success; every fatal error prints a JSON error object to stderr - carrying its
`code`, `stage`, `message` and `details` - and exits with code 2, with nothing
on stdout. An argument the parser rejects is argparse's usage message, also exit
2. Set `DISTILL_TRACEBACK=1` to have an unexpected failure re-raised with its
Python traceback instead of being converted; it is a debugging aid, not a
contract.

Two endings are deliberately not that shape, because neither is a failure to
diagnose. `Ctrl-C` ends the command with Python's own `KeyboardInterrupt`
traceback and exit 130 - the operator stopped their own command. A write to a
stdout nobody is reading (`distill … | head`) ends it at exit 141, the
conventional 128 + `SIGPIPE`, and Distill writes no error record, because
nothing failed. A result small enough to reach the pipe before the reader leaves
is written and the command succeeds as usual: 141 is what a *write with no
reader* produces, not what `| head` always produces.

**stderr carries two kinds of record.** A processing run reports progress there
as it goes - one NDJSON record per event, each carrying `"type":
"distill.progress"`. The fatal error record has no `type` field and is the last
record Distill writes, so a failing run's stderr is read a line at a time and
the record without a `type` is the failure; parsing the whole stream as one
document fails at the first newline. Lines that are not JSON at all come from
libraries Distill imports rather than from Distill.

**"Nothing on stdout" means nothing was written there, not that a write can be
taken back.** A failing command fails before it prints a result, so stdout stays
empty. The exception is a caller who closes the pipe part-way through one:
those bytes have already left, Distill exits 141, and no guarantee about
stdout's contents survives a descriptor the caller broke.

| Command | Purpose |
| --- | --- |
| `process-local-video PATH` | Process one local video file into a bundle. |
| `process-youtube-video URL` | Download and process one personal-use YouTube video. |
| `process-video-directory PATH` | Process every video in a directory. Flags: `--recursive`, `--max-items`, `--continue-on-error/--no-continue-on-error`, plus the processing options. |
| `process-youtube-playlist URL` | Process videos from a YouTube playlist or channel URL. Flags: `--max-items`, `--continue-on-error/--no-continue-on-error`, plus the processing options. |
| `cleanup-cache` | Prune old cache bundles. Flags: `--output-dir`, `--max-age-days`, `--keep-generations`, `--dry-run/--no-dry-run`. |
| `cache-doctor` | Report what is under an output root - bundles, active generations, orphan generations, locks, a prune preview - and change nothing. Flags: `--output-dir`, `--max-age-days`, `--keep-generations`. |
| `get-job-status JOB_ID` | Read a Distill job status record. Flags: `--output-dir`. |
| `filtered-view BUNDLE_KEY` | Print the filtered view of an already-published bundle - the render with the frames the vision model judged redundant left out - as JSON with the document under `markdown`. Changes nothing under the output root. Flags: `--output-dir`. |
| `list-tools` | Print the registered tool names as JSON. |
| `timeout-diagnostics` | Show the configured vs. effective timeout (assumption A-004). |
| `timeout-probe PROBE_MS` | Sleep for a bounded timeout probe; long probes require `DISTILL_ENABLE_LONG_TIMEOUT_PROBE=1`. |
| `local-vision-diagnostics` | Probe the local Rapid-MLX vision server and print the resolved config. Flags: `--caption-frames/--no-caption-frames`, `--local-vision-backend`, `--local-vision-model`, `--local-vision-base-url`, `--local-vision-timeout-sec`, `--local-vision-allow-remote-endpoint/--no-local-vision-allow-remote-endpoint`. |
| `call-tool TOOL [--args JSON]` | Call any registered tool by its MCP-style name with a JSON arguments object. Prints the MCP envelope, so the tool's own JSON is the string at `.result.content[0].text` rather than the printed document itself. |

**`filtered-view` is a reading of a bundle, not a second render of it.** It
takes a bundle key - the directory name under the output root - and prints the
generation active there with the frames the vision model judged redundant
against the surrounding speech left out, under a banner saying exactly that. The
view is non-authoritative: the stored generation render still contains every
frame, this one is produced on demand, is never written back, and does not exist
under the bundle key. Nothing under the output root changes, so it is safe to
run against a cache another process is writing into, and the only copy of a view
is the one you keep:

```bash
distill filtered-view <bundle-key> | jq -r .markdown > view.md
```

**It redacts what the bundle it reads may not have.** The view runs secret
redaction over everything it reads back, whatever policy the run was published
under, and takes nothing from what the manifest claims about itself. For a
bundle published with `--redact-secrets` - the default - that changes nothing,
because the text was capped and redacted at write time. For one published with
`--no-redact-secrets` the view is redacted where the stored render is not, and
carries redaction warnings the generation never recorded: for those bundles it
is deliberately not a faithful projection of the render sitting beside it.

The processing commands (`process-local-video`, `process-youtube-video`,
`process-video-directory`, `process-youtube-playlist`) share these options: `--whisper-model`,
`--whisper-language`, `--ocr`/`--no-ocr`, `--ocr-language`, `--ocr-preprocess`,
`--redact-secrets`/`--no-redact-secrets`, `--artifact-dir`,
`--frame-salience`/`--no-frame-salience`
(whether keyframes are judged against the surrounding speech; on by default, and
part of bundle identity), `--max-keyframes`, `--min-interval-sec`,
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

A batch run (`process-video-directory`, `process-youtube-playlist`) that
continues past a failed item - the default, `--continue-on-error` - reports each
one as the whole error record: `code`, `stage`, `message` and `details`, plus
the item itself under `path` or `url` and the `batch_index` it failed at, so an
item that a re-run would pick up can be told from one that will fail the same
way forever. With `--no-continue-on-error` there is no such report: the first
item's error ends the batch and is raised as the run's own fatal error, so it
reaches you as the error object on stderr, carrying no `batch_index` and no item
key - whatever the failing stage happened to put in `details` is all that ties
it to an item.

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

A vision endpoint is handed images from your sources and the **extracted
text** around them - on-screen text read off each keyframe and, when frame
salience is on, the transcript window surrounding it - so where it points is a
decision rather than a default:

- **Loopback only, unless you say otherwise.** Every address the host resolves
  to must be loopback, checked per request rather than once at configuration
  time. Pass `--local-vision-allow-remote-endpoint` (or set
  `"allow_remote_endpoint": true` in the local-vision config) to reach a server
  elsewhere deliberately.
- **`http` or `https` only.** No other scheme is accepted, with or without the
  opt-out - and `http` only ever speaks to loopback. A non-loopback endpoint
  requires `https` even with the opt-out, so a keyframe or credential never
  crosses the network in cleartext.
- **Redirects are not followed**, and a response body beyond 32 MiB is refused.
  Both hold wherever the endpoint is: they are about what the client does with
  an answer, not about whose answer it is.
- **Credentials.** Set `"api_key_env"` (the name of an environment variable -
  preferred, and it wins over an inline value) or `"api_key"` in the
  local-vision config; the value is sent as `Authorization: Bearer` and never
  appears in logs, errors, renders, or bundles. A credential that is
  *configured but empty* on a non-loopback endpoint is a fatal config error
  (the typo guard); configuring none at all is intentional no-auth.
- **Remote runs are bounded.** With the opt-out set, the vision stage carries
  a run-wide budget (wall clock and received bytes) and 429/5xx answers are
  retried within `Retry-After` bounds, then the remainder of the run degrades
  to OCR-only. A run that may send keyframes off-machine also records a
  `non_local_only_processing` warning in its render, and that fact becomes
  part of bundle identity - the endpoint's address never does.

A *misconfigured* endpoint (bad scheme, credential in the URL, remote without
the opt-out, configured-but-empty credential) is a fatal error at startup: the
config was ignored otherwise. An endpoint that is merely *unavailable at run
time* degrades to OCR-only output with a warning instead.

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
source - transcript text, image text, vision interpretations, salience judgments, link labels - was
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
  pattern names survives it. `--no-redact-secrets` turns it off for *your own
  render and disk output only* - text entering a vision prompt is always
  redacted, wherever the endpoint lives, so the model reads `[REDACTED]` in
  place of a secret either way. It reduces what leaks into a bundle; it does
  not make a bundle safe to publish.

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
