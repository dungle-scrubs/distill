# README refresh verification

## What the page now claims

The opening benefit is: "Get a talk's speech and slide details without watching
it." The description explains that an agent receives one self-contained
markdown file combining speech and timestamped visual readings. It does not
promise error-free transcription or interpretation. The package description
matches the tagline exactly.

The opening example is captured output, not invented output. It uses the
[six-second synthetic recording](../../examples/cache-demo.mp4), whose narration
does not speak the table values. Tesseract recovers those values from the image.
Vision is disabled in this example. The [full artifact](output.md) retains the
untrusted-data boundaries omitted from the short excerpt. See
[provenance](provenance.md) for the source and capture conditions.

## What moved or was cut

The README is 212 lines, down from 561. Quickstart became six goal-oriented
Usage examples. Testing became Contributing, with lint, type, and build checks.

Output layout, detailed configuration, the complete command and flag lists,
cache identity and retention, vision setup and endpoint chains, capability
absence costs, and the full security discussion moved to
[the reference](../../docs/reference.md). The README retains a lookup summary,
installation requirements, and security limits. The centered logo and redundant
Git/project-install variants were cut. License credits remain.

Documentation tests follow the moved claims. Existing assertions remain, flag
checks cover both user documents, and dependency classifications are checked
in both copies of the table. A regression test prevents inline pipe characters
from being parsed as table rows.

## Verification results

- README audit: `pyramid: pass`, all 16 checks `ok`, no advisory exceptions.
- `uv run pytest`: 1,437 passed, 2 skipped. The gated network/live-server tests
  were not enabled.
- `uv run ruff check .`, `uv run ruff format --check .`, and `uv run ty check`:
  passed.
- `uv sync`, `uv lock`, and `uv build`: succeeded. The built wheel contains the
  current README, matching description, version, Python requirement, and PyAV
  compatibility bound.
- All 11 relative links and fragments in the README, reference, and provenance
  resolve. Every non-ellipsis output line in the opening example occurs in the
  captured artifact, in order.

### Commands run

The opening processing command and `cat .distill/cache-demo-*.md` ran with the
project CLI in an isolated working directory. The same media also ran through
a separately installed wheel and recovered both the narration and table text,
with no warnings.

These commands ran against synthetic data, isolated cache/configuration paths,
or their read-only guidance:

- `distill list-tools`
- `distill guide` and `distill guide consuming`
- `distill cache-doctor`
- `distill cleanup-cache --keep-generations 3 --dry-run`
- `distill filtered-view BUNDLE_KEY | jq -r .markdown`, with the actual sample key
- `distill local-vision-diagnostics`, which confirmed that the default server
  was unavailable. No server was started.

The documented uv install command ran against the published package with tool
and executable directories confined to the task folder:

```bash
uv tool install --with 'av<19' --with-executables-from yt-dlp dungle-scrubs-distill
```

Both installed executables were checked, and Whisper's audio decoder passed its
model-free compatibility test in that environment.

### Commands matched to definitions

All 15 subcommands ran with `--help`. The local-video, YouTube, X,
directory, and playlist Usage examples also parsed with the real CLI parser.
Placeholder source URLs, browser-cookie access, and batch processing were not
executed against real accounts or recordings. Their syntax and flags match the
CLI definitions. The pip alternative matches `pip install --help` and the
package metadata. The optional `rapid-mlx serve` command in the reference
matches the installed server binary's `serve --help`; the server was not run.

## Installation defects caught

A clean uv tool installation exposed only `distill`, not its `yt-dlp`
dependency's executable. With unrelated executables removed from `PATH`, a
YouTube invocation failed with `E_MISSING_TOOL` for `yt-dlp`. The documented
`--with-executables-from yt-dlp` flag exposes both commands, as specified by the
[official uv tool guide](https://docs.astral.sh/uv/guides/tools/#installing-tools).

An unconstrained clean installation selected PyAV 19 and produced a
`whisper_failed` warning because `av.open` no longer accepts `metadata_errors`.
The [PyAV 19 changelog](https://pyav.basswood.io/docs/stable/development/changelog.html#v19-0-0)
confirms that removal. The package now requires `av>=11,<19`. Install examples
also supply the bound so they work with the already-published 0.4.0 package.
A model-free WAV decode regression failed on PyAV 19 and passed on 18.1.0.
The installed-package media run then recovered speech and OCR with no warnings.
The lockfile still resolves its existing PyAV 17.0.1.

The reference also corrects the environment-variable account to include
`DISTILL_ARTIFACT_DIR`, and removes an unsupported assertion that independently
produced videos cannot share a sampled fingerprint.

## Confirmed brief

> - Problem (confirmed): Agents usually pull automated YouTube subtitles, which are often wrong and miss information in the visuals.
> - Occasion (confirmed): Dense explanatory videos, talks, tutorials, guides, and seminars, especially those with slides, charts, flowcharts, or tables.
> - Incumbent (confirmed): Relying on automated subtitles or watching the video yourself.
> - Value (confirmed): "Extract the information so I don't have to watch the video," with accurate, relevant information from speech and visuals.
> - Non-goals (confirmed): No strict genre restriction. The focus is recovering information and meaning, especially from text and diagrams in images.

The reusable brief is committed as [`.readme-brief.md`](../../.readme-brief.md).

## Delegation and limits

The source handoff used `scout-agent` on `glm-5.3@pi`, without a retained
session. Reference relocation used `minimax-m3@pi`, session
`01a14f68-2abf-73a7-9ed8-3dc143a22d32`; the lead checked its files and assertions.

An independent read-only review on `qwen3.8-27b@pi/lmstudio`, session
`01a14f6d-d941-7095-897d-2ce60d3338e5`, timed out without findings or a verdict.
It was not restarted. There is no completed independent-review approval.
The reported verification rests on the lead's prose/source review, actual
command captures, package smoke tests, and automated checks.

Task folder: `.scratch/readme-refresh/`. Lasting evidence is committed here;
temporary environments and logs are discarded after the local commit. No
server configuration, credentials, shared tracker, publication, or remote Git
state was changed.
