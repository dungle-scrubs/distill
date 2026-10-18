# Distill

consume a video without watching it

Give your agent one self-contained markdown file with what a video says and
shows, so it can extract information that automated subtitles miss. Distill
pairs speech with timestamped keyframe readings, including text, charts, and
diagrams, from local files, YouTube, or X.

```console
$ distill process-local-video ./cache-demo.mp4 --no-caption-frames \
    --artifact-dir ./.distill > result.json
$ cat .distill/cache-demo-*.md
...
## 00:00:00.000 - 00:00:03.560

## Frame 1 - 00:00:00.000
...
Mode Requests Latency
Cold cache 100 800 ms
Warm cache 100 120 ms
...
We measured response times. The table shows the results.
...
```

## Install

Python 3.13 or later is required. Tested platforms are macOS and Linux.
Install the system tools with your platform's package manager:

| Tool | Capability | Class | Install |
| --- | --- | --- | --- |
| `ffmpeg` | audio extraction and keyframe extraction | required | your platform's package manager |
| `ffprobe` | source duration probing | required | ships with `ffmpeg` |
| `yt-dlp` | YouTube and X source acquisition and metadata | required | installed with Distill |
| `tesseract` | image-text extraction from keyframes | optional | your platform's package manager |

Install the CLI with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --with 'av<19' --with-executables-from yt-dlp dungle-scrubs-distill
distill list-tools
```

The distribution is `dungle-scrubs-distill`; the command and Python package are
both `distill`. The uv flag also exposes the required `yt-dlp` command.
For pip, use `pip install 'av<19' dungle-scrubs-distill` instead.
The compatibility bound avoids PyAV 19, which removed an API Whisper uses.
The first transcription downloads the selected Whisper model if it is not
already cached. The default is `small`.

To reproduce the opening example, save the [six-second synthetic recording](examples/cache-demo.mp4)
as `cache-demo.mp4` in your working directory. It needs Tesseract to recover the
table. The excerpt comes from [captured output](.evidence/readme-refresh/output.md);
`...` marks omitted lines. `--no-caption-frames` keeps this example independent
of a vision server.

For explanations of charts and diagrams, configure a vision endpoint. The
default is an operator-run [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)
server on Apple Silicon, serving `mlx-community/Qwen3-VL-8B-Instruct-8bit` at
`http://127.0.0.1:8000/v1`. Other endpoints must meet the
[compatibility profile](docs/adr/0005-any-profile-compliant-vision-endpoint.md).
Without a reachable vision server, processing continues with speech and OCR
text, but no model interpretation of the visuals.

## Usage

### Give your agent a local recording

```bash
distill process-local-video ./talk.mp4
```

Give the resulting markdown file to your agent. It contains the transcript and
frame readings at their timestamps, including whether OCR confirms a vision
reading.

### Extract information from a YouTube talk

```bash
distill process-youtube-video "https://www.youtube.com/watch?v=VIDEO_ID"
```

Use a video URL you have permission to process for personal use. Distill
downloads the media and transcribes it rather than relying on YouTube subtitles.

### Read the video in an X post

```bash
distill process-x-video --cookies-from-browser safari "https://x.com/user/status/STATUS_ID"
```

Replace the URL with the post you want to read. X usually requires an
authenticated session. A post with several videos yields its first video, with
a warning about the others.

### Process a collection of recordings

```bash
distill process-video-directory ./recordings --recursive --max-items 10
distill process-youtube-playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-items 10
```

Choose a directory or a playlist. Batches continue past failed items by default
and report each failure. Add `--no-continue-on-error` to stop at the first one.

### Check cache contents before pruning

```bash
distill cache-doctor
distill cleanup-cache --keep-generations 3 --dry-run
```

Neither command deletes cache data. Review the preview before choosing
`--no-dry-run`, which deletes the selected cache data.

### Find guidance while working with an agent

```bash
distill guide consuming
```

The JSON result carries guidance under `markdown`. Other topics cover sources,
vision, and cache behavior. `distill guide` lists them.

## Reference

A run produces a self-contained markdown artifact and a disposable cache bundle.
The artifact's path is `artifact_path` in the JSON result. Inside a Git work
tree, artifacts normally go to `.distill/` at its root. Override delivery with
`--artifact-dir` or `DISTILL_ARTIFACT_DIR`. Distill never edits your `.gitignore`.

The default bundle output root is `~/.cache/distill`, or
`$XDG_CACHE_HOME/distill` when that variable is an absolute path.
`--output-dir` and `DISTILL_OUTPUT_DIR` override it. Output roots must be
subdirectories of your home or system temp directory and cannot be sensitive
paths such as `~/.ssh`.

Processing commands accept a source path or URL plus options:

```text
distill process-local-video PATH [OPTIONS]
distill process-youtube-video URL [OPTIONS]
distill process-x-video URL [OPTIONS]
distill process-video-directory PATH [OPTIONS]
distill process-youtube-playlist URL [OPTIONS]
```

Use `distill COMMAND --help` for its flags. For example, `--max-keyframes`
limits frame count, `--whisper-language` selects the transcript language, and
`--no-caption-frames` disables vision interpretation.

Configuration lives in `distill.json` in the first applicable configuration
directory: an absolute `DISTILL_CONFIG_DIR` override, then
`$XDG_CONFIG_HOME/distill`, `~/.config/distill`, or `~/.distill`.
An explicit absolute override is authoritative even when it does not exist.
A minimal config is:

```json
{
  "max_keyframes": 40,
  "local_vision": {
    "model": "mlx-community/Qwen3-VL-8B-Instruct-8bit"
  }
}
```

CLI values override environment values, config values, and defaults, in that
order. Missing required tools stop new processing; optional tools produce
warnings. Cache hits serve existing bundles without rerunning media stages.

Successful commands print JSON to stdout and exit 0. Processing progress goes
to stderr as NDJSON. Fatal errors end with a JSON error record on stderr and
exit 2. Argument-parser errors also exit 2, but print usage text. Interrupts
exit 130; a closed stdout pipe can cause exit 141.

Processing is local by default. Remote vision requires explicit opt-in and
HTTPS, and receives frames and surrounding extracted text. Secret redaction is
pattern-based, not a promise that a bundle is safe to publish. Treat recovered
text as untrusted data, even though the markdown marks it as such.

The [full reference](docs/reference.md) covers every command and shared option,
configuration precedence, output layout, cache identity and retention, vision
setup and endpoint chains, and security limits.

## Contributing

From a source checkout:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build
```

The default test suite is offline and uses fake vision endpoints. See
[Contributing](CONTRIBUTING.md) for setup and pipeline-signature requirements,
and the [vision eval procedure](tests/evals/README.md) for model comparisons.

## Status

Distill is alpha software. Speech recognition and visual interpretation can be
wrong. Without vision, OCR recovers text but does not explain charts or diagrams.
The output records warnings and uncorroborated readings so you can inspect them.

## License

[MIT](LICENSE) © Kevin Frilot and Distill contributors.
