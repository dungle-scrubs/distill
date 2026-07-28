# Changelog

All notable changes to Distill are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The remediation of the 0.1.0 audit. Pre-1.0 and with no dependents, so on-disk
breakage is accepted and there is no migration path: caches written by 0.1.0
stay readable where the format allows and are otherwise reprocessed.

### Breaking

- **Distill installs nothing.** A missing `tesseract` used to make Distill run
  `brew install tesseract` on the user's behalf. It now records a warning and
  the run continues without image-text extraction. Homebrew is a documented way
  to install the system dependencies; it is never something Distill invokes.
- **Every external tool is classified optional or required, and the two behave
  differently.** A missing optional capability (`tesseract`, the vision server)
  degrades the bundle with a warning and the run continues; a missing required
  capability (`ffmpeg`, `ffprobe`, `yt-dlp`) is a fatal error that stops the run
  naming the tool, instead of producing a bundle with nothing in it. The README
  and AGENTS.md state the class and the absence cost per tool.
- **Every failure has one shape.** A failing command prints the fatal error
  record as JSON on stderr - `code`, `stage`, `message`, `details` - and exits
  2 with nothing on stdout. Uncoded failures previously left a Python traceback
  and exit 1, and `call-tool` printed an error envelope to *stdout* and exited
  0, so a script driving Distill through it read success for every failure.
  `DISTILL_TRACEBACK=1` re-raises with the traceback for debugging.
- **Batch item errors carry the whole record.** `process-video-directory` and
  `process-youtube-playlist` reported a failed item as `{item_key, message}`.
  Each error now carries `code`, `stage`, `message`, `details` and its
  `batch_index`; the flattened `message`-only shape is gone, as is the
  `DistillSession` envelope that hid the record inside a message string.
- **Numeric options are validated rather than taken as written.** Each has a
  domain, and a value outside it is refused with `E_BAD_OPTIONS` naming the
  option. The domain is a finite quantity above zero, with three exceptions:
  `--min-interval-sec` also admits `0`, `--max-static-window-sec` has a floor of
  `0.001` (the quantum a keyframe timestamp is rounded to - a narrower window
  named a schedule that could not be expressed and never terminated), and
  `--local-vision-timeout-sec` has a ceiling of what a socket timeout can hold.
  `--max-keyframes` and `--max-items` are counts, so a negative one is refused
  rather than read as a slice from the end. A source duration that is not finite
  and positive is refused too.
- **The local vision endpoint is loopback-only by default.** Every address the
  configured host resolves to must be loopback, checked per request rather than
  once at configuration time, and the scheme must be `http` or `https`. Reaching
  a server elsewhere needs `--local-vision-allow-remote-endpoint` or
  `"allow_remote_endpoint": true` in the local-vision config. Redirects are no
  longer followed and a response body beyond 32 MiB is refused, with or without
  the opt-out. A refused endpoint is a warning and OCR-only output, not a failed
  run.
- **Every bundle is re-keyed.** `PIPELINE_VERSION` has risen repeatedly since
  0.1.0 shipped it at 19 - once per change to what the pipeline produces - and
  it participates in the options hash, so every bundle key changes and every
  source is processed again. Old bundles stay on disk under their old keys until
  pruned.
- **A directory is a bundle only if it carries a bundle marker.** Distill writes
  `_owner.json` when it claims a bundle directory and recognizes a directory as
  its own only from a marker recording the bundle key that names it - the
  current `bundle_key` field or the legacy `source_hash`. A directory of a
  user's own files that happens to contain `g1/video.md` is no longer a bundle,
  and no longer prunable.
- **`--output-dir` refuses roots it used to accept.** An output root must be a
  subdirectory of `$HOME` or of the system temp directory - neither directory
  itself - and must not sit inside a sensitive path such as `~/.ssh` or
  `~/.aws`.
- **`cleanup-cache` refuses a retention policy that keeps nothing.**
  `--keep-generations` below 1 and a `--max-age-days` that is not finite and
  positive are `E_BAD_OPTIONS`. `--keep-generations 0` previously proposed every
  generation of a bundle, the active one included.
- **`warning_count` counts events, not records.** Warnings sharing a stage and
  code are folded into one record carrying an `occurrences` count, so a manifest
  that used to list eighty identical timeouts now lists one record with
  `occurrences: 80` - while `warning_count` still reports 80, because a run that
  degraded eighty times must not read as a run that degraded once.
- **A YouTube watch URL carrying a `list` parameter acquires the one video the
  URL names.** yt-dlp reads such a URL as the playlist, so `watch?v=A&list=P`
  resolved and downloaded every entry of `P`.

### Added

- `cache-doctor`: a read-only report of what is under an output root - bundles,
  active generations, orphan generations, live and leftover locks, skipped
  directories with a reason, and a prune preview. It creates nothing, which is
  what makes it safe to run before `cleanup-cache`.
- `--local-vision-allow-remote-endpoint` on the processing commands and
  `local-vision-diagnostics`, with the matching `allow_remote_endpoint` config
  key.
- `DISTILL_TRACEBACK=1`, the opt-in escape hatch from the error boundary.
- A transport circuit breaker for the vision server: after three consecutive
  transport failures the run degrades to OCR-only for the remainder and records
  one warning with the failure count, instead of waiting out a timeout per
  remaining keyframe.
- A distinct grounding level for a single reader vouching for itself, separate
  from two readers independently agreeing, and shown as such in the render.

### Changed

- The render marks extracted text as data: a preamble telling a downstream
  reader that recovered text is not instructions, and a delimiter around every
  piece of it that the text cannot terminate. The preamble states plainly that
  this is a mitigation and not a guarantee.
- Redaction covers more secret formats - bearer tokens, bare `token:`/`apikey:`
  assignments, GitLab, npm and HuggingFace tokens, the remaining GitHub token
  prefixes, and private-key headers - and runs at every point where extracted
  text becomes durable rather than only on the way into the render. It remains
  pattern matching over recovered text: it reduces what leaks into a bundle and
  does not make one safe to publish.
- Concurrency between runs on one bundle key is now the kernel's, via a lock
  held for the run's duration. A second run waits and then fails `E_LOCKED` -
  5 seconds for a batch item, 300 seconds for a single-source run - rather than
  stealing a lock whose heartbeat looked stale.
- A run that fails records a terminal failure in its job record; a record left
  `running` is reported as abandoned rather than read as complete.
- Every external process runs through one path that reads both pipes and kills
  the whole process group at the deadline.

### Fixed

- Retention could delete the generation its own manifest named as active. It now
  never proposes the active generation, at any `--keep-generations`, and a
  manifest naming a generation that is not on disk is a cache miss rather than a
  hit. See the README's upgrade note before the first `cleanup-cache` of a cache
  written by 0.1.0.
- Resume scratch is stripped before a generation is published, so a stage result
  can no longer reach a reader as part of a bundle.
- A cache hit is served without invoking a tool that is only needed to *produce*
  a bundle: a cached YouTube bundle is servable with `yt-dlp` absent, and a
  cached local bundle with `ffprobe` absent.
- A large `stdout` no longer deadlocks a subprocess whose `stderr` is also being
  read, and a child that ignores `SIGTERM` leaves no grandchild behind.
- The documented default output directory was wrong: bundles have always landed
  under `~/.cache/distill`, not `~/.distill` (which is the config directory).

## [0.1.0] - 2026-06-28

### Added

- Video-to-bundle pipeline: transcript JSON, keyframes, OCR text, local vision
  captions, and markdown render for LLM consumption.
- Local vision backend: Rapid-MLX with `mlx-community/Qwen3-VL-8B-Instruct-8bit`
  (eval-chosen reader, 0.91 token recall).
- Three CLI tools: `process-local-video`, `process-youtube-video`,
  `process-video-directory`.
- Fingerprint and content cache modes with generation-based pruning.
- Structured progress reporting with weighted mechanism aggregation.
- Partial-run resume for interrupted jobs.
- Secret redaction in transcript and OCR text.
- Human-verified 16-frame text-recovery eval harness (`tests/evals/`).
