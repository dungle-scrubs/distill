# Changelog

All notable changes to Distill are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-01

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
