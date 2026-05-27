# Saccade

Saccade converts local videos and personal-use YouTube videos into LLM-readable bundles with transcript JSON, keyframes, OCR text, local vision captions, and markdown.

## Install

```bash
uv add git+<repo-url>
```

For local development:

```bash
uv sync
uv run saccade list-tools
```

Saccade expects system tools for full processing: `ffmpeg`, `ffprobe`, `tesseract`, and `yt-dlp`.

## Local vision backends

Frame interpretation runs against a local vision model. Two backends are supported:

- **Ollama** (default, cross-platform) — model `qwen3-vl:8b`. Requires Ollama running.
- **MLX** (Apple Silicon, optional) — install with `uv sync --extra mlx`. Recommended model:
  `mlx-community/Qwen3-VL-8B-Instruct-8bit`.

On the 16-frame eval (`tests/evals/`), the MLX 8-bit model was the strongest reader
(lowest word-error-rate, best grounding balance) and ran ~28% faster than Ollama on
Apple Silicon; the 4-bit MLX build is smaller/faster but quantization costs accuracy.
A larger model (`qwen3-vl:30b-a3b`) gave no quality gain, so bigger is not better here.

## CLI

```bash
saccade process-local-video ./demo.mp4
saccade process-youtube-video "https://youtu.be/..."
saccade process-video-directory ./recordings --recursive --max-items 10
saccade process-youtube-playlist "https://www.youtube.com/playlist?list=..."
saccade cleanup-cache --keep-generations 3 --dry-run
```
