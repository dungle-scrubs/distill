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

## CLI

```bash
saccade process-local-video ./demo.mp4
saccade process-youtube-video "https://youtu.be/..."
saccade process-video-directory ./recordings --recursive --max-items 10
saccade process-youtube-playlist "https://www.youtube.com/playlist?list=..."
saccade cleanup-cache --keep-generations 3 --dry-run
```
