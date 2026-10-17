#!/bin/bash
# macOS proof leg, run locally on mini: the suite must pass with the optional
# capabilities (ffmpeg, tesseract) absent - they degrade to skips, never
# failures (R-52, ADR-0002). Replaces the macOS leg of the CI matrix
# (GitHub-hosted macOS runners bill at 10x the Linux rate).
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
echo "local macos test OK"
