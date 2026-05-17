"""Cache cleanup helpers for immutable media bundle generations."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any


def cleanup_media_cache(
    root: Path,
    *,
    max_age_days: float | None = None,
    keep_generations: int = 3,
    dry_run: bool = True,
    markdown_names: tuple[str, ...] = ("episode.md", "video.md"),
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - (max_age_days * 86400) if max_age_days is not None else None
    candidates: list[Path] = []

    for bundle in root.iterdir():
        if not bundle.is_dir() or bundle.name.startswith("_"):
            continue
        generations = sorted(
            [path for path in bundle.glob("g*") if path.is_dir() and path.name[1:].isdigit()],
            key=lambda path: int(path.name[1:]),
        )
        for old_generation in generations[: max(0, len(generations) - keep_generations)]:
            candidates.append(old_generation)
        if cutoff is not None:
            marker_mtime = bundle.stat().st_mtime
            manifest = bundle / "_manifest.json"
            if manifest.exists():
                marker_mtime = manifest.stat().st_mtime
            has_media_bundle = any(
                (gen / name).exists() for gen in generations for name in markdown_names
            )
            if has_media_bundle and marker_mtime < cutoff:
                candidates.append(bundle)

    unique_candidates = sorted(set(candidates), key=lambda path: str(path))
    deleted: list[str] = []
    for candidate in unique_candidates:
        if not dry_run and candidate.exists():
            shutil.rmtree(candidate)
            deleted.append(str(candidate))

    return {
        "root": str(root),
        "dry_run": dry_run,
        "candidate_count": len(unique_candidates),
        "deleted_count": len(deleted),
        "candidates": [str(path) for path in unique_candidates],
        "deleted": deleted,
    }
