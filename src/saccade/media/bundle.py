"""Shared immutable generation bundle helpers for media-ingest apps."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import MediaIngestError


@dataclass(frozen=True)
class MediaBundlePaths:
    root: Path
    generation: Path
    manifest: Path
    markdown: Path
    transcript: Path
    assets: Path | None = None


def read_json_manifest(bundle_root: Path) -> dict[str, Any] | None:
    manifest = bundle_root / "_manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text())


def next_generation(bundle_root: Path) -> str:
    existing = [
        int(path.name[1:])
        for path in bundle_root.glob("g*")
        if path.is_dir() and path.name[1:].isdigit()
    ]
    return f"g{max(existing, default=0) + 1}"


def ensure_safe_directory(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    target = path if path.is_absolute() else root / path
    try:
        target.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise MediaIngestError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "bundle path must stay under output_dir",
            {"path": str(target), "output_root": str(resolved_root)},
        ) from exc

    current = resolved_root
    relative_parts = target.resolve(strict=False).relative_to(resolved_root).parts
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise MediaIngestError(
                "E_BAD_OUTPUT_DIR",
                "bundle",
                "output tree must not contain symlink components",
                {"path": str(current), "output_root": str(resolved_root)},
            )
        current.mkdir(exist_ok=True)
    return target


def stage_paths(
    bundle_root: Path,
    *,
    markdown_name: str,
    transcript_name: str = "transcript.json",
    assets_name: str | None = None,
) -> MediaBundlePaths:
    generation_name = next_generation(bundle_root)
    tmp = bundle_root / f".tmp.{generation_name}"
    if tmp.is_symlink():
        raise MediaIngestError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "output tree must not contain symlink components",
            {"path": str(tmp), "output_root": str(bundle_root.resolve())},
        )
    if tmp.exists():
        shutil.rmtree(tmp)
    ensure_safe_directory(tmp, bundle_root)
    assets = None
    if assets_name:
        assets = tmp / assets_name
        ensure_safe_directory(assets, bundle_root)
    return MediaBundlePaths(
        root=bundle_root,
        generation=tmp,
        manifest=bundle_root / "_manifest.json",
        markdown=tmp / markdown_name,
        transcript=tmp / transcript_name,
        assets=assets,
    )


def publish_generation(paths: MediaBundlePaths, manifest: dict[str, Any]) -> MediaBundlePaths:
    generation_name = paths.generation.name.removeprefix(".tmp.")
    final_generation = paths.root / generation_name
    paths.generation.rename(final_generation)
    final_paths = MediaBundlePaths(
        root=paths.root,
        generation=final_generation,
        manifest=paths.manifest,
        markdown=final_generation / paths.markdown.name,
        transcript=final_generation / paths.transcript.name,
        assets=final_generation / paths.assets.name if paths.assets else None,
    )
    manifest = dict(manifest)
    manifest["active_generation"] = generation_name
    tmp_manifest = paths.root / "_manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(paths.manifest)
    return final_paths
