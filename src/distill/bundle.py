"""Generation publishing and response assembly for Distill.

This module writes a **generation**'s files and assembles the response payload a
caller reads. Bundle identity and layout - what a **bundle marker** is, where a
generation lives, and whether a path may be written to - belong to
`bundle_store`, which this module imports rather than restating. M3.3 moves
publishing itself there; M3.6 deletes what remains.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle_store import (
    BundlePaths,
    ensure_safe_directory,
    stage_paths,
    validate_manifest_schema,
)
from .options import DistillOptions
from .release import DISTILL_VERSION
from .source import SourceInfo
from .version import PIPELINE_VERSION


@dataclass
class BundleGeneration:
    paths: BundlePaths
    resume_partial: bool

    @classmethod
    def stage(cls, bundle_root: Path, *, reset: bool = True) -> BundleGeneration:
        return cls(stage_paths(bundle_root, reset=reset), resume_partial=not reset)

    def run_stage(
        self,
        name: str,
        producer: Callable[[], dict[str, Any]],
        *,
        on_resume: Callable[[], None] | None = None,
        after_produce: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        partial = self.read_partial(name)
        if partial is not None:
            if on_resume:
                on_resume()
            return partial

        payload = producer()
        if after_produce:
            after_produce(payload)
        self.write_partial(name, payload)
        return payload

    def read_partial(self, name: str) -> dict[str, Any] | None:
        if not self.resume_partial:
            return None
        partial = read_partial(self.paths, name)
        if isinstance(partial, dict):
            return partial
        if name == "frames" and isinstance(partial, list):
            return {"frames": partial, "warnings": []}
        return None

    def write_partial(self, name: str, payload: Any) -> None:
        write_partial(self.paths, name, payload)

    def publish(self, manifest: dict[str, Any], progress: dict[str, Any]) -> BundlePaths:
        manifest = dict(manifest)
        manifest["progress"] = progress
        return publish_generation(self.paths, manifest)


def publish_generation(paths: BundlePaths, manifest: dict[str, Any]) -> BundlePaths:
    generation_name = paths.generation.name.removeprefix(".tmp.")
    final_generation = paths.root / generation_name
    paths.generation.rename(final_generation)
    final_paths = BundlePaths(
        root=paths.root,
        generation=final_generation,
        frames=final_generation / "frames",
        manifest=paths.manifest,
        transcript=final_generation / "transcript.json",
        markdown=final_generation / "video.md",
    )
    manifest = dict(manifest)
    manifest["active_generation"] = generation_name
    validate_manifest_schema(manifest, require_active_generation=True)
    for frame in manifest.get("frames", []):
        if isinstance(frame, dict) and "path" in frame:
            frame["path"] = str(final_paths.frames / Path(str(frame["path"])).name)
    rewrite_published_partial_paths(paths.generation, final_generation)
    tmp_manifest = paths.root / "_manifest.json.tmp"
    ensure_safe_directory(tmp_manifest, paths.root, create_leaf=False)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(paths.manifest)
    return final_paths


def rewrite_published_partial_paths(staged_generation: Path, final_generation: Path) -> None:
    staged_prefix = str(staged_generation)
    final_prefix = str(final_generation)
    for path in final_generation.glob("_*.json"):
        payload = json.loads(path.read_text())
        rewritten = rewrite_path_prefix(payload, staged_prefix, final_prefix)
        path.write_text(json.dumps(rewritten, indent=2, sort_keys=True, default=str) + "\n")


def rewrite_path_prefix(value: Any, staged_prefix: str, final_prefix: str) -> Any:
    if isinstance(value, str):
        if value == staged_prefix:
            return final_prefix
        if value.startswith(f"{staged_prefix}/"):
            return f"{final_prefix}{value[len(staged_prefix):]}"
        return value
    if isinstance(value, list):
        return [rewrite_path_prefix(item, staged_prefix, final_prefix) for item in value]
    if isinstance(value, dict):
        return {
            key: rewrite_path_prefix(item, staged_prefix, final_prefix)
            for key, item in value.items()
        }
    return value


def patch_manifest_progress(manifest_path: Path, progress: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest["progress"] = progress
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(manifest_path)


def write_bundle_files(
    paths: BundlePaths,
    source: SourceInfo,
    options: DistillOptions,
    transcript: dict[str, Any] | None,
    markdown: str,
    frames: list[dict],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    # R-16: a write refuses to follow a symlink at its target, so a symlink
    # pre-created at video.md or transcript.json cannot redirect it.
    ensure_safe_directory(paths.markdown, paths.root, create_leaf=False)
    paths.markdown.write_text(markdown)
    if transcript is not None:
        ensure_safe_directory(paths.transcript, paths.root, create_leaf=False)
        paths.transcript.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "distill_version": DISTILL_VERSION,
        "source_type": source.source_type,
        "source_hash": source.source_hash,
        "source_resolved_path": str(source.resolved_path),
        "related_links": list(source.related_links or []),
        "duration_sec": source.duration_sec,
        "options": options.public_dict(source.source_type),
        "frame_count": len(frames),
        "transcript_present": transcript is not None,
        "warning_count": len(warnings),
        "frames": response_frames(frames),
        "warnings": warnings,
    }
    return manifest


def partial_path(paths: BundlePaths, name: str) -> Path:
    return paths.generation / f"_{name}.json"


def read_partial(paths: BundlePaths, name: str) -> Any | None:
    path = partial_path(paths, name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_partial(paths: BundlePaths, name: str, payload: Any) -> None:
    path = partial_path(paths, name)
    ensure_safe_directory(path, paths.root, create_leaf=False)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def response_frames(frames: list[dict]) -> list[dict[str, Any]]:
    response = []
    for frame in frames:
        item = {
            "index": int(frame["index"]),
            "timestamp_sec": float(frame["timestamp_sec"]),
            "path": str(frame["path"]),
            "relative_path": str(frame["relative_path"]),
            "ocr_text": str(frame.get("ocr_text", "")),
        }
        visual_interpretation = frame.get("visual_interpretation")
        if isinstance(visual_interpretation, dict):
            item["visual_interpretation"] = visual_interpretation
        response.append(item)
    return response


def response_from_paths(
    paths: BundlePaths,
    source: SourceInfo,
    frames: list[dict],
    transcript_present: bool,
    warnings: list[dict[str, str]],
    cached: bool,
    progress: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    visual_count = sum(
        1 for frame in frames if isinstance(frame.get("visual_interpretation"), dict)
    )
    summary = f"Processed {source.duration_sec:.1f}s video with {len(frames)} keyframes"
    if visual_count:
        summary += f" and visual interpretation for {visual_count} frames"
    response: dict[str, Any] = {
        "markdown_path": str(paths.markdown),
        "transcript_path": str(paths.transcript)
        if transcript_present and paths.transcript.exists()
        else None,
        "manifest_path": str(paths.manifest),
        "frames": response_frames(frames),
        "duration_sec": source.duration_sec,
        "frame_count": len(frames),
        "source_hash": source.source_hash,
        "source_resolved_path": str(source.resolved_path),
        "cached": cached,
        "pipeline_version": PIPELINE_VERSION,
        "distill_version": DISTILL_VERSION,
        "job_id": job_id,
        "summary": summary,
        "warnings": warnings,
    }
    related_links = list(source.related_links or [])
    if related_links:
        response["related_links"] = related_links
    if progress is not None:
        response["progress"] = progress
    return response
