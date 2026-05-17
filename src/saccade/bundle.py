"""Cache layout, generation publishing, and response assembly for Saccade.

This module is the only final artifact writer. It owns manifest reads/writes,
generation directories, active-generation selection, and response payloads.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SaccadeError
from .media.bundle import ensure_safe_directory, next_generation
from .options import SaccadeOptions
from .source import SourceInfo
from .version import PIPELINE_VERSION, SACCADE_VERSION


@dataclass(frozen=True)
class BundlePaths:
    root: Path
    generation: Path
    frames: Path
    manifest: Path
    transcript: Path
    markdown: Path


def read_manifest(bundle_root: Path) -> dict[str, Any] | None:
    manifest = bundle_root / "_manifest.json"
    if not manifest.exists():
        return None
    with manifest.open() as handle:
        data = json.load(handle)
    validate_manifest_schema(data, require_active_generation=True)
    return data


def active_paths(bundle_root: Path) -> BundlePaths | None:
    manifest = read_manifest(bundle_root)
    if not manifest:
        return None
    generation = bundle_root / str(manifest["active_generation"])
    return BundlePaths(
        root=bundle_root,
        generation=generation,
        frames=generation / "frames",
        manifest=bundle_root / "_manifest.json",
        transcript=generation / "transcript.json",
        markdown=generation / "video.md",
    )


def validate_manifest_schema(
    manifest: dict[str, Any],
    *,
    require_active_generation: bool,
) -> None:
    required_types: dict[str, type | tuple[type, ...]] = {
        "pipeline_version": int,
        "saccade_version": str,
        "source_type": str,
        "source_hash": str,
        "source_resolved_path": str,
        "duration_sec": (int, float),
        "options": dict,
        "frame_count": int,
        "transcript_present": bool,
        "warning_count": int,
        "frames": list,
        "warnings": list,
    }
    if require_active_generation:
        required_types["active_generation"] = str
    for key, expected_type in required_types.items():
        value = manifest.get(key)
        if not isinstance(value, expected_type):
            expected_name = (
                " or ".join(item.__name__ for item in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise SaccadeError(
                "E_BAD_MANIFEST",
                "bundle",
                "cache manifest schema is invalid",
                {"field": key, "expected": expected_name},
            )


def stage_paths(bundle_root: Path, *, reset: bool = True) -> BundlePaths:
    generation_name = next_generation(bundle_root)
    tmp = bundle_root / f".tmp.{generation_name}"
    if tmp.is_symlink():
        raise SaccadeError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "output tree must not contain symlink components",
            {"path": str(tmp), "output_root": str(bundle_root.resolve())},
        )
    if tmp.exists() and reset:
        shutil.rmtree(tmp)
    frames = tmp / "frames"
    ensure_safe_directory(frames, bundle_root)
    return BundlePaths(
        root=bundle_root,
        generation=tmp,
        frames=frames,
        manifest=bundle_root / "_manifest.json",
        transcript=tmp / "transcript.json",
        markdown=tmp / "video.md",
    )


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
    tmp_manifest = paths.root / "_manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(paths.manifest)
    return final_paths


def write_bundle_files(
    paths: BundlePaths,
    source: SourceInfo,
    options: SaccadeOptions,
    transcript: dict[str, Any] | None,
    markdown: str,
    frames: list[dict],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    paths.markdown.write_text(markdown)
    if transcript is not None:
        paths.transcript.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "saccade_version": SACCADE_VERSION,
        "source_type": source.source_type,
        "source_hash": source.source_hash,
        "source_resolved_path": str(source.resolved_path),
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
    response = {
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
        "saccade_version": SACCADE_VERSION,
        "job_id": job_id,
        "summary": summary,
        "warnings": warnings,
    }
    if progress is not None:
        response["progress"] = progress
    return response
