"""What one run records and what it hands back.

This module owns two documents: the **manifest** content a run writes for a
**bundle**, and the response payload a caller receives. They are stated together
because they are one description of a run seen from two sides - the manifest is
what a later run reads back as a cache hit, and the response is what this run's
caller reads now. Splitting them is how a cache hit came to report a different
frame shape from the run that produced it.

It owns no I/O and no layout. Where the manifest is written, when it becomes the
**bundle marker**, and what makes a **generation** active are `bundle_store`'s;
this module produces the document and never a path to put it at. It owns no
policy either: which frames exist, whether the run was a cache hit, and what
warnings were raised are all decided by the caller and recorded here as given.

It is a signed module (ADR-0003): every field below is written into a manifest,
so editing it changes bundle content at an unchanged **bundle key**.
"""

from __future__ import annotations

from typing import Any

from .bundle_store import BundleSnapshot
from .options import DistillOptions
from .release import DISTILL_VERSION
from .source import SourceInfo
from .version import PIPELINE_VERSION


def manifest_document(
    source: SourceInfo,
    options: DistillOptions,
    *,
    transcript_present: bool,
    frames: list[dict],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    """The **manifest** content for one published **generation**.

    Everything a later reader needs to decide whether this bundle answers its
    question, and nothing about where the bundle lives: `active_generation` is
    added by the publish, because only the publish knows which generation the
    staging directory became.
    """
    return {
        "pipeline_version": PIPELINE_VERSION,
        "distill_version": DISTILL_VERSION,
        "source_type": source.source_type,
        "source_hash": source.source_hash,
        "source_resolved_path": str(source.resolved_path),
        "related_links": list(source.related_links or []),
        "duration_sec": source.duration_sec,
        "options": options.public_dict(source.source_type),
        "frame_count": len(frames),
        "transcript_present": transcript_present,
        "warning_count": len(warnings),
        "frames": response_frames(frames),
        "warnings": warnings,
    }


def response_frames(frames: list[dict]) -> list[dict[str, Any]]:
    """The frame shape both documents carry, produced once so they cannot diverge.

    Image text and the vision model's reading of the same frame stay separate
    fields: they are different claims about the frame, and merging them loses
    which one a reader is looking at.
    """
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


def run_response(
    snapshot: BundleSnapshot,
    source: SourceInfo,
    frames: list[dict],
    transcript_present: bool,
    warnings: list[dict[str, str]],
    cached: bool,
    progress: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """What a caller receives, whether the run produced the bundle or read it.

    Stated against a `BundleSnapshot` rather than a set of paths, so a cache hit
    and a fresh publish are described by the same code from the same evidence -
    a snapshot exists only where the **active generation** was proven to be on
    disk (R-04).
    """
    visual_count = sum(
        1 for frame in frames if isinstance(frame.get("visual_interpretation"), dict)
    )
    summary = f"Processed {source.duration_sec:.1f}s video with {len(frames)} keyframes"
    if visual_count:
        summary += f" and visual interpretation for {visual_count} frames"
    response: dict[str, Any] = {
        "markdown_path": str(snapshot.markdown),
        "transcript_path": str(snapshot.transcript)
        if transcript_present and snapshot.transcript.exists()
        else None,
        "manifest_path": str(snapshot.manifest_path),
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
