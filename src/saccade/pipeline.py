"""Saccade pipeline orchestration.

This module owns the package-facing processing functions used by the CLI and by
the tool-proxy adapter. It deliberately has no dependency on tool-proxy protocol
helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bundle import (
    active_paths,
    ensure_safe_directory,
    publish_generation,
    read_partial,
    response_from_paths,
    stage_paths,
    write_bundle_files,
    write_partial,
)
from .errors import SaccadeError
from .frame_selection import select_keyframes
from .local_vision import (
    local_vision_config_from_args,
    probe_local_vision,
    try_interpret_image,
)
from .media.cache import cleanup_media_cache
from .media.jobs import read_job_status, write_job_status
from .ocr import ocr_frames
from .options import SaccadeOptions
from .progress import (
    TERMINAL_PROGRESS_STATUSES,
    OverallProgressAggregator,
    ProgressCounter,
    ProgressHeartbeat,
    ProgressReporter,
)
from .redact_secrets import redact_text
from .render import render_markdown
from .source import (
    cached_youtube_source,
    resolve_local_source,
    validate_output_root,
    youtube_source_info,
)
from .vision_prompts import build_technical_frame_prompt

TOOLS = {
    "process_local_video": "Process a local video into a transcript/keyframe markdown bundle",
    "process_youtube_video": "Download and process one YouTube video into a transcript/keyframe markdown bundle",
    "process_video_directory": "Process video files in a local directory into Saccade bundles",
    "process_youtube_playlist": "Process videos from a YouTube playlist or channel URL",
    "cleanup_cache": "Prune old Saccade cache bundles and generations",
    "get_job_status": "Read a Saccade job status record by job id",
}
DEFAULT_CONFIGURED_TIMEOUT_MS = 5_400_000
TIMEOUT_ENV = "TOOL_PROXY_EFFECTIVE_TIMEOUT_MS"
LONG_TIMEOUT_PROBE_ENV = "SACCADE_ENABLE_LONG_TIMEOUT_PROBE"
TIMEOUT_PROBE_LIMIT_MS = 1_000


class SaccadeSession:
    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "process_local_video":
                result = process_local_video(args)
            elif name == "process_youtube_video":
                result = process_youtube_video(args)
            elif name == "process_video_directory":
                result = process_video_directory(args)
            elif name == "process_youtube_playlist":
                result = process_youtube_playlist(args)
            elif name == "cleanup_cache":
                result = cleanup_saccade_cache(args)
            elif name == "get_job_status":
                result = get_job_status(args)
            else:
                raise SaccadeError("E_UNKNOWN_TOOL", "protocol", f"Unknown tool: {name}")
            return {"result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
        except SaccadeError as exc:
            return {"error": {"message": exc.to_json_text()}}
        except Exception as exc:
            error = SaccadeError("E_INTERNAL", "internal", str(exc))
            return {"error": {"message": error.to_json_text()}}


def process_local_video(args: dict[str, Any]) -> dict[str, Any]:
    options = SaccadeOptions.from_args(args)
    source = resolve_local_source(str(args.get("path", "")), options)
    return process_resolved_source(source, options, tool="process_local_video")


def process_youtube_video(args: dict[str, Any]) -> dict[str, Any]:
    options = SaccadeOptions.from_args({**args, "cache_mode": "fingerprint"})
    root = validate_output_root(options.output_dir)
    progress = ProgressReporter(emitter=progress_emitter(options.job_id))
    url = str(args.get("url", ""))
    if not options.force_reprocess:
        cached = cached_youtube_source(url, options, root)
        if cached is not None:
            progress.skip_cached(
                "youtube_download",
                detail={
                    "source": "cached_manifest",
                    "video_id": cached.youtube_video_id,
                },
            )
            return process_resolved_source(
                cached,
                options,
                root,
                progress=progress,
                tool="process_youtube_video",
            )
    source = youtube_source_info(url, options, root, progress=progress)
    return process_resolved_source(
        source,
        options,
        root,
        progress=progress,
        tool="process_youtube_video",
    )


def progress_emitter(job_id: str) -> Any:
    def emit(event: Any) -> None:
        payload = event.to_dict()
        payload["job_id"] = job_id
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)

    return emit


def cache_hit_progress_summary(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    progress_summary = manifest.get("progress")
    if isinstance(progress_summary, dict) and progress_summary_is_terminal(progress_summary):
        return progress_summary

    progress_summary = OverallProgressAggregator().cached_summary({"source": "cache"})
    manifest = dict(manifest)
    manifest["progress"] = progress_summary
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(manifest_path)
    return progress_summary


def progress_summary_is_terminal(progress_summary: dict[str, Any]) -> bool:
    if progress_summary.get("overall_percent") != 100.0:
        return False
    mechanisms = progress_summary.get("mechanisms")
    if not isinstance(mechanisms, dict) or not mechanisms:
        return False
    return all(
        isinstance(state, dict) and state.get("status") in TERMINAL_PROGRESS_STATUSES
        for state in mechanisms.values()
    )


def process_resolved_source(
    source: Any,
    options: SaccadeOptions,
    output_root: Path | None = None,
    progress: ProgressReporter | None = None,
    *,
    tool: str = "process_local_video",
) -> dict[str, Any]:
    output_root = output_root or validate_output_root(options.output_dir)
    bundle_root = output_root / source.source_hash
    ensure_safe_directory(bundle_root, output_root)
    cached_paths = active_paths(bundle_root)
    if cached_paths and not options.force_reprocess:
        manifest = json.loads(cached_paths.manifest.read_text())
        progress_summary = cache_hit_progress_summary(manifest, cached_paths.manifest)
        response = response_from_paths(
            cached_paths,
            source,
            manifest.get("frames", []),
            bool(manifest.get("transcript_present")),
            list(manifest.get("warnings", [])),
            cached=True,
            progress=progress_summary,
            job_id=options.job_id,
        )
        write_job_status(
            output_root, options.job_id, status="completed", tool=tool, result=response
        )
        return response

    progress = progress or ProgressReporter(emitter=progress_emitter(options.job_id))
    heartbeat = ProgressHeartbeat(progress.counter).start()
    staged = stage_paths(bundle_root, reset=not options.resume_partial)
    try:
        warnings = list(source.warnings)
        transcript_partial = read_partial(staged, "transcript") if options.resume_partial else None
        if isinstance(transcript_partial, dict):
            transcript = transcript_partial.get("transcript")
            warnings.extend(transcript_partial.get("warnings", []))
            progress.skip_cached("transcription", detail={"source": "partial_resume"})
            progress.skip_cached("audio_extraction", detail={"source": "partial_resume"})
        else:
            transcript, transcript_warnings = transcribe_with_imports(
                source.resolved_path,
                staged.generation,
                options,
                progress,
            )
            heartbeat.check()
            warnings.extend(transcript_warnings)
            write_partial(
                staged,
                "transcript",
                {"transcript": transcript, "warnings": transcript_warnings},
            )

        frames_partial = read_partial(staged, "frames") if options.resume_partial else None
        if isinstance(frames_partial, list):
            frames = frames_partial
            progress.skip_cached("frame_selection", detail={"source": "partial_resume"})
        else:
            frames, frame_warnings = select_keyframes(
                source.resolved_path,
                staged.frames,
                source.duration_sec,
                options.max_keyframes,
                options.min_interval_sec,
                options.max_static_window_sec,
                progress,
            )
            heartbeat.check()
            warnings.extend(frame_warnings)
            write_partial(staged, "frames", frames)

        ocr_partial = read_partial(staged, "ocr") if options.resume_partial else None
        if isinstance(ocr_partial, dict):
            frames = ocr_partial.get("frames", frames)
            warnings.extend(ocr_partial.get("warnings", []))
            progress.skip_cached("ocr", detail={"source": "partial_resume"})
        else:
            frames, ocr_warnings = ocr_frames(frames, options.ocr_language, options.ocr, progress)
            heartbeat.check()
            warnings.extend(ocr_warnings)
            write_partial(staged, "ocr", {"frames": frames, "warnings": ocr_warnings})
        if options.redact_secrets:
            redaction_partial = (
                read_partial(staged, "redaction") if options.resume_partial else None
            )
            if isinstance(redaction_partial, dict):
                frames = redaction_partial.get("frames", frames)
                warnings.extend(redaction_partial.get("warnings", []))
                progress.skip_cached("redaction", detail={"source": "partial_resume"})
            else:
                redaction_warnings: list[dict[str, str]] = []
                redacted_frames = []
                for index, frame in enumerate(frames):
                    copied = dict(frame)
                    result = redact_text(str(copied.get("ocr_text", "")))
                    copied["ocr_text"] = result.text
                    redaction_warnings.extend(result.warnings)
                    redacted_frames.append(copied)
                    progress.update(
                        "redaction",
                        percent=((index + 1) / max(1, len(frames))) * 100,
                        detail={"frame": index + 1, "frames": len(frames)},
                    )
                frames = redacted_frames
                warnings.extend(redaction_warnings)
                progress.complete("redaction", detail={"frames": len(frames)})
                write_partial(
                    staged,
                    "redaction",
                    {"frames": frames, "warnings": redaction_warnings},
                )
        else:
            progress.skip_cached("redaction", detail={"reason": "disabled"})
        if options.caption_frames:
            vision_partial = (
                read_partial(staged, "local_vision") if options.resume_partial else None
            )
            if isinstance(vision_partial, dict):
                frames = vision_partial.get("frames", frames)
                warnings.extend(vision_partial.get("warnings", []))
                progress.skip_cached("local_vision", detail={"source": "partial_resume"})
            else:
                frames, vision_warnings = interpret_frames_with_local_vision(
                    frames,
                    options,
                    progress,
                )
                warnings.extend(vision_warnings)
                write_partial(
                    staged,
                    "local_vision",
                    {"frames": frames, "warnings": vision_warnings},
                )
        else:
            progress.skip_cached("local_vision", detail={"reason": "disabled"})
        progress.update("rendering", status="running")
        markdown = render_markdown(
            str(source.resolved_path), source.duration_sec, transcript, frames, warnings
        )
        progress.complete("rendering")
        progress.update("bundle_publish", status="running")
        manifest = write_bundle_files(
            staged, source, options, transcript, markdown, frames, warnings
        )
        final_paths = publish_generation(staged, manifest)
        progress.complete("bundle_publish", detail={"generation": final_paths.generation.name})
        progress_summary = progress.aggregator.terminal_summary(progress.states)
        manifest = json.loads(final_paths.manifest.read_text())
        manifest["progress"] = progress_summary
        tmp_manifest = final_paths.manifest.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        tmp_manifest.replace(final_paths.manifest)
    finally:
        heartbeat.stop()
    final_frames = []
    for frame in frames:
        copied = dict(frame)
        copied["path"] = str(final_paths.frames / Path(str(frame["path"])).name)
        final_frames.append(copied)
    response = response_from_paths(
        final_paths,
        source,
        final_frames,
        transcript is not None,
        warnings,
        cached=False,
        progress=progress_summary,
        job_id=options.job_id,
    )
    write_job_status(output_root, options.job_id, status="completed", tool=tool, result=response)
    return response


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def bool_arg(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def process_video_directory(args: dict[str, Any]) -> dict[str, Any]:
    directory = Path(str(args.get("path", ""))).expanduser()
    if not directory.exists() or not directory.is_dir():
        raise SaccadeError(
            "E_BAD_SOURCE", "source", "directory does not exist", {"path": str(directory)}
        )
    options = SaccadeOptions.from_args(args)
    max_items = int(args.get("max_items", 50))
    recursive = bool(args.get("recursive", False))
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in sorted(directory.glob(pattern))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ][:max_items]
    root = validate_output_root(options.output_dir)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        child_args = {**args, "path": str(path), "job_id": f"{options.job_id}-{index}"}
        try:
            result = process_local_video(child_args)
            result["batch_index"] = index
            results.append(result)
        except Exception as exc:
            errors.append({"path": str(path), "message": str(exc)})
            if not bool_arg(args.get("continue_on_error"), True):
                raise
    summary = {
        "job_id": options.job_id,
        "directory": str(directory.resolve()),
        "video_count": len(files),
        "processed_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors,
    }
    write_job_status(
        root, options.job_id, status="completed", tool="process_video_directory", result=summary
    )
    return summary


def youtube_playlist_urls(url: str, max_items: int) -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "webpage_url", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SaccadeError(
            "E_YTDLP",
            "youtube",
            "yt-dlp could not list playlist videos",
            {"stderr": proc.stderr.strip()},
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()][:max_items]


def playlist_folder_name(url: str) -> str:
    parsed = urlparse(url)
    playlist_id = parse_qs(parsed.query).get("list", [""])[0]
    raw_name = playlist_id or f"url-{hashlib.sha256(url.encode()).hexdigest()[:16]}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip(".-")
    return safe_name[:120] or f"url-{hashlib.sha256(url.encode()).hexdigest()[:16]}"


def process_youtube_playlist(args: dict[str, Any]) -> dict[str, Any]:
    url = str(args.get("url", ""))
    if not url:
        raise SaccadeError("E_BAD_URL", "youtube", "url is required")
    options = SaccadeOptions.from_args({**args, "cache_mode": "fingerprint"})
    root = validate_output_root(options.output_dir)
    playlist_root = root / "playlists" / playlist_folder_name(url)
    ensure_safe_directory(playlist_root, root)
    max_items = int(args.get("max_items", 25))
    urls = youtube_playlist_urls(url, max_items)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, video_url in enumerate(urls, start=1):
        child_args = {
            **args,
            "url": video_url,
            "output_dir": str(playlist_root),
            "job_id": f"{options.job_id}-{index}",
        }
        try:
            result = process_youtube_video(child_args)
            result["batch_index"] = index
            results.append(result)
        except Exception as exc:
            errors.append({"url": video_url, "message": str(exc)})
            if not bool_arg(args.get("continue_on_error"), True):
                raise
    summary = {
        "job_id": options.job_id,
        "playlist_url": url,
        "playlist_output_dir": str(playlist_root),
        "video_count": len(urls),
        "processed_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors,
    }
    playlist_summary_path = playlist_root / "playlist.json"
    summary["playlist_summary_path"] = str(playlist_summary_path)
    playlist_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_job_status(
        root, options.job_id, status="completed", tool="process_youtube_playlist", result=summary
    )
    return summary


def cleanup_saccade_cache(args: dict[str, Any]) -> dict[str, Any]:
    root = validate_output_root(args.get("output_dir"))
    return cleanup_media_cache(
        root,
        max_age_days=float(args["max_age_days"]) if args.get("max_age_days") is not None else None,
        keep_generations=int(args.get("keep_generations", 3)),
        dry_run=bool_arg(args.get("dry_run"), True),
        markdown_names=("video.md",),
    )


def get_job_status(args: dict[str, Any]) -> dict[str, Any]:
    root = validate_output_root(args.get("output_dir"))
    job_id = str(args.get("job_id", ""))
    if not job_id:
        raise SaccadeError("E_BAD_JOB", "job", "job_id is required")
    status = read_job_status(root, job_id)
    if status is None:
        raise SaccadeError("E_JOB_NOT_FOUND", "job", "job status not found", {"job_id": job_id})
    return status


def transcribe_with_imports(
    video_path: Path,
    work_dir: Path,
    options: SaccadeOptions,
    progress: ProgressCounter | ProgressReporter,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    from .transcript import transcribe_video

    return transcribe_video(
        video_path,
        work_dir,
        options.whisper_model,
        options.whisper_language,
        options.vad_filter,
        progress,
    )


def interpret_frames_with_local_vision(
    frames: list[dict],
    options: SaccadeOptions,
    progress: ProgressReporter | None = None,
) -> tuple[list[dict], list[dict[str, str]]]:
    config = options.local_vision_config()
    probe = probe_local_vision(config)
    if not probe.available:
        if progress:
            progress.skip_cached("local_vision", detail={"reason": probe.code})
        return frames, [probe.warning()]
    interpreted_frames = []
    warnings: list[dict[str, str]] = []
    if not frames:
        if progress:
            progress.complete("local_vision", detail={"frames": 0})
        return frames, warnings
    for index, frame in enumerate(frames):
        copied = dict(frame)
        prompt = build_technical_frame_prompt(
            "ui_interface",
            ocr_text=str(copied.get("ocr_text", "")) or None,
        )
        result, warning = try_interpret_image(
            config,
            Path(str(copied["path"])),
            prompt.prompt,
            prompt_profile=prompt.profile,
        )
        if warning:
            warnings.append(warning)
        if result:
            copied["visual_interpretation"] = result.public_dict()
        interpreted_frames.append(copied)
        if progress:
            progress.update(
                "local_vision",
                percent=((index + 1) / len(frames)) * 100,
                detail={"frame": index + 1, "frames": len(frames)},
            )
    if progress:
        progress.complete("local_vision", detail={"frames": len(frames)})
    return interpreted_frames, warnings


def list_tools() -> None:
    print(json.dumps({"tools": sorted(TOOLS)}, indent=2))


def configured_timeout_ms() -> int:
    return DEFAULT_CONFIGURED_TIMEOUT_MS


def timeout_diagnostics(effective_timeout_ms: int | None = None) -> dict[str, Any]:
    configured = configured_timeout_ms()
    effective = effective_timeout_ms
    source = "argument"
    if effective is None:
        raw_effective = os.environ.get(TIMEOUT_ENV)
        if raw_effective:
            effective = int(raw_effective)
            source = TIMEOUT_ENV
        else:
            effective = configured
            source = "app.config.json"
    return {
        "configured_timeout_ms": configured,
        "effective_timeout_ms": effective,
        "effective_timeout_source": source,
        "effective_meets_configured": effective >= configured,
        "assumption": "A-004",
    }


def run_timeout_probe(probe_ms: int) -> dict[str, Any]:
    if probe_ms < 0:
        raise SaccadeError(
            "E_BAD_ARGUMENT",
            "timeout",
            "timeout probe duration must be non-negative",
            {"probe_ms": probe_ms},
        )
    long_probe_enabled = os.environ.get(LONG_TIMEOUT_PROBE_ENV) == "1"
    if probe_ms > TIMEOUT_PROBE_LIMIT_MS and not long_probe_enabled:
        raise SaccadeError(
            "E_BAD_ARGUMENT",
            "timeout",
            "long timeout probes require SACCADE_ENABLE_LONG_TIMEOUT_PROBE=1",
            {"probe_ms": probe_ms, "limit_ms": TIMEOUT_PROBE_LIMIT_MS},
        )
    started = time.monotonic()
    time.sleep(probe_ms / 1000)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        **timeout_diagnostics(),
        "probe_requested_ms": probe_ms,
        "probe_elapsed_ms": elapsed_ms,
        "long_probe_enabled": long_probe_enabled,
    }


def local_vision_diagnostics(args: dict[str, Any] | None = None) -> dict[str, Any]:
    config = local_vision_config_from_args(args or {})
    probe = probe_local_vision(config)
    return {
        "config": config.public_dict(),
        "probe": {
            "available": probe.available,
            "backend": probe.backend,
            "model": probe.model,
            "base_url": probe.base_url,
            "code": probe.code,
            "message": probe.message,
            "detail": probe.detail,
        },
        "setup_command": f"ollama pull {config.model}",
        "lm_studio_note": "LM Studio downloads do not satisfy Ollama model availability because the model stores are separate.",
    }
