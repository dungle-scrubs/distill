"""Command line interface for Saccade."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import SaccadeError
from .pipeline import (
    SaccadeSession,
    cleanup_saccade_cache,
    get_job_status,
    list_tools,
    local_vision_diagnostics,
    process_local_video,
    process_video_directory,
    process_youtube_playlist,
    process_youtube_video,
    run_timeout_probe,
    timeout_diagnostics,
)


def _add_common_processing_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--redact-secrets", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--whisper-model")
    parser.add_argument("--whisper-language")
    parser.add_argument("--ocr-language")
    parser.add_argument("--max-keyframes", type=int)
    parser.add_argument("--min-interval-sec", type=float)
    parser.add_argument("--max-static-window-sec", type=float)
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--caption-frames", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local-vision-model")
    parser.add_argument("--local-vision-base-url")
    parser.add_argument("--local-vision-timeout-sec", type=float)
    parser.add_argument("--job-id")
    parser.add_argument("--resume-partial", action=argparse.BooleanOptionalAction, default=None)


def _args_payload(args: argparse.Namespace, keys: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in keys:
        value = getattr(args, key)
        if value is not None:
            payload[key] = value
    return payload


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


PROCESSING_KEYS = (
    "output_dir",
    "force_reprocess",
    "ocr",
    "redact_secrets",
    "whisper_model",
    "whisper_language",
    "ocr_language",
    "max_keyframes",
    "min_interval_sec",
    "max_static_window_sec",
    "max_duration_sec",
    "vad_filter",
    "caption_frames",
    "local_vision_model",
    "local_vision_base_url",
    "local_vision_timeout_sec",
    "job_id",
    "resume_partial",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saccade", description="Video transcript and keyframe bundle CLI"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    local = subcommands.add_parser("process-local-video", help="Process one local video file")
    local.add_argument("path")
    _add_common_processing_options(local)

    youtube = subcommands.add_parser("process-youtube-video", help="Process one YouTube video URL")
    youtube.add_argument("url")
    _add_common_processing_options(youtube)

    directory = subcommands.add_parser(
        "process-video-directory", help="Process videos in a directory"
    )
    directory.add_argument("path")
    directory.add_argument("--recursive", action="store_true")
    directory.add_argument("--max-items", type=int)
    directory.add_argument(
        "--continue-on-error", action=argparse.BooleanOptionalAction, default=None
    )
    directory.add_argument("--output-dir")
    directory.add_argument("--job-id")

    playlist = subcommands.add_parser(
        "process-youtube-playlist", help="Process a YouTube playlist or channel"
    )
    playlist.add_argument("url")
    playlist.add_argument("--max-items", type=int)
    playlist.add_argument(
        "--continue-on-error", action=argparse.BooleanOptionalAction, default=None
    )
    playlist.add_argument("--output-dir")
    playlist.add_argument("--job-id")

    cleanup = subcommands.add_parser("cleanup-cache", help="Prune Saccade cache bundles")
    cleanup.add_argument("--output-dir")
    cleanup.add_argument("--max-age-days", type=float)
    cleanup.add_argument("--keep-generations", type=int)
    cleanup.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)

    status = subcommands.add_parser("get-job-status", help="Read a Saccade job status record")
    status.add_argument("job_id")
    status.add_argument("--output-dir")

    subcommands.add_parser("list-tools", help="List tool names")
    subcommands.add_parser("timeout-diagnostics", help="Show configured timeout diagnostics")

    timeout_probe = subcommands.add_parser(
        "timeout-probe", help="Sleep for a bounded timeout probe"
    )
    timeout_probe.add_argument("probe_ms", type=int)

    vision = subcommands.add_parser("local-vision-diagnostics", help="Probe local vision settings")
    vision.add_argument("--local-vision-model")

    raw = subcommands.add_parser(
        "call-tool", help="Call a package tool by MCP-style name and JSON args"
    )
    raw.add_argument("tool")
    raw.add_argument("--args", default="{}")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "process-local-video":
            _print_json(
                process_local_video({**_args_payload(args, PROCESSING_KEYS), "path": args.path})
            )
        elif args.command == "process-youtube-video":
            _print_json(
                process_youtube_video({**_args_payload(args, PROCESSING_KEYS), "url": args.url})
            )
        elif args.command == "process-video-directory":
            keys = ("recursive", "max_items", "continue_on_error", "output_dir", "job_id")
            _print_json(process_video_directory({**_args_payload(args, keys), "path": args.path}))
        elif args.command == "process-youtube-playlist":
            keys = ("max_items", "continue_on_error", "output_dir", "job_id")
            _print_json(process_youtube_playlist({**_args_payload(args, keys), "url": args.url}))
        elif args.command == "cleanup-cache":
            keys = ("output_dir", "max_age_days", "keep_generations", "dry_run")
            _print_json(cleanup_saccade_cache(_args_payload(args, keys)))
        elif args.command == "get-job-status":
            _print_json(get_job_status(_args_payload(args, ("job_id", "output_dir"))))
        elif args.command == "list-tools":
            list_tools()
        elif args.command == "timeout-diagnostics":
            _print_json(timeout_diagnostics())
        elif args.command == "timeout-probe":
            _print_json(run_timeout_probe(args.probe_ms))
        elif args.command == "local-vision-diagnostics":
            payload = _args_payload(args, ("local_vision_model",))
            _print_json(local_vision_diagnostics(payload))
        elif args.command == "call-tool":
            response = SaccadeSession().call_tool(args.tool, json.loads(args.args))
            _print_json(response)
    except SaccadeError as exc:
        print(exc.to_json_text(), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
