"""Command line interface for Distill.

This is a signed module (ADR-0003). It writes nothing into a bundle itself, but
it decides which parsed values reach `DistillOptions`, and every manifest
records the resulting option set - including `output_dir`, `force_reprocess`,
and `resume_partial`, which are deliberately *not* part of the bundle key.
Changing what this module forwards can therefore change manifest content at an
unchanged bundle key, which is the stale hit the signature exists to catch.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import DistillError
from .options import OPTION_SPECS, PROCESSING_OPTION_NAMES
from .pipeline import (
    DistillSession,
    call_registered_tool,
    list_tools,
    local_vision_diagnostics,
    run_timeout_probe,
    timeout_diagnostics,
)


def _add_common_processing_options(parser: argparse.ArgumentParser) -> None:
    specs = {spec.name: spec for spec in OPTION_SPECS}
    for key in PROCESSING_KEYS:
        flag = f"--{key.replace('_', '-')}"
        spec = specs.get(key)
        if spec and spec.boolean:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif spec and spec.caster in {int, float}:
            parser.add_argument(flag, type=spec.caster)
        elif key == "local_vision_timeout_sec":
            parser.add_argument(flag, type=float)
        elif key in {"caption_frames", "local_vision_allow_remote_endpoint"}:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(flag)


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
    *PROCESSING_OPTION_NAMES,
    "caption_frames",
    "local_vision_backend",
    "local_vision_model",
    "local_vision_base_url",
    "local_vision_timeout_sec",
    "local_vision_allow_remote_endpoint",
)

LOCAL_VISION_DIAGNOSTIC_KEYS = (
    "caption_frames",
    "local_vision_backend",
    "local_vision_model",
    "local_vision_base_url",
    "local_vision_timeout_sec",
    "local_vision_allow_remote_endpoint",
)
"""What `local-vision-diagnostics` forwards, named once so the parser can be checked.

`_args_payload` reads these off the namespace with a bare `getattr`, so a key
here that the subparser never registered is an `AttributeError` on every
invocation of the command rather than a missing option. Stating the list gives
a test something to hold both halves against.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distill", description="Video transcript and keyframe bundle CLI"
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
    _add_common_processing_options(playlist)

    cleanup = subcommands.add_parser("cleanup-cache", help="Prune Distill cache bundles")
    cleanup.add_argument("--output-dir")
    cleanup.add_argument("--max-age-days", type=float)
    cleanup.add_argument("--keep-generations", type=int)
    cleanup.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)

    # Read-only (R-57), and deliberately without a --dry-run flag: there is no
    # other mode. `cleanup-cache --dry-run` previews a command that can delete;
    # this one cannot, which is what makes it safe to reach for first.
    doctor = subcommands.add_parser(
        "cache-doctor", help="Inspect Distill cache bundles, locks and a prune preview"
    )
    doctor.add_argument("--output-dir")
    doctor.add_argument("--max-age-days", type=float)
    doctor.add_argument("--keep-generations", type=int)

    status = subcommands.add_parser("get-job-status", help="Read a Distill job status record")
    status.add_argument("job_id")
    status.add_argument("--output-dir")

    subcommands.add_parser("list-tools", help="List tool names")
    subcommands.add_parser("timeout-diagnostics", help="Show configured timeout diagnostics")

    timeout_probe = subcommands.add_parser(
        "timeout-probe", help="Sleep for a bounded timeout probe"
    )
    timeout_probe.add_argument("probe_ms", type=int)

    vision = subcommands.add_parser("local-vision-diagnostics", help="Probe local vision settings")
    vision.add_argument("--caption-frames", action=argparse.BooleanOptionalAction, default=None)
    vision.add_argument("--local-vision-backend")
    vision.add_argument("--local-vision-model")
    vision.add_argument("--local-vision-base-url")
    vision.add_argument("--local-vision-timeout-sec", type=float)
    vision.add_argument(
        "--local-vision-allow-remote-endpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

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
                call_registered_tool(
                    "process_local_video",
                    {**_args_payload(args, PROCESSING_KEYS), "path": args.path},
                )
            )
        elif args.command == "process-youtube-video":
            _print_json(
                call_registered_tool(
                    "process_youtube_video",
                    {**_args_payload(args, PROCESSING_KEYS), "url": args.url},
                )
            )
        elif args.command == "process-video-directory":
            keys = ("recursive", "max_items", "continue_on_error", "output_dir", "job_id")
            _print_json(
                call_registered_tool(
                    "process_video_directory",
                    {**_args_payload(args, keys), "path": args.path},
                )
            )
        elif args.command == "process-youtube-playlist":
            keys = (*PROCESSING_KEYS, "max_items", "continue_on_error")
            _print_json(
                call_registered_tool(
                    "process_youtube_playlist",
                    {**_args_payload(args, keys), "url": args.url},
                )
            )
        elif args.command == "cleanup-cache":
            keys = ("output_dir", "max_age_days", "keep_generations", "dry_run")
            _print_json(call_registered_tool("cleanup_cache", _args_payload(args, keys)))
        elif args.command == "cache-doctor":
            keys = ("output_dir", "max_age_days", "keep_generations")
            _print_json(call_registered_tool("cache_doctor", _args_payload(args, keys)))
        elif args.command == "get-job-status":
            _print_json(
                call_registered_tool("get_job_status", _args_payload(args, ("job_id", "output_dir")))
            )
        elif args.command == "list-tools":
            list_tools()
        elif args.command == "timeout-diagnostics":
            _print_json(timeout_diagnostics())
        elif args.command == "timeout-probe":
            _print_json(run_timeout_probe(args.probe_ms))
        elif args.command == "local-vision-diagnostics":
            payload = _args_payload(args, LOCAL_VISION_DIAGNOSTIC_KEYS)
            _print_json(local_vision_diagnostics(payload))
        elif args.command == "call-tool":
            response = DistillSession().call_tool(args.tool, json.loads(args.args))
            _print_json(response)
    except DistillError as exc:
        print(exc.to_json_text(), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
