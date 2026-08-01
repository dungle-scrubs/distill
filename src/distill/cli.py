"""Command line interface for Distill, and the error boundary around it.

This is a signed module (ADR-0003). It writes nothing into a bundle itself, but
it decides which parsed values reach `DistillOptions`, and every manifest
records the resulting option set - including `output_dir`, `force_reprocess`,
and `resume_partial`, which are deliberately *not* part of the bundle key.
Changing what this module forwards can therefore change manifest content at an
unchanged bundle key, which is the stale hit the signature exists to catch.

It also owns Distill's outermost error boundary (R-46). Every exception a
command raises leaves here as one shape - the **fatal error** record as JSON on
stderr, exit code 2 - so an operator scripting Distill has one thing to parse
and one code to branch on. Before that, only `DistillError` was converted and
everything else left as a Python traceback with exit 1 (finding 14): an exit
code that says nothing, a stack that names Distill's internals, and no code or
stage to key on.

*Raises*, and not "ends badly", because three ways a command ends are outside
that shape on purpose and each one has a better answer than a record:

- **An argument the parser rejects** is argparse's usage message and exit 2.
  The usage message says what the accepted spellings are; a JSON record would
  not.
- **`Ctrl-C`** is `KeyboardInterrupt`: the operator ending their own command.
  It leaves with the interpreter's own traceback and exit 130, because it is
  not a failure to diagnose and not Distill's to relabel.
- **A caller that stops reading stdout** (`distill … | head`) is exit 141, the
  conventional 128 + SIGPIPE. Nothing failed; the reader left.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, NoReturn

from .errors import INTERNAL_CODE, INTERNAL_STAGE, DistillError
from .options import OPTION_SPECS, PROCESSING_OPTION_NAMES
from .pipeline import (
    DistillSession,
    call_registered_tool,
    filtered_view,
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


class _NobodyIsReadingTheOutput(Exception):
    """Raised where Distill writes its result and finds the pipe has no reader.

    A marker, not a diagnosis. `BrokenPipeError` means "this pipe has no
    reader", and only the pipe carrying the *result* says anything about the
    caller: a **progress** record written to a stderr nobody reads raises the
    same class, and so would any socket or pipe a stage happens to hold. Caught
    around the whole dispatch, `BrokenPipeError` answered all of those with the
    caller-left-early exit, which turns an uncoded failure into something that
    looks like a caller's choice.

    So the two places Distill writes to stdout raise this instead, and `main`
    catches this rather than `BrokenPipeError`. Anything else that breaks a pipe
    reaches the boundary as what it is: an exception a command raised.
    """


def _write_stdout(text: str) -> None:
    """Put `text` on stdout, saying so if the caller has stopped reading."""
    try:
        print(text)
    except BrokenPipeError as gone:
        raise _NobodyIsReadingTheOutput from gone


def _print_json(payload: Any) -> None:
    """The one write to stdout, so the boundary's guarantees cover all of them.

    Every command's result leaves through here - `list-tools` included, which
    used to print from `pipeline` and so sat outside whatever this boundary
    promises about the output stream.
    """
    _write_stdout(json.dumps(payload, indent=2, sort_keys=True))


BROKEN_PIPE_EXIT = 141
"""What a command exits when the caller stopped reading its output: 128 + SIGPIPE.

`distill … | head` is the shell's own idiom, not a Distill failure, and it is
not the operator's mistake either. Reported as an `E_INTERNAL` record it read as
a defect in Distill; reported as the conventional signal exit it reads as what
it is, and a shell that checks `$?` sees the number every other program uses.
"""

GONE_STREAM_ERRORS = (OSError, ValueError)
"""What a stream that is no longer usable answers with, both spellings.

`OSError` is a descriptor that has gone away: closed underneath the stream,
`EPIPE`, `EBADF`. `ValueError: I/O operation on closed file` is the *stream*
having gone away, which is a different object being in a different state, and a
guard that names only the first lets the second out of a boundary that exists so
nothing gets out.
"""


def _discard(stream: Any) -> None:
    """Point `stream`'s descriptor at the null device, and stop worrying about it.

    The interpreter flushes both standard streams on the way out. When the
    reader is gone that flush fails *after* everything this module controls has
    finished, and CPython answers by printing "Exception ignored on flushing
    sys.stdout" and exiting 120 - a second diagnosis, on the error channel, of
    something that was never Distill's failure, replacing the exit code the
    contract names. Re-pointing the descriptor is the standard recipe: the
    buffered bytes have nowhere to go either way, and this is the only way to
    say so before shutdown asks.

    The descriptor may already *be* the null device, and then there is nothing
    to do and something not to do. With file descriptor 1 closed rather than
    dead, `os.open` is handed 1 as the lowest free number, so the open itself
    installs the null device where stdout belongs; `dup2` is then a no-op and
    closing what was opened closes stdout a second time, which is the failure
    this function was called to prevent, one step later.
    """
    if stream is None:
        return
    try:
        target = stream.fileno()
    except GONE_STREAM_ERRORS:
        return
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    if devnull == target:
        return
    try:
        os.dup2(devnull, target)
    except OSError:
        pass
    finally:
        os.close(devnull)


def _flush_stdout_quietly() -> None:
    """Empty the output buffer on a failing path, without adding a second failure.

    Whatever a command wrote before it failed has already been written; what
    must not happen is the interpreter discovering a dead stdout during its own
    shutdown flush and printing noise *after* the error record, on the channel
    the record was just written to.
    """
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.flush()
    except GONE_STREAM_ERRORS:
        _discard(stream)


TRACEBACK_ENV = "DISTILL_TRACEBACK"
"""Set to `1` to have the boundary re-raise instead of converting.

The escape hatch exists because the boundary's whole value - one line, no stack
- is the wrong trade for the one person who has to fix what it names. It is opt
-in rather than a flag so it can be turned on around a command already written,
and it is off by default because a traceback is not a contract.
"""


def _emit_error(error: dict[str, Any]) -> None:
    """Write `error` to the error channel, or write nothing anywhere.

    Two ways the channel is not there, and both used to end somewhere worse
    than silence:

    - **No stream at all.** An interpreter started with file descriptor 2
      closed has `sys.stderr is None`, and `print(..., file=None)` falls back to
      *stdout*. The one channel this boundary promises to keep clean is
      precisely where the error record went, so a caller piping stdout to `jq`
      got a result-shaped error after all.
    - **A stream nobody is reading.** `_fail` is called from inside an `except`
      clause, so an `OSError` raised by the write is caught by nothing: it
      leaves `main` as a second, unrelated failure and the process ends on the
      exit code for "could not flush the std streams" rather than on the one the
      contract names.

    A caller that closed the error channel has said it does not want the
    diagnosis. It has not changed what happened, so the exit code is unchanged.
    """
    stream = sys.stderr
    if stream is None:
        return
    try:
        print(json.dumps(error, sort_keys=True), file=stream)
        stream.flush()
    except GONE_STREAM_ERRORS:
        # A failed flush leaves the bytes in the buffer, where the interpreter's
        # own shutdown flush finds them and fails again - which is the exit-120
        # path, one step further along.
        _discard(stream)


def _fail(error: dict[str, Any]) -> NoReturn:
    """End the command on `error`, the one way any command ends badly.

    stderr, so a caller reading stdout with `jq` gets either a result or
    nothing rather than a result-shaped error. Exit 2, which is also what
    argparse uses for a usage error: both mean "this command produced no
    output", and an operator branching on non-zero does not have to know which
    kind it was.

    The carve-out, stated because the promise is otherwise wider than the
    mechanism: stdout is clean because nothing writes a failure to it, not
    because a write can be undone. A command that fails *after* printing part
    of its result - which is what a caller closing the pipe mid-write produces -
    leaves those bytes where they landed.
    """
    _flush_stdout_quietly()
    _emit_error(error)
    raise SystemExit(2)


def _tool_args(raw: str) -> dict[str, Any]:
    """The `--args` document, or `E_BAD_ARGUMENT` saying why it is not one.

    `E_BAD_ARGUMENT` and not `E_INTERNAL`: an operator who mistyped a JSON
    literal has not found a defect in Distill, and the catch-all's "an
    unexpected error" would send them looking for one. A document that parses
    but is not an object is the same mistake seen a moment later - it reached
    the tool as a list and died on `.get` - so it gets the same answer here.
    """
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise DistillError(
            "E_BAD_ARGUMENT",
            "cli",
            "--args must be a JSON object",
            {"error": str(exc)},
        ) from exc
    if not isinstance(parsed, dict):
        raise DistillError(
            "E_BAD_ARGUMENT",
            "cli",
            "--args must be a JSON object",
            {"received": type(parsed).__name__},
        )
    return parsed


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
    _add_common_processing_options(directory)

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

    # Read-only like `cache-doctor`, and deliberately without a --dry-run flag
    # for the same reason: there is no other mode. A preview exists to stand in
    # for a run that would change something, and this one produces a view on
    # demand that is never written back (D-006), so a `--dry-run` would preview
    # itself. The pair of arguments is `get-job-status`'s - one identifier the
    # record is under, and the root to look for it in - because the question is
    # the same question asked of a bundle instead of a job.
    view = subcommands.add_parser(
        "filtered-view", help="Render the non-authoritative filtered view of a published bundle"
    )
    view.add_argument("bundle_key")
    view.add_argument("--output-dir")

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


def _dispatch(argv: list[str] | None) -> None:
    """Parse `argv` and run the command it names. Everything a boundary covers.

    Parsing is *inside* the boundary and its answer to a typo is unchanged, and
    those two facts are not in tension. Argparse ends a usage error by raising
    `SystemExit`, which is a `BaseException` and passes through a clause naming
    `Exception` untouched - so the usage message still stands alone, still exits
    2, and still carries no error object. What the boundary picks up is the
    other half: a type converter, a default, or an `argparse` action raising
    something that is *not* a usage error. Those ran outside every handler, and
    left as a traceback.
    """
    args = build_parser().parse_args(argv)
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
        keys = (*PROCESSING_KEYS, "recursive", "max_items", "continue_on_error")
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
    elif args.command == "filtered-view":
        _print_json(filtered_view(_args_payload(args, ("bundle_key", "output_dir"))))
    elif args.command == "list-tools":
        _print_json(list_tools())
    elif args.command == "timeout-diagnostics":
        _print_json(timeout_diagnostics())
    elif args.command == "timeout-probe":
        _print_json(run_timeout_probe(args.probe_ms))
    elif args.command == "local-vision-diagnostics":
        payload = _args_payload(args, LOCAL_VISION_DIAGNOSTIC_KEYS)
        _print_json(local_vision_diagnostics(payload))
    elif args.command == "call-tool":
        response = DistillSession().call_tool(args.tool, _tool_args(args.args))
        # The session speaks the MCP envelope, where a failure is a result
        # with an `error` in it. A CLI does not: R-46 is one exit code and
        # one record, so an envelope carrying an error ends the command the
        # way every other failure does rather than printing success-shaped
        # JSON and exiting 0.
        error = response.get("error")
        if isinstance(error, dict):
            _fail(error)
        _print_json(response)
    else:
        # The parser accepted a command this chain has no branch for, which is
        # a defect in this module and not in the operator's argv. Without this
        # arm the chain simply falls off the end: the command exits 0 having
        # printed nothing, which is indistinguishable from a command that
        # succeeded and had nothing to say.
        raise DistillError(
            INTERNAL_CODE,
            INTERNAL_STAGE,
            f"the {args.command} command is registered but no branch dispatches it",
            {"command": args.command},
        )


def main(argv: list[str] | None = None) -> None:
    """Run one command, and convert every way it can fail (R-46).

    `Exception`, not `BaseException`, and the two things that pass through are
    the two the boundary has nothing better to say about. `SystemExit` is
    argparse's answer to a typo and `_fail`'s own; `KeyboardInterrupt` is the
    operator interrupting their own command, which is not a failure to diagnose
    and not Distill's to relabel.

    `_NobodyIsReadingTheOutput` is caught first and separately. It is the one
    ending that is not a failure at all - the caller stopped reading, which
    `| head` does on purpose - so it ends the command on the conventional signal
    exit rather than as a record saying an unexpected error occurred. It is
    raised only where Distill writes its result, so a broken pipe anywhere else
    is still an exception a command raised and is reported as one.
    """
    try:
        _dispatch(argv)
        if sys.stdout is not None:
            # Inside the boundary, deliberately. `print` buffers when stdout is
            # a pipe, so a caller that walked away is not discovered until the
            # buffer is emptied - and left to the interpreter's own shutdown
            # flush, that discovery happens where nothing can handle it.
            try:
                sys.stdout.flush()
            except BrokenPipeError as gone:
                raise _NobodyIsReadingTheOutput from gone
    except _NobodyIsReadingTheOutput:
        _discard(sys.stdout)
        raise SystemExit(BROKEN_PIPE_EXIT) from None
    except DistillError as exc:
        _fail(exc.to_dict())
    except Exception as exc:
        if os.environ.get(TRACEBACK_ENV) == "1":
            raise
        _fail(DistillError.from_unexpected(exc).to_dict())


if __name__ == "__main__":
    main()
