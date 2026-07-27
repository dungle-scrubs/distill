"""The single subprocess path for Distill.

This module owns how an external tool is invoked: the argument list (never a
shell), the child's own process group, the total deadline and the idle timeout
that every invocation carries, concurrent draining of stdout and stderr, the
size cap on captured output, graceful-then-forced termination of the whole
group, and the one table that maps a subprocess failure onto Distill's
`DistillError` taxonomy. It is the generalization of the former
`source.run_json_command`, whose `E_MISSING_TOOL`/`E_COMMAND` mapping it keeps.

It does not own which tool to run, what a tool's output means, whether a
failure is a **degradation** or a **fatal error**, or the retry policy around
one. Callers decide those: a caller that treats its tool as an **optional
capability** catches the error and records a **warning** (ADR-0002), and a
caller that needs the tool lets the error end the run.

Two entry points, one core:

- `run()` captures both streams and returns them.
- `stream()` additionally hands each line to a callback as it arrives, for a
  caller that must consume output live (yt-dlp's download progress).

Why both timeouts (R-30). A total deadline alone either kills a legitimate
long download or is set so high it never protects anything; an idle timeout
alone lets a tool that emits a heartbeat run forever. Every invocation carries
both, and neither has a default meaning "no limit": `total_timeout_sec` is
required, `idle_timeout_sec` defaults to 120 s, and a non-finite or
non-positive value for either is rejected.

Why a process group (R-31). ffmpeg and yt-dlp spawn helpers. Signalling only
the direct child on a timeout leaves those helpers running, holding the pipes
and the network connection - finding 8. The child is started in its own
session, and a timeout signals the group: SIGTERM, a grace period, then
SIGKILL. Signals are sent while the direct child is still unreaped, so the
group id cannot have been recycled onto an unrelated process.

Why a boundary event (R-29 observability). Because this is the only path an
external tool is invoked on, one event emitted here covers every invocation in
the pipeline without a call site remembering to log. Every invocation emits
exactly one, at the end, whether it succeeded, failed, or never started; see
`_boundary_log`.

Known limits. A grandchild that inherits the pipes and outlives its parent is
not killed when the parent exits on its own - only a timeout terminates the
group - and draining is bounded rather than waiting for such a grandchild to
close the pipes, so the tail of that output is dropped instead of hanging the
run. Process control here is POSIX-only.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import DistillError, warning

LOGGER = logging.getLogger(__name__)

# R-30: an invocation with no output for this long is treated as stalled. Named
# rather than repeated at call sites so raising it is one edit.
DEFAULT_IDLE_TIMEOUT_SEC = 120.0
# R-33: per-stream capture cap. Output past it is drained and discarded, so the
# cap bounds memory without re-creating the deadlock it exists alongside.
OUTPUT_CAP_BYTES = 8 * 1024 * 1024
# How long the group gets to honour SIGTERM before SIGKILL follows.
DEFAULT_TERMINATE_GRACE_SEC = 5.0
TRUNCATION_WARNING_CODE = "command_output_truncated"

# How long the readers get to finish once the child is gone. Bounded because a
# grandchild may hold the pipes open indefinitely.
_DRAIN_GRACE_SEC = 1.0
_POLL_INTERVAL_SEC = 0.01
_READ_CHUNK_BYTES = 64 * 1024
# A "line" this long without a terminator is delivered as-is, so a tool that
# never emits a newline cannot grow the pending buffer without bound.
_MAX_LINE_BYTES = 64 * 1024
# Lines handed to a callback per turn of the supervision loop.
_DISPATCH_BATCH = 64
_STDERR_TAIL_CHARS = 2000
_STDOUT = "stdout"
_STDERR = "stderr"


class FailureKind(Enum):
    """Every way an invocation can fail. The error table is keyed by this."""

    MISSING_TOOL = "missing_tool"
    TOTAL_TIMEOUT = "total_timeout"
    IDLE_TIMEOUT = "idle_timeout"
    EXIT_STATUS = "exit_status"
    BAD_JSON = "bad_json"


# The single error-mapping table. `E_MISSING_TOOL` and `E_COMMAND` are carried
# over from `source.run_json_command` so migrating call sites does not change
# which code a caller sees. A caller may override the command code (yt-dlp
# reports `E_YTDLP`); the missing-tool code is fixed, because "not installed"
# means the same thing for every tool.
ERROR_CODES: dict[FailureKind, str] = {
    FailureKind.MISSING_TOOL: "E_MISSING_TOOL",
    FailureKind.TOTAL_TIMEOUT: "E_COMMAND",
    FailureKind.IDLE_TIMEOUT: "E_COMMAND",
    FailureKind.EXIT_STATUS: "E_COMMAND",
    FailureKind.BAD_JSON: "E_COMMAND",
}

_TIMEOUT_KINDS: dict[str, FailureKind] = {
    "total": FailureKind.TOTAL_TIMEOUT,
    "idle": FailureKind.IDLE_TIMEOUT,
}


@dataclass(frozen=True)
class CommandTimeouts:
    """The pair of limits one call site invokes its tool under (R-30).

    Both are required: there is no constructor that produces a half-specified
    pair, so a call site cannot express "bound the total but not the stall" by
    omission. Each is validated the moment it is written down rather than when
    the tool is finally run, which is what makes a wrong limit a startup error
    instead of a surprise an hour into a download.

    A long download is bounded by `idle_sec` rather than `total_sec`: the total
    for such a call site is deliberately generous, because the limit that
    detects a wedged transfer is the absence of output, not the elapsed time.
    """

    total_sec: float
    idle_sec: float

    def __post_init__(self) -> None:
        _validate_timeout("total_sec", self.total_sec)
        _validate_timeout("idle_sec", self.idle_sec)


@dataclass(frozen=True)
class CommandResult:
    """What one invocation produced, including how it was reduced.

    `warnings` carries the **degradation** record for truncated capture, so a
    caller can attach it to the run without knowing the cap exists.
    """

    tool: str
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_sec: float
    warnings: tuple[dict[str, str], ...] = ()

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated


def run(
    argv: Sequence[str],
    *,
    stage: str,
    total_timeout_sec: float,
    idle_timeout_sec: float = DEFAULT_IDLE_TIMEOUT_SEC,
    check: bool = True,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    output_cap_bytes: int = OUTPUT_CAP_BYTES,
    terminate_grace_sec: float = DEFAULT_TERMINATE_GRACE_SEC,
    error_code: str | None = None,
) -> CommandResult:
    """Run `argv` to completion and return what it produced.

    `stage` names the pipeline stage for the error and any **warning**.
    `check` raises on a non-zero exit; a caller that degrades on failure passes
    `check=False` and reads `returncode`. A missing tool always raises, because
    there is no result to return.
    """
    return _execute(
        argv,
        stage=stage,
        total_timeout_sec=total_timeout_sec,
        idle_timeout_sec=idle_timeout_sec,
        check=check,
        cwd=cwd,
        env=env,
        output_cap_bytes=output_cap_bytes,
        terminate_grace_sec=terminate_grace_sec,
        error_code=error_code,
        on_stdout_line=None,
        on_stderr_line=None,
    )


def stream(
    argv: Sequence[str],
    *,
    stage: str,
    total_timeout_sec: float,
    idle_timeout_sec: float = DEFAULT_IDLE_TIMEOUT_SEC,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    check: bool = True,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    output_cap_bytes: int = OUTPUT_CAP_BYTES,
    terminate_grace_sec: float = DEFAULT_TERMINATE_GRACE_SEC,
    error_code: str | None = None,
) -> CommandResult:
    """Run `argv`, handing each output line to a callback as it arrives.

    Callbacks run on the calling thread, not on the reader threads, so a
    caller's progress emitter needs no locking. Lines are delivered without
    their terminator; both `\\n` and `\\r` end a line, because progress-writing
    tools use either. Empty lines are not delivered. The full output is still
    captured and returned.
    """
    return _execute(
        argv,
        stage=stage,
        total_timeout_sec=total_timeout_sec,
        idle_timeout_sec=idle_timeout_sec,
        check=check,
        cwd=cwd,
        env=env,
        output_cap_bytes=output_cap_bytes,
        terminate_grace_sec=terminate_grace_sec,
        error_code=error_code,
        on_stdout_line=on_stdout_line,
        on_stderr_line=on_stderr_line,
    )


def run_json(
    argv: Sequence[str],
    *,
    stage: str,
    total_timeout_sec: float,
    idle_timeout_sec: float = DEFAULT_IDLE_TIMEOUT_SEC,
    **options: Any,
) -> Any:
    """Run `argv` and parse its stdout as JSON, mapping a bad document as a failure."""
    result = run(
        argv,
        stage=stage,
        total_timeout_sec=total_timeout_sec,
        idle_timeout_sec=idle_timeout_sec,
        **options,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise _failure(
            FailureKind.BAD_JSON,
            argv,
            stage=stage,
            details=_failure_details(
                argv,
                exit_status=result.returncode,
                timeout_fired=None,
                stdout_truncated=result.stdout_truncated,
                stderr_truncated=result.stderr_truncated,
                stderr=result.stderr,
                duration_sec=result.duration_sec,
            ),
        ) from exc


# --- failure construction ----------------------------------------------------


def _tool_of(argv: Sequence[str]) -> str:
    return argv[0]


def _message(kind: FailureKind, argv: Sequence[str], *, idle_timeout_sec: float = 0.0) -> str:
    tool = _tool_of(argv)
    return {
        FailureKind.MISSING_TOOL: f"required tool is not installed: {tool}",
        FailureKind.TOTAL_TIMEOUT: f"command exceeded its total deadline: {tool}",
        FailureKind.IDLE_TIMEOUT: (
            f"command produced no output for {idle_timeout_sec:g}s: {tool}"
        ),
        FailureKind.EXIT_STATUS: f"command failed: {tool}",
        FailureKind.BAD_JSON: f"command returned invalid JSON: {tool}",
    }[kind]


def _failure_details(
    argv: Sequence[str],
    *,
    exit_status: int | None,
    timeout_fired: str | None,
    stdout_truncated: bool,
    stderr_truncated: bool,
    stderr: str = "",
    duration_sec: float | None = None,
) -> dict[str, Any]:
    """The structured failure payload every subprocess failure carries.

    Enough to trace a failure back to the run that produced it without
    reproducing it: which tool, which arguments, how it ended, which timeout
    fired if any, and whether what is reported here was itself truncated.
    """
    return {
        "tool": _tool_of(argv),
        "argv": list(argv),
        "exit_status": exit_status,
        "timeout_fired": timeout_fired,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stderr_tail": stderr[-_STDERR_TAIL_CHARS:],
        "duration_sec": duration_sec,
    }


def _failure(
    kind: FailureKind,
    argv: Sequence[str],
    *,
    stage: str,
    details: dict[str, Any],
    error_code: str | None = None,
    idle_timeout_sec: float = 0.0,
) -> DistillError:
    code = ERROR_CODES[kind]
    if error_code is not None and kind is not FailureKind.MISSING_TOOL:
        code = error_code
    return DistillError(
        code,
        stage,
        _message(kind, argv, idle_timeout_sec=idle_timeout_sec),
        details,
    )


# --- the boundary event ------------------------------------------------------

BOUNDARY_EVENT_TYPE = "distill.run_command"
BOUNDARY_EVENT_NAME = "tool_invocation"

# Ordinal of the invocation within this process. A run is one process, so the
# pid plus this ordinal orders every tool Distill spawned and ties each event to
# the run that spawned it without a run id having to be threaded through.
_invocation_sequence = itertools.count(1)


def _boundary_log(
    *,
    outcome: str,
    stage: str,
    argv: Sequence[str],
    sequence: int,
    exit_status: int | None,
    timeout_fired: str | None,
    duration_sec: float,
    stdout_truncated: bool,
    stderr_truncated: bool,
) -> None:
    """Emit the one event this invocation produces (R-29 observability).

    Metadata only. The tool's captured output is deliberately absent: it is
    **extracted text**, it has not passed a **redaction sink**, and a debug log
    is not one. What is here answers which tool ran, under which stage, in what
    order within the run, how it ended, and how long it took - enough to trace a
    hang or a degradation back to the invocation that caused it.
    """
    LOGGER.debug(
        json.dumps(
            {
                "type": BOUNDARY_EVENT_TYPE,
                "event": BOUNDARY_EVENT_NAME,
                "detail": {
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "stage": stage,
                    "tool": _tool_of(argv),
                    "argv": list(argv),
                    "outcome": outcome,
                    "exit_status": exit_status,
                    "timeout_fired": timeout_fired,
                    "duration_sec": round(duration_sec, 6),
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
            },
            sort_keys=True,
        )
    )


# --- capture and draining ----------------------------------------------------


class _CappedSink:
    """Accumulates bytes up to a cap, counting the rest as truncated."""

    def __init__(self, cap_bytes: int) -> None:
        self._cap = cap_bytes
        self._parts: list[bytes] = []
        self._size = 0
        self.truncated = False

    def add(self, chunk: bytes) -> None:
        room = self._cap - self._size
        if room > 0:
            kept = chunk[:room]
            self._parts.append(kept)
            self._size += len(kept)
        if len(chunk) > room:
            self.truncated = True

    def text(self) -> str:
        return b"".join(self._parts).decode("utf-8", errors="replace")


class _ActivityClock:
    """When the child last produced output on either stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def mark(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    @property
    def last(self) -> float:
        with self._lock:
            return self._last


class _StreamReader(threading.Thread):
    """Drains one pipe. One per stream, so neither can starve the other."""

    def __init__(
        self,
        name: str,
        source: Any,
        *,
        sink: _CappedSink,
        clock: _ActivityClock,
        lines: queue.SimpleQueue[tuple[str, str]] | None,
    ) -> None:
        super().__init__(name=f"run_command-{name}", daemon=True)
        self._name = name
        self._source = source
        self._sink = sink
        self._clock = clock
        self._lines = lines
        self._pending = b""

    def run(self) -> None:
        try:
            while True:
                chunk = self._source.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                self._clock.mark()
                self._sink.add(chunk)
                if self._lines is not None:
                    self._split(chunk)
        except (OSError, ValueError):
            # The pipe was closed underneath us during termination.
            pass
        finally:
            if self._lines is not None:
                if self._pending:
                    self._emit(self._pending)
                    self._pending = b""
                self._lines.put((_EOF, ""))

    def _split(self, chunk: bytes) -> None:
        parts = (self._pending + chunk).split(b"\n")
        self._pending = parts.pop()
        for part in parts:
            if part.endswith(b"\r"):
                part = part[:-1]
            for piece in part.split(b"\r"):
                self._emit(piece)
        if len(self._pending) >= _MAX_LINE_BYTES:
            self._emit(self._pending)
            self._pending = b""

    def _emit(self, raw: bytes) -> None:
        if not raw or self._lines is None:
            return
        self._lines.put((self._name, raw.decode("utf-8", errors="replace")))


_EOF = "__eof__"


# --- the core ----------------------------------------------------------------


def _validate_timeout(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number of seconds, got {value!r}")
    return number


def _terminate_group(proc: subprocess.Popen[bytes], grace_sec: float) -> int | None:
    """SIGTERM the child's group, then SIGKILL it if the group is still there.

    Signalling happens before the direct child is reaped, so the group id is
    still reserved and cannot have been recycled onto an unrelated process.
    """
    pgid: int | None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    if pgid is not None and pgid == os.getpgrp():
        # start_new_session did not take effect; signalling this group would
        # signal Distill itself.
        pgid = None

    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        try:
            if pgid is not None:
                os.killpg(pgid, signal_number)
            else:
                proc.send_signal(signal_number)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            return proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            continue
    try:
        return proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
        return proc.returncode


def _deliver(
    name: str,
    text: str,
    on_stdout_line: Callable[[str], None] | None,
    on_stderr_line: Callable[[str], None] | None,
) -> None:
    if name == _STDOUT and on_stdout_line is not None:
        on_stdout_line(text)
    elif name == _STDERR and on_stderr_line is not None:
        on_stderr_line(text)


def _dispatch(
    lines: queue.SimpleQueue[tuple[str, str]],
    timeout_sec: float,
    on_stdout_line: Callable[[str], None] | None,
    on_stderr_line: Callable[[str], None] | None,
) -> None:
    """Hand queued lines to their callbacks, waiting up to `timeout_sec` for one.

    Doubles as the wait in the supervision loop: with no callbacks registered
    nothing is ever queued and this is simply a bounded sleep. A batch rather
    than a single line per turn, so a chatty tool cannot make the queue grow
    faster than the loop empties it.
    """
    try:
        name, text = lines.get(timeout=timeout_sec)
    except queue.Empty:
        return
    _deliver(name, text, on_stdout_line, on_stderr_line)
    for _ in range(_DISPATCH_BATCH - 1):
        try:
            name, text = lines.get_nowait()
        except queue.Empty:
            return
        _deliver(name, text, on_stdout_line, on_stderr_line)


def _drain_queue(
    lines: queue.SimpleQueue[tuple[str, str]],
    on_stdout_line: Callable[[str], None] | None,
    on_stderr_line: Callable[[str], None] | None,
) -> None:
    while True:
        try:
            name, text = lines.get_nowait()
        except queue.Empty:
            return
        _deliver(name, text, on_stdout_line, on_stderr_line)


def _outcome_of(timeout_fired: str | None, exit_status: int | None, *, check: bool) -> str:
    """How the invocation ended, for the boundary event.

    A non-zero exit under `check=False` is reported as it happened rather than
    as success: the caller chose to handle it, which does not make it a clean
    run, and a **degradation** is exactly what an operator comes to this event
    looking for.
    """
    if timeout_fired is not None:
        return _TIMEOUT_KINDS[timeout_fired].value
    if exit_status != 0:
        return FailureKind.EXIT_STATUS.value if check else "nonzero_exit_unchecked"
    return "ok"


def _execute(
    argv: Sequence[str],
    *,
    stage: str,
    total_timeout_sec: float,
    idle_timeout_sec: float,
    check: bool,
    cwd: Path | None,
    env: Mapping[str, str] | None,
    output_cap_bytes: int,
    terminate_grace_sec: float,
    error_code: str | None,
    on_stdout_line: Callable[[str], None] | None,
    on_stderr_line: Callable[[str], None] | None,
) -> CommandResult:
    command = list(argv)
    if not command:
        raise ValueError("argv must name a tool to run")
    total = _validate_timeout("total_timeout_sec", total_timeout_sec)
    idle = _validate_timeout("idle_timeout_sec", idle_timeout_sec)
    grace = _validate_timeout("terminate_grace_sec", terminate_grace_sec)
    if output_cap_bytes <= 0:
        raise ValueError(f"output_cap_bytes must be positive, got {output_cap_bytes!r}")

    started = time.monotonic()
    sequence = next(_invocation_sequence)
    try:
        # An argument list, always. There is no code path here that hands a
        # command line to a shell, so no argument can become syntax.
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            bufsize=0,
            close_fds=True,
            start_new_session=True,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError, IsADirectoryError) as exc:
        duration_sec = time.monotonic() - started
        _boundary_log(
            outcome=FailureKind.MISSING_TOOL.value,
            stage=stage,
            argv=command,
            sequence=sequence,
            exit_status=None,
            timeout_fired=None,
            duration_sec=duration_sec,
            stdout_truncated=False,
            stderr_truncated=False,
        )
        raise _failure(
            FailureKind.MISSING_TOOL,
            command,
            stage=stage,
            details=_failure_details(
                command,
                exit_status=None,
                timeout_fired=None,
                stdout_truncated=False,
                stderr_truncated=False,
                duration_sec=duration_sec,
            ),
        ) from exc

    clock = _ActivityClock()
    sinks = {name: _CappedSink(output_cap_bytes) for name in (_STDOUT, _STDERR)}
    deliver = on_stdout_line is not None or on_stderr_line is not None
    lines: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
    readers = [
        _StreamReader(
            _STDOUT,
            proc.stdout,
            sink=sinks[_STDOUT],
            clock=clock,
            lines=lines if deliver else None,
        ),
        _StreamReader(
            _STDERR,
            proc.stderr,
            sink=sinks[_STDERR],
            clock=clock,
            lines=lines if deliver else None,
        ),
    ]
    for reader in readers:
        reader.start()

    total_deadline = started + total
    timeout_fired: str | None = None
    try:
        while True:
            if proc.poll() is not None:
                break
            now = time.monotonic()
            if now >= total_deadline:
                timeout_fired = "total"
                break
            idle_deadline = clock.last + idle
            if now >= idle_deadline:
                timeout_fired = "idle"
                break
            wait_for = min(total_deadline, idle_deadline, now + _POLL_INTERVAL_SEC) - now
            _dispatch(lines, max(wait_for, 0.0), on_stdout_line, on_stderr_line)
    except BaseException:
        # A callback raised, or the caller was interrupted: the child is still
        # ours to clean up. This path is reachable in production - a progress
        # reporter whose counter has been stopped raises - so it emits the
        # invocation's boundary event too, rather than being the one way a tool
        # can run and leave no trace.
        exit_status = _terminate_group(proc, grace)
        _boundary_log(
            outcome="callback_error",
            stage=stage,
            argv=command,
            sequence=sequence,
            exit_status=exit_status,
            timeout_fired=None,
            duration_sec=time.monotonic() - started,
            stdout_truncated=sinks[_STDOUT].truncated,
            stderr_truncated=sinks[_STDERR].truncated,
        )
        raise

    exit_status = (
        _terminate_group(proc, grace) if timeout_fired is not None else proc.returncode
    )

    deadline = time.monotonic() + _DRAIN_GRACE_SEC
    for reader in readers:
        reader.join(timeout=max(deadline - time.monotonic(), 0.0))
    _drain_queue(lines, on_stdout_line, on_stderr_line)
    # Only close the pipes when no reader is still blocked on one: closing
    # underneath a reader would free an fd number another thread could reopen.
    if all(not reader.is_alive() for reader in readers):
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError):  # already closed
                    pipe.close()

    stdout_sink, stderr_sink = sinks[_STDOUT], sinks[_STDERR]
    duration_sec = time.monotonic() - started
    warnings = tuple(
        warning(
            stage,
            TRUNCATION_WARNING_CODE,
            f"{_tool_of(command)} {name} exceeded {output_cap_bytes} bytes "
            "and was truncated",
        )
        for name, sink in ((_STDOUT, stdout_sink), (_STDERR, stderr_sink))
        if sink.truncated
    )
    details = _failure_details(
        command,
        exit_status=exit_status,
        timeout_fired=timeout_fired,
        stdout_truncated=stdout_sink.truncated,
        stderr_truncated=stderr_sink.truncated,
        stderr=stderr_sink.text(),
        duration_sec=duration_sec,
    )

    _boundary_log(
        outcome=_outcome_of(timeout_fired, exit_status, check=check),
        stage=stage,
        argv=command,
        sequence=sequence,
        exit_status=exit_status,
        timeout_fired=timeout_fired,
        duration_sec=duration_sec,
        stdout_truncated=stdout_sink.truncated,
        stderr_truncated=stderr_sink.truncated,
    )

    if timeout_fired is not None:
        raise _failure(
            _TIMEOUT_KINDS[timeout_fired],
            command,
            stage=stage,
            details=details,
            error_code=error_code,
            idle_timeout_sec=idle,
        )
    if check and exit_status != 0:
        raise _failure(
            FailureKind.EXIT_STATUS,
            command,
            stage=stage,
            details=details,
            error_code=error_code,
        )

    return CommandResult(
        tool=_tool_of(command),
        argv=tuple(command),
        returncode=exit_status if exit_status is not None else -1,
        stdout=stdout_sink.text(),
        stderr=stderr_sink.text(),
        stdout_truncated=stdout_sink.truncated,
        stderr_truncated=stderr_sink.truncated,
        duration_sec=duration_sec,
        warnings=warnings,
    )
