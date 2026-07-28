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
one. For an absent tool that answer is stated once, in `capabilities.py`: a call
site that catches `E_MISSING_TOOL` hands the tool to that table, which returns a
**warning** for an **optional capability** and raises for a **required** one
(ADR-0002). For every other failure the call site decides, and does so knowing
its tool is installed.

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
the direct child leaves those helpers running, holding the pipes and the
network connection - finding 8. The child is started in its own session, and
*every* path out of an invocation signals the group: SIGTERM, a grace period,
then SIGKILL, whether the child hit a deadline, a callback raised, or the child
exited on its own. The last of those is not an afterthought - yt-dlp exits 0
the moment it hands the merge to ffmpeg, so terminating only on a timeout would
report success while the helper kept the socket and a half-written file.
Signals are sent while the direct child is still unreaped, so the group id
cannot have been recycled onto an unrelated process.

The contract that follows, stated so a later reader does not "fix" it back:
Distill terminates the whole process group of every tool it runs, so a tool
that deliberately spawns a process meant to outlive it is not supported here.
For this tool set - ffmpeg, ffprobe, tesseract, yt-dlp - a helper that outlives
its parent is a leak rather than a feature, and a leaked one holds a socket, a
descriptor and a partial file that nothing will ever clean up. Sparing
survivors would be safe only for a tool none of these are.

Why callbacks get their own thread (R-30). A caller's progress callback is
arbitrary code that can block - a progress emitter writing to a full parent
pipe blocks indefinitely. Delivering lines on the supervision thread would put
that block between the deadline checks, so neither limit would be enforced
while it lasted: the one failure this module exists to make impossible. The
supervisor watches the clock, `_Dispatcher` watches the queue, and neither
waits on the other.

Why a boundary event (R-29 observability). Because this is the only path an
external tool is invoked on, one event emitted here covers every invocation in
the pipeline without a call site remembering to log. Every invocation emits
exactly one, at the end, whether it succeeded, failed, or never started; see
`_boundary_log`.

Known limits. A process that escapes the group - one that starts a session of
its own - can inherit the pipes and hold them open after the group is gone, so
draining is bounded rather than waiting for it. Past that bound the readers are
stopped and the pipes closed regardless, so the tail is dropped instead of
hanging the run or leaking a thread and two descriptors per invocation; the
drop is recorded as truncation (R-33) rather than lost in silence. Line
delivery is bounded the same way and for the same reason, except that what a
blocked callback never received is still in the captured output. Process
control here is POSIX-only.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import os
import queue
import select
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import DistillError, WarningRecord, warning

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
# Slack for a stopped reader to notice and exit. It checks between polls, so
# this is headroom on a loaded machine rather than a wait anyone pays for.
_READER_STOP_GRACE_SEC = 0.5
_POLL_INTERVAL_SEC = 0.01
_READ_CHUNK_BYTES = 64 * 1024
# A "line" this long without a terminator is delivered as-is, so a tool that
# never emits a newline cannot grow the pending buffer without bound.
_MAX_LINE_BYTES = 64 * 1024
# How long delivery gets to hand over what is already queued once the child is
# gone. Bounded because a callback can block; what it never received is still in
# the captured output.
_DISPATCH_FINISH_GRACE_SEC = 1.0
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


def silent_tool_timeouts(total_sec: float) -> CommandTimeouts:
    """The pair for a tool that produces no output at all until it is finished.

    An idle timeout detects a stall by watching output stop. A tool that never
    speaks while it works - `tesseract`, `ffprobe -v error` - never resets the
    idle clock, so a shorter idle value detects nothing: it silently *becomes*
    the deadline and quietly replaces the total the call site declared. Such a
    call site therefore states one number, and this constructor is what puts it
    in both places, so the limit that governs is the limit written down.

    The pair is still structurally complete under R-30 - both limits are set,
    and neither means "no limit". What is removed is the ability to tighten one
    of them in isolation for a tool where doing so cannot help.
    """
    return CommandTimeouts(total_sec=total_sec, idle_sec=total_sec)


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
    warnings: tuple[WarningRecord, ...] = ()

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

    Callbacks run on one delivery thread of their own - not on the reader
    threads, which must keep draining, and not on the supervision thread, which
    must keep checking the deadlines. They are serialized there, so a caller's
    progress emitter needs no locking; a callback that blocks costs delivery
    only, and neither stalls capture nor suspends a limit. A callback that
    raises ends the invocation and is re-raised here, on the calling thread.

    Lines are delivered without their terminator; both `\\n` and `\\r` end a
    line, because progress-writing tools use either, and a `\\r` ends one the
    moment it arrives rather than at EOF. Empty lines are not delivered.

    Delivery is bounded by the same cap as capture: once a stream passes
    `output_cap_bytes` its lines stop being handed over, because they are no
    longer being kept either. The drop is the truncation already recorded as a
    **warning**, and it is what bounds the queue of lines waiting for the
    caller - at most one cap's worth per stream. Delivery is also bounded in
    time once the child is gone: what a slow or blocked callback has not taken
    by then is not handed over, and is still in `CommandResult.stdout` /
    `.stderr`.
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
) -> tuple[Any, tuple[dict[str, str], ...]]:
    """Run `argv`, parse its stdout as JSON, and return it with any **warnings**.

    The warnings are `CommandResult.warnings` - truncated capture (R-33). They
    are returned rather than dropped because a caller of `run_json` has no other
    way to reach them, and a truncated document that still parsed is exactly the
    silent loss R-33 exists to prevent.

    A caller's `error_code` covers the whole invocation, this failure included:
    one tool reports under one code, so a caller passing `E_YTDLP` does not get
    `E_COMMAND` for the single case where the tool ran and answered badly.
    """
    result = run(
        argv,
        stage=stage,
        total_timeout_sec=total_timeout_sec,
        idle_timeout_sec=idle_timeout_sec,
        **options,
    )
    try:
        return json.loads(result.stdout), result.warnings
    except json.JSONDecodeError as exc:
        raise _failure(
            FailureKind.BAD_JSON,
            argv,
            stage=stage,
            error_code=options.get("error_code"),
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
    """Accumulates bytes up to a cap, counting the rest as truncated.

    Two ways a stream loses bytes, kept apart because they need different
    **warning** messages: the cap was reached, or the reader was stopped before
    the pipe reached EOF because something outlived the child holding it open.
    `truncated` is the caller-facing answer to "is this the whole output".
    """

    def __init__(self, cap_bytes: int) -> None:
        self._cap = cap_bytes
        self._parts: list[bytes] = []
        self._size = 0
        self.over_cap = False
        self.tail_dropped = False

    @property
    def truncated(self) -> bool:
        return self.over_cap or self.tail_dropped

    def add(self, chunk: bytes) -> None:
        room = self._cap - self._size
        if room > 0:
            kept = chunk[:room]
            self._parts.append(kept)
            self._size += len(kept)
        if len(chunk) > room:
            self.over_cap = True

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
    """Drains one pipe. One per stream, so neither can starve the other.

    Reads are preceded by a bounded `poll`, so the thread never parks in a read
    that only a grandchild's exit could end: `stop` is noticed within one poll
    interval, which is what makes shutdown bounded rather than a leaked thread
    and descriptor per invocation. `poll` rather than `select`, because
    `select` cannot watch a descriptor number past `FD_SETSIZE`.
    """

    def __init__(
        self,
        stream: str,
        source: Any,
        *,
        sink: _CappedSink,
        clock: _ActivityClock,
        lines: queue.SimpleQueue[tuple[str, str]] | None,
    ) -> None:
        super().__init__(name=f"run_command-{stream}", daemon=True)
        # Not `_name`: that is `threading.Thread`'s own backing field, and
        # writing it would rename the thread out of a stack dump.
        self._stream = stream
        self._source = source
        self._sink = sink
        self._clock = clock
        self._lines = lines
        self._pending = b""
        self._stopped = threading.Event()
        self.reached_eof = False

    def stop(self) -> None:
        """Stop reading, recording anything left unread as truncation.

        Called once draining is out of time. A grandchild can hold the write
        end open long after the child is gone, so the bytes this reader never
        got are a loss the caller is told about (R-33) rather than a silent
        gap.
        """
        self._stopped.set()
        if not self.reached_eof:
            self._sink.tail_dropped = True

    def run(self) -> None:
        try:
            poller = select.poll()
            poller.register(self._source, select.POLLIN)
            while not self._stopped.is_set():
                if not poller.poll(_POLL_INTERVAL_SEC * 1000):
                    continue
                chunk = self._source.read(_READ_CHUNK_BYTES)
                if not chunk:
                    self.reached_eof = True
                    break
                self._clock.mark()
                self._sink.add(chunk)
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
        """Cut the buffer into lines on either terminator.

        Both `\\n` and `\\r` end a line, and a lone `\\r` ends one as soon as it
        arrives rather than waiting for a newline that a progress writer never
        sends: ffmpeg's stats are CR-delimited, and holding them in the pending
        buffer would deliver a whole download as one blob at EOF.
        """
        if self._lines is None:
            return
        parts = (self._pending + chunk).replace(b"\r", b"\n").split(b"\n")
        self._pending = parts.pop()
        for part in parts:
            self._emit(part)
        if len(self._pending) >= _MAX_LINE_BYTES:
            self._emit(self._pending)
            self._pending = b""

    def _emit(self, raw: bytes) -> None:
        # Past the capture cap the line was not kept, so it is not delivered
        # either. The cap is checked before the split, so every queued line came
        # out of bytes that fit under it: the queue holds at most one cap's
        # worth of output per stream, which is what bounds it.
        if not raw or self._lines is None or self._sink.over_cap:
            return
        self._lines.put((self._stream, raw.decode("utf-8", errors="replace")))


_EOF = "__eof__"


# --- the core ----------------------------------------------------------------


def _validate_timeout(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number of seconds, got {value!r}")
    return number


def _group_id(proc: subprocess.Popen[bytes]) -> int | None:
    """The group to signal, or `None` if signalling one would be wrong.

    `start_new_session` makes the child the leader of a new group, so the group
    id *is* its pid; `getpgid` is asked first only to confirm that, and its
    answer is preferred when it has one. It does not always have one: macOS
    refuses `getpgid` for a process that has exited but not been reaped, which
    is exactly the state the child is in on the path where it finished by
    itself. Falling back to the pid there is what keeps that path able to reach
    the survivors - and the pid is still this child's, because nothing reaps it
    before the group has been signalled.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:  # exited and unreaped, or never ours
        pgid = proc.pid
    if pgid == os.getpgrp():
        # start_new_session did not take effect; signalling this group would
        # signal Distill itself.
        return None
    return pgid


def _signal_group(
    proc: subprocess.Popen[bytes], pgid: int | None, signal_number: int
) -> None:
    """Signal the whole group, falling back to the child when there is no group."""
    try:
        if pgid is not None:
            os.killpg(pgid, signal_number)
        else:
            proc.send_signal(signal_number)
    except (ProcessLookupError, PermissionError):
        pass


def _has_exited(proc: subprocess.Popen[bytes]) -> bool:
    """Whether the child has exited, without reaping it.

    `Popen.poll` would reap, and reaping releases the pid that is also the group
    id: the kernel holds that id only while the leader is unreaped or the group
    still has members, so a reaped child leaves its group id free to be recycled
    onto an unrelated process before the group is signalled. Every path out of
    an invocation signals the group, so no path may notice the exit by reaping.
    """
    try:
        return (
            os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            is not None
        )
    except OSError:  # already reaped, or never ours
        return True


def _await_exit_without_reaping(proc: subprocess.Popen[bytes], grace_sec: float) -> None:
    """Give the child `grace_sec` to exit, leaving it unreaped either way.

    This is the SIGTERM grace period. It is spent without reaping for the reason
    `_has_exited` states: the SIGKILL that follows must still name a group id
    the kernel is holding for this child.
    """
    deadline = time.monotonic() + grace_sec
    while not _has_exited(proc):
        if time.monotonic() >= deadline:
            return
        time.sleep(_POLL_INTERVAL_SEC)


def _terminate_group(proc: subprocess.Popen[bytes], grace_sec: float) -> int | None:
    """SIGTERM the child's group, then SIGKILL the group whatever the child did.

    Called on every path out of an invocation, not only after a timeout, and the
    forced kill is not an escalation reserved for a child that ignored SIGTERM.
    yt-dlp exits cleanly - on SIGTERM, or on its own the moment the download is
    handed to the ffmpeg it spawned for the merge - while that ffmpeg, holding
    the network and the disk, does not. Returning as soon as the direct child is
    gone therefore orphans exactly the process that matters (R-31, finding 8),
    whether the child was killed or finished by itself. The group SIGKILL always
    runs, and reaping the leader is the last thing that happens.

    Both signals are sent while the direct child is still unreaped, so the group
    id cannot have been recycled onto an unrelated process.
    """
    pgid = _group_id(proc)
    _signal_group(proc, pgid, signal.SIGTERM)
    _await_exit_without_reaping(proc, grace_sec)
    _signal_group(proc, pgid, signal.SIGKILL)
    try:
        return proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
        return proc.returncode


class _Dispatcher(threading.Thread):
    """Hands queued lines to the caller's callbacks, and owns nothing else.

    Delivery has a thread of its own so that supervision never waits on the
    caller: a callback that blocks - a progress emitter writing to a full parent
    pipe - would otherwise sit between two deadline checks and suspend both
    limits for as long as it lasted (R-30). It does not decide when the
    invocation ends, does not touch the child, and does not drain the pipes;
    the readers do that, so a blocked callback cannot stall capture either.

    Callbacks are serialized here, so a caller's progress emitter needs no
    locking of its own. A callback that raises stops delivery and leaves the
    exception in `failure` for the supervision thread to re-raise, which is
    where the caller can catch it.
    """

    def __init__(
        self,
        lines: queue.SimpleQueue[tuple[str, str]],
        on_stdout_line: Callable[[str], None] | None,
        on_stderr_line: Callable[[str], None] | None,
    ) -> None:
        super().__init__(name="run_command-dispatch", daemon=True)
        self._lines = lines
        self._on_stdout_line = on_stdout_line
        self._on_stderr_line = on_stderr_line
        self._finishing = threading.Event()
        self._stopped = threading.Event()
        self.failure: BaseException | None = None

    def finish(self) -> None:
        """Deliver what is already queued, then end. Called once the readers are done."""
        self._finishing.set()

    def stop(self) -> None:
        """Deliver nothing further, whatever is left in the queue.

        A blocked callback cannot be interrupted, so this is what guarantees no
        line reaches the caller after the invocation has returned to it.
        """
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.is_set():
            try:
                name, text = self._lines.get(timeout=_POLL_INTERVAL_SEC)
            except queue.Empty:
                if self._finishing.is_set():
                    return
                continue
            try:
                self._deliver(name, text)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                self.failure = exc
                return

    def _deliver(self, name: str, text: str) -> None:
        if name == _STDOUT and self._on_stdout_line is not None:
            self._on_stdout_line(text)
        elif name == _STDERR and self._on_stderr_line is not None:
            self._on_stderr_line(text)


def _shutdown_dispatch(dispatcher: _Dispatcher | None) -> None:
    """End delivery, giving queued lines a bounded window to be handed over.

    Bounded for the same reason draining is: a callback that blocks cannot be
    interrupted, and waiting on one would hang the return that the deadlines
    just enforced. What a blocked callback never received is still in the
    captured output, so the loss is delivery, not data.
    """
    if dispatcher is None:
        return
    dispatcher.finish()
    dispatcher.join(timeout=_DISPATCH_FINISH_GRACE_SEC)
    dispatcher.stop()


def _shutdown_readers(readers: Sequence[_StreamReader], proc: subprocess.Popen[bytes]) -> None:
    """End draining and release what the invocation held, on every exit path.

    Bounded on purpose: a grandchild that inherited the pipes can hold them open
    long after the child is gone, so the readers get a grace period, are then
    stopped, and their pipes are closed whatever they were doing. Abandoning a
    blocked reader instead leaks a thread and two descriptors per invocation,
    and drops its tail in silence; `stop` records that tail as truncation
    (R-33).
    """
    deadline = time.monotonic() + _DRAIN_GRACE_SEC
    for reader in readers:
        reader.join(timeout=max(deadline - time.monotonic(), 0.0))
    for reader in readers:
        reader.stop()
    for reader in readers:
        reader.join(timeout=_READER_STOP_GRACE_SEC)
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError):  # already closed
                pipe.close()


def _truncation_messages(
    tool: str, stream: str, sink: _CappedSink, cap_bytes: int
) -> list[str]:
    """Every way this stream lost bytes, one **warning** message each (R-33).

    A tail left unread because a helper held the pipe open past the drain
    deadline is the same loss to the caller as output past the cap, and is
    reported the same way: R-33 makes truncation a warning, not a silent gap.
    """
    messages = []
    if sink.over_cap:
        messages.append(f"{tool} {stream} exceeded {cap_bytes} bytes and was truncated")
    if sink.tail_dropped:
        messages.append(
            f"{tool} {stream} was still held open when draining ended, "
            "so its remaining output was truncated"
        )
    return messages


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
    dispatcher = _Dispatcher(lines, on_stdout_line, on_stderr_line) if deliver else None
    if dispatcher is not None:
        dispatcher.start()

    total_deadline = started + total
    timeout_fired: str | None = None
    try:
        while True:
            # Never `proc.poll()`: noticing the exit must not reap it, or the
            # group id below could already have been recycled.
            if _has_exited(proc):
                break
            if dispatcher is not None and dispatcher.failure is not None:
                # Ends the invocation now rather than at the deadline, which is
                # what the old inline delivery did by propagating.
                raise dispatcher.failure
            now = time.monotonic()
            if now >= total_deadline:
                timeout_fired = "total"
                break
            idle_deadline = clock.last + idle
            if now >= idle_deadline:
                timeout_fired = "idle"
                break
            wait_for = min(total_deadline, idle_deadline, now + _POLL_INTERVAL_SEC) - now
            time.sleep(max(wait_for, 0.0))
    except BaseException:
        # A callback raised, or the caller was interrupted: the child is still
        # ours to clean up. This path is reachable in production - a progress
        # reporter whose counter has been stopped raises - so it emits the
        # invocation's boundary event too, rather than being the one way a tool
        # can run and leave no trace.
        exit_status = _terminate_group(proc, grace)
        _shutdown_readers(readers, proc)
        _shutdown_dispatch(dispatcher)
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

    # Unconditional: a child that exited on its own can still have left the rest
    # of its group running (R-31), and its exit status survives the group kill
    # because the kernel is holding it for the reap `_terminate_group` ends with.
    exit_status = _terminate_group(proc, grace)

    _shutdown_readers(readers, proc)
    _shutdown_dispatch(dispatcher)

    stdout_sink, stderr_sink = sinks[_STDOUT], sinks[_STDERR]
    duration_sec = time.monotonic() - started
    warnings = tuple(
        warning(stage, TRUNCATION_WARNING_CODE, message)
        for name, sink in ((_STDOUT, stdout_sink), (_STDERR, stderr_sink))
        for message in _truncation_messages(_tool_of(command), name, sink, output_cap_bytes)
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

    # A callback can raise on the last line it is handed, after the supervision
    # loop has already ended. That is still the caller's exception to see, and
    # the invocation still reports how it ended.
    callback_failure = dispatcher.failure if dispatcher is not None else None

    _boundary_log(
        outcome=(
            "callback_error"
            if callback_failure is not None
            else _outcome_of(timeout_fired, exit_status, check=check)
        ),
        stage=stage,
        argv=command,
        sequence=sequence,
        exit_status=exit_status,
        timeout_fired=timeout_fired,
        duration_sec=duration_sec,
        stdout_truncated=stdout_sink.truncated,
        stderr_truncated=stderr_sink.truncated,
    )

    if callback_failure is not None:
        raise callback_failure
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
