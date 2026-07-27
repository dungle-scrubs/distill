"""Tests for `run_command`, the single subprocess path.

Every test drives a real child process, because the defects this module exists
to close - the pipe-buffer deadlock (finding 3) and the orphaned grandchild
(finding 8) - only exist between real pipes and real process groups; a fake
`Popen` cannot reproduce either.

Timing discipline: no test asserts on elapsed time. Timeouts are used as
budgets that a correct implementation never reaches, and the assertions are on
observable state - process gone, which timeout fired, what was captured - so a
loaded machine makes a test slower rather than red.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from distill import run_command
from distill.errors import DistillError
from distill.run_command import (
    DEFAULT_IDLE_TIMEOUT_SEC,
    ERROR_CODES,
    OUTPUT_CAP_BYTES,
    TRUNCATION_WARNING_CODE,
    CommandTimeouts,
    FailureKind,
    run,
    run_json,
    silent_tool_timeouts,
    stream,
)

PYTHON = sys.executable
# A budget a correct implementation never reaches, used wherever the test is
# about something other than a timeout.
GENEROUS_TOTAL_SEC = 30.0
GENEROUS_IDLE_SEC = 20.0


def child(script: str, *args: str) -> list[str]:
    """An argv running `script` in a fresh interpreter - hermetic and portable."""
    return [PYTHON, "-c", script, *args]


def wait_until_gone(pid: int, timeout_sec: float = 15.0) -> bool:
    """Poll until `pid` no longer exists. Bounded, so a live process fails."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:  # pragma: no cover - pid reused by another user
            return False
        time.sleep(0.02)
    return False


def wait_for(condition: Callable[[], bool], timeout_sec: float = 5.0) -> bool:
    """Poll until `condition` holds. Bounded, and asserts on state, not on time."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def open_descriptors() -> int:
    """How many descriptors this process holds. POSIX-only, like the module."""
    return len(os.listdir("/dev/fd"))


def kill_all(pid_file: Path) -> None:
    """Kill every pid a child recorded, so a failing test leaves nothing behind."""
    for value in pid_file.read_text().split():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(value), signal.SIGKILL)


# --- concurrent draining (finding 3) ----------------------------------------

BIG_STDOUT_BETWEEN_STDERR_LINES = """
import sys
sys.stderr.write("start\\n")
sys.stderr.flush()
sys.stdout.write("x" * 256 * 1024)
sys.stdout.flush()
sys.stderr.write("done\\n")
sys.stderr.flush()
"""


def test_large_stdout_does_not_deadlock_while_stderr_is_read() -> None:
    """FAILS FIRST (finding 3): the pipe-buffer deadlock.

    The child writes 256 KiB to stdout between two stderr lines. A reader that
    consumes stderr to EOF before touching stdout blocks forever: the child is
    blocked writing stdout past the ~64 KiB pipe buffer, so the second stderr
    line never arrives. This is exactly the shape of the yt-dlp downloader.
    """
    result = run(
        child(BIG_STDOUT_BETWEEN_STDERR_LINES),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert result.returncode == 0
    assert len(result.stdout) == 256 * 1024
    assert "start" in result.stderr
    assert "done" in result.stderr


BOTH_STREAMS_OVERFLOW = """
import sys
chunk = "y" * 8192
for _ in range(32):
    sys.stdout.write(chunk)
    sys.stderr.write(chunk)
sys.stdout.flush()
sys.stderr.flush()
"""


def test_both_pipes_are_drained_concurrently() -> None:
    """Both streams overflow the pipe buffer; neither may starve the other."""
    result = run(
        child(BOTH_STREAMS_OVERFLOW),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert result.returncode == 0
    assert len(result.stdout) == 256 * 1024
    assert len(result.stderr) == 256 * 1024


# --- argument list only ------------------------------------------------------


def test_no_shell_invocation_path_exists_anywhere_in_the_package() -> None:
    """No module may hand a command line to a shell, here or elsewhere."""
    package_dir = Path(run_command.__file__).resolve().parent
    offenders = sorted(
        path.name
        for path in package_dir.rglob("*.py")
        if "shell=True" in path.read_text()
    )

    assert offenders == []
    for entry_point in (run, stream, run_json):
        assert "shell" not in inspect.signature(entry_point).parameters


def test_arguments_reach_the_child_literally(tmp_path: Path) -> None:
    """Shell metacharacters are data, not syntax."""
    marker = tmp_path / "written-by-a-shell"
    injected = f"; touch {marker}"

    result = run(
        child("import sys; sys.stdout.write(sys.argv[1])", injected),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert result.stdout == injected
    assert not marker.exists()


# --- deadlines ---------------------------------------------------------------

CHATTY_FOREVER = """
import sys, time
while True:
    sys.stdout.write("tick\\n")
    sys.stdout.flush()
    time.sleep(0.01)
"""


def test_total_deadline_fires_on_a_child_that_never_finishes() -> None:
    """A child that keeps producing output is still bounded by total time."""
    with pytest.raises(DistillError) as failure:
        run(
            child(CHATTY_FOREVER),
            stage="source",
            total_timeout_sec=0.3,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
            terminate_grace_sec=0.2,
        )

    assert failure.value.code == "E_COMMAND"
    assert failure.value.details["timeout_fired"] == "total"


def stalling_child_script(pid_file: Path) -> str:
    return f"""
import os, sys, time
open({str(pid_file)!r}, "w").write(str(os.getpid()))
sys.stdout.write("hello\\n")
sys.stdout.flush()
time.sleep(600)
"""


def test_idle_timeout_fires_separately_from_the_total_deadline(tmp_path: Path) -> None:
    """The child stalls long before the total deadline; idle time is what ends it."""
    with pytest.raises(DistillError) as failure:
        run(
            child(stalling_child_script(tmp_path / "pid")),
            stage="source",
            total_timeout_sec=5.0,
            idle_timeout_sec=0.2,
            terminate_grace_sec=0.2,
        )

    assert failure.value.code == "E_COMMAND"
    assert failure.value.details["timeout_fired"] == "idle"


def test_a_stalled_child_is_killed_by_the_idle_timeout(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"

    with pytest.raises(DistillError) as failure:
        run(
            child(stalling_child_script(pid_file)),
            stage="source",
            total_timeout_sec=5.0,
            idle_timeout_sec=0.2,
            terminate_grace_sec=0.2,
        )

    # Which timeout ended it matters: the total deadline would also have killed
    # this child eventually, and that is not what is under test.
    assert failure.value.details["timeout_fired"] == "idle"
    assert wait_until_gone(int(pid_file.read_text()))


SLOW_BUT_PROGRESSING = """
import sys, time
for index in range(10):
    sys.stdout.write("line %d\\n" % index)
    sys.stdout.flush()
    time.sleep(0.05)
"""


def test_a_slow_but_progressing_child_is_not_killed_by_the_idle_timeout() -> None:
    """Output resets the idle clock, so a long quiet-free download survives."""
    result = run(
        child(SLOW_BUT_PROGRESSING),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=1.0,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [f"line {index}" for index in range(10)]


def test_both_timeouts_are_structurally_required() -> None:
    """There is no default that silently means "no limit" (R-30)."""
    for entry_point in (run, stream, run_json):
        parameters = inspect.signature(entry_point).parameters
        assert parameters["total_timeout_sec"].default is inspect.Parameter.empty
        assert parameters["idle_timeout_sec"].default == DEFAULT_IDLE_TIMEOUT_SEC

    assert DEFAULT_IDLE_TIMEOUT_SEC == 120.0


@pytest.mark.parametrize(
    ("total", "idle"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("inf"), 1.0),
        (float("nan"), 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, float("inf")),
        (1.0, float("nan")),
    ],
)
def test_a_timeout_that_is_not_a_positive_finite_number_is_rejected(
    total: float, idle: float
) -> None:
    with pytest.raises(ValueError):
        run(
            child("pass"),
            stage="source",
            total_timeout_sec=total,
            idle_timeout_sec=idle,
        )


# --- process group and termination (finding 8) -------------------------------


def test_the_child_runs_in_its_own_process_group() -> None:
    result = run(
        child("import os, sys; sys.stdout.write(str(os.getpgrp()))"),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert int(result.stdout) != os.getpgrp()


def sigterm_ignoring_parent_script(pid_file: Path) -> str:
    """A child that ignores SIGTERM and leaves a grandchild behind.

    The grandchild's own output goes to /dev/null: this test is about what is
    still *running* after the deadline, and inheriting the pipes would conflate
    that with whether the helper can drain around a pipe-holding grandchild
    (covered separately).
    """
    return f"""
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen(
    [{PYTHON!r}, "-c", "import time; time.sleep(600)"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
open({str(pid_file)!r}, "w").write("%d %d" % (os.getpid(), grandchild.pid))
sys.stdout.write("spawned\\n")
sys.stdout.flush()
time.sleep(600)
"""


def test_a_sigterm_ignoring_child_leaves_no_grandchild_after_the_deadline(
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 8): killing the child alone orphans the grandchild.

    The child ignores SIGTERM, so only a forced kill of the whole process group
    reaches the grandchild it spawned.
    """
    pid_file = tmp_path / "pids"

    with pytest.raises(DistillError) as failure:
        run(
            child(sigterm_ignoring_parent_script(pid_file)),
            stage="youtube",
            total_timeout_sec=1.0,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
            terminate_grace_sec=0.2,
        )

    child_pid, grandchild_pid = (int(value) for value in pid_file.read_text().split())
    assert wait_until_gone(child_pid), "the child outlived its deadline"
    assert wait_until_gone(grandchild_pid), "the grandchild was orphaned"
    assert failure.value.details["exit_status"] == -signal.SIGKILL


def sigterm_obeying_parent_script(pid_file: Path) -> str:
    """A child that dies on SIGTERM but leaves behind a helper that does not.

    The yt-dlp shape: the tool exits cleanly on the first signal while the
    ffmpeg it spawned for the merge - holding the network and the disk - ignores
    it. The helper's output goes to /dev/null for the same reason as above.
    """
    return f"""
import os, signal, subprocess, sys, time
grandchild = subprocess.Popen(
    [
        {PYTHON!r},
        "-c",
        "import signal, time;"
        " signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
open({str(pid_file)!r}, "w").write("%d %d" % (os.getpid(), grandchild.pid))
sys.stdout.write("spawned\\n")
sys.stdout.flush()
time.sleep(600)
"""


def test_a_grandchild_that_ignores_sigterm_dies_even_when_the_child_obeys_it(
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2): the child honouring SIGTERM ended termination early.

    Reaping the direct child says nothing about the rest of its group. The
    inverse case - a child that ignores SIGTERM - is covered above and does not
    catch this one, because there the escalation to SIGKILL still ran.
    """
    pid_file = tmp_path / "pids"

    with pytest.raises(DistillError) as failure:
        run(
            child(sigterm_obeying_parent_script(pid_file)),
            stage="youtube",
            total_timeout_sec=1.0,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
            terminate_grace_sec=0.2,
        )

    child_pid, grandchild_pid = (int(value) for value in pid_file.read_text().split())
    try:
        # The child died on SIGTERM: this is the path where the old escalation
        # to a group SIGKILL never ran.
        assert failure.value.details["exit_status"] == -signal.SIGTERM
        assert wait_until_gone(child_pid), "the child outlived its deadline"
        assert wait_until_gone(grandchild_pid), "the grandchild was orphaned"
    finally:
        kill_all(pid_file)


def test_the_group_is_asked_to_terminate_before_it_is_forced(tmp_path: Path) -> None:
    """SIGTERM reaches the group first; SIGKILL is the escalation, not the opener."""
    handled = tmp_path / "sigterm-seen"
    script = f"""
import os, signal, sys, time
def _handle(signum, frame):
    open({str(handled)!r}, "w").write("sigterm")
    os._exit(0)
signal.signal(signal.SIGTERM, _handle)
sys.stdout.write("ready\\n")
sys.stdout.flush()
time.sleep(600)
"""

    with pytest.raises(DistillError):
        run(
            child(script),
            stage="source",
            total_timeout_sec=0.3,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

    assert handled.read_text() == "sigterm"


def test_a_grandchild_holding_the_pipes_does_not_hang_the_caller(
    tmp_path: Path,
) -> None:
    """The child exits; a grandchild keeps stdout open. Draining must be bounded."""
    pid_file = tmp_path / "grandchild-pid"
    script = f"""
import os, subprocess, sys, time
grandchild = subprocess.Popen([{PYTHON!r}, "-c", "import time; time.sleep(30)"])
open({str(pid_file)!r}, "w").write(str(grandchild.pid))
sys.stdout.write("bye\\n")
sys.stdout.flush()
"""

    result = run(
        child(script),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert result.returncode == 0
    assert "bye" in result.stdout
    os.kill(int(pid_file.read_text()), signal.SIGKILL)


PIPE_HOLDING_GRANDCHILD = f"""
import subprocess, sys
grandchild = subprocess.Popen([{PYTHON!r}, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "a").write("%d\\n" % grandchild.pid)
sys.stdout.write("bye\\n")
sys.stdout.flush()
"""

_INVOCATIONS = 3


def test_a_pipe_holding_grandchild_leaks_no_thread_or_descriptor(tmp_path: Path) -> None:
    """FAILS FIRST (finding 3): the blocked reader and its pipes were abandoned.

    The grandchild inherits both pipes and outlives the child, so neither reader
    ever reaches EOF. Abandoning them leaked a thread and two descriptors per
    invocation - a pipeline that runs one such tool per stage runs out of both.
    """
    pid_file = tmp_path / "grandchild-pids"
    pid_file.write_text("")
    argv = child(PIPE_HOLDING_GRANDCHILD, str(pid_file))

    try:
        threads_before = threading.active_count()
        descriptors_before = open_descriptors()
        for _ in range(_INVOCATIONS):
            result = run(
                argv,
                stage="source",
                total_timeout_sec=GENEROUS_TOTAL_SEC,
                idle_timeout_sec=GENEROUS_IDLE_SEC,
            )
            assert "bye" in result.stdout

        assert wait_for(lambda: threading.active_count() == threads_before), (
            f"{threading.active_count() - threads_before} reader threads were abandoned"
        )
        assert open_descriptors() <= descriptors_before
    finally:
        kill_all(pid_file)


def test_a_dropped_tail_is_recorded_as_a_warning(tmp_path: Path) -> None:
    """FAILS FIRST (finding 3): what the readers never got was lost in silence.

    R-33 makes truncation a **warning**; a tail dropped because a helper held
    the pipe open past the drain deadline is the same loss to the caller as
    output past the cap, so it is reported the same way.
    """
    pid_file = tmp_path / "grandchild-pids"
    pid_file.write_text("")

    try:
        result = run(
            child(PIPE_HOLDING_GRANDCHILD, str(pid_file)),
            stage="youtube",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

        assert result.stdout_truncated is True
        assert result.stderr_truncated is True
        assert {entry["code"] for entry in result.warnings} == {TRUNCATION_WARNING_CODE}
        assert {entry["stage"] for entry in result.warnings} == {"youtube"}
        messages = " ".join(entry["message"] for entry in result.warnings)
        assert "stdout" in messages
        assert "stderr" in messages
    finally:
        kill_all(pid_file)


# --- output caps -------------------------------------------------------------


def test_the_default_output_cap_is_eight_mebibytes_per_stream() -> None:
    assert OUTPUT_CAP_BYTES == 8 * 1024 * 1024
    for entry_point in (run, stream):
        assert (
            inspect.signature(entry_point).parameters["output_cap_bytes"].default
            == OUTPUT_CAP_BYTES
        )


def test_stdout_is_captured_under_the_default_cap() -> None:
    script = f"import sys; sys.stdout.write('a' * {OUTPUT_CAP_BYTES + 4096})"

    result = run(
        child(script),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    # returncode 0 proves the surplus was still drained rather than left to
    # block the child.
    assert result.returncode == 0
    assert len(result.stdout) == OUTPUT_CAP_BYTES
    assert result.stdout_truncated is True
    assert result.stderr_truncated is False


def test_stderr_is_captured_under_the_default_cap() -> None:
    script = f"import sys; sys.stderr.write('b' * {OUTPUT_CAP_BYTES + 4096})"

    result = run(
        child(script),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert result.returncode == 0
    assert len(result.stderr) == OUTPUT_CAP_BYTES
    assert result.stderr_truncated is True
    assert result.stdout_truncated is False


def test_truncation_is_recorded_as_a_warning() -> None:
    script = "import sys; sys.stdout.write('a' * 4096); sys.stderr.write('b' * 4096)"

    result = run(
        child(script),
        stage="youtube",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        output_cap_bytes=1024,
    )

    assert {entry["code"] for entry in result.warnings} == {TRUNCATION_WARNING_CODE}
    assert {entry["stage"] for entry in result.warnings} == {"youtube"}
    messages = " ".join(entry["message"] for entry in result.warnings)
    assert "stdout" in messages
    assert "stderr" in messages
    assert len(result.warnings) == 2


EXACT_CAP_BYTES = 1000


@pytest.mark.parametrize(
    ("written", "truncated"),
    [(EXACT_CAP_BYTES - 1, False), (EXACT_CAP_BYTES, False), (EXACT_CAP_BYTES + 1, True)],
)
def test_the_cap_is_the_last_byte_kept_not_the_first_dropped(
    written: int, truncated: bool
) -> None:
    """The boundary itself, so the one arithmetic line R-33 rests on is pinned.

    Output of exactly the cap fits; one byte more does not. Without this, `>`
    and `>=` in the sink are indistinguishable and the cap silently loses a
    byte's worth of every stream.
    """
    result = run(
        child(f"import sys; sys.stdout.write('a' * {written})"),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        output_cap_bytes=EXACT_CAP_BYTES,
    )

    assert result.stdout == "a" * min(written, EXACT_CAP_BYTES)
    assert result.stdout_truncated is truncated
    assert (result.warnings != ()) is truncated


def test_output_within_the_cap_records_no_warning() -> None:
    result = run(
        child("import sys; sys.stdout.write('small')"),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert result.warnings == ()
    assert result.stdout_truncated is False


# --- the error table ---------------------------------------------------------


def test_every_failure_kind_has_an_entry_in_one_error_table() -> None:
    assert set(ERROR_CODES) == set(FailureKind)


def test_a_missing_tool_maps_to_e_missing_tool() -> None:
    with pytest.raises(DistillError) as failure:
        run(
            ["distill-tool-that-does-not-exist"],
            stage="source",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

    assert failure.value.code == ERROR_CODES[FailureKind.MISSING_TOOL] == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "distill-tool-that-does-not-exist"


def test_a_non_zero_exit_maps_to_e_command() -> None:
    with pytest.raises(DistillError) as failure:
        run(
            child("raise SystemExit(3)"),
            stage="source",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

    assert failure.value.code == ERROR_CODES[FailureKind.EXIT_STATUS] == "E_COMMAND"
    assert failure.value.details["exit_status"] == 3


def test_a_non_zero_exit_is_returned_rather_than_raised_when_check_is_false() -> None:
    result = run(
        child("raise SystemExit(3)"),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        check=False,
    )

    assert result.returncode == 3


def test_invalid_json_maps_to_e_command() -> None:
    with pytest.raises(DistillError) as failure:
        run_json(
            child("import sys; sys.stdout.write('not json')"),
            stage="source",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

    assert failure.value.code == ERROR_CODES[FailureKind.BAD_JSON] == "E_COMMAND"


def test_run_json_reports_a_bad_document_under_the_callers_error_code() -> None:
    """FAILS FIRST (finding 8): the caller's code was dropped for this one failure.

    One tool reports under one code. Passing `E_YTDLP` and getting `E_COMMAND`
    only when the tool answered with garbage splits one tool's failures across
    two codes, which is exactly the case an operator greps for.
    """
    with pytest.raises(DistillError) as failure:
        run_json(
            child("import sys; sys.stdout.write('not json')"),
            stage="youtube",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
            error_code="E_YTDLP",
        )

    assert failure.value.code == "E_YTDLP"
    assert failure.value.stage == "youtube"


def test_run_json_parses_the_document_a_tool_prints() -> None:
    payload, warnings = run_json(
        child("import sys; sys.stdout.write('{\"format\": {\"duration\": \"3.5\"}}')"),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
    )

    assert payload["format"]["duration"] == "3.5"
    assert warnings == ()


def test_run_json_hands_back_the_truncation_warning_with_the_document() -> None:
    """R-33: a document that parsed is not proof nothing was lost.

    `run_json` is the only reach a caller has to the invocation, so a truncation
    warning it does not return is a warning that cannot exist anywhere.
    """
    document, warnings = run_json(
        child(
            "import sys; sys.stdout.write('{}'); sys.stderr.write('x' * 4096)",
        ),
        stage="source",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        output_cap_bytes=64,
    )

    assert document == {}
    assert [item["code"] for item in warnings] == [TRUNCATION_WARNING_CODE]
    assert "stderr" in warnings[0]["message"]


def test_a_caller_may_override_the_command_error_code() -> None:
    with pytest.raises(DistillError) as failure:
        run(
            child("raise SystemExit(1)"),
            stage="youtube",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
            error_code="E_YTDLP",
        )

    assert failure.value.code == "E_YTDLP"
    assert failure.value.stage == "youtube"


# --- the failure payload -----------------------------------------------------

FAILURE_PAYLOAD_KEYS = {
    "tool",
    "argv",
    "exit_status",
    "timeout_fired",
    "stdout_truncated",
    "stderr_truncated",
}


def test_a_failed_command_records_a_structured_payload() -> None:
    argv = child("import sys; sys.stderr.write('boom\\n'); raise SystemExit(3)")

    with pytest.raises(DistillError) as failure:
        run(
            argv,
            stage="source",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

    details = failure.value.details
    assert set(details) >= FAILURE_PAYLOAD_KEYS
    assert details["tool"] == PYTHON
    assert details["argv"] == argv
    assert details["exit_status"] == 3
    assert details["timeout_fired"] is None
    assert details["stdout_truncated"] is False
    assert details["stderr_truncated"] is False
    assert "boom" in details["stderr_tail"]


def test_a_timed_out_command_records_which_timeout_fired_and_the_truncation_state(
    tmp_path: Path,
) -> None:
    with pytest.raises(DistillError) as failure:
        run(
            child(stalling_child_script(tmp_path / "pid")),
            stage="source",
            total_timeout_sec=5.0,
            idle_timeout_sec=0.2,
            terminate_grace_sec=0.2,
            output_cap_bytes=2,
        )

    details = failure.value.details
    assert set(details) >= FAILURE_PAYLOAD_KEYS
    assert details["timeout_fired"] == "idle"
    assert details["stdout_truncated"] is True
    assert details["tool"] == PYTHON


# --- stream() ----------------------------------------------------------------


def test_stream_delivers_stdout_lines_before_the_child_exits(tmp_path: Path) -> None:
    """The child only finishes once the caller has seen its first line.

    A helper that hands lines over after the child exits deadlocks here and hits
    its deadline, so this proves live delivery without timing anything.
    """
    acknowledgement = tmp_path / "ack"
    script = f"""
import os, sys, time
sys.stdout.write("go\\n")
sys.stdout.flush()
deadline = time.monotonic() + 60
while not os.path.exists({str(acknowledgement)!r}) and time.monotonic() < deadline:
    time.sleep(0.01)
sys.stdout.write("acked\\n")
sys.stdout.flush()
"""

    def acknowledge(line: str) -> None:
        if line == "go":
            acknowledgement.write_text("ok")

    result = stream(
        child(script),
        stage="youtube",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        on_stdout_line=acknowledge,
    )

    assert result.stdout.splitlines() == ["go", "acked"]


def test_stream_delivers_stderr_lines_and_still_captures_them() -> None:
    seen: list[str] = []

    result = stream(
        child("import sys; sys.stderr.write('one\\ntwo\\n')"),
        stage="youtube",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        on_stderr_line=seen.append,
    )

    assert seen == ["one", "two"]
    assert result.stderr == "one\ntwo\n"


PROGRESS_ON_STDOUT_NOISE_ON_STDERR = """
import sys
noise = "n" * 8192
for index in range(32):
    sys.stdout.write("[download] %d%%\\n" % index)
    sys.stdout.flush()
    sys.stderr.write(noise)
sys.stderr.flush()
"""


def test_stream_consumes_progress_while_the_other_stream_overflows() -> None:
    """The yt-dlp shape: progress on stdout, bulk noise on stderr."""
    progress: list[str] = []

    result = stream(
        child(PROGRESS_ON_STDOUT_NOISE_ON_STDERR),
        stage="youtube",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        on_stdout_line=progress.append,
    )

    assert progress == [f"[download] {index}%" for index in range(32)]
    assert len(result.stderr) == 32 * 8192


def carriage_return_progress_script(acknowledgement: Path) -> str:
    """A tool whose progress is CR-delimited, like ffmpeg without `-progress`.

    Nothing further is written until the first line has been delivered, so a
    helper that holds a CR-terminated line in its pending buffer produces
    `unacked` instead of the rest of the progress.
    """
    return f"""
import os, sys, time
sys.stdout.write("frame=1\\r")
sys.stdout.flush()
deadline = time.monotonic() + 5
while not os.path.exists({str(acknowledgement)!r}) and time.monotonic() < deadline:
    time.sleep(0.01)
if os.path.exists({str(acknowledgement)!r}):
    sys.stdout.write("frame=2\\rdone\\n")
else:
    sys.stdout.write("unacked\\n")
sys.stdout.flush()
"""


def test_stream_delivers_a_carriage_return_line_as_it_arrives(tmp_path: Path) -> None:
    """FAILS FIRST (finding 7): a CR-terminated line sat in the pending buffer.

    `stream` documents `\\r` as a line terminator, and ffmpeg's stats use it
    when `-progress pipe:2` is absent. Splitting on newline first delivers a
    whole run of progress as one blob at EOF, which then crowds the real error
    out of the stderr tail.
    """
    acknowledgement = tmp_path / "ack"
    seen: list[str] = []

    def acknowledge(line: str) -> None:
        seen.append(line)
        if line == "frame=1":
            acknowledgement.write_text("ok")

    result = stream(
        child(carriage_return_progress_script(acknowledgement)),
        stage="youtube",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        on_stdout_line=acknowledge,
    )

    assert seen == ["frame=1", "frame=2", "done"]
    assert result.stdout == "frame=1\rframe=2\rdone\n"


CHATTIER_THAN_THE_CAP = """
import sys
for index in range(20000):
    sys.stdout.write("line %05d\\n" % index)
sys.stdout.flush()
"""


def test_line_delivery_stops_at_the_capture_cap() -> None:
    """FAILS FIRST (finding 11): the cap bounded capture but not the line queue.

    Every line was queued whatever the cap said, so a chatty tool grew the queue
    without bound - the one thing the cap exists to prevent. Delivery now stops
    where capture does, and the drop is the truncation already warned about.
    """
    cap_bytes = 1000
    seen: list[str] = []

    result = stream(
        child(CHATTIER_THAN_THE_CAP),
        stage="youtube",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        output_cap_bytes=cap_bytes,
        on_stdout_line=seen.append,
    )

    assert result.stdout_truncated is True
    assert sum(len(line) + 1 for line in seen) <= cap_bytes
    # What is delivered is still the head of the output, in order.
    assert seen == [f"line {index:05d}" for index in range(len(seen))]


def test_the_reader_threads_are_named_for_the_stream_they_drain(tmp_path: Path) -> None:
    """FAILS FIRST (finding 10): the intended names never reached the threads.

    A hung run is diagnosed from a stack dump, and "stdout" alone does not say
    which subsystem is blocked. The child is still running when the names are
    read, so both readers are certainly alive.
    """
    acknowledgement = tmp_path / "ack"
    script = f"""
import os, sys, time
sys.stdout.write("go\\n")
sys.stdout.flush()
deadline = time.monotonic() + 5
while not os.path.exists({str(acknowledgement)!r}) and time.monotonic() < deadline:
    time.sleep(0.01)
"""
    named: set[str] = set()

    def record(_line: str) -> None:
        named.update(thread.name for thread in threading.enumerate())
        acknowledgement.write_text("ok")

    stream(
        child(script),
        stage="youtube",
        total_timeout_sec=GENEROUS_TOTAL_SEC,
        idle_timeout_sec=GENEROUS_IDLE_SEC,
        on_stdout_line=record,
    )

    assert {"run_command-stdout", "run_command-stderr"} <= named


# --- the boundary event (R-29 observability) ---------------------------------


def boundary_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [json.loads(record.message) for record in caplog.records]


def test_every_invocation_emits_one_boundary_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One event per invocation, carrying what ties it to the run that made it."""
    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        run(
            child("import sys; sys.stdout.write('hi')"),
            stage="source",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

    events = boundary_events(caplog)
    assert len(events) == 1
    assert events[0]["type"] == "distill.run_command"
    detail = events[0]["detail"]
    assert detail["stage"] == "source"
    assert detail["tool"] == PYTHON
    assert detail["outcome"] == "ok"
    assert detail["exit_status"] == 0
    assert detail["timeout_fired"] is None
    # The run is one process, so pid plus ordinal is what correlates this
    # invocation to the run that made it and orders it among the others.
    assert detail["pid"] == os.getpid()
    assert isinstance(detail["sequence"], int)
    assert detail["duration_sec"] >= 0


def test_the_boundary_event_carries_no_captured_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Captured output is **extracted text** and a debug log is not a redaction sink.

    The child assembles the secret at runtime, so the string never appears in
    the argv the event does record: what is asserted here is that the tool's
    *output* stayed out, not that nothing recognizable was ever logged.
    """
    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        run(
            child("import sys; s = 'sk' + '-secret'; sys.stdout.write(s); sys.stderr.write(s)"),
            stage="ocr",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
        )

    assert "sk-secret" not in caplog.text


def test_a_failed_invocation_still_emits_its_boundary_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The events an operator needs most are the ones that ended badly."""
    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        with pytest.raises(DistillError):
            run(
                child("import sys; sys.exit(3)"),
                stage="frames",
                total_timeout_sec=GENEROUS_TOTAL_SEC,
                idle_timeout_sec=GENEROUS_IDLE_SEC,
            )
        with pytest.raises(DistillError):
            run(
                ["distill-no-such-tool"],
                stage="frames",
                total_timeout_sec=GENEROUS_TOTAL_SEC,
                idle_timeout_sec=GENEROUS_IDLE_SEC,
            )
        with pytest.raises(DistillError):
            run(
                child("import time; time.sleep(30)"),
                stage="frames",
                total_timeout_sec=GENEROUS_TOTAL_SEC,
                idle_timeout_sec=0.2,
            )

    outcomes = [event["detail"]["outcome"] for event in boundary_events(caplog)]
    assert outcomes == ["exit_status", "missing_tool", "idle_timeout"]


def test_a_callback_that_raises_still_emits_its_boundary_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A progress reporter can raise, and a tool that ran must leave a trace.

    `ProgressReporter.update` raises once its counter is stopped, so this is a
    production path, not a hypothetical one.
    """

    def explode(_line: str) -> None:
        raise RuntimeError("progress reporter is stopped")

    with (
        caplog.at_level(logging.DEBUG, logger="distill.run_command"),
        pytest.raises(RuntimeError),
    ):
        stream(
            child(CHATTY_FOREVER),
            stage="youtube",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
            terminate_grace_sec=0.2,
            on_stdout_line=explode,
        )

    assert [event["detail"]["outcome"] for event in boundary_events(caplog)] == [
        "callback_error"
    ]


def test_an_unchecked_non_zero_exit_is_not_reported_as_ok(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A **degradation** is exactly what an operator comes to this event for."""
    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        run(
            child("import sys; sys.exit(1)"),
            stage="youtube",
            total_timeout_sec=GENEROUS_TOTAL_SEC,
            idle_timeout_sec=GENEROUS_IDLE_SEC,
            check=False,
        )

    assert boundary_events(caplog)[0]["detail"]["outcome"] == "nonzero_exit_unchecked"


# --- the migrated call sites (M2.2) ------------------------------------------

PACKAGE_ROOT = Path(run_command.__file__).parent
# The one module allowed to import `subprocess`. Scope is the whole package,
# with no exemption: `measure.py` is an offline harness rather than runtime
# pipeline code, but it still invokes ffmpeg, and R-29 admits no second path.
SUBPROCESS_OWNER = "run_command.py"

# The names `run_command` exports for invoking a tool. A call to one of these
# is a call site, and every call site is held to R-30 below.
HELPER_NAMES = frozenset({"run", "stream", "run_json"})


def package_modules() -> list[Path]:
    return sorted(
        module
        for module in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in module.parts
    )


def call_sites() -> list[tuple[str, int, dict[str, ast.expr]]]:
    """Every invocation of a `run_command` helper outside `run_command` itself.

    Read off the source rather than a maintained list, so a call site added
    later is covered without anyone remembering to enumerate it. Only calls to
    names the module imported from `run_command` count - an unrelated
    `runner.run(...)` is not a tool invocation.
    """
    found: list[tuple[str, int, dict[str, ast.expr]]] = []
    for module in package_modules():
        if module.name == SUBPROCESS_OWNER:
            continue
        tree = ast.parse(module.read_text())
        imported = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("run_command")
            for alias in node.names
            if alias.name in HELPER_NAMES
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in imported
            ):
                found.append(
                    (
                        module.name,
                        node.lineno,
                        {
                            keyword.arg: keyword.value
                            for keyword in node.keywords
                            if keyword.arg
                        },
                    )
                )
    return found


def declared_timeouts() -> dict[str, CommandTimeouts]:
    """Every `CommandTimeouts` pair the package declares, discovered by import."""
    found: dict[str, CommandTimeouts] = {}
    for module in package_modules():
        if module.name == "__init__.py":
            continue
        imported = importlib.import_module(f"distill.{module.stem}")
        for name, value in vars(imported).items():
            if isinstance(value, CommandTimeouts):
                found[f"{module.stem}.{name}"] = value
    return found


def test_the_package_has_call_sites_to_check() -> None:
    """Guards the two tests below against silently checking an empty set."""
    assert {module for module, _line, _kwargs in call_sites()} == {
        "frame_selection.py",
        "measure.py",
        "ocr.py",
        "source.py",
        "transcript.py",
    }


def test_every_call_site_passes_both_timeouts() -> None:
    """R-30: no invocation inherits a limit by omission, not even the default."""
    missing = [
        f"{module}:{line}"
        for module, line, kwargs in call_sites()
        if not {"total_timeout_sec", "idle_timeout_sec"} <= set(kwargs)
    ]

    assert missing == []


# What a timeout argument must look like: `SOME_TIMEOUTS.total_sec`, reading
# off a declared `CommandTimeouts` pair.
TIMEOUT_FIELDS = {"total_timeout_sec": "total_sec", "idle_timeout_sec": "idle_sec"}


def test_no_call_site_inlines_a_timeout_value() -> None:
    """A literal at a call site is a limit nobody can find, review, or raise."""
    def reads_the_named_field(value: ast.expr, field: str) -> bool:
        return isinstance(value, ast.Attribute) and value.attr == field

    inlined = [
        f"{module}:{line} {argument}"
        for module, line, kwargs in call_sites()
        for argument, field in TIMEOUT_FIELDS.items()
        if argument in kwargs and not reads_the_named_field(kwargs[argument], field)
    ]

    assert inlined == []


def test_every_declared_timeout_pair_bounds_something() -> None:
    pairs = declared_timeouts()

    assert pairs
    for name, timeouts in pairs.items():
        assert timeouts.total_sec > 0 and math.isfinite(timeouts.total_sec), name
        assert timeouts.idle_sec > 0 and math.isfinite(timeouts.idle_sec), name


def test_every_module_with_a_call_site_declares_a_named_timeout_pair() -> None:
    """The pair each call site reads from is declared in the module that runs it."""
    invoking = {module.removesuffix(".py") for module, _line, _kwargs in call_sites()}
    declaring = {name.split(".", 1)[0] for name in declared_timeouts()}

    assert invoking <= declaring


def test_a_timeout_pair_cannot_be_constructed_half_specified() -> None:
    """R-30: there is no constructor that omits one of the two limits."""
    with pytest.raises(TypeError):
        CommandTimeouts(total_sec=60.0)  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        CommandTimeouts(idle_sec=60.0)  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    ("total", "idle"),
    [(0.0, 60.0), (-1.0, 60.0), (float("inf"), 60.0), (60.0, 0.0), (60.0, float("nan"))],
)
def test_a_timeout_pair_rejects_a_limit_that_bounds_nothing(total: float, idle: float) -> None:
    with pytest.raises(ValueError):
        CommandTimeouts(total_sec=total, idle_sec=idle)


# --- silent-by-design tools (finding 4) --------------------------------------


def test_a_silent_tool_pair_puts_the_declared_number_in_both_places() -> None:
    """One number governs, because with no output there is no idle clock.

    A tool that says nothing until it is finished never resets the idle timer,
    so a lower idle value is not a stall detector - it is a quieter deadline
    that replaces the one the call site wrote down.
    """
    timeouts = silent_tool_timeouts(90.0)

    assert timeouts == CommandTimeouts(total_sec=90.0, idle_sec=90.0)


def test_a_silent_tool_call_site_cannot_tighten_its_idle_value_alone() -> None:
    """The constructor takes one limit, so the two cannot drift apart."""
    parameters = inspect.signature(silent_tool_timeouts).parameters

    assert list(parameters) == ["total_sec"]


def test_the_silent_by_design_call_sites_declare_a_single_deadline() -> None:
    """tesseract and `ffprobe -v error` both answer only when they are done."""
    for name in ("ocr.TESSERACT_TIMEOUTS", "source.FFPROBE_TIMEOUTS"):
        timeouts = declared_timeouts()[name]
        assert timeouts.idle_sec == timeouts.total_sec, name
