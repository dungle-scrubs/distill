"""The CLI's error boundary: every exception leaves as the JSON error object (R-46).

Finding 14 is what these cover. `cli.main` caught `DistillError` and nothing
else, so anything Distill had not already coded - an unreadable **job record**,
a malformed `--args`, a **fatal error** the filesystem raised - left the process
as a Python traceback on stderr with exit 1. A traceback is not a contract: an
operator scripting Distill cannot branch on it, the exit code says only "some
python died", and the stack names Distill's internals to whoever ran the
command.

What the boundary owes, stated once because every test here is one half of it:
for an exception a command raised, the **fatal error** record (`code`, `stage`,
`message`, `details`) as JSON on stderr, exit code 2, and no traceback text on
either stream.

*An exception a command raised*, and not every way a command can end. Three
endings are deliberately outside that shape, and each is covered here as what it
is rather than as a gap: an argument the parser rejects is argparse's usage
message and exit 2; `Ctrl-C` is `KeyboardInterrupt`, which leaves with the
interpreter's own traceback and exit 130 because the operator ended their own
command; and a caller that stops reading stdout gets exit 141, because nothing
failed.

Nor is stderr the record and nothing else. A processing run writes NDJSON
**progress** records there as it goes, so the record is the last thing Distill
writes rather than the whole stream -
`test_a_failing_run_leaves_the_error_record_as_the_last_line_of_stderr` is the
one test here that drives a command emitting progress, and `error_object` below
is usable by the others only because the commands they drive emit none.

R-46's other half is at the bottom of this file. A batch reports each item that
failed, and it flattened every failure to `str(exc)` - so a **fatal error** that
travelled the whole way as a code and a stage arrived in the batch report as a
sentence, and two items that failed for different reasons were indistinguishable
to anything but a human reading them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from distill.cli import build_parser, main
from distill.errors import DistillError
from distill.local_vision import MAX_SOCKET_TIMEOUT_SEC

APP = Path(__file__).resolve().parents[1]

FATAL_ERROR_FIELDS = {"code", "stage", "message", "details"}


def error_object(captured: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """The **fatal error** the boundary wrote, proven to be one and only one.

    Both streams are checked for traceback text here rather than in each test:
    a boundary that reports the error object *and* dumps a stack has not stopped
    the leak, it has added a line above it.
    """
    output = captured.readouterr()
    assert "Traceback" not in output.err
    assert "Traceback" not in output.out
    payload = json.loads(output.err)
    assert isinstance(payload, dict)
    assert set(payload) >= FATAL_ERROR_FIELDS
    return payload


def a_corrupt_job_record(root: Path, job_id: str) -> None:
    """A **job record** whose bytes are not text at all.

    `JobStore._parse` already refuses a record that is not JSON and one that
    carries no known status, which is why those two are not finding 14. Bytes
    that are not UTF-8 never reach it: `path.read_text()` raises
    `UnicodeDecodeError` one line earlier, and that is a `ValueError` with
    nothing coded about it.
    """
    jobs = root / "_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{job_id}.json").write_bytes(b'{"status": "completed", "tool": "\xff\xfe"}')


def test_a_corrupt_job_record_is_reported_as_the_json_error_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST (finding 14): the boundary catches `DistillError` and nothing else."""
    a_corrupt_job_record(tmp_path, "corrupt-job")

    with pytest.raises(SystemExit) as exit_info:
        main(["get-job-status", "corrupt-job", "--output-dir", str(tmp_path)])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_INTERNAL"
    assert "UnicodeDecodeError" in json.dumps(payload)


def test_a_corrupt_job_record_exits_two_from_a_real_process(tmp_path: Path) -> None:
    """The exit code the operator's shell sees, not the one a test caught.

    In-process `SystemExit(2)` is the same object either way, but a boundary
    that wrote the error object to stdout, or that let the interpreter print a
    stack on the way out, looks identical from inside `pytest.raises`.
    """
    a_corrupt_job_record(tmp_path, "corrupt-job")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "distill.cli",
            "get-job-status",
            "corrupt-job",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(APP / "src")},
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert json.loads(result.stderr)["code"] == "E_INTERNAL"


# --- The boundary's own streams: what happens when the channel is not there. --
#
# Every test above drives a failure through a boundary whose stderr is a working
# pipe. These drive the same boundary with the channel it writes to already
# gone, which is the state a caller leaves behind by closing a descriptor or by
# walking away from a pipe - and the state in which an error boundary that
# cannot write is at its most dangerous, because the fallback it takes is the
# one nobody chose.

CLI_ENTRY_POINTS: dict[str, list[str]] = {
    "module": [sys.executable, "-m", "distill.cli"],
    "console_script": [str(Path(sys.executable).with_name("distill"))],
}
"""Both ways the boundary is entered, because the recipe has to be on both.

`distill = distill.cli:main` is what an operator types; `python -m distill.cli`
is what the rest of this file drives. They share `main`, and a guard written
into `__main__` rather than into `main` would hold for one and not the other.
"""


def cli_child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(APP / "src")}


def a_reader_that_is_already_gone() -> tuple[int, int]:
    """A pipe with no reader at all, so the first write is `EPIPE` (no race).

    Closing the read end *before* the child is started is what makes this
    deterministic. Handing the child a pipe and closing the read end afterwards
    races the child's first write against the parent's close, and a write that
    wins that race lands in the pipe buffer and succeeds.
    """
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    return read_fd, write_fd


@pytest.mark.parametrize("entry", sorted(CLI_ENTRY_POINTS), ids=sorted(CLI_ENTRY_POINTS))
def test_a_caller_that_stopped_reading_stdout_ends_the_command_at_141(entry: str) -> None:
    """FAILS FIRST: `distill list-tools | head` exits 120 with interpreter noise.

    A caller closing Distill's stdout is `| head`, and it is not a Distill
    failure - it is the shell's own idiom. What it produced was a
    `BrokenPipeError` converted by the catch-all into an `E_INTERNAL` record
    saying an unexpected error ended the command, followed by the interpreter's
    "Exception ignored on flushing sys.stdout" on the way out and exit 120: two
    diagnoses of a defect, for a caller who did exactly what the shell invites.
    """
    _read_fd, write_fd = a_reader_that_is_already_gone()
    try:
        result = subprocess.run(
            [*CLI_ENTRY_POINTS[entry], "list-tools"],
            stdout=write_fd,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=cli_child_env(),
        )
    finally:
        os.close(write_fd)

    assert result.returncode == 141
    assert result.stderr == ""


def test_a_command_that_cannot_reach_stderr_writes_nothing_to_stdout(tmp_path: Path) -> None:
    """FAILS FIRST: `print(file=None)` is `print(file=sys.stdout)`.

    An interpreter started with file descriptor 2 closed has `sys.stderr is
    None` - not a broken stream, no stream - and `print(..., file=None)` falls
    back to stdout. So the one channel the boundary promises to keep clean is
    exactly where the error record went, and a caller piping stdout to `jq` got
    a result-shaped error after all.
    """
    a_corrupt_job_record(tmp_path, "corrupt-job")

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'exec "$0" -m distill.cli get-job-status corrupt-job --output-dir "$1" 2>&-',
            sys.executable,
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=cli_child_env(),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_a_command_whose_stderr_is_a_broken_pipe_still_exits_two(tmp_path: Path) -> None:
    """FAILS FIRST: the write raises inside the `except` clause. Exit 120.

    `_fail` runs from an exception handler, so an `OSError` it raises is not
    caught by anything: it leaves `main` as a second, unrelated failure, the
    interpreter fails to print its traceback to the same dead stream, and the
    process ends on the exit code that means "could not flush the std streams".
    Exit 2 is what the contract says a failing command exits with, and a caller
    that stopped reading the diagnosis has not changed what happened.
    """
    a_corrupt_job_record(tmp_path, "corrupt-job")
    _read_fd, write_fd = a_reader_that_is_already_gone()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "distill.cli",
                "get-job-status",
                "corrupt-job",
                "--output-dir",
                str(tmp_path),
            ],
            stdout=subprocess.PIPE,
            stderr=write_fd,
            text=True,
            check=False,
            env=cli_child_env(),
        )
    finally:
        os.close(write_fd)

    assert result.returncode == 2
    assert result.stdout == ""


A_STREAM_CLOSED_BEFORE_THE_COMMAND = """
import sys
sys.path.insert(0, {src!r})
sys.{stream}.close()
from distill.cli import main
main(["get-job-status", "corrupt-job", "--output-dir", sys.argv[1]])
"""
"""A command whose stream object is closed, which is not the same as a dead fd.

A closed `TextIOWrapper` answers a write or a flush with `ValueError: I/O
operation on closed file`, not with an `OSError` - so a guard written for the
descriptor going away lets the stream going away straight through.
"""


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_a_command_whose_stream_was_closed_still_exits_two(stream: str, tmp_path: Path) -> None:
    """FAILS FIRST: `ValueError` from the guarded flush. Exit 1, with a stack.

    The guards named `OSError`, which is the whole vocabulary of a descriptor
    that has gone away and none of the vocabulary of a *stream* that has. Python
    closes both on `sys.stdout.close()`, and an embedder or a wrapper script that
    tidies up its streams before handing over does the same thing.
    """
    a_corrupt_job_record(tmp_path, "corrupt-job")
    script = tmp_path / "closed.py"
    script.write_text(
        A_STREAM_CLOSED_BEFORE_THE_COMMAND.format(src=str(APP / "src"), stream=stream)
    )

    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        env=cli_child_env(),
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stdout
    if stream == "stdout":
        assert json.loads(result.stderr)["code"] == "E_INTERNAL"


A_DESCRIPTOR_CLOSED_UNDER_A_BUFFERED_WRITE = """
import os, sys
sys.path.insert(0, {src!r})
sys.stdout.write("a partial result")
os.close(1)
from distill.cli import main
main(["get-job-status", "corrupt-job", "--output-dir", sys.argv[1]])
"""
"""Buffered output, and the descriptor under it closed before the command fails.

The buffer is what makes this different from a closed stream: something is
waiting to be written when the boundary reaches for the descriptor, and again
when the interpreter does on the way out.
"""


def test_a_command_whose_stdout_descriptor_vanished_still_exits_two(tmp_path: Path) -> None:
    """FAILS FIRST: exit 120, with the interpreter's flush noise after the record.

    `_discard` opens the null device and `dup2`s it over the descriptor - but
    with file descriptor 1 *closed*, `os.open` is handed 1 as the lowest free
    number, so the null device already is the descriptor, `dup2(1, 1)` does
    nothing, and closing what it opened closes stdout for the second time. The
    shutdown flush then finds a bad descriptor exactly as before.
    """
    a_corrupt_job_record(tmp_path, "corrupt-job")
    script = tmp_path / "vanished.py"
    script.write_text(A_DESCRIPTOR_CLOSED_UNDER_A_BUFFERED_WRITE.format(src=str(APP / "src")))

    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        env=cli_child_env(),
    )

    assert result.returncode == 2
    assert "Exception ignored" not in result.stderr
    assert json.loads(result.stderr)["code"] == "E_INTERNAL"


def test_a_broken_pipe_that_is_not_the_callers_output_is_not_read_as_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: exit 141 for a broken pipe that had nothing to do with stdout.

    `BrokenPipeError` means "this pipe has no reader", and only the one Distill
    writes its *result* to says anything about the caller. A progress record
    written to a stderr nobody is reading raises the same class, and so would any
    socket or pipe a stage holds - and answering those with the caller-left-early
    exit turns an uncoded failure into a success-adjacent one nobody reported.
    """

    def fault(*_args: object, **_kwargs: object) -> Any:
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("distill.cli.timeout_diagnostics", fault)

    with pytest.raises(SystemExit) as exit_info:
        main(["timeout-diagnostics"])

    assert exit_info.value.code == 2
    assert error_object(capsys)["code"] == "E_INTERNAL"


def test_call_tool_with_unparseable_args_is_reported_as_the_json_error_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FAILS FIRST: `json.loads('{bad')` raises a bare `JSONDecodeError`.

    The code is `E_BAD_ARGUMENT` and not `E_INTERNAL`: an operator who mistyped
    a JSON literal has not found a defect in Distill, and an error object saying
    "internal" would send them looking for one.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(["call-tool", "cache_doctor", "--args", "{bad"])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_ARGUMENT"
    assert payload["stage"] == "cli"


def test_call_tool_args_that_are_not_an_object_are_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--args '[]'` parses as JSON and is still not a tool's arguments.

    It reached `call_tool` as a list and died on `.get` - a well-formed document
    of the wrong shape is the same operator mistake as a malformed one, and gets
    the same answer rather than an `AttributeError`.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(["call-tool", "cache_doctor", "--args", "[1, 2]"])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_ARGUMENT"
    assert payload["stage"] == "cli"


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not refused by directory permissions")
def test_a_permission_error_on_the_output_root_is_reported_as_the_json_error_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An output root Distill cannot create is a diagnosis, not a stack.

    `validate_output_root` refuses the roots it has a policy about and then
    calls `mkdir`, which is where the filesystem gets its say. `PermissionError`
    is an `OSError`, so it walked straight past a boundary that named only
    `DistillError`.

    Driven through `cleanup-cache`, because the exception this covers is the one
    `mkdir` raises and `cache-doctor` deliberately never calls it (R-57): the
    read-only command validates its root with `create=False`. It provoked a
    `PermissionError` anyway on Python 3.13, from the `Path.is_dir()` behind
    `survey` - and on 3.14, where `Path.is_dir()` returns `False` for `EACCES`
    instead of raising, the same command succeeded and printed a report. The
    root that cannot be reached is now `cache-doctor`'s own typed refusal
    (`E_OUTPUT_ROOT_UNREADABLE`), which is a different contract from this one;
    what belongs here is a command that really does try to create the root, and
    that is every writing command.
    """
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    unreadable.chmod(0o000)
    try:
        with pytest.raises(SystemExit) as exit_info:
            main(["cleanup-cache", "--output-dir", str(unreadable / "root")])
    finally:
        unreadable.chmod(0o700)

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_INTERNAL"
    assert "PermissionError" in json.dumps(payload)


def registered_subcommands() -> tuple[str, ...]:
    """Every subcommand the parser registers, read off the parser itself.

    Derived rather than listed, so a command added to `build_parser` without an
    entry in `INTERNAL_FAULT_ARGV` fails this module instead of quietly opting
    out of the boundary.
    """
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return tuple(action.choices)
    raise AssertionError("the parser registers no subcommands")


INTERNAL_FAULT_ARGV: dict[str, list[str]] = {
    "process-local-video": ["process-local-video", "video.mp4"],
    "process-youtube-video": ["process-youtube-video", "https://youtu.be/abcdefghijk"],
    "process-video-directory": ["process-video-directory", "clips"],
    "process-youtube-playlist": ["process-youtube-playlist", "https://youtube.com/playlist?list=P"],
    "cleanup-cache": ["cleanup-cache"],
    "cache-doctor": ["cache-doctor"],
    "get-job-status": ["get-job-status", "some-job"],
    "list-tools": ["list-tools"],
    "timeout-diagnostics": ["timeout-diagnostics"],
    "timeout-probe": ["timeout-probe", "1"],
    "local-vision-diagnostics": ["local-vision-diagnostics"],
    "call-tool": ["call-tool", "cache_doctor"],
}
"""One accepted invocation per subcommand, for the sweep below."""

CLI_WORK_SEAMS = (
    "call_registered_tool",
    "list_tools",
    "local_vision_diagnostics",
    "run_timeout_probe",
    "timeout_diagnostics",
)
"""Everything `cli.main` dispatches work to. Faulted together, so every branch fails."""


def test_the_sweep_covers_every_registered_subcommand() -> None:
    assert set(INTERNAL_FAULT_ARGV) == set(registered_subcommands())


@pytest.mark.parametrize("command", sorted(INTERNAL_FAULT_ARGV))
def test_no_command_lets_an_internal_fault_reach_the_user_as_a_traceback(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every subcommand, with the work it dispatches to raising `RuntimeError`.

    Parameterized over the parser's own registry rather than over a list, so the
    claim is about *every* command rather than the ones somebody remembered.
    `RuntimeError` stands in for the whole class the boundary did not name: it
    is not a `DistillError`, carries no code, and is exactly what an unhandled
    defect three modules down looks like from here.
    """

    def fault(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("a fault three modules down")

    for seam in CLI_WORK_SEAMS:
        monkeypatch.setattr(f"distill.cli.{seam}", fault)

    class FaultingSession:
        def call_tool(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
            raise RuntimeError("a fault three modules down")

    monkeypatch.setattr("distill.cli.DistillSession", FaultingSession)

    with pytest.raises(SystemExit) as exit_info:
        main(INTERNAL_FAULT_ARGV[command])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_INTERNAL"
    assert "RuntimeError" in json.dumps(payload)


def test_an_operator_typo_still_gets_argparses_usage_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Parse-time `SystemExit(2)` is the contract, not a leak the boundary missed.

    A command that does not exist, or a flag with no value, is answered by the
    usage message argparse already writes - the same exit code the error object
    uses, and the answer an operator can act on. The boundary is downstream of
    parsing on purpose.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(["no-such-command"])

    assert exit_info.value.code == 2
    output = capsys.readouterr()
    assert "Traceback" not in output.err
    # Python 3.14's argparse colorizes usage when the environment says the
    # terminal can; strip ANSI escapes so the assertion is about the words.
    plain_err = re.sub(r"\x1b\[[0-9;]*m", "", output.err)
    assert "usage: distill" in plain_err
    # And *only* the usage message. A boundary that had swallowed the parser's
    # own `SystemExit` would answer a typo with an error object naming
    # `SystemExit`, which says nothing about how to spell the command.
    assert '"code"' not in output.err


def test_a_probe_no_clock_can_sleep_for_is_refused_as_a_bad_argument(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: `OverflowError` from `time.sleep`, reported as `E_INTERNAL`.

    The same escape `NumericDomain.ceiling` closed for `--local-vision-timeout-
    sec`, at the one door that does not go through the option boundary.
    `run_timeout_probe` already refuses a negative probe and a long one; above
    the range a sleep can represent it refused nothing, and the number an
    operator typed reached `time.sleep` as `OverflowError: timestamp out of
    range for platform time_t`.

    `E_BAD_ARGUMENT` at stage `timeout`, which is what this function already
    answers its floor with. The catch-all is a backstop for defects, and
    reporting an operator's own number as an internal fault is a wrong diagnosis
    with the argument's name thrown away.
    """
    monkeypatch.setenv("DISTILL_ENABLE_LONG_TIMEOUT_PROBE", "1")

    with pytest.raises(SystemExit) as exit_info:
        main(["timeout-probe", str(10**30)])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_ARGUMENT"
    assert payload["stage"] == "timeout"
    assert payload["details"]["probe_ms"] == 10**30


def test_a_registered_command_no_branch_dispatches_is_reported_not_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: the dispatch chain falls off the end and the command exits 0.

    A subcommand registered on the parser without a branch in `_dispatch` prints
    nothing and succeeds, which is indistinguishable from a command that ran and
    had nothing to say - so a script driving Distill reads success for work that
    never happened. `test_the_sweep_covers_every_registered_subcommand` catches
    the omission in this repository's own tests; nothing caught it at runtime.
    """

    def parser_with_an_unrouted_command() -> argparse.ArgumentParser:
        parser = build_parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                action.add_parser("unrouted")
        return parser

    monkeypatch.setattr("distill.cli.build_parser", parser_with_an_unrouted_command)

    with pytest.raises(SystemExit) as exit_info:
        main(["unrouted"])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_INTERNAL"
    assert payload["details"]["command"] == "unrouted"


def test_a_converter_that_fails_at_parse_time_is_converted_like_any_other_fault(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: parsing ran outside the boundary, so this left as a traceback.

    Argparse turns a `ValueError` or a `TypeError` from a `type=` converter into
    a usage error, and turns nothing else into anything: an action, a default or
    a converter that raises for any other reason walks straight out of
    `parse_args`. With parsing outside the `try` there was no handler above it
    at all, which is finding 14 again at the one place the boundary did not
    cover.

    The typo contract is unchanged, and
    `test_an_operator_typo_still_gets_argparses_usage_message` is the other half
    of this pair: argparse ends a usage error with `SystemExit`, a
    `BaseException`, which passes a clause naming `Exception` untouched.
    """

    def explode(_raw: str) -> int:
        raise RuntimeError("a converter three modules down")

    def parser_with_a_faulting_converter() -> argparse.ArgumentParser:
        parser = build_parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for probe_action in action.choices["timeout-probe"]._actions:
                    if probe_action.dest == "probe_ms":
                        probe_action.type = explode
        return parser

    monkeypatch.setattr("distill.cli.build_parser", parser_with_a_faulting_converter)

    with pytest.raises(SystemExit) as exit_info:
        main(["timeout-probe", "1"])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_INTERNAL"
    assert "RuntimeError" in json.dumps(payload)


def test_the_debug_escape_hatch_re_raises_the_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DISTILL_TRACEBACK=1` is how a maintainer gets the stack back.

    The boundary is a one-line diagnosis by design, which is the right default
    for an operator and the wrong one for whoever has to fix what it names. An
    opt-in env var is the whole of the escape: off, nothing changes; on, the
    original exception propagates with its stack intact.
    """

    def fault(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("a fault three modules down")

    monkeypatch.setattr("distill.cli.timeout_diagnostics", fault)
    monkeypatch.setenv("DISTILL_TRACEBACK", "1")

    with pytest.raises(RuntimeError, match="three modules down"):
        main(["timeout-diagnostics"])


def test_a_non_numeric_max_age_days_is_refused_as_a_bad_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: `float(args["max_age_days"])` raises a bare `ValueError`.

    Recorded by M8.1 as M9.1's. `--max-age-days` is `type=float` at the parser,
    so the door this arrives through is `call-tool`, where the arguments are a
    JSON document rather than an argv. `PrunePolicy` already refuses the value
    with `E_BAD_OPTIONS`; the conversion in front of it never let the policy see
    it.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "call-tool",
                "cleanup_cache",
                "--args",
                json.dumps({"output_dir": str(tmp_path), "max_age_days": "soon"}),
            ]
        )

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_OPTIONS"
    assert payload["stage"] == "prune"


def test_a_non_numeric_keep_generations_is_refused_as_a_bad_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same conversion, one line up: `int(args.get("keep_generations"))`."""
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "call-tool",
                "cache_doctor",
                "--args",
                json.dumps({"output_dir": str(tmp_path), "keep_generations": "lots"}),
            ]
        )

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_OPTIONS"
    assert payload["stage"] == "prune"


def test_a_vision_timeout_no_socket_can_hold_is_refused_on_the_run_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: `1e300` is finite, so the run path admitted it (M8 review).

    `_coerce_float` is the config layer's door and stops there, but the run path
    validates the *raw* argument instead of coercing it, deliberately - an
    operator naming a timeout on the command line should be told it is unusable
    rather than silently given the default. It was told nothing: `1e300` cleared
    `math.isfinite` and travelled into `socket.settimeout`, which answered with
    `OverflowError` from the stdlib.

    Refused as `E_BAD_OPTIONS` at the option boundary rather than caught as
    `E_INTERNAL` at the CLI boundary. The catch-all is a backstop for defects,
    and reporting an operator's own number as an internal fault is a wrong
    diagnosis with the option's name thrown away.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "process-local-video",
                str(tmp_path / "video.mp4"),
                "--output-dir",
                str(tmp_path / "out"),
                "--local-vision-timeout-sec",
                "1e300",
            ]
        )

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_OPTIONS"
    assert payload["stage"] == "options"
    assert payload["details"]["local_vision_timeout_sec"] == repr(1e300)


def test_details_json_cannot_write_still_reach_the_operator_as_a_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: `TypeError: Object of type PosixPath is not JSON serializable`.

    A stage raising about a file puts the `Path` in `details` because that is
    what it was holding. `_fail` runs from inside an `except` clause, so the
    `TypeError` the serialization raises is caught by nothing at all: the
    operator gets no error object, and the stack the boundary exists to replace
    is the stack of the boundary itself.
    """

    def fault(*_args: object, **_kwargs: object) -> Any:
        raise DistillError(
            "E_BAD_MEDIA",
            "source",
            "unreadable",
            {"path": Path("/tmp/clip.mp4"), "head": b"\xff\xd8"},
        )

    monkeypatch.setattr("distill.cli.timeout_diagnostics", fault)

    with pytest.raises(SystemExit) as exit_info:
        main(["timeout-diagnostics"])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_MEDIA"
    assert "clip.mp4" in payload["details"]["path"]


@pytest.mark.parametrize(
    "timeout",
    ["-5", "0", "nan", "1e300", repr(MAX_SOCKET_TIMEOUT_SEC)],
    ids=["negative", "zero", "not-a-number", "over-ceiling", "at-ceiling"],
)
def test_a_diagnostics_timeout_outside_the_domain_is_refused_like_a_run_path(
    timeout: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAILS FIRST: the diagnostics door coerced what the run path refuses.

    Two doors, one option, two answers. `--local-vision-timeout-sec -5` on
    `process-local-video` is `E_BAD_OPTIONS` naming the option, which is what
    the README says happens to a value outside a numeric domain. The same flag
    on `local-vision-diagnostics` went straight to the config layer, whose
    contract is to *coerce* - so `-5`, `0`, `nan` and `1e300` all printed
    `timeout_sec: 30.0` and exited 0, telling an operator checking their vision
    settings that the setting they just named is in force.

    Coercion is right for a config file, which a run should not stop for, and
    wrong for a number an operator typed at a diagnostic command whose whole
    purpose is to report what their arguments resolve to.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(["local-vision-diagnostics", "--local-vision-timeout-sec", timeout])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_OPTIONS"
    assert payload["stage"] == "options"
    assert "local_vision_timeout_sec" in payload["details"]


def test_a_diagnostics_timeout_inside_the_domain_is_still_honoured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The refusal is the domain's edge, not the flag's."""
    main(["local-vision-diagnostics", "--local-vision-timeout-sec", "12.5"])

    assert json.loads(capsys.readouterr().out)["config"]["timeout_sec"] == 12.5


def test_an_output_dir_that_is_not_a_path_is_refused_as_a_bad_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FAILS FIRST: `Path(5)` is a `TypeError`, reported as `E_INTERNAL`.

    `--args` is a JSON document, so every argument arrives with a type the
    operator chose. A wrongly typed value inside a well-formed object is a typo
    like any other and gets the answer `PrunePolicy` already gives one - the
    option named, at the boundary that reads it - rather than the catch-all's
    "an unexpected error", which sends the operator looking for a defect in
    Distill.
    """
    with pytest.raises(SystemExit) as exit_info:
        main(["call-tool", "cache_doctor", "--args", json.dumps({"output_dir": 5})])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload["code"] == "E_BAD_OPTIONS"
    assert payload["details"]["output_dir"] == "5"


def test_the_error_object_names_the_exception_without_printing_its_stack(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Enough to diagnose, and nothing an operator has to read a stack for.

    The type and the message are what identify an unexpected failure; the frames
    are what would leak Distill's internals to whoever ran the command. Both
    halves are asserted here so neither can be dropped as an implementation
    detail of the other.
    """

    def fault(*_args: object, **_kwargs: object) -> Any:
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr("distill.cli.timeout_diagnostics", fault)

    with pytest.raises(SystemExit):
        main(["timeout-diagnostics"])

    payload = error_object(capsys)
    assert payload["stage"] == "internal"
    assert payload["details"]["exception"] == "ZeroDivisionError"
    assert "division by zero" in payload["details"]["message"]
    assert 'File "' not in json.dumps(payload)


def test_a_failing_run_leaves_the_error_record_as_the_last_line_of_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error channel is shared with the progress stream, and that is the contract.

    Every other test in this file drives a command that emits no progress, so
    each one could read the whole of stderr as the record. A processing command
    cannot: progress is NDJSON on stderr from the first mechanism onward, so by
    the time a stage fails the error record is the *last* line of a stream, not
    the whole of it. A caller told only "the error object is on stderr" would
    `json.loads` the lot and fail on the first newline.

    What separates them is a field. Every progress record carries
    `type: distill.progress`; the **fatal error** record carries `code`,
    `stage`, `message` and `details` and no `type` at all, so the two are told
    apart by shape - and the record is the last thing Distill writes, so a
    caller wanting only the failure can take it from the end.

    Distill's own lines, and not every line: an imported library writes to
    stderr when it feels like it (on this platform `av` emits a dyld class
    warning), which is why the claim is about the records Distill writes rather
    than about the bytes on the descriptor.
    """
    from test_local_integration import fake_transcribe, make_short_screencast

    from distill import pipeline as distill_pipeline

    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)

    def fail_at_the_render(*_args: object, **_kwargs: object) -> Any:
        raise DistillError("E_BAD_RENDER", "render", "the render could not be written")

    # Transcription is faked because a real one loads a model; keyframe
    # selection is real, because it is the stage whose progress this test is
    # about. The failure is put after it, which is where a real one lands.
    monkeypatch.setattr(distill_pipeline, "transcribe_with_imports", fake_transcribe)
    monkeypatch.setattr(distill_pipeline, "render_markdown", fail_at_the_render)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "process-local-video",
                str(video),
                "--output-dir",
                str(tmp_path / "cache"),
                "--no-ocr",
                "--no-caption-frames",
            ]
        )

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    records = [
        json.loads(line)
        for line in captured.err.splitlines()
        if line.startswith("{") and line.rstrip().endswith("}")
    ]

    assert len(records) > 1, "the command emitted no progress, so it proves nothing"
    assert all(record["type"] == "distill.progress" for record in records[:-1])
    assert "type" not in records[-1]
    assert records[-1]["code"] == "E_BAD_RENDER"
    assert set(records[-1]) >= FATAL_ERROR_FIELDS


def test_a_distill_error_still_reaches_stderr_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The catch-all does not shadow the coded path it was added beside."""

    def raise_error(_name: str, _args: dict[str, Any]) -> dict[str, Any]:
        raise DistillError("E_TEST", "test", "boom", {"detail": 1})

    monkeypatch.setattr("distill.cli.call_registered_tool", raise_error)

    with pytest.raises(SystemExit) as exit_info:
        main(["get-job-status", "any", "--output-dir", str(tmp_path)])

    assert exit_info.value.code == 2
    payload = error_object(capsys)
    assert payload == {
        "code": "E_TEST",
        "stage": "test",
        "message": "boom",
        "details": {"detail": 1},
    }


# --- R-46's other half: what a batch says about an item that failed. ----------


def two_items_failing_differently(item: str, _index: int) -> dict[str, Any]:
    """Two **fatal errors** that differ in code *and* stage, plus one uncoded fault.

    Differing in both is the point: flattening to `str(exc)` loses each
    independently, so an assertion over one code would pass with the stage still
    thrown away.
    """
    if item == "a.mp4":
        raise DistillError("E_BAD_MEDIA", "source", "not a video file", {"path": item})
    if item == "b.mp4":
        raise DistillError("E_LOCKED", "youtube", "another run holds this bundle")
    raise RuntimeError("a fault three modules down")


def test_batch_item_errors_preserve_the_code_of_each_failure() -> None:
    """FAILS FIRST: `errors.append({item_key: item, "message": str(exc)})`."""
    from distill.pipeline import BatchRunner

    runner = BatchRunner(
        job_id="distill-batch",
        tool="process_video_directory",
        item_key="path",
        items=["a.mp4", "b.mp4", "c.mp4"],
        continue_on_error=True,
    )

    _results, errors = runner.run(two_items_failing_differently)

    assert [error["code"] for error in errors] == ["E_BAD_MEDIA", "E_LOCKED", "E_INTERNAL"]


def test_batch_item_errors_preserve_the_stage_of_each_failure() -> None:
    """FAILS FIRST: the stage was never written into the batch report at all."""
    from distill.pipeline import BatchRunner

    runner = BatchRunner(
        job_id="distill-batch",
        tool="process_video_directory",
        item_key="path",
        items=["a.mp4", "b.mp4", "c.mp4"],
        continue_on_error=True,
    )

    _results, errors = runner.run(two_items_failing_differently)

    assert [error["stage"] for error in errors] == ["source", "youtube", "internal"]


def test_a_batch_item_error_carries_the_whole_fatal_error_record() -> None:
    """The record a batch reports is the record every other surface reports.

    Including `details` and the item's own index: a report a caller has to
    correlate by position is a report that cannot be correlated at all once
    `continue_on_error` has skipped some, and `results` already carry
    `batch_index`.
    """
    from distill.pipeline import BatchRunner

    runner = BatchRunner(
        job_id="distill-batch",
        tool="process_video_directory",
        item_key="path",
        items=["a.mp4", "b.mp4"],
        continue_on_error=True,
    )

    _results, errors = runner.run(two_items_failing_differently)

    assert errors == [
        {
            "path": "a.mp4",
            "batch_index": 1,
            "code": "E_BAD_MEDIA",
            "stage": "source",
            "message": "not a video file",
            "details": {"path": "a.mp4"},
        },
        {
            "path": "b.mp4",
            "batch_index": 2,
            "code": "E_LOCKED",
            "stage": "youtube",
            "message": "another run holds this bundle",
            "details": {},
        },
    ]


def test_a_batch_that_stops_on_the_first_error_still_raises_it() -> None:
    """`continue_on_error=False` is unchanged: the item's error ends the batch.

    Recording the failure in the report and re-raising it are different
    questions, and only the first one is what R-46 changes.
    """
    from distill.pipeline import BatchRunner

    runner = BatchRunner(
        job_id="distill-batch",
        tool="process_video_directory",
        item_key="path",
        items=["a.mp4", "b.mp4"],
        continue_on_error=False,
    )

    with pytest.raises(DistillError) as error_info:
        runner.run(two_items_failing_differently)

    assert error_info.value.code == "E_BAD_MEDIA"
