"""The CLI's error boundary: every exception leaves as the JSON error object (R-46).

Finding 14 is what these cover. `cli.main` caught `DistillError` and nothing
else, so anything Distill had not already coded - an unreadable **job record**,
a malformed `--args`, a **fatal error** the filesystem raised - left the process
as a Python traceback on stderr with exit 1. A traceback is not a contract: an
operator scripting Distill cannot branch on it, the exit code says only "some
python died", and the stack names Distill's internals to whoever ran the
command.

What the boundary owes, stated once because every test here is one half of it:
the **fatal error** record (`code`, `stage`, `message`, `details`) as JSON on
stderr, exit code 2, and no traceback text on either stream.

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
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from distill.cli import build_parser, main
from distill.errors import DistillError

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
    """
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    unreadable.chmod(0o000)
    try:
        with pytest.raises(SystemExit) as exit_info:
            main(["cache-doctor", "--output-dir", str(unreadable / "root")])
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
    assert "usage: distill" in output.err
    # And *only* the usage message. A boundary that had swallowed the parser's
    # own `SystemExit` would answer a typo with an error object naming
    # `SystemExit`, which says nothing about how to spell the command.
    assert '"code"' not in output.err


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
    assert "File \"" not in json.dumps(payload)


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
