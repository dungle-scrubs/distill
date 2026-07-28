"""Tests for the CLI dispatch branches in ``distill.cli.main``.

These exercise the subcommand dispatch in ``main()`` directly (in-process),
complementing the subprocess-based scaffold tests. They cover the branches that
can run hermetically: the diagnostic/list commands, cache cleanup against a
temp directory, job-status lookup, and the call-tool wrapper, plus the
DistillError-to-stderr path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.cli import main
from distill.errors import DistillError


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[object, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_list_tools_dispatch_prints_sorted_tools(capsys: pytest.CaptureFixture[str]) -> None:
    """R-57 adds the read-only inspection command to the tool surface.

    `cleanup_cache` keeps its name: D-042 renames no command, so the public
    surface gains the surface that reports and keeps the one that deletes.
    """
    code, out, _ = _run(["list-tools"], capsys)
    assert code is None
    payload = json.loads(out)
    assert payload == {
        "tools": [
            "cache_doctor",
            "cleanup_cache",
            "get_job_status",
            "process_local_video",
            "process_video_directory",
            "process_youtube_playlist",
            "process_youtube_video",
        ]
    }


def test_timeout_diagnostics_dispatch_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["timeout-diagnostics"], capsys)
    assert code is None
    payload = json.loads(out)
    assert payload["configured_timeout_ms"] == 5_400_000
    assert payload["assumption"] == "A-004"


def test_timeout_probe_dispatch_short_probe(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["timeout-probe", "1"], capsys)
    assert code is None
    payload = json.loads(out)
    assert payload["probe_requested_ms"] == 1
    assert payload["long_probe_enabled"] is False


def test_timeout_probe_dispatch_rejects_long_probe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["timeout-probe", "1001"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    payload = json.loads(err)
    assert payload["code"] == "E_BAD_ARGUMENT"
    assert payload["stage"] == "timeout"


def test_cleanup_cache_dispatch_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """R-57: a preview reports what it skipped, not only what it would delete.

    "Considered nothing" and "deleted nothing" are different answers, and a
    payload carrying only `deleted` cannot tell them apart - here the root holds
    one directory that is not a **bundle**, so the empty candidate list has to
    come with the reason it is empty.
    """
    (tmp_path / "notes").mkdir()
    code, out, _ = _run(
        ["cleanup-cache", "--output-dir", str(tmp_path), "--keep-generations", "3", "--dry-run"],
        capsys,
    )
    assert code is None
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert payload["candidate_count"] == 0
    assert payload["deleted"] == []
    assert payload["considered"] == 1
    assert payload["skipped"] == [
        {
            "path": str(tmp_path / "notes"),
            "verdict": "absent",
            "reason": "no bundle marker",
        }
    ]


def test_get_job_status_dispatch_missing_job_is_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["get-job-status", "no-such-job", "--output-dir", str(tmp_path)])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    payload = json.loads(err)
    assert payload["code"] == "E_JOB_NOT_FOUND"
    assert payload["stage"] == "job"


def test_call_tool_dispatch_reports_an_unknown_tool_as_a_fatal_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R-46: `call-tool` is a command, so a failed call is a failed command.

    Classified *defect* under R-46 - the old test pinned the `DistillError`
    being swallowed into a stdout `{"error": ...}` envelope with exit 0, so a
    script calling Distill through `call-tool` saw success for every failure and
    had to inspect the payload to find out otherwise. The session still speaks
    the envelope, which is its contract; the CLI does not.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["call-tool", "does_not_exist", "--args", "{}"])
    assert exc_info.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    payload = json.loads(output.err)
    assert payload["code"] == "E_UNKNOWN_TOOL"
    assert payload["stage"] == "protocol"


def test_call_tool_dispatch_refuses_a_job_id_outside_the_domain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R-18: an empty identifier names no **job record**, so it is refused.

    Classified *defect* under R-18 - the old assertion was
    `"result" in payload or "error" in payload`, which every possible outcome
    satisfies, over an empty `job_id` that R-18 makes a bounded-domain
    rejection. Both halves are pinned now: which answer, and through which exit.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "call-tool",
                "get_job_status",
                "--args",
                json.dumps({"job_id": "", "output_dir": str(tmp_path)}),
            ]
        )
    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "E_BAD_JOB_ID"
    assert payload["stage"] == "job"


def test_local_vision_diagnostics_dispatch_runs_with_every_override(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command runs, rather than only parsing (D-022).

    `test_local_vision_diagnostics_cli_accepts_all_overrides` stops at the
    parser, so a key `main` reads that the subparser never registered parsed
    fine and died on dispatch. This drives the whole command with the probe
    stubbed, which is the only part of it that wants a server.
    """
    from distill.local_vision import LocalVisionProbe

    monkeypatch.setattr(
        "distill.pipeline.probe_local_vision",
        lambda config: LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="",
            message="",
            detail={},
        ),
    )
    code, out, _ = _run(
        [
            "local-vision-diagnostics",
            "--caption-frames",
            "--local-vision-backend",
            "rapid-mlx",
            "--local-vision-model",
            "mlx-community/Qwen3-VL-8B-Instruct-8bit",
            "--local-vision-base-url",
            "http://10.0.0.5:8000/v1",
            "--local-vision-timeout-sec",
            "45",
            "--local-vision-allow-remote-endpoint",
        ],
        capsys,
    )
    assert code is None
    payload = json.loads(out)
    # The opt-out reached the config rather than only the namespace: without it
    # a non-loopback base_url is a fatal E_BAD_OPTIONS before any probe runs.
    assert payload["config"]["base_url"] == "http://10.0.0.5:8000/v1"
    assert payload["probe"]["available"] is True


def test_main_propagates_distill_error_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_error(_name: str, _args: dict) -> dict:
        raise DistillError("E_TEST", "test", "boom")

    monkeypatch.setattr("distill.cli.call_registered_tool", raise_error)
    with pytest.raises(SystemExit) as exc_info:
        main(["get-job-status", "any", "--output-dir", str(tmp_path)])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert json.loads(err)["code"] == "E_TEST"
