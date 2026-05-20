from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]


def test_package_files_exist() -> None:
    for relative in [
        "pyproject.toml",
        "README.md",
        "src/saccade/cli.py",
        "src/saccade/pipeline.py",
    ]:
        assert (APP / relative).exists()


def test_list_tools_returns_both_tools() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "saccade.cli",
            "list-tools",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(APP / "src")},
    )
    assert json.loads(result.stdout) == {
        "tools": [
            "cleanup_cache",
            "get_job_status",
            "process_local_video",
            "process_video_directory",
            "process_youtube_playlist",
            "process_youtube_video",
        ]
    }


def test_timeout_config_is_90_minutes() -> None:
    from saccade.pipeline import configured_timeout_ms

    assert configured_timeout_ms() == 5_400_000


def test_playlist_cli_accepts_child_processing_options() -> None:
    from saccade.cli import build_parser

    args = build_parser().parse_args(
        [
            "process-youtube-playlist",
            "https://www.youtube.com/playlist?list=PLQHpFq3RA7fEJ0z3DABwTPvwre0Vu6OBH",
            "--no-ocr",
            "--no-caption-frames",
            "--whisper-model",
            "base",
        ]
    )

    assert args.command == "process-youtube-playlist"
    assert args.ocr is False
    assert args.caption_frames is False
    assert args.whisper_model == "base"


def test_timeout_diagnostics_report_configured_and_effective_timeout() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "saccade.cli",
            "timeout-diagnostics",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PYTHONPATH": str(APP / "src"),
            "TOOL_PROXY_EFFECTIVE_TIMEOUT_MS": "5400000",
        },
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "assumption": "A-004",
        "configured_timeout_ms": 5_400_000,
        "effective_meets_configured": True,
        "effective_timeout_ms": 5_400_000,
        "effective_timeout_source": "TOOL_PROXY_EFFECTIVE_TIMEOUT_MS",
    }


def test_timeout_probe_is_fast_by_default() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "saccade.cli",
            "timeout-probe",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(APP / "src")},
    )
    payload = json.loads(result.stdout)
    assert payload["probe_requested_ms"] == 1
    assert payload["long_probe_enabled"] is False
    assert payload["effective_meets_configured"] is True


def test_long_timeout_probe_requires_explicit_opt_in() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "saccade.cli",
            "timeout-probe",
            "1001",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(APP / "src")},
    )
    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["code"] == "E_BAD_ARGUMENT"
    assert payload["stage"] == "timeout"
