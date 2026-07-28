from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from distill import frame_selection
from distill.capabilities import EXTERNAL_TOOLS
from distill.errors import DistillError
from distill.frame_selection import filtered_candidates, hamming_distance, select_keyframes
from distill.progress import ProgressReporter
from distill.run_command import (
    OUTPUT_CAP_BYTES,
    TRUNCATION_WARNING_CODE,
    CommandTimeouts,
)


def test_filtered_candidates_respects_interval_and_static_floor() -> None:
    assert filtered_candidates([0, 1, 5], 20, 4, 10) == [0, 5, 15]


def test_static_floor_compares_against_last_force_kept_frame() -> None:
    assert filtered_candidates([0, 9, 18], 20, 4, 10) == [0, 9, 18]


def test_hamming_distance_for_phash_hex() -> None:
    assert hamming_distance("0f", "00") == 4


def test_scene_midpoint_candidates_falls_back_to_adaptive_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTimecode:
        def __init__(self, seconds: float) -> None:
            self.seconds = seconds

        def get_seconds(self) -> float:
            return self.seconds

    class FakeContentDetector:
        pass

    class FakeAdaptiveDetector:
        pass

    def fake_detect(_path: str, detector: object) -> list[tuple[Any, Any]]:
        if isinstance(detector, FakeContentDetector):
            return []
        return [(FakeTimecode(1.0), FakeTimecode(3.0))]

    monkeypatch.setitem(
        __import__("sys").modules,
        "scenedetect",
        type(
            "FakeSceneDetect",
            (),
            {
                "ContentDetector": FakeContentDetector,
                "AdaptiveDetector": FakeAdaptiveDetector,
                "detect": staticmethod(fake_detect),
            },
        ),
    )

    assert frame_selection.scene_midpoint_candidates(Path("demo.mp4"), 10.0) == [2.0]


# A fake ffmpeg that writes the single frame it was asked for, and one that
# fails the way a bad seek does. Both are real executables: run_command spawns a
# real child in its own process group, which no patched call can stand in for.
FAKE_FFMPEG_WRITES_FRAME = """
import pathlib, sys

pathlib.Path(sys.argv[-1]).write_bytes(b"png")
"""

FAKE_FFMPEG_FAILS = """
import sys

sys.stderr.write("Output file #0 does not contain any stream\\n")
sys.exit(1)
"""


def test_absent_ffmpeg_ends_the_run_because_the_table_says_it_is_required(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """ADR-0002 / R-34: a missing **required capability** is a **fatal error**.

    Nothing is installed into the fake PATH, so ffmpeg is genuinely absent and
    run_command's real E_MISSING_TOOL is what reaches the call site. The call
    site does not decide what that costs - it hands the tool to the capability
    table, which classifies ffmpeg required, so the run ends here naming the
    tool rather than degrading and dying later at render under E_NO_CONTENT.
    """
    with pytest.raises(DistillError) as failure:
        frame_selection.extract_frame(Path("demo.mp4"), 1.0, tmp_path / "frame.png")

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.stage == "frames"
    assert failure.value.message.startswith("ffmpeg is not installed or not on PATH; ")
    assert EXTERNAL_TOOLS["ffmpeg"].absence_cost in failure.value.message
    assert failure.value.details["requirement"] == "required"


def test_extract_frame_degrades_when_ffmpeg_fails(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """A non-zero exit reduces the bundle by one keyframe and continues.

    ffmpeg being *required* is about ffmpeg being installed. An installed
    ffmpeg that could not decode one frame costs that keyframe, not the run.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_FAILS)

    extracted, warnings = frame_selection.extract_frame(
        Path("demo.mp4"), 1.5, tmp_path / "frame.png"
    )

    assert extracted is False
    assert warnings == [
        {
            "stage": "frames",
            "code": "frame_extract_failed",
            "message": "could not extract frame at 1.500s",
            "occurrences": 1,
        }
    ]


def test_extract_frame_emits_a_boundary_event(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """R-29: the invocation is visible at the boundary, named by its stage."""
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_FRAME)

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        assert frame_selection.extract_frame(Path("demo.mp4"), 1.0, tmp_path / "f.png") == (
            True,
            [],
        )

    events = [json.loads(record.message) for record in caplog.records]
    assert [(event["detail"]["tool"], event["detail"]["stage"]) for event in events] == [
        ("ffmpeg", "frames")
    ]


def test_frame_selection_reports_scene_and_candidate_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        frame_selection,
        "scene_midpoint_candidates",
        lambda _path, _duration: [0.0, 5.0],
    )

    def fake_extract_frame(
        _video: Path, _timestamp: float, output: Path
    ) -> tuple[bool, list[dict[str, str]]]:
        output.write_bytes(b"png")
        return True, []

    monkeypatch.setattr(frame_selection, "extract_frame", fake_extract_frame)
    monkeypatch.setattr(frame_selection, "phash", lambda _path: "0f")
    progress = ProgressReporter()

    frames, warnings = select_keyframes(
        Path("demo.mp4"),
        tmp_path,
        duration_sec=10.0,
        max_keyframes=2,
        min_interval_sec=1.0,
        max_static_window_sec=90.0,
        progress=progress,
    )

    assert warnings == []
    assert len(frames) == 1
    assert any(
        event.mechanism == "scene_detection" and event.status == "completed"
        for event in progress.events
    )
    assert any(
        event.mechanism == "frame_extraction" and event.percent == 100.0
        for event in progress.events
    )


# A fake ffmpeg that produces the frame and floods stderr past the capture cap
# while it does. ffmpeg is genuinely chatty on stderr, so this is the shape of a
# real invocation that both succeeds and loses part of its own record.
FAKE_FFMPEG_FLOODS_STDERR = f"""
import pathlib, sys

pathlib.Path(sys.argv[-1]).write_bytes(b"png")
sys.stderr.write("x" * ({OUTPUT_CAP_BYTES} + 1024))
"""


def test_a_truncated_frame_grab_still_yields_its_keyframe(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-33: the invocation's own **warning** reaches the caller, and only that.

    Truncation is not failure. A caller reading one warning per grab would
    either drop the keyframe it did produce or drop the warning; both answers
    have to survive, which is why `extract_frame` returns them separately.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_FLOODS_STDERR)
    monkeypatch.setattr(frame_selection, "scene_midpoint_candidates", lambda _p, _d: [0.0])
    monkeypatch.setattr(frame_selection, "phash", lambda _path: "0f")

    frames, warnings = select_keyframes(
        Path("demo.mp4"),
        tmp_path,
        duration_sec=1.0,
        max_keyframes=1,
        min_interval_sec=1.0,
        max_static_window_sec=90.0,
    )

    assert len(frames) == 1
    assert [item["code"] for item in warnings] == [TRUNCATION_WARNING_CODE]


# A fake ffmpeg that is installed and then hangs: the tool is present, so this
# is not a capability question at all.
FAKE_FFMPEG_HANGS = """
import time

time.sleep(600)
"""


def test_a_wedged_but_installed_ffmpeg_still_only_costs_its_keyframe(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The other half of the split: required is about installed, not about healthy.

    ffmpeg is classified a **required capability**, and it is here - it just
    cannot decode this frame in time. That reduces the **bundle** by one
    keyframe, exactly as it did before the classification was enforced.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_HANGS)
    monkeypatch.setattr(
        frame_selection, "FRAME_EXTRACT_TIMEOUTS", CommandTimeouts(total_sec=0.3, idle_sec=0.3)
    )

    extracted, warnings = frame_selection.extract_frame(
        Path("demo.mp4"), 1.0, tmp_path / "frame.png"
    )

    assert extracted is False
    assert [item["code"] for item in warnings] == ["frame_extract_timeout"]
