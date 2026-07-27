from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from distill import frame_selection
from distill.frame_selection import filtered_candidates, hamming_distance, select_keyframes
from distill.progress import ProgressReporter


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


def test_extract_frame_degrades_when_ffmpeg_missing(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """ADR-0002: a keyframe that cannot be grabbed is a **warning**, not a raise.

    Nothing is installed into the fake PATH, so ffmpeg is genuinely absent and
    the degradation is driven by run_command's real E_MISSING_TOOL.
    """
    result = frame_selection.extract_frame(Path("demo.mp4"), 1.0, tmp_path / "frame.png")

    assert result == {
        "stage": "frames",
        "code": "ffmpeg_missing",
        "message": "ffmpeg is not installed; skipping keyframe extraction",
    }


def test_extract_frame_degrades_when_ffmpeg_fails(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """A non-zero exit reduces the bundle by one keyframe and continues."""
    fake_tool("ffmpeg", FAKE_FFMPEG_FAILS)

    result = frame_selection.extract_frame(Path("demo.mp4"), 1.5, tmp_path / "frame.png")

    assert result == {
        "stage": "frames",
        "code": "frame_extract_failed",
        "message": "could not extract frame at 1.500s",
    }


def test_extract_frame_emits_a_boundary_event(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """R-29: the invocation is visible at the boundary, named by its stage."""
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_FRAME)

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        assert frame_selection.extract_frame(Path("demo.mp4"), 1.0, tmp_path / "f.png") is None

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

    def fake_extract_frame(_video: Path, _timestamp: float, output: Path) -> None:
        output.write_bytes(b"png")
        return None

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
