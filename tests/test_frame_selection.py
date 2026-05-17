from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from saccade import frame_selection
from saccade.frame_selection import filtered_candidates, hamming_distance, select_keyframes
from saccade.progress import ProgressReporter


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
