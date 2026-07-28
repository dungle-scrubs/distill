"""R-41: a **warning** that happened N times is one record saying so.

Eighty keyframes that each failed the same way are one finding, not eighty.
The aggregated record with its count *is* the warning carried, so ADR-0002's
promise that every warning reaches the **manifest** is kept by the count
rather than by the repetition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_local_integration import fake_transcribe, make_short_screencast

from distill import pipeline as distill_session
from distill.artifacts import FrameArtifact
from distill.errors import aggregate_warnings, warning
from distill.local_vision import (
    FrameInterpreter,
    LocalVisionConfig,
    LocalVisionFailure,
    LocalVisionProbe,
)
from distill.options import DistillOptions
from distill.progress import ProgressReporter


def test_a_warning_records_one_occurrence_from_the_start() -> None:
    """A record carries its count whether or not anything repeated.

    A shape that only grows the field when it happens twice makes every reader
    of a **manifest** write the `.get("occurrences", 1)` this saves them.
    """
    assert warning("ocr", "ocr_failed", "nope") == {
        "stage": "ocr",
        "code": "ocr_failed",
        "message": "nope",
        "occurrences": 1,
    }


def test_warnings_sharing_a_stage_and_code_become_one_record() -> None:
    repeated = [warning("local_vision", "local_vision_timeout", "timed out")] * 3

    assert aggregate_warnings(repeated) == [
        {
            "stage": "local_vision",
            "code": "local_vision_timeout",
            "message": "timed out",
            "occurrences": 3,
        }
    ]


def test_a_shared_code_under_a_different_stage_stays_its_own_record() -> None:
    """The key is the pair. A stage is where a warning happened, and two stages
    failing the same way are two things to fix."""
    aggregated = aggregate_warnings(
        [
            warning("ocr", "command_output_truncated", "ocr said too much"),
            warning("transcript", "command_output_truncated", "ffmpeg said too much"),
            warning("ocr", "command_output_truncated", "ocr said too much again"),
        ]
    )

    assert [(item["stage"], item["occurrences"]) for item in aggregated] == [
        ("ocr", 2),
        ("transcript", 1),
    ]


def test_the_first_message_is_kept_and_first_appearance_orders_the_result() -> None:
    """What survives is the first of each, in the order they first happened.

    Not the last: the first failure is the one with the context that explains
    the rest, and a reader scanning a manifest reads it in run order.
    """
    aggregated = aggregate_warnings(
        [
            warning("local_vision", "local_vision_timeout", "frame 3 timed out"),
            warning("frames", "frame_extract_failed", "frame 4 was not written"),
            warning("local_vision", "local_vision_timeout", "frame 5 timed out"),
        ]
    )

    assert [item["message"] for item in aggregated] == [
        "frame 3 timed out",
        "frame 4 was not written",
    ]
    assert [item["occurrences"] for item in aggregated] == [2, 1]


def test_an_already_aggregated_record_contributes_its_whole_count() -> None:
    """Aggregation composes, because it runs more than once on the way out.

    A stage aggregates what it produced and the run aggregates every stage's,
    so a record arriving with a count of 3 must add 3 rather than 1 - otherwise
    the second pass would silently discard what the first one counted.
    """
    aggregated = aggregate_warnings(
        [
            warning("local_vision", "local_vision_timeout", "timed out", occurrences=3),
            warning("local_vision", "local_vision_timeout", "timed out again", occurrences=5),
        ]
    )

    assert aggregated == [
        {
            "stage": "local_vision",
            "code": "local_vision_timeout",
            "message": "timed out",
            "occurrences": 8,
        }
    ]


def test_a_record_that_never_carried_a_count_is_counted_as_one() -> None:
    """A warning built as a bare mapping still aggregates."""
    aggregated = aggregate_warnings(
        [
            {"stage": "ocr", "code": "ocr_failed", "message": "nope"},
            {"stage": "ocr", "code": "ocr_failed", "message": "nope"},
        ]
    )

    assert aggregated == [
        {"stage": "ocr", "code": "ocr_failed", "message": "nope", "occurrences": 2}
    ]


def _available_probe(config: LocalVisionConfig) -> LocalVisionProbe:
    return LocalVisionProbe(
        available=True,
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        code="local_vision_available",
        message="available",
        detail={},
    )


def test_the_interpreters_repeated_transport_failures_reach_the_caller_as_one(
    tmp_path: Path,
) -> None:
    """The vision pass hands back an account, not a transcript of its attempts."""
    timeout = LocalVisionFailure("local_vision_timeout", "Local vision timed out.")

    def dead_server(
        _config: LocalVisionConfig,
        _image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[None, dict[str, Any]]:
        return None, timeout.warning()

    frames = []
    for index in range(20):
        image = tmp_path / f"frame{index}.png"
        image.write_bytes(b"png")
        frames.append(
            FrameArtifact(
                index=index + 1,
                timestamp_sec=float(index),
                path=str(image),
                relative_path=f"frames/{image.name}",
                extracted_text="on-screen text",
            )
        )

    interpreter = FrameInterpreter(
        LocalVisionConfig(),
        probe=_available_probe,
        try_interpret=dead_server,
    )
    _interpreted, warnings = interpreter.interpret(frames)

    assert [(item["code"], item["occurrences"]) for item in warnings] == [
        ("local_vision_timeout", 3),
        ("local_vision_transport_breaker_open", 1),
    ]


def test_a_published_manifest_carries_one_record_per_stage_and_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The run's own fold, at the point the **manifest** is written.

    A stage can only aggregate what it produced; only the run sees every
    stage's warnings together, so this is the fold that decides what a reader
    of a published **generation** actually gets.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    def noisy_vision(
        frames: list[FrameArtifact],
        _options: DistillOptions,
        _progress: ProgressReporter,
    ) -> tuple[list[FrameArtifact], list[dict[str, Any]]]:
        return frames, [
            *[warning("local_vision", "local_vision_timeout", "frame timed out")] * 5,
            *[warning("local_vision", "frame_text_ungrounded", "nothing to read")] * 2,
        ]

    monkeypatch.setattr(distill_session, "interpret_frames_with_local_vision", noisy_vision)

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": True,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    manifest = json.loads(Path(response["manifest_path"]).read_text())
    vision = [item for item in manifest["warnings"] if item["stage"] == "local_vision"]
    assert [(item["code"], item["occurrences"]) for item in vision] == [
        ("local_vision_timeout", 5),
        ("frame_text_ungrounded", 2),
    ]
    assert vision == [item for item in response["warnings"] if item["stage"] == "local_vision"]
    # And the reader of the **render** is told the number too, rather than
    # being shown one timeout and left to assume it happened once.
    markdown = Path(response["markdown_path"]).read_text()
    assert "occurrences: 5" in markdown
