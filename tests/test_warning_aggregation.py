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
from distill.artifacts import FrameArtifact, Interpretation
from distill.bundle_store import BundleRun, BundleStore
from distill.errors import aggregate_warnings, warning
from distill.local_vision import (
    FrameInterpreter,
    LocalVisionConfig,
    LocalVisionFailure,
    LocalVisionProbe,
)
from distill.options import DistillOptions
from distill.progress import ProgressReporter
from distill.redact_secrets import redact_text
from distill.response import manifest_document
from distill.source import SourceInfo


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


def test_a_count_survives_the_stage_result_a_resume_reads_it_out_of(
    tmp_path: Path,
) -> None:
    """The count is a number on disk, and the run that resumes folds it again.

    A **stage result** is written as JSON and read back by a later run, which
    then folds every stage's warnings a second time. Two things have to hold
    for a resumed run to publish the same **manifest** as an uninterrupted one:
    the count must survive serialization as a number rather than as the string
    a permissive encoder would make of it, and the second fold must add it
    rather than reset it to one - a resumed 80-keyframe run would otherwise
    report three timeouts as one.
    """
    root = tmp_path / "output"
    root.mkdir()
    run = BundleStore.open(root).begin("b0a1c2d3")
    assert isinstance(run, BundleRun)
    folded = warning("local_vision", "local_vision_timeout", "frame 3 timed out", occurrences=3)

    run.write_stage("local_vision", {"frames": [], "warnings": [folded]})
    recorded = run.read_stage("local_vision")

    assert recorded is not None
    assert recorded["warnings"] == [folded]
    assert recorded["warnings"][0]["occurrences"] == 3
    assert not isinstance(recorded["warnings"][0]["occurrences"], str)
    assert aggregate_warnings(recorded["warnings"])[0]["occurrences"] == 3


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


def test_a_folded_warning_is_counted_by_what_it_says_happened(tmp_path: Path) -> None:
    """The interpreter's own tally reads the count, not the record.

    A carrier hands back **warnings** the **redaction** policy already folded,
    so one record can stand for four confusable matches. Counting records
    rather than occurrences made `debug_info` disagree with the **manifest**
    about the same run.
    """
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    confusable = " ".join(["ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef"] * 4)

    def reader(
        config: LocalVisionConfig,
        _image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        return (
            Interpretation(
                visual_summary=confusable,
                detected_elements=(),
                interpretation="a screen holding four secret-shaped values",
                uncertainty="Low",
                backend=config.backend,
                model=config.model,
                prompt_profile=prompt_profile,
                verbatim_text="on-screen text",
                text_confidence="high",
            ),
            None,
        )

    frame = FrameArtifact(
        index=1,
        timestamp_sec=0.0,
        path=str(image),
        relative_path="frames/frame.png",
        extracted_text="on-screen text",
    )
    interpreter = FrameInterpreter(
        LocalVisionConfig(), probe=_available_probe, try_interpret=reader
    )
    _frames, warnings = interpreter.interpret([frame])

    folded = [item for item in warnings if item["code"] == "possible_confusable_secret"]
    assert [item["occurrences"] for item in folded] == [4]
    assert interpreter.debug_info()["warning_counts"]["possible_confusable_secret"] == 4


def test_two_hundred_confusable_matches_are_one_record_saying_two_hundred() -> None:
    """The count is exact and is not built one record at a time.

    The cap R-41 removed existed because a record per match is a list nobody
    can read; a record per match built and then folded is the same list, held
    in memory first. Text Distill did not write decides how many matches there
    are, so the count is counted rather than accumulated.
    """
    text = " ".join(["ｓｋ-abcdefghijklmnopqrstuvwxyzabcdef"] * 200)

    result = redact_text(text)

    assert [(item["code"], item["occurrences"]) for item in result.warnings] == [
        ("possible_confusable_secret", 200)
    ]


def test_the_manifests_warning_count_counts_events_and_not_records() -> None:
    """R-41 did not get to change what `warning_count` means.

    The field is named for how many warnings a run raised, and a reader who
    sees `2` for a run that timed out eighty times has been told the run went
    well. Folding the list made the record count a count of *kinds*, and the
    two answers were the same number only before the fold existed.
    """
    document = manifest_document(
        SourceInfo(
            source_type="local",
            resolved_path=Path("/tmp/video.mp4"),
            duration_sec=1.0,
            source_fingerprint="fingerprint",
            source_hash="bundle-key",
            warnings=[],
            related_links=None,
        ),
        DistillOptions(),
        transcript_present=False,
        frames=[],
        warnings=aggregate_warnings(
            [
                *[warning("local_vision", "local_vision_timeout", "timed out")] * 80,
                *[warning("ocr", "ocr_failed", "nope")] * 3,
            ]
        ),
    )

    assert len(document["warnings"]) == 2
    assert document["warning_count"] == 83


def test_a_manifest_warning_without_a_count_still_counts_as_one() -> None:
    """A **warning** built as a bare mapping is an event like any other.

    `aggregate_warnings` already tolerates a record with no `occurrences` -
    a probe's, a capability's - so the count over the folded list has to
    tolerate the same shape rather than reading a missing field as zero.
    """
    document = manifest_document(
        SourceInfo(
            source_type="local",
            resolved_path=Path("/tmp/video.mp4"),
            duration_sec=1.0,
            source_fingerprint="fingerprint",
            source_hash="bundle-key",
            warnings=[],
            related_links=None,
        ),
        DistillOptions(),
        transcript_present=False,
        frames=[],
        warnings=[{"stage": "capability", "code": "ffmpeg_missing", "message": "nope"}],
    )

    assert document["warning_count"] == 1
