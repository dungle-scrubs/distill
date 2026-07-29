"""Tests for what a **stage result** must prove before a **resume** believes it.

The seam under test is `BundleRun.read_stage`, the one point at which a run
takes scratch that was on disk before it started and treats it as work it does
not have to do. RV-8 is what that costs when the read trusts the file: no schema
version, so a document a newer Distill cannot parse is parsed anyway; no
**bundle key**, so scratch written under one key is reused under another; no
confinement, so a path recorded in it points wherever it likes.

R-23 fixes all three at the read, and D-030 fixes the consequence: a stage
result that fails validation is *discarded*, and its stage recomputed. That is
the whole point. A stage result is scratch by definition - the stage that
produced it can always produce it again - so ending a run over one would turn a
**resume** optimisation into an outage the first time a schema moved.

Sections 7 and 8 are the two places that consequence had not reached, both
found by the Phase 4 boundary review. A payload whose *shape* had drifted
passed R-23's envelope checks and then crashed the consumer that read a field
off it, and a stage result that could not be *written* - a symlink, a
directory, a read-only file at its path - ended the run at the very write the
recomputation was making. The second was the worse of the two: a failing run
keeps its **staging directory** and `resume_partial` defaults on, so one
stray file made a **bundle key** permanently unrunnable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from distill import pipeline
from distill.artifacts import FrameArtifact, Provenance, RedactionState
from distill.bundle_store import (
    STAGE_RESULT_SCHEMA_VERSION,
    BundleRun,
    BundleStore,
    _read_regular_file,
)
from distill.emit import EMITTER
from distill.errors import DistillError
from distill.options import DistillOptions
from distill.pipeline import ProcessingRun
from distill.progress import ProgressHeartbeat, ProgressReporter
from distill.source import SourceInfo

BUNDLE_KEY = "b0a1c2d3"
OTHER_BUNDLE_KEY = "f9e8d7c6"


def begin_run(store: BundleStore, bundle_key: str = BUNDLE_KEY) -> BundleRun:
    run = store.begin(bundle_key)
    assert isinstance(run, BundleRun)
    return run


def open_run(tmp_path: Path) -> BundleRun:
    root = tmp_path / "output"
    root.mkdir()
    return begin_run(BundleStore.open(root))


def stage_result_file(run: BundleRun, name: str) -> Path:
    return run.paths.generation / f"_{name}.json"


def recorded_document(run: BundleRun, name: str) -> dict[str, Any]:
    return json.loads(stage_result_file(run, name).read_text())


def plant(run: BundleRun, name: str, document: Any) -> None:
    """Put a stage result on disk without going through `write_stage`.

    An interrupted run's scratch is read back by a *later* run, so what
    `read_stage` actually receives is a file some other process wrote. Planting
    it directly is how a test says that - and the documents below are exactly
    the ones a bundle-key collision, a schema change, or a tampered staging
    directory would leave behind.
    """
    stage_result_file(run, name).write_text(json.dumps(document, indent=2) + "\n")


def processing_run(output_root: Path) -> ProcessingRun:
    """A `ProcessingRun` with nothing but what `_run_stage` reads.

    `_run_stage` is the caller that turns a rejected stage result into a
    recomputation, and it touches only the progress reporter. The source and
    the tool are what a full run needs to *produce* a bundle, which this is not
    doing.
    """
    return ProcessingRun(
        source=None,
        options=DistillOptions(),
        output_root=output_root,
        progress=ProgressReporter(),
        tool="test",
    )


# Which stages the fakes below were actually asked to do, newest last. A resume
# is only observable as work *not* done, so a test that cannot see the producers
# run cannot tell a recomputation from a hit.
selected: list[str] = []


def fake_transcribe(*_args: Any, **_kwargs: Any) -> tuple[None, list[dict[str, str]]]:
    selected.append("transcript")
    return None, []


def fake_select_keyframes(
    _video: Path,
    frames_dir: Path,
    *_args: Any,
    redaction: RedactionState,
    **_kwargs: Any,
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    selected.append("frames")
    image = frames_dir / "frame_0001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return [
        FrameArtifact(
            index=1,
            timestamp_sec=0.0,
            path=str(image),
            relative_path="frames/frame_0001.png",
            extracted_text="recomputed frame reading",
            redaction=redaction,
        )
    ], []


def read_frame(run: BundleRun, text: str) -> FrameArtifact:
    """A **frame artifact** an image-text stage would hand back, inside this run.

    Its path is under the run's own **staging directory**, which is what R-23's
    confinement demands of a stage result naming a file.
    """
    return FrameArtifact(
        index=1,
        timestamp_sec=0.0,
        path=str(run.frames_dir / "frame_0001.png"),
        relative_path="frames/frame_0001.png",
        extracted_text=text,
    )


def fake_ocr_frames(
    frames: list[FrameArtifact], *_args: Any
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    selected.append("ocr")
    return frames, []


def resuming_run(
    output_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ProcessingRun:
    """A real second run of `BUNDLE_KEY`, with the media work faked out.

    `execute` is the resume path in full: it takes the run lock, finds no
    **active generation**, reopens the **staging directory** an interrupted run
    left, and reads every **stage result** in it back. Only the three producers
    are fakes, because what they produce is not what is under test and a real
    one would need ffmpeg on the machine running the suite.
    """
    selected.clear()
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"not a video, and never opened")
    monkeypatch.setattr(pipeline, "transcribe_with_imports", fake_transcribe)
    monkeypatch.setattr(pipeline, "select_keyframes", fake_select_keyframes)
    monkeypatch.setattr(pipeline, "ocr_frames", fake_ocr_frames)
    return ProcessingRun(
        source=SourceInfo(
            source_type="local",
            resolved_path=video,
            duration_sec=1.0,
            source_fingerprint="fingerprint",
            source_hash=BUNDLE_KEY,
            warnings=[],
            provenance=Provenance(
                title=video.name,
                duration_sec=1.0,
                processed_at="2026-07-29T14:20:00Z",
            ),
        ),
        options=DistillOptions(caption_frames=False, job_id="job-1"),
        output_root=output_root,
        progress=ProgressReporter(),
        tool="process_local_video",
    )


# 1 and 2. A stage result carries a schema version and the bundle key it belongs to


def test_a_recorded_stage_result_carries_its_schema_version_and_bundle_key(
    tmp_path: Path,
) -> None:
    """R-23: the two facts a read has to check must be facts a write recorded.

    Both live beside the payload rather than inside it: a stage decides what its
    payload holds, and neither the schema nor the **bundle key** is a stage's to
    choose.
    """
    run = open_run(tmp_path)

    run.write_stage("ocr", {"frames": [], "warnings": []})

    document = recorded_document(run, "ocr")
    assert document["schema_version"] == STAGE_RESULT_SCHEMA_VERSION
    assert document["bundle_key"] == BUNDLE_KEY
    assert document["stage"] == "ocr"
    assert document["payload"] == {"frames": [], "warnings": []}
    # Read back through the seam, a caller still gets the payload it recorded:
    # the envelope is the store's, not the stage's.
    assert run.read_stage("ocr") == {"frames": [], "warnings": []}


# 3. FAILS FIRST: a stage result whose bundle key does not match is rejected (RV-8)


def test_a_stage_result_bound_to_another_bundle_key_is_rejected(tmp_path: Path) -> None:
    """RV-8: scratch recorded under one **bundle key** was reused under another.

    A **bundle key** is a **source fingerprint** and an **options hash**
    together, so a stage result carrying a different one describes different
    media, different processing choices, or both. Read back it is not stale - it
    is someone else's answer, and reusing it publishes a **generation** for a
    source that never produced it.
    """
    run = open_run(tmp_path)

    plant(
        run,
        "ocr",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": OTHER_BUNDLE_KEY,
            "stage": "ocr",
            "payload": {"frames": [{"ocr_text": "another bundle's screen"}]},
        },
    )

    assert run.read_stage("ocr") is None


# 4. FAILS FIRST: a stage result containing a path outside the bundle root is rejected


def test_a_stage_result_naming_a_path_outside_the_bundle_root_is_rejected(
    tmp_path: Path,
) -> None:
    """R-23: a path read back out of scratch is a path this run will act on.

    The **frame artifacts** in a stage result name image files, and a resumed
    run reads them, copies them and records them in the **manifest**. A path
    that resolves outside the bundle root makes the resume a reader of whatever
    it points at.
    """
    run = open_run(tmp_path)
    outside = tmp_path / "elsewhere" / "secret.png"

    plant(
        run,
        "frames",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": BUNDLE_KEY,
            "stage": "frames",
            "payload": {
                "frames": [
                    {
                        "index": 0,
                        "path": str(outside),
                        "relative_path": "frames/frame_0000.png",
                    }
                ]
            },
        },
    )

    assert run.read_stage("frames") is None


def test_a_stage_result_naming_paths_inside_the_bundle_root_is_kept(tmp_path: Path) -> None:
    """The confinement has to admit the paths a real run records, or it is a delete.

    A run records an absolute path into its own **staging directory** and a path
    relative to the generation. Both are inside the bundle root, and rejecting
    either would mean no stage result is ever reusable - which passes a
    confinement test and removes **resume**.
    """
    run = open_run(tmp_path)
    inside = run.paths.frames / "frame_0000.png"

    payload = {
        "frames": [
            {
                "index": 0,
                "path": str(inside),
                "relative_path": "frames/frame_0000.png",
            }
        ]
    }
    run.write_stage("frames", payload)

    assert run.read_stage("frames") == payload


# 5. An unknown schema version is rejected rather than partially parsed


def test_an_unknown_schema_version_is_rejected_rather_than_partially_parsed(
    tmp_path: Path,
) -> None:
    """R-23: the version is the reason not to guess, so guessing defeats it.

    A document from a schema this code does not know may use the same field
    names for different things. Reading the fields it recognizes and ignoring
    the rest is how a schema change becomes silent corruption, so an unknown
    version is refused whole.
    """
    run = open_run(tmp_path)

    plant(
        run,
        "ocr",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION + 1,
            "bundle_key": BUNDLE_KEY,
            "stage": "ocr",
            "payload": {"frames": [{"ocr_text": "readable, and not to be read"}]},
        },
    )

    assert run.read_stage("ocr") is None


def test_a_schema_version_that_is_not_an_integer_is_rejected(tmp_path: Path) -> None:
    """`True` is an instance of `int` and equals 1, so equality alone is not the check.

    Scratch corrupted into `"schema_version": true` would otherwise read as
    version 1 - the one version this code trusts completely.
    """
    run = open_run(tmp_path)

    for version in (True, "1", None, [1]):
        plant(
            run,
            "ocr",
            {
                "schema_version": version,
                "bundle_key": BUNDLE_KEY,
                "stage": "ocr",
                "payload": {"frames": []},
            },
        )
        assert run.read_stage("ocr") is None, f"schema_version {version!r} was accepted"


def test_a_stage_result_with_no_schema_version_at_all_is_rejected(tmp_path: Path) -> None:
    """The pre-R-23 document is an unknown version, not a default one.

    Every stage result written before this validation existed is a bare payload
    with no envelope. Treating a missing version as version 1 would trust
    exactly the documents that were never checked (D-015: no deprecation path is
    owed, and one recomputation is the whole cost).
    """
    run = open_run(tmp_path)

    plant(run, "ocr", {"frames": [{"ocr_text": "recorded by an older Distill"}]})

    assert run.read_stage("ocr") is None


def test_a_document_that_cannot_be_validated_at_all_is_a_miss_not_a_failure(
    tmp_path: Path,
) -> None:
    """D-030 covers the validation itself, not only the verdicts it reaches.

    A stage result is written by some other process, so its shape is input, and
    the walk that confines its paths is recursive. A payload nested past
    Python's recursion limit exhausts that walk - and a validator that let the
    `RecursionError` escape would end the run on the one file that exists so the
    run does not have to end.

    Written as text rather than built as a value, because the depth that breaks
    the validator would break the test's own encoder first.
    """
    run = open_run(tmp_path)
    depth = sys.getrecursionlimit() * 2
    nested = "[" * depth + "]" * depth
    stage_result_file(run, "ocr").write_text(
        f'{{"schema_version": {STAGE_RESULT_SCHEMA_VERSION}, "bundle_key": "{BUNDLE_KEY}",'
        f' "stage": "ocr", "payload": {{"frames": {nested}}}}}'
    )

    assert run.read_stage("ocr") is None


# 6. Rejection triggers recomputation, not run failure


def test_a_rejected_stage_result_recomputes_its_stage_rather_than_failing_the_run(
    tmp_path: Path,
) -> None:
    """D-030, and the point of the whole milestone.

    Refusing a stage result is only safe because the alternative is cheap: the
    stage runs. A run that raised instead would fail on scratch it wrote itself,
    which is worse than the trust it replaced - the failure would be
    unrecoverable without deleting the bundle by hand.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    pipeline_run = processing_run(root)

    plant(
        run,
        "ocr",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": OTHER_BUNDLE_KEY,
            "stage": "ocr",
            "payload": {"frames": [{"ocr_text": "another bundle's screen"}]},
        },
    )

    produced: list[str] = []

    def produce() -> dict[str, Any]:
        produced.append("ocr")
        return {"frames": [read_frame(run, "this bundle's screen")], "warnings": []}

    warnings: list[dict[str, str]] = []
    frames = pipeline_run._run_stage(
        run,
        ProgressHeartbeat(pipeline_run.progress.counter).start(),
        warnings,
        "ocr",
        (),
        produce,
        pipeline_run._recovered_frames,
    )

    assert produced == ["ocr"], "the rejected stage result was reused instead of recomputed"
    assert [frame.extracted_text for frame in frames] == ["this bundle's screen"]
    # The recomputation replaced the scratch, so the next resume has a stage
    # result bound to the bundle it actually belongs to.
    assert recorded_document(run, "ocr")["bundle_key"] == BUNDLE_KEY
    resumed = pipeline_run._recovered_frames(run.read_stage("ocr") or {})
    assert resumed is not None
    assert [frame.extracted_text for frame in resumed.value] == ["this bundle's screen"]


def test_a_resume_carries_the_warnings_recorded_with_the_stage_result(tmp_path: Path) -> None:
    """A **warning** describes the run that is qualified by it, which is this one.

    R-58 caps an extracted-text field when the stage result is recorded, and the
    record of that cut is on the envelope. The run that resumes is the run
    reading the shortened text, so it is the one that has to carry the warning -
    otherwise a **generation** is published from truncated text with nothing in
    the bundle saying it was truncated.
    """
    run = open_run(tmp_path)
    truncated = {
        "stage": "artifacts",
        "code": "extracted_text_truncated",
        "message": "payload.frames[0].extracted_text was truncated",
    }

    plant(
        run,
        "ocr",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": BUNDLE_KEY,
            "stage": "ocr",
            "payload": {"frames": [], "warnings": [{"stage": "ocr", "code": "x", "message": "y"}]},
            "warnings": [truncated],
        },
    )

    payload = run.read_stage("ocr")
    assert payload is not None
    assert payload["warnings"] == [{"stage": "ocr", "code": "x", "message": "y"}, truncated]


# 7. A payload whose *shape* drifted is a miss too (boundary review, finding 2)


@pytest.mark.parametrize(
    "drifted",
    [
        pytest.param({"warnings": []}, id="payload-without-frames"),
        pytest.param(
            {"frames": ["frames/frame_0001.png"], "warnings": []}, id="frames-not-documents"
        ),
        pytest.param(
            {"frames": [{"index": 1, "timestamp_sec": 0.0}], "warnings": []},
            id="frame-missing-a-required-field",
        ),
        pytest.param({"frames": {"0": {}}, "warnings": []}, id="frames-not-a-list"),
        pytest.param(
            {
                "frames": [
                    {
                        "index": 1,
                        "timestamp_sec": 0.0,
                        "path": "frames/frame_0001.png",
                        "relative_path": "frames/frame_0001.png",
                    }
                ],
                "warnings": "one warning",
            },
            id="warnings-not-a-list",
        ),
        pytest.param(
            {
                "frames": [
                    {
                        "index": 1,
                        "timestamp_sec": 0.0,
                        "path": "frames/frame_0001.png",
                        "relative_path": "frames/frame_0001.png",
                    }
                ],
                "warnings": ["ocr fell back to the whole frame"],
            },
            id="warnings-that-are-not-warnings",
        ),
        pytest.param(
            {
                "frames": [
                    {
                        "index": 1,
                        "timestamp_sec": 0.0,
                        "path": "frames/frame_0001.png",
                        "relative_path": 42,
                    }
                ],
                "warnings": [],
            },
            id="frame-field-of-the-wrong-type",
        ),
        pytest.param(
            {
                "frames": [
                    {
                        "index": 1,
                        "timestamp_sec": 0.0,
                        "path": "frames/frame_0001.png",
                        "relative_path": "frames/frame_0001.png",
                        "interpretation": {"visual_summary": 42},
                    }
                ],
                "warnings": [],
            },
            id="interpretation-field-of-the-wrong-type",
        ),
    ],
)
def test_a_resumed_payload_whose_shape_drifted_recomputes_its_stage(
    drifted: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-23 validates the envelope; the *shape* of the payload is the stage's own.

    `validated_stage_payload` checks the version, the **bundle key**, the stage
    and the paths, and hands the payload back unexamined - which is correct, the
    store does not own what a payload means. But the pipeline then read the
    fields it expected straight off it, so a document whose shape had drifted -
    an older Distill's spelling, a truncated write, scratch a user edited -
    reached the run as a `KeyError` or an `AttributeError` and surfaced as
    `E_INTERNAL`. That is D-046 and R-23 inverted: the run died over its own
    scratch, which is the one thing a resume must never cost.

    Driven through a real second run of the **bundle key** - a first run leaves
    the **staging directory** behind, `resume_partial` defaults on, and this is
    the run that reads it back.
    """
    root = tmp_path / "output"
    root.mkdir()
    interrupted = begin_run(BundleStore.open(root))
    plant(
        interrupted,
        "frames",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": BUNDLE_KEY,
            "stage": "frames",
            "payload": drifted,
        },
    )
    interrupted.release()

    response = resuming_run(root, tmp_path, monkeypatch).execute()

    assert response["cached"] is False
    assert response["frame_count"] == 1
    assert selected == ["transcript", "frames", "ocr"], (
        "the drifted stage result was used instead of recomputed"
    )
    # The recomputation replaced the scratch, so the *next* resume reads a
    # payload of the shape this stage speaks.
    assert response["warnings"] == []


@pytest.mark.parametrize(
    "drifted",
    [
        pytest.param({"transcript": "everything that was said"}, id="transcript-not-a-document"),
        pytest.param(
            {"transcript": {"language": "en", "language_probability": "very likely"}},
            id="transcript-field-that-will-not-convert",
        ),
        pytest.param(
            {"transcript": {"language": "en", "segments": ["everything that was said"]}},
            id="segments-that-are-not-segments",
        ),
        pytest.param(
            {"transcript": {"language": "en", "segments": "everything that was said"}},
            id="segments-that-are-not-a-collection",
        ),
        pytest.param({"warnings": []}, id="payload-with-no-transcript-field"),
    ],
)
def test_a_resumed_transcript_whose_shape_drifted_recomputes_its_stage(
    drifted: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The **transcript** stage is exposed exactly as the frame stages are.

    Its carrier is rebuilt from a recorded document too, and the document is
    input in the same way: `Transcript.from_document` coerces the fields it
    names, so a drifted one raises where the coercion happens rather than
    where the stage result was read.
    """
    root = tmp_path / "output"
    root.mkdir()
    interrupted = begin_run(BundleStore.open(root))
    plant(
        interrupted,
        "transcript",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": BUNDLE_KEY,
            "stage": "transcript",
            "payload": drifted,
        },
    )
    interrupted.release()

    response = resuming_run(root, tmp_path, monkeypatch).execute()

    assert response["transcript_path"] is None
    assert selected == ["transcript", "frames", "ocr"], (
        "the drifted stage result was used instead of recomputed"
    )


# 8. A stage result that cannot be *written* costs the next resume, not this
#    run (boundary review, finding 1)


def foreign_document(stage: str) -> str:
    """A stage result another bundle's run would leave: readable, and not ours."""
    return json.dumps(
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": OTHER_BUNDLE_KEY,
            "stage": stage,
            "payload": {"frames": [], "warnings": []},
        }
    )


def recompute_ocr(run: BundleRun, pipeline_run: ProcessingRun) -> list[str]:
    """Drive the ocr stage once, returning the extracted text it came back with.

    The whole seam in one call: `read_stage`, the decision, the producer, and
    the `write_stage` that records what the producer returned.
    """
    return [
        frame.extracted_text
        for frame in pipeline_run._run_stage(
            run,
            ProgressHeartbeat(pipeline_run.progress.counter).start(),
            [],
            "ocr",
            (),
            lambda: {"frames": [read_frame(run, "this bundle's screen")], "warnings": []},
            pipeline_run._recovered_frames,
        )
    ]


def test_a_symlink_at_a_stage_result_path_costs_the_resume_not_the_run(
    tmp_path: Path,
) -> None:
    """A stage result that cannot be recorded is scratch lost, never a run lost.

    Rejecting a document at the read is only half of D-030. The stage then
    recomputes, and the recomputation writes its result to the same path - so a
    non-regular file left there put the *write* through the confinement check
    and the run died on `E_BAD_OUTPUT_DIR`, one step after correctly refusing to
    trust what was there. Anything can leave one: a co-tenant, a backup tool, a
    user's own `ln -s`.

    Permanently, which is what made it worse than a run: `abandon` deliberately
    keeps the **staging directory** and `resume_partial` defaults on, so every
    later run of the **bundle key** reached the same write and died the same
    way. A stage result that cannot be written must cost the next **resume**,
    which is one recomputation, and nothing else.

    Confinement is not what gives. The link is refused rather than followed, and
    the file it points at is neither read nor written.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    pipeline_run = processing_run(root)
    outside = tmp_path / "co-tenant.json"
    outside.write_text(foreign_document("ocr"))
    stage_result_file(run, "ocr").symlink_to(outside)

    assert run.read_stage("ocr") is None

    assert recompute_ocr(run, pipeline_run) == ["this bundle's screen"]
    # And again, because the failure this covers was not one run's: the run that
    # came after it read the same directory and reached the same write.
    assert recompute_ocr(run, pipeline_run) == ["this bundle's screen"]

    assert stage_result_file(run, "ocr").is_symlink(), "the link was replaced, not refused"
    assert outside.read_text() == foreign_document("ocr"), "the write followed the link"


def test_a_directory_at_a_stage_result_path_costs_the_resume_not_the_run(
    tmp_path: Path,
) -> None:
    """The same, for what confinement has no opinion about.

    A symlink is refused by `confined_path`. A directory is not - it is inside
    the bundle, reached through nothing - so it passes confinement and is
    refused for what it is instead: a stage result is a regular file, and a
    path holding anything else is a path this run cannot record at.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    pipeline_run = processing_run(root)
    stage_result_file(run, "ocr").mkdir()

    assert recompute_ocr(run, pipeline_run) == ["this bundle's screen"]
    assert stage_result_file(run, "ocr").is_dir()


def test_a_fifo_at_a_stage_result_path_is_refused_rather_than_written_into(
    tmp_path: Path,
) -> None:
    """The refusal is asked before the open, because one kind of file blocks.

    A directory or a socket fails the write with an errno; a fifo does not fail
    it at all. Opening one for writing waits for a reader, so a run with a fifo
    at a stage-result path would hang under its own lock - no error, no
    progress, and the **bundle key** held until somebody killed it. Asking
    `lstat` first is what makes "not a regular file" a refusal rather than a
    write whose behaviour depends on what is at the other end.

    The test attaches a reader precisely so that the unguarded write would
    *succeed*: without one this would prove the guard by hanging, which is not
    a proof anybody can run. Nothing arrives at that reader.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    pipeline_run = processing_run(root)
    fifo = stage_result_file(run, "ocr")
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)

    try:
        assert recompute_ocr(run, pipeline_run) == ["this bundle's screen"]
        # Nothing ever opened the fifo for writing, so the reader is at end of
        # file rather than merely waiting.
        assert os.read(reader, 64) == b""
    finally:
        os.close(reader)


def test_a_stage_result_path_this_process_may_not_write_costs_the_resume(
    tmp_path: Path,
) -> None:
    """And a target that is everything it should be, on a filesystem that says no.

    A read-only file is the reachable case; a full disk and a mount that went
    read-only under the run are the same refusal with a different errno. The
    run holds the payload it was about to record, so what is lost is the next
    **resume**'s head start.
    """
    root = tmp_path / "output"
    root.mkdir()
    store = BundleStore.open(root)
    run = begin_run(store)
    pipeline_run = processing_run(root)
    unwritable = stage_result_file(run, "ocr")
    unwritable.write_text("{}")
    unwritable.chmod(0o444)

    try:
        assert recompute_ocr(run, pipeline_run) == ["this bundle's screen"]
        assert unwritable.read_text() == "{}"
    finally:
        unwritable.chmod(0o644)


def test_a_stage_that_produces_the_wrong_shape_is_reported_rather_than_retried(
    tmp_path: Path,
) -> None:
    """Tolerance is for scratch, and a stage's own output is not scratch.

    The same shape check answers a resumed payload and a fresh one, and only one
    of them means "recompute": a producer whose output this stage cannot read
    will produce it again identically, so retrying is a loop and publishing it
    is worse. It is a defect in the run, reported with the stage that has it -
    not a `KeyError` from whichever consumer reached the missing field first.
    """
    root = tmp_path / "output"
    root.mkdir()
    run = begin_run(BundleStore.open(root))
    pipeline_run = processing_run(root)

    with pytest.raises(DistillError) as raised:
        pipeline_run._run_stage(
            run,
            ProgressHeartbeat(pipeline_run.progress.counter).start(),
            [],
            "ocr",
            (),
            lambda: {"frames": "one frame", "warnings": []},
            pipeline_run._recovered_frames,
        )

    assert raised.value.code == "E_BAD_STAGE_PAYLOAD"
    assert raised.value.details == {"stage": "ocr"}


def test_the_stage_result_file_is_opened_in_a_way_that_cannot_be_redirected(
    tmp_path: Path,
) -> None:
    """The refusal is the kernel's at the open, not only a check that ran first.

    `_recordable_stage_result` asks what is at the path and gives an operator a
    reason; between that question and the open, the path can be replaced. R-16
    has to hold anyway, so both opens are `O_NOFOLLOW` - a link swapped in is
    `ELOOP`, not a write through to whatever it points at - and `O_NONBLOCK`, so
    a fifo swapped in answers `ENXIO` instead of making the run wait under its
    own lock for a reader that never comes.

    Asked of the emitter and the reader directly, because the state check in
    front of them is exactly what a race defeats: a test that went through it
    would be proving the check, not the open.
    """
    outside = tmp_path / "co-tenant.json"
    outside.write_text("{}\n")
    link = tmp_path / "_ocr.json"
    link.symlink_to(outside)
    fifo = tmp_path / "_frames.json"
    os.mkfifo(fifo)

    with pytest.raises(OSError):
        EMITTER.emit(link, '{"payload": {}}')
    with pytest.raises(OSError):
        _read_regular_file(link)
    with pytest.raises(OSError):
        EMITTER.emit(fifo, '{"payload": {}}')

    assert outside.read_text() == "{}\n"


def bundle_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [json.loads(record.message) for record in caplog.records]


def test_a_stage_result_the_run_could_not_use_or_record_says_so_in_the_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both new refusals are silent from the outside, so both are logged.

    A discard costs a recomputation and an unrecordable write costs the next
    **resume**, and neither shows up in what a caller receives - which is the
    point of the fix and also its risk. Unlogged, a **bundle key** whose scratch
    can never be recorded looks exactly like a pipeline that is slow, and the
    directory somebody has to go and look at is not named anywhere.
    """
    root = tmp_path / "output"
    root.mkdir()
    run = begin_run(BundleStore.open(root))
    pipeline_run = processing_run(root)
    plant(
        run,
        "ocr",
        {
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "bundle_key": BUNDLE_KEY,
            "stage": "ocr",
            "payload": {"frames": "one frame"},
        },
    )

    with caplog.at_level(logging.DEBUG, logger="distill.bundle_store"):
        assert recompute_ocr(run, pipeline_run) == ["this bundle's screen"]
        stage_result_file(run, "local_vision").mkdir()
        run.write_stage("local_vision", {"frames": [], "warnings": []})

    reasons = {
        (event["event"], event["detail"]["stage"]): event["detail"]["reason"]
        for event in bundle_events(caplog)
        if "reason" in event["detail"]
    }
    assert reasons[("stage_result_rejected", "ocr")] == "payload_shape_unusable"
    assert reasons[("stage_result_not_recorded", "local_vision")] == "not_a_regular_file"


def test_a_stage_result_reached_through_a_symlink_is_not_read(tmp_path: Path) -> None:
    """A **stage result** is what is at the name, not what the name points at.

    The read followed a link out of the bundle and validated whatever came back,
    so a document planted outside the **output root** could be resumed from as
    long as it carried the right **bundle key** - and the write, which refuses
    the link, would then never replace it. R-16's refusal is about the path a
    run touches, and reading is touching.
    """
    root = tmp_path / "output"
    root.mkdir()
    run = begin_run(BundleStore.open(root))
    outside = tmp_path / "planted.json"
    outside.write_text(
        json.dumps(
            {
                "schema_version": STAGE_RESULT_SCHEMA_VERSION,
                "bundle_key": BUNDLE_KEY,
                "stage": "ocr",
                "payload": {"frames": [], "warnings": []},
            }
        )
    )
    stage_result_file(run, "ocr").symlink_to(outside)

    assert run.read_stage("ocr") is None
