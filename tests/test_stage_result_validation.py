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
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from distill.bundle_store import (
    STAGE_RESULT_SCHEMA_VERSION,
    BundleRun,
    BundleStore,
)
from distill.options import DistillOptions
from distill.pipeline import ProcessingRun
from distill.progress import ProgressHeartbeat, ProgressReporter

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
        return {"frames": [{"ocr_text": "this bundle's screen"}], "warnings": []}

    warnings: list[dict[str, str]] = []
    payload = pipeline_run._run_stage(
        run,
        ProgressHeartbeat(pipeline_run.progress.counter).start(),
        warnings,
        "ocr",
        (),
        produce,
    )

    assert produced == ["ocr"], "the rejected stage result was reused instead of recomputed"
    assert payload == {"frames": [{"ocr_text": "this bundle's screen"}], "warnings": []}
    # The recomputation replaced the scratch, so the next resume has a stage
    # result bound to the bundle it actually belongs to.
    assert recorded_document(run, "ocr")["bundle_key"] == BUNDLE_KEY
    assert run.read_stage("ocr") == payload


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
