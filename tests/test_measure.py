from __future__ import annotations

from pathlib import Path

import pytest

from saccade.measure import (
    CorpusItem,
    build_measurement,
    measurement_schema,
    parse_corpus_args,
)


def test_measurement_output_schema_is_defined() -> None:
    schema = measurement_schema()

    assert schema["schema_version"] == 1
    assert schema["assumptions"] == ["A-001", "A-002", "A-003", "A-004"]
    assert "phash_threshold_sweep" in schema["measurement_fields"]
    assert "timeout_behavior" in schema["measurement_fields"]


def test_measurement_script_handles_three_to_five_screencasts(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"fixture-{index}.mp4"
        path.write_bytes(b"video")
        paths.append(path)

    measurement = build_measurement(
        [CorpusItem(path=path, license_note="owned fixture") for path in paths]
    )

    assert len(measurement["corpus"]) == 3
    assert len(measurement["measurements"]) == 3


def test_measurement_script_rejects_corpus_outside_size_window(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"video")

    with pytest.raises(ValueError, match="3-5"):
        build_measurement([CorpusItem(path=path, license_note="owned fixture")])


def test_corpus_args_require_license_notes(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"video")

    items = parse_corpus_args([f"{path}::owned fixture"])

    assert items[0].path == path.resolve()
    assert items[0].license_note == "owned fixture"
