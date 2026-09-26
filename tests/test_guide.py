"""The caller-facing guidance surface: `distill guide` and the response's
`visual_extraction` field.

Both exist for the same reason. The repository's documents (README,
AGENTS.md, CONTEXT.md) do not follow an installed CLI, and the vision gap on
the poteto-post run showed the cost: a caller can hold a bundle whose frames
were never read as images and not know it. The guide carries the static
facts; `visual_extraction` carries the per-generation facts. Neither enters a
bundle - `guide.py` is exempt from the signature for that reason - so the
tests here hold them as caller contract, not as cache identity.
"""

from __future__ import annotations

import json

import pytest

from distill.cli import main
from distill.errors import DistillError
from distill.guide import GUIDE_TOPICS, guide_payload
from distill.options import (
    DEFAULT_LOCAL_VISION_BASE_URL,
    DEFAULT_LOCAL_VISION_MODEL,
    OPTION_DEFAULTS,
)
from distill.response import visual_extraction
from distill.source import validate_output_root


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[object, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_index_lists_every_topic_and_only_those() -> None:
    payload = guide_payload(None)
    assert payload["topics"] == ["cache", "consuming", "sources", "vision"]
    assert payload["topics"] == list(GUIDE_TOPICS)
    for topic in payload["topics"]:
        assert f"`{topic}`" in payload["markdown"]


def test_every_processing_command_is_documented_in_the_index() -> None:
    """A source kind added to the parser is a source kind the guide must name.

    Read off the parser rather than a second list, so a command registered
    without guide coverage fails here - the same derivation the error-boundary
    sweep uses for subcommands.
    """
    from distill.cli import build_parser

    registered: list[str] = []
    for action in build_parser()._actions:
        if action.__class__.__name__ == "_SubParsersAction":
            registered = [str(name) for name in (action.choices or {})]
    assert registered, "the parser registers no subcommands"
    markdown = guide_payload(None)["markdown"] + "".join(
        guide_payload(topic)["markdown"] for topic in GUIDE_TOPICS
    )
    for command in registered:
        if command.startswith("process-"):
            assert command in markdown, f"{command} is registered but the guide never names it"


def test_each_topic_serves_its_own_markdown() -> None:
    for topic in GUIDE_TOPICS:
        payload = guide_payload(topic)
        assert payload["topic"] == topic
        assert len(payload["markdown"]) > 400
        assert payload["markdown"].startswith("#")


def test_an_unknown_topic_is_a_usage_error_naming_the_valid_ones() -> None:
    with pytest.raises(DistillError) as failure:
        guide_payload("install")
    assert failure.value.code == "E_BAD_OPTIONS"
    assert failure.value.details["topics"] == ["cache", "consuming", "sources", "vision"]


def test_the_sources_topic_holds_the_x_conventions() -> None:
    sources = guide_payload("sources")["markdown"]
    assert "status" in sources
    assert "--cookies" in sources
    assert "--cookies-from-browser" in sources
    assert "first" in sources
    assert "/video/N" in sources


def test_the_vision_topic_pins_the_default_endpoint_to_the_code() -> None:
    vision = guide_payload("vision")["markdown"]
    assert DEFAULT_LOCAL_VISION_BASE_URL in vision
    assert DEFAULT_LOCAL_VISION_MODEL in vision
    assert "options hash" in vision
    assert "ocr_only" in vision


def test_the_consuming_topic_covers_the_boundary_and_salience() -> None:
    consuming = guide_payload("consuming")["markdown"]
    assert "untrusted" in consuming.lower()
    assert "salience" in consuming
    assert "corroboration" in consuming


def test_the_cache_topic_covers_the_inspection_commands() -> None:
    cache = guide_payload("cache")["markdown"]
    assert "cleanup-cache" in cache
    assert "cache-doctor" in cache
    assert "filtered-view" in cache


def test_documented_defaults_are_derived_not_restated() -> None:
    """Defaults in the text are read from the option table at import time.

    A default that changes in `options.py` changes this text at the same
    commit, so the guide cannot outlive the code it describes.
    """
    index = guide_payload(None)["markdown"]
    assert f"`{OPTION_DEFAULTS['whisper_model']}`" in index
    assert _default_output_root_matches(index)
    sources = guide_payload("sources")["markdown"]
    assert f"{OPTION_DEFAULTS['max_duration_sec']} seconds" in sources


def _default_output_root_matches(index: str) -> bool:
    return str(validate_output_root(None, create=False)) in index


def test_the_cli_prints_the_guide_as_json_with_markdown_under_markdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, _ = _run(["guide", "vision"], capsys)
    assert code is None
    payload = json.loads(out)
    assert payload["topic"] == "vision"
    assert "vision endpoint" in payload["markdown"]

    code, out, _ = _run(["guide"], capsys)
    assert code is None
    payload = json.loads(out)
    assert payload["topics"] == list(GUIDE_TOPICS)
    assert "# Distill guide" in payload["markdown"]


def test_the_cli_rejects_an_unknown_topic_through_the_error_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["guide", "install"])
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip().splitlines()[-1])
    assert payload["code"] == "E_BAD_OPTIONS"


def test_visual_extraction_modes_are_derived_from_the_frame_documents() -> None:
    reading = {"visual_summary": "a slide about a trust graph"}
    vision = [{"ocr_text": "text", "visual_interpretation": reading}]
    assert visual_extraction(vision)["mode"] == "vision"

    ocr_only = [{"ocr_text": "text"}]
    payload = visual_extraction(ocr_only)
    assert payload["mode"] == "ocr_only"
    assert "reachable vision endpoint" in payload["advice"]

    none = [{"ocr_text": "  "}]
    payload = visual_extraction(none)
    assert payload["mode"] == "none"
    assert "OCR enabled" in payload["advice"]

    assert visual_extraction([])["mode"] == "none"


def test_visual_extraction_counts_are_per_frame_and_advice_is_absent_when_vision() -> None:
    frames = [
        {"ocr_text": "slide text", "visual_interpretation": {"visual_summary": "a chart"}},
        {"ocr_text": "", "visual_interpretation": {"verbatim_text": "label"}},
        {"ocr_text": "text only"},
    ]
    payload = visual_extraction(frames)
    assert payload == {
        "mode": "vision",
        "frames_total": 3,
        "frames_with_readings": 2,
        "frames_with_ocr_text": 2,
        "advice": None,
    }
