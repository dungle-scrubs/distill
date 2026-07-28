from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.errors import (
    DETAIL_TRUNCATION_SUFFIX,
    DistillError,
    aggregate_warnings,
    warning,
)
from distill.pipeline import DistillSession


def test_distill_error_serializes_to_parseable_json_text() -> None:
    error = DistillError("E_BAD_SOURCE", "source", "missing", {"path": "x"})
    payload = json.loads(error.to_json_text())
    assert payload == {
        "code": "E_BAD_SOURCE",
        "stage": "source",
        "message": "missing",
        "details": {"path": "x"},
    }


def test_warning_shape_and_code_validation() -> None:
    """R-41 changed the record: a **warning** carries how many times it happened.

    Classified *defect* against R-41 - the old equality pinned the record to
    exactly stage/code/message, which is the shape that made eighty identical
    failures eighty entries in a **manifest**. The code domain it also checked
    is unchanged and still checked here; `tests/test_warning_aggregation.py`
    owns the fold itself.
    """
    assert warning("ocr", "ocr_failed", "nope") == {
        "stage": "ocr",
        "code": "ocr_failed",
        "message": "nope",
        "occurrences": 1,
    }
    assert warning("ocr", "ocr_failed", "nope", occurrences=4)["occurrences"] == 4
    with pytest.raises(ValueError):
        warning("ocr", "OCR_FAILED", "bad")
    with pytest.raises(ValueError):
        warning("ocr", "ocr_failed", "never happened", occurrences=0)


def test_the_session_error_channel_carries_the_code_and_stage_as_fields() -> None:
    """R-46 changed the envelope: the error half *is* the **fatal error** record.

    Classified *defect* under R-46 - the old assertion recovered the code and
    the stage by `json.loads`-ing them back out of a `message` string, which is
    the flattening R-46 forbids. Every reader wanting the code had to know the
    message was secretly JSON, and a reader that did not know saw one opaque
    sentence.
    """
    result = DistillSession().call_tool("missing", {})
    assert result["error"] == {
        "code": "E_UNKNOWN_TOOL",
        "stage": "protocol",
        "message": "Unknown tool: missing",
        "details": {},
    }


def test_an_uncoded_failure_reaches_the_session_channel_as_one_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catch-all's mapping, asked of the surface a caller reads.

    The type and message identify the failure; the record shape is the same one
    a coded failure gets, so nothing downstream branches on which kind it was
    before it can read the code.
    """

    def fault(_name: str, _args: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("a fault three modules down")

    monkeypatch.setattr("distill.pipeline.call_registered_tool", fault)
    result = DistillSession().call_tool("cache_doctor", {})

    assert result["error"]["code"] == "E_INTERNAL"
    assert result["error"]["stage"] == "internal"
    assert result["error"]["details"] == {
        "exception": "RuntimeError",
        "message": "a fault three modules down",
    }


def test_a_count_that_is_not_a_number_is_refused_including_a_boolean() -> None:
    """`occurrences: true` is not "it happened once".

    `bool` is an `int` subtype in Python, so a JSON `true` arriving where a
    count belongs passed both the record's own check and the fold's, and
    round-tripped into a published **manifest** as the number of times
    something happened. It is a wrong type like any other wrong type.
    """
    # `occurrences="4"` the type checker already refuses; `occurrences=True`
    # it accepts, because `bool` *is* an `int` to the checker. That is exactly
    # why the check has to exist at runtime too.
    with pytest.raises(ValueError):
        warning("ocr", "ocr_failed", "nope", occurrences=True)


# --- Details that JSON has no word for. ---------------------------------------
#
# A **fatal error** is published as JSON text, and its `details` is whatever the
# stage that raised it put there. Every one of these is a value some stage can
# reach for without thinking - the `Path` it was working on, the bytes it could
# not decode, the ratio that came out `inf` - and each of them ended the
# serialization with a `TypeError` or a `ValueError` rather than a record. At
# the CLI that is fatal twice over: `_fail` runs inside an `except` clause, so
# nothing at all catches what the serialization raises and the operator gets no
# error object, only a stack.


class UnprintableFailure(Exception):
    """An exception whose message cannot be read, which is still a failure.

    Not a contrived shape: `__str__` on a real exception runs a real `__repr__`
    over whatever it was constructed with, and a lazily-formatted message is
    ordinary. `from_unexpected` asks for exactly that string.
    """

    def __str__(self) -> str:
        raise RuntimeError("this exception cannot say what went wrong")


def test_details_that_json_has_no_word_for_still_serialize() -> None:
    """FAILS FIRST: `TypeError: Object of type PosixPath is not JSON serializable`."""
    error = DistillError(
        "E_BAD_SOURCE",
        "source",
        "missing",
        {"path": Path("/tmp/clip.mp4"), "head": b"\xff\xd8", "ratio": float("inf")},
    )

    details = json.loads(error.to_json_text())["details"]

    assert "clip.mp4" in details["path"]
    assert "\\xff" in details["head"]
    assert details["ratio"] == "inf"


def test_a_detail_whose_repr_raises_does_not_take_the_record_with_it() -> None:
    """The last thing a boundary may do is fail while reporting a failure."""

    class Unrepresentable:
        def __repr__(self) -> str:
            raise RuntimeError("no")

    error = DistillError("E_BAD_SOURCE", "source", "missing", {"value": Unrepresentable()})

    assert json.loads(error.to_json_text())["details"]["value"] == "<Unrepresentable>"


def test_details_that_refer_to_themselves_serialize_once_and_stop() -> None:
    """A cycle is a `ValueError: Circular reference detected`, not a record."""
    cycle: dict[str, object] = {"stage": "source"}
    cycle["self"] = cycle

    details = json.loads(DistillError("E_X", "s", "m", {"cycle": cycle}).to_json_text())["details"]

    assert details["cycle"]["stage"] == "source"
    assert details["cycle"]["self"] == "<recursive>"


def test_a_detail_too_long_to_publish_is_capped_rather_than_dropped() -> None:
    """A record is a diagnosis, not a transcript: it is bounded, and says so."""
    details = json.loads(
        DistillError("E_X", "s", "m", {"blob": "x" * 10_000}).to_json_text()
    )["details"]

    assert len(details["blob"]) < 10_000
    assert details["blob"].endswith(DETAIL_TRUNCATION_SUFFIX)


def test_an_exception_that_cannot_say_what_went_wrong_still_becomes_a_record() -> None:
    """FAILS FIRST: `from_unexpected` calls `str(exc)`, which raises.

    The conversion runs at two boundaries and both are already handling a
    failure, so an exception raised *by the conversion* replaces the failure it
    was converting - with something that names neither.
    """
    error = DistillError.from_unexpected(UnprintableFailure())

    assert error.code == "E_INTERNAL"
    assert error.details["exception"] == "UnprintableFailure"
    assert error.details["message"] == "<unreadable: RuntimeError>"

    # The fold does not raise - it repairs, because it reads records other
    # runs and other Distill versions wrote - so there a bool is repaired to
    # one rather than counted as one.
    folded = aggregate_warnings(
        [{"stage": "ocr", "code": "ocr_failed", "message": "nope", "occurrences": True}]
    )

    assert folded == [{"stage": "ocr", "code": "ocr_failed", "message": "nope", "occurrences": 1}]
    # Stated on the type as well as the value, because `True == 1`: the
    # equality above passes on a record still carrying the boolean.
    assert type(folded[0]["occurrences"]) is int
