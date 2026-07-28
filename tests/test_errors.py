from __future__ import annotations

import json

import pytest

from distill.errors import DistillError, warning
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


def test_session_error_channel_contains_json_text() -> None:
    result = DistillSession().call_tool("missing", {})
    payload = json.loads(result["error"]["message"])
    assert payload["code"] == "E_UNKNOWN_TOOL"
    assert payload["stage"] == "protocol"
