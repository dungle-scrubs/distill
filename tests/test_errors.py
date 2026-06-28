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
    assert warning("ocr", "ocr_failed", "nope") == {
        "stage": "ocr",
        "code": "ocr_failed",
        "message": "nope",
    }
    with pytest.raises(ValueError):
        warning("ocr", "OCR_FAILED", "bad")


def test_session_error_channel_contains_json_text() -> None:
    result = DistillSession().call_tool("missing", {})
    payload = json.loads(result["error"]["message"])
    assert payload["code"] == "E_UNKNOWN_TOOL"
    assert payload["stage"] == "protocol"
