from __future__ import annotations

import json
import os
import struct
import urllib.error
import zlib
from pathlib import Path
from typing import Any

import pytest

from saccade.local_vision import (
    DEFAULT_OLLAMA_MODEL,
    LocalVisionConfig,
    load_local_vision_config,
    local_vision_config_from_args,
    parse_interpretation_json,
    probe_local_vision,
    probe_ollama_availability,
    smoke_ollama_image,
    try_interpret_image,
)
from saccade.options import SaccadeOptions
from saccade.pipeline import local_vision_diagnostics


def test_default_local_vision_config_uses_qwen3_vl_8b(tmp_path: Path) -> None:
    config = load_local_vision_config(tmp_path)

    assert config.backend == "ollama"
    assert config.model == DEFAULT_OLLAMA_MODEL
    assert config.caption_frames is True


def test_local_vision_config_loads_from_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "saccade.local-vision.json").write_text(
        json.dumps(
            {
                "model": "qwen3-vl:32b",
                "base_url": "http://127.0.0.1:11435/",
                "timeout_sec": 12,
                "caption_frames": True,
            }
        )
    )
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))

    options = SaccadeOptions.from_args({})

    assert options.caption_frames is True
    assert options.local_vision_model == "qwen3-vl:32b"
    assert options.local_vision_base_url == "http://127.0.0.1:11435"
    assert options.local_vision_timeout_sec == 12.0


def test_nested_saccade_config_is_supported(tmp_path: Path) -> None:
    (tmp_path / "saccade.json").write_text(
        json.dumps({"local_vision": {"model": "qwen3-vl:30b-a3b", "caption_frames": True}})
    )

    config = load_local_vision_config(tmp_path)

    assert config.model == "qwen3-vl:30b-a3b"
    assert config.caption_frames is True


def test_per_call_local_vision_model_override(tmp_path: Path) -> None:
    (tmp_path / "saccade.local-vision.json").write_text(
        json.dumps({"model": "qwen3-vl:32b", "caption_frames": True})
    )

    config = local_vision_config_from_args(
        {"local_vision_model": "qwen3-vl:8b", "caption_frames": "false"},
        tmp_path,
    )

    assert config.model == "qwen3-vl:8b"
    assert config.caption_frames is False


def test_ollama_probe_checks_model_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    probe = probe_ollama_availability(LocalVisionConfig(model="qwen3-vl:8b"))

    assert probe.available is True
    assert probe.model == "qwen3-vl:8b"


def test_ollama_probe_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode()

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    probe = probe_ollama_availability(LocalVisionConfig(model="qwen3-vl:8b"))

    assert probe.available is False
    assert probe.code == "local_vision_model_missing"
    assert "ollama pull qwen3-vl:8b" in probe.message


def test_unsupported_lm_studio_backend_is_not_treated_as_ollama() -> None:
    probe = probe_local_vision(LocalVisionConfig(backend="lmstudio"))

    assert probe.available is False
    assert probe.code == "local_vision_backend_unsupported"


def test_mlx_backend_probe_reports_unavailable_without_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    import saccade.local_vision as lv

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlx_vlm":
            raise ImportError("no mlx_vlm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    probe = lv.probe_local_vision(LocalVisionConfig(backend="mlx", model="mlx-community/x"))

    assert probe.available is False
    assert probe.code == "local_vision_mlx_unavailable"
    assert "mlx-vlm" in probe.message


def test_mlx_backend_probe_available_when_mlx_importable() -> None:
    pytest.importorskip("mlx_vlm")
    probe = probe_local_vision(LocalVisionConfig(backend="mlx", model="mlx-community/x"))

    assert probe.available is True
    assert probe.code == "local_vision_available"


def test_mlx_backend_is_accepted_by_options() -> None:
    options = SaccadeOptions.from_args({"local_vision_backend": "mlx"})

    assert options.local_vision_backend == "mlx"


def test_local_vision_diagnostics_include_pull_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_probe(config: LocalVisionConfig) -> object:
        return type(
            "FakeProbe",
            (),
            {
                "available": False,
                "backend": config.backend,
                "model": config.model,
                "base_url": config.base_url,
                "code": "local_vision_ollama_unavailable",
                "message": "missing server",
                "detail": {},
            },
        )()

    monkeypatch.setattr("saccade.pipeline.probe_local_vision", fake_probe)

    diagnostics = local_vision_diagnostics({"local_vision_model": "qwen3-vl:8b"})

    assert diagnostics["setup_command"] == "ollama pull qwen3-vl:8b"
    assert "LM Studio" in diagnostics["lm_studio_note"]


def test_ollama_image_smoke_reports_image_input_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTagsResponse:
        def __enter__(self) -> FakeTagsResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

    calls = 0

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeTagsResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeTagsResponse()
        raise urllib.error.URLError("image input rejected")

    image = tmp_path / "pixel.png"
    image.write_bytes(b"not really an image")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    probe = smoke_ollama_image(LocalVisionConfig(model="qwen3-vl:8b"), image)

    assert probe.available is False
    assert probe.code == "local_vision_image_smoke_failed"
    assert "image smoke test" in probe.message


def test_interpret_image_returns_structured_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTagsResponse:
        def __enter__(self) -> FakeTagsResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

    class FakeGenerateResponse:
        def __enter__(self) -> FakeGenerateResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "response": json.dumps(
                        {
                            "visual_summary": "A line chart",
                            "detected_elements": ["axis", "trend line"],
                            "interpretation": "Values rise over time.",
                            "uncertainty": "Low",
                        }
                    )
                }
            ).encode()

    calls = 0

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeTagsResponse | FakeGenerateResponse:
        nonlocal calls
        calls += 1
        return FakeTagsResponse() if calls == 1 else FakeGenerateResponse()

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"),
        image,
        "Interpret the chart.",
        prompt_profile="technical",
    )

    assert warning is None
    assert result is not None
    assert result.public_dict() == {
        "visual_summary": "A line chart",
        "detected_elements": ["axis", "trend line"],
        "interpretation": "Values rise over time.",
        "uncertainty": "Low",
        "backend": "ollama",
        "model": "qwen3-vl:8b",
        "prompt_profile": "technical",
        "frame_kind": "",
        "verbatim_text": "",
        "text_confidence": "none",
    }


def test_interpret_image_accepts_qwen_thinking_json_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTagsResponse:
        def __enter__(self) -> FakeTagsResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

    class FakeGenerateResponse:
        def __enter__(self) -> FakeGenerateResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "response": "",
                    "thinking": json.dumps(
                        {
                            "visual_summary": "A dashboard",
                            "detected_elements": ["bar chart"],
                            "interpretation": "The chart compares metrics.",
                            "uncertainty": "Low",
                        }
                    ),
                }
            ).encode()

    calls = 0

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeTagsResponse | FakeGenerateResponse:
        nonlocal calls
        calls += 1
        return FakeTagsResponse() if calls == 1 else FakeGenerateResponse()

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"), image, "Interpret."
    )

    assert warning is None
    assert result is not None
    assert result.visual_summary == "A dashboard"


def test_interpret_image_retries_past_one_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTagsResponse:
        def __enter__(self) -> FakeTagsResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

    class FakeResponse:
        def __init__(self, body: str) -> None:
            self._body = body

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"response": self._body}).encode()

    good = json.dumps(
        {
            "visual_summary": "A slide",
            "detected_elements": ["title"],
            "interpretation": "Recovered on retry.",
            "uncertainty": "Low",
            "verbatim_text": "Recovered",
            "text_confidence": "high",
        }
    )
    calls = 0

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeTagsResponse()  # availability probe
        if calls == 2:
            return FakeResponse("not json")  # first generate attempt: malformed
        return FakeResponse(good)  # retry succeeds

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"), image, "Interpret."
    )

    assert warning is None
    assert result is not None
    assert result.verbatim_text == "Recovered"


def test_interpret_image_model_missing_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTagsResponse:
        def __enter__(self) -> FakeTagsResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": []}).encode()

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeTagsResponse())

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"), image, "Interpret."
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_model_missing"


def test_interpret_image_server_unavailable_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> object:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"), image, "Interpret."
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_ollama_unavailable"


def test_interpret_image_timeout_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTagsResponse:
        def __enter__(self) -> FakeTagsResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

    calls = 0

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeTagsResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeTagsResponse()
        raise TimeoutError("timed out")

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"), image, "Interpret."
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_timeout"


def test_interpret_image_malformed_response_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTagsResponse:
        def __enter__(self) -> FakeTagsResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

    class FakeGenerateResponse:
        def __enter__(self) -> FakeGenerateResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"response": "not json"}).encode()

    calls = 0

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeTagsResponse | FakeGenerateResponse:
        nonlocal calls
        calls += 1
        return FakeTagsResponse() if calls == 1 else FakeGenerateResponse()

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"), image, "Interpret."
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_malformed_response"


def test_interpretation_parser_accepts_fenced_or_prefaced_json() -> None:
    payload = {
        "visual_summary": "A chart",
        "detected_elements": ["bar"],
        "interpretation": "Growth increases.",
        "uncertainty": "Low",
    }

    assert parse_interpretation_json(f"```json\n{json.dumps(payload)}\n```") == payload
    assert parse_interpretation_json(f"Here is the result:\n{json.dumps(payload)}") == payload


def test_interpret_image_cancel_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, warning = try_interpret_image(
        LocalVisionConfig(model="qwen3-vl:8b"), image, "Interpret."
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_cancelled"


@pytest.mark.skipif(
    os.environ.get("SACCADE_RUN_OLLAMA_SMOKE") != "1",
    reason="set SACCADE_RUN_OLLAMA_SMOKE=1 after `ollama pull qwen3-vl:8b`",
)
def test_ollama_qwen3_vl_image_smoke(tmp_path: Path) -> None:
    image = tmp_path / "pixel.png"
    image.write_bytes(_solid_png_bytes())

    probe = smoke_ollama_image(LocalVisionConfig(model=DEFAULT_OLLAMA_MODEL), image)

    assert probe.available, probe.message


def _solid_png_bytes() -> bytes:
    width = height = 32
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
