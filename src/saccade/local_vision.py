"""Local vision backend configuration and availability checks for Saccade.

This module owns local-only vision provider setup. It does not interpret frames
or mutate bundle artifacts; pipeline integration calls it to decide whether a
requested vision pass can run or should degrade to OCR-only output.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .vision_prompts import FRAME_KINDS, TEXT_CONFIDENCE_LEVELS

DEFAULT_OLLAMA_MODEL = "qwen3-vl:8b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SEC = 30.0
CONFIG_FILENAMES = ("saccade.local-vision.json", "saccade.json")


@dataclass(frozen=True)
class LocalVisionConfig:
    backend: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    caption_frames: bool = True

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalVisionProbe:
    available: bool
    backend: str
    model: str
    base_url: str
    code: str
    message: str
    detail: dict[str, Any]

    def warning(self) -> dict[str, str]:
        return {
            "stage": "local_vision",
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class LocalVisionResult:
    visual_summary: str
    detected_elements: list[str]
    interpretation: str
    uncertainty: str
    backend: str
    model: str
    prompt_profile: str
    frame_kind: str = ""
    verbatim_text: str = ""
    text_confidence: str = "none"

    @property
    def has_interpretation(self) -> bool:
        return bool(self.interpretation.strip() or self.detected_elements)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalVisionFailure(Exception):
    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def warning(self) -> dict[str, str]:
        return {
            "stage": "local_vision",
            "code": self.code,
            "message": self.message,
        }


def config_dir() -> Path:
    return Path(os.environ.get("CONFIG_DIR", Path.home() / ".tool-proxy")).expanduser()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_local_vision_config(base_dir: Path | None = None) -> LocalVisionConfig:
    root = (base_dir or config_dir()).expanduser()
    config = LocalVisionConfig()
    for filename in CONFIG_FILENAMES:
        payload = _read_json(root / filename)
        if filename == "saccade.json":
            nested = payload.get("local_vision")
            payload = nested if isinstance(nested, dict) else {}
        config = _config_from_payload(payload, config)
    return config


def local_vision_config_from_args(
    args: dict[str, Any],
    base_dir: Path | None = None,
) -> LocalVisionConfig:
    config = load_local_vision_config(base_dir)
    overrides: dict[str, Any] = {}
    if "caption_frames" in args:
        overrides["caption_frames"] = _coerce_bool(
            args.get("caption_frames"), config.caption_frames
        )
    if "local_vision_backend" in args:
        overrides["backend"] = str(args["local_vision_backend"])
    if "local_vision_model" in args:
        overrides["model"] = str(args["local_vision_model"])
    if "local_vision_base_url" in args:
        overrides["base_url"] = str(args["local_vision_base_url"])
    if "local_vision_timeout_sec" in args:
        overrides["timeout_sec"] = _coerce_float(
            args.get("local_vision_timeout_sec"), config.timeout_sec
        )
    return _config_from_payload(overrides, config)


def _config_from_payload(
    payload: dict[str, Any],
    base: LocalVisionConfig,
) -> LocalVisionConfig:
    if not payload:
        return base
    return replace(
        base,
        backend=str(payload.get("backend", base.backend)),
        model=str(payload.get("model", base.model)),
        base_url=str(payload.get("base_url", base.base_url)).rstrip("/"),
        timeout_sec=_coerce_float(payload.get("timeout_sec"), base.timeout_sec),
        caption_frames=_coerce_bool(
            payload.get("caption_frames", base.caption_frames),
            base.caption_frames,
        ),
    )


def probe_local_vision(config: LocalVisionConfig) -> LocalVisionProbe:
    if config.backend != "ollama":
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_backend_unsupported",
            message=f"Local vision backend '{config.backend}' is not supported; continuing with OCR-only output.",
            detail={"backend": config.backend},
        )
    return probe_ollama_availability(config)


def probe_ollama_availability(config: LocalVisionConfig) -> LocalVisionProbe:
    url = f"{config.base_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_ollama_unavailable",
            message="Ollama is unavailable for local vision; continuing with OCR-only output. Start Ollama and run `ollama pull qwen3-vl:8b`.",
            detail={"error": str(exc), "url": url},
        )
    except (TimeoutError, json.JSONDecodeError) as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_probe_failed",
            message="Ollama local vision probe failed; continuing with OCR-only output.",
            detail={"error": str(exc), "url": url},
        )

    models = payload.get("models", [])
    names = {
        str(model.get("name", ""))
        for model in models
        if isinstance(model, dict) and model.get("name")
    }
    if config.model not in names:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_model_missing",
            message=f"Ollama model '{config.model}' is not pulled; continuing with OCR-only output. Run `ollama pull {config.model}`.",
            detail={"available_models": sorted(names)},
        )
    return LocalVisionProbe(
        available=True,
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        code="local_vision_available",
        message="Ollama local vision is available.",
        detail={"available_models": sorted(names)},
    )


def smoke_ollama_image(config: LocalVisionConfig, image_path: Path) -> LocalVisionProbe:
    probe = probe_ollama_availability(config)
    if not probe.available:
        return probe
    payload = {
        "model": config.model,
        "prompt": "Reply with one short sentence describing this image.",
        "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
        "stream": False,
    }
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_image_smoke_failed",
            message="Ollama responded to model probing but failed the image smoke test.",
            detail={"error": str(exc)},
        )
    if not str(result.get("response", "")).strip():
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_image_smoke_failed",
            message="Ollama image smoke test returned an empty response.",
            detail={"response_keys": sorted(result.keys())},
        )
    return replace(probe, detail={**probe.detail, "image_smoke": "pass"})


def interpret_image(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    *,
    prompt_profile: str = "technical",
) -> LocalVisionResult:
    if config.backend != "ollama":
        raise LocalVisionFailure(
            "local_vision_backend_unsupported",
            f"Local vision backend '{config.backend}' is not supported.",
            {"backend": config.backend},
        )
    probe = probe_ollama_availability(config)
    if not probe.available:
        raise LocalVisionFailure(probe.code, probe.message, probe.detail)
    return _interpret_with_ollama(config, image_path, prompt, prompt_profile)


def try_interpret_image(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    *,
    prompt_profile: str = "technical",
) -> tuple[LocalVisionResult | None, dict[str, str] | None]:
    try:
        return interpret_image(config, image_path, prompt, prompt_profile=prompt_profile), None
    except KeyboardInterrupt:
        failure = LocalVisionFailure(
            "local_vision_cancelled",
            "Local vision was cancelled; continuing with OCR-only output.",
        )
        return None, failure.warning()
    except LocalVisionFailure as exc:
        return None, exc.warning()


def _interpret_with_ollama(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    prompt_profile: str,
) -> LocalVisionResult:
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise LocalVisionFailure(
            "local_vision_image_read_failed",
            "Local vision could not read the frame image; continuing with OCR-only output.",
            {"path": str(image_path), "error": str(exc)},
        ) from exc
    request_prompt = (
        f"{prompt}\n\n"
        "Return compact JSON with string fields frame_kind, verbatim_text, text_confidence, "
        "visual_summary, interpretation, uncertainty, and detected_elements as an array of strings. "
        "Leave verbatim_text empty and set text_confidence to \"none\" if you cannot read the text."
    )
    payload = {
        "model": config.model,
        "prompt": request_prompt,
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "stream": False,
        "format": "json",
    }
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise LocalVisionFailure(
            "local_vision_timeout",
            "Local vision timed out; continuing with OCR-only output.",
            {"timeout_sec": config.timeout_sec},
        ) from exc
    except urllib.error.URLError as exc:
        raise LocalVisionFailure(
            "local_vision_ollama_unavailable",
            "Ollama is unavailable for local vision; continuing with OCR-only output.",
            {"error": str(exc)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise LocalVisionFailure(
            "local_vision_malformed_response",
            "Ollama local vision returned malformed JSON; continuing with OCR-only output.",
            {"error": str(exc)},
        ) from exc
    raw_response = str(payload.get("response") or payload.get("thinking") or "")
    interpreted = parse_interpretation_json(raw_response)
    if interpreted is None:
        raise LocalVisionFailure(
            "local_vision_malformed_response",
            "Ollama local vision returned a malformed interpretation; continuing with OCR-only output.",
            {"response_preview": raw_response[:200]},
        )
    elements = interpreted.get("detected_elements", [])
    if not isinstance(elements, list):
        elements = []
    return LocalVisionResult(
        visual_summary=str(interpreted.get("visual_summary", "")).strip(),
        detected_elements=[str(item) for item in elements],
        interpretation=str(interpreted.get("interpretation", "")).strip(),
        uncertainty=str(interpreted.get("uncertainty", "")).strip(),
        backend=config.backend,
        model=config.model,
        prompt_profile=prompt_profile,
        frame_kind=_normalize_frame_kind(interpreted.get("frame_kind")),
        verbatim_text=str(interpreted.get("verbatim_text", "")).strip(),
        text_confidence=_normalize_text_confidence(interpreted.get("text_confidence")),
    )


def _normalize_frame_kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    return kind if kind in FRAME_KINDS else ""


def _normalize_text_confidence(value: Any) -> str:
    level = str(value or "").strip().lower()
    return level if level in TEXT_CONFIDENCE_LEVELS else "none"


def parse_interpretation_json(raw_response: str) -> dict[str, Any] | None:
    stripped = raw_response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(stripped)
    return parsed if isinstance(parsed, dict) else None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None
