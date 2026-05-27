"""Normalized tool options for Saccade pipeline stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from .errors import SaccadeError
from .local_vision import LocalVisionConfig, local_vision_config_from_args
from .version import PIPELINE_VERSION

DEFAULT_MAX_DURATION_SEC = 7200.0


@dataclass(frozen=True)
class SaccadeOptions:
    whisper_model: str = "small"
    whisper_language: str = "en"
    ocr: bool = True
    ocr_language: str = "eng"
    ocr_preprocess: bool = True
    redact_secrets: bool = True
    max_keyframes: int = 80
    min_interval_sec: float = 4.0
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC
    vad_filter: bool = True
    max_static_window_sec: float = 90.0
    cache_mode: str = "fingerprint"
    output_dir: str | None = None
    force_reprocess: bool = False
    caption_frames: bool = True
    local_vision_backend: str = "ollama"
    local_vision_model: str = "qwen3-vl:8b"
    local_vision_base_url: str = "http://127.0.0.1:11434"
    local_vision_timeout_sec: float = 30.0
    job_id: str = ""
    resume_partial: bool = True

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> SaccadeOptions:
        local_vision = local_vision_config_from_args(args)
        options = cls(
            whisper_model=str(args.get("whisper_model", cls.whisper_model)),
            whisper_language=str(args.get("whisper_language", cls.whisper_language)),
            ocr=bool(args.get("ocr", cls.ocr)),
            ocr_language=str(args.get("ocr_language", cls.ocr_language)),
            ocr_preprocess=bool(args.get("ocr_preprocess", cls.ocr_preprocess)),
            redact_secrets=bool(args.get("redact_secrets", cls.redact_secrets)),
            max_keyframes=int(args.get("max_keyframes", cls.max_keyframes)),
            min_interval_sec=float(args.get("min_interval_sec", cls.min_interval_sec)),
            max_duration_sec=float(args.get("max_duration_sec", cls.max_duration_sec)),
            vad_filter=bool(args.get("vad_filter", cls.vad_filter)),
            max_static_window_sec=float(
                args.get("max_static_window_sec", cls.max_static_window_sec)
            ),
            cache_mode=str(args.get("cache_mode", cls.cache_mode)),
            output_dir=args.get("output_dir"),
            force_reprocess=bool(args.get("force_reprocess", cls.force_reprocess)),
            caption_frames=local_vision.caption_frames,
            local_vision_backend=local_vision.backend,
            local_vision_model=local_vision.model,
            local_vision_base_url=local_vision.base_url,
            local_vision_timeout_sec=local_vision.timeout_sec,
            job_id=str(args.get("job_id") or f"saccade-{uuid4().hex}"),
            resume_partial=bool(args.get("resume_partial", cls.resume_partial)),
        )
        if options.cache_mode not in {"fingerprint", "content"}:
            raise SaccadeError(
                "E_BAD_OPTIONS",
                "options",
                "cache_mode must be 'fingerprint' or 'content'",
                {"cache_mode": options.cache_mode},
            )
        if options.max_keyframes < 1:
            raise SaccadeError("E_BAD_OPTIONS", "options", "max_keyframes must be positive")
        if options.min_interval_sec < 0:
            raise SaccadeError("E_BAD_OPTIONS", "options", "min_interval_sec must be >= 0")
        if options.max_duration_sec <= 0:
            raise SaccadeError("E_BAD_OPTIONS", "options", "max_duration_sec must be positive")
        if options.local_vision_backend != "ollama":
            raise SaccadeError(
                "E_BAD_OPTIONS",
                "options",
                "local_vision_backend must be 'ollama'",
                {"local_vision_backend": options.local_vision_backend},
            )
        return options

    def local_vision_config(self) -> LocalVisionConfig:
        return LocalVisionConfig(
            backend=self.local_vision_backend,
            model=self.local_vision_model,
            base_url=self.local_vision_base_url,
            timeout_sec=self.local_vision_timeout_sec,
            caption_frames=self.caption_frames,
        )

    def cache_payload(self, source_type: str) -> dict[str, Any]:
        payload = {
            "whisper_model": self.whisper_model,
            "whisper_language": self.whisper_language,
            "ocr": self.ocr,
            "ocr_language": self.ocr_language,
            "ocr_preprocess": self.ocr_preprocess,
            "redact_secrets": self.redact_secrets,
            "max_keyframes": self.max_keyframes,
            "min_interval_sec": self.min_interval_sec,
            "max_duration_sec": self.max_duration_sec,
            "vad_filter": self.vad_filter,
            "max_static_window_sec": self.max_static_window_sec,
            "caption_frames": self.caption_frames,
            "local_vision_backend": self.local_vision_backend,
            "local_vision_model": self.local_vision_model,
            "local_vision_base_url": self.local_vision_base_url,
            "local_vision_timeout_sec": self.local_vision_timeout_sec,
            "pipeline_version": PIPELINE_VERSION,
        }
        if source_type == "local":
            payload["cache_mode"] = self.cache_mode
        return payload

    def opts_hash(self, source_type: str) -> str:
        raw = json.dumps(self.cache_payload(source_type), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def public_dict(self, source_type: str) -> dict[str, Any]:
        payload = asdict(self)
        payload["opts_hash"] = self.opts_hash(source_type)
        payload.pop("job_id", None)
        return payload
