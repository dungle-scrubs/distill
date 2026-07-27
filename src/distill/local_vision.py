"""Local vision backend configuration, availability checks, and frame interpretation.

This module owns local-only vision provider setup and the frame-interpretation
pass: it decides whether a requested vision pass can run (or should degrade to
OCR-only output), and when it can, ``FrameInterpreter`` reads each **keyframe**,
requests an **interpretation**, grounds it against the frame's **extracted
text**, and hands back the **frame artifact** carrying both.

It does not own what an interpretation is made of - those field names are
``artifacts.Interpretation``'s, so that the one module reading them back
(``render``) and the one filling them in (this one) cannot drift apart - and it
no longer owns redaction. The policy runs where the model's words enter the
carrier (R-19, D-019); the post-hoc ``_redact_result_fields`` helper this module
used to apply afterwards is gone, along with the window in which an
interpretation existed unredacted.

Distill talks to a local Rapid-MLX server directly over its OpenAI-compatible
HTTP API. The server is assumed to already be running (``rapid-mlx serve
<model>``); Distill probes ``GET <base_url>/models`` for availability and posts
chat-completion requests to ``<base_url>/chat/completions``. Distill does not
manage the server lifecycle, and it has no dependency on any other local
runtime shim.
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .artifacts import FrameArtifact, Interpretation
from .errors import warning
from .grounding import UNGROUNDED, GroundingAssessment, assess_grounding
from .progress import ProgressReporter
from .vision_prompts import FRAME_KINDS, TEXT_CONFIDENCE_LEVELS

DEFAULT_LOCAL_VISION_BACKEND = "rapid-mlx"
DEFAULT_LOCAL_VISION_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"
# Rapid-MLX serves an OpenAI-compatible API under /v1 on port 8000 by default.
DEFAULT_LOCAL_VISION_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TIMEOUT_SEC = 30.0
# Small vision models intermittently emit non-JSON; one retry recovers most of
# them. Transport errors (timeout, unreachable) are not retried.
DEFAULT_MAX_ATTEMPTS = 2
CONFIG_FILENAMES = ("distill.local-vision.json", "distill.json")
DEBUG_ENV = "DISTILL_LOCAL_VISION_DEBUG"
# Cap on in-flight vision requests. Rapid-MLX batches internally, so Distill
# keeps a small fixed pool rather than fanning out unbounded. 1 == serial.
DEFAULT_MAX_PARALLEL = 1
FRAME_READ_FAILURE_CODES = frozenset(
    {
        "local_vision_malformed_response",
        "local_vision_timeout",
        "local_vision_image_read_failed",
    }
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalVisionConfig:
    backend: str = DEFAULT_LOCAL_VISION_BACKEND
    model: str = DEFAULT_LOCAL_VISION_MODEL
    base_url: str = DEFAULT_LOCAL_VISION_BASE_URL
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


ProbeLocalVision = Callable[[LocalVisionConfig], LocalVisionProbe]
TryInterpretImage = Callable[..., tuple["Interpretation | None", "dict[str, str] | None"]]


@dataclass(frozen=True)
class _FrameOutcome:
    """Per-frame interpretation result, merged back into the run in frame order.

    Produced (possibly off a worker thread) by ``_interpret_one`` without
    touching shared interpreter state; ``_merge_outcomes`` is the single ordered
    consumer that records warnings, counters, traces, and progress.
    """

    index: int
    frame: FrameArtifact
    interpreted: bool
    warnings: list[dict[str, str]]
    path: str
    has_extracted_text: bool
    warning_code: str | None


# Injectable HTTP entry points. Production uses urllib against the running
# Rapid-MLX server; hermetic tests monkeypatch these so no real server is
# required. ``HttpRequestor`` returns the decoded JSON body (or raises).
HttpRequestor = Callable[..., "dict[str, Any]"]


def _debug_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return value
    return os.environ.get(DEBUG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _boundary_log(event: str, detail: dict[str, Any]) -> None:
    LOGGER.debug(
        json.dumps(
            {
                "type": "distill.local_vision",
                "event": event,
                "detail": detail,
            },
            sort_keys=True,
        )
    )


def config_dir() -> Path:
    return Path(os.environ.get("DISTILL_CONFIG_DIR", Path.home() / ".distill")).expanduser()


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
        if filename == "distill.json":
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
    if config.backend == DEFAULT_LOCAL_VISION_BACKEND:
        return probe_rapid_mlx_availability(config)
    return LocalVisionProbe(
        available=False,
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        code="local_vision_backend_unsupported",
        message=f"Local vision backend '{config.backend}' is not supported; continuing with OCR-only output.",
        detail={"backend": config.backend},
    )


def probe_rapid_mlx_availability(
    config: LocalVisionConfig,
    *,
    requestor: HttpRequestor | None = None,
) -> LocalVisionProbe:
    """Confirm a Rapid-MLX server is reachable and serving the configured model.

    Probes ``GET <base_url>/models`` (the OpenAI-compatible models list). The
    probe is satisfied when the server responds and the configured model id is
    present in its catalog. Transport failures map onto Distill's existing
    warning codes so the OCR-only fallback behaves as before.
    """
    models_url = _models_url(config.base_url)
    try:
        payload = _http_get_json(requestor, models_url, config.timeout_sec)
    except TimeoutError as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_timeout",
            message="Rapid-MLX local vision probe timed out; continuing with OCR-only output.",
            detail={"error": str(exc), "url": models_url},
        )
    except urllib.error.URLError as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_rapid_mlx_unavailable",
            message="Rapid-MLX is unavailable for local vision; continuing with OCR-only output. Start it with `rapid-mlx serve <model>`.",
            detail={"error": str(exc), "url": models_url},
        )
    except OSError as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_rapid_mlx_unavailable",
            message="Rapid-MLX local vision connection dropped during probe; continuing with OCR-only output.",
            detail={"error": str(exc), "url": models_url},
        )
    except (ValueError, RuntimeError) as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_rapid_mlx_malformed_response",
            message="Rapid-MLX returned a malformed models response; continuing with OCR-only output.",
            detail={"error": str(exc), "url": models_url},
        )

    served = _served_model_ids(payload)
    if config.model not in served:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_model_unavailable",
            message=(
                f"Rapid-MLX is not serving model '{config.model}'; continuing with OCR-only output."
            ),
            detail={
                "configured_model": config.model,
                "served_models": served,
                "url": models_url,
            },
        )
    return LocalVisionProbe(
        available=True,
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        code="local_vision_available",
        message="Rapid-MLX local vision target is available.",
        detail={"served_models": served, "url": models_url},
    )


def interpret_image(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    *,
    prompt_profile: str = "technical",
) -> Interpretation:
    if config.backend != DEFAULT_LOCAL_VISION_BACKEND:
        raise LocalVisionFailure(
            "local_vision_backend_unsupported",
            f"Local vision backend '{config.backend}' is not supported.",
            {"backend": config.backend},
        )
    return _interpret_with_rapid_mlx(config, image_path, prompt, prompt_profile)


def interpret_image_after_probe(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    *,
    prompt_profile: str = "technical",
) -> Interpretation:
    return _interpret_with_rapid_mlx(config, image_path, prompt, prompt_profile)


def try_interpret_image(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    *,
    prompt_profile: str = "technical",
) -> tuple[Interpretation | None, dict[str, str] | None]:
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


def try_interpret_image_after_probe(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    *,
    prompt_profile: str = "technical",
) -> tuple[Interpretation | None, dict[str, str] | None]:
    try:
        return (
            interpret_image_after_probe(
                config,
                image_path,
                prompt,
                prompt_profile=prompt_profile,
            ),
            None,
        )
    except KeyboardInterrupt:
        failure = LocalVisionFailure(
            "local_vision_cancelled",
            "Local vision was cancelled; continuing with OCR-only output.",
        )
        return None, failure.warning()
    except LocalVisionFailure as exc:
        return None, exc.warning()


@dataclass
class FrameInterpreter:
    config: LocalVisionConfig
    progress: ProgressReporter | None = None
    probe: ProbeLocalVision = probe_local_vision
    try_interpret: TryInterpretImage = try_interpret_image
    max_parallel: int = DEFAULT_MAX_PARALLEL
    debug: bool | None = None
    _last_probe: LocalVisionProbe | None = None
    _frame_count: int = 0
    _interpreted_count: int = 0
    _max_parallel: int = 1
    _warning_counts: dict[str, int] = field(default_factory=dict)
    _trace_events: list[dict[str, Any]] = field(default_factory=list)

    def interpret(
        self, frames: list[FrameArtifact]
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        self._reset_run(frames)
        self._log("interpret.start", {"frames": len(frames), "backend": self.config.backend})
        probe = self.probe(self.config)
        self._last_probe = probe
        # Cap the worker pool at the configured admission limit; never unbounded.
        # max_parallel<=1 keeps interpretation serial — the config fallback if
        # parallel-vs-serial is not a measured win (A-004).
        self._max_parallel = max(int(self.max_parallel), 1)
        if not probe.available:
            if self.progress:
                self.progress.skip_cached("local_vision", detail={"reason": probe.code})
            probe_warning = probe.warning()
            self._record_warning(probe_warning)
            self._log("interpret.unavailable", {"code": probe.code})
            return frames, [probe_warning]

        warnings: list[dict[str, str]] = []
        interpreted_frames = frames
        try:
            if not frames:
                if self.progress:
                    self.progress.complete("local_vision", detail={"frames": 0})
                self._log("interpret.complete", {"frames": 0, "warnings": 0})
                return interpreted_frames, warnings

            # Resolve the target once via the probe, then interpret each frame
            # against that endpoint without re-probing. Use the resolved
            # model/base_url when the probe reported them.
            resolved = self._resolved_config(probe)
            self._log(
                "interpret.pool",
                {"frames": len(frames), "max_parallel": self._max_parallel},
            )
            interpreted_frames = self._interpret_frames(resolved, frames, warnings)
            if self.progress:
                self.progress.complete("local_vision", detail={"frames": len(frames)})
            self._log(
                "interpret.complete",
                {
                    "frames": len(interpreted_frames),
                    "interpreted_frames": self._interpreted_count,
                    "warnings": len(warnings),
                },
            )
            return interpreted_frames, warnings
        finally:
            # Nothing to release: Rapid-MLX manages its own lifecycle. The
            # finally block stays so the run-state semantics read the same as
            # the prior leasing path without leaking a no-op callback.
            pass

    def _resolved_config(self, probe: LocalVisionProbe) -> LocalVisionConfig:
        if probe.backend != DEFAULT_LOCAL_VISION_BACKEND or not (probe.base_url and probe.model):
            return self.config
        return replace(
            self.config,
            backend=DEFAULT_LOCAL_VISION_BACKEND,
            model=probe.model,
            base_url=probe.base_url,
        )

    def _interpret_frames(
        self,
        config: LocalVisionConfig,
        frames: list[FrameArtifact],
        warnings: list[dict[str, str]],
    ) -> list[FrameArtifact]:
        """Interpret every frame, capping concurrency at the admission limit.

        Each frame is interpreted into its own local warning list and interpreted
        flag; results are merged back in frame order so output ordering, warning
        ordering, and run-state counters are identical whether the pool ran
        serially (``max_parallel<=1``) or in parallel. ``_interpret_frame`` never
        mutates shared interpreter state, so it is safe to run on worker threads.
        """
        frame_count = len(frames)
        if self._max_parallel <= 1 or frame_count <= 1:
            outcomes = [
                self._interpret_one(config, index, frame, frame_count)
                for index, frame in enumerate(frames)
            ]
        else:
            workers = min(self._max_parallel, frame_count)
            ordered: list[_FrameOutcome | None] = [None] * frame_count
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for outcome in pool.map(
                    lambda item: self._interpret_one(config, item[0], item[1], frame_count),
                    list(enumerate(frames)),
                ):
                    ordered[outcome.index] = outcome
            outcomes = [outcome for outcome in ordered if outcome is not None]
        return self._merge_outcomes(outcomes, warnings)

    def _interpret_one(
        self,
        config: LocalVisionConfig,
        index: int,
        frame: FrameArtifact,
        frame_count: int,
    ) -> _FrameOutcome:
        self._ensure_frame_invariants(index, frame, frame_count)
        local_warnings: list[dict[str, str]] = []
        interpreted_frame, was_interpreted = self._interpret_frame(
            config, index, frame, frame_count, local_warnings
        )
        # The first frame warning (if any) is the read/transport warning surfaced
        # by _try_interpret_frame; trailing ones are grounding and carrier warnings.
        warning_code = local_warnings[0].get("code") if local_warnings else None
        return _FrameOutcome(
            index=index,
            frame=interpreted_frame,
            interpreted=was_interpreted,
            warnings=local_warnings,
            path=frame.path,
            has_extracted_text=bool(frame.extracted_text.strip()),
            warning_code=warning_code,
        )

    def _merge_outcomes(
        self,
        outcomes: list[_FrameOutcome],
        warnings: list[dict[str, str]],
    ) -> list[FrameArtifact]:
        frame_count = len(outcomes)
        frames: list[FrameArtifact] = []
        for outcome in outcomes:
            self._trace(
                "frame.start",
                {
                    "index": outcome.index + 1,
                    "path": outcome.path,
                    "has_extracted_text": outcome.has_extracted_text,
                },
            )
            frames.append(outcome.frame)
            if outcome.interpreted:
                self._interpreted_count += 1
            for frame_warning in outcome.warnings:
                warnings.append(frame_warning)
                self._record_warning(frame_warning)
            if self.progress:
                self.progress.update(
                    "local_vision",
                    percent=((outcome.index + 1) / frame_count) * 100,
                    detail={"frame": outcome.index + 1, "frames": frame_count},
                )
            self._trace(
                "frame.complete",
                {
                    "index": outcome.index + 1,
                    "interpreted": outcome.interpreted,
                    "warning_code": outcome.warning_code,
                },
            )
        return frames

    def debug_info(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend,
            "model": self.config.model,
            "debug_enabled": self._debug_enabled,
            "last_probe": None
            if self._last_probe is None
            else {
                "available": self._last_probe.available,
                "code": self._last_probe.code,
                "model": self._last_probe.model,
            },
            "frame_count": self._frame_count,
            "interpreted_count": self._interpreted_count,
            "max_parallel": self._max_parallel,
            "warning_counts": dict(sorted(self._warning_counts.items())),
            "trace_events": list(self._trace_events),
        }

    @property
    def _debug_enabled(self) -> bool:
        return _debug_enabled(self.debug)

    def _interpret_frame(
        self,
        config: LocalVisionConfig,
        index: int,
        frame: FrameArtifact,
        frame_count: int,
        warnings: list[dict[str, str]],
    ) -> tuple[FrameArtifact, bool]:
        """Interpret a single frame without mutating shared interpreter state.

        Appends any frame warnings to the supplied (per-frame) ``warnings`` list
        and returns the interpreted frame plus whether a result was produced.
        Trace events, progress updates, and counters are emitted by the caller in
        frame order so the parallel path stays deterministic and thread-safe.

        Nothing is mutated in place because nothing can be: a **frame artifact**
        is frozen, so what comes back is the next frame in the chain and the
        model's words went through the **redaction** policy on the way into it.
        """
        from .vision_prompts import build_technical_frame_prompt

        self._ensure_frame_invariants(index, frame, frame_count)
        prompt = build_technical_frame_prompt(ocr_text=frame.extracted_text or None)
        result, frame_warning = self._try_interpret_frame(
            config,
            Path(frame.path),
            prompt.prompt,
            prompt_profile=prompt.profile,
        )
        if frame_warning:
            warnings.append(frame_warning)
        if result:
            return self._interpreted(frame, result, index, warnings), True
        if frame_warning and frame_warning.get("code") in FRAME_READ_FAILURE_CODES:
            unusable = GroundingAssessment(
                UNGROUNDED,
                None,
                f"vision model produced no usable output ({frame_warning.get('code')})",
            )
            carried, carrier_warnings = frame.with_interpretation(
                None, grounding=unusable.public_dict()
            )
            warnings.extend(carrier_warnings)
            return carried, False
        return frame, False

    def _try_interpret_frame(
        self,
        config: LocalVisionConfig,
        image_path: Path,
        prompt: str,
        *,
        prompt_profile: str,
    ) -> tuple[Interpretation | None, dict[str, str] | None]:
        if self.try_interpret is try_interpret_image:
            return try_interpret_image_after_probe(
                config,
                image_path,
                prompt,
                prompt_profile=prompt_profile,
            )
        return self.try_interpret(
            config,
            image_path,
            prompt,
            prompt_profile=prompt_profile,
        )

    def _interpreted(
        self,
        frame: FrameArtifact,
        reading: Interpretation,
        index: int,
        warnings: list[dict[str, str]],
    ) -> FrameArtifact:
        """The frame carrying the model's reading and Distill's grounding of it.

        **Grounding** is assessed against the reading the model returned, and
        the reading is then handed to the carrier, which is where the
        **redaction** policy runs over every field of it (R-19). Assessing
        first is deliberate: grounding compares two readers' words, and a
        policy that replaced a secret in one of them would make the comparison
        about what redaction did rather than about what was read.
        """
        assessment = assess_grounding(
            ocr_text=frame.extracted_text,
            verbatim_text=reading.verbatim_text,
            text_confidence=reading.text_confidence,
            has_interpretation=reading.has_interpretation,
        )
        carried, carrier_warnings = frame.with_interpretation(
            reading, grounding=assessment.public_dict()
        )
        warnings.extend(carrier_warnings)
        if assessment.level == UNGROUNDED:
            warnings.append(
                warning(
                    "local_vision",
                    "frame_text_ungrounded",
                    f"frame {frame.index or index + 1} interpretation is not grounded in "
                    f"readable text: {assessment.reason}",
                )
            )
        return carried

    def _reset_run(self, frames: list[FrameArtifact]) -> None:
        self._last_probe = None
        self._frame_count = len(frames)
        self._interpreted_count = 0
        self._max_parallel = 1
        self._warning_counts = {}
        self._trace_events = []

    def _record_warning(self, warning_payload: dict[str, str]) -> None:
        code = warning_payload.get("code", "unknown")
        self._warning_counts[code] = self._warning_counts.get(code, 0) + 1

    def _log(self, event: str, detail: dict[str, Any]) -> None:
        if self._debug_enabled:
            self._trace(event, detail)
        _boundary_log(event, detail)

    def _trace(self, event: str, detail: dict[str, Any]) -> None:
        if not self._debug_enabled:
            return
        self._trace_events.append({"event": event, "detail": detail})

    @staticmethod
    def _ensure_frame_invariants(index: int, frame: FrameArtifact, frame_count: int) -> None:
        if frame_count <= 0:
            raise AssertionError("frame_count must be positive while interpreting frames")
        if not 0 <= index < frame_count:
            raise AssertionError("frame index must be within frame_count")
        if not frame.path:
            raise AssertionError(f"frame at index {index} is missing required 'path'")


def _interpret_with_rapid_mlx(
    config: LocalVisionConfig,
    image_path: Path,
    prompt: str,
    prompt_profile: str,
    *,
    requestor: HttpRequestor | None = None,
) -> Interpretation:
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
        'Leave verbatim_text empty and set text_confidence to "none" if you cannot read the text.'
    )
    data_uri = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    body = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    completions_url = _completions_url(config.base_url)
    last_preview = ""
    for _attempt in range(DEFAULT_MAX_ATTEMPTS):
        try:
            envelope = _http_post_json(requestor, completions_url, body, config.timeout_sec)
        except TimeoutError as exc:
            raise LocalVisionFailure(
                "local_vision_timeout",
                "Local vision timed out; continuing with OCR-only output.",
                {"timeout_sec": config.timeout_sec, "error": str(exc)},
            ) from exc
        except urllib.error.URLError as exc:
            raise LocalVisionFailure(
                "local_vision_rapid_mlx_unavailable",
                "Rapid-MLX local vision target was unreachable during generation; continuing with OCR-only output.",
                {"error": str(exc), "url": completions_url},
            ) from exc
        except OSError as exc:
            # e.g. ConnectionResetError while reading the response body.
            raise LocalVisionFailure(
                "local_vision_rapid_mlx_unavailable",
                "Rapid-MLX local vision connection dropped during generation; continuing with OCR-only output.",
                {"error": str(exc), "url": completions_url},
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise LocalVisionFailure(
                "local_vision_malformed_response",
                "Rapid-MLX local vision returned malformed JSON; continuing with OCR-only output.",
                {"error": str(exc), "url": completions_url},
            ) from exc
        raw_response = _chat_content(envelope)
        interpreted = parse_interpretation_json(raw_response)
        if interpreted is not None:
            return _result_from_payload(interpreted, config, prompt_profile)
        last_preview = raw_response[:200]
    raise LocalVisionFailure(
        "local_vision_malformed_response",
        "Rapid-MLX local vision returned a malformed interpretation; continuing with OCR-only output.",
        {"response_preview": last_preview, "attempts": DEFAULT_MAX_ATTEMPTS},
    )


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _served_model_ids(payload: Any) -> list[str]:
    """Best-effort extraction of model ids from an OpenAI-style /v1/models body."""
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for entry in data:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict):
            model_id = entry.get("id") or entry.get("model")
            if isinstance(model_id, str):
                ids.append(model_id)
    return ids


def _chat_content(envelope: Any) -> str:
    if not isinstance(envelope, dict):
        return ""
    choices = envelope.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def _http_get_json(
    requestor: HttpRequestor | None,
    url: str,
    timeout_sec: float,
) -> dict[str, Any]:
    if requestor is not None:
        payload = requestor(method="GET", url=url, timeout=timeout_sec)
    else:
        payload = _urlopen_json("GET", url, body=None, timeout_sec=timeout_sec)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def _http_post_json(
    requestor: HttpRequestor | None,
    url: str,
    body: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    if requestor is not None:
        payload = requestor(method="POST", url=url, body=body, timeout=timeout_sec)
    else:
        payload = _urlopen_json("POST", url, body=body, timeout_sec=timeout_sec)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def _urlopen_json(method: str, url: str, body: dict[str, Any] | None, timeout_sec: float) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Surface HTTP error bodies (e.g. model-not-loaded) as a RuntimeError so
        # the probe/interpret paths can map them onto warning codes.
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:200]}") from exc
    except http.client.IncompleteRead as exc:
        # A server that closes the connection mid-body (crash/restart) yields a
        # truncated read; treat it as a malformed response so we degrade cleanly.
        raise RuntimeError(f"incomplete response from {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"non-JSON response from {url}: {raw[:200]!r}") from exc


def _result_from_payload(
    interpreted: dict[str, Any],
    config: LocalVisionConfig,
    prompt_profile: str,
) -> Interpretation:
    elements = interpreted.get("detected_elements", [])
    if not isinstance(elements, list):
        elements = []
    return Interpretation(
        visual_summary=str(interpreted.get("visual_summary", "")).strip(),
        detected_elements=tuple(str(item) for item in elements),
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
