"""Normalized tool options for Distill pipeline stages."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from .errors import DistillError
from .local_vision import (
    DEFAULT_LOCAL_VISION_BACKEND,
    DEFAULT_LOCAL_VISION_BASE_URL,
    DEFAULT_LOCAL_VISION_MODEL,
    DEFAULT_TIMEOUT_SEC,
    LocalVisionConfig,
    local_vision_config_from_args,
)
from .version import PIPELINE_VERSION

DEFAULT_MAX_DURATION_SEC = 7200.0


@dataclass(frozen=True)
class OptionSpec:
    name: str
    default: Any
    caster: Callable[[Any], Any]
    boolean: bool = False
    cache_key: bool = True


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


OPTION_SPECS: tuple[OptionSpec, ...] = (
    OptionSpec("whisper_model", "small", str),
    OptionSpec("whisper_language", "en", str),
    OptionSpec("ocr", True, bool, boolean=True),
    OptionSpec("ocr_language", "eng", str),
    OptionSpec("ocr_preprocess", True, bool, boolean=True),
    OptionSpec("redact_secrets", True, bool, boolean=True),
    OptionSpec("max_keyframes", 80, int),
    OptionSpec("min_interval_sec", 4.0, float),
    OptionSpec("max_duration_sec", DEFAULT_MAX_DURATION_SEC, float),
    OptionSpec("vad_filter", True, bool, boolean=True),
    OptionSpec("max_static_window_sec", 90.0, float),
    OptionSpec("cache_mode", "fingerprint", str, cache_key=False),
    OptionSpec("output_dir", None, lambda value: value, cache_key=False),
    OptionSpec("force_reprocess", False, bool, boolean=True, cache_key=False),
    OptionSpec("job_id", "", str, cache_key=False),
    OptionSpec("resume_partial", True, bool, boolean=True, cache_key=False),
)
OPTION_DEFAULTS = {spec.name: spec.default for spec in OPTION_SPECS}


@dataclass(frozen=True)
class NumericDomain:
    """The values one numeric option may take (R-03, R-47).

    Every numeric option is a quantity of something a run needs at least some
    of - frames, seconds, items - so the floor is zero for all of them and the
    only questions left are whether zero itself is a policy and whether the
    quantity is counted or measured. `admits_zero` answers the first,
    `integral` the second. Nothing here is infinite or unknown: a NaN limit
    cannot fire at all and an infinite one is an unbounded run reporting a
    bound.
    """

    integral: bool = False
    admits_zero: bool = False


NUMERIC_OPTION_DOMAINS: dict[str, NumericDomain] = {
    "max_keyframes": NumericDomain(integral=True),
    # Zero is a spacing policy, not a broken one: it means no gap is required
    # between two keyframes. It is the only numeric option zero means something
    # to, which is why the domain is per option rather than one global floor.
    "min_interval_sec": NumericDomain(admits_zero=True),
    "max_duration_sec": NumericDomain(),
    "max_static_window_sec": NumericDomain(),
    "local_vision_timeout_sec": NumericDomain(),
    # Not a `DistillOptions` field - `max_items` bounds a batch rather than a
    # bundle, so it is not part of the options hash - but it is an operator's
    # number arriving at the same boundary and it is refused the same way.
    "max_items": NumericDomain(integral=True),
}
"""Every numeric option a run is configured with, and the values each admits.

A table rather than a check per call site, because the check that is written
where a value is *used* is the one that is missing at the second use: the
duration cap was validated in `from_args` and the batch limit nowhere, and
`max_items=-1` became a slice taken from the wrong end.

Retention's numbers - `keep_generations` and `max_age_days` - are not here.
They belong to `PrunePolicy`, which validates them where the policy is built
(R-03) and reports them at the `prune` stage rather than as run options.
"""


def _bad_number(name: str, value: Any, domain: NumericDomain) -> DistillError:
    quantity = "a whole number" if domain.integral else "a finite number"
    floor = "0 or greater" if domain.admits_zero else "greater than 0"
    return DistillError(
        "E_BAD_OPTIONS",
        "options",
        f"{name} must be {quantity} {floor}",
        # `repr`, because the value that provoked this may be a NaN, and a
        # fatal error is published as JSON that a strict reader has to parse.
        {name: repr(value)},
    )


def validated_number(name: str, value: Any) -> int | float:
    """The value `name` may take, or `E_BAD_OPTIONS` naming what was wrong.

    Rejection happens here, at the boundary, because a number that is refused
    is still attached to the option it came from; three stages later a NaN is
    just a comparison that answers no. Booleans are refused for the reason
    `PrunePolicy` refuses them - `True` meaning 1 is a coincidence rather than
    an instruction - and a value that is not a number at all is refused as the
    same option error rather than escaping as a `ValueError`.

    It does not decide what any option *means*; `NUMERIC_OPTION_DOMAINS` owns
    that, and an option missing from it is a programming error, not an input.
    """
    domain = NUMERIC_OPTION_DOMAINS[name]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise _bad_number(name, value, domain)
    try:
        number = float(value)
    except ValueError:
        raise _bad_number(name, value, domain) from None
    if not math.isfinite(number):
        raise _bad_number(name, value, domain)
    if number < 0 or (number == 0 and not domain.admits_zero):
        raise _bad_number(name, value, domain)
    if domain.integral:
        # `int(2.7)` answered 2. Truncating an operator's number is the quiet
        # reinterpretation this validation exists to remove.
        if number != int(number):
            raise _bad_number(name, value, domain)
        return int(number)
    return number


def validated_count(name: str, value: Any) -> int:
    """`validated_number` for an option counted in whole units, typed as one."""
    return int(validated_number(name, value))


CACHE_OPTION_NAMES = tuple(spec.name for spec in OPTION_SPECS if spec.cache_key)
PROCESSING_OPTION_NAMES = tuple(spec.name for spec in OPTION_SPECS if spec.name != "cache_mode")
LOCAL_VISION_OPTION_NAMES = (
    "caption_frames",
    "local_vision_backend",
    "local_vision_model",
    "local_vision_base_url",
    "local_vision_timeout_sec",
    "local_vision_allow_remote_endpoint",
)


@dataclass(frozen=True)
class TranscriptionConfig:
    whisper_model: str
    whisper_language: str
    vad_filter: bool


@dataclass(frozen=True)
class FrameSelectionConfig:
    max_keyframes: int
    min_interval_sec: float
    max_static_window_sec: float


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool
    language: str
    preprocess: bool


@dataclass(frozen=True)
class CacheConfig:
    cache_mode: str
    output_dir: str | None
    force_reprocess: bool
    resume_partial: bool


@dataclass(frozen=True)
class DistillOptions:
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
    local_vision_backend: str = DEFAULT_LOCAL_VISION_BACKEND
    local_vision_model: str = DEFAULT_LOCAL_VISION_MODEL
    local_vision_base_url: str = DEFAULT_LOCAL_VISION_BASE_URL
    local_vision_timeout_sec: float = DEFAULT_TIMEOUT_SEC
    local_vision_allow_remote_endpoint: bool = False
    job_id: str = ""
    resume_partial: bool = True

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> DistillOptions:
        local_vision = local_vision_config_from_args(args)
        values: dict[str, Any] = {}
        for spec in OPTION_SPECS:
            default = OPTION_DEFAULTS[spec.name]
            raw_value = args.get(spec.name, default)
            if spec.boolean:
                values[spec.name] = _coerce_bool(raw_value, bool(default))
            elif spec.name in NUMERIC_OPTION_DOMAINS:
                # Before the `None` branch below: `None` is a legitimate value
                # for `output_dir` and an unusable one for a quantity.
                values[spec.name] = validated_number(spec.name, raw_value)
            elif raw_value is None:
                values[spec.name] = None
            else:
                values[spec.name] = spec.caster(raw_value)
        options = cls(
            whisper_model=values["whisper_model"],
            whisper_language=values["whisper_language"],
            ocr=values["ocr"],
            ocr_language=values["ocr_language"],
            ocr_preprocess=values["ocr_preprocess"],
            redact_secrets=values["redact_secrets"],
            max_keyframes=values["max_keyframes"],
            min_interval_sec=values["min_interval_sec"],
            max_duration_sec=values["max_duration_sec"],
            vad_filter=values["vad_filter"],
            max_static_window_sec=values["max_static_window_sec"],
            cache_mode=values["cache_mode"],
            output_dir=values["output_dir"],
            force_reprocess=values["force_reprocess"],
            caption_frames=local_vision.caption_frames,
            local_vision_backend=local_vision.backend,
            local_vision_model=local_vision.model,
            local_vision_base_url=local_vision.base_url,
            # The raw argument when the caller supplied one, because the config
            # layer's coercion answers an unusable number with the default -
            # reasonable for a config file, and silence where an operator just
            # named a timeout on the command line.
            local_vision_timeout_sec=validated_number(
                "local_vision_timeout_sec",
                args.get("local_vision_timeout_sec", local_vision.timeout_sec),
            ),
            local_vision_allow_remote_endpoint=local_vision.allow_remote_endpoint,
            job_id=str(values["job_id"] or f"distill-{uuid4().hex}"),
            resume_partial=values["resume_partial"],
        )
        if options.cache_mode not in {"fingerprint", "content"}:
            raise DistillError(
                "E_BAD_OPTIONS",
                "options",
                "cache_mode must be 'fingerprint' or 'content'",
                {"cache_mode": options.cache_mode},
            )
        # Every numeric floor is `NUMERIC_OPTION_DOMAINS`, applied above as the
        # value is read. A check here would run after construction, which is
        # after `local_vision_config_from_args` has already spent the value.
        if options.local_vision_backend != "rapid-mlx":
            raise DistillError(
                "E_BAD_OPTIONS",
                "options",
                "local_vision_backend must be 'rapid-mlx'",
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
            allow_remote_endpoint=self.local_vision_allow_remote_endpoint,
        )

    def transcription_config(self) -> TranscriptionConfig:
        return TranscriptionConfig(
            whisper_model=self.whisper_model,
            whisper_language=self.whisper_language,
            vad_filter=self.vad_filter,
        )

    def frame_selection_config(self) -> FrameSelectionConfig:
        return FrameSelectionConfig(
            max_keyframes=self.max_keyframes,
            min_interval_sec=self.min_interval_sec,
            max_static_window_sec=self.max_static_window_sec,
        )

    def ocr_config(self) -> OcrConfig:
        return OcrConfig(
            enabled=self.ocr,
            language=self.ocr_language,
            preprocess=self.ocr_preprocess,
        )

    def cache_config(self) -> CacheConfig:
        return CacheConfig(
            cache_mode=self.cache_mode,
            output_dir=self.output_dir,
            force_reprocess=self.force_reprocess,
            resume_partial=self.resume_partial,
        )

    def cache_payload(self, source_type: str) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in CACHE_OPTION_NAMES}
        payload.update({name: getattr(self, name) for name in LOCAL_VISION_OPTION_NAMES})
        payload["pipeline_version"] = PIPELINE_VERSION
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
