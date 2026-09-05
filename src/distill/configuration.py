"""Read and fold configuration once per resolution.

CLI > environment > file > default. DISTILL_OUTPUT_DIR and
DISTILL_ARTIFACT_DIR set the two output locations. distill.json is strict for
processing; distill.local-vision.json and its nested local_vision settings use
forgiving coercion. Directory discovery stays in config.py.

This module does not own diagnostic variables: DISTILL_TRACEBACK,
DISTILL_LOCAL_VISION_DEBUG, DISTILL_EFFECTIVE_TIMEOUT_MS, and
DISTILL_ENABLE_LONG_TIMEOUT_PROBE are read by their diagnostic boundaries.
Option types and defaults belong to options.py; settled endpoint validation
belongs to local_vision.py. Credentials never enter the public result view.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import config_dir
from .errors import DistillError, errno_name
from .frame_selection import MAX_CANDIDATE_SCHEDULE
from .local_vision import (
    ENDPOINT_FIELD_NAMES,
    MAX_SOCKET_TIMEOUT_SEC,
    LocalVisionConfig,
    SecretCredential,
    _with_validated_endpoint,
)
from .options import (
    GENERAL_OPTION_NAMES,
    NUMERIC_OPTION_DOMAINS,
    OPTION_SPECS,
    DistillOptions,
    annotate_configured_refusal,
    coerce_bool,
    validated_number,
    validated_option_type,
)

GENERAL_CONFIG_FILENAME = "distill.json"
LOCAL_VISION_SECTION = "local_vision"
CONFIG_FILENAMES = ("distill.local-vision.json", GENERAL_CONFIG_FILENAME)
OPTION_ENV_VARIABLES = {"artifact_dir": "DISTILL_ARTIFACT_DIR", "output_dir": "DISTILL_OUTPUT_DIR"}


def _bad_config_file(path: Path, message: str, **details: Any) -> DistillError:
    return DistillError("E_BAD_OPTIONS", "options", message, {"path": str(path), **details})


def _read_json_strict(path: Path) -> dict[str, Any]:
    """Strict reader for ``distill.json`` general schema (D-011).

    Absence is no configuration; unparsable, unreadable or not-an-object is
    ``E_BAD_OPTIONS`` at stage ``options`` naming the path.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        try:
            path.lstat()
        except (FileNotFoundError, NotADirectoryError):
            return {}
        except OSError as exc:
            raise _bad_config_file(
                path,
                "config file presence could not be checked",
                errno=errno_name(exc),
            ) from exc
        raise _bad_config_file(
            path,
            "config file points to a missing target",
            errno="ENOENT",
        ) from None
    except NotADirectoryError:
        return {}
    except UnicodeDecodeError as exc:
        raise _bad_config_file(path, "config file is not UTF-8 text", error=str(exc)) from exc
    except OSError as exc:
        raise _bad_config_file(
            path, "config file could not be read", errno=errno_name(exc)
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_config_file(path, "config file is not valid JSON", error=str(exc)) from exc
    if not isinstance(payload, dict):
        raise _bad_config_file(
            path, "config file must hold a JSON object", received=type(payload).__name__
        )
    return payload


def _read_json_forgiving(path: Path) -> dict[str, Any]:
    """Forgiving reader for local-vision files - a broken file is defaults."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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
    """A configured number, or the default when what arrived is not one.

    Coercion rather than refusal is this layer's contract - a config file that
    names an unusable timeout should not stop a run - but what it coerces *to*
    has to be a number the thing downstream can take. `inf` cleared the `> 0`
    test and reached `socket.settimeout`, which answered with an uncaught
    `OverflowError` from the stdlib; so does any finite value past
    `MAX_SOCKET_TIMEOUT_SEC`, which is why the bound is the socket's and not
    just `math.isfinite`. `nan` clears nothing and would have disabled the
    timeout by always comparing false. All of them are unusable in the way a
    string is, so all of them get the default.

    `bool` is refused for the reason `manifest_duration` refuses it: `True` is
    an `int` in Python, and a config file saying `"timeout_sec": true` is not a
    one-second timeout.
    """
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed) or parsed <= 0 or parsed >= MAX_SOCKET_TIMEOUT_SEC:
        return default
    return parsed


def _with_chain(config: LocalVisionConfig) -> LocalVisionConfig:
    """Every settled config carries an **endpoint chain**, so there is one path.

    A config that named no `endpoints` still names an endpoint - its top-level
    fields are one - and deriving that entry here is what keeps resolution from
    growing a second path for "the old shape". A one-entry chain that the
    multi-entry code does not handle is a one-entry chain nobody tested.

    Derived after validation, not before: the entry has to mirror the endpoint
    the run will actually use, and validation is what settles that.
    """
    if config.endpoints is not None:
        return config
    return replace(
        config,
        endpoints=(
            LocalVisionConfig(
                model=config.model,
                base_url=config.base_url,
                credential=config.credential,
                credential_configured=config.credential_configured,
                credential_env=config.credential_env,
                allow_remote_endpoint=config.allow_remote_endpoint,
            ),
        ),
    )


def _chain_after_overrides(
    config: LocalVisionConfig, overrides: dict[str, Any]
) -> LocalVisionConfig:
    """What `--local-vision-model` / `--local-vision-base-url` do to a chain.

    <!-- P3-D-016 --> The two flags are not symmetric, because identity is not.
    ADR-0004 keeps the address out of the **options hash**, so moving an
    endpoint's address reaches the same reader at a different place and changes
    no candidate key; the model *is* identity, and applying it to one entry
    would leave the rest naming readers nobody asked for under keys that
    describe them.

    So: the address alone moves entry 0 and leaves the chain otherwise intact.
    The model alone against a chain of more than one is refused, because
    choosing which entry it meant is a decision Distill does not get to make
    silently. Both together name one endpoint completely, and a run told to use
    that endpoint should not still carry others it might fall through to - so
    the chain is replaced.
    """
    chain = config.endpoints
    if chain is None:
        return config
    names_model = "model" in overrides
    names_address = "base_url" in overrides
    if names_model and names_address:
        return replace(config, endpoints=None)
    if names_model:
        if len(chain) > 1:
            raise DistillError(
                "E_BAD_OPTIONS",
                "local_vision",
                "--local-vision-model names one model but 'endpoints' names "
                f"{len(chain)} endpoints, and the model decides which bundle a run "
                "publishes under. Name --local-vision-base-url too to use a single "
                "endpoint, or edit the chain.",
                {"endpoints": len(chain)},
            )
        return replace(config, endpoints=None)
    if names_address:
        moved = replace(chain[0], base_url=str(overrides["base_url"]).rstrip("/"))
        return replace(config, endpoints=(moved, *chain[1:]))
    return config


def _resolved_credential(
    payload: dict[str, Any], base: LocalVisionConfig
) -> tuple[SecretCredential | None, bool, str]:
    """D-016: `api_key_env` names an env var and wins over inline `api_key`.

    Returns (credential, configured, env_name). `configured` is True whenever
    either key appeared, even if the resolved value is empty - validation
    needs that distinction to fail closed on a remote endpoint.
    """
    if "api_key" not in payload and "api_key_env" not in payload:
        return base.credential, base.credential_configured, base.credential_env
    value = str(payload.get("api_key") or "")
    env_name = str(payload.get("api_key_env") or "")
    if env_name:
        value = os.environ.get(env_name, "")
    if not value:
        return None, True, env_name
    return SecretCredential(value), True, env_name


def _endpoints_from_payload(
    payload: dict[str, Any],
    inherited: tuple[LocalVisionConfig, ...] | None,
) -> tuple[LocalVisionConfig, ...] | None:
    """The **endpoint chain** this layer configured, or the one it inherited.

    <!-- P3-D-010 --> Every entry is folded onto a fresh `LocalVisionConfig`,
    never onto the surrounding config: inheriting would give entry 2 entry 1's
    credential and its remote authorization, and the outgoing `Authorization`
    header is where that would first be visible.

    A layer that names no `endpoints` leaves the inherited chain alone. What a
    layer that *does* name one should do to an earlier layer's - replace it
    rather than concatenate - is asserted separately, and is why this returns
    the new chain whole rather than extending.
    """
    configured = payload.get("endpoints")
    if not isinstance(configured, list):
        return inherited
    return tuple(
        _config_from_payload(entry, LocalVisionConfig())
        for entry in configured
        if isinstance(entry, dict)
    )


def _config_from_payload(
    payload: dict[str, Any],
    base: LocalVisionConfig,
) -> LocalVisionConfig:
    if not payload:
        return base
    credential, credential_configured, credential_env = _resolved_credential(payload, base)
    return replace(
        base,
        endpoints=_endpoints_from_payload(payload, base.endpoints),
        credential=credential,
        credential_configured=credential_configured,
        credential_env=credential_env,
        backend=str(payload.get("backend", base.backend)),
        model=str(payload.get("model", base.model)),
        base_url=str(payload.get("base_url", base.base_url)).rstrip("/"),
        timeout_sec=_coerce_float(payload.get("timeout_sec"), base.timeout_sec),
        caption_frames=_coerce_bool(
            payload.get("caption_frames", base.caption_frames),
            base.caption_frames,
        ),
        allow_remote_endpoint=_coerce_bool(
            payload.get("allow_remote_endpoint", base.allow_remote_endpoint),
            base.allow_remote_endpoint,
        ),
    )


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Validated general and vision settings from the same configuration reads."""

    options: DistillOptions
    local_vision: LocalVisionConfig
    configured_from: dict[str, Path]


class ResolvedOptions(dict[str, Any]):
    """Arguments with file origins for values not superseded by another layer."""

    def __init__(self, values: Mapping[str, Any], *, configured_from: Mapping[str, Path]) -> None:
        super().__init__(values)
        self.configured_from = dict(configured_from)


def environment_options() -> dict[str, Any]:
    return {
        name: value
        for name, variable in OPTION_ENV_VARIABLES.items()
        if (value := os.environ.get(variable))
    }


def _resolve_general_layers(
    args: Mapping[str, Any], general_keys: Iterable[str], root: Path
) -> tuple[ResolvedOptions, dict[str, Any]]:
    payload = _read_json_strict(root / GENERAL_CONFIG_FILENAME)
    known = frozenset(general_keys)
    values = {key: value for key, value in payload.items() if key != LOCAL_VISION_SECTION}
    unknown = sorted(set(values) - known)
    if unknown:
        raise _bad_config_file(
            root / GENERAL_CONFIG_FILENAME,
            "config file names unknown options",
            unknown_options=unknown,
        )
    values = {
        key: value for key, value in values.items() if not (key == "output_dir" and value == "")
    }
    origins = dict.fromkeys(values, root / GENERAL_CONFIG_FILENAME)
    for layer in (environment_options(), args):
        for key, value in layer.items():
            if value is None and key in values:
                continue
            values[key] = value
            origins.pop(key, None)
    return ResolvedOptions(values, configured_from=origins), payload


def resolve_options(
    args: Mapping[str, Any],
    *,
    general_keys: Iterable[str] = GENERAL_OPTION_NAMES,
    base_dir: Path | None = None,
) -> ResolvedOptions:
    """Resolve general settings for tools that do not process a source."""
    resolved, _ = _resolve_general_layers(args, general_keys, base_dir or config_dir())
    return resolved


def general_config(base_dir: Path | None = None) -> dict[str, Any]:
    payload = _read_json_strict((base_dir or config_dir()) / GENERAL_CONFIG_FILENAME)
    return {key: value for key, value in payload.items() if key != LOCAL_VISION_SECTION}


def _resolve_vision_layers(
    args: Mapping[str, Any], root: Path, raw_general: dict[str, Any] | None = None
) -> LocalVisionConfig:
    config = LocalVisionConfig()
    named: list[str] = []
    for filename in CONFIG_FILENAMES:
        payload = (
            raw_general
            if filename == GENERAL_CONFIG_FILENAME and raw_general is not None
            else _read_json_forgiving(root / filename)
        )
        if filename == GENERAL_CONFIG_FILENAME:
            nested = payload.get(LOCAL_VISION_SECTION)
            payload = nested if isinstance(nested, dict) else {}
        named.extend(key for key in ENDPOINT_FIELD_NAMES if key in payload)
        config = _config_from_payload(payload, config)
    config = replace(config, top_level_endpoint_fields=tuple(dict.fromkeys(named)))
    overrides = {
        spec.vision_field: args[spec.name]
        for spec in OPTION_SPECS
        if spec.vision_field is not None and spec.name in args
    }
    config = _chain_after_overrides(config, overrides)
    return _with_chain(_with_validated_endpoint(_config_from_payload(overrides, config)))


def load_local_vision_config(base_dir: Path | None = None) -> LocalVisionConfig:
    """Read the forgiving vision-only view without general-option validation."""
    return _resolve_vision_layers({}, (base_dir or config_dir()).expanduser())


def local_vision_config_from_args(
    args: dict[str, Any], base_dir: Path | None = None
) -> LocalVisionConfig:
    return resolve_run_config(args, base_dir=base_dir).local_vision


def resolve_run_config(args: dict[str, Any], *, base_dir: Path | None = None) -> ResolvedRunConfig:
    root = base_dir or config_dir()
    resolved, payload = _resolve_general_layers(args, GENERAL_OPTION_NAMES, root)
    local_vision = _resolve_vision_layers(args, root.expanduser(), payload)
    values: dict[str, Any] = {}
    for spec in OPTION_SPECS:
        if spec.vision_field is not None:
            continue
        try:
            raw = validated_option_type(spec, resolved.get(spec.name, spec.default))
            if spec.boolean:
                values[spec.name] = coerce_bool(raw, bool(spec.default))
            elif spec.name in NUMERIC_OPTION_DOMAINS:
                values[spec.name] = validated_number(spec.name, raw)
            else:
                values[spec.name] = None if raw is None else spec.caster(raw)
        except DistillError as exc:
            annotate_configured_refusal(exc, resolved, spec.name)
            raise
    try:
        timeout = validated_number(
            "local_vision_timeout_sec",
            resolved.get("local_vision_timeout_sec", local_vision.timeout_sec),
        )
    except DistillError as exc:
        annotate_configured_refusal(exc, resolved, "local_vision_timeout_sec")
        raise
    local_vision = replace(local_vision, timeout_sec=timeout)
    values.update(
        {
            spec.name: getattr(local_vision, spec.vision_field)
            for spec in OPTION_SPECS
            if spec.vision_field is not None
        }
    )
    values["job_id"] = str(values["job_id"] or f"distill-{uuid4().hex}")
    options = DistillOptions(
        **values,
        local_vision_credential=local_vision.credential,
        local_vision_credential_configured=local_vision.credential_configured,
        local_vision_credential_env=local_vision.credential_env,
        local_vision_endpoints=local_vision.endpoints,
    )
    if options.cache_mode not in {"fingerprint", "content"}:
        raise annotate_configured_refusal(
            DistillError(
                "E_BAD_OPTIONS",
                "options",
                "cache_mode must be 'fingerprint' or 'content'",
                {"cache_mode": options.cache_mode},
            ),
            resolved,
            "cache_mode",
        )
    if options.local_vision_backend != "rapid-mlx":
        raise DistillError(
            "E_BAD_OPTIONS",
            "options",
            "local_vision_backend must be 'rapid-mlx'",
            {"local_vision_backend": options.local_vision_backend},
        )
    if options.max_duration_sec / options.max_static_window_sec > MAX_CANDIDATE_SCHEDULE:
        raise annotate_configured_refusal(
            DistillError(
                "E_BAD_OPTIONS",
                "options",
                f"max_duration_sec and max_static_window_sec would build a keyframe schedule of more than {MAX_CANDIDATE_SCHEDULE} candidates; widen the window or lower the duration cap",
                {
                    "max_duration_sec": options.max_duration_sec,
                    "max_static_window_sec": options.max_static_window_sec,
                },
            ),
            resolved,
            "max_static_window_sec",
        )
    return ResolvedRunConfig(options, local_vision, resolved.configured_from)
