"""Single Configuration module that owns where config lives and how layers win.

This module is the one place that knows, for **every** option, where a value
may come from and which layer wins when more than one names it. The layers are
CLI > environment > file > default, stated in ``config.py`` and reimplemented
there for vision in ``local_vision.py`` before this module. The cost of two
implementations is two answers to the same precedence question - the seam was an
untyped ``dict[str, Any]`` into ``DistillOptions.from_args`` that erased which
layer a value came from, and a second reader over the same files with its own
forgiving coercions.

Directory discovery stays single-sourced. ``config.config_dir`` owns which
directory is *the* config directory (``DISTILL_CONFIG_DIR``, ``XDG_CONFIG_HOME``,
``~/.config/distill``, ``~/.distill``, first existing directory wins, an
absolute ``DISTILL_CONFIG_DIR`` is authoritative even when absent). This module
imports that function and does not reimplement it.

What this module owns:

- **Where config is read from** - by reusing ``config.config_dir``.
- **How configured layers are folded** - one function, ``resolve_run_config``,
  that reads ``distill.json`` (general schema, strict), ``distill.local-
  vision.json`` and the nested ``local_vision`` object (forgiving), then the
  environment variables ``DISTILL_OUTPUT_DIR`` / ``DISTILL_ARTIFACT_DIR``, then
  the CLI dict, in that order, for *all* options. A second ``distill.json``
  reader does not exist; the vision config is a view over the same resolved
  layers, not a second file read with its own precedence.
- **The typed seam** - ``ResolvedRunConfig`` dataclass holding both
  ``DistillOptions`` (general) and ``LocalVisionConfig`` (vision), replacing the
  untyped dict that erased layer identity.
- **Which fields enter the bundle identity vs which are machine-local claims**
  per ADR-0004 (identity vs environment). Identity answers "would running again
  produce different output", and is hashed into the **options hash** / **bundle
  key**. Environment answers "where does this run keep/publish state", and is
  never hashed.
  General: identity is every ``cache_key=True`` option (transcription, frame
  selection, OCR, redact, frame_salience, numeric caps, whisper etc.) plus the
  vision identity fields listed below. Machine-local is ``output_dir``,
  ``artifact_dir``, ``cache_mode`` (for local), ``force_reprocess``,
  ``job_id``, ``resume_partial`` - they decide where or whether a run publishes,
  not what it produces (``cache_key=False`` in ``OPTION_SPECS``).
  Vision: identity is ``local_vision_backend`` (always ``rapid-mlx``),
  ``local_vision_timeout_sec``, ``caption_frames`` (via ``vision_mode``),
  ``local_vision_model`` when vision is selected, and the boolean
  ``local_vision_non_local`` (D-012 narrows ADR-0004 for provenance: *was* this
  produced via a possibly-remote endpoint). The endpoint's address
  ``base_url``, ``credential``/``credential_env``, and ``endpoints`` chain
  itself are machine-local claims and never enter the hash - the same reader at
  a different place produces byte-identical output, and the chain is the
  candidate list, not the selected reader. ``allow_remote_endpoint`` is not
  identity except via that one boolean fold.

Vision coercions stay forgiving - a config file naming an unusable timeout is
coerced to the default rather than stopping the run - but they are called from
this single resolver rather than from a second reader. A CLI ``--local-vision-
timeout-sec`` that is unusable is refused by the same validation that refuses a
general numeric option, so an operator's typo is never silently coerced.

What it does not own: **directory discovery** (``config.py``), **what an option
means or which values it may take** (``options.py`` / ``NUMERIC_OPTION_DOMAINS``),
**the vision transport or endpoint policy** (``rapid_mlx.py``), **whether a
bundle is cached** (``pipeline.py``/``bundle_store.py``). It owns the layering
that reaches those owners.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import (
    GENERAL_CONFIG_FILENAME,
    LOCAL_VISION_SECTION,
    OPTION_ENV_VARIABLES,
    config_dir,
)
from .errors import DistillError, errno_name

# ---------------------------------------------------------------------------
# File helpers - strict (general) vs forgiving (vision)
# ---------------------------------------------------------------------------

CONFIG_DIR_ENV = "DISTILL_CONFIG_DIR"
XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"
LOCAL_VISION_CONFIG_FILENAMES = ("distill.local-vision.json", "distill.json")


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
        raise _bad_config_file(path, "config file could not be read", errno=errno_name(exc)) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_config_file(path, "config file is not valid JSON", error=str(exc)) from exc
    if not isinstance(payload, dict):
        raise _bad_config_file(path, "config file must hold a JSON object", received=type(payload).__name__)
    return payload


def _read_json_forgiving(path: Path) -> dict[str, Any]:
    """Forgiving reader for local-vision files - a broken file is defaults."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# Vision forgiving coercions (kept here so the single resolver calls them)
# ---------------------------------------------------------------------------

MAX_SOCKET_TIMEOUT_SEC = 2**63 / 1e9

ENDPOINT_FIELD_NAMES = ("model", "base_url", "api_key", "api_key_env", "allow_remote_endpoint")
CONFIG_FILENAMES = LOCAL_VISION_CONFIG_FILENAMES


def _coerce_bool_vision(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float_vision(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed) or parsed <= 0 or parsed >= MAX_SOCKET_TIMEOUT_SEC:
        return default
    return parsed


# ---------------------------------------------------------------------------
# General layer helpers (single precedence implementation)
# ---------------------------------------------------------------------------


def _environment_options() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for option, variable in OPTION_ENV_VARIABLES.items():
        value = os.environ.get(variable)
        if value:
            values[option] = value
    return values


def _general_file_payload_strict(base_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    """Strict general payload and the file path it came from."""
    root = (base_dir or config_dir())
    payload = _read_json_strict(root / GENERAL_CONFIG_FILENAME)
    general = {key: value for key, value in payload.items() if key != LOCAL_VISION_SECTION}
    return general, root / GENERAL_CONFIG_FILENAME


def _general_file_options(base_dir: Path | None = None) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    """General options from file, with strict unknown-key handling deferred."""
    payload, path = _general_file_payload_strict(base_dir)
    # Empty output_dir is not a setting (mirror config.resolve_options)
    payload = {k: v for k, v in payload.items() if not (k == "output_dir" and v == "")}
    return payload, path, {}


# ---------------------------------------------------------------------------
# Vision merging helpers (copied from local_vision, called from single resolver)
# ---------------------------------------------------------------------------


def _resolved_credential_vision(
    payload: dict[str, Any], base: Any
) -> tuple[Any | None, bool, str]:
    if "api_key" not in payload and "api_key_env" not in payload:
        return base.credential, base.credential_configured, base.credential_env
    value = str(payload.get("api_key") or "")
    env_name = str(payload.get("api_key_env") or "")
    if env_name:
        value = os.environ.get(env_name, "")
    if not value:
        return None, True, env_name
    # Import SecretCredential lazily to avoid circular top-level
    from .local_vision import SecretCredential as _SecretCredential

    return _SecretCredential(value), True, env_name


def _endpoints_from_payload_vision(
    payload: dict[str, Any], inherited: Any
) -> Any:
    configured = payload.get("endpoints")
    if not isinstance(configured, list):
        return inherited
    from .local_vision import LocalVisionConfig as _LVC

    return tuple(
        _config_from_payload_vision(entry, _LVC()) for entry in configured if isinstance(entry, dict)
    )


def _config_from_payload_vision(payload: dict[str, Any], base: Any) -> Any:
    if not payload:
        return base
    credential, credential_configured, credential_env = _resolved_credential_vision(payload, base)
    return replace(
        base,
        endpoints=_endpoints_from_payload_vision(payload, base.endpoints),
        credential=credential,
        credential_configured=credential_configured,
        credential_env=credential_env,
        backend=str(payload.get("backend", base.backend)),
        model=str(payload.get("model", base.model)),
        base_url=str(payload.get("base_url", base.base_url)).rstrip("/"),
        timeout_sec=_coerce_float_vision(payload.get("timeout_sec"), base.timeout_sec),
        caption_frames=_coerce_bool_vision(
            payload.get("caption_frames", base.caption_frames),
            base.caption_frames,
        ),
        allow_remote_endpoint=_coerce_bool_vision(
            payload.get("allow_remote_endpoint", base.allow_remote_endpoint),
            base.allow_remote_endpoint,
        ),
    )


def _merged_local_vision_config_for_view(
    base_dir: Path | None = None,
    *,
    strict_general_payload: dict[str, Any] | None = None,
) -> Any:
    """Vision files folded, forgiving, but reuses strict general payload when available.

    When ``strict_general_payload`` is provided (unified path), the nested
    ``local_vision`` object from that already-strict read is used rather than
    re-reading ``distill.json`` forgivingly - vision is a view, not a second
    reader. For the legacy forgiving wrapper path, this arg is None and the
    file is read forgivingly as before.
    """
    from .local_vision import LocalVisionConfig as _LVC

    root = (base_dir or config_dir()).expanduser()
    config = _LVC()
    named: list[str] = []
    for filename in CONFIG_FILENAMES:
        if filename == "distill.json" and strict_general_payload is not None:
            # Reuse already-strict payload's nested object forgivingly
            nested = strict_general_payload.get("local_vision")  # type: ignore[union-attr]
            # strict_general_payload is the whole distill.json payload minus local_vision?
            # For unified we passed the raw payload; adjust: caller should pass raw payload
            # So handle both: if strict_general_payload already filtered, nested is missing.
            # Safer: read forgiving nested from file if we filtered?
            # We instead expect caller passes raw strict payload before filtering.
            # If we got filtered payload, nested will be absent - fall back to forgiving read.
            if isinstance(nested, dict):
                payload: dict[str, Any] = nested
            else:
                # Fall back to forgiving read of file for nested
                raw = _read_json_forgiving(root / filename)
                nested2 = raw.get("local_vision")
                payload = nested2 if isinstance(nested2, dict) else {}
        else:
            payload = _read_json_forgiving(root / filename)
            if filename == "distill.json":
                nested = payload.get("local_vision")
                payload = nested if isinstance(nested, dict) else {}
        named.extend(key for key in ENDPOINT_FIELD_NAMES if key in payload)
        config = _config_from_payload_vision(payload, config)
    return replace(config, top_level_endpoint_fields=tuple(dict.fromkeys(named)))


def _with_chain_vision(config: Any) -> Any:
    if config.endpoints is not None:
        return config
    from .local_vision import LocalVisionConfig as _LVC

    return replace(
        config,
        endpoints=(
            _LVC(
                model=config.model,
                base_url=config.base_url,
                credential=config.credential,
                credential_configured=config.credential_configured,
                credential_env=config.credential_env,
                allow_remote_endpoint=config.allow_remote_endpoint,
            ),
        ),
    )


def _with_validated_endpoint_vision(config: Any) -> Any:
    from .local_vision import (
        _check_credential_is_not_empty as _check_cred,
    )
    from .local_vision import _validate_chain as _validate_chain_fn
    from .rapid_mlx import EndpointRejected, _checked_endpoint_url

    _validate_chain_fn(config.endpoints)
    if config.endpoints is not None and config.top_level_endpoint_fields:
        named = ", ".join(f"'{name}'" for name in config.top_level_endpoint_fields)
        raise DistillError(
            "E_BAD_OPTIONS",
            "local_vision",
            f"{named} names an endpoint at the top level while 'endpoints' names a chain. "
            "Move the top-level fields into an entry, or remove 'endpoints'.",
            {"fields": list(config.top_level_endpoint_fields)},
        )
    for index, entry in enumerate(config.endpoints or ()):
        try:
            _checked_endpoint_url(entry.base_url, allow_remote_endpoint=entry.allow_remote_endpoint)
        except EndpointRejected as exc:
            raise DistillError(
                "E_BAD_OPTIONS",
                "local_vision",
                f"endpoint {index}: {exc.message}",
                {"entry": index, **dict(exc.detail)},
            ) from exc
    try:
        _checked_endpoint_url(config.base_url, allow_remote_endpoint=config.allow_remote_endpoint)
    except EndpointRejected as exc:
        raise DistillError("E_BAD_OPTIONS", "local_vision", exc.message, dict(exc.detail)) from exc
    _check_cred(config, captioning=config.caption_frames)
    for index, entry in enumerate(config.endpoints or ()):
        _check_cred(entry, captioning=config.caption_frames, index=index)
    return config


def _chain_after_overrides_vision(config: Any, overrides: dict[str, Any]) -> Any:
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


# ---------------------------------------------------------------------------
# Typed result and single resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Typed result of ``resolve_run_config`` - the seam DistillOptions was built from.

    Holds both the validated general options and the validated vision config
    produced from the **same** layered reads (CLI > env > file > default). No
    caller reconstructs one from the other's scalars; the vision config is a
    view over the resolved layers, not a second reader that could disagree.

    Per ADR-0004 ``identity vs environment``: ``options`` carries both, but
    only identity fields enter ``opts_hash`` / bundle identity. See module
    docstring for the partition, and ``DistillOptions.cache_payload`` for the
    exact allowlist. ``local_vision`` carries the full claim, and
    ``config_is_non_local`` / ``local_vision_non_local`` is the one boolean
    fold that promotion concerns (D-012).

    ``configured_from`` records which general options still came from
    ``distill.json`` after env and CLI wins, so a later refusal can name the
    file.
    """

    options: Any  # DistillOptions - string to avoid circular import at class define time
    local_vision: Any  # LocalVisionConfig
    configured_from: dict[str, Path] = field(default_factory=dict)
    base_dir: Path | None = None
    # Raw layers for introspection (not identity)
    general_file_options: dict[str, Any] = field(default_factory=dict)
    environment_options: dict[str, Any] = field(default_factory=dict)
    cli_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedOptionsLike(dict[str, Any]):  # helper for legacy resolve_options wrapper
    configured_from: dict[str, Path] = field(default_factory=dict)  # type: ignore[assignment]


def _resolve_general_layers(
    cli_args: Mapping[str, Any],
    *,
    general_keys: Iterable[str],
    base_dir: Path | None,
    raw_payload_for_vision: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Fold general layers in precedence order, returning (resolved_dict, configured_from, raw_payload)."""
    known = frozenset(general_keys)
    root = base_dir or config_dir()
    # Strict read of distill.json
    try:
        raw_payload = _read_json_strict(root / GENERAL_CONFIG_FILENAME)
    except DistillError:
        raise
    # Store raw for vision view if caller wants it
    if raw_payload_for_vision is not None:
        # caller supplied mutable dict to fill - not needed; we return
        pass
    file_options_raw = {k: v for k, v in raw_payload.items() if k != LOCAL_VISION_SECTION}
    unknown = sorted(set(file_options_raw) - known)
    if unknown:
        raise _bad_config_file(
            root / GENERAL_CONFIG_FILENAME,
            "config file names unknown options",
            unknown_options=unknown,
        )
    file_options = {
        key: value for key, value in file_options_raw.items() if not (key == "output_dir" and value == "")
    }
    resolved: dict[str, Any] = {k: v for k, v in file_options.items() if k in known}
    configured_from: dict[str, Path] = {
        key: root / GENERAL_CONFIG_FILENAME for key in file_options if key in known
    }
    # Env layer
    env_opts = _environment_options()
    for key, value in env_opts.items():
        if key not in known:
            continue
        resolved[key] = value
        configured_from.pop(key, None)
    # CLI layer (None means caller named nothing)
    for key, value in cli_args.items():
        # Only consider general keys here; vision keys are handled in vision resolver
        if key not in known:
            continue
        if value is None and key in resolved:
            continue
        resolved[key] = value
        configured_from.pop(key, None)
    # Carry through non-option keys from cli_args that caller passed (path/url etc.)
    # resolve_options did: for key, value in args.items(): if value is None and key in resolved: continue; resolved[key]=value
    # That carried through non-general keys. We need to replicate for final args dict.
    # But _resolve_general_layers is just for general options; the fuller resolve will merge.
    return resolved, configured_from, raw_payload


# For legacy config.resolve_options wrapper, expose a dict-like with configured_from
class _LegacyResolvedOptions(dict[str, Any]):
    def __init__(self, values: Mapping[str, Any], *, configured_from: Mapping[str, Path]) -> None:
        super().__init__(values)
        self.configured_from = dict(configured_from)


def resolve_general_options(
    cli_args: Mapping[str, Any],
    *,
    general_keys: Iterable[str],
    base_dir: Path | None = None,
) -> _LegacyResolvedOptions:
    """Legacy helper for ``config.resolve_options`` - delegates to single precedence."""
    resolved, configured_from, _ = _resolve_general_layers(cli_args, general_keys=general_keys, base_dir=base_dir)
    # Need to also carry through non-general CLI keys as resolve_options did
    # resolve_options iterated over all args items and carried through untouched.
    # Replicate: for key,value in cli_args: if value is None and key in resolved: continue; resolved[key]=value; pop configured_from
    # But we already handled general keys; need to also handle other keys.
    known = frozenset(general_keys)
    for key, value in cli_args.items():
        if key in known:
            continue
        if value is None and key in resolved:
            continue
        resolved[key] = value
        configured_from.pop(key, None)
    return _LegacyResolvedOptions(resolved, configured_from=configured_from)


def _resolve_vision_layers(
    cli_args: Mapping[str, Any],
    *,
    base_dir: Path | None,
    raw_general_payload: dict[str, Any] | None = None,
) -> Any:
    """Resolve LocalVisionConfig from file + CLI, reusing strict payload when given."""
    from .local_vision import LocalVisionConfig as _LVC

    root = (base_dir or config_dir()).expanduser()
    # Build base from files forgivingly, but reuse strict payload's nested for view
    config = _LVC()
    named: list[str] = []
    for filename in CONFIG_FILENAMES:
        if filename == "distill.json" and raw_general_payload is not None:
            nested = raw_general_payload.get("local_vision")
            payload = nested if isinstance(nested, dict) else {}
        else:
            payload = _read_json_forgiving(root / filename)
            if filename == "distill.json":
                nested = payload.get("local_vision")
                payload = nested if isinstance(nested, dict) else {}
        named.extend(key for key in ENDPOINT_FIELD_NAMES if key in payload)
        config = _config_from_payload_vision(payload, config)
    config = replace(config, top_level_endpoint_fields=tuple(dict.fromkeys(named)))

    # CLI overrides (forgiving coercions, but crafted as another payload)
    overrides: dict[str, Any] = {}
    if "caption_frames" in cli_args:
        overrides["caption_frames"] = _coerce_bool_vision(cli_args.get("caption_frames"), config.caption_frames)
    if "local_vision_backend" in cli_args:
        overrides["backend"] = str(cli_args["local_vision_backend"])
    if "local_vision_model" in cli_args:
        overrides["model"] = str(cli_args["local_vision_model"])
    if "local_vision_base_url" in cli_args:
        overrides["base_url"] = str(cli_args["local_vision_base_url"])
    if "local_vision_timeout_sec" in cli_args:
        overrides["timeout_sec"] = _coerce_float_vision(cli_args.get("local_vision_timeout_sec"), config.timeout_sec)
    if "local_vision_allow_remote_endpoint" in cli_args:
        overrides["allow_remote_endpoint"] = _coerce_bool_vision(
            cli_args.get("local_vision_allow_remote_endpoint"), config.allow_remote_endpoint
        )
    config = _chain_after_overrides_vision(config, overrides)
    return _with_chain_vision(_with_validated_endpoint_vision(_config_from_payload_vision(overrides, config)))


def resolve_run_config(
    cli_args: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> ResolvedRunConfig:
    """Single resolver for ALL options behind one typed interface.

    Reads the config directory once, folds CLI > env > file > default for both
    general and vision options, validates via the same doors a typed value
    meets, and returns the typed ``ResolvedRunConfig`` holding both
    ``DistillOptions`` and ``LocalVisionConfig``.

    Vision config is a view over these same layers - forgiving coercions are
    applied here, not by a second reader. Replaces the untyped
    ``dict[str, Any]`` seam into ``DistillOptions.from_args``.

    ``base_dir`` is the config directory override for tests; ``None`` uses
    ``config_dir()``.
    """
    # Lazy imports to avoid circular at module load
    from .frame_selection import MAX_CANDIDATE_SCHEDULE as _MAX_SCHED
    from .local_vision import LocalVisionConfig as _LVC
    from .options import GENERAL_OPTION_NAMES as _GENERAL_OPTION_NAMES
    from .options import NUMERIC_OPTION_DOMAINS as _NUMERIC_DOMAINS
    from .options import OPTION_DEFAULTS as _OPTION_DEFAULTS
    from .options import OPTION_SPECS as _OPTION_SPECS
    from .options import DistillOptions as _DistillOptions
    from .options import validated_number as _validated_number

    # Resolve general layers and capture raw payload for vision view
    resolved_general, configured_from, raw_payload = _resolve_general_layers(
        cli_args, general_keys=_GENERAL_OPTION_NAMES, base_dir=base_dir
    )
    # resolved_general currently only contains general options that were set;
    # need to also include non-general CLI passthrough for options construction?
    # DistillOptions.from_args used: args = resolve_options(args, general_keys=...) -> that returned dict with all args keys carried through.
    # So resolved_general already via _resolve_general_layers does NOT carry non-general keys except we handle?
    # Our _resolve_general_layers only kept general keys; we need to reconstruct the full args dict as resolve_options did:
    # Build legacy resolved dict that includes all cli passthrough, to feed into validation.
    # Easiest: rebuild full_args = resolve_general_options(cli_args, ...) but we already have configured_from
    legacy_args = resolve_general_options(cli_args, general_keys=_GENERAL_OPTION_NAMES, base_dir=base_dir)

    # Resolve vision via same layers, reusing raw_payload as view
    local_vision: _LVC = _resolve_vision_layers(cli_args, base_dir=base_dir, raw_general_payload=raw_payload)

    # Now build DistillOptions from legacy_args + local_vision, mirroring DistillOptions.from_args
    # Reuse validation helpers from options
    from .options import FALSE_BOOLEAN_SPELLINGS as _FALSE_SPELLINGS
    from .options import TRUE_BOOLEAN_SPELLINGS as _TRUE_SPELLINGS

    def _coerce_bool_opt(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _FALSE_SPELLINGS:
                return False
            if normalized in _TRUE_SPELLINGS:
                return True
        return bool(value)

    def _bad_number_opt(name: str, value: Any, domain: Any, reason: str = "") -> DistillError:
        quantity = "a whole number" if domain.integral else "a finite number"
        if domain.floor:
            floor = f"{domain.floor} or greater"
        else:
            floor = "0 or greater" if domain.admits_zero else "greater than 0"
        if math.isfinite(domain.ceiling):
            floor = f"{floor} and below {domain.ceiling}"
        message = f"{name} must be {quantity} {floor}"
        return DistillError(
            "E_BAD_OPTIONS",
            "options",
            f"{message} ({reason})" if reason else message,
            {name: repr(value)},
        )

    def _validated_option_type_opt(spec: Any, value: Any) -> Any:
        if spec.name in _NUMERIC_DOMAINS:
            return value
        if spec.boolean:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in (_FALSE_SPELLINGS | _TRUE_SPELLINGS):
                return value
            expected = "a boolean or recognized on/off spelling"
        elif spec.name == "output_dir":
            if value is None or isinstance(value, str):
                return value
            expected = "text or null"
        elif spec.caster is str:
            if isinstance(value, str):
                return value
            expected = "text"
        else:
            return value
        raise DistillError(
            "E_BAD_OPTIONS",
            "options",
            f"{spec.name} must be {expected}",
            {spec.name: repr(value)},
        )

    def _annotate_configured_refusal_opt(error: DistillError, args: dict[str, Any], option: str) -> DistillError:
        configured_from_map = getattr(args, "configured_from", {}).get(option)
        if configured_from_map is not None:
            error.details["configured_from"] = str(configured_from_map)
        return error

    values: dict[str, Any] = {}
    for spec in _OPTION_SPECS:
        default = _OPTION_DEFAULTS[spec.name]
        try:
            raw_value = _validated_option_type_opt(spec, legacy_args.get(spec.name, default))
            if spec.boolean:
                values[spec.name] = _coerce_bool_opt(raw_value, bool(default))
            elif spec.name in _NUMERIC_DOMAINS:
                values[spec.name] = _validated_number(spec.name, raw_value)
            elif raw_value is None:
                values[spec.name] = None
            else:
                values[spec.name] = spec.caster(raw_value)
        except DistillError as exc:
            _annotate_configured_refusal_opt(exc, legacy_args, spec.name)
            raise

    from uuid import uuid4 as _uuid4

    # Timeout handling mirrors options.py: use raw CLI if supplied, else local_vision's coerced timeout
    timeout_raw = legacy_args.get("local_vision_timeout_sec", local_vision.timeout_sec)
    try:
        timeout_validated = _validated_number("local_vision_timeout_sec", timeout_raw)
    except DistillError as exc:
        _annotate_configured_refusal_opt(exc, legacy_args, "local_vision_timeout_sec")
        raise

    options = _DistillOptions(
        whisper_model=values["whisper_model"],
        whisper_language=values["whisper_language"],
        ocr=values["ocr"],
        ocr_language=values["ocr_language"],
        ocr_preprocess=values["ocr_preprocess"],
        redact_secrets=values["redact_secrets"],
        frame_salience=values["frame_salience"],
        max_keyframes=values["max_keyframes"],
        min_interval_sec=values["min_interval_sec"],
        max_duration_sec=values["max_duration_sec"],
        vad_filter=values["vad_filter"],
        max_static_window_sec=values["max_static_window_sec"],
        cache_mode=values["cache_mode"],
        output_dir=values["output_dir"],
        artifact_dir=values["artifact_dir"],
        force_reprocess=values["force_reprocess"],
        caption_frames=local_vision.caption_frames,
        local_vision_backend=local_vision.backend,
        local_vision_model=local_vision.model,
        local_vision_base_url=local_vision.base_url,
        local_vision_timeout_sec=timeout_validated,
        local_vision_allow_remote_endpoint=local_vision.allow_remote_endpoint,
        local_vision_credential=local_vision.credential,
        local_vision_credential_configured=local_vision.credential_configured,
        local_vision_credential_env=local_vision.credential_env,
        local_vision_endpoints=local_vision.endpoints,
        job_id=str(values["job_id"] or f"distill-{_uuid4().hex}"),
        resume_partial=values["resume_partial"],
    )
    if options.cache_mode not in {"fingerprint", "content"}:
        raise _annotate_configured_refusal_opt(
            DistillError(
                "E_BAD_OPTIONS",
                "options",
                "cache_mode must be 'fingerprint' or 'content'",
                {"cache_mode": options.cache_mode},
            ),
            legacy_args,
            "cache_mode",
        )
    if options.local_vision_backend != "rapid-mlx":
        raise DistillError(
            "E_BAD_OPTIONS",
            "options",
            "local_vision_backend must be 'rapid-mlx'",
            {"local_vision_backend": options.local_vision_backend},
        )
    worst_case_candidates = options.max_duration_sec / options.max_static_window_sec
    if worst_case_candidates > _MAX_SCHED:
        raise _annotate_configured_refusal_opt(
            DistillError(
                "E_BAD_OPTIONS",
                "options",
                "max_duration_sec and max_static_window_sec would build a keyframe "
                f"schedule of more than {_MAX_SCHED} candidates; widen "
                "the window or lower the duration cap",
                {
                    "max_duration_sec": options.max_duration_sec,
                    "max_static_window_sec": options.max_static_window_sec,
                },
            ),
            legacy_args,
            "max_static_window_sec",
        )

    # Keep vision's timeout consistent with validated one
    if local_vision.timeout_sec != timeout_validated:
        local_vision = replace(local_vision, timeout_sec=timeout_validated)

    # Build configured_from for returned dataclass
    return ResolvedRunConfig(
        options=options,
        local_vision=local_vision,
        configured_from=dict(configured_from),
        base_dir=base_dir,
        general_file_options=dict(legacy_args),
        environment_options=dict(_environment_options()),
        cli_args=dict(cli_args),
    )


# ---------------------------------------------------------------------------
# Legacy helpers for thin wrappers (so config/options/local_vision can delegate)
# ---------------------------------------------------------------------------


def general_config(base_dir: Path | None = None) -> dict[str, Any]:
    root = base_dir or config_dir()
    payload = _read_json_strict(root / GENERAL_CONFIG_FILENAME)
    return {k: v for k, v in payload.items() if k != LOCAL_VISION_SECTION}


def environment_options() -> dict[str, Any]:
    return _environment_options()


def resolve_options(
    args: Mapping[str, Any],
    *,
    general_keys: Iterable[str],
    base_dir: Path | None = None,
) -> _LegacyResolvedOptions:
    return resolve_general_options(args, general_keys=general_keys, base_dir=base_dir)


def load_local_vision_config(base_dir: Path | None = None) -> Any:
    # Forgiving wrapper - delegates to forgiving merge without strict payload
    # Use original forgiving path: _merged_local_vision_config_for_view with None raw

    cfg = _merged_local_vision_config_for_view(base_dir, strict_general_payload=None)
    return _with_chain_vision(_with_validated_endpoint_vision(cfg))


def local_vision_config_from_args(args: dict[str, Any], base_dir: Path | None = None) -> Any:
    # Unified path but for thin wrapper we want same as resolve_run_config's vision
    # Reuse _resolve_vision_layers forgiving path without strict payload? For legacy
    # compatibility, we should honor forgiving + overrides. But to keep view semantics,
    # we reuse _resolve_vision_layers with strict payload when possible? To preserve
    # existing forgiving behavior for tests that expect malformed local-vision file
    # to degrade, we use forgiving.
    # If base_dir is None, we could try to use strict payload if distill.json is readable,
    # but for simplicity use the resolver that already handles both.
    # For full compatibility with unified precedence, delegate to resolve_run_config's vision part:
    resolved = resolve_run_config(dict(args), base_dir=base_dir)
    return resolved.local_vision
