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

It does own where that client is allowed to talk and how much it will listen to
(R-43, R-44): the scheme, the loopback rule and its opt-out, the refusal to
follow a redirect, and the 32 MiB bound on a response body. That policy lives
here rather than in a transport module of its own because there is one client
and one endpoint (ADR-0001), and a rule kept next to the requests it governs
cannot be reached around by adding a caller.
"""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import logging
import os
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, NoReturn

from .artifacts import FrameArtifact, Interpretation, document_carries_a_reading
from .errors import DistillError, WarningRecord, aggregate_warnings, warning
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
# Consecutive transport failures after which the run stops attempting keyframes
# (R-40). Three, because one is noise and two is a coincidence.
CONSECUTIVE_TRANSPORT_FAILURE_LIMIT = 3
# The failures that say something about the transport rather than about the
# answer: nothing arrived. A refused or reset connection reaches the caller as
# the unavailable code, a deadline as the timeout code.
TRANSPORT_FAILURE_CODES = frozenset(
    {
        "local_vision_timeout",
        "local_vision_rapid_mlx_unavailable",
    }
)
# Failures that arrived *as a response*: the transport carried them, so they are
# evidence it works. A success is the other member of this class and needs no
# code. Everything else (an unreadable image file, a cancelled run) happened on
# this side of the wire and says nothing either way.
DELIVERED_RESPONSE_CODES = frozenset({"local_vision_malformed_response"})
BREAKER_WARNING_CODE = "local_vision_transport_breaker_open"
ENDPOINT_REJECTED_CODE = "local_vision_endpoint_rejected"
FRAME_READ_FAILURE_CODES = frozenset(
    {
        "local_vision_malformed_response",
        "local_vision_timeout",
        "local_vision_image_read_failed",
        ENDPOINT_REJECTED_CODE,
    }
)
# R-43. The two schemes the OpenAI-compatible API is served over; anything else
# names a different protocol, and a vision endpoint is not a file or a gopher
# hole no matter who wrote the config.
ALLOWED_ENDPOINT_SCHEMES = frozenset({"http", "https"})
DEFAULT_SCHEME_PORTS = {"http": 80, "https": 443}
# R-44. A chat-completion envelope is kilobytes; 32 MiB is orders of magnitude
# past any real one, and past it the read stops rather than the process growing
# to whatever the far end decided to send.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
# An HTTP error body is quoted into a message, never parsed, and the quote is
# 200 characters. This is how much of one is worth reading to produce it.
ERROR_BODY_PREVIEW_BYTES = 2048
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalVisionConfig:
    backend: str = DEFAULT_LOCAL_VISION_BACKEND
    model: str = DEFAULT_LOCAL_VISION_MODEL
    base_url: str = DEFAULT_LOCAL_VISION_BASE_URL
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    caption_frames: bool = True
    allow_remote_endpoint: bool = False
    """R-43's opt-out, and only that: it permits a `base_url` off loopback.

    It does not widen the scheme, re-enable redirects, or lift the response
    cap. Those hold wherever the endpoint is, because they are about what the
    client will do with an answer rather than about whose answer it is.
    """

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

    def warning(self) -> dict[str, Any]:
        return warning("local_vision", self.code, self.message)


class LocalVisionFailure(Exception):
    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def warning(self) -> dict[str, Any]:
        return warning("local_vision", self.code, self.message)


class EndpointRejected(LocalVisionFailure):
    """R-43: the endpoint the client was about to speak to is not allowed.

    A `LocalVisionFailure` and not a separate hierarchy, because to every
    caller it is the same event as a timeout: no reading came back and the run
    continues on OCR-only output. It carries a `reason` so the log and the
    **warning** both say *which* rule refused, not merely that one did.
    """

    def __init__(self, reason: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ENDPOINT_REJECTED_CODE, message, {"reason": reason, **(detail or {})})
        self.reason = reason


ProbeLocalVision = Callable[[LocalVisionConfig], LocalVisionProbe]
AddressResolver = Callable[[str, int], list[str]]
"""Host and port to the addresses they resolve to, for the loopback check.

Injectable because a test that asked the machine's resolver would be asserting
against `/etc/hosts`, and because D-029's rebind - the same name answering
differently twice - has no other seam to arrive through.
"""
TryInterpretImage = Callable[..., tuple["Interpretation | None", "WarningRecord | None"]]


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
    warnings: list[WarningRecord]
    path: str
    has_extracted_text: bool
    # The code of the read/transport failure this frame's attempt ended in, or
    # `None` if a response arrived. Not "the first warning": a frame that was
    # read successfully can still warn about **grounding**, and the breaker
    # must not read that as the transport having failed.
    warning_code: str | None
    # The breaker state change this frame's outcome caused, if any. Carried
    # rather than logged where it happened so the ordered consumer emits it in
    # frame order even when the pool ran the frames out of order.
    breaker_transition: dict[str, Any] | None = None


class _TransportBreaker:
    """One run's consecutive-transport-failure count, and the state it decides.

    Owns the count, the open/closed state, and the record of what tripped it:
    which attempt, which **keyframe**, which code. It is asked `admit()` before
    every attempt and told `record()` after every one, and those two calls are
    the whole interface.

    It does not own what a skipped keyframe becomes - the caller hands the
    **frame artifact** back untouched, which is the OCR-only **degradation**
    ADR-0002 asks for - and it does not own the **warning**: it supplies the
    counts, `summary_warning` phrases them. It never closes again either. A
    half-open probe would be a fourth timeout to learn what three already
    said, and a server that died mid-run is not coming back inside the run.

    Which failures count is a distinction, not a list: a **transport** failure
    means nothing arrived, so the run has no evidence the server is there. A
    malformed body arrived, so it is evidence the transport works and resets
    the count - a wrong server and a dead server are different findings, and
    only one of them is worth stopping for. Failures on this side of the wire
    (an unreadable image, a cancellation) neither count nor reset.

    Thread-safe because the pool may attempt keyframes concurrently, and with
    ``max_parallel > 1`` "consecutive" means consecutive in *completion* order
    rather than keyframe order. That is the only order that exists there - the
    keyframes were in flight together - and the consequence is worth stating: a
    fast success landing between two slow timeouts resets a count that keyframe
    order would have kept climbing, so a partly-working server takes longer to
    give up on than a dead one. A dead one returns no successes at all, which
    is the case this exists for. Attempts already dispatched when it opens
    still run, and ``DEFAULT_MAX_PARALLEL`` is 1, so the ordinary run has one
    order.
    """

    def __init__(self, limit: int = CONSECUTIVE_TRANSPORT_FAILURE_LIMIT) -> None:
        self._limit = max(int(limit), 1)
        self._lock = threading.Lock()
        self._consecutive = 0
        self._attempts = 0
        self._transport_failures = 0
        self._skipped = 0
        self._opened: dict[str, Any] | None = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened is not None

    def admit(self) -> bool:
        """Whether a **keyframe** may be attempted, counting the skip when not."""
        with self._lock:
            if self._opened is None:
                return True
            self._skipped += 1
            return False

    def record(self, *, frame_number: int, code: str | None) -> dict[str, Any] | None:
        """Fold one attempt's outcome in; the state transition it caused, or `None`."""
        with self._lock:
            self._attempts += 1
            if self._opened is not None:
                # An attempt that passed admission before the trip returns
                # after it. Nothing it says is a state change, because the
                # breaker does not close: reporting one would put a reset in
                # the log of a run still skipping every remaining keyframe, and
                # zero the count that explains why.
                if code in TRANSPORT_FAILURE_CODES:
                    self._transport_failures += 1
                return None
            if code in TRANSPORT_FAILURE_CODES:
                self._consecutive += 1
                self._transport_failures += 1
                if self._consecutive < self._limit:
                    return None
                self._opened = {
                    "state": "open",
                    "consecutive_failures": self._consecutive,
                    "limit": self._limit,
                    "attempt": self._attempts,
                    "frame": frame_number,
                    "code": code,
                }
                return dict(self._opened)
            if code is not None and code not in DELIVERED_RESPONSE_CODES:
                return None
            cleared, self._consecutive = self._consecutive, 0
            if not cleared:
                return None
            return {
                "state": "closed",
                "cleared_failures": cleared,
                "attempt": self._attempts,
                "frame": frame_number,
            }

    def summary_warning(self, frame_count: int) -> WarningRecord | None:
        """The one **warning** R-40 asks for, or `None` if the breaker held.

        One record for every keyframe the run gave up on, carrying the count
        that tripped it and the count that cost - which is the whole point of
        the breaker: 77 timeouts said this once each.
        """
        with self._lock:
            opened = self._opened
            skipped = self._skipped
        if opened is None:
            return None
        return warning(
            "local_vision",
            BREAKER_WARNING_CODE,
            f"local vision stopped after {opened['consecutive_failures']} consecutive "
            f"transport failures ({opened['code']}) at keyframe {opened['frame']}; "
            f"{skipped} of {frame_count} keyframes continue with OCR-only output.",
        )

    def state(self) -> dict[str, Any]:
        """What the breaker saw, for `debug_info`."""
        with self._lock:
            return {
                "open": self._opened is not None,
                "limit": self._limit,
                "consecutive_failures": self._consecutive,
                "transport_failures": self._transport_failures,
                "skipped_keyframes": self._skipped,
                "opened_by": None if self._opened is None else dict(self._opened),
            }


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


def _reject_endpoint(reason: str, message: str, **detail: Any) -> NoReturn:
    """Refuse an endpoint, saying why, in the log and in the exception alike.

    Every rejection goes through here so the observability is a property of
    the rule rather than of whoever wrote the call site: there is one event,
    it always carries the reason, and no new rule can forget to emit it.
    """
    _boundary_log("endpoint_rejected", {"reason": reason, **detail})
    raise EndpointRejected(reason, message, detail)


def _resolve_addresses(host: str, port: int) -> list[str]:
    """The addresses `host` answers with - without a lookup when it is one.

    An address literal is already the answer, so the common configuration
    (`http://127.0.0.1:8000/v1`) never consults a resolver at all.
    """
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        _reject_endpoint(
            "unresolvable",
            f"Local vision endpoint host '{host}' does not resolve; "
            "continuing with OCR-only output.",
            host=host,
            error=str(exc),
        )
    return [str(info[4][0]) for info in infos]


def _checked_endpoint_url(url: str, *, allow_remote_endpoint: bool) -> tuple[str, int]:
    """R-43's static half: what can be decided about `url` without asking anyone.

    The scheme, the presence of a host, and - when the host is written as an
    address - whether that address is loopback. A host written as a *name*
    cannot be settled here without a lookup, so it is settled per request
    instead (D-029); this is the check a config can be rejected by, not the
    authoritative one.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ALLOWED_ENDPOINT_SCHEMES:
        _reject_endpoint(
            "scheme",
            f"Local vision endpoint scheme '{parts.scheme}' is not allowed; "
            f"use {' or '.join(sorted(ALLOWED_ENDPOINT_SCHEMES))}.",
            url=url,
            scheme=parts.scheme,
        )
    try:
        # Both raise on a malformed authority - a port out of range, a bracket
        # that never closes - and a rejection saying so is the answer a caller
        # can act on. An unhandled `ValueError` here would reach the CLI as a
        # traceback about something typed in a config file.
        host, configured_port = parts.hostname, parts.port
    except ValueError as exc:
        _reject_endpoint(
            "unparsable_url",
            f"Local vision endpoint '{url}' is not a usable URL: {exc}",
            url=url,
        )
    if not host:
        _reject_endpoint("host_missing", f"Local vision endpoint '{url}' names no host.", url=url)
    port = configured_port or DEFAULT_SCHEME_PORTS[parts.scheme]
    if not allow_remote_endpoint:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_loopback:
            _reject_endpoint(
                "non_loopback_host",
                f"Local vision endpoint host '{host}' is not loopback; set "
                "local_vision_allow_remote_endpoint to reach one deliberately.",
                url=url,
                host=host,
            )
    return host, port


def _check_resolved_address(
    host: str,
    port: int,
    *,
    allow_remote_endpoint: bool,
    resolver: AddressResolver | None = None,
) -> list[str]:
    """R-43's authoritative half: every address `host` resolves to is loopback.

    Every address, not the first: a name answering with both 127.0.0.1 and a
    link-local address is a name that can steer the very next connection off
    loopback, and which of its answers gets connected to is not this function's
    to know.
    """
    if allow_remote_endpoint:
        return []
    addresses = (resolver or _resolve_addresses)(host, port)
    if not addresses:
        _reject_endpoint(
            "unresolvable",
            f"Local vision endpoint host '{host}' resolved to no address.",
            host=host,
        )
    for address in addresses:
        try:
            resolved = ipaddress.ip_address(address)
        except ValueError:
            # A resolver that answered with something that is not an address
            # (a scoped `fe80::1%en0`, a name, anything) has not been shown to
            # be loopback, and unproven is refused rather than assumed.
            _reject_endpoint(
                "unparsable_address",
                f"Local vision endpoint host '{host}' resolved to '{address}', "
                "which is not an address.",
                host=host,
                address=address,
            )
        if not resolved.is_loopback:
            _reject_endpoint(
                "non_loopback_address",
                f"Local vision endpoint host '{host}' resolves to {address}, which is not "
                "loopback; set local_vision_allow_remote_endpoint to reach one deliberately.",
                host=host,
                address=address,
            )
    return addresses


def _with_validated_endpoint(config: LocalVisionConfig) -> LocalVisionConfig:
    """The same config, once its `base_url` is one Distill will speak to.

    A rejected endpoint is a **fatal error** here rather than a **warning**:
    unlike a server that is merely down, nothing about the run will make this
    address allowed, and silently degrading to OCR-only would hide that the
    operator's configuration was ignored.
    """
    try:
        _checked_endpoint_url(
            config.base_url, allow_remote_endpoint=config.allow_remote_endpoint
        )
    except EndpointRejected as exc:
        raise DistillError(
            "E_BAD_OPTIONS",
            "local_vision",
            exc.message,
            {"base_url": config.base_url, **exc.detail},
        ) from exc
    return config


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


def _merged_local_vision_config(base_dir: Path | None = None) -> LocalVisionConfig:
    """The configuration files, folded together, not yet validated.

    Unvalidated on purpose: the endpoint that matters is the one the run will
    use, and a per-call override is allowed to rescue a file that names a
    forbidden one. Only the settled config is checked.
    """
    root = (base_dir or config_dir()).expanduser()
    config = LocalVisionConfig()
    for filename in CONFIG_FILENAMES:
        payload = _read_json(root / filename)
        if filename == "distill.json":
            nested = payload.get("local_vision")
            payload = nested if isinstance(nested, dict) else {}
        config = _config_from_payload(payload, config)
    return config


def load_local_vision_config(base_dir: Path | None = None) -> LocalVisionConfig:
    return _with_validated_endpoint(_merged_local_vision_config(base_dir))


def local_vision_config_from_args(
    args: dict[str, Any],
    base_dir: Path | None = None,
) -> LocalVisionConfig:
    config = _merged_local_vision_config(base_dir)
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
    if "local_vision_allow_remote_endpoint" in args:
        overrides["allow_remote_endpoint"] = _coerce_bool(
            args.get("local_vision_allow_remote_endpoint"), config.allow_remote_endpoint
        )
    return _with_validated_endpoint(_config_from_payload(overrides, config))


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
        allow_remote_endpoint=_coerce_bool(
            payload.get("allow_remote_endpoint", base.allow_remote_endpoint),
            base.allow_remote_endpoint,
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
        payload = _http_get_json(
            requestor,
            models_url,
            config.timeout_sec,
            allow_remote_endpoint=config.allow_remote_endpoint,
        )
    except EndpointRejected as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code=exc.code,
            message=f"{exc.message} Continuing with OCR-only output.",
            detail={**exc.detail, "url": models_url},
        )
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
) -> tuple[Interpretation | None, WarningRecord | None]:
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
) -> tuple[Interpretation | None, WarningRecord | None]:
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
    _breaker: _TransportBreaker = field(default_factory=_TransportBreaker)

    def interpret(
        self, frames: list[FrameArtifact]
    ) -> tuple[list[FrameArtifact], list[WarningRecord]]:
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

        warnings: list[WarningRecord] = []
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
            # R-41 where the vision pass is done producing warnings: eighty
            # keyframes failing the same way is one finding with a count on it,
            # not eighty records of the same sentence. `_warning_counts` still
            # has the unfolded tally for `debug_info`.
            folded = aggregate_warnings(warnings)
            self._log(
                "interpret.complete",
                {
                    "frames": len(interpreted_frames),
                    "interpreted_frames": self._interpreted_count,
                    "warnings": len(folded),
                    "warning_occurrences": sum(item["occurrences"] for item in folded),
                },
            )
            return interpreted_frames, folded
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
        warnings: list[WarningRecord],
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
        interpreted = self._merge_outcomes(outcomes, warnings)
        breaker_warning = self._breaker.summary_warning(frame_count)
        if breaker_warning is not None:
            # Last, so it reads as the account of the run rather than as one
            # more per-frame failure: the frames it speaks for are above it.
            warnings.append(breaker_warning)
            self._record_warning(breaker_warning)
        return interpreted

    def _interpret_one(
        self,
        config: LocalVisionConfig,
        index: int,
        frame: FrameArtifact,
        frame_count: int,
    ) -> _FrameOutcome:
        self._ensure_frame_invariants(index, frame, frame_count)
        if not self._breaker.admit():
            # R-40's **degradation**: the keyframe is not attempted and not
            # warned about on its own - the breaker's single warning speaks for
            # all of them - and it is handed back carrying the **extracted
            # text** OCR already read, which is what OCR-only means.
            return _FrameOutcome(
                index=index,
                frame=frame,
                interpreted=False,
                warnings=[],
                path=frame.path,
                has_extracted_text=bool(frame.extracted_text.strip()),
                warning_code=None,
            )
        local_warnings: list[WarningRecord] = []
        interpreted_frame, was_interpreted, read_code = self._interpret_frame(
            config, index, frame, frame_count, local_warnings
        )
        transition = self._breaker.record(frame_number=index + 1, code=read_code)
        return _FrameOutcome(
            index=index,
            frame=interpreted_frame,
            interpreted=was_interpreted,
            warnings=local_warnings,
            path=frame.path,
            has_extracted_text=bool(frame.extracted_text.strip()),
            warning_code=read_code,
            breaker_transition=transition,
        )

    def _merge_outcomes(
        self,
        outcomes: list[_FrameOutcome],
        warnings: list[WarningRecord],
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
            if outcome.breaker_transition is not None:
                transition = outcome.breaker_transition
                self._log(
                    "breaker.open" if transition["state"] == "open" else "breaker.reset",
                    transition,
                )
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
            "breaker": self._breaker.state(),
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
        warnings: list[WarningRecord],
    ) -> tuple[FrameArtifact, bool, str | None]:
        """Interpret a single frame without mutating shared interpreter state.

        Appends any frame warnings to the supplied (per-frame) ``warnings`` list
        and returns the interpreted frame, whether a result was produced, and the
        code of the read failure it ended in (``None`` when a response arrived).
        That third value is the breaker's evidence and is returned rather than
        recovered from the warning list, because the warning list also holds
        **grounding** warnings a *successful* read produced.
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
        if result is not None and not result.carries_a_reading:
            # R-39 where a reading arrives by any route: the transport path
            # rejects an empty payload as malformed, and a reading that reached
            # here saying nothing is the same non-answer one step later. It is
            # not attached and it is not counted, so a run cannot report a
            # frame as interpreted on the strength of an empty object.
            result = None
        read_code = frame_warning.get("code") if frame_warning else None
        if result:
            return self._interpreted(frame, result, index, warnings), True, read_code
        if read_code in FRAME_READ_FAILURE_CODES:
            unusable = GroundingAssessment(
                UNGROUNDED,
                None,
                f"vision model produced no usable output ({read_code})",
            )
            carried, carrier_warnings = frame.with_interpretation(
                None, grounding=unusable.public_dict()
            )
            warnings.extend(carrier_warnings)
            return carried, False, read_code
        return frame, False, read_code

    def _try_interpret_frame(
        self,
        config: LocalVisionConfig,
        image_path: Path,
        prompt: str,
        *,
        prompt_profile: str,
    ) -> tuple[Interpretation | None, WarningRecord | None]:
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
        warnings: list[WarningRecord],
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
            carries_a_reading=reading.carries_a_reading,
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
        # A breaker is a fact about one run: an interpreter reused for a second
        # run starts closed, or a server that recovered between runs would
        # never be spoken to again.
        self._breaker = _TransportBreaker()

    def _record_warning(self, warning_payload: WarningRecord) -> None:
        """Tally one **warning** by what it says happened, not by being one record.

        A carrier hands back warnings the **redaction** policy already folded,
        so one record can stand for four confusable matches. Counting records
        made `debug_info` and the **manifest** disagree about the same run.
        """
        code = warning_payload.get("code", "unknown")
        occurrences = warning_payload.get("occurrences", 1)
        counted = occurrences if isinstance(occurrences, int) and occurrences >= 1 else 1
        self._warning_counts[code] = self._warning_counts.get(code, 0) + counted

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
            envelope = _http_post_json(
                requestor,
                completions_url,
                body,
                config.timeout_sec,
                allow_remote_endpoint=config.allow_remote_endpoint,
            )
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
    *,
    allow_remote_endpoint: bool = False,
) -> dict[str, Any]:
    if requestor is not None:
        payload = requestor(method="GET", url=url, timeout=timeout_sec)
    else:
        payload = _urlopen_json(
            "GET",
            url,
            body=None,
            timeout_sec=timeout_sec,
            allow_remote_endpoint=allow_remote_endpoint,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def _http_post_json(
    requestor: HttpRequestor | None,
    url: str,
    body: dict[str, Any],
    timeout_sec: float,
    *,
    allow_remote_endpoint: bool = False,
) -> dict[str, Any]:
    if requestor is not None:
        payload = requestor(method="POST", url=url, body=body, timeout=timeout_sec)
    else:
        payload = _urlopen_json(
            "POST",
            url,
            body=body,
            timeout_sec=timeout_sec,
            allow_remote_endpoint=allow_remote_endpoint,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


class _RedirectsAreRejected(urllib.request.HTTPRedirectHandler):
    """R-43 (RV-7): a 3xx ends the request instead of starting another one.

    Validating an endpoint and then following wherever it points validates
    nothing - the second request is to an address no rule ever saw. Refusing
    outright, rather than re-validating the target, is the smaller rule: the
    only endpoint Distill talks to is the one the operator configured.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,  # noqa: ARG002 - urllib's signature; the body is not read
        code: int,
        msg: str,
        headers: Any,  # noqa: ARG002 - urllib's signature; no header is trusted
        newurl: str,
    ) -> NoReturn:
        _reject_endpoint(
            "redirect",
            f"Local vision endpoint answered {code} {msg} redirecting to another "
            "address; redirects are not followed.",
            url=req.full_url,
            status=code,
            location=newurl,
        )


def _build_opener(*handlers: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    """The vision client's opener: the default chain, minus two of its habits.

    Redirects are not followed, and no proxy is consulted. An empty
    `ProxyHandler` rather than the default one because the default reads
    `HTTP_PROXY` from the environment: a validated loopback endpoint would
    still be dialled through whatever that names, which sends the **keyframe**
    to a host no rule here ever saw. The endpoint is local by rule, so there is
    nothing for a proxy to do.

    Built by a function so a test can put a fake transport under the *same*
    chain production uses, instead of asserting against a chain it assembled
    itself.
    """
    return urllib.request.build_opener(
        _RedirectsAreRejected, urllib.request.ProxyHandler({}), *handlers
    )


_OPENER = _build_opener()


def _urlopen_json(
    method: str,
    url: str,
    body: dict[str, Any] | None,
    timeout_sec: float,
    *,
    allow_remote_endpoint: bool = False,
    resolver: AddressResolver | None = None,
) -> Any:
    """One request to the vision endpoint, validated before and bounded after.

    The address is resolved and checked here, on every request, rather than
    once when the config was read (D-029): a name that answered with 127.0.0.1
    while the config was being validated is free to answer with something else
    by the time the **keyframe** is posted, and only the check standing between
    the resolution and the connection sees that.

    What that does not close, stated so nobody reads it as more: the connection
    resolves the name again, so a name that changes its answer between this
    check and that connect is not caught. Closing it means connecting to the
    address this function validated rather than to the name, which is a socket
    Distill would have to open itself. The bound here is per request, which is
    what D-029 asks for; it is not per packet.
    """
    host, port = _checked_endpoint_url(url, allow_remote_endpoint=allow_remote_endpoint)
    _check_resolved_address(
        host, port, allow_remote_endpoint=allow_remote_endpoint, resolver=resolver
    )
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
        with _OPENER.open(request, timeout=timeout_sec) as response:
            # One byte past the cap, so a body that is exactly at it still
            # arrives and a body over it is recognisable without being read.
            # The header is not consulted: a Content-Length is the sender's
            # claim about the sender, and R-44 is a bound on this process.
            payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            # A delivered response, in M7.2's sense: it arrived, so it is
            # evidence the transport works and must not count toward the
            # breaker's consecutive-failure tally. It is unusable for the same
            # reason a truncated body is - hence malformed, not unavailable.
            raise RuntimeError(
                f"response from {url} exceeds the {MAX_RESPONSE_BYTES} byte cap"
            )
        raw = payload.decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Surface HTTP error bodies (e.g. model-not-loaded) as a RuntimeError so
        # the probe/interpret paths can map them onto warning codes. Bounded
        # like any other body (R-44), and tighter: only a preview of it is ever
        # used, so reading a gigabyte to quote 200 characters of it is a cost
        # with no reader.
        detail = (
            exc.read(ERROR_BODY_PREVIEW_BYTES).decode("utf-8", errors="replace")
            if hasattr(exc, "read")
            else ""
        )
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
    """The reading a payload `parse_interpretation_json` accepted describes.

    Every field is optional here because the payload has already been checked
    for one that is not (R-39): what is missing from a validated payload is a
    field the model left out, not an answer that said nothing.
    """
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
    """The model's answer as an interpretation payload, or `None` if it is not one.

    `None` means malformed, and covers three things the caller handles
    identically: text that is not JSON, JSON that is not an object, and an
    object that carries no reading (R-39). The third is why this is not a bare
    parser - a server that is up and answers `{}` for every keyframe parses
    perfectly, and counting that as an interpretation is what makes a dead
    model look like a working one.
    """
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
    if not isinstance(parsed, dict) or not document_carries_a_reading(parsed):
        return None
    return parsed


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
