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

The transport, the OpenAI-style envelope parsing, and the endpoint policy
(R-43, R-44: the scheme, the loopback rule and its opt-out, the refusal to
follow a redirect, the 32 MiB bound) are ``rapid_mlx``'s - the one
OpenAI-compatible client, whose default endpoint is a local Rapid-MLX server
(ADR-0005, superseding ADR-0001). This module drives that client: it
owns the configuration, decides whether a pass can run, and runs the
interpretation. The endpoint policy still lives next to the requests it governs,
because both moved together; splitting them so a caller could reach around the
policy is the thing that was avoided, not the file they share.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .artifacts import FrameArtifact, Interpretation
from .config import config_dir as general_config_dir
from .errors import DistillError, WarningRecord, aggregate_warnings, occurrences_of, warning
from .grounding import UNGROUNDED, GroundingAssessment, assess_grounding
from .progress import ProgressReporter
from .rapid_mlx import (
    _OPENER as _OPENER,
)
from .rapid_mlx import (  # noqa: F401  re-exported: the client this module drives
    ALLOWED_ENDPOINT_SCHEMES as ALLOWED_ENDPOINT_SCHEMES,
)
from .rapid_mlx import (
    DEFAULT_SCHEME_PORTS as DEFAULT_SCHEME_PORTS,
)
from .rapid_mlx import (
    DEFAULT_VISION_STAGE_BUDGET_BYTES as DEFAULT_VISION_STAGE_BUDGET_BYTES,
)
from .rapid_mlx import (
    DEFAULT_VISION_STAGE_BUDGET_SEC as DEFAULT_VISION_STAGE_BUDGET_SEC,
)
from .rapid_mlx import (
    ENDPOINT_REJECTED_CODE as ENDPOINT_REJECTED_CODE,
)
from .rapid_mlx import (
    ERROR_BODY_PREVIEW_BYTES as ERROR_BODY_PREVIEW_BYTES,
)
from .rapid_mlx import (
    MAX_RESPONSE_BYTES as MAX_RESPONSE_BYTES,
)
from .rapid_mlx import (
    AddressResolver as AddressResolver,
)
from .rapid_mlx import (
    EndpointRejected as EndpointRejected,
)
from .rapid_mlx import (
    HttpRequestor as HttpRequestor,
)
from .rapid_mlx import (
    LocalVisionFailure as LocalVisionFailure,
)
from .rapid_mlx import (
    SecretCredential as SecretCredential,
)
from .rapid_mlx import (
    VisionStageBudget as VisionStageBudget,
)
from .rapid_mlx import (
    _boundary_log as _boundary_log,
)
from .rapid_mlx import (
    _build_opener as _build_opener,
)
from .rapid_mlx import (
    _chat_content as _chat_content,
)
from .rapid_mlx import (
    _check_resolved_address as _check_resolved_address,
)
from .rapid_mlx import (
    _checked_endpoint_url as _checked_endpoint_url,
)
from .rapid_mlx import (
    _completions_url as _completions_url,
)
from .rapid_mlx import (
    _extract_first_json_object as _extract_first_json_object,
)
from .rapid_mlx import (
    _http_get_json as _http_get_json,
)
from .rapid_mlx import (
    _http_post_json as _http_post_json,
)
from .rapid_mlx import (
    _models_url as _models_url,
)
from .rapid_mlx import (
    _normalize_frame_kind as _normalize_frame_kind,
)
from .rapid_mlx import (
    _normalize_text_confidence as _normalize_text_confidence,
)
from .rapid_mlx import (
    _RedirectsAreRejected as _RedirectsAreRejected,
)
from .rapid_mlx import (
    _reject_endpoint as _reject_endpoint,
)
from .rapid_mlx import (
    _resolve_addresses as _resolve_addresses,
)
from .rapid_mlx import (
    _served_model_ids as _served_model_ids,
)
from .rapid_mlx import (
    _static_host_is_loopback as _static_host_is_loopback,
)
from .rapid_mlx import (
    _urlopen_json as _urlopen_json,
)
from .rapid_mlx import (
    parse_frame_salience as parse_frame_salience,
)
from .rapid_mlx import (
    parse_interpretation_json as parse_interpretation_json,
)
from .transcript import select_transcript_window

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
DELIVERED_RESPONSE_CODES = frozenset(
    {"local_vision_malformed_response", "local_vision_retry_exhausted"}
)
# 429/5xx handling (D-008): the server asked for a pause, so honor it - but
# bounded on both axes, attempts and seconds, because Retry-After is the
# endpoint's claim about the endpoint and the run's time is this process's.
RETRYABLE_MAX_RETRIES = 2
RETRY_AFTER_CAP_SEC = 10.0
DEFAULT_RETRY_BACKOFF_SEC = 1.0
# Injectable so tests assert the pause without sleeping through it.
_sleep = time.sleep
# Failures that condemn every remaining attempt the moment one happens: a
# rejected credential will be rejected again on the next keyframe, and a
# rejected endpoint is a property of the config, not of the frame - retrying
# either is sending more keyframes to an endpoint that already said no
# (D-016, D-008).
IMMEDIATE_FAILURE_CODES = frozenset(
    {
        "local_vision_auth_rejected",
        # An endpoint that kept answering 429/5xx through the bounded retries
        # will do it again for the next keyframe; retrying the run into a
        # rate limiter is the amplification D-008 exists to prevent.
        "local_vision_retry_exhausted",
        ENDPOINT_REJECTED_CODE,
        VisionStageBudget.CODE,
    }
)
BREAKER_WARNING_CODE = "local_vision_transport_breaker_open"
FRAME_READ_FAILURE_CODES = frozenset(
    {
        "local_vision_malformed_response",
        "local_vision_retry_exhausted",
        "local_vision_timeout",
        "local_vision_image_read_failed",
        "local_vision_auth_rejected",
        VisionStageBudget.CODE,
        ENDPOINT_REJECTED_CODE,
    }
)
# R-43. The two schemes the OpenAI-compatible API is served over; anything else
# names a different protocol, and a vision endpoint is not a file or a gopher
# hole no matter who wrote the config.
# R-44. A chat-completion envelope is kilobytes; 32 MiB is orders of magnitude
# past any real one, and past it the read stops rather than the process growing
# to whatever the far end decided to send.
# An HTTP error body is quoted into a message, never parsed, and the quote is
# 200 characters. This is how much of one is worth reading to produce it.
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
    client will do with an answer rather than about whose answer it is. And
    the widened host narrows the scheme: a non-loopback endpoint must speak
    `https` (D-008) - plain `http` is loopback-only, opt-out or not.
    """
    credential: SecretCredential | None = field(default=None, repr=False, metadata={"secret": True})
    """The endpoint credential, in its non-serializable carrier (D-007).

    The carrier itself redacts every text form and refuses serialization;
    `repr=False` and the `public_dict` exclusion are belt-and-braces on top.
    It participates in equality (the carrier compares by value) so configs
    differing only by credential are never interchangeable.
    """
    # Whether api_key/api_key_env was present in any config layer, and the env
    # var it named. Neither is secret (a *name* is diagnostics, not a value);
    # together they let validation tell "intentionally no auth" apart from
    # "meant to authenticate and the value went missing" (D-016).
    credential_configured: bool = False
    credential_env: str = ""
    budget: VisionStageBudget | None = field(
        default=None, repr=False, compare=False, metadata={"runtime": True}
    )
    """The run's vision-stage budget, attached by the interpreter to its
    per-run resolved config. Runtime state, not configuration: excluded from
    identity, `repr`, and `public_dict`."""
    endpoints: tuple[LocalVisionConfig, ...] | None = None
    """The configured **endpoint chain**, in preference order.

    `None` and `()` are different facts and the distinction is load-bearing:
    `None` is "no chain was configured", which the top-level fields answer for
    and `_with_chain` turns into a one-entry chain; `()` is "a chain was
    configured and it names no endpoints", which is a configuration mistake and
    is refused. Collapsing them would read an empty array as the defaults and
    run an endpoint the operator did not ask for.

    After the config is settled this is always a tuple of at least one entry.

    Entries are `LocalVisionConfig` values rather than a type of their own, so
    this module keeps no knowledge of what a chain *means* - order, selection,
    and candidate keys all belong to `vision_chain.py`. What lives here is what
    already lived here: reading configuration layers into the fields an endpoint
    is made of.

    <!-- P3-D-010 --> Each entry is parsed from a fresh default rather than from
    the surrounding config, so entry 2 cannot inherit entry 1's credential or
    its remote authorization.
    """
    top_level_endpoint_fields: tuple[str, ...] = field(
        default=(), repr=False, compare=False, metadata={"runtime": True}
    )
    """Which top-level endpoint fields the config *files* named, if any.

    Parse bookkeeping in the same spirit as `credential_configured`: recorded
    while layers are folded, consumed once by the settle-time check, and never
    part of what a **bundle** records - it describes how the configuration was
    written rather than what the endpoint is.

    Carried rather than checked in place because P3-D-013 is about *any* layer:
    one file naming a chain and another naming a top-level `base_url` reads as
    good configuration in each file alone, and the conflict exists only once
    they are folded together.
    """

    def public_dict(self) -> dict[str, Any]:
        # Excluded by the `secret` metadata flag and by carrier type - a rule
        # about secrets, not about one field spelling. Built by field access
        # rather than asdict, which deep-copies into the refusing carrier.
        public: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if (
                f.metadata.get("secret")
                or f.metadata.get("runtime")
                or isinstance(value, SecretCredential)
            ):
                continue
            if f.name == "endpoints":
                # Each entry answers for itself, by the same rules: an entry
                # holds a credential exactly as the outer config does, and a
                # tuple of dataclasses reaching a serializer would take those
                # carriers with it. Recursing keeps the secret rule in one
                # place rather than restating it per field spelling.
                public[f.name] = None if value is None else [e.public_dict() for e in value]
                continue
            public[f.name] = value
        return public


def config_is_non_local(config: LocalVisionConfig) -> bool:
    """D-012's predicate, in one place: this run MAY send keyframes off the
    producing machine. True when the operator opted into a remote endpoint,
    the scheme is not `http` (which the transport proves loopback on every
    request, opt-in or not - D-008), and the host is not a literal loopback
    address. An `https` hostname counts as remote: it cannot be settled
    statically, and provenance folds fail closed."""
    if not config.allow_remote_endpoint:
        return False
    try:
        scheme = urllib.parse.urlsplit(config.base_url).scheme.lower()
    except ValueError:
        return True  # unparsable fails closed, like an unsettleable name
    if scheme == "http":
        return False
    return not _static_host_is_loopback(config.base_url)


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


ProbeLocalVision = Callable[[LocalVisionConfig], LocalVisionProbe]
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
            if code in IMMEDIATE_FAILURE_CODES:
                # One rejection condemns the rest. It arrived AS a response,
                # so it is not a transport failure and is not counted as one;
                # the taxonomy at DELIVERED_RESPONSE_CODES stays true.
                self._opened = {
                    "state": "open",
                    "consecutive_failures": self._consecutive,
                    "limit": 1,
                    "attempt": self._attempts,
                    "frame": frame_number,
                    "code": code,
                }
                return dict(self._opened)
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
        """The one **warning** R-40 asks for, or `None` if it cost nothing.

        One record for every keyframe the run gave up on, carrying the count
        that tripped it and the count that cost - which is the whole point of
        the breaker: 77 timeouts said this once each.

        `None` when the breaker held *and* when it opened without refusing a
        keyframe - a trip on the last one, where the failures that tripped it
        already have their own warnings and there is nothing left to give up
        on. A record there would report a **degradation** that did not happen,
        in a sentence saying zero keyframes were affected.

        The count is of keyframes *not attempted*, and says so. "Continue with
        OCR-only output" describes them and also describes the ones that failed
        before the trip, which are not in this number - so a reader adding the
        sentence up got a different run than the one that happened.
        """
        with self._lock:
            opened = self._opened
            skipped = self._skipped
        if opened is None or not skipped:
            return None
        immediate_phrases = {
            "local_vision_auth_rejected": "the endpoint rejected the credential",
            ENDPOINT_REJECTED_CODE: "the endpoint was rejected",
            VisionStageBudget.CODE: "the vision stage's run-wide budget was spent",
            "local_vision_retry_exhausted": "the endpoint kept rate-limiting",
        }
        phrase = immediate_phrases.get(opened["code"])
        if phrase is not None:
            reason = f"{phrase} ({opened['code']})"
        else:
            reason = (
                f"{opened['consecutive_failures']} consecutive transport failures "
                f"({opened['code']})"
            )
        return warning(
            "local_vision",
            BREAKER_WARNING_CODE,
            f"local vision stopped after {reason} at keyframe {opened['frame']}; "
            f"{skipped} of {frame_count} keyframes were not attempted and continue "
            "with OCR-only output.",
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


def _debug_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return value
    return os.environ.get(DEBUG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


ENDPOINT_FIELD_NAMES = ("model", "base_url", "api_key", "api_key_env", "allow_remote_endpoint")
"""The config keys that describe *one* **vision endpoint**.

Named as a set rather than checked inline because two rules need the same
answer: these are what a chain entry is made of, and they are what a top-level
config names when it configures the single endpoint the old way. `backend`,
`timeout_sec` and `caption_frames` are deliberately absent - those are
chain-wide and belong at the top level next to an `endpoints` array.
"""

MAX_ENDPOINTS = 4
"""How many **vision endpoints** one **endpoint chain** may name.

<!-- P3-D-005 --> A bound rather than a preference: each entry costs a probe on
a cold run, and the chain is walked in order, so an unbounded chain turns one
misconfigured run into a long walk through endpoints that are not there. Four is
enough for the cases the chain exists for (a cloud reader with a local
fallback, or a couple of each) and small enough that the walk stays short.
"""


def _validate_chain(endpoints: tuple[LocalVisionConfig, ...] | None) -> None:
    """The chain's shape, checked once the config is settled.

    Called from `_with_validated_endpoint` rather than while merging layers,
    for the reason `_merged_local_vision_config` states: a per-call override is
    allowed to rescue a file that names something unusable, and naming both
    `--local-vision-model` and `--local-vision-base-url` replaces the chain
    outright. Only the settled config is checked.

    `None` is not a chain anybody configured and is nothing to check - the
    top-level fields answer for it, and `_with_chain` derives the one entry.
    """
    if endpoints is None:
        return
    if not endpoints:
        raise DistillError(
            "E_BAD_OPTIONS",
            "local_vision",
            "'endpoints' was configured but names no endpoint. Remove it to use "
            "the default endpoint, or pass --no-caption-frames to run without vision.",
            {"endpoints": 0},
        )
    if len(endpoints) > MAX_ENDPOINTS:
        raise DistillError(
            "E_BAD_OPTIONS",
            "local_vision",
            f"'endpoints' names {len(endpoints)} endpoints; at most {MAX_ENDPOINTS} are allowed.",
            {"endpoints": len(endpoints), "max": MAX_ENDPOINTS},
        )
    # <!-- P3-D-020 --> ADR-0004 keeps the address out of identity, so what
    # names a reader is its model and whether it is remote. Two entries
    # agreeing on both derive the same candidate key, and at resolution time
    # there is nothing left to tell them apart with: whichever answered first
    # would publish under that key and the other would be served its
    # generation while the manifest named a different address.
    seen: dict[tuple[str, bool], int] = {}
    for index, entry in enumerate(endpoints):
        identity = (entry.model, config_is_non_local(entry))
        first = seen.get(identity)
        if first is not None:
            where = "remote" if identity[1] else "local"
            raise DistillError(
                "E_BAD_OPTIONS",
                "local_vision",
                f"endpoints {first} and {index} both name model '{entry.model}' "
                f"{where}ly, so they cannot be told apart by bundle key. "
                "Give them different models, or remove one.",
                # Both indices, because "two of your endpoints collide" leaves
                # the operator to work out which two out of four.
                {"entries": [first, index], "model": entry.model, "remote": identity[1]},
            )
        seen[identity] = index


def _with_validated_endpoint(config: LocalVisionConfig) -> LocalVisionConfig:
    """The same config, once its `base_url` is one Distill will speak to.

    A rejected endpoint is a **fatal error** here rather than a **warning**:
    unlike a server that is merely down, nothing about the run will make this
    address allowed, and silently degrading to OCR-only would hide that the
    operator's configuration was ignored.
    """
    _validate_chain(config.endpoints)
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
        # R-43's loopback rule and D-008's scheme rule are properties of one
        # endpoint, so a chain has to be held to them once per entry. Checking
        # only the top-level `base_url` would clear a chain whose first entry
        # is loopback and let entry 1 - the one that actually leaves the
        # machine - reach the network unexamined.
        try:
            _checked_endpoint_url(entry.base_url, allow_remote_endpoint=entry.allow_remote_endpoint)
        except EndpointRejected as exc:
            raise DistillError(
                "E_BAD_OPTIONS",
                "local_vision",
                f"endpoint {index}: {exc.message}",
                # The index, because "one of your endpoints is not allowed" is
                # not something an operator can act on. The rejection's own
                # detail still decides whether the URL itself can be named
                # (D-007).
                {"entry": index, **dict(exc.detail)},
            ) from exc
    try:
        _checked_endpoint_url(config.base_url, allow_remote_endpoint=config.allow_remote_endpoint)
    except EndpointRejected as exc:
        raise DistillError(
            "E_BAD_OPTIONS",
            "local_vision",
            exc.message,
            # The rejection's own detail is the whole story: sites that can
            # safely name the URL already include a sanitized `url`, and the
            # ones that omit it do so because the URL may carry a credential
            # (D-007). Re-adding the raw base_url here defeated exactly that.
            dict(exc.detail),
        ) from exc
    if (
        config.caption_frames
        and config.credential_configured
        and config.credential is None
        and config_is_non_local(config)
    ):
        # D-016's typo guard: a credential the operator *configured* resolved
        # to nothing, on a config that says remote is intended and whose host
        # is not literally loopback. Without allow_remote_endpoint the
        # per-request resolved-address check already proves the endpoint
        # local (D-029), so a hostname like `localhost` is not punished.
        # Configuring no credential at all remains intentional no-auth.
        named = (
            f" (env var '{config.credential_env}' is unset or empty)"
            if config.credential_env
            else ""
        )
        raise DistillError(
            "E_BAD_OPTIONS",
            "local_vision",
            f"local_vision credential is configured but empty{named}; refusing "
            "to send keyframes to a non-loopback endpoint without it.",
            {"reason": "credential_configured_but_empty", "api_key_env": config.credential_env},
        )
    return config


config_dir = general_config_dir
"""Where local-vision configuration lives: the general config directory.

Re-exported rather than resolved again here, because two answers to "where does
config live" is a machine where `distill.json` is read from one directory and
`distill.local-vision.json` from another. `config.py` owns the resolution; this
module owns only its own forgiving reader and coercions.
"""


def _read_json(path: Path) -> dict[str, Any]:
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


MAX_SOCKET_TIMEOUT_SEC = 2**63 / 1e9
"""The largest timeout `socket.settimeout` can hold, in seconds (~292 years).

CPython stores a timeout as nanoseconds in a signed 64-bit integer, so this is
where the representable range ends and `OverflowError: timestamp out of range
for platform time_t` begins. It is a fact about the call downstream rather than
a policy about how long a probe may wait: `inf` is not the only number a socket
cannot take, and a door that coerces has to land inside the range or it has not
coerced anything.
"""


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


def _merged_local_vision_config(base_dir: Path | None = None) -> LocalVisionConfig:
    """The configuration files, folded together, not yet validated.

    Unvalidated on purpose: the endpoint that matters is the one the run will
    use, and a per-call override is allowed to rescue a file that names a
    forbidden one. Only the settled config is checked.
    """
    root = (base_dir or config_dir()).expanduser()
    config = LocalVisionConfig()
    named: list[str] = []
    for filename in CONFIG_FILENAMES:
        payload = _read_json(root / filename)
        if filename == "distill.json":
            nested = payload.get("local_vision")
            payload = nested if isinstance(nested, dict) else {}
        # Recorded per layer and folded afterwards, because the conflict
        # P3-D-013 names can span two files that are each fine alone.
        named.extend(key for key in ENDPOINT_FIELD_NAMES if key in payload)
        config = _config_from_payload(payload, config)
    return replace(config, top_level_endpoint_fields=tuple(dict.fromkeys(named)))


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


def load_local_vision_config(base_dir: Path | None = None) -> LocalVisionConfig:
    return _with_chain(_with_validated_endpoint(_merged_local_vision_config(base_dir)))


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
    return _with_chain(_with_validated_endpoint(_config_from_payload(overrides, config)))


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
    """Confirm a vision endpoint is reachable and will serve the configured model.

    Probes ``GET <base_url>/models``. A model present in the catalog satisfies
    the probe as before; a model the catalog omits is settled by attempting a
    minimal image-bearing completion (D-008 - /models is advisory, a proxy's
    catalog is routinely incomplete), so the unlisted path may load the model
    and costs up to a second timeout. Transport failures map onto Distill's
    existing warning codes so the OCR-only fallback behaves as before.
    """
    models_url = _models_url(config.base_url)
    try:
        payload = _http_get_json(
            requestor,
            models_url,
            config.timeout_sec,
            allow_remote_endpoint=config.allow_remote_endpoint,
            credential=config.credential,
            budget=config.budget,
        )
    except EndpointRejected as exc:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code=exc.code,
            message=f"{exc.message} Continuing with OCR-only output.",
            # The rejection's own detail already carries a SANITIZED url when
            # one is safe to echo; overriding it with the raw models_url
            # defeated exactly that.
            detail=dict(exc.detail),
        )
    except LocalVisionFailure as exc:
        # Raised typed from the transport (e.g. auth_rejected). Its message
        # and detail are already credential- and body-free by construction.
        # A retryable (429/5xx) has no retry loop on the probe path: the
        # server erroring IS the probe's answer, reported as unavailable
        # with the status - "try again shortly" is the operator action.
        code, message = exc.code, exc.message
        if exc.code == "local_vision_retryable":
            code = "local_vision_rapid_mlx_unavailable"
            message = (
                f"Rapid-MLX answered HTTP {exc.detail.get('status')} to the "
                "probe; continuing with OCR-only output."
            )
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code=code,
            message=message,
            detail=dict(exc.detail),
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
        # An HTTP error or a non-envelope answer from /models says nothing
        # about completions (D-008: many providers and proxies serve no
        # catalog at all). The catalog becomes advisory-absent and the
        # attempted completion decides; only a transport-level failure above
        # stops the probe, because both endpoints share a server.
        _boundary_log("models_catalog_unavailable", {"error": str(exc)[:200], "url": models_url})
        payload = {"data": []}

    served = _served_model_ids(payload)
    if config.model not in served:
        # D-008: /models is advisory, not authoritative - a proxy's catalog is
        # routinely incomplete. A model the catalog omits is settled by
        # attempting a completion; only a failed attempt makes it unavailable.
        attempt_failure = _attempt_minimal_completion(config, requestor)
        if attempt_failure is not None:
            if attempt_failure.code == "local_vision_model_unavailable":
                message = (
                    f"Rapid-MLX is not serving model '{config.model}' and a "
                    "completion attempt failed; continuing with OCR-only output."
                )
            else:
                # An auth rejection, endpoint refusal, or transport fault is
                # its own event; reporting it as a missing model would hide
                # the one cause the operator can act on.
                message = attempt_failure.message
            return LocalVisionProbe(
                available=False,
                backend=config.backend,
                model=config.model,
                base_url=config.base_url,
                code=attempt_failure.code,
                message=message,
                detail={
                    "configured_model": config.model,
                    "served_models": served,
                    "url": models_url,
                    "completion_error": attempt_failure.message,
                    "completion_code": attempt_failure.code,
                    **{
                        k: v
                        for k, v in attempt_failure.detail.items()
                        if k not in {"configured_model", "served_models", "url"}
                    },
                },
            )
        return LocalVisionProbe(
            available=True,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_available",
            message=(
                f"Model '{config.model}' is not in the /models catalog but a "
                "completion succeeded; proceeding."
            ),
            detail={"served_models": served, "url": models_url, "model_not_listed": True},
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


def _attempt_minimal_completion(
    config: LocalVisionConfig,
    requestor: HttpRequestor | None,
) -> LocalVisionFailure | None:
    """One tiny completion carrying a 1x1 PNG image part, or the typed failure
    it produced. The image part is the point: the probe's question is "will
    this endpoint read a keyframe for me", and a text-only ping would authorize
    an image run against an endpoint that only routes text (D-008).

    Failures keep their own class: an auth rejection, a timeout, or a policy
    refusal must never be reported as a missing model.
    """
    body = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ping"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PROBE_PIXEL_B64}"},
                    },
                ],
            }
        ],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        envelope = _http_post_json(
            requestor,
            _completions_url(config.base_url),
            body,
            config.timeout_sec,
            allow_remote_endpoint=config.allow_remote_endpoint,
            credential=config.credential,
            budget=config.budget,
        )
        if not envelope.get("choices"):
            # A 200 without a completion envelope (several proxies answer
            # errors this way) is not evidence the endpoint serves the model.
            return LocalVisionFailure(
                "local_vision_model_unavailable",
                "completion attempt returned no completion envelope",
                {"envelope_keys": sorted(envelope.keys())},
            )
    except LocalVisionFailure as exc:
        # Typed already (auth_rejected, endpoint rejection, transport code):
        # pass it through untouched so the cause survives to the surface. A
        # retryable becomes unavailability: the attempt has no retry loop.
        if exc.code == "local_vision_retryable":
            return LocalVisionFailure(
                "local_vision_rapid_mlx_unavailable",
                f"completion attempt answered HTTP {exc.detail.get('status')}",
                dict(exc.detail),
            )
        return exc
    except TimeoutError as exc:
        return LocalVisionFailure(
            "local_vision_timeout",
            "completion attempt timed out; continuing with OCR-only output.",
            {"error": str(exc)},
        )
    except (urllib.error.URLError, OSError) as exc:
        return LocalVisionFailure(
            "local_vision_rapid_mlx_unavailable",
            f"completion attempt could not reach the endpoint: {exc}",
            {"error": str(exc)},
        )
    except (RuntimeError, ValueError) as exc:
        # An HTTP error or non-envelope answer to a well-formed completion for
        # this model: the model-shaped failure.
        return LocalVisionFailure(
            "local_vision_model_unavailable",
            f"completion attempt failed: {exc}",
            {"error": str(exc)},
        )
    return None


# The smallest valid PNG (1x1 transparent pixel), used only by the probe's
# attempted completion. A constant, so the probe body is byte-stable.
_PROBE_PIXEL_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
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
    try_interpret: TryInterpretImage = try_interpret_image_after_probe
    max_parallel: int = DEFAULT_MAX_PARALLEL
    debug: bool | None = None
    budget_sec: float = DEFAULT_VISION_STAGE_BUDGET_SEC
    budget_bytes: int = DEFAULT_VISION_STAGE_BUDGET_BYTES
    # D-017: the toggle gates window construction, so a disabled run never
    # asks the model for a judgment and records no salience at all.
    frame_salience: bool = True
    _last_probe: LocalVisionProbe | None = None
    _frame_count: int = 0
    _interpreted_count: int = 0
    _max_parallel: int = 1
    _transcript_segments: tuple[Any, ...] = ()
    _salience_requested: int = 0
    _salience_recorded: int = 0
    _warning_counts: dict[str, int] = field(default_factory=dict)
    _trace_events: list[dict[str, Any]] = field(default_factory=list)
    _breaker: _TransportBreaker = field(default_factory=_TransportBreaker)

    def interpret(
        self,
        frames: list[FrameArtifact],
        *,
        transcript_segments: Iterable[Any] | None = None,
    ) -> tuple[list[FrameArtifact], list[WarningRecord]]:
        self._transcript_segments = tuple(transcript_segments or ()) if self.frame_salience else ()
        self._reset_run(frames)
        self._log("interpret.start", {"frames": len(frames), "backend": self.config.backend})
        provenance_warnings: list[WarningRecord] = []
        if config_is_non_local(self.config):
            # D-012: both provenance surfaces - this warning and the identity
            # fold in options.cache_payload - read the SAME predicate on the
            # same input, before any I/O (including the probe's), so a bundle
            # keyed non-local always carries the disclosure.
            non_local = warning(
                "local_vision",
                "non_local_only_processing",
                "this run was configured to send keyframes and surrounding "
                "transcript text to a non-loopback vision endpoint; the "
                "generation may not be local-only processing.",
            )
            self._record_warning(non_local)
            provenance_warnings.append(non_local)
        probing_config = self._probing_config()
        probe = self.probe(probing_config)
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
            return frames, [*provenance_warnings, probe_warning]

        warnings: list[WarningRecord] = [*provenance_warnings]
        if probe.detail.get("model_not_listed"):
            # Available, but on the strength of an attempted completion rather
            # than the catalog (D-008) - the run says so once, out loud.
            advisory = warning(
                "local_vision",
                "local_vision_model_not_listed",
                f"model '{probe.model}' is not in the endpoint's /models "
                "catalog; proceeding because a completion succeeded.",
            )
            self._record_warning(advisory)
            warnings.append(advisory)
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
            resolved = self._resolved_config(probe, probing_config)
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

    def _probing_config(self) -> LocalVisionConfig:
        # The budget starts when the STAGE does - before the probe - and one
        # instance rides every request the stage makes, so the probe's GET
        # and attempted completion charge the same bounds as the frames
        # (D-008: run-wide means the run, not the run minus its first two
        # requests). Attached only when the config opts into a remote
        # endpoint: remote cost is the threat, and a slow-but-working local
        # model must not be half-degraded by a run-wide clock.
        if self.config.budget is None and self.config.allow_remote_endpoint:
            budget = VisionStageBudget(
                wall_clock_sec=float(self.budget_sec), max_bytes=int(self.budget_bytes)
            )
            return replace(self.config, budget=budget)
        return self.config

    def _resolved_config(
        self, probe: LocalVisionProbe, probing: LocalVisionConfig
    ) -> LocalVisionConfig:
        # `probing` is the config the probe actually used - the SAME budget
        # instance keeps charging through the frames (one run, one budget).
        if probe.backend != DEFAULT_LOCAL_VISION_BACKEND or not (probe.base_url and probe.model):
            return probing
        return replace(
            probing,
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
            "salience_requested": self._salience_requested,
            "salience_recorded": self._salience_recorded,
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
        window = (
            select_transcript_window(self._transcript_segments, frame.timestamp_sec)
            if self._transcript_segments
            else ""
        )
        prompt = build_technical_frame_prompt(
            ocr_text=frame.extracted_text or None,
            transcript_window=window or None,
        )
        result, frame_warning = self._try_interpret_frame(
            config,
            Path(frame.path),
            prompt.prompt,
            prompt_profile=prompt.profile,
        )
        if window:
            self._salience_requested += 1
        if result is not None and result.salience is not None and self.frame_salience and window:
            self._salience_recorded += 1
        if result is not None and (not self.frame_salience or not window):
            # Not asked means not recorded: whether the toggle is off or this
            # frame simply had no transcript window, a model that volunteers
            # the fields anyway does not get to write a judgment against
            # speech it was never shown (D-017, D-003).
            result = replace(result, salience=None)
        if frame_warning:
            warnings.append(frame_warning)
        read_code = frame_warning.get("code") if frame_warning else None
        if result is not None and not result.carries_a_reading:
            # R-39 where a reading arrives by any route: the transport path
            # rejects an empty payload as malformed, and a reading that reached
            # here saying nothing is the same non-answer one step later. It is
            # not attached and it is not counted, so a run cannot report a
            # frame as interpreted on the strength of an empty object.
            #
            # And it leaves by the same door, because it is the same finding: a
            # reading dropped silently left `read_code` unset, so no warning was
            # raised, no **grounding** was attached, and the breaker read the
            # attempt as a success. From every seam the frame was
            # indistinguishable from one the model read and had nothing to
            # remark on - which is the claim R-39 exists to refuse. Malformed
            # rather than a transport code: the response arrived, so it is
            # evidence the transport works and must reset the breaker's tally
            # rather than count toward it.
            result = None
            read_code = "local_vision_malformed_response"
            warnings.append(
                warning(
                    "local_vision",
                    read_code,
                    f"frame {frame.index or index + 1} interpretation carried no reading",
                )
            )
        if result:
            return self._interpreted(frame, result, warnings), True, read_code
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
        # Whatever `try_interpret` is - the probe-skipping default, or a fake a
        # test injected - it is called the one way. The default already skips
        # the backend re-check the probe just did, so there is no default to
        # tell apart from an injection here: the `is try_interpret_image` guard
        # that used to do that was a test seam reaching into production control
        # flow, and it is gone.
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
        warnings: list[WarningRecord],
    ) -> FrameArtifact:
        """The frame carrying the model's reading and Distill's grounding of it.

        **Grounding** is assessed against the reading the model returned, and
        the reading is then handed to the carrier, which is where the
        **redaction** policy runs over every field of it (R-19). Assessing
        first is deliberate: grounding compares two readers' words on the
        SAME text. Since M2.6 the model reads prompt-side-redacted text, so
        the comparison side passes through the same sink - on the default
        path an idempotent no-op (the carrier already redacted), and under
        --no-redact-secrets the one change that keeps a secret-bearing frame
        from being marked uncorroborated for echoing [REDACTED] honestly.
        """
        from .redact_secrets import redact_for_prompt

        assessment = assess_grounding(
            ocr_text=redact_for_prompt(frame.extracted_text),
            verbatim_text=redact_for_prompt(reading.verbatim_text),
            text_confidence=reading.text_confidence,
            has_interpretation=reading.has_interpretation,
            carries_a_reading=reading.carries_a_reading,
        )
        carried, carrier_warnings = frame.with_interpretation(
            reading, grounding=assessment.public_dict()
        )
        warnings.extend(carrier_warnings)
        # No ungrounded warning: since M3.1 (D-002) the vision reading is the
        # authoritative one, so "not grounded in readable text" is not a defect
        # to report - it is the ordinary case for a frame the image-text reader
        # could not read. The level is still recorded on the artifact and the
        # render carries its neutral note; a warning would re-deliver the
        # retired confidence verdict through a channel consumers read as a
        # credibility signal, and would reintroduce the document-sourced reason
        # text the render just stopped printing.
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
        made `debug_info` and the **manifest** disagree about the same run, so
        the count is read the one way `errors.occurrences_of` reads it.

        Keyed on the code alone, where `aggregate_warnings` keys on (stage,
        code) - deliberately, and worth stating because the two numbers can
        differ. This is a debug tally for one stage's run, and the warnings
        reaching it are the vision pass's own plus the ones a carrier raised
        while redacting the model's words (`redaction`, `artifacts`). A code
        that appeared under two stages is one entry here and two records in the
        **manifest**; the total is the same either way, and it is the total
        this field is read for.
        """
        code = warning_payload.get("code", "unknown")
        self._warning_counts[code] = self._warning_counts.get(code, 0) + occurrences_of(
            warning_payload
        )

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


def _post_completion_with_retries(
    requestor: HttpRequestor | None,
    completions_url: str,
    body: dict[str, Any],
    config: LocalVisionConfig,
) -> dict[str, Any]:
    """One completion POST, honoring 429/5xx Retry-After within bounds.

    The server's requested pause is honored but capped, the retries are
    counted, and when both are spent the frame degrades with its own code -
    a rate limit is a delivered response, so it must not look like a
    transport fault to the breaker (D-008).
    """
    retries_left = RETRYABLE_MAX_RETRIES
    while True:
        try:
            return _http_post_json(
                requestor,
                completions_url,
                body,
                config.timeout_sec,
                allow_remote_endpoint=config.allow_remote_endpoint,
                credential=config.credential,
                budget=config.budget,
            )
        except LocalVisionFailure as exc:
            if exc.code != "local_vision_retryable":
                raise
            if retries_left <= 0:
                status = exc.detail.get("status")
                raise LocalVisionFailure(
                    "local_vision_retry_exhausted",
                    f"Local vision endpoint kept answering HTTP {status} after "
                    f"{RETRYABLE_MAX_RETRIES} retries; continuing with OCR-only "
                    "output for this keyframe.",
                    {"status": status, "retries": RETRYABLE_MAX_RETRIES},
                ) from exc
            retries_left -= 1
            retry_after = exc.detail.get("retry_after")
            requested = float(retry_after) if retry_after is not None else DEFAULT_RETRY_BACKOFF_SEC
            # Clamped on both ends: a negative or NaN Retry-After is the
            # endpoint's bug, not a crash in this process, and zero means
            # "retry now", not "use the default".
            _sleep(max(0.0, min(requested, RETRY_AFTER_CAP_SEC)))


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
    # The prompt builder owns the response schema (it already names every
    # field, salience included when a window exists); a second field list
    # appended here was the LAST thing the model read, and it omitted the
    # salience fields - so no run ever recorded one.
    request_prompt = prompt
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
        # D-008: never a stream. A streamed answer defeats the absolute
        # deadline and the response cap alike - the bound is on the whole
        # answer, not on each drip of it.
        "stream": False,
    }
    completions_url = _completions_url(config.base_url)
    last_preview = ""
    for _attempt in range(DEFAULT_MAX_ATTEMPTS):
        try:
            envelope = _post_completion_with_retries(
                requestor,
                completions_url,
                body,
                config,
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
    salience = parse_frame_salience(interpreted)
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
        salience=None if salience is None else salience.document(),
    )
