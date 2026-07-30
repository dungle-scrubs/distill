"""The Rapid-MLX vision client: how Distill talks to the local server and what it will listen to.

This module owns the transport (the HTTP request, the opener that follows no
redirect and consults no proxy, the 32 MiB read bound), the OpenAI-style
envelope parsing, and the endpoint policy (scheme, the loopback rule and its
per-request re-check, the refusal to follow a redirect). It is one
OpenAI-compatible client, not a provider abstraction: `backend` names the wire
protocol rather than a vendor, Rapid-MLX is the default endpoint rather than the
only one (ADR-0005, superseding ADR-0001), and no backend branch belongs here.

The endpoint policy lives next to the requests it governs, which is the point:
a rule kept beside the request it guards cannot be reached around by adding a
caller. `local_vision` owns the configuration and the frame-interpretation pass
and calls into this module; it does not reimplement any of what is here.
"""

from __future__ import annotations

import hmac
import http.client
import ipaddress
import json
import logging
import math
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, NoReturn

from .artifacts import (
    SALIENCE_REASON_MAX_CHARS,
    FrameSalience,
    document_carries_a_reading,
)
from .errors import warning
from .redact_secrets import redact_for_prompt
from .vision_prompts import FRAME_KINDS, TEXT_CONFIDENCE_LEVELS

# The boundary events (`endpoint_rejected`, and every `_boundary_log` record)
# keep the channel they had before this module existed: they carry
# `"type": "distill.local_vision"` and a consumer filters on that name, so the
# client logging under its own module name would silently move an observability
# channel plan 00 hardened. The events are the local-vision subsystem's; the
# code moving files does not change whose events they are.
LOGGER = logging.getLogger("distill.local_vision")

ENDPOINT_REJECTED_CODE = "local_vision_endpoint_rejected"


ALLOWED_ENDPOINT_SCHEMES = frozenset({"http", "https"})


DEFAULT_SCHEME_PORTS = {"http": 80, "https": 443}


MAX_RESPONSE_BYTES = 32 * 1024 * 1024


ERROR_BODY_PREVIEW_BYTES = 2048


# Injectable clock for the absolute deadline, so a test can drip a body
# through simulated time instead of real sleeps.
_monotonic = time.monotonic

# The absolute deadline is enforced between chunk reads, so the chunk is how
# often a dripping response gets measured against the clock.
_READ_CHUNK_BYTES = 64 * 1024


# Run-wide vision-stage budget defaults (D-008). They exist to bound REMOTE
# cost - the interpreter only attaches a budget when the config opts into a
# remote endpoint, because a slow local model is not the threat and must not
# be half-degraded by a run-wide clock. The probe runs before the budget is
# created, so its two requests are bounded by timeout_sec alone.
# 3600s clears the run's own worst case (default 80 keyframes x 30s
# timeout = 2400s) with margin; the bound exists for runaway remote cost,
# not to race a working endpoint.
DEFAULT_VISION_STAGE_BUDGET_SEC = 3600.0
DEFAULT_VISION_STAGE_BUDGET_BYTES = 256 * 1024 * 1024


class VisionStageBudget:
    """Run-wide bounds for the vision stage: wall clock and received bytes.

    Charged by the transport as chunks arrive, checked before each request.
    Exhaustion raises the typed budget failure, which the interpreter treats
    as immediate: the remainder degrades to OCR-only and every frame already
    interpreted is kept (D-008).
    """

    CODE = "local_vision_budget_exhausted"

    def __init__(self, *, wall_clock_sec: float, max_bytes: int) -> None:
        if wall_clock_sec <= 0 or max_bytes <= 0:
            raise ValueError("vision stage budget bounds must be positive")
        self._deadline = _monotonic() + wall_clock_sec
        self._wall_clock_sec = wall_clock_sec
        self._max_bytes = max_bytes
        self._remaining = max_bytes
        self._lock = threading.Lock()
        self._exhausted = False

    def _exhaust(self, why: str) -> NoReturn:
        if not self._exhausted:
            # Latched: one run, one budget_exhausted event, however many
            # threads cross the line together.
            self._exhausted = True
            _boundary_log(
                "budget_exhausted",
                {"why": why, "wall_clock_sec": self._wall_clock_sec, "max_bytes": self._max_bytes},
            )
        raise LocalVisionFailure(
            self.CODE,
            f"the vision stage's run-wide budget is spent ({why}); remaining "
            "keyframes continue with OCR-only output.",
            {"why": why, "wall_clock_sec": self._wall_clock_sec, "max_bytes": self._max_bytes},
        )

    def check(self) -> None:
        with self._lock:
            if _monotonic() > self._deadline:
                self._exhaust("wall clock")
            if self._remaining <= 0:
                self._exhaust("bytes")

    def charge(self, received_bytes: int) -> None:
        # Both bounds, per chunk: crossing either mid-body cuts the response
        # off in flight - the bound is on the run, not on how the endpoint
        # chunks it or how long it takes to drip.
        with self._lock:
            self._remaining -= received_bytes
            if self._remaining < 0:
                self._exhaust("bytes")
            if _monotonic() > self._deadline:
                self._exhaust("wall clock")


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


class SecretCredential:
    """The non-serializable carrier for a vision-endpoint credential (D-007).

    ``reveal()`` is the only door to the value. Every text form redacts, and
    every generic serialization path - JSON, pickle, deepcopy - is refused
    rather than redacted, because a copy that silently dropped the secret
    would *look* safe while a copy that kept it would leak; failing loudly is
    the only honest behavior for both.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretCredential(***)"

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError("SecretCredential cannot be serialized or copied")

    def __eq__(self, other: object) -> bool:
        # Value equality (constant-time out of caution): two configs differing
        # only by credential must never compare interchangeable. Cache keys
        # deliberately exclude the credential; this is about equality, not
        # about putting secrets in a key.
        if not isinstance(other, SecretCredential):
            return NotImplemented
        return hmac.compare_digest(self._value.encode(), other._value.encode())

    def __hash__(self) -> int:
        return hash(self._value)


AddressResolver = Callable[[str, int], list[str]]


# The injectable request seam is credential-blind by design: a fake server
# fed through `requestor` never sees headers, so credential behavior MUST be
# tested at the opener level (patch `_OPENER`), never through this seam - a
# credential test written against it passes vacuously.
HttpRequestor = Callable[..., "dict[str, Any]"]


def _request_headers(credential: SecretCredential | None) -> dict[str, str]:
    """The request headers, with `Authorization: Bearer` exactly when a
    credential is carried - never an empty or placeholder header (D-016)."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if credential is not None:
        headers["Authorization"] = f"Bearer {credential.reveal()}"
    return headers


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

    Every rejection *this module makes* goes through here, so the
    observability is a property of the rule rather than of whoever wrote the
    call site: there is one event, it always carries the reason, and no new
    rule can forget to emit it.

    Not every refusal is one of this module's. urllib turns a 3xx it cannot
    act on into an `HTTPError` before `_RedirectsAreRejected` sees it, so a
    redirect with no `Location`, or one naming a scheme urllib will not open,
    ends the request without an `endpoint_rejected` event. Nothing is followed
    in either case - which is why the fix there was the message an operator
    gets, not a handler override to route the refusal back through here.
    """
    for url_key in ("url", "location"):
        # Centrally, so no rejection site can forget: an echoed URL never
        # carries userinfo, query, fragment, or path (D-007). `location` is a
        # server-chosen value and gets the same treatment.
        if url_key in detail:
            detail[url_key] = _safe_url_for_message(str(detail[url_key]))
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
        # Unavailability, not a policy verdict: nothing was rejected, the host
        # could not be FOUND. The transport-failure code routes this to the
        # 3-strike breaker like a refused TCP connect - a transient DNS flap
        # must not condemn a whole run the way a real rejection does.
        _boundary_log("endpoint_unresolvable", {"host": host, "error": str(exc)})
        raise LocalVisionFailure(
            "local_vision_rapid_mlx_unavailable",
            f"Local vision endpoint host '{host}' does not resolve; "
            "continuing with OCR-only output.",
            {"host": host, "error": str(exc)},
        ) from exc
    return [str(info[4][0]) for info in infos]


def _parse_retry_after(header: str) -> float | None:
    """RFC 9110 Retry-After: delay-seconds or an HTTP-date, else None.

    Non-finite and unparsable values are None - the caller's default backoff
    covers them, and an endpoint's arithmetic is never this process's crash.
    """
    if not header:
        return None
    try:
        seconds = float(header)
        return seconds if math.isfinite(seconds) else None
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        return None
    import datetime as _dt

    return (target - _dt.datetime.now(_dt.UTC)).total_seconds()


def _safe_url_for_message(url: str) -> str:
    """The echoable form of an operator-supplied URL: userinfo, query, and
    fragment stripped. String surgery rather than urlsplit, because this runs
    precisely when the URL may have failed to parse - and a secret must not
    survive the echo either way."""
    trimmed = url.split("#", 1)[0].split("?", 1)[0]
    scheme, sep, rest = trimmed.partition("://")
    if not sep:
        scheme, sep, rest = "", "", trimmed
    if "@" in rest:
        rest = rest.rsplit("@", 1)[1]
    # Authority only: a path can carry a key too (`/v1/sk-...`), and the
    # diagnostic value of an echoed URL is the scheme, host, and port.
    rest = rest.split("/", 1)[0]
    return f"{scheme}{sep}{rest}"


def _static_host_is_loopback(url: str) -> bool:
    """True only when the host is a *literal* loopback address. A hostname
    cannot be settled without a lookup, so callers deciding statically must
    treat a name as potentially remote."""
    try:
        host = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _checked_endpoint_url(url: str, *, allow_remote_endpoint: bool) -> tuple[str, str, int]:
    """R-43's static half: what can be decided about `url` without asking anyone.

    The scheme, the presence of a host, and - when the host is written as an
    address - whether that address is loopback. A host written as a *name*
    cannot be settled here without a lookup, so it is settled per request
    instead (D-029); this is the check a config can be rejected by, not the
    authoritative one.
    """
    try:
        # All three raise on a malformed authority, at different moments: a
        # bracket that never closes stops `urlsplit` itself, a port out of
        # range is only noticed when `parts.port` is read. Splitting first and
        # guarding only the reads left the earlier one escaping as
        # `ValueError("Invalid IPv6 URL")` - a traceback out of the CLI about
        # something an operator typed in a config file. A rejection saying so
        # is the answer they can act on.
        parts = urllib.parse.urlsplit(url)
        scheme = parts.scheme
        host, configured_port = parts.hostname, parts.port
    except ValueError as exc:
        _reject_endpoint(
            "unparsable_url",
            f"Local vision endpoint '{_safe_url_for_message(url)}' is not a usable URL: {exc}",
            url=_safe_url_for_message(url),
        )
    if parts.username is not None or parts.password is not None:
        # D-007: no credential-in-URL. Only the sanitized form is echoed - the
        # thing being rejected is exactly the thing that must not appear in an
        # error surface.
        _reject_endpoint(
            "credential_in_url",
            f"Local vision endpoint '{_safe_url_for_message(url)}' embeds "
            "userinfo; credentials belong in api_key_env, never in base_url.",
            url=_safe_url_for_message(url),
        )
    if parts.query or parts.fragment:
        # Key names are safe to echo and are what the operator needs to see;
        # values are exactly what must not be echoed (D-007).
        keys = [key for key, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
        _reject_endpoint(
            "query_in_base_url",
            f"Local vision endpoint '{_safe_url_for_message(url)}' carries a "
            f"query or fragment ({', '.join(keys) if keys else 'fragment'}); "
            "a base_url takes no parameters, and credentials belong in "
            "api_key_env.",
            url=_safe_url_for_message(url),
            query_keys=keys,
        )
    if host is not None and ("%" in host or "@" in host):
        # Percent-escaped userinfo survives urlsplit as a "hostname"; a real
        # hostname never contains '%' or '@'. Rejected without echoing the
        # host at all - it may be a mangled credential.
        _reject_endpoint(
            "invalid_host_characters",
            "Local vision endpoint host contains characters that do not "
            "belong in a hostname (percent-escapes or '@'); if this was an "
            "attempt to embed a credential, use api_key_env instead.",
        )
    if scheme not in ALLOWED_ENDPOINT_SCHEMES:
        _reject_endpoint(
            "scheme",
            f"Local vision endpoint scheme '{scheme}' is not allowed; "
            f"use {' or '.join(sorted(ALLOWED_ENDPOINT_SCHEMES))}.",
            url=url,
            scheme=scheme,
        )
    if not host:
        _reject_endpoint(
            "host_missing",
            f"Local vision endpoint '{_safe_url_for_message(url)}' names no host.",
            url=url,
        )
    port = configured_port or DEFAULT_SCHEME_PORTS[scheme]
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if not allow_remote_endpoint:
        if address is not None and not address.is_loopback:
            _reject_endpoint(
                "non_loopback_host",
                f"Local vision endpoint host '{host}' is not loopback; set "
                "local_vision_allow_remote_endpoint to reach one deliberately.",
                url=url,
                host=host,
            )
    elif scheme == "http" and address is not None and not address.is_loopback:
        # A remote endpoint requires https: a bearer credential (or a keyframe)
        # must never cross the network in cleartext. `http` exists for loopback
        # only, wherever the opt-in stands (D-008).
        _reject_endpoint(
            "http_off_loopback",
            f"Local vision endpoint host '{host}' is not loopback and the "
            "scheme is http; a remote endpoint requires https.",
            url=url,
            host=host,
        )
    return scheme, host, port


def _check_resolved_address(
    host: str,
    port: int,
    *,
    allow_remote_endpoint: bool,
    scheme: str,
    resolver: AddressResolver | None = None,
) -> list[str]:
    """R-43's authoritative half: every address `host` resolves to is loopback.

    Every address, not the first: a name answering with both 127.0.0.1 and a
    link-local address is a name that can steer the very next connection off
    loopback, and which of its answers gets connected to is not this function's
    to know.

    The remote opt-in exempts only `https` (D-008): `http` must prove every
    resolved address loopback wherever the opt-in stands, because a bearer
    credential or a keyframe must never cross the network in cleartext.
    """
    if allow_remote_endpoint and scheme != "http":
        return []
    addresses = (resolver or _resolve_addresses)(host, port)
    if not addresses:
        _boundary_log("endpoint_unresolvable", {"host": host})
        raise LocalVisionFailure(
            "local_vision_rapid_mlx_unavailable",
            f"Local vision endpoint host '{host}' resolved to no address; "
            "continuing with OCR-only output.",
            {"host": host},
        )
    reject_reason, reject_hint = (
        ("http_off_loopback", "a remote endpoint requires https")
        if allow_remote_endpoint
        else (
            "non_loopback_address",
            "set local_vision_allow_remote_endpoint to reach one deliberately",
        )
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
                reject_reason,
                f"Local vision endpoint host '{host}' resolves to {address}, which is not "
                f"loopback; {reject_hint}.",
                host=host,
                address=address,
            )
    return addresses


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
    credential: SecretCredential | None = None,
    budget: VisionStageBudget | None = None,
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
            credential=credential,
            budget=budget,
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
    credential: SecretCredential | None = None,
    budget: VisionStageBudget | None = None,
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
            credential=credential,
            budget=budget,
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
    credential: SecretCredential | None = None,
    budget: VisionStageBudget | None = None,
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
    # The deadline is absolute from entry: policy resolution (which may hit
    # DNS), connect, send, and every chunk of the answer all charge it. One
    # caveat stated rather than hidden: a single blocked socket operation is
    # bounded by the per-op timeout, so the true worst case is deadline plus
    # one socket operation (D-008).
    deadline = _monotonic() + timeout_sec
    scheme, host, port = _checked_endpoint_url(url, allow_remote_endpoint=allow_remote_endpoint)
    _check_resolved_address(
        host,
        port,
        allow_remote_endpoint=allow_remote_endpoint,
        scheme=scheme,
        resolver=resolver,
    )
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=_request_headers(credential),
    )
    # The deadline is absolute and covers the whole request - connect, send,
    # and every chunk of the answer. urllib's timeout is per socket operation,
    # so without this a body arriving one drip at a time never times out
    # (D-008).
    if budget is not None:
        budget.check()
    deadline = _monotonic() + timeout_sec
    try:
        with _OPENER.open(request, timeout=timeout_sec) as response:
            # One byte past the cap, so a body that is exactly at it still
            # arrives and a body over it is recognisable without being read.
            # The header is not consulted: a Content-Length is the sender's
            # claim about the sender, and R-44 is a bound on this process.
            chunks: list[bytes] = []
            received = 0
            while received <= MAX_RESPONSE_BYTES:
                if _monotonic() > deadline:
                    raise TimeoutError(f"response did not complete within {timeout_sec} seconds")
                # read1, not read: BufferedIOBase.read(n) blocks until n bytes
                # accumulate, so a dripping body would never return to the
                # deadline check. read1 yields whatever one raw read produces,
                # keeping the clock honest per drip.
                chunk = response.read1(min(_READ_CHUNK_BYTES, MAX_RESPONSE_BYTES + 1 - received))
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if budget is not None:
                    # Charged as it arrives, so a dripping over-budget body is
                    # cut off mid-flight, not billed after the fact.
                    budget.charge(len(chunk))
            payload = b"".join(chunks)
        if len(payload) > MAX_RESPONSE_BYTES:
            # A delivered response, in M7.2's sense: it arrived, so it is
            # evidence the transport works and must not count toward the
            # breaker's consecutive-failure tally. It is unusable for the same
            # reason a truncated body is - hence malformed, not unavailable.
            raise RuntimeError(f"response from {url} exceeds the {MAX_RESPONSE_BYTES} byte cap")
        raw = payload.decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            # D-007: an auth failure's error surface carries neither the
            # credential nor the endpoint's error body - a reflected body is
            # the provider's text, and quoting it here would hand a hostile
            # endpoint a channel into Distill's own error output.
            _boundary_log("auth_rejected", {"status": exc.code, "url": _safe_url_for_message(url)})
            raise LocalVisionFailure(
                "local_vision_auth_rejected",
                f"Local vision endpoint rejected the credential (HTTP {exc.code}); "
                "continuing with OCR-only output.",
                {"status": exc.code, "url": _safe_url_for_message(url)},
            ) from exc
        if exc.code == 429 or 500 <= exc.code < 600:
            retry_after_header = ""
            if getattr(exc, "headers", None) is not None:
                retry_after_header = str(exc.headers.get("Retry-After") or "")
            retry_after = _parse_retry_after(retry_after_header)
            raise LocalVisionFailure(
                "local_vision_retryable",
                f"Local vision endpoint answered HTTP {exc.code}; retrying is "
                "allowed within bounds.",
                {"status": exc.code, "retry_after": retry_after},
            ) from exc
        # Surface HTTP error bodies (e.g. model-not-loaded) as a RuntimeError so
        # the probe/interpret paths can map them onto warning codes. Bounded
        # like any other body (R-44), and tighter: only a preview of it is ever
        # used. Read only after the auth branch, and NEVER on an authenticated
        # request: a server that echoes the Authorization header into its
        # error body would otherwise reflect the credential into diagnostics.
        if credential is not None:
            detail = "<error body withheld: authenticated request>"
        elif hasattr(exc, "read"):
            detail = exc.read(ERROR_BODY_PREVIEW_BYTES).decode("utf-8", errors="replace")
        else:
            detail = ""
        if 300 <= exc.code < 400:
            # A redirect this module never got to refuse: urllib turns a 3xx it
            # cannot act on - no `Location`, or one naming a scheme it will not
            # open - into an `HTTPError` before `_RedirectsAreRejected` is
            # consulted. A 3xx carries no body, so quoting one leaves an
            # operator with `HTTP 302 from …: ` and nothing to act on. Nothing
            # is followed either way; only the message was the mystery.
            raise RuntimeError(
                f"HTTP {exc.code} from {_safe_url_for_message(url)}: the endpoint answered "
                "a redirect that could not be followed, and redirects are not followed in "
                "any case"
            ) from exc
        # The URL is sanitized (authority only) and the endpoint's text is a
        # bounded 200-char preview: diagnostic enough for "model not loaded",
        # too small to be a channel.
        raise RuntimeError(
            f"HTTP {exc.code} from {_safe_url_for_message(url)}: {detail[:200]}"
        ) from exc
    except http.client.IncompleteRead as exc:
        # A server that closes the connection mid-body (crash/restart) yields a
        # truncated read; treat it as a malformed response so we degrade cleanly.
        raise RuntimeError(f"incomplete response from {_safe_url_for_message(url)}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = "<body withheld: authenticated request>" if credential is not None else raw[:200]
        raise ValueError(
            f"non-JSON response from {_safe_url_for_message(url)}: {preview!r}"
        ) from exc


def _normalize_frame_kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    return kind if kind in FRAME_KINDS else ""


def _normalize_text_confidence(value: Any) -> str:
    level = str(value or "").strip().lower()
    return level if level in TEXT_CONFIDENCE_LEVELS else "none"


def parse_frame_salience(payload: Any) -> FrameSalience | None:
    """The model's salience judgment, strictly validated, or None.

    `adds_information` must be a JSON boolean: a "false" string, a number, a
    missing field, or a non-mapping payload all yield ABSENT salience rather
    than a guessed value (D-003) - absence is honest, coercion is invention.
    """
    if not isinstance(payload, dict):
        return None
    adds_information = payload.get("adds_information")
    if not isinstance(adds_information, bool):
        return None
    raw_reason = payload.get("reason")
    # A non-string reason is refused, not stringified: repr()ing a structure
    # into a reader-facing sentence is the coercion this parser exists to
    # avoid. REDACTED before the cap is measured (D-010's ordering, same as
    # the prompt path): cutting first can split a credential mid-syntax so
    # the carrier's redactor no longer matches it, storing the secret whole.
    reason = redact_for_prompt(raw_reason).strip() if isinstance(raw_reason, str) else ""
    truncated = len(reason) > SALIENCE_REASON_MAX_CHARS
    if truncated:
        reason = reason[:SALIENCE_REASON_MAX_CHARS]
    return FrameSalience(
        adds_information=adds_information, reason=reason, reason_truncated=truncated
    )


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
