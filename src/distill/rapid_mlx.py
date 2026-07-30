"""The Rapid-MLX vision client: how Distill talks to the local server and what it will listen to.

This module owns the transport (the HTTP request, the opener that follows no
redirect and consults no proxy, the 32 MiB read bound), the OpenAI-style
envelope parsing, and the endpoint policy (scheme, the loopback rule and its
per-request re-check, the refusal to follow a redirect). It is a client for the
one backend Distill supports (ADR-0001) - not a provider abstraction, and no
backend branch belongs here.

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
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, NoReturn

from .artifacts import document_carries_a_reading
from .errors import warning
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


def _checked_endpoint_url(url: str, *, allow_remote_endpoint: bool) -> tuple[str, int]:
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
        _reject_endpoint("host_missing", f"Local vision endpoint '{url}' names no host.", url=url)
    port = configured_port or DEFAULT_SCHEME_PORTS[scheme]
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
        headers=_request_headers(credential),
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
        # Surface HTTP error bodies (e.g. model-not-loaded) as a RuntimeError so
        # the probe/interpret paths can map them onto warning codes. Bounded
        # like any other body (R-44), and tighter: only a preview of it is ever
        # used, so reading a gigabyte to quote 200 characters of it is a cost
        # with no reader. Read only after the auth branch: a 401/403 body is
        # never used, so a hostile one is never decoded at all.
        detail = (
            exc.read(ERROR_BODY_PREVIEW_BYTES).decode("utf-8", errors="replace")
            if hasattr(exc, "read")
            else ""
        )
        if 300 <= exc.code < 400:
            # A redirect this module never got to refuse: urllib turns a 3xx it
            # cannot act on - no `Location`, or one naming a scheme it will not
            # open - into an `HTTPError` before `_RedirectsAreRejected` is
            # consulted. A 3xx carries no body, so quoting one leaves an
            # operator with `HTTP 302 from …: ` and nothing to act on. Nothing
            # is followed either way; only the message was the mystery.
            raise RuntimeError(
                f"HTTP {exc.code} from {url}: the endpoint answered a redirect that could not "
                "be followed, and redirects are not followed in any case"
            ) from exc
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:200]}") from exc
    except http.client.IncompleteRead as exc:
        # A server that closes the connection mid-body (crash/restart) yields a
        # truncated read; treat it as a malformed response so we degrade cleanly.
        raise RuntimeError(f"incomplete response from {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"non-JSON response from {url}: {raw[:200]!r}") from exc


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
