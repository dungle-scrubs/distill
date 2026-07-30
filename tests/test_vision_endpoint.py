"""Where the vision client is allowed to talk, and how much it will listen to.

R-43: the endpoint is validated - scheme restricted to `http`/`https`, host
restricted to loopback unless the operator opts out - redirects are not
followed, and the *resolved* address is checked on every request (D-029), not
once when the config was read. R-44: a response body stops at 32 MiB.

Loopback here means the address, never the spelling: a name that resolves to
127.0.0.1 is loopback and a name that resolves to 169.254.169.254 is not, so
every test drives a fake resolver rather than asking the machine's.
"""

from __future__ import annotations

import http.client
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from distill.errors import DistillError
from distill.local_vision import (
    ERROR_BODY_PREVIEW_BYTES,
    MAX_RESPONSE_BYTES,
    LocalVisionConfig,
    LocalVisionFailure,
    _build_opener,
    _urlopen_json,
    local_vision_config_from_args,
    probe_local_vision,
    try_interpret_image,
)
from distill.options import DistillOptions

LOOPBACK_URL = "http://127.0.0.1:8000/v1"
# https, not http: since M2.3 (D-008) a non-loopback endpoint requires TLS
# even with the remote opt-in, so the URL every opt-in test reaches for is
# one the policy actually permits.
PRIVATE_URL = "https://10.0.0.5:8000/v1"
MODELS_URL = f"{LOOPBACK_URL}/models"
LINK_LOCAL_URL = "http://169.254.169.254/latest/meta-data/"


class _CannedResponse:
    """The part of an HTTP response `_urlopen_json` and urllib's handlers read.

    `endless` is a body that never runs out: every `read(size)` returns `size`
    bytes and an unbounded read fails outright, so a test cannot pass by
    accident on a cap that is not there.
    """

    def __init__(
        self,
        url: str,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        endless: bool = False,
    ) -> None:
        self.url = url
        self.code = status
        self.status = status
        self.msg = "canned"
        self.headers = http.client.HTTPMessage()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        self._body = body
        self._endless = endless
        self._offset = 0
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._endless:
            if size < 0:
                raise AssertionError("an unbounded read of an endless body never returns")
            return b"a" * size
        if size < 0:
            chunk, self._offset = self._body[self._offset :], len(self._body)
            return chunk
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        return None

    def info(self) -> http.client.HTTPMessage:
        return self.headers

    def geturl(self) -> str:
        return self.url

    def getcode(self) -> int:
        return self.code

    def __enter__(self) -> _CannedResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _FakeTransport(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    """A transport that serves canned responses and records what was asked for.

    An `HTTPHandler`/`HTTPSHandler` subclass so `build_opener` installs it
    *instead of* the real ones: everything above it in the handler chain -
    redirects included - is the production chain, and nothing below it is a
    socket. Both schemes, because since M2.3 the opt-in tests reach for https
    URLs (a remote endpoint requires TLS).
    """

    def __init__(self, responses: dict[str, _CannedResponse]) -> None:
        self.responses = responses
        self.requested: list[str] = []
        self.dialled: list[str] = []

    def http_open(self, req: urllib.request.Request) -> Any:
        self.requested.append(req.full_url)
        # What urllib would have connected to, which a proxy rewrites and the
        # URL does not show.
        self.dialled.append(req.host)
        if req.full_url not in self.responses:
            raise AssertionError(f"unexpected request: {req.full_url}")
        return self.responses[req.full_url]

    https_open = http_open


def _json_response(url: str, payload: Any) -> _CannedResponse:
    return _CannedResponse(url, body=json.dumps(payload).encode())


def _redirect_response(url: str, location: str) -> _CannedResponse:
    return _CannedResponse(url, status=302, headers={"Location": location})


def test_a_non_loopback_host_is_rejected_without_the_opt_out(tmp_path: Path) -> None:
    """R-43: a configured endpoint off loopback is refused, not honored.

    The vision client posts a **keyframe** the run just captured to whatever
    the config names. A host that is not loopback is a host that is not this
    machine, so it is rejected at the seam that settles the config rather than
    discovered at the wire.
    """
    with pytest.raises(DistillError, match="loopback"):
        local_vision_config_from_args({"local_vision_base_url": PRIVATE_URL}, tmp_path)

    (tmp_path / "distill.local-vision.json").write_text(json.dumps({"base_url": PRIVATE_URL}))
    with pytest.raises(DistillError, match="loopback"):
        local_vision_config_from_args({}, tmp_path)

    with pytest.raises(DistillError, match="loopback"):
        DistillOptions.from_args({"local_vision_base_url": PRIVATE_URL})


def test_the_endpoint_scheme_is_restricted_to_http_and_https(tmp_path: Path) -> None:
    """R-43: a scheme that is not HTTP is refused, opt-out or not.

    The opt-out is about *where* the endpoint is, so it must not widen *what*
    the endpoint is: `file://` and `gopher://` are rejected either way.
    """
    for url in ("file:///etc/passwd", "gopher://127.0.0.1:70/", "ftp://127.0.0.1/v1"):
        with pytest.raises(DistillError, match="scheme"):
            local_vision_config_from_args({"local_vision_base_url": url}, tmp_path)
        with pytest.raises(DistillError, match="scheme"):
            local_vision_config_from_args(
                {"local_vision_base_url": url, "local_vision_allow_remote_endpoint": True},
                tmp_path,
            )

    for url in (LOOPBACK_URL, "https://127.0.0.1:8443/v1", "http://[::1]:8000/v1"):
        assert local_vision_config_from_args(
            {"local_vision_base_url": url}, tmp_path
        ).base_url == url.rstrip("/")


@pytest.mark.parametrize("source_type", ("local", "youtube"))
def test_the_opt_out_permits_a_non_loopback_host(
    source_type: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-43: an operator who says so explicitly can point the client elsewhere.

    The flag travels with the local-vision policy but not bundle identity. On a
    loopback endpoint both values permit the same reader and produce the same
    output. On a remote endpoint false produces no bundle, so the flag cannot
    distinguish two bundle contents under ADR-0004.
    """
    config = local_vision_config_from_args(
        {"local_vision_base_url": PRIVATE_URL, "local_vision_allow_remote_endpoint": True},
        tmp_path,
    )

    assert config.base_url == PRIVATE_URL
    assert config.allow_remote_endpoint is True

    models_url = f"{PRIVATE_URL}/models"
    completions_url = f"{PRIVATE_URL}/chat/completions"
    transport = _FakeTransport(
        {
            models_url: _json_response(models_url, {"data": [{"id": config.model}]}),
            completions_url: _json_response(
                completions_url,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"visual_summary": "Remote endpoint reached"})
                            }
                        }
                    ]
                },
            ),
        }
    )
    monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))

    probe = probe_local_vision(config)
    assert probe.available is True

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    interpretation, failure = try_interpret_image(config, image, "Interpret.")
    assert interpretation is not None
    assert interpretation.visual_summary == "Remote endpoint reached"
    assert failure is None
    assert transport.requested == [models_url, completions_url]

    options = DistillOptions.from_args(
        {"local_vision_base_url": PRIVATE_URL, "local_vision_allow_remote_endpoint": True}
    )

    assert options.local_vision_base_url == PRIVATE_URL
    assert options.local_vision_config() == LocalVisionConfig(
        backend=options.local_vision_backend,
        model=options.local_vision_model,
        base_url=PRIVATE_URL,
        timeout_sec=options.local_vision_timeout_sec,
        caption_frames=options.caption_frames,
        allow_remote_endpoint=True,
    )
    # Same loopback endpoint on both sides, so only the policy flag differs.
    permitted = DistillOptions.from_args({"local_vision_allow_remote_endpoint": True})
    assert permitted.local_vision_base_url == DistillOptions.from_args({}).local_vision_base_url
    assert permitted.opts_hash(source_type) == DistillOptions.from_args({}).opts_hash(source_type)


def test_a_redirect_to_a_link_local_address_is_not_followed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RV-7: a loopback endpoint cannot hand the client an address off loopback.

    The server the config points at is loopback and passes every check; its
    answer is `302 Location: 169.254.169.254`, the cloud metadata address. A
    client that follows it has validated nothing, so the redirect is not
    followed: the request fails carrying the reason, and the link-local address
    is never asked for.
    """
    transport = _FakeTransport(
        {
            MODELS_URL: _redirect_response(MODELS_URL, LINK_LOCAL_URL),
            LINK_LOCAL_URL: _json_response(LINK_LOCAL_URL, {"data": [{"id": "leaked"}]}),
        }
    )
    monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))

    with pytest.raises(LocalVisionFailure) as caught:
        _urlopen_json("GET", MODELS_URL, None, 5.0)

    assert caught.value.code == "local_vision_endpoint_rejected"
    assert caught.value.detail["reason"] == "redirect"
    assert transport.requested == [MODELS_URL]


def test_redirects_are_disabled_rather_than_re_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-43: even a redirect *to loopback* is refused.

    The rule is that the client speaks to the configured endpoint and nowhere
    else. A redirect whose target would itself pass every address check is the
    case that tells the two designs apart: re-validating the target would
    follow this one, disabling redirects does not.
    """
    other_loopback = "http://127.0.0.1:9999/v1/models"
    transport = _FakeTransport(
        {
            MODELS_URL: _redirect_response(MODELS_URL, other_loopback),
            other_loopback: _json_response(other_loopback, {"data": [{"id": "elsewhere"}]}),
        }
    )
    monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))

    with pytest.raises(LocalVisionFailure) as caught:
        _urlopen_json("GET", MODELS_URL, None, 5.0)

    assert caught.value.detail["reason"] == "redirect"
    assert caught.value.detail["location"] == other_loopback
    assert transport.requested == [MODELS_URL]


def test_the_resolved_address_is_checked_on_every_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-029: a name that changes its answer between requests is caught.

    The same configured endpoint, resolved twice: 127.0.0.1 while the first
    request is made, then the cloud metadata address for the second. Validating
    the configured URL once would have admitted both, because the URL never
    changed - only what it resolves to did.
    """
    named_url = "http://vision.test:8000/v1/models"
    transport = _FakeTransport({named_url: _json_response(named_url, {"data": []})})
    monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))
    answers = [["127.0.0.1"], ["169.254.169.254"]]
    asked: list[tuple[str, int]] = []

    def rebinding_resolver(host: str, port: int) -> list[str]:
        asked.append((host, port))
        return answers.pop(0)

    assert _urlopen_json("GET", named_url, None, 5.0, resolver=rebinding_resolver) == {"data": []}

    with pytest.raises(LocalVisionFailure) as caught:
        _urlopen_json("GET", named_url, None, 5.0, resolver=rebinding_resolver)

    assert caught.value.detail["reason"] == "non_loopback_address"
    assert caught.value.detail["address"] == "169.254.169.254"
    assert asked == [("vision.test", 8000), ("vision.test", 8000)]
    assert transport.requested == [named_url]


def test_a_redirect_urllib_will_not_act_on_still_tells_the_operator_it_was_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The redirect that never reaches `_RedirectsAreRejected`.

    A 302 whose `Location` names a scheme urllib will not open - or carries no
    `Location` at all - is refused by urllib itself, before the handler that
    exists to refuse redirects is consulted. Nothing leaks: the request is not
    followed either way. What the operator got was `HTTP 302 from …: ` with an
    empty quote, because a 3xx has no body, and no clue that the endpoint had
    answered with a redirect at all.
    """
    for location in ({"Location": "file:///etc/passwd"}, {}):
        response = _CannedResponse(MODELS_URL, status=302, headers=location)
        transport = _FakeTransport({MODELS_URL: response})
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))

        with pytest.raises(RuntimeError) as caught:
            _urlopen_json("GET", MODELS_URL, None, 5.0)

        message = str(caught.value)
        assert "302" in message
        assert "redirect" in message
        # And it is not a rejection this module made, so it does not claim to
        # be one.
        assert not isinstance(caught.value, LocalVisionFailure)


def test_the_scheme_is_re_checked_on_the_request_not_only_on_the_config() -> None:
    """R-43: the per-request check is the static one too, not just the resolved one.

    The resolved tier cannot carry this on its own. `urllib`'s default handler
    chain - which `_build_opener` keeps, minus redirects and proxies - still
    holds the File, FTP and Data handlers, and a `file://` URL has an empty
    host that resolves to loopback, so a request validated only by address
    would open `/etc/passwd` and hand it back as the endpoint's answer.
    """
    with pytest.raises(LocalVisionFailure) as caught:
        _urlopen_json("GET", "file:///etc/passwd", None, 5.0)

    assert caught.value.detail["reason"] == "scheme"


def test_every_address_a_name_answers_with_is_checked_not_only_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-029: one answer, two addresses, and the second one is off loopback.

    A name is free to answer with a list, and which entry the connect picks is
    not this check's to know - so a name answering with both 127.0.0.1 and the
    cloud metadata address can steer the very next connection off loopback
    while passing any check that stops at the first entry. The rejection has to
    name the address that caused it, because "this name resolves somewhere it
    should not" is not an answer an operator can act on.
    """
    named_url = "http://vision.test:8000/v1/models"
    transport = _FakeTransport({named_url: _json_response(named_url, {"data": []})})
    monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))

    with pytest.raises(LocalVisionFailure) as caught:
        _urlopen_json(
            "GET",
            named_url,
            None,
            5.0,
            resolver=lambda host, port: ["127.0.0.1", "169.254.169.254"],
        )

    assert caught.value.detail["reason"] == "non_loopback_address"
    assert caught.value.detail["address"] == "169.254.169.254"
    # Refused before the request, not after it.
    assert transport.requested == []


def test_a_response_body_beyond_the_cap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-44: the read stops at 32 MiB, and what stopped it is not the header.

    The body advertises nothing and never ends, so a client trusting a
    `Content-Length` or reading to EOF never returns. The failure is the
    malformed-response one rather than a transport one: the response arrived,
    which is what M7.2 counts as evidence the transport works - it is unusable
    for the same reason a truncated body is.
    """
    endless = _CannedResponse(MODELS_URL, endless=True)
    monkeypatch.setattr(
        "distill.rapid_mlx._OPENER", _build_opener(_FakeTransport({MODELS_URL: endless}))
    )

    with pytest.raises(RuntimeError, match=str(MAX_RESPONSE_BYTES)):
        _urlopen_json("GET", MODELS_URL, None, 5.0)

    # Chunked since M2.5 (the absolute deadline is enforced between chunks):
    # the reads still stop at exactly one byte past the cap, never beyond.
    assert sum(endless.read_sizes) == MAX_RESPONSE_BYTES + 1
    assert all(size > 0 for size in endless.read_sizes)

    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    completions_url = f"{LOOPBACK_URL}/chat/completions"
    monkeypatch.setattr(
        "distill.rapid_mlx._OPENER",
        _build_opener(
            _FakeTransport({completions_url: _CannedResponse(completions_url, endless=True)})
        ),
    )

    result, failure = try_interpret_image(
        LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
    )

    assert result is None
    assert failure is not None
    assert failure["code"] == "local_vision_malformed_response"


def test_every_rejected_endpoint_is_logged_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """R-43's observability: each refusal says which rule refused, and about what.

    One event per rejection, at the boundary the rest of this module logs at,
    naming the reason - an operator whose run degraded to OCR-only can tell a
    scheme they typed wrong from a name that resolved somewhere unexpected
    without reproducing the run.
    """
    transport = _FakeTransport({MODELS_URL: _redirect_response(MODELS_URL, LINK_LOCAL_URL)})
    monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))

    with caplog.at_level(logging.DEBUG, logger="distill.local_vision"):
        with pytest.raises(DistillError):
            local_vision_config_from_args({"local_vision_base_url": PRIVATE_URL}, tmp_path)
        with pytest.raises(LocalVisionFailure):
            _urlopen_json("GET", MODELS_URL, None, 5.0)
        with pytest.raises(LocalVisionFailure):
            _urlopen_json(
                "GET",
                "http://vision.test:8000/v1/models",
                None,
                5.0,
                resolver=lambda host, port: ["169.254.169.254"],
            )

    events = [json.loads(record.message) for record in caplog.records]
    rejected = [event for event in events if event["event"] == "endpoint_rejected"]

    assert [event["detail"]["reason"] for event in rejected] == [
        "non_loopback_host",
        "redirect",
        "non_loopback_address",
    ]
    assert rejected[0]["detail"]["host"] == "10.0.0.5"
    assert rejected[1]["detail"]["location"] == LINK_LOCAL_URL
    assert rejected[2]["detail"]["address"] == "169.254.169.254"
    assert all(event["type"] == "distill.local_vision" for event in rejected)


def test_a_rejected_endpoint_degrades_the_probe_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0002: local vision is optional, so a refused endpoint warns and continues.

    The probe is the first request a run makes, and a name that resolves off
    loopback is refused there. That must reach the run the way every other
    unavailable vision target does - an unavailable **probe** carrying a
    **warning** with the reason - rather than as an exception escaping the
    capability that was allowed to be missing.
    """
    monkeypatch.setattr(
        "distill.rapid_mlx._resolve_addresses",
        lambda host, port: ["169.254.169.254"],
    )

    probe = probe_local_vision(LocalVisionConfig(base_url="http://vision.test:8000/v1"))

    assert probe.available is False
    assert probe.code == "local_vision_endpoint_rejected"
    assert probe.detail["reason"] == "non_loopback_address"
    assert probe.detail["address"] == "169.254.169.254"
    assert probe.warning()["code"] == "local_vision_endpoint_rejected"


def test_an_unusable_endpoint_is_refused_rather_than_raising_a_bare_value_error(
    tmp_path: Path,
) -> None:
    """A malformed endpoint is a rejection with a reason, not a stray `ValueError`.

    Both halves of the check parse attacker- or operator-supplied text: the
    URL from a config file, and whatever the resolver hands back. Neither may
    reach the caller as an exception from the parsing library, because that is
    the shape no caller in this module handles.
    """
    # A port out of range: `urlsplit` accepts the string and `parts.port`
    # raises when it is read.
    with pytest.raises(DistillError, match="not a usable URL"):
        local_vision_config_from_args(
            {"local_vision_base_url": "http://127.0.0.1:99999/v1"}, tmp_path
        )

    # A bracket that never closes: `urlsplit` itself raises, before there is a
    # `parts` to read anything off. Same typo class, same operator, and the
    # docstring above claims both - so both are asserted.
    with pytest.raises(DistillError, match="not a usable URL"):
        local_vision_config_from_args({"local_vision_base_url": "http://[::1:8000/v1"}, tmp_path)

    with pytest.raises(LocalVisionFailure) as caught:
        _urlopen_json(
            "GET",
            "http://vision.test:8000/v1/models",
            None,
            5.0,
            resolver=lambda host, port: ["not-an-address"],
        )

    assert caught.value.detail["reason"] == "unparsable_address"


def test_an_environment_proxy_cannot_take_the_request_off_the_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-43: `HTTP_PROXY` does not get to redirect what the address check allowed.

    urllib's default opener reads the proxy out of the environment, so a
    validated loopback endpoint would still be dialled through whatever
    `HTTP_PROXY` names - and the request carries the **keyframe** as base64.
    The URL never changes, which is why this asserts on the host dialled rather
    than on the one requested.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.9:3128")
    monkeypatch.setenv("http_proxy", "http://10.0.0.9:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    transport = _FakeTransport({MODELS_URL: _json_response(MODELS_URL, {"data": []})})
    monkeypatch.setattr("distill.rapid_mlx._OPENER", _build_opener(transport))

    assert _urlopen_json("GET", MODELS_URL, None, 5.0) == {"data": []}
    assert transport.dialled == ["127.0.0.1:8000"]


def test_an_error_body_is_read_under_a_bound_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-44 on the path that quotes the body instead of parsing it.

    A `404` whose body never ends is read to build the message a **warning**
    carries. Only a preview of it is ever used, so the read stops long before
    the response cap - the endless body proves it stopped at all. A `500` is
    retryable since M2.5 and reads no body at all: the strongest bound is the
    read that never happens.
    """
    endless = _CannedResponse(MODELS_URL, status=404, endless=True)
    monkeypatch.setattr(
        "distill.rapid_mlx._OPENER", _build_opener(_FakeTransport({MODELS_URL: endless}))
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        _urlopen_json("GET", MODELS_URL, None, 5.0)

    assert endless.read_sizes == [ERROR_BODY_PREVIEW_BYTES]

    retryable = _CannedResponse(MODELS_URL, status=500, endless=True)
    monkeypatch.setattr(
        "distill.rapid_mlx._OPENER", _build_opener(_FakeTransport({MODELS_URL: retryable}))
    )

    with pytest.raises(LocalVisionFailure, match="HTTP 500"):
        _urlopen_json("GET", MODELS_URL, None, 5.0)

    assert retryable.read_sizes == []
