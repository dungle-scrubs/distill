from __future__ import annotations

import io
import json
import logging
import os
import struct
import urllib.error
import zlib
from pathlib import Path
from typing import Any

import pytest
from untrusted_blocks import SENTINEL, assert_delimited, attack

from distill.artifacts import FrameArtifact, Interpretation
from distill.errors import DistillError
from distill.grounding import UNGROUNDED
from distill.local_vision import (
    BREAKER_WARNING_CODE,
    DEFAULT_LOCAL_VISION_BASE_URL,
    DEFAULT_LOCAL_VISION_MODEL,
    DEFAULT_TIMEOUT_SEC,
    FrameInterpreter,
    LocalVisionConfig,
    LocalVisionFailure,
    LocalVisionProbe,
    _interpret_with_rapid_mlx,
    config_dir,
    load_local_vision_config,
    local_vision_config_from_args,
    parse_interpretation_json,
    probe_local_vision,
    probe_rapid_mlx_availability,
    try_interpret_image,
)
from distill.options import DistillOptions
from distill.pipeline import local_vision_diagnostics

DEFAULT_MODEL = DEFAULT_LOCAL_VISION_MODEL


def _frame(index: int, image: Path, *, extracted_text: str = "") -> FrameArtifact:
    """One **frame artifact** as `frame_selection` would have produced it.

    The interpreter takes carriers now, so a test that fed it a mapping would be
    testing a shape nothing produces.
    """
    return FrameArtifact(
        index=index,
        timestamp_sec=float(index),
        path=str(image),
        relative_path=f"frames/{image.name}",
        extracted_text=extracted_text,
    )


def _summary(frame: FrameArtifact) -> str:
    reading = frame.reading
    assert reading is not None
    return reading.visual_summary


def _models_body(*model_ids: str) -> dict[str, Any]:
    return {"data": [{"id": mid} for mid in model_ids]}


def _chat_envelope(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _frame_json(**overrides: Any) -> str:
    payload = {
        "visual_summary": "A line chart",
        "detected_elements": ["axis", "trend line"],
        "interpretation": "Values rise over time.",
        "uncertainty": "Low",
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeRapidMlx:
    """Records requests and serves canned /v1/models and /v1/chat/completions."""

    def __init__(
        self,
        *,
        models: list[str] | None = None,
        chat_content: str | None = None,
        models_error: Exception | None = None,
        chat_error: Exception | None = None,
    ) -> None:
        self.models = models if models is not None else [DEFAULT_MODEL]
        self.chat_content = chat_content
        self.models_error = models_error
        self.chat_error = chat_error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, method: str, url: str, body: Any = None, timeout: float = 30.0) -> Any:
        self.calls.append({"method": method, "url": url, "body": body, "timeout": timeout})
        if method == "GET" and url.rstrip("/").endswith("/models"):
            if self.models_error is not None:
                raise self.models_error
            return _models_body(*self.models)
        if method == "POST" and url.rstrip("/").endswith("/chat/completions"):
            if self.chat_error is not None:
                raise self.chat_error
            return _chat_envelope(self.chat_content or "")
        raise RuntimeError(f"unexpected request {method} {url}")


def test_default_local_vision_config_targets_rapid_mlx(tmp_path: Path) -> None:
    config = load_local_vision_config(tmp_path)

    assert config.backend == "rapid-mlx"
    assert config.model == DEFAULT_MODEL
    assert config.base_url == DEFAULT_LOCAL_VISION_BASE_URL
    assert config.caption_frames is True


def test_local_vision_config_loads_from_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "distill.local-vision.json").write_text(
        json.dumps(
            {
                "backend": "rapid-mlx",
                "model": "qwen3-vl:32b",
                "base_url": "http://127.0.0.1:9000/v1",
                "timeout_sec": 12,
                "caption_frames": True,
            }
        )
    )
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))

    options = DistillOptions.from_args({})

    assert options.caption_frames is True
    assert options.local_vision_backend == "rapid-mlx"
    assert options.local_vision_model == "qwen3-vl:32b"
    assert options.local_vision_base_url == "http://127.0.0.1:9000/v1"
    assert options.local_vision_timeout_sec == 12.0

    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))

    options_with_generic_config_dir = DistillOptions.from_args({})

    assert options_with_generic_config_dir.local_vision_model == DEFAULT_MODEL
    assert options_with_generic_config_dir.local_vision_base_url == DEFAULT_LOCAL_VISION_BASE_URL


def test_nested_distill_config_is_supported(tmp_path: Path) -> None:
    (tmp_path / "distill.json").write_text(
        json.dumps({"local_vision": {"model": "qwen3-vl:30b-a3b", "caption_frames": True}})
    )

    config = load_local_vision_config(tmp_path)

    assert config.model == "qwen3-vl:30b-a3b"
    assert config.caption_frames is True


def test_per_call_local_vision_model_override(tmp_path: Path) -> None:
    (tmp_path / "distill.local-vision.json").write_text(
        json.dumps({"model": "qwen3-vl:32b", "caption_frames": True})
    )

    config = local_vision_config_from_args(
        {"local_vision_model": "qwen3-vl:8b", "caption_frames": "false"},
        tmp_path,
    )

    assert config.model == "qwen3-vl:8b"
    assert config.caption_frames is False


def test_probe_hits_models_endpoint_with_configured_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeRapidMlx(models=["other-model", DEFAULT_MODEL])
    config = LocalVisionConfig(
        base_url="http://127.0.0.1:49181/v1",
        model=DEFAULT_MODEL,
        timeout_sec=12,
    )

    probe = probe_rapid_mlx_availability(config, requestor=server)

    assert server.calls == [
        {"method": "GET", "url": "http://127.0.0.1:49181/v1/models", "body": None, "timeout": 12}
    ]
    assert probe.available is True
    assert probe.backend == "rapid-mlx"
    assert probe.model == DEFAULT_MODEL
    assert probe.base_url == "http://127.0.0.1:49181/v1"
    assert probe.code == "local_vision_available"
    assert probe.detail["served_models"] == ["other-model", DEFAULT_MODEL]


def test_probe_reports_unavailable_when_server_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LocalVisionConfig()

    probe = probe_rapid_mlx_availability(
        config,
        requestor=FakeRapidMlx(models_error=urllib.error.URLError("connection refused")),
    )

    assert probe.available is False
    assert probe.code == "local_vision_rapid_mlx_unavailable"
    assert "connection refused" in probe.detail["error"]


def test_probe_reports_model_unavailable_when_model_not_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Since M2.4 (D-008) an unlisted model alone does not hard-skip: the
    probe attempts a completion, and only that attempt failing too makes the
    model unavailable."""
    config = LocalVisionConfig(model="qwen3-vl:32b")

    probe = probe_rapid_mlx_availability(
        config,
        requestor=FakeRapidMlx(
            models=["other"], chat_error=RuntimeError("HTTP 404: model not found")
        ),
    )

    assert probe.available is False
    assert probe.code == "local_vision_model_unavailable"
    assert probe.detail["configured_model"] == "qwen3-vl:32b"
    assert probe.detail["served_models"] == ["other"]
    assert "404" in probe.detail["completion_error"]


def test_probe_reports_malformed_models_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Since the Phase 2 landing review, a bad /models answer no longer vetoes
    the endpoint (many providers serve no catalog): the catalog becomes
    advisory-absent and the attempted completion decides. Both failing is
    what reports unavailable."""
    probe = probe_rapid_mlx_availability(
        LocalVisionConfig(),
        requestor=FakeRapidMlx(
            models_error=ValueError("non-JSON response"),
            chat_error=RuntimeError("HTTP 404: no completions either"),
        ),
    )

    assert probe.available is False
    assert probe.code == "local_vision_model_unavailable"

    working = probe_rapid_mlx_availability(
        LocalVisionConfig(),
        requestor=FakeRapidMlx(models_error=ValueError("non-JSON response"), chat_content="pong"),
    )

    assert working.available is True
    assert working.detail["model_not_listed"] is True


def test_probe_reports_timeout() -> None:
    config = LocalVisionConfig()

    probe = probe_rapid_mlx_availability(
        config, requestor=FakeRapidMlx(models_error=TimeoutError("timed out"))
    )

    assert probe.available is False
    assert probe.code == "local_vision_timeout"


def test_unsupported_backend_is_rejected() -> None:
    probe = probe_local_vision(LocalVisionConfig(backend="ollama"))

    assert probe.available is False
    assert probe.code == "local_vision_backend_unsupported"


def test_rapid_mlx_backend_is_accepted_by_options() -> None:
    options = DistillOptions.from_args({"local_vision_backend": "rapid-mlx"})

    assert options.local_vision_backend == "rapid-mlx"


def test_non_rapid_mlx_backend_is_rejected_by_options() -> None:
    with pytest.raises(DistillError, match="must be 'rapid-mlx'"):
        DistillOptions.from_args({"local_vision_backend": "ollama"})


def test_config_dir_env_overrides_a_config_planted_under_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hermeticity: the autouse fixture's DISTILL_CONFIG_DIR beats the real HOME.

    The suite pins DISTILL_CONFIG_DIR at an empty throwaway directory precisely so
    a developer's ~/.distill config cannot steer a test run. This asserts that
    precedence explicitly - DEFAULT_MODEL alone would be true for the wrong reason.
    """
    home_config_dir = tmp_path / ".distill"
    home_config_dir.mkdir()
    (home_config_dir / "distill.local-vision.json").write_text(
        json.dumps({"model": "mlx-community/non-default-model"})
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    hermetic_config_dir = Path(os.environ["DISTILL_CONFIG_DIR"])

    assert hermetic_config_dir != home_config_dir
    assert config_dir() == hermetic_config_dir
    assert DistillOptions.from_args({}).local_vision_model == DEFAULT_MODEL


def test_config_dir_falls_back_to_home_when_env_var_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The HOME fallback the hermeticity fixture deliberately overrides still works.

    Without DISTILL_CONFIG_DIR, ``config_dir()`` resolves ``~/.distill``, so a
    config living there is read. This is the assertion that catches a regression
    in the fallback; the hermeticity test above cannot, because its env var wins.
    """
    home_config_dir = tmp_path / ".distill"
    home_config_dir.mkdir()
    configured_model = "mlx-community/non-default-model"
    assert configured_model != DEFAULT_MODEL
    (home_config_dir / "distill.local-vision.json").write_text(
        json.dumps({"model": configured_model})
    )
    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert config_dir() == home_config_dir
    assert DistillOptions.from_args({}).local_vision_model == configured_model


def test_local_vision_diagnostics_describe_rapid_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_probe(config: LocalVisionConfig) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_rapid_mlx_unavailable",
            message="missing",
            detail={},
        )

    monkeypatch.setattr("distill.pipeline.probe_local_vision", fake_probe)

    diagnostics = local_vision_diagnostics({})
    options = DistillOptions.from_args({})

    assert diagnostics["setup_command"] == f"rapid-mlx serve {DEFAULT_MODEL}"
    assert options.local_vision_model == DEFAULT_MODEL
    assert "127.0.0.1:8000" in diagnostics["rapid_mlx_note"]
    assert diagnostics["probe"]["backend"] == "rapid-mlx"
    assert "release_warning" not in diagnostics


@pytest.mark.parametrize(
    "supplied",
    [
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="-inf"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(True, id="true"),
        pytest.param(1e300, id="finite_but_past_what_a_socket_holds"),
    ],
)
def test_a_vision_timeout_no_socket_can_take_is_answered_by_the_default(
    supplied: object,
    tmp_path: Path,
) -> None:
    """A config *file* naming an unusable timeout gets the default, not a refusal.

    `_coerce_float` is the config layer's door, and its whole contract is to
    answer an unusable number with the default rather than refuse it - which is
    what a config file should do, since a file nobody is looking at should not
    stop a run. It admitted `inf` because `inf > 0`, so the number travelled all
    the way into `socket.settimeout` and came back as `OverflowError: timestamp
    out of range for platform time_t`.

    `inf` is not the only number a socket cannot take, which is why the bound
    here is the socket's own: a timeout is nanoseconds in a signed 64-bit
    integer, so `1e300` is finite, positive, and just as much an
    `OverflowError`. `nan` and `true` are the same door from the other side: a
    NaN timeout is a comparison that always answers no, and a config file saying
    `"timeout_sec": true` is not a one-second timeout.

    Driven through the file, deliberately. The *override* door - the flag an
    operator typed - refuses these instead of coercing them, because a command
    that reports what your arguments resolve to must not answer a rejected
    argument by printing the default as though it were in force. That half is
    `test_a_diagnostics_timeout_outside_the_domain_is_refused_like_a_run_path`,
    and the two doors are the recorded scoping rather than an inconsistency.
    """
    (tmp_path / "distill.local-vision.json").write_text(json.dumps({"timeout_sec": supplied}))

    config = load_local_vision_config(tmp_path)

    assert config.timeout_sec == DEFAULT_TIMEOUT_SEC


def test_interpret_image_returns_structured_result(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    server = FakeRapidMlx(
        chat_content=_frame_json(
            verbatim_text="Quarterly revenue",
            text_confidence="high",
            frame_kind="slide",
        )
    )
    config = LocalVisionConfig()

    result = _interpret_with_rapid_mlx(config, image, "Interpret.", "technical", requestor=server)

    request = server.calls[0]
    assert request["method"] == "POST"
    assert request["url"] == f"{config.base_url}/chat/completions"
    assert request["body"]["model"] == config.model
    assert request["body"]["temperature"] == 0
    assert request["body"]["response_format"] == {"type": "json_object"}
    # The image is sent as a base64 data URI in an image_url content part.
    content = request["body"]["messages"][0]["content"]
    assert any(
        part.get("type") == "image_url"
        and part["image_url"]["url"].startswith("data:image/png;base64,")
        for part in content
    )
    assert result.backend == "rapid-mlx"
    assert result.verbatim_text == "Quarterly revenue"
    assert result.text_confidence == "high"
    assert result.frame_kind == "slide"


def test_the_boundary_survives_from_the_frame_to_the_request_body(tmp_path: Path) -> None:
    """R-28 asserted where the model actually reads it, not where it is built.

    `vision_prompts` can delimit **extracted text** perfectly and the model
    still receive none of it: the interpreter has to pass the frame's text to
    the builder, and the request assembly has to carry the built prompt through
    rather than replace it. Both links are checked here against one hostile
    frame, because a boundary tested only at the builder is a boundary that
    would survive `local_vision` dropping it.
    """
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    payload = attack("frame")
    sent: dict[str, str] = {}

    def capture(
        config: LocalVisionConfig,
        image_path: Path,
        prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        sent["prompt"] = prompt
        return Interpretation(backend=config.backend, prompt_profile=prompt_profile), None

    interpreter = FrameInterpreter(
        LocalVisionConfig(), probe=_available_probe, try_interpret=capture
    )
    interpreter.interpret([_frame(1, image, extracted_text=payload)])

    server = FakeRapidMlx(chat_content=_frame_json(verbatim_text="", text_confidence="none"))
    _interpret_with_rapid_mlx(
        LocalVisionConfig(), image, sent["prompt"], "technical", requestor=server
    )
    text = next(
        part["text"]
        for part in server.calls[0]["body"]["messages"][0]["content"]
        if part.get("type") == "text"
    )

    assert_delimited(text, payload, SENTINEL)


def test_interpret_image_retries_past_one_malformed_response(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    attempts = {"n": 0}

    def requestor(*, method: str, url: str, body: Any = None, timeout: float = 30.0) -> Any:
        if not url.rstrip("/").endswith("/chat/completions"):
            return _models_body(DEFAULT_MODEL)
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _chat_envelope("not json at all")
        return _chat_envelope(_frame_json(verbatim_text="Hello", text_confidence="high"))

    result = _interpret_with_rapid_mlx(
        LocalVisionConfig(), image, "Interpret.", "technical", requestor=requestor
    )

    assert attempts["n"] == 2
    assert result.verbatim_text == "Hello"


def test_interpret_image_reports_unreachable_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    # The probe succeeds (server up, model served); generation then fails to
    # connect, which must surface as a transport-unavailable warning.
    monkeypatch.setattr(
        "distill.rapid_mlx._urlopen_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("connection refused")),
    )

    result, warning = try_interpret_image(
        LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_rapid_mlx_unavailable"


def test_interpret_image_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    monkeypatch.setattr(
        "distill.rapid_mlx._urlopen_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    result, warning = try_interpret_image(
        LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_timeout"


def test_interpret_image_malformed_response_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    # Probe succeeds; the chat completion returns a body whose content is not
    # valid JSON, so both retry attempts are exhausted -> malformed response.
    monkeypatch.setattr(
        "distill.rapid_mlx._urlopen_json",
        lambda method, url, body=None, timeout_sec=30.0, **_: (
            _models_body(DEFAULT_MODEL)
            if url.rstrip("/").endswith("/models")
            else _chat_envelope("definitely not json")
        ),
    )

    result, warning = try_interpret_image(
        LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_malformed_response"


def test_interpretation_parser_accepts_fenced_or_prefaced_json() -> None:
    payload = {
        "visual_summary": "A chart",
        "detected_elements": ["bar"],
        "interpretation": "Growth increases.",
        "uncertainty": "Low",
    }

    assert parse_interpretation_json(f"```json\n{json.dumps(payload)}\n```") == payload
    assert parse_interpretation_json(f"Here is the result:\n{json.dumps(payload)}") == payload


def test_an_empty_object_is_rejected_rather_than_counted_as_an_interpretation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-39: `{}` is a malformed response, not a reading of the keyframe.

    A server that is up and answers `{}` for every **keyframe** is otherwise
    indistinguishable from a working one: the parser hands back a dict, the
    interpret path builds an all-empty **interpretation** from it, and the run
    reports a reading it does not have.
    """
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(
        "distill.rapid_mlx._urlopen_json",
        lambda method, url, body=None, timeout_sec=30.0, **_: (
            _models_body(DEFAULT_MODEL)
            if url.rstrip("/").endswith("/models")
            else _chat_envelope("{}")
        ),
    )

    assert parse_interpretation_json("{}") is None
    assert parse_interpretation_json("Here is the result:\n{}") is None

    result, warning = try_interpret_image(
        LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_malformed_response"


def test_a_payload_needs_one_non_empty_substantive_field(tmp_path: Path) -> None:
    """R-39: metadata is not content, and whitespace is not content.

    Each field that carries what the model saw admits the payload on its own;
    a payload holding only the fields that describe a reading - its kind, its
    confidence, its hedge - describes nothing, and neither does one whose
    substantive fields are blank, whitespace, or an empty list.
    """
    for field_name, value in (
        ("visual_summary", "A line chart"),
        ("interpretation", "Values rise over time."),
        ("detected_elements", ["axis"]),
        ("verbatim_text", "Quarterly revenue"),
    ):
        assert parse_interpretation_json(json.dumps({field_name: value})) == {field_name: value}

    metadata_only = {
        "frame_kind": "slide",
        "text_confidence": "high",
        "uncertainty": "The axis labels are small.",
    }
    assert parse_interpretation_json(json.dumps(metadata_only)) is None

    blank = {
        "visual_summary": "   ",
        "interpretation": "",
        "detected_elements": ["", "  \t"],
        "verbatim_text": "\n",
        "frame_kind": "slide",
        "text_confidence": "high",
    }
    assert parse_interpretation_json(json.dumps(blank)) is None

    # A substantive field of the wrong type carries nothing either.
    assert parse_interpretation_json(json.dumps({"detected_elements": "axis"})) is None
    assert parse_interpretation_json(json.dumps({"visual_summary": {}})) is None


def test_the_response_summary_does_not_claim_an_empty_interpretation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-39: the run's own count of readings excludes the ones that are empty.

    Two **keyframes**, one answered with a reading and one answered `{}`. The
    interpreted count is what a caller reads to know how much of the run the
    vision pass actually covered, so it must say one.
    """
    for index in range(2):
        (tmp_path / f"frame{index}.png").write_bytes(b"png")

    def fake_urlopen(
        method: str,
        url: str,
        body: Any = None,
        timeout_sec: float = 30.0,
        **_: Any,
    ) -> Any:
        if url.rstrip("/").endswith("/models"):
            return _models_body(DEFAULT_MODEL)
        text = next(
            part["text"] for part in body["messages"][0]["content"] if part.get("type") == "text"
        )
        content = _frame_json(verbatim_text="frame0", text_confidence="high")
        return _chat_envelope(content if "frame0" in text else "{}")

    monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)

    interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe, debug=True)
    frames, warnings = interpreter.interpret(
        [
            _frame(index + 1, tmp_path / f"frame{index}.png", extracted_text=f"frame{index}")
            for index in range(2)
        ]
    )

    debug_info = interpreter.debug_info()
    assert debug_info["interpreted_count"] == 1
    assert debug_info["warning_counts"]["local_vision_malformed_response"] == 1
    assert frames[0].reading is not None
    assert frames[1].reading is None
    assert [w["code"] for w in warnings] == ["local_vision_malformed_response"]
    complete = next(
        event for event in debug_info["trace_events"] if event["event"] == "interpret.complete"
    )
    assert complete["detail"]["interpreted_frames"] == 1


def test_a_reading_that_says_nothing_is_neither_carried_nor_counted(tmp_path: Path) -> None:
    """R-39 one step past the parser, where a reading arrives already built.

    The transport path rejects an empty payload, but the interpreter counted a
    frame as interpreted whenever a reading object came back at all. An
    `Interpretation` holding nothing but its backend is the same non-answer with
    a dataclass around it, and a **frame artifact** that carried it would make
    `render` announce a visual interpretation with nothing under the heading.

    It is the *same* non-answer, so it takes the same route out: the
    malformed-response **warning**, the `UNGROUNDED` assessment saying the
    model produced no usable output, and a breaker that counts it as a
    delivered response. Dropping it silently left a frame whose reading was
    discarded looking, from every seam, exactly like a frame the vision pass
    read successfully and had nothing to remark on.
    """
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def empty_reading(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        return (
            Interpretation(
                backend=config.backend,
                model=config.model,
                prompt_profile=prompt_profile,
                frame_kind="slide",
                text_confidence="high",
                uncertainty="   ",
            ),
            None,
        )

    interpreter = FrameInterpreter(
        LocalVisionConfig(), probe=_available_probe, try_interpret=empty_reading
    )

    frames, warnings = interpreter.interpret([_frame(1, image, extracted_text="Hello")])

    assert frames[0].reading is None
    assert [w["code"] for w in warnings] == ["local_vision_malformed_response"]
    assert interpreter.debug_info()["interpreted_count"] == 0
    # The frame says why it has no reading, rather than looking like a frame
    # that had nothing to say.
    assert frames[0].grounding is not None
    assert frames[0].grounding["level"] == UNGROUNDED
    assert "no usable output" in frames[0].grounding["reason"]
    # A response arrived, so the breaker takes it as evidence the transport
    # works: it is not a consecutive transport failure and it clears the tally.
    assert interpreter.debug_info()["breaker"]["transport_failures"] == 0


def test_a_truncated_json_body_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-39: a body that stops mid-value is malformed at either layer.

    A server killed mid-write truncates the HTTP body; a model that runs out of
    output budget truncates the JSON it was asked for. Both must reach the
    malformed-response **warning** and neither may raise or be counted.
    """
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    truncated = '{"visual_summary": "A line ch'

    assert parse_interpretation_json(truncated) is None

    monkeypatch.setattr(
        "distill.rapid_mlx._urlopen_json",
        lambda method, url, body=None, timeout_sec=30.0, **_: (
            _models_body(DEFAULT_MODEL)
            if url.rstrip("/").endswith("/models")
            else _chat_envelope(truncated)
        ),
    )

    result, warning = try_interpret_image(
        LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
    )

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_malformed_response"

    # The same shape one layer down: the HTTP body itself ends mid-value, so no
    # envelope is ever parsed. The stub above has to come off first, or the real
    # decode this half is about never runs.
    monkeypatch.undo()
    monkeypatch.setattr(
        "distill.rapid_mlx._OPENER",
        _FakeOpener(b'{"choices": [{"message": {"content": "'),
    )

    body_result, body_warning = try_interpret_image(
        LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
    )

    assert body_result is None
    assert body_warning is not None
    assert body_warning["code"] == "local_vision_malformed_response"


class _FakeHttpResponse:
    """The minimum of an ``http.client.HTTPResponse`` that ``_urlopen_json`` uses."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk, self._offset = self._payload[self._offset :], len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    read1 = read


class _FakeOpener:
    """Stands in for the vision client's opener, serving one canned response.

    The opener and not ``urllib.request.urlopen``: the client opens through its
    own opener now, the one built without redirect following (R-43).
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def open(self, request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        return _FakeHttpResponse(self._payload)


def test_interpret_image_cancel_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    # Generation is the only network call on the interpret path (interpret_image
    # no longer probes); raising KeyboardInterrupt there exercises the cancel path.
    monkeypatch.setattr(
        "distill.rapid_mlx._urlopen_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    result, warning = try_interpret_image(LocalVisionConfig(), image, "Interpret.")

    assert result is None
    assert warning is not None
    assert warning["code"] == "local_vision_cancelled"


def _available_probe(config: LocalVisionConfig) -> LocalVisionProbe:
    return LocalVisionProbe(
        available=True,
        backend=config.backend,
        model="Qwen3-VL-8B-8bit",
        base_url=config.base_url,
        code="local_vision_available",
        message="available",
        detail={},
    )


def test_frame_interpreter_debug_info_tracks_run_state(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")

    def fake_try_interpret(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        return (
            Interpretation(
                visual_summary=f"summary for {image_path.name}",
                detected_elements=("caption",),
                interpretation="A readable caption.",
                uncertainty="Low",
                backend=config.backend,
                model=config.model,
                prompt_profile=prompt_profile,
                verbatim_text="Hello",
                text_confidence="high",
            ),
            None,
        )

    interpreter = FrameInterpreter(
        LocalVisionConfig(),
        probe=_available_probe,
        try_interpret=fake_try_interpret,
        debug=True,
    )

    frames, warnings = interpreter.interpret([_frame(1, image, extracted_text="Hello")])

    assert warnings == []
    reading = frames[0].reading
    assert reading is not None
    assert reading.visual_summary == "summary for frame.png"
    debug_info = interpreter.debug_info()
    assert debug_info["backend"] == "rapid-mlx"
    assert debug_info["last_probe"] == {
        "available": True,
        "code": "local_vision_available",
        "model": "Qwen3-VL-8B-8bit",
    }
    assert debug_info["frame_count"] == 1
    assert debug_info["interpreted_count"] == 1
    assert debug_info["max_parallel"] == 1
    assert debug_info["warning_counts"] == {}
    assert debug_info["trace_events"] == [
        {"event": "interpret.start", "detail": {"frames": 1, "backend": "rapid-mlx"}},
        {"event": "interpret.pool", "detail": {"frames": 1, "max_parallel": 1}},
        {
            "event": "frame.start",
            "detail": {"index": 1, "path": str(image), "has_extracted_text": True},
        },
        {
            "event": "frame.complete",
            "detail": {"index": 1, "interpreted": True, "warning_code": None},
        },
        {
            "event": "interpret.complete",
            "detail": {
                "frames": 1,
                "interpreted_frames": 1,
                "warnings": 0,
                "warning_occurrences": 0,
            },
        },
    ]


def test_frame_interpreter_caps_pool_at_configured_max_parallel(tmp_path: Path) -> None:
    import threading
    import time

    for index in range(4):
        (tmp_path / f"frame{index}.png").write_bytes(b"png")

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_try_interpret(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return (
            Interpretation(
                visual_summary=image_path.name,
                detected_elements=(),
                interpretation="ok",
                uncertainty="Low",
                backend=config.backend,
                model=config.model,
                prompt_profile=prompt_profile,
            ),
            None,
        )

    interpreter = FrameInterpreter(
        LocalVisionConfig(),
        probe=_available_probe,
        try_interpret=fake_try_interpret,
        max_parallel=2,
    )

    frames = [
        # Image text grounds the interpretation so no extraneous warnings appear.
        _frame(index + 1, tmp_path / f"frame{index}.png", extracted_text=f"frame{index}")
        for index in range(4)
    ]
    interpreted, _ = interpreter.interpret(frames)

    assert len(interpreted) == 4
    # Concurrency never exceeds the configured cap of 2.
    assert peak <= 2
    assert peak >= 2  # and the pool did run in parallel
    # Output order is preserved despite parallel execution.
    assert [_summary(frame) for frame in interpreted] == [
        "frame0.png",
        "frame1.png",
        "frame2.png",
        "frame3.png",
    ]
    assert interpreter.debug_info()["interpreted_count"] == 4


def test_frame_interpreter_runs_serially_when_max_parallel_is_one(tmp_path: Path) -> None:
    import threading
    import time

    for index in range(3):
        (tmp_path / f"frame{index}.png").write_bytes(b"png")

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_try_interpret(
        config: LocalVisionConfig,
        image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str = "technical",
    ) -> tuple[Interpretation, None]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return (
            Interpretation(
                visual_summary=image_path.name,
                detected_elements=(),
                interpretation="ok",
                uncertainty="Low",
                backend=config.backend,
                model=config.model,
                prompt_profile=prompt_profile,
            ),
            None,
        )

    interpreter = FrameInterpreter(
        LocalVisionConfig(),
        probe=_available_probe,
        try_interpret=fake_try_interpret,
        max_parallel=1,
    )

    frames = [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(3)]
    interpreted, _ = interpreter.interpret(frames)

    # max_parallel<=1 is the serial fallback (A-004): never overlaps.
    assert peak == 1
    assert len(interpreted) == 3


def test_frame_interpreter_records_unavailable_probe_warning() -> None:
    def fake_probe(config: LocalVisionConfig) -> LocalVisionProbe:
        return LocalVisionProbe(
            available=False,
            backend=config.backend,
            model=config.model,
            base_url=config.base_url,
            code="local_vision_rapid_mlx_unavailable",
            message="missing",
            detail={},
        )

    interpreter = FrameInterpreter(
        LocalVisionConfig(),
        probe=fake_probe,
        debug=True,
    )

    unreached = _frame(1, Path("/tmp/frame.png"))
    frames, warnings = interpreter.interpret([unreached])

    assert frames == [unreached]
    # Classified *defect* against R-41: the record now carries its occurrence
    # count, and the probe's warning is built by the one builder every other
    # warning goes through rather than assembled as a bare mapping beside it.
    assert warnings == [
        {
            "stage": "local_vision",
            "code": "local_vision_rapid_mlx_unavailable",
            "message": "missing",
            "occurrences": 1,
        }
    ]
    assert interpreter.debug_info()["warning_counts"] == {"local_vision_rapid_mlx_unavailable": 1}


def test_frame_interpreter_asserts_frame_path_invariant() -> None:
    interpreter = FrameInterpreter(
        LocalVisionConfig(),
        probe=_available_probe,
    )

    pathless = FrameArtifact(index=1, timestamp_sec=0.0, path="", relative_path="")

    with pytest.raises(AssertionError, match="missing required 'path'"):
        interpreter.interpret([pathless])


@pytest.mark.skipif(
    os.environ.get("DISTILL_RUN_RAPID_MLX_SMOKE") != "1",
    reason="set DISTILL_RUN_RAPID_MLX_SMOKE=1 after starting `rapid-mlx serve <model>`",
)
def test_rapid_mlx_qwen3_vl_image_smoke(tmp_path: Path) -> None:
    image = tmp_path / "pixel.png"
    image.write_bytes(_solid_png_bytes())

    result, warning = try_interpret_image(LocalVisionConfig(), image, "Describe this image.")

    assert warning is None
    assert result is not None


def _solid_png_bytes() -> bytes:
    width = height = 32
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class TestSecretCredential:
    """The non-serializable carrier (D-007): reveal() is the only door to the value."""

    def test_carrier_redacts_every_text_form_and_refuses_serialization(self) -> None:
        import copy
        import pickle

        from distill.local_vision import SecretCredential

        secret = SecretCredential("sk-super-secret-value")

        assert secret.reveal() == "sk-super-secret-value"
        assert "sk-super-secret-value" not in repr(secret)
        assert "sk-super-secret-value" not in str(secret)
        assert "sk-super-secret-value" not in f"{secret}"
        with pytest.raises(TypeError):
            json.dumps(secret)
        with pytest.raises(TypeError):
            pickle.dumps(secret)
        with pytest.raises(TypeError):
            copy.deepcopy(secret)

    def test_config_never_exposes_the_credential_in_any_serialized_form(self) -> None:
        from dataclasses import asdict, replace

        from distill.local_vision import SecretCredential

        config = replace(LocalVisionConfig(), credential=SecretCredential("sk-super-secret-value"))

        assert "credential" not in config.public_dict()
        assert "sk-super-secret-value" not in json.dumps(config.public_dict())
        assert "sk-super-secret-value" not in repr(config)
        # asdict deep-copies field values and the carrier refuses deepcopy, so
        # generic dataclass serialization fails loudly instead of leaking.
        with pytest.raises(TypeError):
            asdict(config)

    @pytest.mark.parametrize(
        "url",
        [
            "http://user:sk-super-secret-value@127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/v1?api_key=sk-super-secret-value",
        ],
    )
    def test_credential_bearing_base_url_is_rejected_without_echoing_it(self, url: str) -> None:
        from distill.local_vision import _checked_endpoint_url
        from distill.rapid_mlx import EndpointRejected

        with pytest.raises(EndpointRejected) as excinfo:
            _checked_endpoint_url(url, allow_remote_endpoint=True)

        assert "sk-super-secret-value" not in str(excinfo.value)
        assert "sk-super-secret-value" not in json.dumps(excinfo.value.detail)

    def test_loader_rejection_of_credential_bearing_url_does_not_echo_it(self) -> None:
        with pytest.raises(DistillError) as excinfo:
            local_vision_config_from_args(
                {"local_vision_base_url": "http://user:sk-super-secret-value@127.0.0.1:8000/v1"}
            )

        assert "sk-super-secret-value" not in str(excinfo.value)
        assert "sk-super-secret-value" not in json.dumps(getattr(excinfo.value, "details", {}))

    @pytest.mark.parametrize(
        "url",
        [
            # Port out of range: rejected as unparsable BEFORE the credential
            # screen can run, so the echo path itself must redact.
            "http://user:sk-super-secret-value@127.0.0.1:99999/v1",
            # Percent-encoded userinfo parses as a hostname, evading the
            # userinfo check; the host echo must not leak it either.
            "http://user%3Ask-super-secret-value%40127.0.0.1:8000/v1",
        ],
    )
    def test_malformed_credential_bearing_urls_never_echo_the_secret(self, url: str) -> None:
        from distill.local_vision import _checked_endpoint_url
        from distill.rapid_mlx import EndpointRejected

        with pytest.raises(EndpointRejected) as excinfo:
            _checked_endpoint_url(url, allow_remote_endpoint=True)

        assert "sk-super-secret-value" not in str(excinfo.value)
        assert "sk-super-secret-value" not in json.dumps(excinfo.value.detail)

    def test_configs_differing_only_by_credential_are_not_equal(self) -> None:
        from dataclasses import replace

        from distill.local_vision import SecretCredential

        base = LocalVisionConfig()
        with_a = replace(base, credential=SecretCredential("sk-credential-a"))
        with_b = replace(base, credential=SecretCredential("sk-credential-b"))
        with_a_again = replace(base, credential=SecretCredential("sk-credential-a"))

        # A future config-keyed cache must never serve a response fetched
        # under a different credential.
        assert with_a != with_b
        assert with_a != base
        assert with_a == with_a_again


class TestCredentialResolution:
    """D-016: api_key_env preferred over inline api_key, both under local_vision."""

    def test_inline_api_key_resolves_to_a_carried_credential(self, tmp_path: Path) -> None:
        (tmp_path / "distill.local-vision.json").write_text(
            json.dumps({"api_key": "sk-inline-credential"})
        )

        config = load_local_vision_config(tmp_path)

        assert config.credential is not None
        assert config.credential.reveal() == "sk-inline-credential"

    def test_api_key_env_wins_over_inline_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "distill.local-vision.json").write_text(
            json.dumps({"api_key": "sk-inline-loses", "api_key_env": "DISTILL_TEST_VISION_KEY"})
        )
        monkeypatch.setenv("DISTILL_TEST_VISION_KEY", "sk-from-env-wins")

        config = load_local_vision_config(tmp_path)

        assert config.credential is not None
        assert config.credential.reveal() == "sk-from-env-wins"

    def test_credential_survives_the_options_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "distill.local-vision.json").write_text(
            json.dumps({"api_key": "sk-round-trip"})
        )
        monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))

        options = DistillOptions.from_args({})
        config = options.local_vision_config()

        assert config.credential is not None
        assert config.credential.reveal() == "sk-round-trip"
        # The identity/cache surfaces stay credential-free and serializable.
        assert "sk-round-trip" not in json.dumps(options.public_dict("local"))
        assert "sk-round-trip" not in json.dumps(options.cache_payload("local"))

    def test_bearer_header_present_exactly_when_a_credential_is(self) -> None:
        from distill.local_vision import SecretCredential
        from distill.rapid_mlx import _request_headers

        with_credential = _request_headers(SecretCredential("sk-header-secret"))
        without_credential = _request_headers(None)

        assert with_credential["Authorization"] == "Bearer sk-header-secret"
        assert "Authorization" not in without_credential
        assert without_credential["Content-Type"] == "application/json"
        assert without_credential["Accept"] == "application/json"

    def test_interpret_request_carries_bearer_exactly_when_config_does(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from dataclasses import replace

        from distill.local_vision import SecretCredential

        class _RecordingOpener(_FakeOpener):
            def __init__(self, payload: bytes) -> None:
                super().__init__(payload)
                self.requests: list[Any] = []

            def open(self, request: Any, timeout: float | None = None) -> _FakeHttpResponse:
                self.requests.append(request)
                return super().open(request, timeout)

        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        payload = json.dumps(_chat_envelope(_frame_json())).encode()
        recorder = _RecordingOpener(payload)
        monkeypatch.setattr("distill.rapid_mlx._OPENER", recorder)

        with_credential = replace(
            LocalVisionConfig(), credential=SecretCredential("sk-wire-secret")
        )
        try_interpret_image(with_credential, image, "Interpret.", prompt_profile="technical")
        try_interpret_image(LocalVisionConfig(), image, "Interpret.", prompt_profile="technical")

        first, second = recorder.requests
        assert first.get_header("Authorization") == "Bearer sk-wire-secret"
        assert second.get_header("Authorization") is None

    def test_configured_but_empty_credential_on_remote_endpoint_is_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "distill.local-vision.json").write_text(
            json.dumps(
                {
                    "api_key_env": "DISTILL_TEST_UNSET_KEY",
                    "base_url": "https://vision.example.com/v1",
                    "allow_remote_endpoint": True,
                }
            )
        )
        monkeypatch.delenv("DISTILL_TEST_UNSET_KEY", raising=False)

        with pytest.raises(DistillError) as excinfo:
            load_local_vision_config(tmp_path)

        assert "DISTILL_TEST_UNSET_KEY" in str(excinfo.value)

    def test_no_credential_configured_on_remote_endpoint_is_intentional_no_auth(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "distill.local-vision.json").write_text(
            json.dumps({"base_url": "https://vision.example.com/v1", "allow_remote_endpoint": True})
        )

        config = load_local_vision_config(tmp_path)

        assert config.credential is None
        assert config.credential_configured is False

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_rejection_degrades_without_echoing_body_or_credential(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        status: int,
    ) -> None:
        from dataclasses import replace
        from email.message import Message

        from distill.local_vision import SecretCredential

        class _AuthRejectingOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                raise urllib.error.HTTPError(
                    request.full_url,
                    status,
                    "denied",
                    Message(),
                    io.BytesIO(b'{"error": "server-error-body-text"}'),
                )

        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _AuthRejectingOpener())
        config = replace(LocalVisionConfig(), credential=SecretCredential("sk-auth-secret"))

        with caplog.at_level(logging.DEBUG, logger="distill.local_vision"):
            result, failure_warning = try_interpret_image(
                config, image, "Interpret.", prompt_profile="technical"
            )

        assert result is None
        assert failure_warning is not None
        assert failure_warning["code"] == "local_vision_auth_rejected"
        surface = json.dumps(failure_warning)
        assert "sk-auth-secret" not in surface
        assert "server-error-body-text" not in surface
        assert any('"event": "auth_rejected"' in record.message for record in caplog.records)

    def test_one_auth_rejection_degrades_the_remaining_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for index in range(3):
            (tmp_path / f"frame{index}.png").write_bytes(b"png")
        attempts: list[str] = []

        def fake_urlopen(
            method: str, url: str, body: Any = None, timeout_sec: float = 30.0, **_: Any
        ) -> Any:
            attempts.append(url)
            raise LocalVisionFailure(
                "local_vision_auth_rejected",
                "Local vision endpoint rejected the credential (HTTP 401); "
                "continuing with OCR-only output.",
                {"status": 401},
            )

        monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)
        monkeypatch.setattr("distill.local_vision._urlopen_json", fake_urlopen, raising=False)

        interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe, debug=True)
        frames, warnings = interpreter.interpret(
            [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(3)]
        )

        # One attempt, not three: the first 401 degrades the remainder.
        assert len(attempts) == 1
        assert all(frame.reading is None for frame in frames)
        codes = {w["code"] for w in warnings}
        assert "local_vision_auth_rejected" in codes

    def test_auth_rejected_frame_carries_the_ungrounded_assessment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "frame0.png").write_bytes(b"png")

        def fake_urlopen(*_args: Any, **_kwargs: Any) -> Any:
            raise LocalVisionFailure(
                "local_vision_auth_rejected",
                "Local vision endpoint rejected the credential (HTTP 401); "
                "continuing with OCR-only output.",
                {"status": 401},
            )

        monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)

        interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe)
        frames, _warnings = interpreter.interpret([_frame(1, tmp_path / "frame0.png")])

        # FRAME_READ_FAILURE_CODES membership is what attaches the explicit
        # "vision produced no usable output" grounding to the failed frame.
        assert frames[0].reading is None
        assert frames[0].grounding is not None
        assert frames[0].grounding.get("level") == UNGROUNDED

    def test_probe_request_carries_the_bearer_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        from distill.local_vision import SecretCredential

        captured: dict[str, Any] = {}

        def fake_get(requestor, url, timeout_sec, **kwargs):  # noqa: ANN001
            captured.update(kwargs)
            return _models_body(DEFAULT_MODEL)

        monkeypatch.setattr("distill.local_vision._http_get_json", fake_get)
        config = replace(LocalVisionConfig(), credential=SecretCredential("sk-probe-secret"))

        probe = probe_rapid_mlx_availability(config)

        assert probe.available is True
        assert captured["credential"] is config.credential

    def test_probe_auth_rejection_yields_unavailable_probe_with_auth_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from email.message import Message

        class _AuthRejectingOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                raise urllib.error.HTTPError(
                    request.full_url, 401, "denied", Message(), io.BytesIO(b"{}")
                )

        monkeypatch.setattr("distill.rapid_mlx._OPENER", _AuthRejectingOpener())

        probe = probe_rapid_mlx_availability(LocalVisionConfig())

        assert probe.available is False
        assert probe.code == "local_vision_auth_rejected"

    def test_empty_inline_api_key_is_the_same_typo_guard(self, tmp_path: Path) -> None:
        (tmp_path / "distill.local-vision.json").write_text(
            json.dumps(
                {
                    "api_key": "",
                    "base_url": "https://vision.example.com/v1",
                    "allow_remote_endpoint": True,
                }
            )
        )

        with pytest.raises(DistillError):
            load_local_vision_config(tmp_path)

    def test_configured_but_empty_credential_on_loopback_is_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "distill.local-vision.json").write_text(
            json.dumps({"api_key": "", "base_url": "http://localhost:8000/v1"})
        )

        config = load_local_vision_config(tmp_path)

        # allow_remote_endpoint is False, so the per-request resolved-address
        # check owns locality; the typo guard must not punish the local default.
        assert config.credential is None
        assert config.credential_configured is True


class TestEndpointPolicyHttps:
    """M2.3: http only ever speaks to loopback; a remote endpoint requires https."""

    def test_http_to_a_non_loopback_literal_is_rejected_even_with_remote_allowed(
        self,
    ) -> None:
        from distill.local_vision import _checked_endpoint_url
        from distill.rapid_mlx import EndpointRejected

        with pytest.raises(EndpointRejected) as excinfo:
            _checked_endpoint_url("http://203.0.113.7:8000/v1", allow_remote_endpoint=True)

        assert excinfo.value.reason == "http_off_loopback"
        assert "https" in excinfo.value.message

    def test_http_name_resolving_off_machine_is_rejected_even_with_remote_allowed(
        self,
    ) -> None:
        from distill.local_vision import _check_resolved_address
        from distill.rapid_mlx import EndpointRejected

        with pytest.raises(EndpointRejected) as excinfo:
            _check_resolved_address(
                "plain.example.com",
                8000,
                allow_remote_endpoint=True,
                scheme="http",
                resolver=lambda _host, _port: ["203.0.113.7"],
            )

        assert excinfo.value.reason == "http_off_loopback"

    def test_https_name_resolving_off_machine_is_permitted_with_remote_allowed(
        self,
    ) -> None:
        from distill.local_vision import _check_resolved_address

        addresses = _check_resolved_address(
            "vision.example.com",
            443,
            allow_remote_endpoint=True,
            scheme="https",
            resolver=lambda _host, _port: ["203.0.113.7"],
        )

        assert addresses == []

    def test_one_endpoint_rejection_degrades_the_remaining_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from distill.rapid_mlx import EndpointRejected

        for index in range(3):
            (tmp_path / f"frame{index}.png").write_bytes(b"png")
        attempts: list[str] = []

        def fake_urlopen(
            method: str, url: str, body: Any = None, timeout_sec: float = 30.0, **_: Any
        ) -> Any:
            attempts.append(url)
            # A rebind is the rejection that actually reaches a frame: the
            # config validated, the probe passed, and the name changed its
            # answer mid-run (D-029).
            raise EndpointRejected(
                "non_loopback_address",
                "Local vision endpoint host resolves to 203.0.113.7, which is "
                "not loopback; set local_vision_allow_remote_endpoint to reach "
                "one deliberately.",
            )

        monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)

        interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe)
        frames, warnings = interpreter.interpret(
            [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(3)]
        )

        # The same config produces the same rejection on every frame; one is
        # enough to condemn the rest.
        assert len(attempts) == 1
        assert all(frame.reading is None for frame in frames)
        summary = next(w for w in warnings if w["code"] == BREAKER_WARNING_CODE)
        assert "the endpoint was rejected" in summary["message"]
        assert "consecutive transport failures" not in summary["message"]

    def test_a_transient_dns_failure_gets_transport_strikes_not_condemnation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unresolvable name is unavailability, not a policy verdict: it
        must carry the transport-failure code (3-strike breaker), never the
        rejection code that condemns the run on the first flap."""
        import distill.rapid_mlx as rapid_mlx_module
        from distill.rapid_mlx import EndpointRejected, _resolve_addresses

        def failing_getaddrinfo(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("temporary failure in name resolution")

        monkeypatch.setattr(rapid_mlx_module.socket, "getaddrinfo", failing_getaddrinfo)

        with pytest.raises(LocalVisionFailure) as excinfo:
            _resolve_addresses("vision.test", 8000)

        assert excinfo.value.code == "local_vision_rapid_mlx_unavailable"
        assert not isinstance(excinfo.value, EndpointRejected)

    def test_the_production_opener_verifies_tls(self) -> None:
        """M2.3's premise is TLS for anything remote; pin that the opener's
        HTTPS handler actually verifies certificates and hostnames, so a
        future _build_opener change cannot silently turn verification off."""
        import ssl
        import urllib.request as _ur

        from distill.rapid_mlx import _build_opener

        opener = _build_opener()
        handlers = getattr(opener, "handlers", [])
        https_handlers = [h for h in handlers if isinstance(h, _ur.HTTPSHandler)]

        assert https_handlers, "the opener must have an HTTPS handler"
        for handler in https_handlers:
            context = handler._context  # noqa: SLF001 - the pin is the point
            assert context.verify_mode is ssl.CERT_REQUIRED
            assert context.check_hostname is True


class TestAttemptCompletionAvailability:
    """M2.4: /models is advisory; a missing listing warns, a completion decides."""

    def test_model_absent_from_catalog_is_available_when_a_completion_succeeds(
        self,
    ) -> None:
        server = FakeRapidMlx(models=["other-model"], chat_content="pong")
        config = LocalVisionConfig(model=DEFAULT_MODEL)

        probe = probe_rapid_mlx_availability(config, requestor=server)

        assert probe.available is True
        assert probe.code == "local_vision_available"
        assert probe.detail["model_not_listed"] is True
        assert probe.detail["served_models"] == ["other-model"]
        # The decision came from an attempted completion, not /models alone.
        assert any(
            call["method"] == "POST" and call["url"].endswith("/chat/completions")
            for call in server.calls
        )

    def test_interpreter_surfaces_an_advisory_warning_for_an_unlisted_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "frame0.png").write_bytes(b"png")

        def fake_urlopen(
            method: str, url: str, body: Any = None, timeout_sec: float = 30.0, **_: Any
        ) -> Any:
            if url.rstrip("/").endswith("/models"):
                return _models_body("some-other-model")
            return _chat_envelope(_frame_json())

        monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)

        interpreter = FrameInterpreter(LocalVisionConfig())
        frames, warnings = interpreter.interpret([_frame(1, tmp_path / "frame0.png")])

        # The frame is read (no hard-skip), and the run says why that is safe.
        assert frames[0].reading is not None
        advisory = [w for w in warnings if w["code"] == "local_vision_model_not_listed"]
        assert len(advisory) == 1
        assert "catalog" in advisory[0]["message"]

    def test_a_provider_rejecting_the_image_part_degrades_that_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Off-profile (D-008): an endpoint that 400s the image content part
        is out of profile for that frame - the frame degrades to OCR-only, the
        run continues, and the hostile error body is bounded, not parsed."""
        from email.message import Message

        class _ImageRejectingOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                raise urllib.error.HTTPError(
                    request.full_url,
                    400,
                    "bad request",
                    Message(),
                    io.BytesIO(b'{"error": "image content part not supported"}'),
                )

        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _ImageRejectingOpener())

        result, failure_warning = try_interpret_image(
            LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
        )

        assert result is None
        assert failure_warning is not None
        assert failure_warning["code"] == "local_vision_malformed_response"

    def test_the_attempted_completion_is_minimal_and_carries_an_image_part(
        self,
    ) -> None:
        server = FakeRapidMlx(models=["other-model"], chat_content="pong")

        probe_rapid_mlx_availability(LocalVisionConfig(), requestor=server)

        [attempt] = [c for c in server.calls if c["method"] == "POST"]
        body = attempt["body"]
        assert body["max_tokens"] == 1
        assert body["stream"] is False
        [message] = body["messages"]
        part_types = [part["type"] for part in message["content"]]
        # The probe's question is "will this endpoint read a keyframe" - a
        # text-only ping would authorize an image run against a text router.
        assert "image_url" in part_types

    def test_an_auth_rejection_during_the_attempt_keeps_its_own_code(self) -> None:
        server = FakeRapidMlx(
            models=["other-model"],
            chat_error=LocalVisionFailure(
                "local_vision_auth_rejected",
                "Local vision endpoint rejected the credential (HTTP 401); "
                "continuing with OCR-only output.",
                {"status": 401},
            ),
        )

        probe = probe_rapid_mlx_availability(LocalVisionConfig(), requestor=server)

        assert probe.available is False
        # The operator's signal is the credential, not a missing model.
        assert probe.code == "local_vision_auth_rejected"
        assert "credential" in probe.message

    def test_a_200_without_a_completion_envelope_is_not_availability(self) -> None:
        class _ErrorBodyServer(FakeRapidMlx):
            def __call__(
                self, *, method: str, url: str, body: Any = None, timeout: float = 30.0
            ) -> Any:
                if method == "POST":
                    self.calls.append(
                        {"method": method, "url": url, "body": body, "timeout": timeout}
                    )
                    return {"error": {"message": "model not found"}}
                return super().__call__(method=method, url=url, body=body, timeout=timeout)

        server = _ErrorBodyServer(models=["other-model"])

        probe = probe_rapid_mlx_availability(LocalVisionConfig(), requestor=server)

        assert probe.available is False
        assert probe.code == "local_vision_model_unavailable"


class TestBoundedRemoteBehavior:
    """M2.5: stream off, absolute deadline, run-wide budget, bounded retries."""

    def test_the_interpret_request_sets_stream_false(self) -> None:
        server = FakeRapidMlx(chat_content=_frame_json())

        result = _interpret_with_rapid_mlx(
            LocalVisionConfig(),
            Path(__file__),
            "Interpret.",
            "technical",
            requestor=server,
        )

        assert result is not None
        [attempt] = [c for c in server.calls if c["method"] == "POST"]
        assert attempt["body"]["stream"] is False

    def test_a_slow_drip_trips_the_absolute_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """urllib's timeout is per socket operation; a body arriving one chunk
        at a time never trips it. The deadline is absolute: the whole answer
        arrives within timeout_sec or the request is a timeout."""
        from distill.rapid_mlx import _urlopen_json

        clock = {"now": 0.0}
        monkeypatch.setattr("distill.rapid_mlx._monotonic", lambda: clock["now"])

        class _DrippingResponse:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

            def read(self, size: int = -1) -> bytes:
                # Each chunk costs 10 simulated seconds and never finishes.
                clock["now"] += 10.0
                return b"x" * min(size if size > 0 else 65536, 65536)

            read1 = read

        class _DrippingOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                return _DrippingResponse()

        monkeypatch.setattr("distill.rapid_mlx._OPENER", _DrippingOpener())

        with pytest.raises(TimeoutError):
            _urlopen_json(
                "POST",
                "http://127.0.0.1:8000/v1/chat/completions",
                {"model": "m"},
                timeout_sec=30.0,
            )

    def test_the_byte_budget_degrades_the_remainder_and_keeps_prior_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for index in range(3):
            (tmp_path / f"frame{index}.png").write_bytes(b"png")
        payload = json.dumps(_chat_envelope(_frame_json())).encode()
        opened = {"n": 0}

        class _CountingOpener(_FakeOpener):
            def open(self, request: Any, timeout: float | None = None) -> Any:
                opened["n"] += 1
                return super().open(request, timeout)

        monkeypatch.setattr("distill.rapid_mlx._OPENER", _CountingOpener(payload))

        interpreter = FrameInterpreter(
            # The budget bounds remote cost, so it attaches only to a config
            # that opted into a remote endpoint (loopback + opt-in is legal).
            LocalVisionConfig(allow_remote_endpoint=True),
            probe=_available_probe,
            # Enough for one response, not two: the second request must find
            # the budget spent and the third must never be attempted.
            budget_bytes=len(payload) + 10,
        )
        frames, warnings = interpreter.interpret(
            [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(3)]
        )

        assert frames[0].reading is not None  # already-interpreted frames kept
        assert frames[1].reading is None
        assert frames[2].reading is None
        assert opened["n"] <= 2  # the third frame was never attempted
        codes = {w["code"] for w in warnings}
        assert "local_vision_budget_exhausted" in codes
        summary = next(w for w in warnings if w["code"] == BREAKER_WARNING_CODE)
        # The R-40 sentence is the operator's whole account: a spent budget
        # must never read as "0 consecutive transport failures".
        assert "budget was spent" in summary["message"]
        assert "consecutive transport failures" not in summary["message"]

    def test_the_wall_clock_budget_degrades_the_remainder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for index in range(3):
            (tmp_path / f"frame{index}.png").write_bytes(b"png")
        clock = {"now": 0.0}
        monkeypatch.setattr("distill.rapid_mlx._monotonic", lambda: clock["now"])
        payload = json.dumps(_chat_envelope(_frame_json())).encode()

        class _SlowFrameOpener(_FakeOpener):
            def open(self, request: Any, timeout: float | None = None) -> Any:
                clock["now"] += 100.0  # each frame costs 100 simulated seconds
                return super().open(request, timeout)

        monkeypatch.setattr("distill.rapid_mlx._OPENER", _SlowFrameOpener(payload))

        interpreter = FrameInterpreter(
            LocalVisionConfig(timeout_sec=1000.0, allow_remote_endpoint=True),
            probe=_available_probe,
            budget_sec=150.0,
        )
        frames, warnings = interpreter.interpret(
            [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(3)]
        )

        assert frames[0].reading is not None
        assert frames[1].reading is None
        assert frames[2].reading is None
        assert "local_vision_budget_exhausted" in {w["code"] for w in warnings}

    def test_a_429_honors_retry_after_with_bounded_retries_then_degrades(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from email.message import Message

        opened = {"n": 0}
        sleeps: list[float] = []

        class _RateLimitingOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                opened["n"] += 1
                headers = Message()
                headers["Retry-After"] = "3"
                raise urllib.error.HTTPError(
                    request.full_url, 429, "too many requests", headers, io.BytesIO(b"{}")
                )

        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _RateLimitingOpener())
        monkeypatch.setattr("distill.local_vision._sleep", sleeps.append)

        result, failure_warning = try_interpret_image(
            LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
        )

        assert result is None
        assert failure_warning is not None
        assert failure_warning["code"] == "local_vision_retry_exhausted"
        # One original attempt plus two bounded retries, each honoring the
        # server's requested pause.
        assert opened["n"] == 3
        assert sleeps == [3.0, 3.0]

    def test_a_local_run_carries_no_budget(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The budget bounds remote cost (D-008); a slow-but-working local
        model must never be half-degraded by a run-wide clock."""
        for index in range(3):
            (tmp_path / f"frame{index}.png").write_bytes(b"png")
        payload = json.dumps(_chat_envelope(_frame_json())).encode()
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _FakeOpener(payload))

        interpreter = FrameInterpreter(
            LocalVisionConfig(),  # no remote opt-in
            probe=_available_probe,
            budget_bytes=1,  # would exhaust instantly if it applied
            budget_sec=0.000001,
        )
        frames, warnings = interpreter.interpret(
            [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(3)]
        )

        assert all(frame.reading is not None for frame in frames)
        assert "local_vision_budget_exhausted" not in {w["code"] for w in warnings}

    def test_an_exhausted_budget_blocks_the_request_before_it_is_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """check() runs before open: a spent budget never sends the request,
        which is the only thing that stops traffic once the run is over
        budget (the breaker covers the interpreter path; this covers every
        path that carries a budget)."""
        from distill.rapid_mlx import VisionStageBudget, _urlopen_json

        clock = {"now": 0.0}
        monkeypatch.setattr("distill.rapid_mlx._monotonic", lambda: clock["now"])
        opened = {"n": 0}

        class _NeverOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                opened["n"] += 1
                raise AssertionError("a spent budget must not send a request")

        monkeypatch.setattr("distill.rapid_mlx._OPENER", _NeverOpener())
        budget = VisionStageBudget(wall_clock_sec=5.0, max_bytes=1024)
        clock["now"] = 100.0

        with pytest.raises(LocalVisionFailure) as excinfo:
            _urlopen_json(
                "POST",
                "http://127.0.0.1:8000/v1/chat/completions",
                {"model": "m"},
                timeout_sec=30.0,
                budget=budget,
            )

        assert excinfo.value.code == "local_vision_budget_exhausted"
        assert opened["n"] == 0

    def test_a_429_without_retry_after_uses_the_default_backoff(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from email.message import Message

        sleeps: list[float] = []

        class _RateLimitingOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                raise urllib.error.HTTPError(
                    request.full_url, 503, "busy", Message(), io.BytesIO(b"{}")
                )

        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _RateLimitingOpener())
        monkeypatch.setattr("distill.local_vision._sleep", sleeps.append)

        result, failure_warning = try_interpret_image(
            LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
        )

        assert result is None
        assert failure_warning is not None
        assert failure_warning["code"] == "local_vision_retry_exhausted"
        assert sleeps == [1.0, 1.0]

    @pytest.mark.parametrize("retry_after", ["-1", "nan", "0"])
    def test_hostile_retry_after_values_never_crash_the_pause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retry_after: str
    ) -> None:
        from email.message import Message

        sleeps: list[float] = []

        class _HostileOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                headers = Message()
                headers["Retry-After"] = retry_after
                raise urllib.error.HTTPError(
                    request.full_url, 429, "limited", headers, io.BytesIO(b"{}")
                )

        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _HostileOpener())
        monkeypatch.setattr("distill.local_vision._sleep", sleeps.append)

        result, failure_warning = try_interpret_image(
            LocalVisionConfig(), image, "Interpret.", prompt_profile="technical"
        )

        assert result is None
        assert failure_warning is not None
        # The endpoint's bug stays the endpoint's: still a rate-limit story,
        # never a crash misreported as malformed JSON.
        assert failure_warning["code"] == "local_vision_retry_exhausted"
        assert all(0.0 <= pause <= 10.0 for pause in sleeps)

    def test_one_rate_limited_frame_degrades_the_remaining_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from email.message import Message

        for index in range(3):
            (tmp_path / f"frame{index}.png").write_bytes(b"png")
        opened = {"n": 0}

        class _RateLimitingOpener:
            def open(self, request: Any, timeout: float | None = None) -> Any:
                opened["n"] += 1
                raise urllib.error.HTTPError(
                    request.full_url, 429, "limited", Message(), io.BytesIO(b"{}")
                )

        monkeypatch.setattr("distill.rapid_mlx._OPENER", _RateLimitingOpener())
        monkeypatch.setattr("distill.local_vision._sleep", lambda _pause: None)

        interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe, debug=True)
        frames, warnings = interpreter.interpret(
            [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(3)]
        )

        # Frame 1 spends its bounded retries (3 requests); frames 2-3 are
        # never attempted - retrying a run into a rate limiter is the
        # amplification the immediate trip prevents.
        assert opened["n"] == 3
        assert all(frame.reading is None for frame in frames)
        assert interpreter.debug_info()["breaker"]["transport_failures"] == 0
        summary = next(w for w in warnings if w["code"] == BREAKER_WARNING_CODE)
        assert "rate-limiting" in summary["message"]
        assert "consecutive transport failures" not in summary["message"]


class TestPromptSideRedaction:
    """M2.6 (D-010): extracted text is redacted before it enters any vision
    prompt, regardless of endpoint locality and regardless of the render-side
    --no-redact-secrets flag."""

    def test_ocr_text_is_redacted_inside_the_built_prompt(self) -> None:
        from distill.vision_prompts import build_technical_frame_prompt

        prompt = build_technical_frame_prompt(
            ocr_text="export API_KEY=sk-prompt-secret-value\nplain slide text"
        )

        assert "sk-prompt-secret-value" not in prompt.prompt
        assert "[REDACTED]" in prompt.prompt
        assert "plain slide text" in prompt.prompt

    def test_url_userinfo_credentials_are_redacted(self) -> None:
        from distill.redact_secrets import redact_text

        result = redact_text(
            "curl http://admin:sk-userinfo-secret@vision.example.com:8000/v1/models"
        )

        assert "sk-userinfo-secret" not in result.text
        assert "vision.example.com" in result.text  # the host stays diagnostic

    def test_prompts_are_identical_local_vs_remote_and_ignore_the_render_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """D-010's cache-coherence half: the prompt depends on the frame, not
        on where the endpoint lives or what --no-redact-secrets says."""
        from dataclasses import replace

        image = tmp_path / "frame.png"
        image.write_bytes(b"png")
        captured: list[str] = []
        payload = json.dumps(_chat_envelope(_frame_json())).encode()

        class _CapturingOpener(_FakeOpener):
            def open(self, request: Any, timeout: float | None = None) -> Any:
                body = json.loads(request.data.decode())
                [message] = body["messages"]
                captured.append(next(p["text"] for p in message["content"] if p["type"] == "text"))
                return super().open(request, timeout)

        monkeypatch.setattr("distill.rapid_mlx._OPENER", _CapturingOpener(payload))
        # DISABLED, not the default: under the default policy the carrier
        # already redacted extracted_text at construction, so only the
        # --no-redact-secrets frame exercises the prompt-side sink at all.
        from distill.artifacts import RedactionState

        secret_frame = FrameArtifact(
            index=1,
            timestamp_sec=1.0,
            path=str(image),
            relative_path=f"frames/{image.name}",
            extracted_text="API_KEY=sk-uniform-secret",
            redaction=RedactionState.DISABLED,
        )
        local = LocalVisionConfig()
        remote = replace(
            LocalVisionConfig(),
            base_url="https://10.0.0.5:8000/v1",
            allow_remote_endpoint=True,
        )
        # The remote request would resolve/connect; the capture happens at the
        # opener so the fake serves both. Resolved-address check: 10.0.0.5 is
        # a literal, allow_remote + https passes the static check, and the
        # resolved check exempts https with the opt-in.
        for config in (local, remote):
            interpreter = FrameInterpreter(config, probe=_available_probe)
            interpreter.interpret([secret_frame])

        assert len(captured) == 2
        assert captured[0] == captured[1]
        assert "sk-uniform-secret" not in captured[0]
        assert "[REDACTED]" in captured[0]


class TestNonLocalProvenance:
    """M2.7 (D-012): a run that may send keyframes off-machine says so, and
    'was remote' folds into bundle identity; the address never does."""

    def test_a_remote_opted_run_records_the_non_local_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from dataclasses import replace

        (tmp_path / "frame0.png").write_bytes(b"png")
        payload = json.dumps(_chat_envelope(_frame_json())).encode()
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _FakeOpener(payload))
        remote = replace(
            LocalVisionConfig(),
            base_url="https://10.0.0.5:8000/v1",
            allow_remote_endpoint=True,
        )

        interpreter = FrameInterpreter(remote, probe=_available_probe)
        _frames, warnings = interpreter.interpret([_frame(1, tmp_path / "frame0.png")])

        assert "non_local_only_processing" in {w["code"] for w in warnings}

    def test_a_loopback_run_records_no_non_local_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "frame0.png").write_bytes(b"png")
        payload = json.dumps(_chat_envelope(_frame_json())).encode()
        monkeypatch.setattr("distill.rapid_mlx._OPENER", _FakeOpener(payload))

        interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe)
        _frames, warnings = interpreter.interpret([_frame(1, tmp_path / "frame0.png")])

        assert "non_local_only_processing" not in {w["code"] for w in warnings}

    def test_was_remote_folds_into_bundle_identity_but_the_address_does_not(
        self,
    ) -> None:
        local = DistillOptions.from_args({})
        remote = DistillOptions.from_args(
            {
                "local_vision_base_url": "https://10.0.0.5:8000/v1",
                "local_vision_allow_remote_endpoint": True,
            }
        )
        another_remote = DistillOptions.from_args(
            {
                "local_vision_base_url": "https://198.51.100.7:9000/v1",
                "local_vision_allow_remote_endpoint": True,
            }
        )

        # Remote- and local-produced bundles never share a key (D-012)...
        assert local.opts_hash("local") != remote.opts_hash("local")
        # ...but WHERE the remote endpoint lives stays a machine-local claim:
        # two different remote addresses produce the same identity, and the
        # address never enters the hashed payload.
        assert remote.opts_hash("local") == another_remote.opts_hash("local")
        assert "10.0.0.5" not in json.dumps(remote.cache_payload("local"))
        assert "10.0.0.5" not in json.dumps(remote.public_dict("local"))

    def test_the_self_contained_render_carries_the_non_local_warning(self) -> None:
        from distill.errors import warning as make_warning
        from distill.render import render_markdown

        non_local = make_warning(
            "local_vision",
            "non_local_only_processing",
            "this run was configured to send keyframes to a non-loopback "
            "vision endpoint; the generation may not be local-only "
            "processing.",
        )

        frame = _frame(1, Path("frames/frame1.png"), extracted_text="on-screen text")
        render = render_markdown(
            "example.mp4",
            12.0,
            None,
            [frame],
            [non_local],
        )

        assert "## Warnings" in render
        assert "non_local_only_processing" in render

    @pytest.mark.parametrize(
        ("base_url", "allow_remote", "expected"),
        [
            # A hostname WITHOUT the opt-in never folds: the per-request
            # resolved check keeps it loopback-or-rejected.
            ("https://vision.example.com/v1", False, False),
            # The same hostname WITH the opt-in fails closed as remote.
            ("https://vision.example.com/v1", True, True),
            # http can never leave the machine (proved per request, opt-in
            # or not), so localhost-over-http is not a non-local claim.
            ("http://localhost:8000/v1", True, False),
            ("https://10.0.0.5:8000/v1", True, True),
            ("http://127.0.0.1:8000/v1", True, False),
        ],
    )
    def test_the_non_local_predicate_terms(
        self, base_url: str, allow_remote: bool, expected: bool
    ) -> None:
        from dataclasses import replace

        from distill.local_vision import config_is_non_local

        config = replace(LocalVisionConfig(), base_url=base_url, allow_remote_endpoint=allow_remote)

        assert config_is_non_local(config) is expected

    def test_the_warning_survives_an_unavailable_endpoint(self, tmp_path: Path) -> None:
        """The fold is computed from options before any I/O; the disclosure
        must reach the render even when the endpoint never answers."""
        from dataclasses import replace

        (tmp_path / "frame0.png").write_bytes(b"png")

        def unavailable_probe(config: LocalVisionConfig) -> LocalVisionProbe:
            return LocalVisionProbe(
                available=False,
                backend=config.backend,
                model=config.model,
                base_url=config.base_url,
                code="local_vision_rapid_mlx_unavailable",
                message="down",
                detail={},
            )

        remote = replace(
            LocalVisionConfig(),
            base_url="https://10.0.0.5:8000/v1",
            allow_remote_endpoint=True,
        )
        interpreter = FrameInterpreter(remote, probe=unavailable_probe)
        _frames, warnings = interpreter.interpret([_frame(1, tmp_path / "frame0.png")])

        codes = {w["code"] for w in warnings}
        assert "non_local_only_processing" in codes
        assert "local_vision_rapid_mlx_unavailable" in codes


def test_end_to_end_credential_budget_and_provenance_compose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The composed remote path, config file to wire: api_key_env resolves
    into a bearer on BOTH the probe and the frame request, the run-wide
    budget rides both, the non-local disclosure lands in the warnings, the
    fold lands in identity, and neither credential nor address reaches the
    bundle surfaces."""

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "distill.local-vision.json").write_text(
        json.dumps(
            {
                "api_key_env": "DISTILL_E2E_VISION_KEY",
                "base_url": "https://10.0.0.5:8000/v1",
                "allow_remote_endpoint": True,
            }
        )
    )
    monkeypatch.setenv("DISTILL_E2E_VISION_KEY", "sk-e2e-secret")
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "frame0.png").write_bytes(b"png")

    options = DistillOptions.from_args({})
    config = options.local_vision_config()

    auth_headers: list[str | None] = []
    budgets_seen: list[object] = []

    class _RecordingOpener:
        def open(self, request: Any, timeout: float | None = None) -> Any:
            auth_headers.append(request.get_header("Authorization"))
            body = (
                json.dumps(_models_body(DEFAULT_MODEL))
                if request.full_url.rstrip("/").endswith("/models")
                else json.dumps(_chat_envelope(_frame_json()))
            ).encode()

            class _Resp:
                def __enter__(self) -> Any:
                    return self

                def __exit__(self, *exc: object) -> None:
                    return None

                def __init__(self) -> None:
                    self._offset = 0

                def read1(self, size: int = -1) -> bytes:
                    chunk = body[self._offset : self._offset + (size if size > 0 else len(body))]
                    self._offset += len(chunk)
                    return chunk

                read = read1

            return _Resp()

    monkeypatch.setattr("distill.rapid_mlx._OPENER", _RecordingOpener())

    original_post = __import__("distill.rapid_mlx", fromlist=["_http_post_json"])._http_post_json

    def spying_post(*args: Any, **kwargs: Any) -> Any:
        budgets_seen.append(kwargs.get("budget"))
        return original_post(*args, **kwargs)

    monkeypatch.setattr("distill.local_vision._http_post_json", spying_post)

    interpreter = FrameInterpreter(config)
    frames, warnings = interpreter.interpret([_frame(1, tmp_path / "frame0.png")])

    # Both requests authenticated with the resolved credential.
    assert auth_headers and all(h == "Bearer sk-e2e-secret" for h in auth_headers)
    # The frame request carried the run's budget (remote-opted config).
    assert budgets_seen and budgets_seen[-1] is not None
    assert frames[0].reading is not None
    assert "non_local_only_processing" in {w["code"] for w in warnings}
    # Identity folds; the credential reaches NO surface, and the address
    # stays out of the bundle surfaces (it remains a machine-local claim in
    # the local diagnostics view, per ADR-0004).
    assert options.cache_payload("local")["local_vision_non_local"] is True
    bundle_surfaces = (
        json.dumps(options.cache_payload("local")),
        json.dumps(options.public_dict("local")),
    )
    for surface in bundle_surfaces:
        assert "sk-e2e-secret" not in surface
        assert "10.0.0.5" not in surface
    assert "sk-e2e-secret" not in json.dumps(config.public_dict(), default=str)


class TestSalienceSchema:
    """M4.3 (D-003/D-018): adds_information is a strict boolean; anything
    else yields absent salience, never a guessed value."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"adds_information": True, "reason": "shows a diagram"},
            {"adds_information": False, "reason": "restates the speech"},
        ],
    )
    def test_a_strict_boolean_parses(self, payload: dict) -> None:
        from distill.rapid_mlx import parse_frame_salience

        salience = parse_frame_salience(payload)

        assert salience is not None
        assert salience.adds_information is payload["adds_information"]
        assert salience.reason == payload["reason"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"adds_information": "false", "reason": "stringly"},
            {"adds_information": "true", "reason": "stringly"},
            {"adds_information": 1, "reason": "inty"},
            {"adds_information": None, "reason": "nully"},
            {"reason": "missing the field"},
            {},
            "not a mapping",
            None,
        ],
    )
    def test_anything_else_is_absent_not_guessed(self, payload: object) -> None:
        from distill.rapid_mlx import parse_frame_salience

        assert parse_frame_salience(payload) is None

    def test_an_oversize_reason_is_truncated_with_a_notice(self) -> None:
        from distill.rapid_mlx import SALIENCE_REASON_MAX_CHARS, parse_frame_salience

        salience = parse_frame_salience(
            {"adds_information": True, "reason": "r" * (SALIENCE_REASON_MAX_CHARS + 50)}
        )

        assert salience is not None
        assert len(salience.reason) == SALIENCE_REASON_MAX_CHARS
        assert salience.reason_truncated is True

    def test_a_reason_exactly_at_the_cap_is_not_flagged_truncated(self) -> None:
        from distill.rapid_mlx import SALIENCE_REASON_MAX_CHARS, parse_frame_salience

        salience = parse_frame_salience(
            {"adds_information": True, "reason": "r" * SALIENCE_REASON_MAX_CHARS}
        )

        assert salience is not None
        assert salience.reason_truncated is False

    def test_a_non_string_reason_is_refused_not_stringified(self) -> None:
        from distill.rapid_mlx import parse_frame_salience

        salience = parse_frame_salience(
            {"adds_information": True, "reason": {"why": "a structure"}}
        )

        assert salience is not None
        assert salience.reason == ""


class TestSalienceRecording:
    """M4.4 (D-003/D-018): validated salience lands on the frame artifact;
    processing never drops a frame because of it."""

    def test_validated_salience_is_recorded_on_every_frame_kept(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for index in range(2):
            (tmp_path / f"frame{index}.png").write_bytes(b"png")

        def fake_urlopen(
            method: str, url: str, body: Any = None, timeout_sec: float = 30.0, **_: Any
        ) -> Any:
            if url.rstrip("/").endswith("/models"):
                return _models_body(DEFAULT_MODEL)
            content = _frame_json(
                adds_information=False, reason="restates the speech"
            )
            return _chat_envelope(content)

        monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)
        segments = ({"start": 0.0, "end": 20.0, "text": "the speaker explains"},)

        interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe)
        frames, _warnings = interpreter.interpret(
            [_frame(index + 1, tmp_path / f"frame{index}.png") for index in range(2)],
            transcript_segments=segments,
        )

        # Never dropped, and every kept frame carries the judgment.
        assert len(frames) == 2
        for frame in frames:
            assert frame.salience is not None
            assert frame.salience["adds_information"] is False
            assert frame.salience["reason"] == "restates the speech"

    def test_invalid_salience_stays_absent_on_the_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "frame0.png").write_bytes(b"png")

        def fake_urlopen(
            method: str, url: str, body: Any = None, timeout_sec: float = 30.0, **_: Any
        ) -> Any:
            if url.rstrip("/").endswith("/models"):
                return _models_body(DEFAULT_MODEL)
            return _chat_envelope(_frame_json(adds_information="false", reason="stringly"))

        monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)
        segments = ({"start": 0.0, "end": 20.0, "text": "context"},)

        interpreter = FrameInterpreter(LocalVisionConfig(), probe=_available_probe)
        frames, _warnings = interpreter.interpret(
            [_frame(1, tmp_path / "frame0.png")], transcript_segments=segments
        )

        assert frames[0].reading is not None  # the reading itself is fine
        assert frames[0].salience is None  # absent, never guessed


class TestFrameSalienceToggle:
    """M4.5 (D-017): top-level frame_salience, default on, user-disableable,
    cache_key=True - on and off yield different bundle keys."""

    def test_defaults_on_and_is_disableable_and_keys_the_cache(self) -> None:
        default = DistillOptions.from_args({})
        disabled = DistillOptions.from_args({"frame_salience": False})

        assert default.frame_salience is True
        assert disabled.frame_salience is False
        assert default.opts_hash("local") != disabled.opts_hash("local")
        assert default.cache_payload("local")["frame_salience"] is True

    def test_disabled_salience_builds_no_window_and_records_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "frame0.png").write_bytes(b"png")
        prompts: list[str] = []

        def fake_urlopen(
            method: str, url: str, body: Any = None, timeout_sec: float = 30.0, **_: Any
        ) -> Any:
            if url.rstrip("/").endswith("/models"):
                return _models_body(DEFAULT_MODEL)
            [message] = body["messages"]
            prompts.append(
                next(p["text"] for p in message["content"] if p["type"] == "text")
            )
            return _chat_envelope(
                _frame_json(adds_information=True, reason="claims to add")
            )

        monkeypatch.setattr("distill.rapid_mlx._urlopen_json", fake_urlopen)
        segments = ({"start": 0.0, "end": 20.0, "text": "context"},)

        interpreter = FrameInterpreter(
            LocalVisionConfig(), probe=_available_probe, frame_salience=False
        )
        frames, _warnings = interpreter.interpret(
            [_frame(1, tmp_path / "frame0.png")], transcript_segments=segments
        )

        assert "adds_information" not in prompts[0]
        assert frames[0].salience is None
