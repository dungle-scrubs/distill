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
    config = LocalVisionConfig(model="qwen3-vl:32b")

    probe = probe_rapid_mlx_availability(config, requestor=FakeRapidMlx(models=["other"]))

    assert probe.available is False
    assert probe.code == "local_vision_model_unavailable"
    assert probe.detail["configured_model"] == "qwen3-vl:32b"
    assert probe.detail["served_models"] == ["other"]


def test_probe_reports_malformed_models_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def requestor(*, method: str, url: str, body: Any = None, timeout: float = 30.0) -> Any:
        raise RuntimeError("HTTP 500 from server: internal error")

    probe = probe_rapid_mlx_availability(LocalVisionConfig(), requestor=requestor)

    assert probe.available is False
    assert probe.code == "local_vision_rapid_mlx_malformed_response"


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

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


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
