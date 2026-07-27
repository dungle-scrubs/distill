"""Tests for the typed carriers **extracted text** travels in.

The invariant under test is R-19/R-20's seam: extracted text is redacted where
it *enters* a carrier, and a carrier whose **redaction** policy was never
applied cannot be serialized into a **generation** or a **render**.

Enforcement is layered (D-022) and these tests are written against the layer
each one can actually reach: construction runs the policy (D-019), so it is
tested for redacted text and a recorded state; the type expresses intent, so it
is tested for frozen-ness and for immutable nesting; the serializer performs the
runtime check, so it is tested for refusal of a state construction cannot
produce. None of them claims the set is a guarantee that no secret survives -
what ran is a policy of patterns, and a secret shaped like none of them is
carried through untouched.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
from test_local_integration import fake_transcribe, make_short_screencast

from distill import pipeline as distill_session
from distill.artifacts import (
    EXTRACTED_TEXT_LIMIT_BYTES,
    TRUNCATION_WARNING_CODE,
    Carrier,
    FrameArtifact,
    RedactionPolicyNotApplied,
    RedactionState,
    StageResult,
    Transcript,
    serialize,
)
from distill.bundle_store import BundleRun, BundleStore


def make_frame(**overrides: Any) -> FrameArtifact:
    fields: dict[str, Any] = {
        "index": 1,
        "timestamp_sec": 1.5,
        "path": "/tmp/bundle/frames/frame_001.png",
        "relative_path": "frames/frame_001.png",
        "extracted_text": "AWS console",
        "phash": "ffff0000ffff0000",
        "source_candidate_index": 0,
    }
    fields.update(overrides)
    return FrameArtifact(**fields)


def make_transcript(**overrides: Any) -> Transcript:
    fields: dict[str, Any] = {
        "language": "en",
        "language_probability": 0.99,
        "segments": [
            {"start_sec": 0.0, "end_sec": 1.0, "text": "hello there", "words": []},
        ],
    }
    fields.update(overrides)
    return Transcript(**fields)


def prose(size_bytes: int) -> str:
    """`size_bytes` of ASCII filler with no secret-shaped run in it.

    The filler cannot be one long run of a single character. An unbroken
    alphanumeric run of 40 characters or more is exactly what
    `redact_secrets`' generic base64 rule matches, and M4.2 (D-019) redacts at
    construction *before* the cap - so such a filler would test the redaction
    policy rather than the cap the test is about.
    """
    unit = "slide text "
    return (unit * (size_bytes // len(unit) + 1))[:size_bytes]


def never_constructed(carrier: Any) -> Any:
    """A carrier bearing `NOT_APPLIED`, which construction can no longer produce.

    D-019 made construction the point the **redaction** policy runs, so this
    state is only reachable by writing past the freeze: a subclass that
    overrode `__post_init__`, or exactly this. `serialize`'s refusal is defence
    in depth against that route, which makes this the only way to reach the
    layer under test.
    """
    object.__setattr__(carrier, "redaction", RedactionState.NOT_APPLIED)
    return carrier


# 1. FrameArtifact and Transcript defined as frozen carriers


def test_frame_artifact_and_transcript_are_frozen() -> None:
    """A carrier's own fields cannot be reassigned after construction.

    Frozen is the first layer only: it stops rebinding a field, which is why
    the nested-collection test below exists as well.
    """
    frame: Any = make_frame(redaction=RedactionState.APPLIED)
    transcript: Any = make_transcript(redaction=RedactionState.APPLIED)

    # Typed as Any deliberately: `ty` rejects both assignments statically, and
    # the static rejection is the layer being tested everywhere else. What is
    # under test here is that the *runtime* refuses too, which is the layer an
    # unchecked caller meets.
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.extracted_text = "sk-live-0123456789abcdefghij"
    with pytest.raises(dataclasses.FrozenInstanceError):
        transcript.language = "fr"


# 2. FAILS FIRST: serializing a carrier whose redaction policy was not applied
#    raises


def test_serializing_a_frame_whose_redaction_policy_was_not_applied_raises() -> None:
    """Finding 4's shape: a frame reaches a writer without the policy having run."""
    frame = never_constructed(make_frame(extracted_text="export API_KEY=sk-live-0123456789ab"))

    assert frame.redaction is RedactionState.NOT_APPLIED
    with pytest.raises(RedactionPolicyNotApplied) as raised:
        serialize(frame)
    assert raised.value.code == "E_REDACTION_POLICY_NOT_APPLIED"


def test_serializing_a_transcript_whose_redaction_policy_was_not_applied_raises() -> None:
    """Finding 15's shape: the transcript took a path the redaction stage did not cover."""
    transcript = never_constructed(make_transcript())

    with pytest.raises(RedactionPolicyNotApplied):
        serialize(transcript)


def test_serializing_a_stage_result_whose_redaction_policy_was_not_applied_raises() -> None:
    """A **stage result** is a **redaction sink** too - resume scratch is still on disk."""
    stage_result = never_constructed(
        StageResult(
            schema_version=1,
            bundle_key="a" * 64,
            stage="ocr",
            payload={"frames": []},
        )
    )

    with pytest.raises(RedactionPolicyNotApplied):
        serialize(stage_result)


# 3. The runtime check is performed in the SERIALIZER, not only in the type


def test_the_check_lives_in_the_serializer_and_not_only_in_the_type() -> None:
    """The policy runs at construction, and the serializer still refuses.

    Two layers, and the second is not redundant. D-019 makes construction the
    point the policy runs, so a constructed carrier is never `NOT_APPLIED` -
    but the type cannot enforce that against a subclass that overrides
    `__post_init__` or a caller that writes past the freeze. The serializer is
    the last point before the text becomes durable, so the refusal stays there
    for the carrier that was never really constructed.
    """
    constructed = make_frame(extracted_text="export API_KEY=sk-live-0123456789abcdefghij")
    assert constructed.redaction is RedactionState.APPLIED
    assert "sk-live-0123456789abcdefghij" not in serialize(constructed)["extracted_text"]

    with pytest.raises(RedactionPolicyNotApplied):
        serialize(never_constructed(make_frame()))


def test_a_carrier_rebuilt_through_replace_is_redacted_again() -> None:
    """Every route into a carrier runs the policy, not just the literal constructor.

    `dataclasses.replace` re-runs `__post_init__`, so a carrier rebuilt around
    new text is redacted on its own terms rather than inheriting the state its
    predecessor earned. That is the route by which a redacted carrier would
    otherwise become an unredacted one carrying an `APPLIED` label.
    """
    applied = make_frame()
    assert serialize(applied)["extracted_text"] == "AWS console"

    rebuilt = dataclasses.replace(applied, extracted_text="export API_KEY=sk-live-0123456789ab")
    assert rebuilt.redaction is RedactionState.APPLIED
    assert "sk-live-0123456789ab" not in serialize(rebuilt)["extracted_text"]


def test_the_serialized_document_records_which_policy_state_produced_it() -> None:
    """The state is bundle content: a reader can tell redacted output from opted-out output."""
    assert serialize(make_frame(redaction=RedactionState.APPLIED))["redaction"] == "applied"
    assert serialize(make_frame(redaction=RedactionState.DISABLED))["redaction"] == "disabled"


def test_a_serialized_document_is_json_writable() -> None:
    """The serializer, not `_payload`, is what a writer can use.

    The internal document is immutable and `json` cannot encode it, so the
    naive bypass fails loudly instead of writing a mangled `mappingproxy`
    repr into a bundle.
    """
    frame = make_frame(
        redaction=RedactionState.APPLIED,
        interpretation={"visual_summary": "a console", "detected_elements": ["a button"]},
    )

    document = serialize(frame)
    assert json.loads(json.dumps(document))["interpretation"]["detected_elements"] == ["a button"]
    with pytest.raises(TypeError):
        json.dumps(frame._payload())


# 4. An explicitly disabled policy satisfies the check


def test_an_explicitly_disabled_policy_serializes(  # D-020
) -> None:
    """`--no-redact-secrets` is retained: the check is on the policy, not the text.

    The text below is still secret-shaped. That is the point - a DISABLED
    policy is applied, so the carrier serializes with the text intact.
    """
    secret = "export API_KEY=sk-live-0123456789abcdefghij"
    frame = make_frame(extracted_text=secret, redaction=RedactionState.DISABLED)
    transcript = make_transcript(redaction=RedactionState.DISABLED)

    assert serialize(frame)["extracted_text"] == secret
    assert serialize(transcript)["segments"][0]["text"] == "hello there"


def test_disabled_is_a_policy_state_and_not_inferred_from_the_text(  # D-020
) -> None:
    """Unchanged text does not mean the policy ran, and changed text does not mean it did not.

    Both directions are wrong to infer, and both are exercised here: text with
    nothing secret-shaped in it comes out of a policy that ran completely
    unchanged, and text a user pasted `[REDACTED]` into looks redacted under a
    policy that was explicitly disabled. The state is modelled, never inferred.
    """
    innocuous = make_frame(extracted_text="the title slide")
    pre_redacted = make_frame(
        extracted_text="API_KEY=[REDACTED]",
        redaction=RedactionState.DISABLED,
    )

    assert serialize(innocuous)["extracted_text"] == "the title slide"
    assert innocuous.redaction is RedactionState.APPLIED
    assert pre_redacted.redaction is RedactionState.DISABLED
    assert RedactionState.DISABLED.policy_applied is True
    assert RedactionState.APPLIED.policy_applied is True
    assert RedactionState.NOT_APPLIED.policy_applied is False


# 5. --no-redact-secrets still produces a bundle


def test_no_redact_secrets_still_produces_a_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The opt-out reaches a published **generation**, end to end.

    R-20 is a refusal, and a refusal wired to the wrong state fails closed on
    exactly the users who opted out. This runs the real pipeline with
    `redact_secrets` off and asserts a bundle came back.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    assert response["cached"] is False
    assert Path(response["markdown_path"]).is_file()
    assert Path(response["transcript_path"]).is_file()
    assert response["frames"]


# 6. An extracted-text field beyond 256 KiB is truncated with a WARNING


def test_extracted_text_beyond_the_cap_is_truncated_with_a_warning() -> None:
    """R-58: the cap is per field, and truncation is recorded rather than silent."""
    oversized = prose(EXTRACTED_TEXT_LIMIT_BYTES + 1)
    frame = make_frame(extracted_text=oversized, redaction=RedactionState.APPLIED)

    document = serialize(frame)
    assert len(document["extracted_text"].encode()) == EXTRACTED_TEXT_LIMIT_BYTES
    codes = [item["code"] for item in document["warnings"]]
    assert codes == [TRUNCATION_WARNING_CODE]
    assert "extracted_text" in document["warnings"][0]["message"]


def test_the_cap_counts_bytes_and_never_splits_a_character() -> None:
    """256 KiB is a byte budget; a multi-byte character must not be cut in half."""
    oversized = "é" * EXTRACTED_TEXT_LIMIT_BYTES  # two bytes each
    frame = make_frame(extracted_text=oversized, redaction=RedactionState.APPLIED)

    text = serialize(frame)["extracted_text"]
    assert len(text.encode()) <= EXTRACTED_TEXT_LIMIT_BYTES
    assert text == "é" * (EXTRACTED_TEXT_LIMIT_BYTES // 2)


def test_a_field_exactly_at_the_cap_is_not_truncated() -> None:
    """The boundary is inclusive: 256 KiB is allowed, 256 KiB + 1 is not."""
    exact = prose(EXTRACTED_TEXT_LIMIT_BYTES)
    frame = make_frame(extracted_text=exact, redaction=RedactionState.APPLIED)

    document = serialize(frame)
    assert document["extracted_text"] == exact
    assert document["warnings"] == []


def test_a_secret_straddling_the_cap_is_redacted_before_it_is_cut() -> None:
    """The 256 KiB cap must not decide what the redaction policy is allowed to see.

    This is the primary site of the ordering `link_label` already got wrong
    once. A value crossing the cap is sliced into a prefix that matches no
    pattern, and that prefix is then durable in full: inverting the two lines
    in `_freeze` published `...slide text sl` + `sk-live-0123`, twelve
    characters of the key, into the **manifest**.

    Both halves of the outcome are asserted, because each fails under the
    inverted order for a different reason: no prefix of the key survives, and
    nothing was truncated at all - redaction shortened the field back under the
    cap, so there is no **warning** to raise.
    """
    secret = "sk-live-0123456789abcdefghij"
    # The cut lands eleven characters into the key: long enough to be a durable
    # prefix, short enough that no pattern in `redact_secrets` matches it.
    text = f"{prose(EXTRACTED_TEXT_LIMIT_BYTES - 12)} {secret}"
    assert len(text.encode()) > EXTRACTED_TEXT_LIMIT_BYTES

    document = serialize(make_frame(extracted_text=text))

    assert document["extracted_text"].endswith("[REDACTED]")
    assert "sk-" not in document["extracted_text"]
    assert document["warnings"] == []


def test_the_cap_reaches_extracted_text_nested_inside_an_interpretation() -> None:
    """An **interpretation** is extracted text too - the model echoes the screen back."""
    frame = make_frame(
        redaction=RedactionState.APPLIED,
        interpretation={
            "visual_summary": prose(EXTRACTED_TEXT_LIMIT_BYTES + 1),
            "detected_elements": [prose(EXTRACTED_TEXT_LIMIT_BYTES + 1)],
        },
    )

    document = serialize(frame)
    assert len(document["interpretation"]["visual_summary"].encode()) == EXTRACTED_TEXT_LIMIT_BYTES
    assert (
        len(document["interpretation"]["detected_elements"][0].encode())
        == EXTRACTED_TEXT_LIMIT_BYTES
    )
    messages = sorted(item["message"] for item in document["warnings"])
    assert len(messages) == 2
    assert "interpretation.detected_elements[0]" in messages[0]
    assert "interpretation.visual_summary" in messages[1]


def test_the_cap_reaches_transcript_segment_text() -> None:
    """R-21: the transcript is extracted text on the same terms as keyframe text."""
    transcript = make_transcript(
        redaction=RedactionState.APPLIED,
        segments=[
            {"start_sec": 0.0, "end_sec": 1.0, "text": prose(EXTRACTED_TEXT_LIMIT_BYTES + 1)}
        ],
    )

    document = serialize(transcript)
    assert len(document["segments"][0]["text"].encode()) == EXTRACTED_TEXT_LIMIT_BYTES
    assert [item["code"] for item in document["warnings"]] == [TRUNCATION_WARNING_CODE]


# 7. No aggregate bundle cap is imposed


def test_no_aggregate_cap_is_imposed_across_a_bundles_carriers() -> None:
    """D-045: 80 keyframes at the per-field cap is bounded enough.

    Eighty frames each holding a full 256 KiB field is ~20 MiB of extracted
    text. Every one of them serializes whole: an aggregate cap would add a
    second failure mode for no extra protection.
    """
    at_cap = prose(EXTRACTED_TEXT_LIMIT_BYTES)
    frames = [
        make_frame(index=index, extracted_text=at_cap, redaction=RedactionState.APPLIED)
        for index in range(80)
    ]

    documents = [serialize(frame) for frame in frames]
    total = sum(len(document["extracted_text"].encode()) for document in documents)
    assert total == 80 * EXTRACTED_TEXT_LIMIT_BYTES
    assert all(document["warnings"] == [] for document in documents)


# 8. Nested collections immutable so post-redaction mutation is impossible


def test_mutating_the_dict_a_carrier_was_built_from_does_not_reach_the_carrier() -> None:
    """The carrier copies on the way in; the caller keeps no handle into it."""
    interpretation = {"visual_summary": "a console", "detected_elements": ["a button"]}
    frame = make_frame(interpretation=interpretation, redaction=RedactionState.APPLIED)

    interpretation["visual_summary"] = "sk-live-0123456789abcdefghij"
    interpretation["detected_elements"].append("sk-live-0123456789abcdefghij")

    assert serialize(frame)["interpretation"] == {
        "visual_summary": "a console",
        "detected_elements": ["a button"],
    }


def test_nested_collections_inside_a_carrier_cannot_be_mutated() -> None:
    """Frozen is not enough when a field holds a list or a dict.

    Post-redaction mutation is how redacted extracted text becomes unredacted
    again between construction and the writer, so the nesting is immutable all
    the way down rather than one level deep.
    """
    frame = make_frame(
        redaction=RedactionState.APPLIED,
        interpretation={"detected_elements": ["a button"], "nested": {"verbatim_text": "OK"}},
    )
    interpretation: Any = frame.interpretation
    warnings: Any = frame.warnings

    with pytest.raises(TypeError):
        interpretation["visual_summary"] = "leaked"
    with pytest.raises(TypeError):
        interpretation["nested"]["verbatim_text"] = "leaked"
    with pytest.raises(AttributeError):
        interpretation["detected_elements"].append("leaked")
    with pytest.raises(AttributeError):
        warnings.append({"stage": "artifacts", "code": "x", "message": "y"})


def test_a_serialized_document_is_a_copy_the_caller_cannot_write_back_through() -> None:
    """Thawing for JSON must not hand out a window into the carrier."""
    frame = make_frame(redaction=RedactionState.APPLIED, interpretation={"verbatim_text": "OK"})

    document = serialize(frame)
    document["interpretation"]["verbatim_text"] = "leaked"

    assert serialize(frame)["interpretation"]["verbatim_text"] == "OK"


def test_a_carrier_naming_an_extracted_text_field_it_does_not_have_is_refused() -> None:
    """A misnamed region reads as capped and behaves as uncapped - the one silent hole.

    R-58 is only as good as each carrier's declaration of which of its fields
    hold extracted text, so a name that is not a field is a construction-time
    failure rather than a field nobody notices went uncapped.
    """

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Miscarrier(Carrier):
        EXTRACTED_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("ocr_text",)

        extracted_text: str = ""

        def _payload(self) -> Mapping[str, Any]:
            return MappingProxyType({"extracted_text": self.extracted_text})

    with pytest.raises(AssertionError, match="ocr_text"):
        Miscarrier(redaction=RedactionState.APPLIED)


def test_a_mutable_value_a_carrier_cannot_freeze_is_refused() -> None:
    """An unsupported type is a defect at construction, not a hole at the writer."""
    with pytest.raises(TypeError):
        make_frame(interpretation={"detected_elements": {"a set is not frozen"}})


def test_a_stage_results_payload_is_redacted_before_it_reaches_the_disk(tmp_path: Path) -> None:
    """`StageResult`'s payload region is a claim, so it is exercised (D-022).

    A **stage result** is durable the moment it is written - it sits in the
    **staging directory** an interrupted run resumes from - which is what makes
    it a **redaction sink** although no reader is ever served one (finding 4).
    Naming `payload` an extracted-text region is how that holds for text a
    stage built by hand rather than took from a carrier that was capped
    already, and nothing failed when the region was emptied: this is the test
    that does.

    Asserted on the bytes on disk rather than on the document `serialize`
    returned, because the region exists for what durability costs.
    """
    store = BundleStore.open(tmp_path / "output")
    run = store.begin("abc123")
    assert isinstance(run, BundleRun)
    secret = "sk-live-0123456789abcdefghij"

    try:
        run.write_stage("ocr", {"frames": [{"note": f"API_KEY={secret}"}]})
        recorded = (run.paths.generation / "_ocr.json").read_text()
    finally:
        run.release()

    assert secret not in recorded
    assert "API_KEY=[REDACTED]" in recorded


def test_bytes_are_refused_rather_than_transcoded_past_the_policy() -> None:
    """A `bytes` value is a `Sequence`, and freezing it silently made it a tuple of ints.

    Which is a value the redaction policy never saw - it is not a `str`, so no
    pattern ran over it - and one the cap never measured, and one `json`
    encodes happily, so it survives a **publish** and a **resume** intact. That
    is the refusal above being claimed and not held: an unfreezable type is a
    defect at construction precisely so it cannot become a hole at the writer.
    Every byte-shaped type is refused, not just the one that is immutable.
    """
    for value in (b"sk-live-0123456789abcdefghij", bytearray(b"secret"), memoryview(b"secret")):
        with pytest.raises(TypeError, match="carrier cannot freeze"):
            make_frame(interpretation={"verbatim_text": value})
