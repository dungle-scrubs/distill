"""The carriers **extracted text** travels in, and the one serializer that lets it out.

This module owns the shape of a **frame artifact**, a **transcript** and a
**stage result** in memory: their fields, *when* the **redaction** policy runs
over them and the state that records it, the per-field 256 KiB cap on extracted
text (R-58), the immutability of everything nested inside them, and the runtime
refusal to serialize a carrier whose redaction policy was never applied (R-20).

When the policy runs is the answer to D-019: at construction, over every
extracted-text region, before the carrier exists as a value anything else can
read. That is what R-19 means by "before it can be written anywhere" - a
**stage result** is on disk long before a **generation** is published, so a
policy that ran later only ever cleaned up a copy (finding 4).

It does not own the redaction policy itself - which values are secret-shaped and
what replaces them is `redact_secrets.py`'s - nor any file write, nor the
markdown **render**, nor the delimiting that keeps extracted text data rather
than instruction. Beyond the policy it imports nothing from Distill except the
error and warning vocabulary, deliberately: a carrier is what the producing
stages hand to the writers, so it must not depend on either side.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar

from .errors import DistillError, warning
from .redact_secrets import redact_text

# R-58 / D-045: each individual extracted-text field is capped, and no aggregate
# bundle cap is imposed on top of it. Eighty keyframes at this cap already bounds
# a bundle's extracted text; a second, aggregate limit would add a failure mode
# (a run that dies on its last frame for a reason no single field explains)
# without protecting anything the per-field cap leaves exposed.
EXTRACTED_TEXT_LIMIT_BYTES = 256 * 1024

WARNING_STAGE = "artifacts"
TRUNCATION_WARNING_CODE = "extracted_text_truncated"
REDACTION_POLICY_NOT_APPLIED_CODE = "E_REDACTION_POLICY_NOT_APPLIED"


class RedactionState(StrEnum):
    """Whether the **redaction** policy has been applied to a carrier's text.

    D-020: this records the state of the *policy*, never a judgement about the
    text. `DISABLED` is an applied policy - the user asked for
    `--no-redact-secrets` and got it - which is what keeps that flag working
    under R-20. Inferring the state from whether the text changed would get both
    cases wrong: text with no secret in it is unchanged by a policy that ran,
    and text a user pasted `[REDACTED]` into looks redacted without one.

    Passed *into* a carrier it is a choice, and the only choice is `DISABLED`:
    every other value means "run the policy", which construction then does
    (D-019). So `NOT_APPLIED` never survives `__post_init__`, and `APPLIED` is
    a fact the constructor established rather than a claim a producer made.
    A carrier bearing `NOT_APPLIED` therefore reached the serializer without
    being constructed - a subclass that skipped `__post_init__`, or a write
    through `object.__setattr__` past the freeze - which is why `serialize`
    still refuses it.
    """

    NOT_APPLIED = "not_applied"
    APPLIED = "applied"
    DISABLED = "disabled"

    @property
    def policy_applied(self) -> bool:
        return self is not RedactionState.NOT_APPLIED


class RedactionPolicyNotApplied(DistillError):
    """A carrier reached the serializer before the redaction policy ran.

    Fatal rather than a **warning**: the alternative to refusing is writing
    text that may carry a secret into a **generation**, and no bundle is better
    than a bundle that quietly contains one.
    """

    def __init__(self, carrier: str) -> None:
        super().__init__(
            REDACTION_POLICY_NOT_APPLIED_CODE,
            WARNING_STAGE,
            f"{carrier} cannot be serialized: its redaction policy has not been applied",
            {"carrier": carrier},
        )


def _cap_extracted_text(text: str, *, path: str, warnings: list[Mapping[str, str]]) -> str:
    """Return `text` within the per-field cap, recording a **warning** if it was cut.

    The budget is bytes, because KiB is a byte unit and a character is not a
    fixed number of them; the cut lands on a character boundary, so truncating
    cannot produce text that is not valid UTF-8.

    Nothing is appended to mark the truncation. A marker would be Distill's own
    words placed inside extracted text, which is exactly what the
    untrusted-data boundary exists to prevent; the warning carried with the
    bundle is the record.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= EXTRACTED_TEXT_LIMIT_BYTES:
        return text
    truncated = encoded[:EXTRACTED_TEXT_LIMIT_BYTES].decode("utf-8", errors="ignore")
    warnings.append(
        _frozen_warning(
            warning(
                WARNING_STAGE,
                TRUNCATION_WARNING_CODE,
                f"{path} held {len(encoded)} bytes of extracted text and was truncated to the "
                f"{EXTRACTED_TEXT_LIMIT_BYTES} byte per-field cap",
            )
        )
    )
    return truncated


def _frozen_warning(payload: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(payload))


def _redact(text: str, *, warnings: list[Mapping[str, str]]) -> str:
    """Return `text` with secret-shaped values replaced, keeping the policy's warnings.

    Every string in an extracted-text region goes through this, every time,
    with no size threshold and no batching: spike A-002 measured a full
    80-keyframe run's redaction at 10.3 ms, which is 0.034% of a 30 s run
    against a 5% budget, so the cheapest arrangement to reason about is also
    fast enough.

    Redaction runs before the cap, because replacing a secret can lengthen the
    text and the cap is what bounds what a carrier holds.
    """
    result = redact_text(text)
    warnings.extend(_frozen_warning(item) for item in result.warnings)
    return result.text


def _freeze(
    value: Any,
    *,
    path: str,
    warnings: list[Mapping[str, str]],
    cap: bool,
    redact: bool,
) -> Any:
    """Copy `value` into an immutable equivalent, redacting and capping on the way.

    Copying rather than wrapping matters: a `MappingProxyType` over the
    caller's dict is a window the caller can still write through, and the
    mutation this guards against is exactly the one that happens *after* the
    redaction policy ran.

    A type this cannot freeze is refused rather than passed through, so a
    mutable value is a defect at construction instead of a hole at the writer.

    Only the *values* inside an extracted-text region are redacted, never the
    mapping keys: a key is the schema's word for a field, chosen by Distill,
    and is not extracted text.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        text = _redact(value, warnings=warnings) if redact else value
        return _cap_extracted_text(text, path=path, warnings=warnings) if cap else text
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze(
                    item, path=f"{path}.{key}", warnings=warnings, cap=cap, redact=redact
                )
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence):
        return tuple(
            _freeze(item, path=f"{path}[{index}]", warnings=warnings, cap=cap, redact=redact)
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} holds {type(value).__name__}, which a carrier cannot freeze")


def _thaw(value: Any) -> Any:
    """Rebuild `value` as plain, JSON-encodable, mutable Python.

    The carrier's own document is immutable and `json` cannot encode a
    `MappingProxyType`, so this is the step that makes a document writable -
    and the copy is what stops a caller writing back into the carrier through
    the document it was handed.
    """
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, kw_only=True)
class Carrier(ABC):
    """What every carrier of **extracted text** has in common.

    Two things: the recorded state of the **redaction** policy, and the
    **warnings** that qualify what it holds. Subclasses declare which of their
    fields are extracted-text regions; every string inside such a region is
    passed through the redaction policy (R-19) and capped individually (R-58),
    and everything, region or not, is frozen.

    Construction is the **redaction sink** (D-019). A carrier therefore cannot
    hold unredacted extracted text at all, except under the one policy that
    says so - `DISABLED`, which is the `--no-redact-secrets` opt-out and is
    recorded as such. What this base still does not do is decide whether the
    text is *safe*: it applies a policy of patterns, and a secret shaped like
    nothing in `redact_secrets.SECRET_PATTERNS` passes through it untouched.
    """

    EXTRACTED_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ()

    redaction: RedactionState = RedactionState.NOT_APPLIED
    warnings: tuple[Mapping[str, str], ...] = ()

    def __post_init__(self) -> None:
        # A region named here that is not a field would read as "capped" and
        # behave as "not capped", which is the one way R-58 could be quietly
        # absent from a carrier. Checked rather than trusted, because the cost
        # of the mistake is unbounded extracted text nobody is told about.
        declared = {field.name for field in dataclasses.fields(self)}
        unknown = sorted(set(self.EXTRACTED_TEXT_FIELDS) - declared)
        if unknown:
            raise AssertionError(
                f"{type(self).__name__} names extracted-text fields it does not have: {unknown}"
            )
        # The policy runs unless it was explicitly disabled, on every route in -
        # the literal constructor and `dataclasses.replace` alike, since replace
        # re-runs this. A carrier rebuilt around new text is redacted again
        # rather than inheriting the state its predecessor earned.
        #
        # Its predecessor's **warnings** do carry forward, because replace
        # passes them in and nothing here can tell a warning the caller raised
        # from one an earlier incarnation produced. A carrier rebuilt around
        # *shorter* text therefore keeps a truncation warning that no longer
        # describes what it holds. That is a stale record rather than a leak,
        # and the alternative - dropping warnings a producer deliberately
        # attached - loses a **degradation** nobody would then be told about.
        apply_policy = self.redaction is not RedactionState.DISABLED
        collected: list[Mapping[str, str]] = []
        for field in dataclasses.fields(self):
            if field.name in {"warnings", "redaction"}:
                continue
            region = field.name in self.EXTRACTED_TEXT_FIELDS
            object.__setattr__(
                self,
                field.name,
                _freeze(
                    getattr(self, field.name),
                    path=field.name,
                    warnings=collected,
                    cap=region,
                    redact=region and apply_policy,
                ),
            )
        object.__setattr__(
            self,
            "redaction",
            RedactionState.APPLIED if apply_policy else RedactionState.DISABLED,
        )
        object.__setattr__(
            self,
            "warnings",
            (*(_frozen_warning(item) for item in self.warnings), *collected),
        )

    @abstractmethod
    def _payload(self) -> Mapping[str, Any]:
        """This carrier's document, still immutable and not yet checked.

        Deliberately private and deliberately unencodable: `serialize` is the
        only supported way out, and a caller who reaches past it gets a
        structure `json` refuses rather than a silently mangled bundle file.
        """


@dataclass(frozen=True, kw_only=True)
class FrameArtifact(Carrier):
    """A **keyframe** and everything Distill derived from it.

    `extracted_text` is what the image-text reader recovered; `interpretation`
    is the vision model's structured reading, which echoes the screen back and
    is therefore extracted text in every field it has. `grounding` is Distill's
    own assessment of the two, so it is frozen but not capped.
    """

    EXTRACTED_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("extracted_text", "interpretation")

    index: int
    timestamp_sec: float
    path: str
    relative_path: str
    phash: str = ""
    source_candidate_index: int = -1
    extracted_text: str = ""
    interpretation: Mapping[str, Any] | None = None
    grounding: Mapping[str, Any] | None = None

    def _payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "index": self.index,
                "timestamp_sec": self.timestamp_sec,
                "path": self.path,
                "relative_path": self.relative_path,
                "phash": self.phash,
                "source_candidate_index": self.source_candidate_index,
                "extracted_text": self.extracted_text,
                "interpretation": self.interpretation,
                "grounding": self.grounding,
                "redaction": self.redaction,
                "warnings": self.warnings,
            }
        )


@dataclass(frozen=True, kw_only=True)
class Transcript(Carrier):
    """The timed, segmented speech recovered from a **source**'s audio.

    Every string under `segments` is extracted text - the segment text and the
    per-word text alike - so the whole region is capped, not just the field a
    render happens to print today.
    """

    EXTRACTED_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("segments",)

    language: str
    segments: tuple[Mapping[str, Any], ...]
    language_probability: float = 0.0

    def _payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "language": self.language,
                "language_probability": self.language_probability,
                "segments": self.segments,
                "redaction": self.redaction,
                "warnings": self.warnings,
            }
        )


@dataclass(frozen=True, kw_only=True)
class StageResult(Carrier):
    """One completed stage's output, kept so an interrupted run can **resume**.

    It is a carrier and not a plain dict because a stage result is durable: it
    is written under a **staging directory**, which makes it a **redaction
    sink** even though no reader is ever served it. Finding 4 is what happens
    when that is forgotten.

    `payload` is treated as an extracted-text region. Its contents are usually
    documents produced by carriers that were capped already, so the second pass
    normally finds nothing to do - it is here so that a stage recording
    something it built by hand is capped on the same terms.

    The schema version and **bundle key** are carried here; validating them on
    read is M4.3's, and this module does not do it.
    """

    EXTRACTED_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("payload",)

    schema_version: int
    bundle_key: str
    stage: str
    payload: Mapping[str, Any] = MappingProxyType({})

    def _payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "bundle_key": self.bundle_key,
                "stage": self.stage,
                "payload": self.payload,
                "redaction": self.redaction,
                "warnings": self.warnings,
            }
        )


def serialize(carrier: Carrier) -> dict[str, Any]:
    """Turn a carrier into the plain document a writer may make durable.

    This is where R-20 is enforced, and the reason it is enforced *here* is
    D-022: Python cannot carry the guarantee in a type. The layers, and what
    each one actually catches:

    - construction runs the policy (D-019), so the text a carrier holds is
      already redacted and there is no window in which it is not;
    - the emitter (Phase 5) is the choke point every durable write goes
      through, and catches a writer that forgets, but only a writer that uses
      it;
    - this check catches a carrier whose state says the policy never ran, at
      the last point before its text becomes durable and on every route into
      the serializer.

    Since construction cannot produce `NOT_APPLIED`, this check is defence in
    depth rather than the protection: what reaches it is a carrier assembled
    around `__post_init__`, and refusing that is worth more than trusting it.

    What it does *not* catch, precisely: this reads the state, so it catches a
    bypass that leaves the state saying the policy never ran - a write through
    `object.__setattr__`, or a subclass that skipped construction and did not
    set the field. A subclass that overrides `__post_init__` without calling
    `super()` *and* declares `APPLIED` serializes raw text, and nothing here
    can tell that from a carrier that earned it. Nor does any layer catch a
    policy that ran and missed: `APPLIED` means the patterns in
    `redact_secrets` were applied to every extracted-text region, not that
    every secret in that text matched one.
    """
    if not carrier.redaction.policy_applied:
        raise RedactionPolicyNotApplied(type(carrier).__name__)
    return _thaw(carrier._payload())


__all__ = [
    "EXTRACTED_TEXT_LIMIT_BYTES",
    "REDACTION_POLICY_NOT_APPLIED_CODE",
    "TRUNCATION_WARNING_CODE",
    "WARNING_STAGE",
    "Carrier",
    "FrameArtifact",
    "RedactionPolicyNotApplied",
    "RedactionState",
    "StageResult",
    "Transcript",
    "serialize",
]
