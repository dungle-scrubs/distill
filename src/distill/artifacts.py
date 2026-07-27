"""The carriers **extracted text** travels in, and the one serializer that lets it out.

This module owns the shape of a **frame artifact**, a **transcript** and a
**stage result** in memory: their fields, the recorded state of the
**redaction** policy that produced them, the per-field 256 KiB cap on extracted
text (R-58), the immutability of everything nested inside them, and the runtime
refusal to serialize a carrier whose redaction policy was never applied (R-20).

It does not own the redaction policy itself - that is `redact_secrets.py` - nor
when the policy runs, nor any file write, nor the markdown **render**, nor the
delimiting that keeps extracted text data rather than instruction. It imports
nothing from Distill except the error and warning vocabulary, deliberately: a
carrier is what the producing stages hand to the writers, so it must not depend
on either side.
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


def _freeze(
    value: Any,
    *,
    path: str,
    warnings: list[Mapping[str, str]],
    cap: bool,
) -> Any:
    """Copy `value` into an immutable equivalent, capping extracted text on the way.

    Copying rather than wrapping matters: a `MappingProxyType` over the
    caller's dict is a window the caller can still write through, and the
    mutation this guards against is exactly the one that happens *after* the
    redaction policy ran.

    A type this cannot freeze is refused rather than passed through, so a
    mutable value is a defect at construction instead of a hole at the writer.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _cap_extracted_text(value, path=path, warnings=warnings) if cap else value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze(item, path=f"{path}.{key}", warnings=warnings, cap=cap)
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence):
        return tuple(
            _freeze(item, path=f"{path}[{index}]", warnings=warnings, cap=cap)
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
    capped individually (R-58) and everything, region or not, is frozen.

    What this base does not do is decide whether the text is safe. It records
    what a producer says it did and refuses to serialize when the producer says
    nothing at all.
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
        collected: list[Mapping[str, str]] = []
        for field in dataclasses.fields(self):
            if field.name == "warnings":
                continue
            object.__setattr__(
                self,
                field.name,
                _freeze(
                    getattr(self, field.name),
                    path=field.name,
                    warnings=collected,
                    cap=field.name in self.EXTRACTED_TEXT_FIELDS,
                ),
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

    - the type states the intent and holds the state, and catches nothing on
      its own - a producer can construct a carrier whose policy never ran, and
      must be able to, because the carrier is where the text is held while the
      policy runs;
    - the emitter (Phase 5) is the choke point every durable write goes
      through, and catches a writer that forgets, but only a writer that uses
      it;
    - this check catches the carrier at the last point before its text becomes
      durable, on every route into the serializer.

    What none of them catch: a producer that declares `APPLIED` or `DISABLED`
    without running the policy. The state is a claim by the producing stage,
    and this function verifies that a claim was made, not that it was true.
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
