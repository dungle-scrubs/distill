"""The carriers **extracted text** travels in, and the one serializer that lets it out.

This module owns the shape of a **frame artifact**, a **transcript** and a
**stage result** in memory: their fields, the field names of the
**interpretation** nested inside a frame artifact, *when* the **redaction**
policy runs over them and the state that records it, the per-field 256 KiB cap
on extracted text (R-58), the immutability of everything nested inside them, and
the runtime refusal to serialize a carrier whose redaction policy was never
applied (R-20).

Owning those names is what makes it the *only* description of a frame: before
M4.4, `frame_selection`, `ocr`, `local_vision`, `render` and `bundle_store` each
restated the schema as string keys on a bare dict, which is how the redaction
policy came to run somewhere other than where the text entered (finding 4).
Producing stages now hand each other carriers, and a reader asks the carrier.

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
from functools import cache
from types import MappingProxyType, NoneType, UnionType
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints

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


@dataclass(frozen=True, slots=True)
class _PolicyText:
    """Text the **redaction** policy has already had its turn over.

    A type rather than a comment, because the cap and the policy have now been
    written in the wrong order twice - once in `link_label`, once here. A cut
    applied first decides what the policy is allowed to see: a value straddling
    the cut is sliced into a prefix that matches no pattern, and that prefix is
    then durable in full. Inverting the two lines in `_freeze` published twelve
    characters of a live key into a **manifest** and passed the whole suite.

    `_redact` is what produces one and `_cap_extracted_text` is what consumes
    one, so the inversion no longer type-checks and no longer runs - the cap
    asked for a `str` before and asks for a step that has happened now. What it
    does not claim is anything about the text itself: the policy is patterns,
    and a secret shaped like none of them passes through untouched.
    """

    text: str

    @classmethod
    def policy_did_not_run(cls, text: str) -> _PolicyText:
        """Text the cap may still take, although no policy ran over it.

        Correct in exactly two cases, both already decided by the caller: the
        field is not an extracted-text region, or the run is under
        `--no-redact-secrets` - which D-020 records as an applied policy rather
        than an absent one. Named so the one route past the policy is legible
        instead of implicit.
        """
        return cls(text)


def _cap_extracted_text(text: _PolicyText, *, path: str, warnings: list[Mapping[str, str]]) -> str:
    """Return `text` within the per-field cap, recording a **warning** if it was cut.

    The budget is bytes, because KiB is a byte unit and a character is not a
    fixed number of them; the cut lands on a character boundary, so truncating
    cannot produce text that is not valid UTF-8.

    Nothing is appended to mark the truncation. A marker would be Distill's own
    words placed inside extracted text, which is exactly what the
    untrusted-data boundary exists to prevent; the warning carried with the
    bundle is the record.
    """
    encoded = text.text.encode("utf-8")
    if len(encoded) <= EXTRACTED_TEXT_LIMIT_BYTES:
        return text.text
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


def _redact(text: str, *, warnings: list[Mapping[str, str]]) -> _PolicyText:
    """Return `text` with secret-shaped values replaced, keeping the policy's warnings.

    Every string in an extracted-text region goes through this, every time,
    with no size threshold and no batching: spike A-002 measured a full
    80-keyframe run's redaction at 10.3 ms, which is 0.034% of a 30 s run
    against a 5% budget, so the cheapest arrangement to reason about is also
    fast enough.

    Redaction runs before the cap, because replacing a secret can lengthen the
    text and the cap is what bounds what a carrier holds - and because a cap
    applied first hides a straddling secret from the policy. `_PolicyText` is
    what makes that order the only one that compiles.
    """
    result = redact_text(text)
    warnings.extend(_frozen_warning(item) for item in result.warnings)
    return _PolicyText(result.text)


def is_path_field(key: str) -> bool:
    """Whether a field name is one of Distill's names for a filesystem path.

    A path Distill built is Distill's own words - an output root the user named,
    a **bundle key** it computed, a `frame_%04d.png` it chose - so it is not
    **extracted text** and the redaction policy has no business rewriting it.
    Left in, it was: a bundle key is 64 hex characters, which is exactly what a
    generic secret pattern matches, so a **stage result** recorded the frame's
    image at a path whose middle had become `[REDACTED]`, and the **resume**
    that read it back rejected the whole document as naming a path outside the
    bundle root. Every stage after the transcript was silently recomputed.

    Stated here rather than at each site, because "which key names a path" is
    one rule with two readers: this one decides what the policy may rewrite, and
    `bundle_store` decides what must stay inside the bundle root (R-23). Two
    copies of it are two answers about the same field.

    Precisely what this does not do: recognize a path recorded under a name that
    does not say "path". Such a field is treated as extracted text - which is
    the safe direction, since the cost is a mangled path rather than a durable
    secret.
    """
    return key == "path" or key.endswith("_path")


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
    and is not extracted text. Nor is the value under a key that names a path -
    see `is_path_field`, which is why a mapping's path fields are frozen like
    everything else and passed through the policy like nothing else.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        text = (
            _redact(value, warnings=warnings)
            if redact
            else _PolicyText.policy_did_not_run(value)
        )
        return _cap_extracted_text(text, path=path, warnings=warnings) if cap else text.text
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze(
                    item,
                    path=f"{path}.{key}",
                    warnings=warnings,
                    cap=cap and not is_path_field(str(key)),
                    redact=redact and not is_path_field(str(key)),
                )
                for key, item in value.items()
            }
        )
    if isinstance(value, bytes | bytearray | memoryview):
        # Refused ahead of the Sequence branch, which every byte-shaped type
        # satisfies. Frozen there, each byte becomes an `int`: a value no
        # pattern ran over, no cap measured, and `json` encodes happily - so it
        # is durable, unredacted and indistinguishable from a list of numbers a
        # stage meant to record. Refusing is what the docstring above promises,
        # and this is what makes it true of bytes as well (finding 6).
        raise TypeError(f"{path} holds {type(value).__name__}, which a carrier cannot freeze")
    if isinstance(value, Sequence):
        return tuple(
            _freeze(item, path=f"{path}[{index}]", warnings=warnings, cap=cap, redact=redact)
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} holds {type(value).__name__}, which a carrier cannot freeze")


@cache
def _declared_types(carrier: type) -> Mapping[str, Any]:
    """The resolved annotations of `carrier`'s fields.

    Cached per class: the annotations are written once and `from __future__ import
    annotations` makes them strings, so resolving them is a per-class cost rather
    than a per-frame one.
    """
    return get_type_hints(carrier)


def _fits_declaration(value: Any, declared: Any) -> bool:
    """Whether `value` is something the field declaring `declared` may hold.

    Deliberately shallow: it answers the shape of the field, not of everything
    nested inside it. A `segments` that must hold mappings is checked to hold
    mappings, and what is *in* a segment stays `transcript.py`'s (D-041).

    Two allowances, both about documents rather than about types. A field
    declared `float` takes an `int`, because JSON writes `0` for a timestamp
    that happens to be whole; and a field declared as a tuple takes a list,
    because that is what JSON gives back for one. Neither reaches the carrier's
    own state - `_freeze` copies both into the immutable form the declaration
    asks for.

    `bool` is refused where an `int` is declared, for the reason
    `validated_stage_payload` refuses it in a schema version: `True` is an
    instance of `int` and equals 1, so a document whose `index` is a boolean
    would otherwise be a frame numbered 1.
    """
    origin = get_origin(declared)
    if origin is UnionType or origin is Union:
        return any(_fits_declaration(value, arm) for arm in get_args(declared))
    if declared is Any:
        return True
    if declared is NoneType:
        return value is None
    if origin is not None:
        if isinstance(origin, type) and issubclass(origin, Mapping):
            return isinstance(value, Mapping)
        if isinstance(origin, type) and issubclass(origin, Sequence):
            if isinstance(value, str) or not isinstance(value, list | tuple):
                return False
            arms = [arm for arm in get_args(declared) if arm is not Ellipsis]
            return not arms or all(_fits_declaration(item, arms[0]) for item in value)
        return isinstance(value, origin)
    if declared is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if declared is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(declared, type):
        return isinstance(value, declared)
    return True


def _refuse_undeclared_values(instance: Any) -> None:
    """Refuse a value a field's own declaration does not allow.

    Construction is where a document stops being a document, so it is where the
    fields it claims have to be the fields this class declares. A **resume**
    rebuilds carriers from JSON another process wrote (R-23), and a dataclass
    checks nothing: a `relative_path` of `42` built a `FrameArtifact` that
    survived freezing, redaction and the write, and then ended the run in
    `render` on `42.strip()` - an `E_INTERNAL` over the run's own scratch, which
    is the one thing a stage result must never cost (D-030, D-046).

    Stated against the annotations rather than a second list of expected types,
    so a field added later is covered by having been declared. Raised as
    `TypeError` because that is what a caller rebuilding a carrier from a
    document already handles: the pipeline's revivers turn it into a stage miss
    and recompute, which is what an unusable document is worth.
    """
    declared = _declared_types(type(instance))
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        if not _fits_declaration(value, declared[field.name]):
            raise TypeError(
                f"{type(instance).__name__}.{field.name} holds "
                f"{type(value).__name__}, which its declaration does not allow"
            )


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
        # Before anything is frozen, because a value the declaration does not
        # allow must not become part of the carrier's state at all - and
        # because freezing accepts far more than a field does: `_freeze` copies
        # any scalar, so an `int` where a path belongs survives it intact.
        _refuse_undeclared_values(self)
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

    def _raised_since(self, previous: Carrier) -> list[dict[str, str]]:
        """The **warnings** this carrier's construction added to the inherited ones.

        A carrier built from another one carries its predecessor's warnings
        forward - `__post_init__` appends to them and never removes - so the
        difference is what *this* construction had to say. Stages report the
        difference, because a stage that reported the whole list would report
        an earlier stage's truncation again every time it touched the frame,
        and a **manifest** would count one lost field as several.
        """
        return [dict(item) for item in self.warnings[len(previous.warnings) :]]

    @abstractmethod
    def _payload(self) -> Mapping[str, Any]:
        """This carrier's document, still immutable and not yet checked.

        Deliberately private and deliberately unencodable: `serialize` is the
        only supported way out, and a caller who reaches past it gets a
        structure `json` refuses rather than a silently mangled bundle file.
        """


@dataclass(frozen=True)
class Interpretation:
    """The vision model's structured reading of one **keyframe**.

    Defined here rather than beside the backend that produces it because an
    **interpretation** is part of a **frame artifact** (CONTEXT.md: a frame
    artifact has one keyframe, at most one interpretation, and one grounding).
    One definition of these field names is the point: `local_vision` fills them
    in, `render` reads them out, and neither spells them as string keys.

    Read-only by the time anybody but its producer sees it. The **redaction**
    policy runs over `document()`'s output when it enters `FrameArtifact`, so
    an interpretation reconstructed by `from_document` holds the text the
    policy left, not the text the model produced.

    It is not a carrier: it holds no policy state and never reaches a writer on
    its own. The frame artifact it belongs to is what makes it durable, and
    that is where the policy runs (D-019).

    `SUBSTANTIVE_FIELDS` names the fields that carry what the model saw, as
    opposed to the ones that describe the reading (`frame_kind`,
    `text_confidence`, `uncertainty`) or the backend that produced it. It lives
    here for the reason the field names do: the module that fills a reading in
    decides whether a response was a reading at all (R-39), and it should not
    have to restate which fields that question is about.
    """

    SUBSTANTIVE_FIELDS: ClassVar[tuple[str, ...]] = (
        "visual_summary",
        "detected_elements",
        "interpretation",
        "verbatim_text",
    )

    visual_summary: str = ""
    detected_elements: tuple[str, ...] = ()
    interpretation: str = ""
    uncertainty: str = ""
    backend: str = ""
    model: str = ""
    prompt_profile: str = ""
    frame_kind: str = ""
    verbatim_text: str = ""
    text_confidence: str = "none"

    def __post_init__(self) -> None:
        # On the same terms as a carrier, and for the same reason: a reading is
        # rebuilt from the mapping a frame artifact froze, which a **resume**
        # read back off disk. A `visual_summary` of `42` reaches `render` as
        # `42.strip()` otherwise.
        _refuse_undeclared_values(self)

    @property
    def has_interpretation(self) -> bool:
        """Whether the model said anything about the frame beyond reading it."""
        return bool(self.interpretation.strip() or self.detected_elements)

    @property
    def carries_a_reading(self) -> bool:
        """Whether this says anything about the **keyframe** at all (R-39)."""
        return document_carries_a_reading(self.document())

    def document(self) -> dict[str, Any]:
        """This reading as the plain mapping a **frame artifact** carries."""
        return {
            **dataclasses.asdict(self),
            "detected_elements": list(self.detected_elements),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any] | None) -> Interpretation | None:
        """Rebuild a reading from a frame artifact's mapping, or `None` if absent.

        Tolerant on the way in, because the mapping may have been read back off
        disk by a **resume** or out of a **manifest** written by an older
        Distill: an unknown key is dropped and a missing one takes its default,
        so a document that has drifted costs the fields that drifted rather
        than the frame.
        """
        if document is None:
            return None
        fields = {field.name for field in dataclasses.fields(cls)}
        values = {key: value for key, value in document.items() if key in fields}
        elements = values.get("detected_elements", ())
        values["detected_elements"] = tuple(str(item) for item in elements)
        return cls(**values)


def document_carries_a_reading(document: Mapping[str, Any] | None) -> bool:
    """Whether a mapping says anything about the **keyframe** (R-39).

    True when at least one of `Interpretation.SUBSTANTIVE_FIELDS` holds text.
    Emptiness is judged the way a reading judges it - stripped - so a mapping
    whose substantive fields are all blank, whitespace, or an empty list is as
    empty as `{}`, and so is one holding nothing but the metadata that
    describes a reading.

    A field counts only in the shape it is declared with. Anything else is a
    value `Interpretation` would drop or mangle rather than read: a
    `detected_elements` of `"axis"` is not a list and becomes no elements at
    all, and one of `[{"label": "axis"}]` would become the single element
    `"{'label': 'axis'}"`.

    A question about a mapping rather than about an `Interpretation`, because
    both callers have one and only one of them can afford to build a reading:
    `local_vision` asks it of what a model returned before there is a reading
    to ask, and `response` asks it of a **manifest** a **cache** hit read back,
    which is a document that may not rebuild at all. It is a function rather
    than a method for the reason the field names are here: the answer must not
    differ between the module that fills a reading in and the module that
    counts them.
    """
    if not isinstance(document, Mapping):
        return False
    return any(
        _substantive_text(name, document.get(name)) for name in Interpretation.SUBSTANTIVE_FIELDS
    )


def _substantive_text(name: str, value: Any) -> str:
    """The text one substantive field would contribute to a reading, if any."""
    if name == "detected_elements":
        # The one substantive field declared as a sequence of strings.
        if not isinstance(value, list | tuple):
            return ""
        return "".join(item.strip() for item in value if isinstance(item, str))
    return value.strip() if isinstance(value, str) else ""


@dataclass(frozen=True, kw_only=True)
class FrameArtifact(Carrier):
    """A **keyframe** and everything Distill derived from it.

    `extracted_text` is what the image-text reader recovered; `interpretation`
    is the vision model's structured reading, which echoes the screen back and
    is therefore extracted text in every field it has. `grounding` is Distill's
    own assessment of the two, so it is frozen but not capped.

    This is the only description of a frame's shape. `frame_selection` produces
    one, `ocr` and `local_vision` add to it, and `render` reads it; none of them
    names a field of it as a string key, which is what stops the schema from
    being restated in five modules and the **redaction** policy from running
    somewhere other than where the text enters (R-19, M4.4).

    Adding to a frame means building the next one: the carrier is frozen, so
    `with_extracted_text` and `with_interpretation` return a new artifact whose
    construction re-runs the policy over what was added. The policy itself is
    not re-chosen on the way - `redaction` travels with the frame from the
    moment `frame_selection` creates it, so `--no-redact-secrets` survives every
    stage without each stage being told about it separately.
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

    @property
    def reading(self) -> Interpretation | None:
        """The **interpretation** as a typed value, or `None` if there is not one.

        The field holds a mapping because that is what the policy caps and
        freezes; this is how a reader gets at it without knowing what is in it.
        """
        return Interpretation.from_document(self.interpretation)

    def with_extracted_text(self, text: str) -> tuple[FrameArtifact, list[dict[str, str]]]:
        """The same frame carrying what the image-text reader recovered from it.

        Returns the **warnings** the new frame's construction raised as well as
        the frame, and only those - the ones it inherited are already recorded
        against the stage that raised them. A stage that reported the whole list
        would report R-58's truncation of the image text again for every later
        stage that touched the frame.
        """
        added = dataclasses.replace(self, extracted_text=text)
        return added, added._raised_since(self)

    def with_interpretation(
        self,
        reading: Interpretation | None,
        *,
        grounding: Mapping[str, Any] | None = None,
    ) -> tuple[FrameArtifact, list[dict[str, str]]]:
        """The same frame carrying a vision reading and Distill's assessment of it.

        Both together, because a **grounding** is an assessment *of* an
        interpretation: recording one without the other is what lets an
        ungrounded reading be served as though nobody had checked it. A frame
        the model produced nothing for still takes a grounding, which is the
        case `reading=None` covers.

        Returns the warnings this construction raised, on the same terms as
        `with_extracted_text`.
        """
        added = dataclasses.replace(
            self,
            interpretation=None if reading is None else reading.document(),
            grounding=grounding,
        )
        return added, added._raised_since(self)

    def relocated(self, path: str) -> FrameArtifact:
        """The same frame, its **keyframe** image now at `path`.

        A **publish** renames the **staging directory**, so the absolute path a
        run recorded while staging is not where a reader will find the image.
        """
        return dataclasses.replace(self, path=path)

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, redaction: RedactionState
    ) -> FrameArtifact:
        """Rebuild a frame artifact from a document a **stage result** recorded.

        A **resume** reads its scratch back off disk as plain JSON, so the
        carriers a stage produced have to be reconstituted before the next stage
        can add to them.

        `redaction` is the *resuming run's* policy and is required, never the
        state the document claims. The document is something another process
        wrote, so its claims are input (R-23's premise): a forged stage result
        recording `disabled` would otherwise downgrade a run that asked for
        redaction, and the text it carries would be published raw. Taking the
        run's policy instead cannot be wrong, because `redact_secrets`
        participates in the **options hash** and so in the **bundle key** - a
        document under this bundle key was written by a run under this policy.

        Warnings recorded on the document come back with it, and this
        reconstruction adds none of its own in any reachable case: the text was
        already capped and redacted before it was written, so the policy running
        again over it changes nothing and has nothing to report.

        Unknown keys are dropped rather than refused - the document is scratch,
        and a field this Distill does not know about is one an older one wrote.
        A value a field's declaration does not allow is a different thing and is
        refused, by `__post_init__`, along with a nested **interpretation** that
        cannot become one: a frame promises its `reading`, so the refusal has to
        land where the document is being rebuilt - where the caller's answer is
        to recompute the stage - rather than later, in a render, where it is a
        run ending over its own scratch.

        What this does not reach is what a **grounding** or a **transcript**
        segment *holds*: those are `grounding.py`'s and `transcript.py`'s
        vocabulary, read by `render`, and hardening the render against the text
        it is given is R-24's, not a carrier's.
        """
        fields = {field.name for field in dataclasses.fields(cls)}
        values = {key: value for key, value in document.items() if key in fields}
        values["redaction"] = redaction
        values["warnings"] = tuple(values.get("warnings", ()))
        artifact = cls(**values)
        if artifact.interpretation is not None:
            Interpretation.from_document(artifact.interpretation)
        return artifact

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

    @classmethod
    def from_document(cls, document: Mapping[str, Any], *, redaction: RedactionState) -> Transcript:
        """Rebuild a transcript from a document a **stage result** recorded.

        The resume route, on the same terms as `FrameArtifact.from_document`,
        including that `redaction` is the resuming run's policy rather than the
        one the document claims for itself.

        What a segment holds is not this carrier's - `transcript.py` decides
        that - so segments are passed through as the documents they are and
        only frozen and capped here.
        """
        return cls(
            language=str(document.get("language", "")),
            language_probability=float(document.get("language_probability", 0.0)),
            segments=tuple(document.get("segments", ())),
            redaction=redaction,
            warnings=tuple(document.get("warnings", ())),
        )

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

    The schema version and **bundle key** are carried here so that whoever reads
    the document back can check them (R-23). Checking them is not this module's:
    `bundle_store` writes the carrier, reads it, and decides whether a
    **resume** may believe it - which it can only do because it is the one that
    knows which bundle is asking.
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
    "Interpretation",
    "RedactionPolicyNotApplied",
    "RedactionState",
    "StageResult",
    "Transcript",
    "is_path_field",
    "serialize",
]
