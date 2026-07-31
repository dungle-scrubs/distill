"""Chain resolution: turning a configured **endpoint chain** into one outcome.

This module owns the **candidate keys** a run could publish under, and (later)
the walk that picks between them. It consumes `options.py` for identity and
`local_vision.py` for the shape of an endpoint.

It does not talk to an endpoint. Nothing here opens a socket, reads a file, or
asks whether a server is up: deriving what a run *might* serve has to happen
before anything is asked, because a cache hit at any preference level means the
network is never touched at all. Availability is `local_vision.py`'s to answer
and the walk's to consume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from .local_vision import LocalVisionConfig
from .options import (
    VISION_MODE_CHAIN_EXHAUSTED,
    VISION_MODE_DISABLED,
    VISION_MODE_SELECTED,
    DistillOptions,
)


@dataclass(frozen=True)
class CandidateKey:
    """One **bundle key** a run could publish under, and what it would mean.

    `entry` is the index in the **endpoint chain** whose reading this key
    describes, or `None` when no reader produced it - a disabled run or an
    exhausted chain. It is positional information *about the run*, not part of
    identity: `opts_hash` is derived from the reader and the outcome, so the
    same endpoint keeps its key when the chain is reordered.
    """

    entry: int | None
    vision_mode: str
    opts_hash: str


def candidate_keys(
    options: DistillOptions,
    chain: tuple[LocalVisionConfig, ...],
    source_type: str,
) -> tuple[CandidateKey, ...]:
    """Every key this run could serve, in preference order.

    One per entry, describing the bundle that entry would produce, then the
    exhausted key. The exhausted key is last because it is the outcome of
    having tried everything, and it exists at all because a degraded run still
    produces a bundle - one a later run must be able to serve without walking
    the whole chain again.

    A disabled run has exactly one key and no chain to walk. It is not an
    outcome of trying, so it is neither a selection nor an exhaustion; giving it
    the exhausted key would let a deliberate `--no-caption-frames` be served a
    degraded bundle (P3-D-012).

    Derived by re-hashing `options` with the entry's identity-bearing fields
    substituted, rather than by assembling a payload here. Identity is
    `options.py`'s to define, and a second place that knew how to build it would
    be a second place to get it wrong - the failure ADR-0003's signature exists
    to catch late and this avoids entirely.
    """
    if not options.caption_frames:
        return (
            CandidateKey(
                entry=None,
                vision_mode=VISION_MODE_DISABLED,
                opts_hash=replace(options, vision_mode=VISION_MODE_DISABLED).opts_hash(source_type),
            ),
        )
    keys = [
        CandidateKey(
            entry=index,
            vision_mode=VISION_MODE_SELECTED,
            opts_hash=_as_entry(options, entry).opts_hash(source_type),
        )
        for index, entry in enumerate(chain)
    ]
    keys.append(
        CandidateKey(
            entry=None,
            vision_mode=VISION_MODE_CHAIN_EXHAUSTED,
            opts_hash=replace(options, vision_mode=VISION_MODE_CHAIN_EXHAUSTED).opts_hash(
                source_type
            ),
        )
    )
    return tuple(keys)


@dataclass(frozen=True)
class EntryOutcome:
    """What happened to one entry during the walk, for the operator to read.

    <!-- P3-D-018 --> An entry passed over is a fact the run has to be able to
    explain: "we used the second endpoint" is not actionable without "because
    the first answered with the wrong model". Recorded per entry and correlated
    to its index, which is the only stable name an entry has - the address is a
    **machine-local claim** and never enters a **bundle**.
    """

    entry: int
    outcome: str
    detail: str = ""


CACHE_HIT = "cache_hit"
"""A candidate key was already on disk, so nothing was asked of any endpoint."""

SELECTED = "selected"
"""This endpoint answered and will produce the run's **interpretations**."""

UNAVAILABLE = "unavailable"
"""This endpoint was asked and could not serve - unreachable, unauthenticated,
or serving a different model. The ordinary case an **endpoint chain** absorbs."""


@dataclass(frozen=True)
class ResolvedRun:
    """The one outcome resolution settles on, and the evidence for it.

    <!-- P3-D-015 --> Every consumer reads the selection from this object rather
    than deriving its own. The partial implementation that sets a bundle key and
    stops leaves the options holding the *previous* entry's fields, so the
    pipeline calls one endpoint while the manifest records another's options
    under a third's key - three artifacts, each internally consistent, agreeing
    on a lie.
    """

    entry: int | None
    vision_mode: str
    opts_hash: str
    endpoint: LocalVisionConfig | None
    served_from_cache: bool
    evidence: tuple[EntryOutcome, ...] = ()


def resolve_chain(
    options: DistillOptions,
    chain: tuple[LocalVisionConfig, ...],
    source_type: str,
    *,
    cached: Callable[[str], bool],
    probe: Callable[[LocalVisionConfig], bool],
) -> ResolvedRun:
    """The one endpoint - or the one cached bundle - this run will use.

    Two phases, and the order is the whole point. <!-- P3-D-011 --> Phase 1
    scans every candidate key against the cache and asks nothing of the network:
    if any of them is already on disk the run has its answer, and an offline
    machine can still serve a bundle it already has. Probing first and consulting
    the cache second would make reachability a precondition for reading
    something already written.

    A hit *below* the top of the chain is still a hit. The preference order says
    which endpoint to ask when work must be done; it does not say that a bundle
    produced by a less-preferred reader should be rebuilt by a better one.

    `cached` and `probe` are injected rather than reached for, so the walk can be
    tested without a store or a server - and so this module keeps its promise to
    talk to no endpoint itself.
    """
    keys = candidate_keys(options, chain, source_type)
    if not options.caption_frames:
        # No reader was asked for, so none is asked. The single key is neither a
        # selection nor an exhaustion, and probing here would be a network call
        # made on behalf of an operator who said not to.
        disabled = keys[0]
        return ResolvedRun(
            entry=None,
            vision_mode=disabled.vision_mode,
            opts_hash=disabled.opts_hash,
            endpoint=None,
            served_from_cache=False,
        )
    for key in keys if not options.force_reprocess else ():
        # Phase 1 is skipped entirely under `--force-reprocess` rather than
        # consulted and ignored: a run told to reprocess must reach the
        # endpoints, and a scan whose result is discarded is one that can still
        # pick the wrong answer if the skip is ever made conditional.
        if cached(key.opts_hash):
            return ResolvedRun(
                entry=key.entry,
                vision_mode=key.vision_mode,
                opts_hash=key.opts_hash,
                endpoint=None if key.entry is None else chain[key.entry],
                served_from_cache=True,
                evidence=(
                    () if key.entry is None else (EntryOutcome(entry=key.entry, outcome=CACHE_HIT),)
                ),
            )
    # Phase 2: nothing on disk, so the endpoints are asked - in preference
    # order, and only now.
    evidence: list[EntryOutcome] = []
    for index, entry in enumerate(chain):
        if probe(entry):
            evidence.append(EntryOutcome(entry=index, outcome=SELECTED))
            selected = keys[index]
            return ResolvedRun(
                entry=index,
                vision_mode=selected.vision_mode,
                opts_hash=selected.opts_hash,
                endpoint=entry,
                served_from_cache=False,
                evidence=tuple(evidence),
            )
        evidence.append(EntryOutcome(entry=index, outcome=UNAVAILABLE))
    # Every endpoint was asked and none could serve. That is **degradation**,
    # and it publishes under the exhausted key rather than the disabled one:
    # the operator asked for vision and did not get it, which is a different
    # bundle from one where vision was never wanted (P3-D-012).
    exhausted = keys[-1]
    return ResolvedRun(
        entry=None,
        vision_mode=exhausted.vision_mode,
        opts_hash=exhausted.opts_hash,
        endpoint=None,
        served_from_cache=False,
        evidence=tuple(evidence),
    )


def _as_entry(options: DistillOptions, entry: LocalVisionConfig) -> DistillOptions:
    """`options` as it would be had this entry produced the run.

    Only the identity-bearing fields move. The address moves too, but not
    because it is identity - it is not (ADR-0004) - it moves because
    `local_vision_non_local` is derived from it, and that boolean *is* identity
    (D-012). Substituting the model without the address would ask whether the
    *previous* endpoint was remote while naming this one's model.
    """
    return replace(
        options,
        vision_mode=VISION_MODE_SELECTED,
        local_vision_model=entry.model,
        local_vision_base_url=entry.base_url,
        local_vision_allow_remote_endpoint=entry.allow_remote_endpoint,
    )
