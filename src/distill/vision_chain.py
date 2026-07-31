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
