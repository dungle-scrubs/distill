"""Candidate keys: the bundle keys one **endpoint chain** could publish under.

Resolution has to know every key a run *might* serve before it asks any
endpoint anything, because a cache hit at any preference level means no network
at all. These tests are about that derivation and nothing else - no probing, no
selection, no cache.
"""

from __future__ import annotations

import pytest

from distill.local_vision import LocalVisionConfig
from distill.options import (
    VISION_MODE_CHAIN_EXHAUSTED,
    VISION_MODE_DISABLED,
    VISION_MODE_SELECTED,
    DistillOptions,
)
from distill.vision_chain import candidate_keys

LOCAL_A = LocalVisionConfig(model="qwen3-vl:8b", base_url="http://127.0.0.1:8000/v1")
LOCAL_B = LocalVisionConfig(model="qwen3-vl:32b", base_url="http://127.0.0.1:9000/v1")
REMOTE_A = LocalVisionConfig(
    model="qwen3-vl:8b", base_url="https://10.0.0.5/v1", allow_remote_endpoint=True
)


def test_one_key_per_entry_plus_the_exhausted_key() -> None:
    """Every outcome a run could reach gets a key, in preference order.

    The exhausted key is last because it is the outcome of having tried
    everything, and it is *present* because a degraded run still produces a
    bundle - one that must be servable on a later run without walking the whole
    chain again.
    """
    keys = candidate_keys(DistillOptions(), (LOCAL_A, LOCAL_B), "local")

    assert [key.entry for key in keys] == [0, 1, None]
    assert [key.vision_mode for key in keys] == [
        VISION_MODE_SELECTED,
        VISION_MODE_SELECTED,
        VISION_MODE_CHAIN_EXHAUSTED,
    ]
    # Distinct, because each names a different reader or a different outcome.
    assert len({key.opts_hash for key in keys}) == 3


def test_the_entry_keys_name_the_entry_that_would_produce_them() -> None:
    """A candidate key describes the reader, not the position.

    Swapping preference order must not change which key a given endpoint
    publishes under: a bundle produced by the 8b model is the same bundle
    whether that endpoint was tried first or second, and a chain reordered
    between runs must still hit it.
    """
    forward = candidate_keys(DistillOptions(), (LOCAL_A, LOCAL_B), "local")
    reversed_chain = candidate_keys(DistillOptions(), (LOCAL_B, LOCAL_A), "local")

    assert forward[0].opts_hash == reversed_chain[1].opts_hash
    assert forward[1].opts_hash == reversed_chain[0].opts_hash
    # And the exhausted key does not depend on the order either.
    assert forward[-1].opts_hash == reversed_chain[-1].opts_hash


def test_remoteness_is_part_of_the_key_but_the_address_is_not() -> None:
    """ADR-0004 and D-012 together: which reader, and whether it was remote.

    `REMOTE_A` serves the same model as `LOCAL_A`, so if remoteness were not in
    the key they would collide - which is exactly the collision M1.1 refuses to
    let an operator configure. Moving a *local* endpoint's address changes
    nothing, because the address is not identity.
    """
    local, remote = candidate_keys(DistillOptions(), (LOCAL_A, REMOTE_A), "local")[:2]
    assert local.opts_hash != remote.opts_hash

    moved = LocalVisionConfig(model=LOCAL_A.model, base_url="http://127.0.0.1:8100/v1")
    assert candidate_keys(DistillOptions(), (moved,), "local")[0].opts_hash == local.opts_hash


def test_a_disabled_run_has_one_key_and_no_chain_to_walk() -> None:
    """`--no-caption-frames` is not an outcome of trying; it is not trying.

    So there is one key and it is neither a selection nor an exhaustion. Giving
    a disabled run the exhausted key would let a deliberate opt-out be served a
    degraded bundle, which is the confusion P3-D-012 exists to prevent.
    """
    keys = candidate_keys(DistillOptions(caption_frames=False), (LOCAL_A, LOCAL_B), "local")

    assert [key.entry for key in keys] == [None]
    assert keys[0].vision_mode == VISION_MODE_DISABLED


def test_derivation_touches_no_network_and_no_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 1 of resolution scans keys before anything is asked or opened.

    Asserted by making both impossible rather than by inspection: a derivation
    that reached for either would raise here, and a later change that quietly
    added a probe or a stat would fail this test rather than slow every run
    down invisibly.
    """
    import socket

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("candidate-key derivation must be pure computation")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr("pathlib.Path.open", forbidden)
    monkeypatch.setattr("pathlib.Path.exists", forbidden)

    keys = candidate_keys(DistillOptions(), (LOCAL_A, REMOTE_A), "local")

    assert len(keys) == 3
