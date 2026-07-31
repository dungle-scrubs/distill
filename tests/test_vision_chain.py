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
from distill.vision_chain import (
    SELECTED,
    SKIPPED,
    UNAVAILABLE,
    EntryOutcome,
    candidate_keys,
    resolve_chain,
)

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


# --- M2.1: the resolution walk ---------------------------------------------


def never_probed(config: object) -> bool:
    """A probe that fails the test if resolution reaches the network."""
    raise AssertionError("a cache hit must not probe any endpoint")


def test_a_cache_hit_at_any_preference_level_touches_no_network() -> None:
    """<!-- P3-D-011 --> Phase 1 scans every candidate key before asking anything.

    The point of deriving all the keys up front: if any of them is already on
    disk, the run has its answer and no endpoint needs to be reachable. Probing
    first and consulting the cache second would make an offline machine unable
    to serve a bundle it already has.

    Asserted with a probe that raises rather than by counting calls, so the test
    fails loudly at the moment resolution reaches for the network instead of
    afterwards on a mismatch.
    """
    chain = (LOCAL_A, LOCAL_B)
    keys = candidate_keys(DistillOptions(), chain, "local")
    # The *second* entry's key is the cached one - a hit below the top of the
    # chain still means no probing, which is the case a "probe entry 0 first"
    # implementation would get wrong.
    cached = {keys[1].opts_hash}

    resolved = resolve_chain(
        DistillOptions(),
        chain,
        "local",
        cached=lambda key: 1 if key in cached else None,
        probe=never_probed,
    )

    assert resolved.served_from_cache is True
    assert resolved.entry == 1
    assert resolved.opts_hash == keys[1].opts_hash


def test_nothing_cached_selects_the_first_available_endpoint() -> None:
    """The ordinary cold run: preference order decides who is asked first."""
    resolved = resolve_chain(
        DistillOptions(),
        (LOCAL_A, LOCAL_B),
        "local",
        cached=lambda key: None,
        probe=lambda config: True,
    )

    assert resolved.entry == 0
    assert resolved.endpoint is LOCAL_A
    assert resolved.served_from_cache is False


def test_an_unavailable_entry_is_passed_over_and_the_reason_is_recorded() -> None:
    """<!-- P3-D-018 --> "we used the second endpoint" is not actionable alone.

    An operator who configured a cloud reader first and finds their bundles
    produced locally needs to know the cloud endpoint was asked and could not
    serve - which entry, and that it was asked rather than skipped. The index is
    what carries that: an endpoint's address is a **machine-local claim** and
    never enters a bundle.
    """
    resolved = resolve_chain(
        DistillOptions(),
        (LOCAL_A, LOCAL_B),
        "local",
        cached=lambda key: None,
        probe=lambda config: config is not LOCAL_A,
    )

    assert resolved.entry == 1
    assert resolved.endpoint is LOCAL_B
    assert resolved.evidence == (
        EntryOutcome(entry=0, outcome=UNAVAILABLE),
        EntryOutcome(entry=1, outcome=SELECTED),
    )


def test_every_endpoint_unavailable_and_nothing_cached_exhausts_the_chain() -> None:
    """<!-- P3-D-012 --> A walked-out chain still produces a bundle, under its
    own key.

    Not the disabled key: the operator asked for vision and did not get it,
    which is a different bundle from one where vision was never wanted. Every
    entry is recorded as asked, so the degradation is explicable rather than
    just visible.
    """
    resolved = resolve_chain(
        DistillOptions(),
        (LOCAL_A, LOCAL_B),
        "local",
        cached=lambda key: None,
        probe=lambda config: False,
    )

    assert resolved.entry is None
    assert resolved.endpoint is None
    assert resolved.vision_mode == VISION_MODE_CHAIN_EXHAUSTED
    assert [outcome.outcome for outcome in resolved.evidence] == [UNAVAILABLE, UNAVAILABLE]


def test_force_reprocess_bypasses_the_cache_and_probes_from_the_top() -> None:
    """`--force-reprocess` means do the work, not find a reason not to.

    Phase 1 is skipped entirely rather than consulted and ignored: a run told to
    reprocess must reach the endpoints, and a cache scan whose result is thrown
    away is a scan that can still pick the wrong answer if the skip is ever
    made conditional.
    """
    asked: list[str] = []

    def probe(config: LocalVisionConfig) -> bool:
        asked.append(config.model)
        return True

    resolved = resolve_chain(
        DistillOptions(force_reprocess=True),
        (LOCAL_A, LOCAL_B),
        "local",
        cached=lambda key: 1,
        probe=probe,
    )

    assert resolved.served_from_cache is False
    assert resolved.entry == 0
    assert asked == [LOCAL_A.model]


def test_a_disabled_run_is_settled_without_probing_anything() -> None:
    """`--no-caption-frames` asks for no reader, so none is asked for.

    Its one key is neither a selection nor an exhaustion, and a probe here would
    be a network call made on behalf of an operator who said not to.
    """
    resolved = resolve_chain(
        DistillOptions(caption_frames=False),
        (LOCAL_A, LOCAL_B),
        "local",
        cached=lambda key: None,
        probe=never_probed,
    )

    assert resolved.entry is None
    assert resolved.endpoint is None
    assert resolved.vision_mode == VISION_MODE_DISABLED
    assert resolved.evidence == ()


def test_a_generation_with_no_interpretations_under_a_selected_key_is_a_miss() -> None:
    """<!-- P3-D-019 --> A `selected` key promises a reader's work is in there.

    A generation under that key holding zero **interpretations** is not the
    bundle the key describes - it is the residue of a run that selected an
    endpoint and then got nothing out of it. Serving it would answer a request
    for a vision reading with a bundle that has none, forever, because the key
    would keep hitting.

    Zero is only meaningful under a `selected` key. A disabled or exhausted run
    is *expected* to hold none, which is why the rule is per vision mode rather
    than a blanket "generations must be non-empty".
    """
    chain = (LOCAL_A, LOCAL_B)
    keys = candidate_keys(DistillOptions(), chain, "local")
    empty = {keys[0].opts_hash: 0, keys[1].opts_hash: 3}

    resolved = resolve_chain(
        DistillOptions(),
        chain,
        "local",
        cached=lambda key: empty.get(key),
        probe=never_probed,
    )

    # Entry 0's generation is empty, so the walk passes it over and serves
    # entry 1's - still without probing.
    assert resolved.entry == 1
    assert resolved.served_from_cache is True


def test_an_exhausted_bundle_is_served_rather_than_walked_again() -> None:
    """A degraded run's bundle is a bundle, and its key is meant to hit.

    Without this the machine that could not reach any endpoint walks the whole
    chain again on every run, paying every timeout, to arrive at the reading it
    already has on disk.
    """
    chain = (LOCAL_A, LOCAL_B)
    keys = candidate_keys(DistillOptions(), chain, "local")

    resolved = resolve_chain(
        DistillOptions(),
        chain,
        "local",
        cached=lambda key: 0 if key == keys[-1].opts_hash else None,
        probe=never_probed,
    )

    assert resolved.entry is None
    assert resolved.vision_mode == VISION_MODE_CHAIN_EXHAUSTED
    assert resolved.served_from_cache is True


def test_an_offline_machine_serves_the_local_reading_rather_than_ocr_only() -> None:
    """<!-- P3-D-011 --> The case the earlier drafts got wrong.

    Chain is `[cloud, local]`, only the *local* key is cached, and nothing is
    reachable. The right answer is the cached local vision reading - a real
    reading, produced by the second-preference endpoint - not a degraded
    OCR-only bundle. Preference order says who to *ask*; it does not say a
    bundle from a less-preferred reader should be thrown away.
    """
    chain = (REMOTE_A, LOCAL_B)
    keys = candidate_keys(DistillOptions(), chain, "local")

    resolved = resolve_chain(
        DistillOptions(),
        chain,
        "local",
        cached=lambda key: 5 if key == keys[1].opts_hash else None,
        probe=lambda config: False,
    )

    assert resolved.entry == 1
    assert resolved.endpoint is LOCAL_B
    assert resolved.vision_mode == VISION_MODE_SELECTED
    assert resolved.served_from_cache is True


def test_a_memoized_endpoint_is_skipped_rather_than_asked_again() -> None:
    """**Skipped** and **unavailable** are different facts, and both are reported.

    An endpoint asked *this run* and found wanting cost a round trip and means
    "we just checked". One passed over on the memo cost nothing and means "we
    already know". An operator reading diagnostics needs to tell those apart -
    a chain that looks slow because every entry is probed every run is a
    different problem from one where the memo is working.
    """
    chain = (LOCAL_A, LOCAL_B)
    asked: list[str] = []

    def probe(config: LocalVisionConfig) -> bool:
        asked.append(config.model)
        return True

    resolved = resolve_chain(
        DistillOptions(),
        chain,
        "local",
        cached=lambda key: None,
        probe=probe,
        skip=lambda config: config is LOCAL_A,
    )

    # Entry 0 was never asked, so it cost no round trip.
    assert asked == [LOCAL_B.model]
    assert resolved.entry == 1
    assert resolved.evidence == (
        EntryOutcome(entry=0, outcome=SKIPPED),
        EntryOutcome(entry=1, outcome=SELECTED),
    )


def test_a_chain_skipped_all_the_way_down_still_exhausts_rather_than_hangs() -> None:
    """Every entry memoized is still an answer, not a reason to ask anyway.

    The memo exists so a run on a machine that cannot reach anything is fast.
    Falling back to probing when every entry is memoized would give that run the
    full cost of the chain precisely when the memo said it was pointless.
    """
    resolved = resolve_chain(
        DistillOptions(),
        (LOCAL_A, LOCAL_B),
        "local",
        cached=lambda key: None,
        probe=never_probed,
        skip=lambda config: True,
    )

    assert resolved.entry is None
    assert resolved.vision_mode == VISION_MODE_CHAIN_EXHAUSTED
    assert [o.outcome for o in resolved.evidence] == [SKIPPED, SKIPPED]
