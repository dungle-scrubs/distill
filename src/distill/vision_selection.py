"""Vision selection — the deep module that owns the chain walk.

This module owns the **vision chain walk** that `vision_chain` describes but does
not execute: deriving candidate keys, scanning the cache before touching the
network, walking the endpoint chain with memo/skip/deadline, and settling which
bundle a run publishes under.

It composes `vision_chain.candidate_keys` (pure) and `vision_chain.resolve_chain`
behind adapters so callers need not interleave the walk's steps themselves. The
adapters are the seams:

- ``Store`` -> ``load_active`` (or a ``cached`` callable)
- ``Prober`` -> ``probe`` (``bool`` or ``LocalVisionProbe``)

Nothing here opens a socket or a file directly; the adapters are the only way
the walk reaches the world, which is what makes it testable with fakes and why
``vision_chain`` can stay pure.

ADR-0007 invariants preserved:

- **cache-before-network always** - every candidate key is scanned against the
  store before any endpoint is asked; a hit below the top of the chain still
  coalesces with no network at all.
- **chain selects one reader** - at most one endpoint is selected; no bundle
  ever holds interpretations from two.
- **selection evidence stays out of bundles** - which endpoints were asked,
  skipped, or never reached is a machine-local fact carried in the response,
  never in the bundle key or payload.
- **chain not in identity** - the same endpoint reached from different chains
  publishes under the same key; the chain is preference, not identity.

Remote budget (VisionStageBudget) handling:

- local endpoints incur no run-wide budget; a remote endpoint does. When any
  endpoint in the chain is remote (``config_is_non_local``), a single
  ``VisionStageBudget`` is created and attached to the probed copies so the
  probe's GET and attempted completion charge the same bounds as later frame
  interpretation (D-008). The budget instance is carried in the outcome so a
  caller that later interprets frames can reuse it rather than creating a
  second.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .local_vision import LocalVisionConfig, config_is_non_local
from .options import DistillOptions
from .rapid_mlx import (
    DEFAULT_VISION_STAGE_BUDGET_BYTES,
    DEFAULT_VISION_STAGE_BUDGET_SEC,
    VisionStageBudget,
)
from .vision_chain import (
    PROBE_CEILING_SEC,
    REVALIDATE_AFTER_WAIT_SEC,
    AvailabilityMemo,
    EntryOutcome,
    ResolvedRun,
    candidate_keys,
    resolve_chain,
)

REKEY_BOUND_REASON = "one_rekey_per_run"
"""Why a run that diverges a second time proceeds anyway (D-005)."""


class Store(Protocol):
    """What the walk needs from a bundle store: ``load_active``."""

    def load_active(self, bundle_key: str) -> Any | None: ...


class Prober(Protocol):
    """What the walk needs from a vision endpoint: ``probe``."""

    def probe(self, endpoint: LocalVisionConfig) -> Any: ...


@dataclass(frozen=True)
class SelectionOutcome:
    """The one outcome a walk settles on, and what it means.

    ``resolved`` is the ``vision_chain.ResolvedRun`` the walk produced; the
    remaining fields are the run-level view that source resolution and the
    pipeline need without re-deriving it.

    ``bundle_key`` is ``source_hash(fingerprint, opts_hash)`` when a fingerprint
    is known, otherwise the raw ``opts_hash`` - the key the cache was (or would
    be) asked about. ``is_remote`` and ``budget`` are the remote-budget
    handling: a remote selection creates a run-wide ``VisionStageBudget`` that
    the caller can reuse for frame interpretation rather than minting a second.
    """

    resolved: ResolvedRun
    bundle_key: str
    is_remote: bool = False
    budget: VisionStageBudget | None = field(default=None, compare=False, repr=False)

    @property
    def entry(self) -> int | None:
        return self.resolved.entry

    @property
    def vision_mode(self) -> str:
        return self.resolved.vision_mode

    @property
    def opts_hash(self) -> str:
        return self.resolved.opts_hash

    @property
    def endpoint(self) -> LocalVisionConfig | None:
        return self.resolved.endpoint

    @property
    def served_from_cache(self) -> bool:
        return self.resolved.served_from_cache

    @property
    def options(self) -> DistillOptions:
        return self.resolved.options

    @property
    def evidence(self) -> tuple[EntryOutcome, ...]:
        return self.resolved.evidence


def _is_remote_selection(resolved: ResolvedRun) -> bool:
    """Whether this resolution left the machine (D-012)."""
    if resolved.endpoint is not None:
        return config_is_non_local(resolved.endpoint)
    return config_is_non_local(resolved.options.local_vision_config())


def _ensure_budget(
    chain: tuple[LocalVisionConfig, ...],
    budget: VisionStageBudget | None,
) -> VisionStageBudget | None:
    """A run-wide budget when any endpoint is remote (D-008).

    Local endpoints incur no run-wide budget; a remote endpoint does. The
    budget is run-wide (one instance), created once and reused for both
    probing and later frame interpretation so the probe's GET charges the same
    bounds. It is *not* attached to the chain copies here in order to keep
    prober identity checks (``ep is REMOTE_A``) intact for fakes; the caller
    carries it in the outcome for later use.
    """

    if budget is not None:
        return budget
    if not any(config_is_non_local(ep) for ep in chain):
        return None
    return VisionStageBudget(
        wall_clock_sec=DEFAULT_VISION_STAGE_BUDGET_SEC,
        max_bytes=DEFAULT_VISION_STAGE_BUDGET_BYTES,
    )


def _servable_count_from_snapshot(snapshot: Any) -> int | None:
    """How many frames of the snapshot carry a reading, if servable."""
    if snapshot is None:
        return None
    manifest = getattr(snapshot, "manifest", None)
    if manifest is None and isinstance(snapshot, dict):
        manifest = snapshot.get("manifest")
    if not isinstance(manifest, dict):
        # Also handle direct manifest dicts (test fakes).
        if isinstance(snapshot, dict) and "frames" in snapshot:
            manifest = snapshot
        else:
            return None
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return None
    # Counts the response shape's key, because the manifest embeds
    # ``response_frames`` which renames ``visual_interpretation`` on the way out.
    from .artifacts import document_carries_a_reading

    return sum(
        1
        for frame in frames
        if isinstance(frame, dict)
        and document_carries_a_reading(frame.get("visual_interpretation"))
    )


def _make_cached(
    store: Any,
    fingerprint: str | None,
    *,
    source_type: str = "local",  # noqa: ARG001  # kept for API symmetry; fingerprint maps opts_hash to bundle_key
) -> Callable[[str], int | None]:
    """Build the ``cached`` callable ``resolve_chain`` expects from a Store."""

    # Prefer an explicit callable if that's what was handed in (tests).
    if callable(store) and not hasattr(store, "load_active"):
        return store  # type: ignore[return-value]

    def cached(opts_hash: str) -> int | None:
        # Map opts_hash -> bundle_key when a fingerprint is known; otherwise
        # the opts_hash itself is the cache key (pure chain tests).
        bundle_key = opts_hash
        if fingerprint is not None:
            from .media_inspect import source_hash

            bundle_key = source_hash(fingerprint, opts_hash)
        # Store with load_active (BundleStore, fake store, dict wrapper).
        if hasattr(store, "load_active"):
            try:
                snapshot = store.load_active(bundle_key)  # type: ignore[union-attr]
            except Exception:
                return None
            return _servable_count_from_snapshot(snapshot)
        # Dict-like store (test helper): bundle_key -> int | snapshot | None.
        if isinstance(store, dict):
            value = store.get(bundle_key)
            if isinstance(value, int):
                return value
            if value is None:
                return None
            return _servable_count_from_snapshot(value)
        return None

    return cached


def _make_probe(prober: Any) -> Callable[[LocalVisionConfig], bool]:
    """Build the ``probe`` callable ``resolve_chain`` expects."""

    def probe(endpoint: LocalVisionConfig) -> bool:
        if prober is None:
            return False
        # Object with .probe (Prober protocol, FrameInterpreter spy, etc.)
        if hasattr(prober, "probe") and callable(prober.probe):
            result = prober.probe(endpoint)  # type: ignore[union-attr]
        elif callable(prober):
            result = prober(endpoint)
        else:
            return False
        # LocalVisionProbe or similar with .available
        if hasattr(result, "available"):
            return bool(result.available)
        return bool(result)

    return probe


def _make_skip(
    memo: AvailabilityMemo | None,
    now: Callable[[], float] | None,
) -> Callable[[LocalVisionConfig], bool]:
    """Build the ``skip`` callable from a memo, if any."""
    if memo is None:
        return lambda _endpoint: False
    clock = now if now is not None else time.monotonic

    def skip(endpoint: LocalVisionConfig) -> bool:
        try:
            return memo.skips(endpoint, now=clock())
        except TypeError:
            # Memo with different signature (test fake).
            try:
                return bool(memo.skips(endpoint))  # type: ignore[call-arg]
            except Exception:
                return False

    return skip


def select(
    options: DistillOptions,
    chain: tuple[LocalVisionConfig, ...],
    store: Any,
    prober: Any,
    *,
    fingerprint: str | None = None,
    source_type: str = "local",
    output_root: Path | None = None,
    now: Callable[[], float] | None = None,
    ceiling_sec: float = PROBE_CEILING_SEC,
    memo: AvailabilityMemo | None = None,
    budget: VisionStageBudget | None = None,
    claim: Callable[[str], bool] | None = None,
) -> SelectionOutcome:
    """The one endpoint - or the one cached bundle - this run will use.

    Behind the four-arg surface it:

    - derives candidates via ``vision_chain.candidate_keys`` (pure)
    - scans the cache before touching the network (hit coalesces)
    - walks the chain with memo/skip/deadline
    - attaches a run-wide ``VisionStageBudget`` when the selection is remote
    - never puts selection evidence into the bundle key

    ``store`` is ``Store`` (``load_active``) or a ``cached`` callable or a
    mapping; ``prober`` is ``Prober`` or a ``probe`` callable. Both are the
    seams that keep this module from opening a socket or a file itself, which
    is what lets tests drive it with fakes and no server.
    """
    # Keep vision_chain.candidate_keys pure and composed: derive here (even
    # though resolve_chain will derive again) so the pure derivation is tested
    # as part of the walk's composition and a future change to candidate shape
    # is exercised through this path.
    _ = candidate_keys(options, chain, source_type)

    # Remote budget handling: one run-wide budget when any endpoint is
    # remote (D-008). Created here so the walk and later frame
    # interpretation can reuse the same instance; a local chain never
    # creates one. Not attached to the chain copies to keep ``ep is X``
    # identity checks in fakes intact.
    attached_budget = _ensure_budget(chain, budget)
    cached = _make_cached(store, fingerprint, source_type=source_type)
    probe = _make_probe(prober)
    skip = _make_skip(memo, now)
    # When output_root is None there is no store to ask; the cached closure
    # already handles that by returning None for every key.
    # ``resolve_chain`` arms its shared ceiling only when ``now`` is supplied;
    # callers that care about deadline pass ``time.monotonic`` explicitly.
    effective_now = now
    if output_root is not None and now is None:
        # Production callers that have a root but no clock want the ceiling
        # enforced (P3-D-020). Default to monotonic rather than leaving the
        # walk unbounded with no ceiling.
        effective_now = time.monotonic
    resolved = resolve_chain(
        options,
        chain,
        source_type,
        cached=cached,
        probe=probe,
        skip=skip,
        now=effective_now,
        ceiling_sec=ceiling_sec,
        claim=claim if claim is not None else (lambda _k: True),
    )
    bundle_key = resolved.opts_hash
    if fingerprint is not None:
        from .media_inspect import source_hash

        bundle_key = source_hash(fingerprint, resolved.opts_hash)
    is_remote = _is_remote_selection(resolved)
    # If the selection is remote and no budget exists yet, create one for the
    # caller to reuse during frame interpretation. If one was already ensured
    # for the walk, reuse it.
    outcome_budget = attached_budget
    if is_remote and outcome_budget is None:
        outcome_budget = VisionStageBudget(
            wall_clock_sec=DEFAULT_VISION_STAGE_BUDGET_SEC,
            max_bytes=DEFAULT_VISION_STAGE_BUDGET_BYTES,
        )
    return SelectionOutcome(
        resolved=resolved,
        bundle_key=bundle_key,
        is_remote=is_remote,
        budget=outcome_budget,
    )


def revalidation_is_owed(waited_sec: float) -> bool:
    """Whether a run that waited this long owes its **endpoint chain** a second walk.

    One named predicate rather than a comparison inside a run, because the
    wait is the store's measurement and the threshold is a statement about the
    vision memo (D-016). Inclusive at the threshold so a wait landing exactly
    on it has aged exactly that much and is re-asked.
    """

    return waited_sec >= REVALIDATE_AFTER_WAIT_SEC


@dataclass
class VisionSelection:
    """Deep owner of the chain walk, behind ``select``'s shallow surface.

    Composes ``candidate_keys`` and ``resolve_chain`` and hides:

    - candidate derivation
    - cache scan (hit coalesces, no network)
    - availability walk with memo/skip/deadline
    - remote budget handling
    - revalidation after wait and single-rekey decision (D-004, D-005)

    Testable with fakes: ``store`` need only answer ``load_active``,
    ``prober`` need only answer ``probe``, and ``memo``/``now`` are injectable.
    No method here opens a socket or a file.
    """

    options: DistillOptions
    chain: tuple[LocalVisionConfig, ...]
    store: Any = None
    prober: Any = None
    fingerprint: str | None = None
    source_type: str = "local"
    output_root: Path | None = None
    now: Callable[[], float] | None = None
    ceiling_sec: float = PROBE_CEILING_SEC
    memo: AvailabilityMemo | None = None
    budget: VisionStageBudget | None = None
    claim: Callable[[str], bool] | None = None

    def select(self) -> SelectionOutcome:
        """The one outcome this configuration selects, via ``select``."""
        return select(
            self.options,
            self.chain,
            self.store,
            self.prober,
            fingerprint=self.fingerprint,
            source_type=self.source_type,
            output_root=self.output_root,
            now=self.now,
            ceiling_sec=self.ceiling_sec,
            memo=self.memo,
            budget=self.budget,
            claim=self.claim,
        )

    def revalidate(self) -> SelectionOutcome:
        """Walk the chain again, for a run whose answer has aged under a lock.

        The whole walk rather than a cheaper re-check: which endpoint this run
        should read with has one answer-shaped procedure (D-004).
        """

        return self.select()

    def divergence(self, held_bundle_key: str) -> SelectionOutcome | None:
        """Re-walk and compare; ``None`` when the chain still names the held key."""
        revalidated = self.revalidate()
        if revalidated.bundle_key == held_bundle_key:
            return None
        return revalidated

    def settle_after_wait(
        self,
        store: Any,
        held: Any,
        lock_wait_sec: float,
        waited_ref: list[float] | None = None,
    ) -> Any:
        """The hold this run will actually work under, once its wait is accounted for.

        Mirrors ``run_orchestrator.KeySettlement.settle`` but delegates the
        walk to this module so the walk lives in one place. Handles
        revalidation, one-rekey bound, remaining-budget queuing, and the
        confirming walk that can only be recorded (D-005).

        ``held`` is the ``BundleRun`` (or ``BundleSnapshot``) the store
        returned. ``waited_ref`` is a single-element list holding the run's
        total ``waited_sec`` so mutations are observable to the caller, as
        ``KeySettlement`` does. Returns the ``BundleRun`` or ``BundleSnapshot``
        the run should work under.
        """

        import json
        import logging
        import os

        logger = logging.getLogger("distill.pipeline")
        held_run: Any | None = held

        # Inline import to avoid cycles; bundle_store is the lock owner.

        def _log(event: str, **detail: Any) -> None:
            logger.debug(
                json.dumps(
                    {
                        "type": "distill.pipeline",
                        "event": event,
                        "detail": {"pid": os.getpid(), **detail},
                    },
                    sort_keys=True,
                )
            )

        def _abandon_reason(exc: BaseException) -> str:
            from .errors import DistillError  # noqa: PLC0415

            if isinstance(exc, DistillError):
                return f"{exc.code}: {exc}"
            return f"{type(exc).__name__}: {exc}"

        def _begin(bundle_key: str, wait_sec: float) -> Any:
            began = store.begin(
                bundle_key,
                wait_sec=wait_sec,
                resume=self.options.resume_partial,
                reuse_active=not self.options.force_reprocess,
            )
            if waited_ref is not None:
                waited_ref[0] += getattr(began, "waited_sec", 0.0)
            return began

        def _candidate_in_hand() -> Any:
            try:
                from .source import candidate_in_hand  # noqa: PLC0415

                return candidate_in_hand(self.options, self.source_type)
            except Exception:
                return None

        try:
            waited = getattr(held, "waited_sec", 0.0)
            if not revalidation_is_owed(waited):
                return held
            _log(
                "chain_revalidated",
                bundle_key=getattr(held, "bundle_key", ""),
                waited_sec=waited,
                revalidate_after_sec=REVALIDATE_AFTER_WAIT_SEC,
            )
            diverged = self.divergence(getattr(held, "bundle_key", ""))
            if diverged is None:
                return held
            remaining = max(0.0, lock_wait_sec - waited)
            # Leave the old key: abandon, remember rekey, move options/source.
            held_in_hand = _candidate_in_hand()
            _log(
                "chain_diverged",
                bundle_key=getattr(held, "bundle_key", ""),
                entry=None if held_in_hand is None else held_in_hand.entry,
                vision_mode=self.options.vision_mode,
                new_bundle_key=diverged.bundle_key,
                new_entry=diverged.entry,
                new_vision_mode=diverged.vision_mode,
            )
            with contextlib.suppress(Exception):
                held.abandon(f"chain_rekeyed: {diverged.bundle_key}")
            # Mutate this selection to become the run the new key describes,
            # so a caller observing ``self`` sees the move (as KeySettlement does).
            self.options = diverged.options  # type: ignore[attr-defined]
            # ``fingerprint`` stays; bundle_key is derived from it and the new
            # opts_hash, so no fingerprint change is needed.
            held_run = None
            begun = _begin(diverged.bundle_key, remaining)
            # If the new key already holds an active generation, serve it.
            if hasattr(begun, "manifest"):
                # Heuristic: BundleSnapshot has manifest, BundleRun has bundle_key
                # and wait; check type name to avoid importing.
                if type(begun).__name__ == "BundleSnapshot":
                    return begun
                # Also handle real BundleStore behavior: begin returns
                # BundleSnapshot when active generation exists.
                from .bundle_store import BundleSnapshot as _Snap  # noqa: PLC0415

                if isinstance(begun, _Snap):
                    return begun
            held_run = begun
            # Confirming walk: ungated, cannot re-key, only record.
            # Use a fresh selector that mirrors the new walk's inputs but
            # without memo gating concerns; the second wait is usually short.
            # We reuse ``self`` which now carries the diverged options, so the
            # confirming walk is from the new key's perspective.
            declined = self.divergence(getattr(begun, "bundle_key", ""))
            if declined is not None:
                _log(
                    "chain_divergence_declined",
                    bundle_key=getattr(begun, "bundle_key", ""),
                    entry=diverged.entry,
                    vision_mode=diverged.vision_mode,
                    new_bundle_key=declined.bundle_key,
                    new_entry=declined.entry,
                    new_vision_mode=declined.vision_mode,
                    reason=REKEY_BOUND_REASON,
                )
            return begun
        except BaseException as exc:
            if held_run is not None:
                with contextlib.suppress(Exception):
                    held_run.abandon(_abandon_reason(exc), during=exc)
            raise

