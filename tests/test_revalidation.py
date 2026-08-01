"""Re-asking the chain after a wait long enough to have outlived the answer.

Chain resolution happens during source resolution, which is *before* a run takes
the lock on the **bundle key** it resolved to. A run that then waits out a
contended lock arrives at its own work holding an availability answer it
gathered minutes ago - and the memo that answer came from has a life measured in
the same minutes (D-004).

The seam this file starts from is the threshold itself. It is a relationship
between three constants, two of which live in `bundle_store.py` and one in
`vision_chain.py`, so these tests import across that boundary deliberately: the
store must not learn what a vision endpoint is (D-016), but a test may relate
what two modules each know. Asserting the relation here is what keeps it from
becoming a coincidence nobody would notice breaking.

The second seam is `ProcessingRun.execute`, the one place that holds both the
wait `BundleStore.begin` measured and the inputs a second walk needs. The wait
is real here - a lock another descriptor holds, polled out on a clock that only
moves when the waiter sleeps - so what the run acts on is the number `_take_lock`
computed rather than one a test handed it. Nothing reaches a server: the chain's
probe is the module-level seam `source._probe_endpoint`, and the media stages are
fakes, because what they produce is not what is under test.

Divergence is driven through the memo, on that same clock. An endpoint recorded
unavailable shortly before the run is *skipped* during resolution - passed over
without a socket - and the wait that follows is long enough to carry the clock
past the memo's life, so the second walk asks it and gets a different answer.
That is D-004's worst case written as a scenario rather than asserted as a
number: `WAIT_THAT_OUTLIVES_THE_MEMO` is the wait a single-source run can
actually spend, and it is exactly enough to expire the answer it queued with.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_bundle_locking import FakeClock, hold_the_lock, lock_is_held, lock_path

from distill import bundle_store, pipeline, source
from distill.artifacts import FrameArtifact, Provenance, RedactionState
from distill.bundle_store import (
    BATCH_ITEM_LOCK_WAIT_SEC,
    MANIFEST_NAME,
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleStore,
)
from distill.errors import DistillError
from distill.local_vision import LocalVisionConfig
from distill.media_inspect import source_hash
from distill.options import (
    VISION_MODE_CHAIN_EXHAUSTED,
    VISION_MODE_SELECTED,
    DistillOptions,
)
from distill.pipeline import REKEY_BOUND_REASON, ProcessingRun
from distill.progress import ProgressReporter
from distill.source import SourceInfo, _resolved_for
from distill.vision_chain import (
    DEFAULT_MEMO_TTL_SEC,
    REVALIDATE_AFTER_WAIT_SEC,
    AvailabilityMemo,
    resolve_chain,
)


def test_a_contended_batch_item_can_never_wait_its_way_to_revalidation() -> None:
    """A batch item's whole lock budget is spent well short of the threshold.

    A playlist item gives up on a contended key after 5 s and lets the batch
    proceed, so the longest it can possibly have held a stale availability
    answer is 5 s against a memo measured in minutes (D-006). Re-keying such a
    run would buy nothing and charge every contended batch a second resolution,
    so the threshold has to sit strictly above the budget - not merely above
    the waits observed in practice.
    """
    assert BATCH_ITEM_LOCK_WAIT_SEC < REVALIDATE_AFTER_WAIT_SEC


def test_a_single_source_run_that_waits_it_out_lands_past_the_threshold() -> None:
    """Spending the whole single-source budget is enough to owe a second walk.

    The run a user is watching waits 300 s for a contended key rather than
    failing, and the memo its availability answer came from is trusted for the
    same 300 s - so the worst case is not a slightly stale answer but an expired
    one (D-004). The threshold has to be strictly inside that budget, or the
    exact case it exists for would be the one case that never triggers it.
    """
    assert REVALIDATE_AFTER_WAIT_SEC < SINGLE_SOURCE_LOCK_WAIT_SEC


ENDPOINT = LocalVisionConfig(model="a-served-reader", base_url="http://127.0.0.1:8000/v1")
FINGERPRINT = "the-fingerprint-of-a-fixture"

PREFERRED = LocalVisionConfig(model="the-preferred-reader", base_url="http://127.0.0.1:8000/v1")
FALLBACK = LocalVisionConfig(model="the-fallback-reader", base_url="http://127.0.0.1:8001/v1")
CHAIN = (PREFERRED, FALLBACK)
"""Two readers, so a walk has somewhere else to land and the key can move.

One endpoint can only diverge into the exhausted key, which is degradation
rather than re-selection. The scenario the re-key exists for is the ordinary
one: the preferred reader was down when the run resolved and is up by the time
the run gets its turn.
"""

MEMO_PREDATES_THE_RUN = 1.0
"""How long before the run's clock starts the memo's answer was written.

The memo is another run's record, so it is dated before this one begins. A
second of it is enough to make the arithmetic below exact rather than
borderline.
"""

WAIT_THAT_OUTLIVES_THE_MEMO = DEFAULT_MEMO_TTL_SEC - MEMO_PREDATES_THE_RUN
"""The longest wait a single-source run can spend, which is exactly enough.

<!-- D-004 --> The two budgets are equal: a run waits 300 s for a contended key
and an unavailability answer is trusted for 300 s. So the worst case is not a
stale answer, it is an expired one - and it takes the *whole* budget to reach.
Derived from the memo rather than written as 299, because the point being made
is that these two numbers meet.
"""


def _fake_transcribe(*_args: Any, **_kwargs: Any) -> tuple[None, list[dict[str, str]]]:
    return None, []


def _fake_select_keyframes(
    _video: Path,
    frames_dir: Path,
    *_args: Any,
    redaction: RedactionState,
    **_kwargs: Any,
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    image = frames_dir / "frame_0001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return [
        FrameArtifact(
            index=1,
            timestamp_sec=0.0,
            path=str(image),
            relative_path="frames/frame_0001.png",
            extracted_text="a frame this test never reads",
            redaction=redaction,
        )
    ], []


def _fake_ocr_frames(
    frames: list[FrameArtifact], *_args: Any
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    return frames, []


def _fake_interpret(
    frames: list[FrameArtifact], *_args: Any, **_kwargs: Any
) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
    return frames, []


@dataclass
class Contended:
    """One `execute()` that came out of a wait, and what the wait cost.

    `probes` holds only the endpoints asked *after* the run started: the walk
    that settled the **bundle key** happens before there is a run at all, so
    counting from zero at `execute` is what makes "asked a second time" a
    countable thing rather than an inference from a total.
    """

    response: dict[str, Any]
    probes: list[LocalVisionConfig]
    bundle_key: str
    root: Path
    lock_events: list[dict[str, Any]]
    pipeline_events: list[dict[str, Any]]


def _structured(caplog: pytest.LogCaptureFixture, logger: str) -> list[dict[str, Any]]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == logger and record.message.startswith("{")
    ]


def _options(chain: tuple[LocalVisionConfig, ...]) -> DistillOptions:
    return DistillOptions(caption_frames=True, local_vision_endpoints=chain, job_id="job-1")


def _key_of(chain: tuple[LocalVisionConfig, ...], entry: int) -> str:
    """The **bundle key** this run publishes under when `entry` is the reader.

    Walked rather than hand-derived: `opts_hash` is what the walk settles on and
    a test that assembled one itself would be asserting its own arithmetic. The
    seams are `resolve_chain`'s own, so nothing here needs a store or a socket.
    """
    resolved = resolve_chain(
        _options(chain),
        chain,
        "local",
        cached=lambda _opts_hash: None,
        probe=lambda endpoint: endpoint == chain[entry],
    )
    return source_hash(FINGERPRINT, resolved.opts_hash)


def _exhausted_key(chain: tuple[LocalVisionConfig, ...]) -> str:
    """The key a run publishes under when every entry was asked and none served.

    A degraded bundle, and a real one: the operator asked for a reading and did
    not get one, which is a different bundle from a run that never wanted one.
    """
    resolved = resolve_chain(
        _options(chain),
        chain,
        "local",
        cached=lambda _opts_hash: None,
        probe=lambda _endpoint: False,
    )
    return source_hash(FINGERPRINT, resolved.opts_hash)


def _run_that_waited(
    seconds: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    *,
    chain: tuple[LocalVisionConfig, ...] = (ENDPOINT,),
    answers: Callable[[LocalVisionConfig, int], bool] = lambda _endpoint, _walk: True,
    skipped_at_resolution: tuple[LocalVisionConfig, ...] = (),
    held_throughout: tuple[str, ...] = (),
) -> Contended:
    """A whole run of one source, held out of its own **bundle key** for `seconds`.

    The contention is the kernel's - a descriptor this process holds, which
    `flock` refuses to the run exactly as another process's would - and the
    clock is the store's injected one, so the budget is spent in the poll loop
    without a second of real time passing. `BundleStore.open` is replaced for
    the duration because `execute` opens its own store: production has no seam
    for a clock there, and the wait is the input under test.

    `skipped_at_resolution` names the endpoints an earlier run found
    unavailable. They are recorded in a real `AvailabilityMemo` read on the same
    clock the wait spends, so a wait that outlives the memo expires it for the
    second walk without the test saying so - which is the whole scenario D-004
    describes. `answers` is asked which walk it is on, so an endpoint can be up
    for one and down for the next.

    `held_throughout` names bundle keys some other run never gives up, for the
    case where the key a re-key moves *to* is itself contended.
    """
    root = tmp_path / "output"
    root.mkdir()
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"not a video, and never opened")

    clock = FakeClock()
    memo = AvailabilityMemo()
    for endpoint in skipped_at_resolution:
        memo.record_unavailable(endpoint, now=clock.now)
    clock.now = MEMO_PREDATES_THE_RUN
    until = clock.now + seconds

    probes: list[LocalVisionConfig] = []
    walk: dict[str, int] = {"index": -1}

    def resolving(*args: Any, **kwargs: Any) -> Any:
        """`resolve_chain` with the memo and the clock this test drives.

        Production wires neither yet, so the seams `resolve_chain` already
        accepts are filled in here rather than reached around: the walk stays
        the real one, and what changes between two walks is only what an
        endpoint answered and how much of the memo's life had run.
        """
        walk["index"] += 1
        return resolve_chain(
            *args,
            skip=lambda endpoint: memo.skips(endpoint, now=clock.now),
            now=clock.monotonic,
            **kwargs,
        )

    def probe(endpoint: LocalVisionConfig) -> bool:
        probes.append(endpoint)
        return answers(endpoint, walk["index"])

    monkeypatch.setattr(source, "resolve_chain", resolving)
    monkeypatch.setattr(source, "_probe_endpoint", probe)
    options = _options(chain)
    # The walk source resolution performs, run here because that is where it
    # happens: the run this test drives starts from its outcome, holding a
    # **bundle key** an endpoint answered for before the wait began.
    resolution = _resolved_for(options, FINGERPRINT, "local", root)
    bundle_key = source_hash(FINGERPRINT, resolution.opts_hash)
    probes.clear()

    holder: dict[str, int | None] = {"fd": hold_the_lock(root, bundle_key)}
    others = [hold_the_lock(root, key) for key in held_throughout]

    def sleep(interval: float) -> None:
        before = clock.now
        clock.sleep(interval)
        if clock.now == before:
            # A poll loop asked for a delay this clock cannot represent: the
            # interval has fallen below the float's resolution at the time it
            # has reached. Real time keeps moving and would end the wait, so a
            # test left to spin here would hang forever on what is a fault in
            # the *budget* being spent, not in the loop spending it.
            raise AssertionError(
                f"the fake clock stopped at {before}: a wait longer than this "
                "test can represent is being spent"
            )
        fd = holder["fd"]
        if fd is not None and clock.now >= until:
            os.close(fd)
            holder["fd"] = None

    monkeypatch.setattr(
        BundleStore,
        "open",
        classmethod(lambda cls, root, **_kwargs: cls(Path(root).resolve(), clock.monotonic, sleep)),
    )
    monkeypatch.setattr(pipeline, "transcribe_with_imports", _fake_transcribe)
    monkeypatch.setattr(pipeline, "select_keyframes", _fake_select_keyframes)
    monkeypatch.setattr(pipeline, "ocr_frames", _fake_ocr_frames)
    monkeypatch.setattr(pipeline, "interpret_frames_with_local_vision", _fake_interpret)
    caplog.set_level(logging.DEBUG)

    run = ProcessingRun(
        source=SourceInfo(
            source_type="local",
            resolved_path=video,
            duration_sec=1.0,
            source_fingerprint=FINGERPRINT,
            source_hash=bundle_key,
            warnings=[],
            provenance=Provenance(
                title=video.name,
                duration_sec=1.0,
                processed_at="2026-08-01T09:00:00Z",
            ),
            resolved_options=resolution.options,
        ),
        options=resolution.options,
        output_root=root,
        progress=ProgressReporter(),
        tool="process_local_video",
    )
    try:
        response = run.execute()
    finally:
        fd = holder["fd"]
        if fd is not None:
            os.close(fd)
        for other in others:
            os.close(other)
    return Contended(
        response=response,
        probes=probes,
        bundle_key=bundle_key,
        root=root,
        lock_events=_structured(caplog, bundle_store.LOGGER.name),
        pipeline_events=_structured(caplog, pipeline.LOGGER.name),
    )


def _counted(events: list[dict[str, Any]], name: str) -> int:
    return sum(1 for event in events if event["event"] == name)


def test_a_run_that_waited_past_the_threshold_re_asks_and_keeps_its_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FAILS FIRST: the wait is measured, and nothing acts on it.

    The run walked the chain before it queued, waited out a contended key for
    longer than the answer it was carrying is worth trusting, and then read
    against that answer anyway (D-004). Asking again is the whole feature.

    The second walk agrees here - the same endpoint answers, so it names the
    same key - and agreement must cost nothing: the lock in hand was taken on
    that key, so there is nothing to release and nothing to take again. One
    acquisition and one release over the whole run is what "kept the lock" is,
    stated in the only terms an operator could check.
    """
    contended = _run_that_waited(REVALIDATE_AFTER_WAIT_SEC + 50.0, tmp_path, monkeypatch, caplog)

    assert contended.probes == [ENDPOINT]
    assert contended.response["source_hash"] == contended.bundle_key
    assert _counted(contended.lock_events, "lock_acquired") == 1
    assert _counted(contended.lock_events, "lock_released") == 1


def test_a_wait_below_the_threshold_leaves_the_chain_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A whole batch-item budget spent waiting is still not worth re-asking.

    The guard on the other side of D-006. Revalidation is not free - the walk
    scans every candidate key against the store before it would probe - and an
    answer five seconds old is an answer, so a threshold that triggered on any
    non-zero wait would charge every contended item of every playlist for
    staleness that cannot have occurred.
    """
    contended = _run_that_waited(BATCH_ITEM_LOCK_WAIT_SEC, tmp_path, monkeypatch, caplog)

    assert contended.probes == []
    assert contended.response["source_hash"] == contended.bundle_key


def test_revalidation_says_which_key_waited_how_long_and_against_what(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second walk nobody can see is a second walk nobody can explain.

    Revalidation is invisible from the outside: it costs a cache scan, usually
    changes nothing, and happens only to runs that were already slow for another
    reason. Three fields make it accountable - which **bundle key** was being
    re-asked about, how long the wait was, and the threshold that wait was
    measured against - so a run that re-asked and a run that did not are told
    apart by the numbers rather than by inference from the total.

    `bundle_key` is the correlation field: the store logged `lock_acquired`
    under it moments earlier, and reading the two together is how an operator
    sees a wait turn into a decision.
    """
    contended = _run_that_waited(REVALIDATE_AFTER_WAIT_SEC + 50.0, tmp_path, monkeypatch, caplog)

    revalidated = [
        event for event in contended.pipeline_events if event["event"] == "chain_revalidated"
    ]
    assert len(revalidated) == 1
    detail = revalidated[0]["detail"]
    assert detail["bundle_key"] == contended.bundle_key
    assert detail["waited_sec"] == pytest.approx(REVALIDATE_AFTER_WAIT_SEC + 50.0, abs=0.05)
    assert detail["revalidate_after_sec"] == REVALIDATE_AFTER_WAIT_SEC

    acquired = [event for event in contended.lock_events if event["event"] == "lock_acquired"]
    assert [event["detail"]["bundle_key"] for event in acquired] == [contended.bundle_key]


def test_a_diverging_second_walk_publishes_under_the_key_it_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FAILS FIRST: the second walk names a key, and the run ignores it.

    <!-- D-005 --> Re-resolution *is* re-keying. `opts_hash` is derived from the
    reader and the outcome, so a walk that lands on a different endpoint lands
    on a different **bundle key** - and the lock in hand was taken on the old
    one. There is no version of this where the run keeps its lock and honours
    the new answer, so it gives the lock up and begins again.

    The scenario is the ordinary one rather than a contrived one: the preferred
    reader was recorded unavailable a moment before the run resolved, so
    resolution skipped it without a socket and settled on the fallback. The run
    then spent the whole budget waiting for the fallback's key, and by the time
    it had it the memo had expired - so the preferred reader is asked, answers,
    and names a key this run does not hold.

    Both halves are asserted, because a re-key that only half happened is worse
    than none: the new key carries the **active generation**, and the old one
    carries no **manifest** at all - not a published bundle, not a partial one.
    """
    contended = _run_that_waited(
        WAIT_THAT_OUTLIVES_THE_MEMO,
        tmp_path,
        monkeypatch,
        caplog,
        chain=CHAIN,
        skipped_at_resolution=(PREFERRED,),
    )

    rekeyed = _key_of(CHAIN, 0)
    assert contended.bundle_key == _key_of(CHAIN, 1)
    assert contended.response["source_hash"] == rekeyed

    store = BundleStore.open(contended.root)
    assert store.load_active(rekeyed) is not None
    assert store.load_active(contended.bundle_key) is None
    assert list(store.bundle_root(contended.bundle_key).rglob(MANIFEST_NAME)) == []
    assert _counted(contended.lock_events, "lock_acquired") == 2


def test_the_divergence_names_both_keys_both_entries_and_both_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run that changed its mind has to say what it changed it from.

    A re-key is the one thing revalidation does that an operator would notice
    from outside - a bundle appears under a key nobody asked for, and the key
    the run was queued under is left holding nothing. Explaining that takes both
    sides: the old key correlates with the `lock_acquired` this run already
    logged, the new key with the generation it is about to publish, and the two
    entries with their **vision modes** say which reader it started toward and
    which one it finished on.

    Reader identity is not the endpoint address, which is a machine-local claim
    and never enters a bundle, so the entry index and the mode are what an
    operator gets - and the mode is what tells a move down the chain from a run
    that gave up on the chain entirely.

    The harder divergence is used here deliberately: *both* readers were down
    when this run resolved, so it queued as a chain-exhausted run - an
    **interpretation**-free bundle it would have published as degradation. One
    came back during the wait. An event carrying only the entries would report
    `None -> 0` and leave the reader to guess whether the run had been disabled;
    the modes are what make it say "gave up on the chain, then did not".
    """
    contended = _run_that_waited(
        WAIT_THAT_OUTLIVES_THE_MEMO,
        tmp_path,
        monkeypatch,
        caplog,
        chain=CHAIN,
        skipped_at_resolution=CHAIN,
    )

    assert contended.bundle_key == _exhausted_key(CHAIN)
    diverged = [event for event in contended.pipeline_events if event["event"] == "chain_diverged"]
    assert len(diverged) == 1
    detail = diverged[0]["detail"]
    assert detail["bundle_key"] == contended.bundle_key
    assert detail["new_bundle_key"] == _key_of(CHAIN, 0)
    assert detail["entry"] is None
    assert detail["new_entry"] == 0
    assert detail["vision_mode"] == VISION_MODE_CHAIN_EXHAUSTED
    assert detail["new_vision_mode"] == VISION_MODE_SELECTED

    acquired = [
        event["detail"]["bundle_key"]
        for event in contended.lock_events
        if event["event"] == "lock_acquired"
    ]
    assert acquired == [detail["bundle_key"], detail["new_bundle_key"]]


def test_a_second_divergence_is_recorded_rather_than_acted_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The re-key is bounded at one, and the bound is what makes it safe.

    <!-- D-005 --> Re-keying is not free: it gives up a lock and queues again,
    and each queue can cost a wait. A run that re-keyed every time the chain
    disagreed with it would follow a flapping endpoint around the chain,
    spending a wait budget per lap and finishing no work - and a chain that
    flaps is exactly the chain this whole feature triggers on.

    So the second answer is taken as information rather than as an instruction.
    The preferred reader comes back during the wait, the run re-keys to it, and
    by the time the run asks once more it has gone away again - and the run
    proceeds on the key it just took rather than moving back. Two acquisitions
    over the whole run is the assertion that says so, because a third would be
    the loop this bound exists to prevent.

    Proceeding is a real cost, accepted with the bound: the run will read with
    an endpoint that has stopped answering, and publish a `selected` bundle with
    no **interpretations** in it - which `_is_servable` then refuses to serve, so
    a later run redoes the work rather than inheriting the mistake. The declined
    event is how an operator sees that happen instead of guessing at it.
    """
    contended = _run_that_waited(
        WAIT_THAT_OUTLIVES_THE_MEMO,
        tmp_path,
        monkeypatch,
        caplog,
        chain=CHAIN,
        skipped_at_resolution=(PREFERRED,),
        answers=lambda endpoint, walk: not (endpoint == PREFERRED and walk == 2),
    )

    rekeyed = _key_of(CHAIN, 0)
    assert contended.response["source_hash"] == rekeyed
    assert _counted(contended.lock_events, "lock_acquired") == 2
    assert _counted(contended.pipeline_events, "chain_diverged") == 1

    declined = [
        event
        for event in contended.pipeline_events
        if event["event"] == "chain_divergence_declined"
    ]
    assert len(declined) == 1
    detail = declined[0]["detail"]
    assert detail["reason"] == REKEY_BOUND_REASON
    assert detail["bundle_key"] == rekeyed
    assert detail["new_bundle_key"] == contended.bundle_key
    assert detail["entry"] == 0
    assert detail["new_entry"] == 1
    assert detail["vision_mode"] == VISION_MODE_SELECTED
    assert detail["new_vision_mode"] == VISION_MODE_SELECTED


def test_a_re_key_queues_on_what_is_left_of_the_budget_and_fails_as_the_new_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A re-keyed run must not be able to wait twice over, or fail as the wrong key.

    The wait budget is a promise to whoever is watching: five minutes for a
    contended key, and then the run gives up. A re-key that queued afresh would
    turn that into ten - the run waits out one key, learns it wants another, and
    starts the whole budget again on a key that may be just as contended. So the
    second queue gets what is left of the first, and here that is a second,
    because the divergence took the entire budget to discover.

    What the failure *names* is the other half. The lock this run could not take
    is the new key's; the old key's lock it released itself, deliberately, on
    its way here. An `E_LOCKED` reporting the old key would send an operator to
    look at a lock nobody holds - and `bundle_key` is the field every other
    event in this run correlates on, so naming the wrong one is not a cosmetic
    error.
    """
    rekeyed = _key_of(CHAIN, 0)

    with pytest.raises(DistillError) as raised:
        _run_that_waited(
            WAIT_THAT_OUTLIVES_THE_MEMO,
            tmp_path,
            monkeypatch,
            caplog,
            chain=CHAIN,
            skipped_at_resolution=(PREFERRED,),
            held_throughout=(rekeyed,),
        )

    assert raised.value.code == "E_LOCKED"
    assert raised.value.details["bundle_key"] == rekeyed
    assert raised.value.details["bundle_key"] != _key_of(CHAIN, 1)
    assert raised.value.details["wait_budget_sec"] == pytest.approx(
        SINGLE_SOURCE_LOCK_WAIT_SEC - WAIT_THAT_OUTLIVES_THE_MEMO, abs=0.05
    )


def test_a_second_walk_that_fails_gives_the_key_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deciding the key happens outside the block that gives the lock up.

    Everything before the first stage - the second walk, the release, the
    re-begin - runs with a **bundle key** held and no `__exit__` waiting to give
    it back. The walk is not inert either: it scans every candidate against the
    store, so a bundle directory this run cannot read fails it. A run that died
    there while holding the lock would leave the key unusable for as long as the
    process lived, which for a long-running server is until it is restarted.

    Asserted on the lock itself rather than on a log line: a fresh descriptor is
    refused exactly as another run's would be, so taking it is the only proof
    that the key came back.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise DistillError("E_INTERNAL", "bundle", "the store could not be read", {})

    monkeypatch.setattr(pipeline, "revalidate_chain", refuse)

    with pytest.raises(DistillError) as raised:
        _run_that_waited(
            WAIT_THAT_OUTLIVES_THE_MEMO,
            tmp_path,
            monkeypatch,
            caplog,
            chain=CHAIN,
            skipped_at_resolution=(PREFERRED,),
        )

    assert raised.value.code == "E_INTERNAL"
    root = tmp_path / "output"
    assert lock_is_held(lock_path(root, _key_of(CHAIN, 1))) is False
