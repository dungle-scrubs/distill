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
"""

from __future__ import annotations

from distill.bundle_store import BATCH_ITEM_LOCK_WAIT_SEC, SINGLE_SOURCE_LOCK_WAIT_SEC
from distill.vision_chain import REVALIDATE_AFTER_WAIT_SEC


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
