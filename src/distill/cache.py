"""The `cleanup-cache` tool's adapter onto **prune**.

This module owns one thing: the argument and payload shape the public
`cleanup_cache` tool has always had, expressed over `BundleStore.plan_prune` and
`BundleStore.apply_prune`. The command keeps its name (D-042) while the
vocabulary underneath it is **prune**.

It owns no policy of its own. Which **generations** survive, which bundles
expire, what counts as a live run and what may be deleted at all are decided in
`bundle_store` - this module only turns a tool call into a `PrunePolicy` and a
plan or an outcome back into JSON. Placing any rule here is what produced the
duplicate, weaker prune the audit found.

M3.6 migrates the caller onto the store directly and deletes this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle_store import BundleStore, PrunePolicy


def cleanup_cache(
    root: Path,
    *,
    max_age_days: float | None,
    keep_generations: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Prune the cache under `root`, or report what pruning would remove.

    `dry_run` is the plan alone: a `PrunePlan` is advisory (D-023), so producing
    one and not applying it is exactly what a preview is, and there is no second
    code path whose answer could differ from what a real run would do.

    The payload reports what was skipped and how many directories were
    considered alongside what was deleted, so an empty result says which kind of
    empty it is (R-57).
    """
    root.mkdir(parents=True, exist_ok=True)
    store = BundleStore.open(root)
    plan = store.plan_prune(
        PrunePolicy(keep_generations=keep_generations, max_age_days=max_age_days)
    )
    payload: dict[str, Any] = {
        "root": str(store.root),
        "dry_run": dry_run,
        "candidate_count": len(plan.targets),
        "candidates": [str(target.path) for target in plan.targets],
        "considered": plan.considered,
        "skipped": [skip.to_dict() for skip in plan.skipped],
        "skipped_count": len(plan.skipped),
    }
    if dry_run:
        return {**payload, "deleted_count": 0, "deleted": [], "results": []}

    outcome = store.apply_prune(plan)
    return {
        **payload,
        "deleted_count": len(outcome.deleted),
        "deleted": [str(path) for path in outcome.deleted],
        "results": [result.to_dict() for result in outcome.results],
    }
