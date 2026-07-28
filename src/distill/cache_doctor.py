"""The read-only inspection surface over an output root (R-57).

This module owns one thing: the payload `distill cache-doctor` prints. It turns
a `BundleStore.survey` and a `BundleStore.plan_prune` into one document
answering, for a root a user names: which directories are **bundles**, what the
**bundle marker** verdict was for every directory looked at, each bundle's
**active generation** and **orphan generations**, which run locks are live and
which are leftovers, and what a **prune** would remove.

It exists because findings 1 and 2 were both invisible until they were
irreversible. A directory Distill never wrote was proposed for deletion, and
retention deleted the **active generation** out from under a manifest that still
named it; in neither case was there a command that would have said so first.
The skips are what carry the requirement: a directory the walk declined to treat
as a bundle is reported with its reason, so "considered nothing" and "deleted
nothing" are different answers.

It is read-only, and that is a property of the whole path rather than a promise
made here: `survey` and `plan_prune` create nothing, the root is validated
without being created, and no prune is applied. The `dry_run` flag on
`cleanup-cache` is a different thing - that command can be told to delete, and
this one cannot be.

What it does not own: bundle layout, marker rules, retention or expiry policy,
and liveness - every one of those is decided in `bundle_store` and only
formatted here. It owns no deletion at all, and no repair: a doctor that could
fix things would need the same destructive powers whose consequences it exists
to preview. Nor does it own the output-root policy, which is
`source.validate_output_root`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle_store import BundleStore, PrunePolicy


def inspect_cache(
    root: Path,
    *,
    keep_generations: int,
    max_age_days: float | None,
) -> dict[str, Any]:
    """Report the state of every bundle under `root`, and what a prune would do.

    The preview is a real `PrunePlan` under the policy the caller names, not a
    second implementation of the prune rules: producing a plan and declining to
    apply it is exactly what a preview is (D-023), so the answer here cannot
    differ from what `cleanup-cache` would actually propose.

    `keep_generations` and `max_age_days` are validated by `PrunePolicy` on
    construction (R-03), so an inspection asked under a policy that would be
    refused is refused here too rather than previewing a prune nobody can run.
    """
    store = BundleStore.open(root)
    survey = store.survey()
    policy = PrunePolicy(keep_generations=keep_generations, max_age_days=max_age_days)
    return {**survey.to_dict(), "prune_preview": store.plan_prune(policy).to_dict()}
