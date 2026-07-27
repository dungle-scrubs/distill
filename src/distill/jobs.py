"""Small JSON job-status records for synchronous Distill tools.

A job record is state another process reads while the run that owns it is still
writing: a caller polls it to learn whether the work finished. It is therefore
written by atomic replace (R-14, D-033) - a record caught half-written is
unparseable, which a poller cannot tell apart from a job that never started.

This module does not own the durable-write mechanism, only the record's shape
and where it lives; the replace comes from `bundle_store.atomic_write_text`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .bundle_store import atomic_write_text, ensure_safe_directory


def safe_job_id(job_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in job_id)


JOB_DIR_NAME = "_jobs"


def job_dir(root: Path) -> Path:
    """Where job records live, refusing to follow a symlink at `_jobs` (R-16).

    `_jobs` is a component below the root the writes are confined to, not the
    confinement root itself: a root is only ever compared against, never
    inspected, so passing `_jobs` as the root would leave the one component an
    attacker can pre-create as a link unchecked.
    """
    # The confinement root is created rather than validated: `ensure_safe_directory`
    # compares against a root, so somebody has to make it, and the record store's
    # own root is the caller's to choose.
    root.mkdir(parents=True, exist_ok=True)
    return ensure_safe_directory(root / JOB_DIR_NAME, root)


def job_path(root: Path, job_id: str) -> Path:
    return job_dir(root) / f"{safe_job_id(job_id)}.json"


def write_job_status(
    root: Path,
    job_id: str,
    *,
    status: str,
    tool: str,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "tool": tool,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    path = job_path(root, job_id)
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n", root=root)
    return payload


def read_job_status(root: Path, job_id: str) -> dict[str, Any] | None:
    path = job_path(root, job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())
