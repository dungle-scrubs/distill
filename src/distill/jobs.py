"""Small JSON job-status records for synchronous Distill tools."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def safe_job_id(job_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in job_id)


def job_dir(root: Path) -> Path:
    path = root / "_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    job_path(root, job_id).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def read_job_status(root: Path, job_id: str) -> dict[str, Any] | None:
    path = job_path(root, job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())
