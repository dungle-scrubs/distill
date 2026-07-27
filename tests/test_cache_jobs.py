"""Unit tests for the public cache/jobs facades.

The `cleanup-cache` tool keeps its name (D-042) and is now an adapter onto
**prune**: `BundleStore.plan_prune` proposes, `BundleStore.apply_prune` decides
under the lock. These tests hold the tool's shape against **bundles** that carry
a **bundle marker** - the retention tests here used to prove the tool deleted
generations out of directories Distill never wrote, which is finding 1, and one
of them passed `keep_generations=0`, which is finding 2's input.
`tests/test_bundle_prune.py` owns the prune rules themselves.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from distill.cache import cleanup_cache
from distill.errors import DistillError
from distill.jobs import job_dir, job_path, read_job_status, safe_job_id, write_job_status


def _write_bundle(
    root: Path,
    bundle_key: str,
    generations: tuple[str, ...],
    active: str,
) -> Path:
    """A marked **bundle**: generations plus the manifest that proves it is one."""
    bundle = root / bundle_key
    bundle.mkdir(parents=True)
    for name in generations:
        (bundle / name).mkdir()
        (bundle / name / "transcript.json").write_text("{}")
        (bundle / name / "video.md").write_text("# transcript")
    (bundle / "_manifest.json").write_text(
        json.dumps(
            {
                "pipeline_version": 1,
                "distill_version": "0.1.0",
                "source_type": "local",
                "bundle_key": bundle_key,
                "source_resolved_path": "/tmp/video.mp4",
                "duration_sec": 1.0,
                "options": {},
                "frame_count": 0,
                "transcript_present": True,
                "warning_count": 0,
                "frames": [],
                "warnings": [],
                "active_generation": active,
            },
            indent=2,
        )
        + "\n"
    )
    return bundle


def _backdate(directory: Path, days: float) -> None:
    stamp = time.time() - days * 86400
    for path in sorted(directory.rglob("*"), reverse=True):
        os.utime(path, (stamp, stamp))
    os.utime(directory, (stamp, stamp))


def test_cleanup_cache_keeps_newest_generations(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    bundle = _write_bundle(root, "abc123", ("g1", "g2", "g3", "g4"), active="g4")

    result = cleanup_cache(root, max_age_days=None, keep_generations=2, dry_run=False)

    # g1 and g2 are pruned; g3 and the active g4 are retained.
    assert set(result["deleted"]) == {str(bundle / "g1"), str(bundle / "g2")}
    assert (bundle / "g3").exists()
    assert (bundle / "g4").exists()
    assert result["deleted_count"] == 2


def test_cleanup_cache_dry_run_does_not_delete(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    bundle = _write_bundle(root, "abc123", ("g1", "g2", "g3", "g4"), active="g4")

    result = cleanup_cache(root, max_age_days=None, keep_generations=2, dry_run=True)

    assert result["deleted"] == []
    assert result["candidate_count"] == 2
    assert (bundle / "g1").exists()


def test_cleanup_cache_prunes_old_bundles_by_age(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    old_bundle = _write_bundle(root, "old", ("g1",), active="g1")
    new_bundle = _write_bundle(root, "new", ("g1",), active="g1")
    _backdate(old_bundle, days=40)

    result = cleanup_cache(root, max_age_days=30, keep_generations=99, dry_run=False)

    # Expiry takes the whole aged bundle, active generation included (D-018).
    assert result["deleted"] == [str(old_bundle)]
    assert new_bundle.exists()
    assert not old_bundle.exists()


def test_cleanup_cache_skips_private_dirs_and_says_so(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    (root / "_internal").mkdir(parents=True)
    (root / "_internal" / "g1").mkdir()
    (root / "_internal" / "g1" / "video.md").write_text("# x")

    result = cleanup_cache(root, max_age_days=None, keep_generations=1, dry_run=False)

    assert result["candidate_count"] == 0
    assert (root / "_internal").exists()
    skipped = {entry["path"]: entry["verdict"] for entry in result["skipped"]}
    assert skipped[str(root / "_internal")] == "reserved"


def test_cleanup_cache_refuses_keep_generations_below_one(tmp_path: Path) -> None:
    """Finding 2's input, refused at the tool boundary (R-03)."""
    root = tmp_path / "cache"
    _write_bundle(root, "abc123", ("g1",), active="g1")

    with pytest.raises(DistillError) as raised:
        cleanup_cache(root, max_age_days=None, keep_generations=0, dry_run=False)

    assert raised.value.code == "E_BAD_OPTIONS"


def test_cleanup_cache_reports_what_it_considered(tmp_path: Path) -> None:
    """"Considered nothing" and "deleted nothing" are different answers (R-57)."""
    root = tmp_path / "cache"
    (root / "notes").mkdir(parents=True)

    result = cleanup_cache(root, max_age_days=None, keep_generations=1, dry_run=False)

    assert result["deleted"] == []
    assert result["considered"] == 1
    assert result["skipped"][0]["reason"] == "no bundle marker"


def test_write_then_read_job_status_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    payload = write_job_status(
        root,
        "job-1",
        status="completed",
        tool="process_local_video",
        result={"manifest_path": "/tmp/manifest.json"},
    )

    assert payload["status"] == "completed"
    assert payload["tool"] == "process_local_video"
    assert payload["job_id"] == "job-1"
    assert "updated_at" in payload

    record = read_job_status(root, "job-1")
    assert record == payload

    written = json.loads(job_path(root, "job-1").read_text())
    assert written["result"]["manifest_path"] == "/tmp/manifest.json"


def test_read_job_status_returns_none_when_missing(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    assert read_job_status(root, "absent") is None


def test_write_job_status_records_error(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    payload = write_job_status(
        root,
        "job-err",
        status="failed",
        tool="process_local_video",
        error={"code": "E_INTERNAL", "message": "boom"},
    )

    assert payload["error"] == {"code": "E_INTERNAL", "message": "boom"}
    assert "result" not in payload


def test_safe_job_id_sanitizes_path_characters() -> None:
    assert safe_job_id("job/../escape") == "job____escape"
    assert safe_job_id("ok-id_1") == "ok-id_1"


def test_job_dir_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    first = job_dir(root)
    second = job_dir(root)
    assert first == second
    assert first.is_dir()
