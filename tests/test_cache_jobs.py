"""Unit tests for the public cache/jobs facades."""

from __future__ import annotations

import json
import time
from pathlib import Path

from distill.cache import cleanup_cache
from distill.jobs import job_dir, job_path, read_job_status, safe_job_id, write_job_status


def _make_generation(bundle: Path, gen: str, *, with_markdown: bool = True) -> Path:
    generation = bundle / gen
    generation.mkdir(parents=True)
    (generation / "transcript.json").write_text("{}")
    if with_markdown:
        (generation / "video.md").write_text("# transcript")
    return generation


def _touch_mtime(path: Path, mtime: float) -> None:
    import os

    os.utime(path, (mtime, mtime))


def test_cleanup_cache_keeps_newest_generations(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    bundle = root / "abc123"
    for gen in ("g1", "g2", "g3", "g4"):
        _make_generation(bundle, gen)

    result = cleanup_cache(root, max_age_days=None, keep_generations=2, dry_run=False)

    # g1 and g2 are pruned; g3 and g4 are retained.
    assert set(result["deleted"]) == {str(bundle / "g1"), str(bundle / "g2")}
    assert (bundle / "g3").exists()
    assert (bundle / "g4").exists()
    assert result["deleted_count"] == 2


def test_cleanup_cache_dry_run_does_not_delete(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    bundle = root / "abc123"
    for gen in ("g1", "g2", "g3", "g4"):
        _make_generation(bundle, gen)

    result = cleanup_cache(root, max_age_days=None, keep_generations=2, dry_run=True)

    assert result["deleted"] == []
    assert result["candidate_count"] == 2
    assert (bundle / "g1").exists()


def test_cleanup_cache_prunes_old_bundles_by_age(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    old_bundle = root / "old"
    new_bundle = root / "new"
    _make_generation(old_bundle, "g1")
    _make_generation(new_bundle, "g1")

    now = time.time()
    _touch_mtime(old_bundle / "_manifest.json" if (old_bundle / "_manifest.json").exists() else old_bundle,
                 now - 40 * 86400)
    _touch_mtime(new_bundle / "_manifest.json" if (new_bundle / "_manifest.json").exists() else new_bundle,
                 now)

    result = cleanup_cache(root, max_age_days=30, keep_generations=99, dry_run=False)

    # The old bundle (marker older than cutoff, has a video.md) is pruned whole;
    # the new one survives.
    assert any(str(old_bundle) == path for path in result["deleted"])
    assert new_bundle.exists()
    assert not old_bundle.exists()


def test_cleanup_cache_skips_private_dirs(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    (root / "_internal").mkdir(parents=True)
    (root / "_internal" / "g1").mkdir()
    (root / "_internal" / "g1" / "video.md").write_text("# x")

    result = cleanup_cache(root, max_age_days=None, keep_generations=0, dry_run=False)

    assert result["candidate_count"] == 0
    assert (root / "_internal").exists()


def test_public_cleanup_cache_passes_video_md_markdown_name(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    bundle = root / "hash"
    _make_generation(bundle, "g1", with_markdown=True)

    result = cleanup_cache(root, max_age_days=None, keep_generations=0, dry_run=True)

    # The public facade uses markdown_names=("video.md",) so the video.md-bearing
    # generation is recognized as a bundle candidate.
    assert result["candidate_count"] == 1


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
