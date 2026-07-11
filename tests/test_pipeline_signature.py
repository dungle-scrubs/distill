from __future__ import annotations

import hashlib
import json
from pathlib import Path

from distill.version import PIPELINE_SIGNATURE, PIPELINE_VERSION, SIGNED_MODULES

HISTORY_PATH = Path(__file__).resolve().parent / "pipeline_signature_history.json"


def _recompute_signature() -> str:
    scripts = Path(__file__).resolve().parents[1] / "src" / "distill"
    digest = hashlib.sha256()
    for name in SIGNED_MODULES:
        digest.update((scripts / name).read_bytes())
    return digest.hexdigest()


def _load_history() -> dict[str, int]:
    return {sig: int(version) for sig, version in json.loads(HISTORY_PATH.read_text()).items()}


def test_pipeline_signature_matches_output_affecting_modules() -> None:
    # The recorded signature must match the current bytes of the signed modules.
    # If this fails, a signed module changed: recompute PIPELINE_SIGNATURE.
    assert _recompute_signature() == PIPELINE_SIGNATURE


def test_signature_change_requires_a_version_bump() -> None:
    """Couple a signature change to a required PIPELINE_VERSION bump.

    ``pipeline_signature_history.json`` maps every published signature to the
    ``PIPELINE_VERSION`` it shipped under. Because ``PIPELINE_VERSION`` is part
    of the cache identity (``opts_hash``), reusing a version after a signed
    module changed would serve stale-logic bundles as cache hits. This test
    forces the pairing to stay 1:1 and monotonic, so recomputing the signature
    without bumping the version fails here.
    """
    history = _load_history()

    # The current signature must be registered against the current version.
    assert PIPELINE_SIGNATURE in history, (
        "signed modules changed: add the new PIPELINE_SIGNATURE to "
        "pipeline_signature_history.json under a new PIPELINE_VERSION"
    )
    assert history[PIPELINE_SIGNATURE] == PIPELINE_VERSION

    # Each distinct signature owns a distinct version (no reuse), and the current
    # version is the newest — so a signature change cannot keep an old version.
    assert len(set(history.values())) == len(history), "a PIPELINE_VERSION is reused across signatures"
    assert max(history.values()) == PIPELINE_VERSION
