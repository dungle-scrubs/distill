from __future__ import annotations

import hashlib
from pathlib import Path

from saccade.version import PIPELINE_SIGNATURE, PIPELINE_VERSION


def test_pipeline_signature_matches_output_affecting_modules() -> None:
    scripts = Path(__file__).resolve().parents[1] / "src" / "saccade"
    digest = hashlib.sha256()
    for name in [
        "redact_secrets.py",
        "frame_selection.py",
        "transcript.py",
        "ocr.py",
        "bundle.py",
        "options.py",
        "local_vision.py",
        "vision_prompts.py",
        "grounding.py",
        "pipeline.py",
    ]:
        digest.update((scripts / name).read_bytes())
    assert PIPELINE_VERSION == 10
    assert digest.hexdigest() == PIPELINE_SIGNATURE
