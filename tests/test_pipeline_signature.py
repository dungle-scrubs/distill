from __future__ import annotations

import hashlib
from pathlib import Path

from distill.version import PIPELINE_SIGNATURE, PIPELINE_VERSION


def test_pipeline_signature_matches_output_affecting_modules() -> None:
    scripts = Path(__file__).resolve().parents[1] / "src" / "distill"
    digest = hashlib.sha256()
    for name in [
        "redact_secrets.py",
        "frame_selection.py",
        "transcript.py",
        "ocr.py",
        "bundle.py",
        "links.py",
        "options.py",
        "local_vision.py",
        "vision_prompts.py",
        "grounding.py",
        "pipeline.py",
    ]:
        digest.update((scripts / name).read_bytes())
    assert PIPELINE_VERSION == 19
    assert digest.hexdigest() == PIPELINE_SIGNATURE
