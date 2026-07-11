"""Version constants for cache-affecting Distill output."""

from __future__ import annotations

PIPELINE_VERSION = 20
DISTILL_VERSION = "0.1.0"

# Output-affecting source files covered by PIPELINE_SIGNATURE. This is the single
# source of truth: the signature test and the AGENTS.md recompute snippet both
# read this list, so a newly added output-affecting module cannot silently
# escape the signature. Editing any of these modules requires bumping
# PIPELINE_VERSION and recomputing PIPELINE_SIGNATURE (test_pipeline_signature
# enforces both).
SIGNED_MODULES = (
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
)

# Hash of output-affecting source files covered by the pipeline signature test.
PIPELINE_SIGNATURE = "be83164da16d1fb7ea0bdcfed72080e3d7a2ac4ca6ec55fcfe2930b5d869cdec"
