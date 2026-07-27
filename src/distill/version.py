"""Pipeline signature, pipeline version, and every module's signed/exempt disposition.

This module owns `PIPELINE_VERSION` and `PIPELINE_SIGNATURE` (the cache identity
of Distill's own output-affecting source) and the recorded disposition of every
module under `src/distill/` - either signed by the signature or exempted with a
reason. It does not own the released package version: that is `DISTILL_VERSION`
in `release.py`, which is bundle content and is therefore signed.

It does not own cache lookup, the bundle key, or any output. Those live in
`pipeline.py`, `bundle_store.py` and `response.py`; this module only supplies
the constants they mix into the bundle key.

Per ADR-0003 signed-ness is a property, not a list: if editing a module can
change bundle content, it is a signed module. `tests/test_pipeline_signature.py`
enumerates `src/distill/` from disk, recursively, and fails when a module is
neither signed nor exempted, so adding a module forces an explicit decision.

What this mechanism does not catch: it detects unversioned change to Distill's
own source. It does not detect output changes caused by package data, a
dependency upgrade, an external tool such as ffmpeg or Tesseract, or a model
swap, and it cannot tell whether an exemption reason below is still true. It is
not a security control - the signature is a change detector, not a signature in
the cryptographic sense, and nothing verifies who produced it.
"""

from __future__ import annotations

PIPELINE_VERSION = 26

# Output-affecting source files covered by PIPELINE_SIGNATURE, named by their
# path relative to `src/distill/` in posix form so that modules in subpackages
# cannot collide with top-level ones. Editing any of these requires bumping
# PIPELINE_VERSION and recomputing PIPELINE_SIGNATURE (test_pipeline_signature
# enforces both). The signature is computed by iterating this tuple, so the
# order is part of the hash: entries are kept in alphabetical order to keep that
# ordering mechanical rather than incidental, and test_pipeline_signature
# enforces that too.
SIGNED_MODULES = (
    "bundle_store.py",
    "capabilities.py",
    "cli.py",
    "errors.py",
    "frame_selection.py",
    "grounding.py",
    "links.py",
    "local_vision.py",
    "ocr.py",
    "options.py",
    "pipeline.py",
    "progress.py",
    "redact_secrets.py",
    "release.py",
    "render.py",
    "response.py",
    "run_command.py",
    "source.py",
    "transcript.py",
    "vision_prompts.py",
)

# Modules deliberately outside the signature, keyed the same way, each with the
# recorded reason its contents cannot change bundle content. An exemption is a
# claim about the module's role; if that role changes, the module becomes
# signed.
EXEMPT_MODULES: dict[str, str] = {
    "__init__.py": (
        "Re-exports DISTILL_VERSION from the signed release.py and nothing "
        "else. response.py stamps manifests from release.py directly, so this "
        "re-export reaches no bundle and editing it cannot change bundle "
        "content."
    ),
    "cache_doctor.py": (
        "Reports on bundles under an output root and never writes, deletes or "
        "repairs anything: it formats a BundleStore survey and an unapplied "
        "PrunePlan. It cannot make a served bundle differ from what the current "
        "code would produce, because it produces no bundle content and reaches "
        "no bundle write path at all. Read-only is what this exemption rests "
        "on; a repair mode added here would make the module signed."
    ),
    "job_store.py": (
        "Owns job records and their identifier domain, written under the cache "
        "root's _jobs/ directory. Those records live outside every bundle "
        "generation, are never published into a bundle, and are never read back "
        "as pipeline input: a run reads a job record to answer a caller, never "
        "to decide what to produce. Rejecting an out-of-domain job id can end a "
        "run before it produces a bundle, which changes whether a bundle exists "
        "rather than what a bundle contains."
    ),
    "measure.py": (
        "Offline spike-tuning harness, not imported by cli.py or pipeline.py. It "
        "writes measurement JSON outside any bundle and is exercised only by "
        "tests and the eval harness."
    ),
    "version.py": (
        "Holds PIPELINE_SIGNATURE itself and so cannot hash itself: any signed "
        "value written here would change the bytes being hashed. What remains "
        "in this module is covered another way. PIPELINE_VERSION is written "
        "into every manifest, but it also participates in the bundle key, so "
        "changing it gives every bundle a new address rather than a stale hit. "
        "SIGNED_MODULES and EXEMPT_MODULES are never written into a bundle; "
        "they only decide what the signature covers, and test_pipeline_signature "
        "checks them against the modules actually on disk."
    ),
}

# Hash of output-affecting source files covered by the pipeline signature test.
PIPELINE_SIGNATURE = "9d98001987fadec8758746316c91fc8105cc944ef2f0d5971fb7422b96540968"
