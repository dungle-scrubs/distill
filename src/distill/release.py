"""The released package version, stamped into every manifest.

This module owns `DISTILL_VERSION`: the version the Distill package is released
under. `bundle.py` writes it into every manifest as `distill_version`, so its
value is durable bundle content and this module is a signed module (ADR-0003) -
editing it changes what the pipeline would produce for an unchanged bundle key.
It must stay in step with `version` in `pyproject.toml`, which is the version
the wheel is published under.

It does not own the pipeline signature, the pipeline version, or the
signed/exempt tables; those live in `version.py`. The split exists because
`version.py` holds `PIPELINE_SIGNATURE` and so cannot be hashed into it, while
this constant can be and must be.
"""

from __future__ import annotations

DISTILL_VERSION = "0.2.0"
