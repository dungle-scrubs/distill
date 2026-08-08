"""Bundle layout — the filesystem shape a **bundle** has on disk.

This is the deep module behind the ``bundle_store`` façade (Candidate 02).
It owns the question a **bundle** is: which directory holds one, what proves
it (the **bundle marker**), and where every file under a **bundle key** lives.
It is the only place that names ``_manifest.json``, ``_owner.json``, ``g<N>``,
``.tmp.g<N>``, ``frames/``, ``video.md`` or ``transcript.json`` as literals,
and the only place that decides whether a path may be written to.

The invariant it holds — *a directory is a bundle only if it carries a bundle
marker* — is proved by ``MarkerVerdict``. A marker is a non-symlink regular
file holding a JSON object whose recorded bundle identity equals the directory
name, accepting either the current ``bundle_key`` field or the legacy
``source_hash`` field (D-017). The **published marker** is additionally
schema-valid because it must name an **active generation**.

``BundleStore`` remains the façade that callers import from; this module is
where the layout decisions live, hidden behind the narrow ``BundlePaths``
and ``MarkerVerdict`` interfaces. Deleting this module would require
reimplementing bundle recognition and path confinement in every caller —
the deletion test that makes it deep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"
"""The **published marker**: present only once a generation has been published."""

OWNERSHIP_MARKER_NAME = "_owner.json"
"""The **ownership marker** ``begin`` writes first, so a directory is identifiable
as Distill-owned from its first moment — before any manifest exists (R-11,
D-025)."""

RENDER_NAME = "video.md"
SELF_CONTAINED_RENDER_NAME = "video.self-contained.md"
TRANSCRIPT_NAME = "transcript.json"
FRAMES_DIR_NAME = "frames"
GENERATION_PREFIX = "g"
STAGING_PREFIX = ".tmp."

SCRATCH_DIR_NAME = "_scratch"
STAGE_RESULT_PREFIX = "_"
STAGE_RESULT_SUFFIX = ".json"
STAGE_RESULT_GLOB = f"{STAGE_RESULT_PREFIX}*{STAGE_RESULT_SUFFIX}"
STAGE_RESULT_SCHEMA_VERSION = 1

MarkerKind = Literal["published", "owned", "not_bundle", "unreadable"]


@dataclass(frozen=True)
class BundlePaths:
    """Where every file under a **bundle key** lives."""

    root: Path
    generation: Path
    frames: Path
    manifest: Path
    transcript: Path
    markdown: Path
    self_contained_markdown: Path


@dataclass(frozen=True)
class MarkerVerdict:
    """Why a directory is, or is not, a **bundle**."""

    kind: MarkerKind
    reason: str
    bundle_key: str | None = None
    manifest: dict[str, Any] | None = None

    @property
    def is_bundle(self) -> bool:
        """Whether the directory is a **bundle**: something may be served from it."""
        return self.kind == "published"

    @property
    def is_distill_owned(self) -> bool:
        """Whether Distill wrote the directory, published or not."""
        return self.kind in ("published", "owned")
