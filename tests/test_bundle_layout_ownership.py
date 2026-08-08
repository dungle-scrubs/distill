"""One module owns bundle layout, and a grep can prove it.

D-041 removes the bundle-reading code from `pipeline.py` and `source.py` in
favour of `BundleStore` calls, and the Gate 3->4 criterion states the result as
a property of the whole package rather than of those two files: *no module
outside `bundle_store.py` references generation naming or bundle layout*. A
criterion checked once by hand is a criterion that decays, so it is checked
here.

What this test can see: a literal layout name written into another module's
source. What it cannot see: a layout name assembled at runtime from fragments, a
name reached through a variable, or a caller that re-derives a path from a
string `bundle_store` handed it. It makes the easy way back to the duplicated
knowledge loud, and nothing more.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "distill"

OWNERS = frozenset({"bundle_store.py", "bundle_layout.py"})
"""Modules allowed to name the layout they own (deep split, Candidate 02).

``bundle_store.py`` remains the façade and the bundle identity owner;
``bundle_layout.py`` owns the filesystem layout names (``BundlePaths``,
``MarkerVerdict``) as the deep module behind that façade. Both may name the
layout literally; no other module may.
"""
OWNER = "bundle_store.py"  # legacy alias for prose

LAYOUT_NAMES = {
    "_manifest.json": "the bundle marker's filename",
    "_owner.json": "the ownership marker's filename",
    "video.md": "the render's filename",
    "transcript.json": "the transcript's filename",
    ".tmp.": "the staging directory prefix",
    "_locks": "the lock directory's name",
}
"""Literal on-disk names that only `bundle_store.py` may write down."""

GENERATION_NAMING = re.compile(r"""["']g\d+["']|g\{|removeprefix\(|GENERATION_PREFIX""")
"""How a module spells a **generation** name if it is deriving one itself.

A quoted `g1`, an f-string building `g{n}`, a `removeprefix` peeling the staging
prefix off one, or the prefix constant imported out of its owner.
"""

EXEMPT: dict[str, set[str]] = {
    # Prose about what this module deliberately does not write.
    "render.py": {"video.md"},
    "version.py": {"video.md", "transcript.json"},
}
"""Modules allowed one named mention, with the reason it is not a reference."""


def _sources() -> dict[str, str]:
    return {
        path.relative_to(PACKAGE_DIR).as_posix(): path.read_text()
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        if "__pycache__" not in path.relative_to(PACKAGE_DIR).parts
        and path.name not in OWNERS
    }


def _names(name: str) -> re.Pattern[str]:
    """Match `name` as a whole word, so a longer name is not a false hit.

    `source.py`'s `_youtube_locks` holds the **acquisition lease**, which is
    keyed by **lock key** and is not the bundle store's `_locks` directory; a
    substring match calls one the other.
    """
    return re.compile(rf"(?<![\w.]){re.escape(name)}")


def test_no_module_outside_the_store_names_a_bundle_layout_file() -> None:
    """R-57/D-041: the marker, the render, the transcript, staging, the locks."""
    offences = [
        f"{module} names {name!r} ({why})"
        for module, text in _sources().items()
        for name, why in LAYOUT_NAMES.items()
        if _names(name).search(text) and name not in EXEMPT.get(module, set())
    ]
    assert offences == []


def test_no_module_outside_the_store_derives_a_generation_name() -> None:
    """A generation is `g<N>` because `bundle_store` says so, nowhere else."""
    offences = [
        f"{module} derives a generation name: {match.group(0)!r}"
        for module, text in _sources().items()
        for match in GENERATION_NAMING.finditer(text)
    ]
    assert offences == []
