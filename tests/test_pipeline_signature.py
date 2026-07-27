"""Tests for the pipeline signature and its completeness.

These tests enforce that the signature covers the current bytes of every signed
module, that a signature change is paired with a `PIPELINE_VERSION` bump, and
that every module under `src/distill/` carries an explicit disposition: signed,
or exempted with a recorded reason.

What the completeness check does not do: it makes omission an explicit act, not
a proof of correctness. It cannot tell whether an exemption reason is true, and
it does not see output changes that come from package data or a dependency
upgrade rather than from Distill's own source.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from distill.release import DISTILL_VERSION
from distill.version import (
    EXEMPT_MODULES,
    PIPELINE_SIGNATURE,
    PIPELINE_VERSION,
    SIGNED_MODULES,
)

HISTORY_PATH = Path(__file__).resolve().parent / "pipeline_signature_history.json"
PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "distill"


def _modules_under(root: Path) -> set[str]:
    """Enumerate `*.py` modules under `root`, named relative to it in posix form.

    Recursive, because a module in a subpackage affects output exactly as much
    as a top-level one, and relative rather than bare names, because two files
    with the same basename in different directories are different modules.
    """
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.relative_to(root).parts
    }


def _package_modules() -> set[str]:
    """The `*.py` modules that exist on disk under `src/distill/`."""
    return _modules_under(PACKAGE_DIR)


def _assert_every_module_is_classified(module_names: set[str]) -> None:
    """Fail when a named module has no recorded disposition.

    Pure over the set of names so the negative case can pass a module name that
    does not exist, rather than writing a file into the package under test.
    """
    unclassified = sorted(module_names - set(SIGNED_MODULES) - set(EXEMPT_MODULES))
    assert not unclassified, (
        f"modules under src/distill/ are neither signed nor exempted: {unclassified}. "
        "Add each to SIGNED_MODULES if editing it can change bundle content, "
        "or to EXEMPT_MODULES with a recorded reason."
    )


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


def test_every_module_is_signed_or_exempted() -> None:
    """Every module under `src/distill/` carries an explicit disposition.

    Per ADR-0003 signed-ness is a property, not a list: if editing a module can
    change bundle content, it is signed. Enumerating the directory - rather than
    trusting `SIGNED_MODULES` to be complete - is what makes an omission a
    failing test instead of a silently stale cache.
    """
    _assert_every_module_is_classified(_package_modules())


def test_signed_and_exempt_modules_exist_and_do_not_overlap() -> None:
    """Both dispositions name real modules, and no module carries both."""
    on_disk = _package_modules()
    missing_signed = sorted(set(SIGNED_MODULES) - on_disk)
    missing_exempt = sorted(set(EXEMPT_MODULES) - on_disk)
    assert not missing_signed, f"SIGNED_MODULES names modules that do not exist: {missing_signed}"
    assert not missing_exempt, f"EXEMPT_MODULES names modules that do not exist: {missing_exempt}"

    both = sorted(set(SIGNED_MODULES) & set(EXEMPT_MODULES))
    assert not both, f"modules are both signed and exempted: {both}"

    assert len(set(SIGNED_MODULES)) == len(SIGNED_MODULES), "SIGNED_MODULES contains a duplicate"

    # The signature hashes SIGNED_MODULES in order, so the order is part of the
    # hash. Alphabetical is the recorded convention; assert it so appending a
    # new entry at the end fails here instead of quietly making the ordering
    # incidental again.
    assert list(SIGNED_MODULES) == sorted(SIGNED_MODULES), "SIGNED_MODULES is not in alphabetical order"


def test_every_exemption_records_a_reason() -> None:
    """An exemption without a stated reason is not a recorded decision."""
    unexplained = sorted(name for name, reason in EXEMPT_MODULES.items() if not reason.strip())
    assert not unexplained, f"exempted modules without a recorded reason: {unexplained}"


def test_a_new_module_fails_the_check_until_signed_or_exempted() -> None:
    """A module added to the package fails the completeness check on sight.

    The added module is nested, because a module in a future subpackage is the
    omission this check most needs to catch. No file is written: the check is a
    pure function over module names, so the package under test stays untouched
    even if this process is killed mid-test.
    """
    added = "vision/prompts.py"
    assert added not in set(SIGNED_MODULES) | set(EXEMPT_MODULES)

    with pytest.raises(AssertionError) as failure:
        _assert_every_module_is_classified(_package_modules() | {added})

    assert added in str(failure.value)


def test_enumeration_recurses_into_subpackages_and_ignores_caches(tmp_path: Path) -> None:
    """Enumeration sees nested modules, names them relative, and skips `__pycache__`.

    A non-recursive walk would leave a module in a subpackage neither signed,
    nor exempted, nor flagged - the silent omission the check exists to
    prevent. Run against a temporary tree so the real package is never written
    to.
    """
    (tmp_path / "vision").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "top.py").write_text("")
    (tmp_path / "vision" / "prompts.py").write_text("")
    (tmp_path / "vision" / "top.py").write_text("")
    (tmp_path / "__pycache__" / "cached.py").write_text("")

    assert _modules_under(tmp_path) == {"top.py", "vision/prompts.py", "vision/top.py"}


def test_released_version_matches_the_packaging_metadata() -> None:
    """`DISTILL_VERSION` and pyproject's `version` must not drift apart.

    Splitting `DISTILL_VERSION` out of `version.py` so it could be signed left
    the same value recorded in two places: `release.py`, which stamps every
    manifest, and `pyproject.toml`, which the wheel is published under. Nothing
    but this test keeps them equal, and a mismatch is invisible until a bundle
    claims one version while the installed package reports another.
    """
    pyproject = tomllib.loads((PACKAGE_DIR.parents[1] / "pyproject.toml").read_text())

    assert pyproject["project"]["version"] == DISTILL_VERSION
