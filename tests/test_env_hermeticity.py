"""Tests for the autouse hermeticity fixture in ``tests/conftest.py`` (R-53).

The fixture is the only thing standing between the suite and the developer's
shell. Every DISTILL_* variable production reads is an input that, if inherited,
silently changes what the code under test does - so the fixture's neutralization
is itself production code for the suite, and gets tested like it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import neutralize_distill_environment

LEAKED_VARIABLES = (
    # Read by pipeline.configured/effective timeout resolution.
    "DISTILL_EFFECTIVE_TIMEOUT_MS",
    # Read by pipeline's timeout-probe guard; leaking it flips two invariants.
    "DISTILL_ENABLE_LONG_TIMEOUT_PROBE",
    # Read by local_vision's debug toggle.
    "DISTILL_LOCAL_VISION_DEBUG",
    # Read by config's environment layer; leaking it moves every run's output
    # root, so a test would publish bundles into the developer's own root.
    "DISTILL_OUTPUT_DIR",
    # A variable nobody has written yet: prefix-wide clearing must cover it.
    "DISTILL_SOME_FUTURE_SETTING",
)


def test_fixture_clears_every_inherited_distill_variable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    config_dir = home / ".distill"
    config_dir.mkdir(parents=True)

    with pytest.MonkeyPatch.context() as leaked_shell:
        for name in LEAKED_VARIABLES:
            leaked_shell.setenv(name, "leaked-from-the-developer-shell")

        with pytest.MonkeyPatch.context() as environment:
            neutralize_distill_environment(environment, home=home, config_dir=config_dir)

            for name in LEAKED_VARIABLES:
                assert name not in os.environ
            assert os.environ["DISTILL_CONFIG_DIR"] == str(config_dir)
            assert os.environ["HOME"] == str(home)

        # The context manager restores the shell's values, so the fixture's
        # neutralization is scoped to the test and does not corrupt the process.
        for name in LEAKED_VARIABLES:
            assert os.environ[name] == "leaked-from-the-developer-shell"


def test_fixture_clears_an_inherited_xdg_config_home(tmp_path: Path) -> None:
    """The one config input outside the DISTILL_ namespace (R-53).

    Config resolution walks `$XDG_CONFIG_HOME/distill`, so a variable the prefix
    clearing cannot see would hand the suite the developer's own config
    directory. It is cleared, not redirected: the throwaway HOME then carries
    the whole chain.
    """
    home = tmp_path / "home"
    config_dir = home / ".distill"
    config_dir.mkdir(parents=True)

    with pytest.MonkeyPatch.context() as leaked_shell:
        leaked_shell.setenv("XDG_CONFIG_HOME", "/somewhere/the/developer/configured")

        with pytest.MonkeyPatch.context() as environment:
            neutralize_distill_environment(environment, home=home, config_dir=config_dir)

            assert "XDG_CONFIG_HOME" not in os.environ


def test_fixture_keeps_test_selection_switches(tmp_path: Path) -> None:
    """DISTILL_RUN_* gates which tests run; clearing it would disable the smokes."""
    home = tmp_path / "home"
    config_dir = home / ".distill"
    config_dir.mkdir(parents=True)

    with pytest.MonkeyPatch.context() as leaked_shell:
        leaked_shell.setenv("DISTILL_RUN_NETWORK_TESTS", "1")

        with pytest.MonkeyPatch.context() as environment:
            neutralize_distill_environment(environment, home=home, config_dir=config_dir)

            assert os.environ["DISTILL_RUN_NETWORK_TESTS"] == "1"


def test_autouse_fixture_has_already_neutralized_this_process() -> None:
    """The fixture under test is autouse, so this test runs inside its effect."""
    assert "DISTILL_ENABLE_LONG_TIMEOUT_PROBE" not in os.environ
    assert "DISTILL_EFFECTIVE_TIMEOUT_MS" not in os.environ
    assert Path(os.environ["DISTILL_CONFIG_DIR"]).is_dir()
