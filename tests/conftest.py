from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

# Distill takes runtime input from the DISTILL_* namespace: the config dir, the
# effective timeout, the long-probe opt-in, the local-vision debug toggle. Any of
# them inherited from the developer's shell changes what production code does
# mid-test, so the fixture clears the whole prefix instead of an enumerated list.
# A variable added to src/distill later is then hermetic by default rather than
# hermetic by somebody remembering to extend a list here.
DISTILL_ENV_PREFIX = "DISTILL_"
# The one carve-out: DISTILL_RUN_* switches select which tests run (the network
# and rapid-mlx smokes). They are read by the suite, never by src/distill, so
# clearing them would silently disable the very tests they exist to enable.
TEST_SELECTION_ENV_PREFIX = "DISTILL_RUN_"


def neutralize_distill_environment(
    environment: pytest.MonkeyPatch,
    *,
    home: Path,
    config_dir: Path,
) -> None:
    """Point Distill's environment inputs at throwaway directories.

    Exposed as a plain function so a test can drive it directly and prove the
    neutralization actually happens; the autouse fixture below is its only
    production caller.
    """
    for name in list(os.environ):
        if name.startswith(DISTILL_ENV_PREFIX) and not name.startswith(
            TEST_SELECTION_ENV_PREFIX
        ):
            environment.delenv(name, raising=False)
    environment.delenv("CONFIG_DIR", raising=False)
    environment.setenv("HOME", str(home))
    environment.setenv("USERPROFILE", str(home))
    environment.setenv("DISTILL_CONFIG_DIR", str(config_dir))


@pytest.fixture
def fake_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, str], Path]:
    """Install a fake external tool on `PATH` and return the installer.

    Every external tool now runs through `run_command`, which spawns a real
    child in its own process group and drains real pipes. A patched `Popen` can
    no longer stand in for a tool, so a fake is a real executable: a Python
    script with the test interpreter in its shebang, needing no shell.

    `PATH` is replaced by the fake directory alone, so a tool a test does not
    install is genuinely absent - a degradation test cannot pass merely because
    the machine running it happens to lack ffmpeg, nor fail because it has it.
    """
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(bin_dir))

    def install(name: str, script: str) -> Path:
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{script}")
        path.chmod(0o755)
        return path

    return install


@pytest.fixture(autouse=True)
def hermetic_user_directories(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    test_home = tmp_path_factory.mktemp("home")
    config_dir = test_home / ".distill"
    config_dir.mkdir()

    with pytest.MonkeyPatch.context() as environment:
        neutralize_distill_environment(
            environment, home=test_home, config_dir=config_dir
        )
        yield


def unused_pid() -> int:
    """A pid no live process holds, for standing in for a holder that is gone.

    Acquisition decides a lock is abandoned by asking the kernel whether the
    recorded pid still exists, so a test about an abandoned lock needs a pid
    that genuinely does not. Searching downward from the top of the pid space
    finds one without spawning and reaping a child, whose pid the kernel could
    hand straight back to something else.
    """
    for candidate in range(2**15 - 1, 1024, -1):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except OSError:
            continue
    raise RuntimeError("no unused pid found")
