"""Where a run's options come from when nobody typed them on the command line.

The subject is `distill.config`: which directory configuration is read from,
which keys of `distill.json` are general options rather than local-vision ones,
and which layer wins when more than one of them names the same option.

Read through the seams an operator actually reaches - `DistillOptions.from_args`
and `validate_output_root` - rather than through the loader alone, because a
resolution order that is right inside the loader and never applied to a run is
the bug this milestone exists to remove.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from distill import config, local_vision, pipeline
from distill.config import OPTION_ENV_VARIABLES, resolve_options
from distill.errors import DistillError
from distill.options import GENERAL_OPTION_NAMES, DistillOptions
from distill.source import validate_output_root

# The DISTILL_* variables that steer diagnosis rather than a run's options: a
# traceback, a debug dump, the effective timeout, the long-probe opt-in. None of
# them is an option layer, and the resolution order must not adopt one.
DIAGNOSTIC_VARIABLES = (
    "DISTILL_TRACEBACK",
    "DISTILL_LOCAL_VISION_DEBUG",
    "DISTILL_EFFECTIVE_TIMEOUT_MS",
    "DISTILL_ENABLE_LONG_TIMEOUT_PROBE",
)


def write_config(directory: Path, payload: dict[str, object]) -> Path:
    """Plant a general `distill.json` in `directory` and return the file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "distill.json"
    path.write_text(json.dumps(payload))
    return path


def home() -> Path:
    """The throwaway home the hermeticity fixture pinned this test process to."""
    return Path(os.environ["HOME"])


def output_root(options: DistillOptions) -> Path:
    """The root a run configured this way would publish under, without creating it."""
    return validate_output_root(options.output_dir, create=False)


def test_a_general_key_in_distill_json_reaches_the_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`distill.json` is a Distill config file, not a local-vision config file.

    Only its nested `local_vision` object was ever read, so every general key an
    operator wrote beside that object - a keyframe cap, an output root - was
    read past in silence.
    """
    write_config(tmp_path, {"max_keyframes": 5})
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))

    assert DistillOptions.from_args({}).max_keyframes == 5


def test_output_dir_in_distill_json_sets_the_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = home() / "configured-root"
    write_config(tmp_path, {"output_dir": str(configured)})
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))

    assert output_root(DistillOptions.from_args({})) == configured.resolve()


def test_distill_output_dir_sets_the_output_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one option an environment variable may set, and the variable that sets it."""
    configured = home() / "environment-root"
    monkeypatch.setenv("DISTILL_OUTPUT_DIR", str(configured))

    assert output_root(DistillOptions.from_args({})) == configured.resolve()


def test_each_layer_overrides_the_one_below_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI > env > file > default, taken one layer at a time.

    One option carried through all four layers, because the order is only
    proven by a case where each layer has something below it to overrule.
    """
    default_root = home() / ".cache" / "distill"
    from_file = home() / "file-root"
    from_environment = home() / "environment-root"
    from_command_line = home() / "command-line-root"

    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    assert output_root(DistillOptions.from_args({})) == default_root.resolve()

    write_config(tmp_path, {"output_dir": str(from_file)})
    assert output_root(DistillOptions.from_args({})) == from_file.resolve()

    monkeypatch.setenv("DISTILL_OUTPUT_DIR", str(from_environment))
    assert output_root(DistillOptions.from_args({})) == from_environment.resolve()

    command_line = DistillOptions.from_args({"output_dir": str(from_command_line)})
    assert output_root(command_line) == from_command_line.resolve()


def test_a_diagnostic_variable_changes_no_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four diagnostic variables are not an option layer and never were.

    They are read where they are used - the error boundary, the vision debug
    dump, the timeout resolution - and adopting one into the resolution order
    would make a debugging switch change what a run produces.
    """
    unconfigured = DistillOptions.from_args({"job_id": "fixed"})

    for name in DIAGNOSTIC_VARIABLES:
        monkeypatch.setenv(name, "1")

    assert DistillOptions.from_args({"job_id": "fixed"}) == unconfigured


def test_a_config_under_xdg_config_home_is_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    write_config(tmp_path / "distill", {"max_keyframes": 11})

    assert DistillOptions.from_args({}).max_keyframes == 11


def test_a_config_under_home_config_distill_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    write_config(home() / ".config" / "distill", {"max_keyframes": 12})

    assert DistillOptions.from_args({}).max_keyframes == 12


def test_a_config_under_the_dot_distill_directory_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oldest location still resolves, so an existing install keeps working."""
    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    write_config(home() / ".distill", {"max_keyframes": 13})

    assert DistillOptions.from_args({}).max_keyframes == 13


def test_the_config_directories_do_not_merge_with_each_other(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One directory is *the* config directory; the rest are not consulted at all.

    Merging would make an option's value depend on which of four directories
    happened to exist on the machine, so a key that lives only in a lower
    directory has to be absent rather than filled in from there.
    """
    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    write_config(tmp_path / "distill", {"max_keyframes": 21})
    write_config(home() / ".config" / "distill", {"max_keyframes": 22, "whisper_model": "medium"})
    write_config(home() / ".distill", {"max_keyframes": 23, "whisper_language": "de"})

    options = DistillOptions.from_args({})

    assert options.max_keyframes == 21
    assert options.whisper_model == "small"
    assert options.whisper_language == "en"


def test_distill_config_dir_wins_over_every_other_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home() / "xdg"))
    write_config(tmp_path, {"max_keyframes": 31})
    write_config(home() / "xdg" / "distill", {"max_keyframes": 32})
    write_config(home() / ".config" / "distill", {"max_keyframes": 33})
    write_config(home() / ".distill", {"max_keyframes": 34})

    assert DistillOptions.from_args({}).max_keyframes == 31


@pytest.mark.parametrize(
    "tool",
    [pipeline.cache_doctor, pipeline.cleanup_distill_cache],
    ids=["cache-doctor", "cleanup-cache"],
)
def test_a_root_reading_tool_reads_the_configured_root(
    tool: Callable[[dict[str, object]], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The tools that never build options resolve the same layers anyway.

    A doctor reporting on `~/.cache/distill` while every run publishes into the
    configured root is a report about the wrong directory, and it reads as an
    empty cache rather than as a question asked in the wrong place.
    """
    configured = home() / "configured-root"
    write_config(tmp_path, {"output_dir": str(configured)})
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))

    assert tool({})["root"] == str(configured.resolve())


def test_a_job_lookup_reads_the_configured_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The same, for the tool whose answer is a record rather than a root.

    Job records live under the output root, so a lookup against the default one
    reports a job that ran as missing. The root a lookup validates is created,
    which is what makes which root it used observable.
    """
    configured = home() / "configured-root"
    write_config(tmp_path, {"output_dir": str(configured)})
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))

    with pytest.raises(DistillError) as refusal:
        pipeline.get_job_status({"job_id": "distill-nothing"})

    assert refusal.value.code == "E_JOB_NOT_FOUND"
    assert configured.is_dir()


def test_the_environment_layer_names_only_the_output_root() -> None:
    """The environment sets one option, and no diagnostic variable is that option.

    Held against the table rather than against behaviour, because the failure
    this guards against is a variable *added* to the layer: a debug toggle read
    as an option would change what a run produces, and the run that proves it
    would be the one nobody ran.
    """
    assert set(OPTION_ENV_VARIABLES) == {"output_dir"}
    assert set(OPTION_ENV_VARIABLES.values()) == {"DISTILL_OUTPUT_DIR"}
    assert not set(OPTION_ENV_VARIABLES.values()) & set(DIAGNOSTIC_VARIABLES)


def test_a_config_file_cannot_introduce_a_key_that_is_not_an_option(
    tmp_path: Path,
) -> None:
    """A config file sets options, and `general_keys` is the whole list of them.

    A file that names something else - a source path, a batch limit, a typo -
    contributes nothing, so a `distill.json` can never hand a tool an argument
    the caller did not give it.
    """
    write_config(tmp_path, {"path": "/etc/passwd", "max_keyframes": 9, "nonsense": 1})

    resolved = resolve_options(
        {"url": "https://youtu.be/abc"},
        general_keys=GENERAL_OPTION_NAMES,
        base_dir=tmp_path,
    )

    assert resolved == {"url": "https://youtu.be/abc", "max_keyframes": 9}


def test_the_local_vision_section_is_not_a_general_option(tmp_path: Path) -> None:
    """`{"local_vision": {...}}` is a section with another owner, not an option.

    The general schema is every other top-level key, so the section is dropped
    here rather than offered as a value for an option called `local_vision`.
    `local_vision.py` reads the same file for its own half.
    """
    write_config(tmp_path, {"local_vision": {"model": "qwen3-vl:32b"}, "max_keyframes": 4})

    assert config.general_config(tmp_path) == {"max_keyframes": 4}
    assert local_vision.load_local_vision_config(tmp_path).model == "qwen3-vl:32b"


def test_an_unused_flag_does_not_overrule_a_configured_value(tmp_path: Path) -> None:
    """`None` in the argument mapping is a flag nobody typed, not a request.

    That is how a parser spells an option left off the command line, and a
    configured value has to survive one - otherwise the file layer would be
    overwritten by the absence of the layer above it.
    """
    write_config(tmp_path, {"max_keyframes": 7})

    resolved = resolve_options(
        {"max_keyframes": None},
        general_keys=GENERAL_OPTION_NAMES,
        base_dir=tmp_path,
    )

    assert resolved == {"max_keyframes": 7}


def test_local_vision_reads_the_same_config_directory() -> None:
    """One answer to where config lives, for both readers.

    `distill.json` and `distill.local-vision.json` are read from the same
    directory or an operator has two config directories and no way to tell
    which file is being read from which.
    """
    assert local_vision.config_dir is config.config_dir


def test_the_module_comment_states_what_it_owns() -> None:
    """The module's own inputs, named in the comment that claims to describe it.

    Derived from the module rather than restated here: a variable or a filename
    this module starts reading without saying so fails, which is the drift a
    hand-written comment about ownership actually suffers.
    """
    comment = config.__doc__ or ""

    for name in (
        config.CONFIG_DIR_ENV,
        config.XDG_CONFIG_HOME_ENV,
        config.GENERAL_CONFIG_FILENAME,
        config.LOCAL_VISION_SECTION,
        *config.OPTION_ENV_VARIABLES.values(),
        *DIAGNOSTIC_VARIABLES,
    ):
        assert name in comment, f"the module comment does not mention {name}"
    assert "does not own" in comment
