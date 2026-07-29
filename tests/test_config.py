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
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from distill import cli, config, local_vision, pipeline
from distill.config import OPTION_ENV_VARIABLES, resolve_options
from distill.errors import DistillError
from distill.local_vision import DEFAULT_TIMEOUT_SEC
from distill.options import GENERAL_OPTION_NAMES, DistillOptions
from distill.source import validate_output_root
from distill.version import SIGNED_MODULES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

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


def test_an_absent_explicit_config_directory_is_authoritative_and_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit missing root means no config, never a lower stale config."""
    explicit = tmp_path / "not-created"
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(explicit))
    write_config(home() / ".config" / "distill", {"max_keyframes": 33})

    assert config.config_dir() == explicit
    assert DistillOptions.from_args({}).max_keyframes == 80


def test_an_explicit_config_path_that_is_not_a_directory_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "config-file"
    explicit.write_text("not a directory")
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(explicit))

    with pytest.raises(DistillError) as refusal:
        DistillOptions.from_args({})

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.stage == "options"
    assert refusal.value.details == {"path": str(explicit), "errno": "ENOTDIR"}


@pytest.mark.parametrize(
    ("variable", "relative_directory"),
    [
        ("DISTILL_CONFIG_DIR", Path("relative-config")),
        ("XDG_CONFIG_HOME", Path("relative-xdg") / "distill"),
    ],
)
def test_a_relative_config_environment_path_is_skipped(
    variable: str,
    relative_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Config environment paths never turn the process directory into policy."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv(
        variable,
        str(relative_directory.parent if variable == "XDG_CONFIG_HOME" else relative_directory),
    )
    write_config(tmp_path / relative_directory, {"max_keyframes": 7})
    write_config(home() / ".config" / "distill", {"max_keyframes": 9})

    assert DistillOptions.from_args({}).max_keyframes == 9


def test_an_unreadable_explicit_config_parent_is_a_typed_process_refusal(
    tmp_path: Path,
) -> None:
    """A real process reports EACCES for the config root instead of E_INTERNAL."""
    blocked_parent = tmp_path / "blocked"
    explicit = blocked_parent / "config"
    explicit.mkdir(parents=True)
    blocked_parent.chmod(0o000)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "distill.cli", "cache-doctor"],
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": str(PACKAGE_ROOT / "src"),
                "DISTILL_CONFIG_DIR": str(explicit),
                "HOME": str(home()),
            },
        )
    finally:
        blocked_parent.chmod(0o700)

    assert result.returncode == 2
    assert result.stdout == ""
    record = json.loads(result.stderr)
    assert record["code"] == "E_BAD_OPTIONS"
    assert record["stage"] == "options"
    assert record["details"] == {"path": str(explicit), "errno": "EACCES"}


def test_an_unreadable_implicit_config_candidate_is_a_typed_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "distill"
    monkeypatch.delenv("DISTILL_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    real_stat = Path.stat

    def deny_candidate(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == candidate:
            raise PermissionError(13, "Permission denied", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", deny_candidate)

    with pytest.raises(DistillError) as refusal:
        config.config_dir()

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.stage == "options"
    assert refusal.value.details == {"path": str(candidate), "errno": "EACCES"}


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


def test_the_directory_command_forwards_common_processing_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    def capture(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        seen.update(tool=tool, args=args)
        return {"ok": True}

    monkeypatch.setattr(cli, "call_registered_tool", capture)

    cli.main(
        [
            "process-video-directory",
            "recordings",
            "--max-keyframes",
            "17",
            "--no-ocr",
            "--local-vision-model",
            "reader",
            "--output-dir",
            "bundles",
            "--job-id",
            "batch",
        ]
    )

    assert capsys.readouterr().err == ""
    assert seen == {
        "tool": "process_video_directory",
        "args": {
            "path": "recordings",
            "recursive": False,
            "max_keyframes": 17,
            "ocr": False,
            "local_vision_model": "reader",
            "output_dir": "bundles",
            "job_id": "batch",
        },
    }


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


def test_a_general_config_file_cannot_pin_a_job_identifier(tmp_path: Path) -> None:
    """Run identity is per invocation even when reuse controls are configured."""
    path = write_config(tmp_path, {"job_id": "shared-run"})

    with pytest.raises(DistillError) as refusal:
        resolve_options({}, general_keys=GENERAL_OPTION_NAMES, base_dir=tmp_path)

    assert "job_id" not in GENERAL_OPTION_NAMES
    assert {"force_reprocess", "resume_partial"} <= set(GENERAL_OPTION_NAMES)
    assert refusal.value.details == {
        "path": str(path),
        "unknown_options": ["job_id"],
    }


def test_an_unknown_general_config_key_is_refused_with_the_file_and_keys(
    tmp_path: Path,
) -> None:
    """A typo cannot silently produce a bundle using the default option."""
    path = write_config(
        tmp_path,
        {"path": "/etc/passwd", "max_keyframes": 9, "nonsense": 1},
    )

    with pytest.raises(DistillError) as refusal:
        resolve_options(
            {"url": "https://youtu.be/abc"},
            general_keys=GENERAL_OPTION_NAMES,
            base_dir=tmp_path,
        )

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.stage == "options"
    assert refusal.value.details == {
        "path": str(path),
        "unknown_options": ["nonsense", "path"],
    }


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


def test_an_empty_file_output_root_is_not_a_setting(tmp_path: Path) -> None:
    """An empty file value follows the environment layer's no-setting rule."""
    write_config(tmp_path, {"output_dir": "", "max_keyframes": 7})

    resolved = resolve_options(
        {},
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


# --- What a configured value is refused by, and what a bad file is worth. -----
#
# A config file is an operator typing an option a day earlier, so the rules it
# meets have to be the rules the command line meets: one door, one refusal, one
# message. The file itself is the other half - a `distill.json` nobody can parse
# is a file somebody wrote and got wrong, which is not the same thing as a file
# that is not there.

MALFORMED_JSON = '{"max_keyframes": 5,'
"""A `distill.json` an operator saved mid-edit: valid up to the point it stops."""


def write_text_config(directory: Path, text: str) -> Path:
    """Plant a `distill.json` holding `text`, whatever `text` is or is not."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "distill.json"
    path.write_text(text)
    return path


def refusal_site(error: DistillError) -> tuple[str, str]:
    """The module and function that raised: which door the value was refused at.

    The innermost frame, because that is where the decision was made. A
    configured value and a typed one refused at the same site are refused by the
    same code, which is the claim this milestone rests on and the one an
    equality of two messages cannot make on its own.
    """
    traceback = error.__traceback__
    assert traceback is not None
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    code = traceback.tb_frame.f_code
    return Path(code.co_filename).name, code.co_name


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("ocr", {}),
        ("ocr", "nope"),
        ("whisper_model", {"name": "small"}),
        ("whisper_model", None),
        ("output_dir", {}),
    ],
    ids=[
        "boolean-object",
        "boolean-typo",
        "text-object",
        "text-null",
        "output-root-object",
    ],
)
def test_general_json_values_must_have_the_option_type(
    option: str,
    value: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """JSON types and boolean typos are refused before Python can coerce them."""
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    write_config(tmp_path, {option: value})

    with pytest.raises(DistillError) as configured:
        DistillOptions.from_args({})
    with pytest.raises(DistillError) as typed:
        DistillOptions.from_args({option: value})

    assert configured.value.code == typed.value.code == "E_BAD_OPTIONS"
    assert configured.value.stage == typed.value.stage == "options"
    assert configured.value.message == typed.value.message
    assert option in configured.value.message


@pytest.mark.parametrize(
    "value",
    [-5, 0, 2.5, "many"],
    ids=["below-the-floor", "zero", "not-whole", "not-a-number"],
)
def test_a_configured_number_is_refused_where_a_typed_one_is(
    value: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One door for both, proven by the record and by the frame that raised it.

    A config file that accepted a keyframe cap the command line refuses would be
    a second, softer set of rules for the same option - and the run that noticed
    would be the one that produced a **bundle** nobody could ask for again.
    """
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    path = write_config(tmp_path, {"max_keyframes": value})

    with pytest.raises(DistillError) as configured:
        DistillOptions.from_args({})
    with pytest.raises(DistillError) as typed:
        DistillOptions.from_args({"max_keyframes": value})

    assert configured.value.code == "E_BAD_OPTIONS"
    assert configured.value.stage == "options"
    assert configured.value.code == typed.value.code
    assert configured.value.stage == typed.value.stage
    assert configured.value.message == typed.value.message
    assert configured.value.details["configured_from"] == str(path)
    assert "configured_from" not in typed.value.details
    assert refusal_site(configured.value) == refusal_site(typed.value)
    assert refusal_site(configured.value)[0] == "options.py"


def test_a_malformed_general_config_file_is_refused_rather_than_read_as_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST: an unparsable `distill.json` was silently `{}` (D-011).

    Every option in it - the keyframe cap, the output root - was then the
    default, and the run went ahead and produced a **bundle** configured by
    nothing the operator wrote. A file that cannot be parsed is not a file
    without options in it.
    """
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    write_text_config(tmp_path, MALFORMED_JSON)

    with pytest.raises(DistillError) as refusal:
        DistillOptions.from_args({})

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.stage == "options"


def test_a_malformed_general_config_file_names_the_file_and_the_parse_failure(
    tmp_path: Path,
) -> None:
    """Which file, and where it stopped parsing - the two things a fix needs.

    Config resolution walks four directories, so "your config is broken" without
    a path sends an operator to look in the one they did not edit.
    """
    path = write_text_config(tmp_path, MALFORMED_JSON)

    with pytest.raises(DistillError) as refusal:
        config.general_config(tmp_path)

    details = refusal.value.details
    assert details["path"] == str(path)
    assert "line 1" in details["error"]


def test_a_non_utf8_general_config_file_is_refused_and_names_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "distill.json"
    path.write_bytes(b'{"max_keyframes": "\xff"}')

    with pytest.raises(DistillError) as refusal:
        config.general_config(tmp_path)

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.stage == "options"
    assert refusal.value.message == "config file is not UTF-8 text"
    assert refusal.value.details["path"] == str(path)


def test_a_general_config_file_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    """A JSON array parses and still names no option.

    Well-formed and unusable is the same mistake as malformed seen a moment
    later - it reaches the loader as a list and sets nothing - so it is answered
    the same way rather than passed over as an empty configuration.
    """
    write_text_config(tmp_path, '["max_keyframes"]')

    with pytest.raises(DistillError) as refusal:
        config.general_config(tmp_path)

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.details["received"] == "list"


def test_an_unreadable_general_config_file_is_refused_rather_than_treated_as_absent(
    tmp_path: Path,
) -> None:
    """A file this process may not read is a refusal, never a silent default.

    The same discipline `source_path_kind` holds to (D-022): absent is a fact,
    but "there is no configuration here" about a file whose contents are denied
    to us is a claim with nothing behind it, and it would run with defaults
    while the operator's own options sat on disk unread.
    """
    path = write_config(tmp_path, {"max_keyframes": 5})
    path.chmod(0o000)
    if os.access(path, os.R_OK):  # pragma: no cover - root reads anything
        pytest.skip("this process can read a mode 000 file")

    with pytest.raises(DistillError) as refusal:
        config.general_config(tmp_path)

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.stage == "options"
    assert refusal.value.details == {"path": str(path), "errno": "EACCES"}


def test_an_absent_general_config_file_is_not_a_refusal(tmp_path: Path) -> None:
    """No `distill.json` is the ordinary case, and stays the ordinary case.

    Stated as its own test because the refusals above are one over-eager `except`
    away from making a machine with no config file unable to run at all.
    """
    assert config.general_config(tmp_path) == {}


def test_a_broken_general_config_symlink_is_refused_as_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "distill.json"
    path.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(DistillError) as refusal:
        config.general_config(tmp_path)

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert refusal.value.stage == "options"
    assert refusal.value.details == {"path": str(path), "errno": "ENOENT"}


def test_a_local_vision_value_is_coerced_where_a_general_value_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two owners of one file, keeping the contracts they each had.

    `local_vision.py`'s reader coerces an unusable value to the default, which is
    right for a section whose failure mode is a slower probe. The general schema
    refuses, because a keyframe cap silently replaced by 80 is a **bundle** the
    operator did not ask for. The same file proves both, so the scope of the
    coercion is the section rather than the file it lives in.
    """
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    write_config(tmp_path, {"local_vision": {"timeout_sec": "soon"}, "max_keyframes": 5})

    assert local_vision.load_local_vision_config(tmp_path).timeout_sec == DEFAULT_TIMEOUT_SEC
    assert DistillOptions.from_args({}).max_keyframes == 5

    write_config(tmp_path, {"local_vision": {"timeout_sec": "soon"}, "max_keyframes": "many"})

    with pytest.raises(DistillError) as refusal:
        DistillOptions.from_args({})

    assert refusal.value.code == "E_BAD_OPTIONS"
    assert local_vision.load_local_vision_config(tmp_path).timeout_sec == DEFAULT_TIMEOUT_SEC


def test_a_malformed_local_vision_file_is_still_read_forgivingly(tmp_path: Path) -> None:
    """The malformed rule is the general schema's, and does not spread.

    `distill.local-vision.json` keeps its own forgiving reader: a broken one
    leaves the defaults in place and a run captions frames with them. Only the
    general file is fatal, which is what D-011 decided and what makes the two
    readers separate rather than duplicated.
    """
    (tmp_path / "distill.local-vision.json").write_text(MALFORMED_JSON)

    assert local_vision.load_local_vision_config(tmp_path).timeout_sec == DEFAULT_TIMEOUT_SEC


def test_a_non_utf8_local_vision_file_still_degrades_to_defaults(
    tmp_path: Path,
) -> None:
    (tmp_path / "distill.local-vision.json").write_bytes(b'{"timeout_sec": "\xff"}')

    assert local_vision.load_local_vision_config(tmp_path).timeout_sec == DEFAULT_TIMEOUT_SEC


def test_a_configured_option_changes_the_options_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Why this module is signed: what it resolves reaches the **bundle key**.

    A configured `max_keyframes` produces a different **options hash** from the
    default one, so editing this module can change which **bundle** a run
    publishes - the property ADR-0003 makes a module signed for.
    """
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    default_hash = DistillOptions.from_args({}).opts_hash("local")

    write_config(tmp_path, {"max_keyframes": 5})

    assert DistillOptions.from_args({}).opts_hash("local") != default_hash
    assert "config.py" in SIGNED_MODULES


def test_file_and_cli_values_produce_the_same_options_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    write_config(tmp_path, {"max_keyframes": 17})

    from_file = DistillOptions.from_args({}).opts_hash("local")
    from_cli = DistillOptions.from_args({"max_keyframes": 17}).opts_hash("local")

    assert from_file == from_cli


def test_file_output_roots_do_not_change_the_options_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISTILL_CONFIG_DIR", str(tmp_path))
    write_config(tmp_path, {"output_dir": str(home() / "first")})
    first = DistillOptions.from_args({}).opts_hash("local")

    write_config(tmp_path, {"output_dir": str(home() / "second")})
    second = DistillOptions.from_args({}).opts_hash("local")

    assert first == second


def test_a_malformed_config_file_leaves_the_cli_as_the_error_object(tmp_path: Path) -> None:
    """The refusal an operator's shell sees: the JSON record on stderr, exit 2.

    Driven through a real child process because the boundary's promise is about
    the process - an in-process raise proves the code and the stage and says
    nothing about the exit code, the stream, or whether a traceback came out
    with it. `cache-doctor` reads config and writes nothing, so this is the
    cheapest command that has to refuse.
    """
    write_text_config(tmp_path, MALFORMED_JSON)

    result = subprocess.run(
        [sys.executable, "-m", "distill.cli", "cache-doctor"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(PACKAGE_ROOT / "src"),
            "DISTILL_CONFIG_DIR": str(tmp_path),
            "HOME": str(home()),
        },
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    record = json.loads(result.stderr)
    assert record["code"] == "E_BAD_OPTIONS"
    assert record["stage"] == "options"
    assert record["details"]["path"] == str(tmp_path / "distill.json")


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
