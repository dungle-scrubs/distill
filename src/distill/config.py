"""Where a run's options come from before anyone types one, and which layer wins.

This module owns three things:

- **The config directory.** `DISTILL_CONFIG_DIR`, then `$XDG_CONFIG_HOME/distill`,
  then `~/.config/distill`, then `~/.distill`; the first that exists is *the*
  config directory and the rest are not read. Directories do not merge, because
  an option whose value depends on which of four directories happens to exist on
  a particular machine is a value nobody can predict from the file they edited.
- **The general schema over `distill.json`.** Every top-level key of that file
  except the nested `local_vision` object, which belongs to another owner (see
  below). A key here names an option; a key naming nothing is passed over.
- **The resolution order.** CLI > environment > file > default (the environment
  layer being `DISTILL_OUTPUT_DIR`, the one option a machine rather than a run
  decides), applied by
  folding the layers into the argument mapping that reaches
  `DistillOptions.from_args`. A configured value therefore arrives at exactly
  the door a typed value arrives at, and is validated by the same code rather
  than by a second, softer set of rules.

What it does not own:

- **Local-vision configuration.** `distill.local-vision.json` and the nested
  `local_vision` object stay with `local_vision.py`, which keeps its own
  forgiving reader and its own coercions. It resolves *where* config lives
  through this module, so the two agree on the directory and disagree about
  nothing else.
- **What an option means, or which values it may take.** `options.py` owns the
  option table and the numeric domains; this module is told which keys are
  general options and never interprets one.
- **The diagnostic environment variables.** `DISTILL_TRACEBACK`,
  `DISTILL_LOCAL_VISION_DEBUG`, `DISTILL_EFFECTIVE_TIMEOUT_MS` and
  `DISTILL_ENABLE_LONG_TIMEOUT_PROBE` steer diagnosis, not a run's options. They
  are read where they are used and are not part of this resolution order; a
  debugging switch must not change what a run produces.
- **Output root policy.** Whether a root is a place Distill may write is
  `source.validate_output_root`'s question; this module only decides which
  string is offered to it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CONFIG_DIR_ENV = "DISTILL_CONFIG_DIR"
XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"
GENERAL_CONFIG_FILENAME = "distill.json"
LOCAL_VISION_SECTION = "local_vision"

OPTION_ENV_VARIABLES: dict[str, str] = {
    "output_dir": "DISTILL_OUTPUT_DIR",
}
"""The options an environment variable may set, and the variable that sets one.

A table rather than a prefix rule, so the environment layer is a list somebody
decided on: a `DISTILL_*` variable is an option input only by appearing here,
and the diagnostic variables cannot become option inputs by being named well.
The output root is the one option that belongs in the environment - it is where
a machine keeps its bundles, which is a property of the machine rather than of
the processing being asked for (ADR-0004).
"""


def config_dir_candidates() -> tuple[Path, ...]:
    """The directories configuration may live in, highest precedence first.

    `DISTILL_CONFIG_DIR` and `$XDG_CONFIG_HOME` contribute a candidate only when
    they are set to something: an empty variable is a variable nobody meant to
    set, and reading `distill.json` out of the process's working directory
    because `XDG_CONFIG_HOME=""` was exported is not configuration, it is an
    accident.
    """
    candidates: list[Path] = []
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        candidates.append(Path(override).expanduser())
    xdg_config_home = os.environ.get(XDG_CONFIG_HOME_ENV)
    if xdg_config_home:
        candidates.append(Path(xdg_config_home).expanduser() / "distill")
    home = Path.home()
    candidates.append(home / ".config" / "distill")
    candidates.append(home / ".distill")
    return tuple(candidates)


def config_dir() -> Path:
    """The one directory configuration is read from.

    The first candidate that exists as a directory, and when none does, the
    highest-precedence candidate - so an operator who points `DISTILL_CONFIG_DIR`
    at a directory they have not created yet is answered with the directory they
    named rather than with whatever older location happens to be on the machine.

    A path that exists but is not a directory is not a config directory, so
    resolution continues past it: a file called `distill` under
    `$XDG_CONFIG_HOME` holds no `distill.json` and never will.
    """
    candidates = config_dir_candidates()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _read_json(path: Path) -> dict[str, Any]:
    """The JSON object at `path`, or nothing when there is not one there.

    Absence, unreadability and malformed content all answer the same way here
    today: no configuration. That is right for a file that is not there and
    wrong for one an operator wrote and got wrong, which is why this reader is
    private and separate from `local_vision.py`'s - the two are about to
    disagree on what a malformed file means.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def general_config(base_dir: Path | None = None) -> dict[str, Any]:
    """The general options `distill.json` states, unvalidated and unfiltered.

    Unvalidated because validation belongs at the door every option goes
    through, not at a second one here (a configured number refused by a rule
    this module invented would be refused differently from a typed one).

    The nested `local_vision` object is dropped rather than passed on: it is a
    section belonging to another owner, and a file that says
    `{"local_vision": {...}}` is not asking to set an option called
    `local_vision`.
    """
    payload = _read_json((base_dir or config_dir()) / GENERAL_CONFIG_FILENAME)
    return {key: value for key, value in payload.items() if key != LOCAL_VISION_SECTION}


def environment_options() -> dict[str, Any]:
    """The options the environment sets, read through `OPTION_ENV_VARIABLES`.

    An empty value is not a setting: `DISTILL_OUTPUT_DIR=` exports a variable
    with nothing in it, and treating that as an output root of `""` would
    silently mean the default while looking like a choice.
    """
    values: dict[str, Any] = {}
    for option, variable in OPTION_ENV_VARIABLES.items():
        value = os.environ.get(variable)
        if value:
            values[option] = value
    return values


def resolve_options(
    args: Mapping[str, Any],
    *,
    general_keys: Iterable[str],
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """`args`, with the configured layers folded in underneath it.

    The order is CLI > environment > file > default. The default layer is not
    here: an option nobody set is simply absent from the result, and the option
    table downstream answers for it - so a default lives in one place instead of
    being copied into the loader.

    `general_keys` says which keys are options, and comes from the caller
    because `options.py` owns that vocabulary. A file or a variable can set only
    those; every other key of `args` (a path, a URL, a batch limit) is carried
    through untouched but is never *introduced* by a config file.

    A key present in `args` with the value `None` is a caller that named
    nothing, not a caller asking for the default: that is how the CLI spells an
    unused flag, and a configured value has to survive it.
    """
    known = frozenset(general_keys)
    resolved: dict[str, Any] = {
        key: value for key, value in general_config(base_dir).items() if key in known
    }
    resolved.update(
        {key: value for key, value in environment_options().items() if key in known}
    )
    for key, value in args.items():
        if value is None and key in resolved:
            continue
        resolved[key] = value
    return resolved
