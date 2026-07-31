"""Where a run's options come from before anyone types one, and which layer wins.

This module owns three things:

- **The config directory.** An absolute `DISTILL_CONFIG_DIR` is authoritative,
  including when it is absent. Without one, `$XDG_CONFIG_HOME/distill`, then
  `~/.config/distill`, then `~/.distill` are considered; the first directory
  that exists is *the* config directory and the rest are not read. Relative
  environment paths are skipped. Directories do not merge, because an option
  whose value depends on which directory happens to exist on a particular
  machine is a value nobody can predict from the file they edited.
- **The general schema over `distill.json`.** Every top-level key of that file
  except the nested `local_vision` object, which belongs to another owner (see
  below). A key here names an option; a key naming nothing is refused. What is
  also not passed over is the file: absent is no configuration, but unparsable,
  unreadable or not-an-object is `E_BAD_OPTIONS` at stage `options` naming the
  path, because a run that falls back to the defaults for a file somebody wrote
  produces a **bundle** nobody asked for (D-011).
- **The resolution order.** CLI > environment > file > default (the environment
  layer being `DISTILL_OUTPUT_DIR` and `DISTILL_ARTIFACT_DIR`, the two options
  a machine or a project rather than a run decides - where derived state lives
  and where the deliverable lands), applied by
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
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from .errors import DistillError, errno_name

CONFIG_DIR_ENV = "DISTILL_CONFIG_DIR"
XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"
GENERAL_CONFIG_FILENAME = "distill.json"
LOCAL_VISION_SECTION = "local_vision"

OPTION_ENV_VARIABLES: dict[str, str] = {
    "artifact_dir": "DISTILL_ARTIFACT_DIR",
    "output_dir": "DISTILL_OUTPUT_DIR",
}
"""The options an environment variable may set, and the variable that sets one.

A table rather than a prefix rule, so the environment layer is a list somebody
decided on: a `DISTILL_*` variable is an option input only by appearing here,
and the diagnostic variables cannot become option inputs by being named well.

Both entries answer *where something goes*, and neither changes what a run
produces. The output root is where a machine keeps its bundles, a property of
the machine rather than of the processing being asked for (ADR-0004). The
artifact directory is where the deliverable lands, which a caller standing in
a project - an editor, an agent, an MCP server - decides once for everything
it runs, not per invocation. An option that changed the reading itself would
not belong here.
"""


class ResolvedOptions(dict[str, Any]):
    """Resolved values plus the file origin of values still supplied by it."""

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        configured_from: Mapping[str, Path],
    ) -> None:
        super().__init__(values)
        self.configured_from = dict(configured_from)


def config_dir_candidates() -> tuple[Path, ...]:
    """The directories configuration may live in, highest precedence first.

    `DISTILL_CONFIG_DIR` and `$XDG_CONFIG_HOME` contribute a candidate only when
    they are set to an absolute path. An empty variable is a variable nobody
    meant to set, and a relative one would read `distill.json` relative to the
    process working directory. Neither is configuration.
    """
    candidates: list[Path] = []
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_absolute():
            candidates.append(override_path)
    xdg_config_home = os.environ.get(XDG_CONFIG_HOME_ENV)
    if xdg_config_home:
        xdg_path = Path(xdg_config_home).expanduser()
        if xdg_path.is_absolute():
            candidates.append(xdg_path / "distill")
    home = Path.home()
    candidates.append(home / ".config" / "distill")
    candidates.append(home / ".distill")
    return tuple(candidates)


def _config_directory_kind(path: Path) -> Literal["directory", "other", "absent"]:
    """Classify one candidate, refusing when the filesystem gives no answer."""
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return "absent"
    except OSError as exc:
        raise _bad_config_file(
            path,
            "config directory could not be read",
            errno=errno_name(exc),
        ) from exc
    return "directory" if stat.S_ISDIR(info.st_mode) else "other"


def config_dir() -> Path:
    """The one directory configuration is read from.

    An absolute `DISTILL_CONFIG_DIR` explicit override is authoritative even
    when it is absent: absence means an explicit-but-empty configuration, not
    permission to read a lower stale directory. If it exists but is not a
    directory, or cannot be classified, it is a typed refusal.

    Only the implicit XDG and home candidates fall through. An absent path or
    one that is not a directory contributes no config, and the first existing
    directory wins.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        explicit = Path(override).expanduser()
        if explicit.is_absolute():
            kind = _config_directory_kind(explicit)
            if kind == "absent":
                return explicit
            if kind == "other":
                raise _bad_config_file(
                    explicit,
                    "config directory is not a directory",
                    errno="ENOTDIR",
                )
            return explicit
    candidates = config_dir_candidates()
    for candidate in candidates:
        if _config_directory_kind(candidate) == "directory":
            return candidate
    return candidates[0]


def _bad_config_file(path: Path, message: str, **details: Any) -> DistillError:
    """The one refusal this module raises, so all three name the same things.

    `E_BAD_OPTIONS` at stage `options`, because a config file is an operator
    typing options a day earlier and the answer they get should not depend on
    when they typed them. The path is always in the details: resolution walks
    four directories, and "your config is broken" without a filename sends
    somebody to look in the one they did not edit.
    """
    return DistillError("E_BAD_OPTIONS", "options", message, {"path": str(path), **details})


def _read_json(path: Path) -> dict[str, Any]:
    """The JSON object at `path`, or nothing when the file is not there.

    Absence is the only one of these that means no configuration (D-011). A file
    an operator wrote and got wrong - unparsable, unreadable, or holding
    something that is not an object - is refused, because the alternative is a
    run that quietly uses the defaults for every option the file was meant to
    set and produces a **bundle** nobody asked for.

    Not there is a `stat` this process was allowed to make: `ENOENT` and the
    `ENOTDIR` of a path whose parent is a file are absence, and every other
    `OSError` - `EACCES` above all - is a refusal, for the reason
    `source_path_kind` refuses one (D-022). "There is no configuration here"
    about a file we were denied is a claim with nothing behind it.

    Separate from `local_vision.py`'s reader rather than shared with it: that
    one stays forgiving, because a broken local-vision section costs a run its
    captions and this one costs a run its options.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        try:
            path.lstat()
        except (FileNotFoundError, NotADirectoryError):
            return {}
        except OSError as exc:
            raise _bad_config_file(
                path,
                "config file presence could not be checked",
                errno=errno_name(exc),
            ) from exc
        raise _bad_config_file(
            path,
            "config file points to a missing target",
            errno="ENOENT",
        ) from None
    except NotADirectoryError:
        return {}
    except UnicodeDecodeError as exc:
        raise _bad_config_file(
            path,
            "config file is not UTF-8 text",
            error=str(exc),
        ) from exc
    except OSError as exc:
        raise _bad_config_file(
            path, "config file could not be read", errno=errno_name(exc)
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_config_file(path, "config file is not valid JSON", error=str(exc)) from exc
    if not isinstance(payload, dict):
        raise _bad_config_file(
            path, "config file must hold a JSON object", received=type(payload).__name__
        )
    return payload


def general_config(base_dir: Path | None = None) -> dict[str, Any]:
    """The general options `distill.json` states, unvalidated and unfiltered.

    Unvalidated because validation belongs at the door every option goes
    through, not at a second one here (a configured number refused by a rule
    this module invented would be refused differently from a typed one). The
    *file* is this module's question and is checked here - whether it can be
    read and holds an object - because no later door ever sees a file that did
    not parse.

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
) -> ResolvedOptions:
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
    root = base_dir or config_dir()
    file_options = general_config(root)
    unknown = sorted(set(file_options) - known)
    if unknown:
        raise _bad_config_file(
            root / GENERAL_CONFIG_FILENAME,
            "config file names unknown options",
            unknown_options=unknown,
        )
    file_options = {
        key: value
        for key, value in file_options.items()
        if not (key == "output_dir" and value == "")
    }
    resolved = ResolvedOptions(
        {key: value for key, value in file_options.items() if key in known},
        configured_from={
            key: root / GENERAL_CONFIG_FILENAME for key in file_options if key in known
        },
    )
    for key, value in environment_options().items():
        if key not in known:
            continue
        resolved[key] = value
        resolved.configured_from.pop(key, None)
    for key, value in args.items():
        if value is None and key in resolved:
            continue
        resolved[key] = value
        resolved.configured_from.pop(key, None)
    return resolved
