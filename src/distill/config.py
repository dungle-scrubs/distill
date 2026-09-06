"""Find the configuration directory using DISTILL_CONFIG_DIR and XDG_CONFIG_HOME.

An absolute DISTILL_CONFIG_DIR is authoritative, even if absent. Otherwise the
first existing XDG, ~/.config/distill, or ~/.distill directory wins.
This module does not own file reading or option precedence; configuration.py does.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Literal

from .errors import DistillError, errno_name

CONFIG_DIR_ENV = "DISTILL_CONFIG_DIR"
XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"


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
