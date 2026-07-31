"""Where the final **artifact** lands, and what it is called.

Distill produces two different things with two different lifetimes, and this
module exists because conflating them is what made the output hard to find.

The **bundle** is derived state: content-addressed, generation-numbered,
prunable, and deliberately disposable - it lives under the cache root so a
cache cleaner may reclaim it without losing anything that cannot be rebuilt
from the source. The **artifact** is the one self-contained document a caller
actually consumes. It is not derived state to be reclaimed; it is the output,
and it belongs where the work is being done - normally the project the caller
is standing in.

So the two resolve separately. `output_dir` moves the cache root and answers
"where does derived state live". `artifact_dir` answers "where does the
deliverable go", and its default walks toward the project rather than toward
`$HOME`.

Nothing here writes to a repository's git configuration. Whether `.distill`
is committed or ignored is the operator's decision, and a tool that silently
edited `.gitignore` would be making it for them.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .redact_secrets import redact_text

ARTIFACT_DIR_ENV = "DISTILL_ARTIFACT_DIR"
XDG_DATA_HOME_ENV = "XDG_DATA_HOME"
PROJECT_ARTIFACT_DIRNAME = ".distill"

# A filename, not a path. Everything outside this class is replaced, so a
# source-chosen title or filename cannot climb out of the artifact directory
# or name a hidden file.
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
# Enough hash to separate two recordings that share a filename, short enough
# that the name stays readable.
_FINGERPRINT_CHARACTERS = 8


def resolve_artifact_dir(
    *,
    explicit: str | None,
    env: dict[str, str],
    configured: str | None,
    cwd: Path,
) -> Path:
    """The directory the final artifact is written to, highest precedence first.

    1. an explicit `--artifact-dir`
    2. `DISTILL_ARTIFACT_DIR`
    3. a `.distill` that already exists at or above `cwd` - someone already
       decided where this project's artifacts go, and a global default that
       scattered a second one beside it would be ignoring that decision
    4. `artifact_dir` from configuration
    5. `<git work tree root>/.distill` - the *root*, not `cwd`, so a run
       invoked from a subdirectory does not scatter artifacts through the tree
    6. `$XDG_DATA_HOME/distill/artifacts` - data, not cache: a deliverable a
       cache cleaner may delete is not a deliverable

    The caller passes `env` and `cwd` rather than this reading the process, so
    the order is testable without mutating global state.
    """
    if explicit:
        return Path(explicit).expanduser()
    from_env = env.get(ARTIFACT_DIR_ENV)
    if from_env:
        return Path(from_env).expanduser()
    adopted = _existing_project_dir(cwd)
    if adopted is not None:
        return adopted
    if configured:
        return Path(configured).expanduser()
    work_tree = _git_work_tree_root(cwd)
    if work_tree is not None:
        return work_tree / PROJECT_ARTIFACT_DIRNAME
    data_home = env.get(XDG_DATA_HOME_ENV)
    root = (
        Path(data_home).expanduser()
        if data_home and Path(data_home).expanduser().is_absolute()
        else Path.home() / ".local" / "share"
    )
    return root / "distill" / "artifacts"


def artifact_entry_name(youtube_video_id: str | None, source_hash: str, stem: str) -> str:
    """The artifact's filename stem: stable across runs of the same source.

    A YouTube id already names the recording uniquely and reads well in a
    directory listing. A local file has only its own name, which is not unique
    - two talks are both `keynote.mp4` - so it carries a slice of the source
    hash as well.

    The name is redacted first, and unconditionally. A bundle already refuses
    to archive a credential-shaped source name, and an artifact that wrote one
    back out would undo that in the worse place: the cache is a private
    directory, while this lands in a project that may well be committed.
    `--no-redact-secrets` governs what the operator sees inside their own
    reading, never what gets written as a filename beside it.
    """
    if youtube_video_id:
        return _safe_name(youtube_video_id)
    fingerprint = source_hash[:_FINGERPRINT_CHARACTERS]
    safe_stem = _safe_name(redact_text(stem).text)
    return f"{safe_stem}-{fingerprint}" if safe_stem else fingerprint


def write_artifact(self_contained_render: Path, artifact_dir: Path, entry: str) -> Path:
    """Copy the self-contained render into `artifact_dir` as `<entry>.md`.

    A copy, not a symlink: the render lives in a cache advertised as deletable,
    and a link into reclaimed space is a broken artifact. The self-contained
    render is the one built to travel - it carries no reference to `frames/`
    or to any machine-local path - so copying it is complete rather than a
    fragment of a bundle.

    Overwrites: an artifact is regenerable from its source, the bundle keeps
    generation history, and a caller that reprocessed a video wants the new
    reading under the name they already know.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / f"{entry}.md"
    shutil.copyfile(self_contained_render, destination)
    return destination


def _existing_project_dir(cwd: Path) -> Path | None:
    """The nearest `.distill` directory at or above `cwd`, if one exists."""
    for directory in (cwd, *cwd.parents):
        candidate = directory / PROJECT_ARTIFACT_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def _git_work_tree_root(cwd: Path) -> Path | None:
    """The root of the git work tree containing `cwd`, or `None`.

    Found by walking for `.git` rather than by asking git: this module must not
    import `subprocess` (only `run_command` may - the one place process
    spawning is reviewed), and spawning a process per run to learn a path that
    a directory walk answers would be cost with no return.

    `.git` is accepted as either a directory or a file, because a linked
    worktree and a submodule both record their git directory in a `.git` file.
    A run inside one belongs to that tree, which is the same answer
    `git rev-parse --show-toplevel` gives.
    """
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return None


def _safe_name(value: str) -> str:
    """`value` reduced to a filename this module is willing to write."""
    collapsed = _UNSAFE_NAME_CHARACTERS.sub("-", value).strip("-. ")
    return collapsed


def artifact_dir_from_environment(cwd: Path | None = None) -> Path:
    """`resolve_artifact_dir` against the real process environment."""
    return resolve_artifact_dir(
        explicit=None,
        env=dict(os.environ),
        configured=None,
        cwd=cwd or Path.cwd(),
    )
