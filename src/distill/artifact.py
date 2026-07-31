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

This module writes **out of** a bundle and never into one, which is what its
ADR-0003 exemption rests on. `emit_artifact` enforces that rather than
assuming it: an artifact directory at or under the output root is refused, so
no `--artifact-dir` and no adopted directory can turn a copy-out into a write
into the cache.

Nothing here writes to a repository's git configuration. Whether `.distill`
is committed or ignored is the operator's decision, and a tool that silently
edited `.gitignore` would be making it for them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .bundle_store import atomic_write_text
from .redact_secrets import redact_text

XDG_DATA_HOME_ENV = "XDG_DATA_HOME"
PROJECT_ARTIFACT_DIRNAME = ".distill"

# A filename, not a path. Everything outside this class is replaced, so a
# source-chosen title or filename cannot climb out of the artifact directory
# or name a hidden file.
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
# Enough fingerprint to separate two recordings that share a filename, short
# enough that the name stays readable.
_FINGERPRINT_CHARACTERS = 12
# Leaves room for `-<fingerprint>.md` inside the 255-byte component limit that
# ext4, APFS and NTFS all impose. Measured in bytes, because that is what the
# filesystem counts and a name is not necessarily ASCII on the way in.
_MAX_NAME_BYTES = 200


def resolve_artifact_dir(
    *,
    explicit: str | None,
    env: dict[str, str],
    cwd: Path,
) -> Path:
    """The directory the final artifact is written to, highest precedence first.

    1. `explicit` - what the caller configured, by any route. Distill's own
       layering (CLI > `DISTILL_ARTIFACT_DIR` > config file) has already
       happened in `config.py`, which owns that order for every option; this
       module receiving one value rather than three is what keeps the two from
       disagreeing about which layer wins.
    2. a `.distill` that already exists at or above `cwd` **within the same
       work tree** - someone already decided where this project's artifacts go
    3. `<git work tree root>/.distill` - the *root*, not `cwd`, so a run
       invoked from a subdirectory does not scatter artifacts through the tree
    4. `$XDG_DATA_HOME/distill/artifacts` - data, not cache: a deliverable a
       cache cleaner may delete is not a deliverable

    The walk in (2) stops at the work tree root, and outside a work tree does
    not happen at all. An unbounded walk reaches `$HOME`, where `~/.distill` is
    Distill's own legacy config directory: every repository on the machine
    without its own `.distill` would have adopted it and written artifacts into
    the operator's configuration.

    `env` and `cwd` are passed in rather than read here, so the order is
    testable without mutating global state.
    """
    if explicit:
        return Path(explicit).expanduser()
    work_tree = _git_work_tree_root(cwd)
    if work_tree is not None:
        adopted = _existing_project_dir(cwd, boundary=work_tree)
        return adopted if adopted is not None else work_tree / PROJECT_ARTIFACT_DIRNAME
    data_home = env.get(XDG_DATA_HOME_ENV)
    root = (
        Path(data_home).expanduser()
        if data_home and Path(data_home).expanduser().is_absolute()
        else Path.home() / ".local" / "share"
    )
    return root / "distill" / "artifacts"


def artifact_entry_name(youtube_video_id: str | None, fingerprint: str, stem: str) -> str:
    """The artifact's filename stem: stable across runs of the same source.

    A YouTube id already names the recording uniquely and reads well in a
    directory listing. A local file has only its own name, which is not unique
    - two talks are both `keynote.mp4` - so it carries a slice of the **source
    fingerprint** as well.

    The fingerprint, deliberately, and not the bundle key: the bundle key folds
    in the processing options and the pipeline version, so naming the artifact
    after it would give the same recording a new filename on every option
    change and every Distill upgrade, leaving the stale ones behind in
    somebody's project. The fingerprint identifies the source, which is what
    the caller means by "the reading of this video".

    The name is redacted first, and unconditionally. A bundle already refuses
    to archive a credential-shaped source name, and an artifact that wrote one
    back out would undo that in the worse place: the cache is a private
    directory, while this lands in a project that may well be committed.
    `--no-redact-secrets` governs what the operator sees inside their own
    reading, never what gets written as a filename beside it.
    """
    if youtube_video_id:
        return _safe_name(youtube_video_id) or _safe_name(fingerprint[:_FINGERPRINT_CHARACTERS])
    suffix = _safe_name(fingerprint[:_FINGERPRINT_CHARACTERS])
    safe_stem = _capped(_safe_name(redact_text(stem).text))
    return f"{safe_stem}-{suffix}" if safe_stem else suffix


def emit_artifact(
    self_contained_render: Path,
    artifact_dir: Path,
    entry: str,
    *,
    output_root: Path,
) -> Path:
    """Copy the self-contained render to `<artifact_dir>/<entry>.md`.

    A copy, not a symlink: the render lives in a cache advertised as deletable,
    and a link into reclaimed space is a broken artifact. The self-contained
    render is the one built to travel - it carries no reference to `frames/`
    or to any machine-local path - so copying it is complete rather than a
    fragment of a bundle.

    Refuses an artifact directory at or under `output_root`. That is the
    module's ADR-0003 exemption stated as a check rather than as a hope: a
    cache root passed as `--artifact-dir` (or adopted, since validating an
    output root creates it, and a created `.distill` is then a `.distill` that
    exists) would otherwise have this writing into the store it claims never to
    touch.

    Written through the bundle store's confined atomic emitter, which is the
    same defense the store itself needed (S1): the destination is a predictable
    path inside a directory a repository can contain, so a symlink pre-created
    at that name would redirect the write - `shutil.copyfile` would follow it
    and truncate whatever it points at. The emitter opens the final component
    `O_NOFOLLOW` and replaces the name rather than writing through it, and the
    write is atomic, so a reader never sees a half-copied document and two
    concurrent runs of the same video cannot interleave.

    Overwrites: an artifact is regenerable from its source, the bundle keeps
    generation history, and a caller that reprocessed a video wants the new
    reading under the name they already know.
    """
    if entry != _safe_name(entry) or not entry:
        raise ValueError(f"artifact entry must be one safe filename component: {entry!r}")
    resolved_dir = artifact_dir.expanduser().resolve()
    resolved_root = output_root.expanduser().resolve()
    if resolved_dir == resolved_root or resolved_root in resolved_dir.parents:
        raise ValueError(f"artifact directory is inside the output root: {resolved_dir}")
    resolved_dir.mkdir(parents=True, exist_ok=True)
    destination = resolved_dir / f"{entry}.md"
    atomic_write_text(destination, self_contained_render.read_text(), root=resolved_dir)
    return destination


def _existing_project_dir(cwd: Path, *, boundary: Path) -> Path | None:
    """The nearest existing `.distill` from `cwd` up to `boundary`, inclusive."""
    for directory in (cwd, *cwd.parents):
        candidate = directory / PROJECT_ARTIFACT_DIRNAME
        if candidate.is_dir():
            return candidate
        if directory == boundary:
            return None
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
    return _UNSAFE_NAME_CHARACTERS.sub("-", value).strip("-. ")


def _capped(value: str) -> str:
    """`value` truncated to `_MAX_NAME_BYTES`, on a character boundary."""
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_NAME_BYTES:
        return value
    return encoded[:_MAX_NAME_BYTES].decode("utf-8", "ignore").rstrip("-. ")


def artifact_dir_from_environment(cwd: Path | None = None) -> Path:
    """`resolve_artifact_dir` against the real process environment."""
    return resolve_artifact_dir(
        explicit=None,
        env=dict(os.environ),
        cwd=cwd or Path.cwd(),
    )
