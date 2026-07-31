"""Where the final artifact lands, and what it is called.

The **cache** is content-addressed, prunable and deliberately disposable; the
**artifact** is the one self-contained document a caller consumes. They are
different concepts with different lifetimes, so they resolve separately: the
cache answers "where does derived state live", the artifact answers "where does
the deliverable go", and the answer to the second is normally the project being
worked in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from distill.artifact import artifact_entry_name, emit_artifact, resolve_artifact_dir
from distill.errors import DistillError


def _git_repo(root: Path) -> Path:
    """A work tree marker, which is all the resolver looks for."""
    (root / ".git").mkdir(parents=True)
    return root


def _render(root: Path, text: str = "# reading\n") -> Path:
    path = root / "video.self-contained.md"
    path.write_text(text)
    return path


def test_an_explicit_directory_wins_over_everything(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    explicit = tmp_path / "chosen"

    resolved = resolve_artifact_dir(explicit=str(explicit), env={}, cwd=repo)

    assert resolved == explicit


def test_an_existing_project_directory_is_adopted(tmp_path: Path) -> None:
    """A `.distill` already in the tree is a decision someone made; honour it
    rather than creating a second one at the work tree root."""
    repo = _git_repo(tmp_path / "repo")
    nested = repo / "src" / "deep"
    nested.mkdir(parents=True)
    (repo / "src" / ".distill").mkdir()

    resolved = resolve_artifact_dir(explicit=None, env={}, cwd=nested)

    assert resolved == repo / "src" / ".distill"


def test_adoption_stops_at_the_work_tree_root(tmp_path: Path) -> None:
    """An unbounded walk reaches `$HOME`, where `~/.distill` is Distill's own
    legacy config directory: every repository on the machine without its own
    would have adopted it and written artifacts into the operator's config."""
    (tmp_path / ".distill").mkdir()
    repo = _git_repo(tmp_path / "repo")

    resolved = resolve_artifact_dir(explicit=None, env={}, cwd=repo)

    assert resolved == repo / ".distill"


def test_a_nested_repository_does_not_adopt_the_outer_one(tmp_path: Path) -> None:
    outer = _git_repo(tmp_path / "outer")
    (outer / ".distill").mkdir()
    inner = _git_repo(outer / "vendor" / "inner")

    resolved = resolve_artifact_dir(explicit=None, env={}, cwd=inner)

    assert resolved == inner / ".distill"


def test_a_git_work_tree_defaults_to_its_root_not_the_working_directory(
    tmp_path: Path,
) -> None:
    """An agent invoked from a subdirectory must not scatter a second
    `.distill` there: artifacts belong to the project, not to the cwd."""
    repo = _git_repo(tmp_path / "repo")
    nested = repo / "src" / "deep"
    nested.mkdir(parents=True)

    resolved = resolve_artifact_dir(explicit=None, env={}, cwd=nested)

    assert resolved == repo / ".distill"


def test_outside_a_repository_the_artifact_is_user_data_not_cache(
    tmp_path: Path,
) -> None:
    """`XDG_DATA_HOME`, not `XDG_CACHE_HOME`: a deliverable that a cache cleaner
    may delete is not a deliverable."""
    resolved = resolve_artifact_dir(
        explicit=None,
        env={"XDG_DATA_HOME": str(tmp_path / "data")},
        cwd=tmp_path / "not-a-repo",
    )

    assert resolved == tmp_path / "data" / "distill" / "artifacts"


@pytest.mark.parametrize(
    ("video_id", "fingerprint", "stem", "expected"),
    [
        ("F3lL98Pj90o", "abc123def456789", "irrelevant", "F3lL98Pj90o"),
        (None, "abc123def456789", "keynote", "keynote-abc123def456"),
        (None, "abc123def456789", "", "abc123def456"),
    ],
)
def test_the_entry_name_is_stable_and_collision_resistant(
    video_id: str | None, fingerprint: str, stem: str, expected: str
) -> None:
    """A YouTube id names itself. A local file gets its stem plus a slice of the
    source fingerprint, so two recordings both called `talk.mp4` do not
    overwrite each other's artifact."""
    assert artifact_entry_name(video_id, fingerprint, stem) == expected


def test_a_credential_shaped_source_name_is_not_written_into_the_project() -> None:
    """The bundle refuses to archive a credential-shaped source name; an
    artifact filename that carried one would undo that in a directory the
    operator may commit."""
    secret_name = f"ghp_{'a' * 36}"

    name = artifact_entry_name(None, "abc123def456789", secret_name)

    assert secret_name not in name
    assert name == "REDACTED-abc123def456"


def test_the_entry_name_refuses_path_separators() -> None:
    """The name becomes a filename directly, so a source-chosen string must not
    be able to climb out of the artifact directory."""
    name = artifact_entry_name(None, "abc123def456789", "../../etc/passwd")

    assert "/" not in name
    assert ".." not in name


def test_a_very_long_source_name_stays_a_writable_filename(tmp_path: Path) -> None:
    """255 bytes is the component limit on ext4, APFS and NTFS alike, and a
    refused write would cost the artifact silently."""
    name = artifact_entry_name(None, "abc123def456789", "k" * 400)

    written = emit_artifact(
        _render(tmp_path), tmp_path / "out", name, output_root=tmp_path / "cache"
    )

    assert written.is_file()
    assert len(written.name.encode("utf-8")) <= 255


def test_a_symlink_at_the_destination_is_refused_not_followed(tmp_path: Path) -> None:
    """`.distill/<video-id>.md` is a predictable path inside a directory a
    repository can contain. A copy that followed a link pre-created there would
    truncate whatever it points at (S1). The store's confinement refuses the
    write instead, and the run degrades to no artifact rather than to a
    destroyed file."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    victim = tmp_path / "precious.txt"
    victim.write_text("do not touch")
    (artifacts / "F3lL98Pj90o.md").symlink_to(victim)

    with pytest.raises(DistillError, match="symlink"):
        emit_artifact(
            _render(tmp_path), artifacts, "F3lL98Pj90o", output_root=tmp_path / "cache"
        )

    assert victim.read_text() == "do not touch"


def test_writing_into_the_cache_is_refused(tmp_path: Path) -> None:
    """This module's ADR-0003 exemption says it writes out of a bundle and never
    into one. Validating an output root creates it, and a created `.distill` is
    then a `.distill` that exists - so without this check `--output-dir
    ./.distill` would have the artifact land inside the store."""
    cache = tmp_path / "cache"
    cache.mkdir()

    with pytest.raises(ValueError, match="inside the output root"):
        emit_artifact(_render(tmp_path), cache / "bundle-key", "video", output_root=cache)

    with pytest.raises(ValueError, match="inside the output root"):
        emit_artifact(_render(tmp_path), cache, "video", output_root=cache)


def test_an_entry_that_is_not_one_filename_is_refused(tmp_path: Path) -> None:
    """`artifact_entry_name` sanitizes, but the write boundary does not take
    that on trust: a caller reaching this directly must not be able to climb."""
    for entry in ("../escape", "nested/name", ""):
        with pytest.raises(ValueError, match="one safe filename component"):
            emit_artifact(
                _render(tmp_path), tmp_path / "out", entry, output_root=tmp_path / "cache"
            )


def test_the_environment_reaches_a_caller_that_skipped_the_config_layer(
    tmp_path: Path,
) -> None:
    """`config.py` folds `DISTILL_ARTIFACT_DIR` into the options a CLI run
    resolves, so this branch is only reached by a caller that constructed
    `DistillOptions` directly. Without it that caller silently ignores the
    operator's environment and writes into whatever repository it is standing
    in - which is how the test suite came to write into this one."""
    repo = _git_repo(tmp_path / "repo")

    resolved = resolve_artifact_dir(
        explicit=None,
        env={"DISTILL_ARTIFACT_DIR": str(tmp_path / "from-env")},
        cwd=repo,
    )

    assert resolved == tmp_path / "from-env"
