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

from distill.artifact import artifact_entry_name, resolve_artifact_dir


def _git_repo(root: Path) -> Path:
    """A work tree marker, which is all the resolver looks for."""
    (root / ".git").mkdir(parents=True)
    return root


def test_an_explicit_directory_wins_over_everything(tmp_path: Path) -> None:
    explicit = tmp_path / "chosen"

    resolved = resolve_artifact_dir(
        explicit=str(explicit),
        env={"DISTILL_ARTIFACT_DIR": str(tmp_path / "from-env")},
        configured=str(tmp_path / "from-config"),
        cwd=tmp_path,
    )

    assert resolved == explicit


def test_the_environment_wins_over_config_and_defaults(tmp_path: Path) -> None:
    resolved = resolve_artifact_dir(
        explicit=None,
        env={"DISTILL_ARTIFACT_DIR": str(tmp_path / "from-env")},
        configured=str(tmp_path / "from-config"),
        cwd=tmp_path,
    )

    assert resolved == tmp_path / "from-env"


def test_an_existing_project_directory_is_adopted_before_config(tmp_path: Path) -> None:
    """A `.distill` already in the tree is a decision someone made; honour it
    rather than a global default that would scatter a second one."""
    project = tmp_path / "work"
    nested = project / "src" / "deep"
    nested.mkdir(parents=True)
    (project / ".distill").mkdir()

    resolved = resolve_artifact_dir(
        explicit=None,
        env={},
        configured=str(tmp_path / "from-config"),
        cwd=nested,
    )

    assert resolved == project / ".distill"


def test_configuration_wins_when_no_project_directory_exists(tmp_path: Path) -> None:
    resolved = resolve_artifact_dir(
        explicit=None,
        env={},
        configured=str(tmp_path / "from-config"),
        cwd=tmp_path,
    )

    assert resolved == tmp_path / "from-config"


def test_a_git_work_tree_defaults_to_its_root_not_the_working_directory(
    tmp_path: Path,
) -> None:
    """An agent invoked from a subdirectory must not scatter a second
    `.distill` there: artifacts belong to the project, not to the cwd."""
    repo = _git_repo(tmp_path / "repo")
    nested = repo / "src" / "deep"
    nested.mkdir(parents=True)

    resolved = resolve_artifact_dir(explicit=None, env={}, configured=None, cwd=nested)

    assert resolved == repo / ".distill"


def test_outside_a_repository_the_artifact_is_user_data_not_cache(
    tmp_path: Path,
) -> None:
    """`XDG_DATA_HOME`, not `XDG_CACHE_HOME`: a deliverable that a cache cleaner
    may delete is not a deliverable."""
    resolved = resolve_artifact_dir(
        explicit=None,
        env={"XDG_DATA_HOME": str(tmp_path / "data")},
        configured=None,
        cwd=tmp_path,
    )

    assert resolved == tmp_path / "data" / "distill" / "artifacts"


@pytest.mark.parametrize(
    ("video_id", "source_hash", "stem", "expected"),
    [
        ("F3lL98Pj90o", "abc123def456", "irrelevant", "F3lL98Pj90o"),
        (None, "abc123def456", "keynote", "keynote-abc123de"),
        (None, "abc123def456", "", "abc123de"),
    ],
)
def test_the_entry_name_is_stable_and_collision_resistant(
    video_id: str | None, source_hash: str, stem: str, expected: str
) -> None:
    """A YouTube id names itself. A local file gets its stem plus a slice of the
    source hash, so two recordings both called `talk.mp4` do not overwrite each
    other's artifact."""
    assert artifact_entry_name(video_id, source_hash, stem) == expected


def test_a_credential_shaped_source_name_is_not_written_into_the_project() -> None:
    """The bundle refuses to archive a credential-shaped source name; an
    artifact filename that carried one would undo that in a directory the
    operator may commit."""
    secret_name = f"ghp_{'a' * 36}"

    name = artifact_entry_name(None, "abc123def456", secret_name)

    assert secret_name not in name
    assert name == "REDACTED-abc123de"


def test_the_entry_name_refuses_path_separators(tmp_path: Path) -> None:
    """The name becomes a filename directly, so a source-chosen string must not
    be able to climb out of the artifact directory."""
    name = artifact_entry_name(None, "abc123def456", "../../etc/passwd")

    assert "/" not in name
    assert ".." not in name
