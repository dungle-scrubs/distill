"""The `filtered-view` command: the read-side view reached through `main(argv)`.

What is under test here is the *surface*, not the view - `test_filtered_view.py`
holds what the markdown says, and these drive the whole command so the argument
glue, the option layers, the JSON on stdout and the error boundary are exercised
by the path an operator actually takes. The fixture is a bundle a real run could
have published, built through `BundleStore.open` / `begin` / `commit` by
`test_filtered_view.publish`, because a command reading a hand-written manifest
would agree with a shape no run produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
from test_filtered_view import BUNDLE_KEY, frame, publish

from distill.cli import build_parser, main
from distill.render import FILTERED_VIEW_BANNER


def tree_digests(root: Path) -> dict[str, str]:
    """Every file under `root`, keyed by its path, valued by its contents.

    A digest per file rather than one over the whole tree: a command that wrote
    something back should say *what* it wrote, and a single hash answers only
    that something moved. Directories are absent because an empty one is not a
    byte, and a file appearing or vanishing changes the key set either way.
    """
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def subcommand_options(command: str) -> set[str]:
    """Every option string one subcommand registers, read off the parser."""
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return {
                string
                for subaction in action.choices[command]._actions
                for string in subaction.option_strings
            }
    raise AssertionError("the parser registers no subcommands")


def test_the_command_takes_a_bundle_key_and_an_output_root_and_no_dry_run() -> None:
    """`get-job-status`'s shape, and `cache-doctor`'s omission.

    Read-only means there is no other mode, so a `--dry-run` here would preview
    itself: the view is produced on demand and never written back (D-006).
    Asserted as the *whole* option set rather than as an absence, because a flag
    added later without a decision is the thing this is holding.
    """
    parsed = build_parser().parse_args(["filtered-view", BUNDLE_KEY, "--output-dir", "/tmp/out"])

    assert parsed.bundle_key == BUNDLE_KEY
    assert parsed.output_dir == "/tmp/out"
    assert subcommand_options("filtered-view") == {"-h", "--help", "--output-dir"}


def test_the_command_prints_the_view_under_its_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: argv in, the view on stdout, carrying the banner it is under.

    JSON like every other command (R-46), with the document under `markdown` -
    so the banner is on stdout both as the surface's own bytes and as the thing
    a caller parsing the payload reads.
    """
    root = tmp_path / "output"
    publish(
        root,
        [
            frame(root, 1, text="kept one", salience={"adds_information": True, "reason": "new"}),
            frame(root, 2, text="dropped one", salience={"adds_information": False}),
        ],
    )

    main(["filtered-view", BUNDLE_KEY, "--output-dir", str(root)])

    out = capsys.readouterr().out
    assert FILTERED_VIEW_BANNER in out
    payload = json.loads(out)
    assert payload["bundle_key"] == BUNDLE_KEY
    assert payload["root"] == str(root.resolve())
    assert FILTERED_VIEW_BANNER in payload["markdown"]
    assert "kept one" in payload["markdown"]
    assert "dropped one" not in payload["markdown"]


def test_an_absent_generation_leaves_as_the_error_record_and_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The typed refusal through the boundary: exit 2, one record, clean stdout.

    A key naming nothing servable is the ordinary way this command is mistyped,
    so what an operator gets has to be the record every other failure is - the
    code to branch on, on stderr - rather than a stack naming Distill's
    internals.
    """
    root = tmp_path / "output"
    root.mkdir()

    with pytest.raises(SystemExit) as exit_info:
        main(["filtered-view", BUNDLE_KEY, "--output-dir", str(root)])

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    payload = json.loads(captured.err)
    assert payload["code"] == "E_NO_ACTIVE_GENERATION"
    assert payload["stage"] == "filtered_view"
    assert payload["details"]["bundle_key"] == BUNDLE_KEY


def test_the_command_leaves_the_bundle_directory_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Never written back, asserted against the disk rather than against intent.

    The banner claims the stored generation still holds every frame, and the
    view exists only as long as the caller keeps what was printed. Both claims
    are one property: nothing under the output root changed. Hashed per file,
    before and after, so a rewritten manifest and a new file are both failures
    rather than one being invisible.
    """
    root = tmp_path / "output"
    publish(root, [frame(root, 1, text="kept one", salience={"adds_information": True})])
    before = tree_digests(root)

    main(["filtered-view", BUNDLE_KEY, "--output-dir", str(root)])

    assert capsys.readouterr().out != ""
    assert tree_digests(root) == before
