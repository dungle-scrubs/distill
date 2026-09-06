"""The configuration snapshot and its dependency direction are observable contracts."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from distill.configuration import resolve_run_config


@pytest.mark.parametrize("nested", [None, {}, {"model": "recorded-reader"}])
def test_each_configuration_document_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: dict[str, Any] | None
) -> None:
    general = tmp_path / "distill.json"
    vision = tmp_path / "distill.local-vision.json"
    payload: dict[str, Any] = {"max_keyframes": 7}
    if nested is not None:
        payload["local_vision"] = nested
    general.write_text(json.dumps(payload))
    vision.write_text(json.dumps({"model": "earlier-reader", "caption_frames": False}))
    reads: Counter[Path] = Counter()
    read_text = Path.read_text

    def counted(path: Path, *args: Any, **kwargs: Any) -> str:
        if path in {general, vision}:
            reads[path] += 1
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    result = resolve_run_config({}, base_dir=tmp_path)
    assert reads == {general: 1, vision: 1}
    assert result.options.max_keyframes == 7
    assert result.options.caption_frames is False
    assert result.local_vision.model == (nested or {}).get("model", "earlier-reader")
    assert result.local_vision == result.options.local_vision_config()


def test_value_and_directory_modules_do_not_import_the_resolver() -> None:
    package = Path(__file__).resolve().parents[1] / "src/distill"
    for name in ("options", "local_vision", "config"):
        tree = ast.parse((package / f"{name}.py").read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert "configuration" not in imports, name
