from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_eval_module(name: str) -> ModuleType:
    """Load one module from the eval tools directory."""
    eval_root = Path(__file__).resolve().parent / "evals"
    eval_root_string = str(eval_root)
    if eval_root_string not in sys.path:
        sys.path.insert(0, eval_root_string)
    spec = importlib.util.spec_from_file_location(name, eval_root / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_score_module() -> ModuleType:
    """Load the eval scorer module."""
    return load_eval_module("score")
