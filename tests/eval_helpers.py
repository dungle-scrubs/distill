from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_score_module() -> ModuleType:
    eval_root = Path(__file__).resolve().parent / "evals"
    sys.path.insert(0, str(eval_root))
    score_spec = importlib.util.spec_from_file_location("score", eval_root / "score.py")
    assert score_spec is not None
    assert score_spec.loader is not None
    score = importlib.util.module_from_spec(score_spec)
    sys.modules["score"] = score
    score_spec.loader.exec_module(score)
    return score
