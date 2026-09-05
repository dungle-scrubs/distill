"""Inject dependencies at the pipeline's construction boundary in integration tests."""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest

from distill import acquisition, pipeline
from distill.local_vision import FrameInterpreter
from distill.run_orchestrator import ProcessingRun, interpret_frames


def configure_run(monkeypatch: pytest.MonkeyPatch, **dependencies: Any) -> None:
    """Keep real ProcessingRun construction while replacing selected dependencies."""
    current = pipeline.ProcessingRun
    configured = dict(current.keywords) if isinstance(current, partial) else {}
    interpreter_options: dict[str, Any] = {}
    interpret = configured.get("interpret_frames")
    if isinstance(interpret, partial):
        factory = interpret.keywords.get("interpreter_factory")
        if isinstance(factory, partial):
            interpreter_options.update(factory.keywords)
    for name in ("probe", "try_interpret"):
        if name in dependencies:
            interpreter_options[name] = dependencies.pop(name)
    if interpreter_options:
        configured["interpret_frames"] = partial(
            interpret_frames, interpreter_factory=partial(FrameInterpreter, **interpreter_options)
        )
    configured.update(dependencies)
    monkeypatch.setattr(pipeline, "ProcessingRun", partial(ProcessingRun, **configured))


def configure_downloader(monkeypatch: pytest.MonkeyPatch, **dependencies: Any) -> None:
    """Use the same configured constructor for direct and source-resolution tests."""
    constructor = acquisition.YoutubeDownloader
    factory = partial(constructor, **dependencies)
    monkeypatch.setattr(acquisition, "YoutubeDownloader", factory)
