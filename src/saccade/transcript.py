"""Saccade wrappers for shared audio extraction and transcription helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .media import transcript as media_transcript
from .media.transcript import (
    MIN_CONFIDENCE,
    VAD_DROP_HIGH_RATIO,
    VAD_DROP_WARNING_RATIO,
    FasterWhisperAdapter,
    WhisperAdapter,
    parse_ffmpeg_progress_time,
    transcribe_audio,
    vad_drop_warnings,
)
from .progress import ProgressCounter, ProgressReporter


def extract_audio(
    media_path: Path,
    audio_path: Path,
    progress: ProgressReporter | None = None,
    duration_sec: float | None = None,
) -> dict[str, str] | None:
    media_transcript.subprocess = subprocess
    return media_transcript.extract_audio(media_path, audio_path, progress, duration_sec)


def transcribe_video(
    video_path: Path,
    work_dir: Path,
    model_name: str,
    language: str,
    vad_filter: bool,
    progress: ProgressCounter | ProgressReporter | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    audio_path = work_dir / "audio.wav"
    audio_warning = extract_audio(
        video_path,
        audio_path,
        progress if isinstance(progress, ProgressReporter) else None,
    )
    if audio_warning:
        if isinstance(progress, ProgressReporter):
            progress.complete("audio_extraction", detail={"warning": audio_warning["code"]})
            progress.skip_cached("transcription", detail={"reason": audio_warning["code"]})
        return None, [audio_warning]
    transcript, warnings = transcribe_audio(audio_path, model_name, language, vad_filter, progress)
    if isinstance(progress, ProgressReporter):
        progress.complete(
            "transcription", detail={"segments": len((transcript or {}).get("segments", []))}
        )
    return transcript, warnings


__all__ = [
    "MIN_CONFIDENCE",
    "VAD_DROP_HIGH_RATIO",
    "VAD_DROP_WARNING_RATIO",
    "FasterWhisperAdapter",
    "WhisperAdapter",
    "extract_audio",
    "parse_ffmpeg_progress_time",
    "transcribe_audio",
    "transcribe_video",
    "vad_drop_warnings",
]
