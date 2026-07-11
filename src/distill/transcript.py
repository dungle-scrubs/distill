"""Audio extraction and faster-whisper transcription helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Protocol

from .errors import warning
from .progress import ProgressCounter, ProgressReporter

MIN_CONFIDENCE = -1.0
VAD_DROP_WARNING_RATIO = 0.50
VAD_DROP_HIGH_RATIO = 0.90


class WhisperAdapter(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
    ) -> tuple[Any, Any]: ...


class FasterWhisperAdapter:
    def __init__(self, model_name: str) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="auto", compute_type="auto")

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
    ) -> tuple[Any, Any]:
        return self.model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
        )


def parse_ffmpeg_progress_time(line: str) -> float | None:
    if line.startswith("out_time_ms="):
        try:
            return float(line.split("=", 1)[1]) / 1_000_000
        except ValueError:
            return None
    if line.startswith("out_time="):
        raw = line.split("=", 1)[1].strip()
        parts = raw.split(":")
        if len(parts) != 3:
            return None
        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
        except ValueError:
            return None
        return hours * 3600 + minutes * 60 + seconds
    return None


def extract_audio(
    media_path: Path,
    audio_path: Path,
    progress: ProgressReporter | None = None,
    duration_sec: float | None = None,
) -> dict[str, str] | None:
    if progress:
        progress.update("audio_extraction", percent=0.0)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if progress and duration_sec and duration_sec > 0:
        command.extend(["-progress", "pipe:2"])
    command.append(str(audio_path))

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        # ffmpeg not installed: degrade to no-transcript rather than crashing.
        return warning("transcript", "audio_extract_failed", "ffmpeg is not installed")
    stderr_lines: list[str] = []
    if proc.stderr:
        for line in proc.stderr:
            stderr_lines.append(line)
            progress_sec = parse_ffmpeg_progress_time(line)
            if progress_sec is not None and progress and duration_sec and duration_sec > 0:
                progress.update(
                    "audio_extraction",
                    percent=(progress_sec / duration_sec) * 100,
                    detail={
                        "processed_sec": progress_sec,
                        "duration_sec": duration_sec,
                    },
                )
    _stdout, stderr_tail = proc.communicate()
    if stderr_tail:
        stderr_lines.append(stderr_tail)
    if proc.returncode != 0:
        return warning("transcript", "audio_extract_failed", "ffmpeg could not extract audio")
    if progress:
        progress.complete(
            "audio_extraction",
            detail={"duration_sec": duration_sec} if duration_sec is not None else None,
        )
    return None


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str,
    vad_filter: bool,
    progress: ProgressCounter | ProgressReporter | None = None,
    adapter: WhisperAdapter | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    warnings: list[dict[str, str]] = []
    if adapter is None:
        try:
            adapter = FasterWhisperAdapter(model_name)
        except ImportError:
            return None, [
                warning(
                    "transcript",
                    "faster_whisper_missing",
                    "faster-whisper is not installed",
                )
            ]

    try:
        segments_iter, info = adapter.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
        )
        segments: list[dict[str, Any]] = []
        dropped = 0
        for segment in segments_iter:
            avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
            if avg_logprob < MIN_CONFIDENCE:
                dropped += 1
                continue
            words = [
                {
                    "word": getattr(word, "word", ""),
                    "start": float(getattr(word, "start", segment.start)),
                    "end": float(getattr(word, "end", segment.end)),
                    "probability": float(getattr(word, "probability", 0.0)),
                }
                for word in getattr(segment, "words", []) or []
            ]
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": str(segment.text).strip(),
                    "avg_logprob": avg_logprob,
                    "words": words,
                }
            )
            if isinstance(progress, ProgressReporter):
                duration = float(getattr(info, "duration", 0.0) or 0.0)
                percent = (float(segment.end) / duration) * 100 if duration > 0 else None
                progress.update(
                    "transcription",
                    percent=percent,
                    detail={
                        "segment_end_sec": float(segment.end),
                        "segments": len(segments),
                    },
                )
            elif progress:
                progress.increment()
        if dropped:
            warnings.append(
                warning(
                    "transcript",
                    "low_confidence_segments_dropped",
                    f"dropped {dropped} low-confidence segments",
                )
            )
        warnings.extend(vad_drop_warnings(info, vad_filter))
        return {
            "language": getattr(info, "language", language),
            "language_probability": float(getattr(info, "language_probability", 0.0)),
            "segments": segments,
        }, warnings
    except Exception as exc:
        return None, [warning("transcript", "whisper_failed", f"Whisper failed: {exc}")]


def vad_drop_warnings(info: Any, vad_filter: bool) -> list[dict[str, str]]:
    if not vad_filter:
        return []
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    duration_after_vad = float(getattr(info, "duration_after_vad", duration) or 0.0)
    if duration <= 0 or duration_after_vad >= duration:
        return []
    drop_ratio = max(0.0, min(1.0, 1 - (duration_after_vad / duration)))
    if drop_ratio >= VAD_DROP_HIGH_RATIO:
        return [
            warning(
                "transcript",
                "vad_drop_ratio_high",
                f"VAD removed {drop_ratio:.0%} of audio",
            )
        ]
    if drop_ratio >= VAD_DROP_WARNING_RATIO:
        return [
            warning(
                "transcript",
                "vad_drop_ratio_warning",
                f"VAD removed {drop_ratio:.0%} of audio",
            )
        ]
    return []


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
