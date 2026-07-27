from __future__ import annotations

import json
import logging
from ast import literal_eval
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from distill import transcript
from distill.progress import ProgressReporter
from distill.transcript import (
    extract_audio,
    parse_ffmpeg_progress_time,
    transcribe_audio,
    transcribe_video,
)


@dataclass(frozen=True)
class FakeWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    words: list[FakeWord]


@dataclass(frozen=True)
class FakeInfo:
    language: str = "en"
    language_probability: float = 0.99
    duration: float = 10.0
    duration_after_vad: float = 10.0


class FakeWhisperAdapter:
    def __init__(self, info: FakeInfo) -> None:
        self.info = info
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
    ) -> tuple[list[FakeSegment], FakeInfo]:
        self.calls.append(
            {
                "audio_path": audio_path,
                "language": language,
                "vad_filter": vad_filter,
            }
        )
        return (
            [
                FakeSegment(
                    start=0.0,
                    end=1.0,
                    text="hello",
                    avg_logprob=-0.1,
                    words=[FakeWord("hello", 0.0, 1.0, 0.98)],
                )
            ],
            self.info,
        )


def test_whisper_adapter_is_mockable(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    adapter = FakeWhisperAdapter(FakeInfo())

    transcript, warnings = transcribe_audio(
        audio,
        "small",
        "en",
        True,
        adapter=adapter,
    )

    assert warnings == []
    assert transcript is not None
    assert transcript["segments"][0]["text"] == "hello"
    assert adapter.calls == [{"audio_path": audio, "language": "en", "vad_filter": True}]


def test_vad_drop_ratio_warning_tiers_are_emitted(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    _, medium_warnings = transcribe_audio(
        audio,
        "small",
        "en",
        True,
        adapter=FakeWhisperAdapter(FakeInfo(duration=10.0, duration_after_vad=4.0)),
    )
    _, high_warnings = transcribe_audio(
        audio,
        "small",
        "en",
        True,
        adapter=FakeWhisperAdapter(FakeInfo(duration=10.0, duration_after_vad=0.5)),
    )

    assert medium_warnings == [
        {
            "stage": "transcript",
            "code": "vad_drop_ratio_warning",
            "message": "VAD removed 60% of audio",
        }
    ]
    assert high_warnings == [
        {
            "stage": "transcript",
            "code": "vad_drop_ratio_high",
            "message": "VAD removed 95% of audio",
        }
    ]


def test_ffmpeg_progress_time_parses_supported_formats() -> None:
    assert parse_ffmpeg_progress_time("out_time_ms=2500000") == 2.5
    assert parse_ffmpeg_progress_time("out_time=00:01:02.500000") == 62.5
    assert parse_ffmpeg_progress_time("progress=continue") is None


# ffmpeg is told to write its `-progress` blocks to stderr ("-progress pipe:2"),
# so unlike yt-dlp, stderr is the stream this call site must read (R-32). The
# fakes below are real executables, because run_command spawns a real child.
FAKE_FFMPEG_ONE_BLOCK = """
import pathlib, sys

argv = sys.argv[1:]
output = pathlib.Path(argv[-1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(b"audio")
(output.parent / "ffmpeg-argv.txt").write_text(repr(argv))
if "-progress" in argv:
    # One real -progress block: the same instant reported under several keys.
    for line in [
        "frame=125",
        "fps=25.00",
        "stream_0_0_q=-0.0",
        "bitrate=  32.0kbits/s",
        "total_size=20480",
        "out_time_us=5000000",
        "out_time_ms=5000000",
        "out_time=00:00:05.000000",
        "dup_frames=0",
        "drop_frames=0",
        "speed=  50x",
        "progress=continue",
    ]:
        sys.stderr.write(line + "\\n")
    sys.stderr.flush()
"""


def fake_ffmpeg_failing(noise_lines: int) -> str:
    """A fake ffmpeg that emits one progress block, then noise, then dies."""
    return f"""
import sys

sys.stderr.write("out_time=00:00:05.000000\\n")
sys.stderr.write("progress=continue\\n")
for index in range({noise_lines}):
    sys.stderr.write("noise line " + str(index) + "\\n")
sys.stderr.write("video.mp4: Invalid data found when processing input\\n")
sys.exit(1)
"""


def test_transcribe_video_enables_audio_extraction_progress(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_tool("ffmpeg", FAKE_FFMPEG_ONE_BLOCK)
    monkeypatch.setattr(transcript, "transcribe_audio", lambda *_a, **_k: (None, []))

    transcribe_video(
        tmp_path / "video.mp4",
        tmp_path,
        "small",
        "en",
        True,
        ProgressReporter(),
        duration_sec=10.0,
    )

    command = literal_eval((tmp_path / "ffmpeg-argv.txt").read_text())
    progress_index = command.index("-progress")
    assert command[progress_index : progress_index + 2] == ["-progress", "pipe:2"]


def test_extract_audio_reports_media_time_progress(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_tool("ffmpeg", FAKE_FFMPEG_ONE_BLOCK)
    monkeypatch.setattr(transcript, "transcribe_audio", lambda *_a, **_k: (None, []))
    progress = ProgressReporter()

    _result, warnings = transcribe_video(
        tmp_path / "video.mp4",
        tmp_path,
        "small",
        "en",
        True,
        progress,
        duration_sec=10.0,
    )

    # 5 s of 10 s is exactly 50%: the percent is derived from the threaded
    # duration_sec, so an inexact assertion here would not notice the
    # duration failing to reach extract_audio at all.
    assert any(
        event.mechanism == "audio_extraction" and event.percent == 50.0
        for event in progress.events
    )
    # The 50% event is emitted inside the stderr loop, before the returncode is
    # ever checked, so the mid-run percent alone cannot tell a successful
    # extraction from one that emitted progress and then failed.
    assert warnings == []
    assert progress.states["audio_extraction"].status == "completed"


def test_extract_audio_emits_one_event_per_ffmpeg_progress_block(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    fake_tool("ffmpeg", FAKE_FFMPEG_ONE_BLOCK)
    progress = ProgressReporter()

    extract_audio(
        tmp_path / "video.mp4",
        tmp_path / "audio.wav",
        progress,
        duration_sec=10.0,
    )

    running_events = [
        event
        for event in progress.events
        if event.mechanism == "audio_extraction" and event.status == "running"
    ]
    # One 0% start plus exactly one tick for the single block.
    assert [event.percent for event in running_events] == [0.0, 50.0]


def test_extract_audio_failure_reports_bounded_ffmpeg_stderr(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    fake_tool("ffmpeg", fake_ffmpeg_failing(transcript.STDERR_TAIL_LINES + 40))

    result = extract_audio(
        tmp_path / "video.mp4",
        tmp_path / "audio.wav",
        ProgressReporter(),
        duration_sec=10.0,
    )

    assert result is not None
    assert result["code"] == "audio_extract_failed"
    assert "Invalid data found when processing input" in result["message"]
    # Bounded: only the tail is kept, and progress keys are never diagnostic.
    assert "noise line 40" not in result["message"]
    assert "out_time" not in result["message"]
    assert len(result["message"]) <= transcript.STDERR_TAIL_CHARS + 64


def test_transcription_progress_uses_segment_end_over_duration(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    progress = ProgressReporter()

    transcribe_audio(
        audio,
        "small",
        "en",
        True,
        progress=progress,
        adapter=FakeWhisperAdapter(FakeInfo(duration=2.0)),
    )

    transcription_events = [
        event for event in progress.events if event.mechanism == "transcription"
    ]
    assert transcription_events[-1].percent == 50.0


def test_faster_whisper_missing_degrades_without_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    def _raise(_model_name: str) -> object:
        raise ImportError("faster-whisper is not installed")

    monkeypatch.setattr(transcript, "FasterWhisperAdapter", _raise)

    result, warnings = transcribe_audio(audio, "small", "en", True)

    assert result is None
    assert warnings == [
        {
            "stage": "transcript",
            "code": "faster_whisper_missing",
            "message": "faster-whisper is not installed",
        }
    ]


def test_whisper_failure_is_caught_as_warning(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    class FailingAdapter:
        def transcribe(self, *_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
            raise RuntimeError("model exploded")

    result, warnings = transcribe_audio(audio, "small", "en", True, adapter=FailingAdapter())

    assert result is None
    assert warnings[0]["code"] == "whisper_failed"
    assert "model exploded" in warnings[0]["message"]


def test_low_confidence_segments_are_dropped_with_warning(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    class MixedAdapter:
        def transcribe(self, *_args: Any, **_kwargs: Any) -> tuple[list[FakeSegment], FakeInfo]:
            return (
                [
                    FakeSegment(0.0, 1.0, "kept", avg_logprob=-0.2, words=[]),
                    FakeSegment(1.0, 2.0, "dropped", avg_logprob=-1.5, words=[]),
                ],
                FakeInfo(),
            )

    transcript_result, warnings = transcribe_audio(
        audio, "small", "en", True, adapter=MixedAdapter()
    )

    assert transcript_result is not None
    assert [segment["text"] for segment in transcript_result["segments"]] == ["kept"]
    assert any(
        w["code"] == "low_confidence_segments_dropped" and "dropped 1" in w["message"]
        for w in warnings
    )


def test_extract_audio_degrades_when_ffmpeg_missing(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """ADR-0002: the run continues with a **warning** rather than raising.

    Nothing is installed into the fake PATH, so ffmpeg is genuinely absent and
    run_command's E_MISSING_TOOL is the error being degraded - not a patched
    exception standing in for one.
    """
    result = extract_audio(tmp_path / "video.mp4", tmp_path / "audio.wav")

    assert result == {
        "stage": "transcript",
        "code": "audio_extract_failed",
        "message": "ffmpeg is not installed",
    }


def test_extract_audio_emits_a_boundary_event(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """R-29: the invocation is visible at the boundary, named by its stage."""
    fake_tool("ffmpeg", FAKE_FFMPEG_ONE_BLOCK)

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        extract_audio(tmp_path / "video.mp4", tmp_path / "audio.wav")

    events = [json.loads(record.message) for record in caplog.records]
    assert [(event["detail"]["tool"], event["detail"]["stage"]) for event in events] == [
        ("ffmpeg", "transcript")
    ]


def test_transcribe_video_returns_audio_warning_without_transcribing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_warning = {
        "stage": "transcript",
        "code": "audio_extract_failed",
        "message": "ffmpeg is not installed",
    }
    monkeypatch.setattr(transcript, "extract_audio", lambda *_a, **_k: audio_warning)

    def _should_not_run(*_a: Any, **_k: Any) -> tuple[Any, Any]:
        raise AssertionError("transcription must not run after an audio-extract failure")

    monkeypatch.setattr(transcript, "transcribe_audio", _should_not_run)

    result, warnings = transcribe_video(
        tmp_path / "video.mp4", tmp_path, "small", "en", True
    )

    assert result is None
    assert warnings == [audio_warning]
