from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saccade.progress import ProgressReporter
from saccade.transcript import (
    extract_audio,
    parse_ffmpeg_progress_time,
    transcribe_audio,
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


def test_extract_audio_reports_media_time_progress(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    class FakePopen:
        returncode = 0

        def __init__(self, command: list[str], **_kwargs: object) -> None:
            assert "-progress" in command
            self.stderr = iter(["out_time_ms=5000000\n"])

        def communicate(self) -> tuple[str, str]:
            return "", ""

    monkeypatch.setattr("saccade.transcript.subprocess.Popen", FakePopen)
    progress = ProgressReporter()

    warning = extract_audio(
        tmp_path / "video.mp4",
        tmp_path / "audio.wav",
        progress,
        duration_sec=10.0,
    )

    assert warning is None
    assert any(
        event.mechanism == "audio_extraction" and event.percent == 50.0 for event in progress.events
    )
    assert progress.events[-1].status == "completed"


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
