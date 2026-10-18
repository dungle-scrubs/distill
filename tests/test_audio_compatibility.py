"""Exercise Whisper's audio decoder without a model, network, or external tools."""

from __future__ import annotations

import struct
import wave
from pathlib import Path


def test_whisper_can_decode_audio_with_installed_pyav(tmp_path: Path) -> None:
    from faster_whisper.audio import decode_audio

    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<1600h", *([1000] * 1600)))

    samples = decode_audio(str(path), sampling_rate=16000)

    assert samples.shape == (1600,)
    assert abs(float(samples[0]) - 1000 / 32768) < 1e-6
