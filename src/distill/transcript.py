"""Audio extraction and faster-whisper transcription helpers."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .capabilities import MISSING_TOOL_CODE, missing_tool_consequence
from .errors import DistillError, WarningRecord, warning
from .progress import ProgressCounter, ProgressReporter
from .run_command import CommandTimeouts, stream

MIN_CONFIDENCE = -1.0
# The one code every extraction failure reports under, named so the caller can
# say why it skipped transcription without being handed the warning back.
AUDIO_EXTRACT_FAILED_CODE = "audio_extract_failed"
VAD_DROP_WARNING_RATIO = 0.50
VAD_DROP_HIGH_RATIO = 0.90

# ffmpeg writes one "key=value" line per field in a -progress block. Only
# "out_time=" drives the audio_extraction mechanism: a block also carries
# "out_time_ms=" (and "out_time_us="), and acting on more than one key would
# emit duplicate events for the same tick.
FFMPEG_PROGRESS_TIME_KEY = "out_time="
FFMPEG_PROGRESS_KEYS = frozenset(
    {
        "bitrate",
        "drop_frames",
        "dup_frames",
        "fps",
        "frame",
        "out_time",
        "out_time_ms",
        "out_time_us",
        "progress",
        "speed",
        "total_size",
    }
)

# A degraded run must be able to say why (ADR-0002), so ffmpeg's own error text
# is kept - but bounded, since -progress makes stderr unbounded in the happy case.
STDERR_TAIL_LINES = 20
STDERR_TAIL_CHARS = 600

# Extraction is bounded by silence rather than by length (R-30): decoding a long
# recording is legitimately slow, but ffmpeg emits either a -progress block or a
# stats line while it works, so two minutes without a byte means it is wedged.
AUDIO_EXTRACT_TIMEOUTS = CommandTimeouts(total_sec=3 * 60 * 60.0, idle_sec=120.0)


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


def is_ffmpeg_progress_line(line: str) -> bool:
    """True when a stderr line is a -progress key=value field, not ffmpeg's log."""
    if "=" not in line:
        return False
    key = line.split("=", 1)[0].strip()
    return key in FFMPEG_PROGRESS_KEYS or (key.startswith("stream_") and key.endswith("_q"))


def format_stderr_tail(lines: Iterable[str]) -> str:
    tail = " | ".join(stripped for line in lines if (stripped := line.strip()))
    if len(tail) > STDERR_TAIL_CHARS:
        tail = tail[-STDERR_TAIL_CHARS:]
    return tail


def extract_audio(
    media_path: Path,
    audio_path: Path,
    progress: ProgressReporter | None = None,
    duration_sec: float | None = None,
) -> tuple[bool, list[WarningRecord]]:
    """Extract the audio track, reporting whether it landed and what it cost.

    Two answers rather than one warning, because they are independent: a
    successful extraction can still record a **warning** (truncated capture,
    R-33), and a caller that read only a warning would either lose it or read
    it as a failure.
    """
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

    stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)

    def read_stderr(line: str) -> None:
        """Consume one ffmpeg stderr line: a -progress field, or diagnostics.

        ffmpeg is told to write its -progress blocks to stderr ("-progress
        pipe:2"), so this is the stream to parse here - unlike yt-dlp, whose
        progress is on stdout (R-32).
        """
        if is_ffmpeg_progress_line(line):
            # A block reports one instant under several keys; only the
            # out_time= field drives an event, so a block emits exactly one.
            if not line.startswith(FFMPEG_PROGRESS_TIME_KEY):
                return
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
            return
        stderr_tail.append(line)

    try:
        result = stream(
            command,
            stage="transcript",
            total_timeout_sec=AUDIO_EXTRACT_TIMEOUTS.total_sec,
            idle_timeout_sec=AUDIO_EXTRACT_TIMEOUTS.idle_sec,
            on_stderr_line=read_stderr,
            check=False,
        )
    except DistillError as exc:
        # An absent tool is the capability table's decision, not this call
        # site's: ffmpeg is classified a **required capability**, so this call
        # raises rather than returning (ADR-0002, R-34). A wedged ffmpeg that is
        # installed is the degradation ADR-0002 does cover - the run loses its
        # transcript and keeps its keyframes.
        if exc.code == MISSING_TOOL_CODE:
            return False, [missing_tool_consequence("transcript", "ffmpeg", cause=exc)]
        return False, [
            warning(
                "transcript",
                AUDIO_EXTRACT_FAILED_CODE,
                f"ffmpeg could not extract audio: {exc.message}",
            )
        ]
    warnings = list(result.warnings)
    if result.returncode != 0:
        reason = format_stderr_tail(stderr_tail)
        message = "ffmpeg could not extract audio"
        if reason:
            message = f"{message}: {reason}"
        warnings.append(warning("transcript", AUDIO_EXTRACT_FAILED_CODE, message))
        return False, warnings
    if progress:
        progress.complete(
            "audio_extraction",
            detail={"duration_sec": duration_sec} if duration_sec is not None else None,
        )
    return True, warnings


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str,
    vad_filter: bool,
    progress: ProgressCounter | ProgressReporter | None = None,
    adapter: WhisperAdapter | None = None,
) -> tuple[dict[str, Any] | None, list[WarningRecord]]:
    warnings: list[WarningRecord] = []
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


def vad_drop_warnings(info: Any, vad_filter: bool) -> list[WarningRecord]:
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
    duration_sec: float | None = None,
) -> tuple[dict[str, Any] | None, list[WarningRecord]]:
    audio_path = work_dir / "audio.wav"
    extracted, audio_warnings = extract_audio(
        video_path,
        audio_path,
        progress if isinstance(progress, ProgressReporter) else None,
        duration_sec,
    )
    if not extracted:
        if isinstance(progress, ProgressReporter):
            progress.complete("audio_extraction", detail={"warning": AUDIO_EXTRACT_FAILED_CODE})
            progress.skip_cached("transcription", detail={"reason": AUDIO_EXTRACT_FAILED_CODE})
        return None, audio_warnings
    transcript, warnings = transcribe_audio(audio_path, model_name, language, vad_filter, progress)
    if isinstance(progress, ProgressReporter):
        progress.complete(
            "transcription", detail={"segments": len((transcript or {}).get("segments", []))}
        )
    # Extraction succeeded, but it may still have recorded truncated capture
    # (R-33); those warnings belong in the bundle beside transcription's own.
    return transcript, [*audio_warnings, *warnings]


__all__ = [
    "AUDIO_EXTRACT_FAILED_CODE",
    "MIN_CONFIDENCE",
    "VAD_DROP_HIGH_RATIO",
    "VAD_DROP_WARNING_RATIO",
    "FFMPEG_PROGRESS_KEYS",
    "FFMPEG_PROGRESS_TIME_KEY",
    "STDERR_TAIL_CHARS",
    "STDERR_TAIL_LINES",
    "FasterWhisperAdapter",
    "WhisperAdapter",
    "extract_audio",
    "format_stderr_tail",
    "is_ffmpeg_progress_line",
    "parse_ffmpeg_progress_time",
    "transcribe_audio",
    "transcribe_video",
    "vad_drop_warnings",
]


# The transcript context a keyframe is judged against (D-003): segments
# overlapping [timestamp - radius, timestamp + radius], in transcript order.
# 30s of speech either side is enough to say whether the frame adds anything
# beyond what is being said, without dragging in a different topic.
SALIENCE_WINDOW_RADIUS_SEC = 30.0
# ~4x the prompt's 1200-char cap: room for redaction to shorten text before
# the authoritative cut, without unbounded accumulation.
_WINDOW_CHAR_BOUND = 4800


def segment_bounds(segment: Any) -> tuple[float, float] | None:
    """Where one **transcript** segment sits on the recording's clock, or `None`.

    `None` is "this segment cannot be placed", and it covers every way that
    happens: not a mapping, either bound missing, a bound that is not a number,
    and a bound that is `nan` or infinite. Missing is included deliberately -
    a segment read as starting at zero because it said nothing has been moved
    to the beginning of the recording, which is a claim about when somebody
    spoke.

    One reading of the question, because two readers ask it and they must not
    answer differently: this module skips what it cannot place when it builds
    a salience window, and a read-side render refuses a document that holds
    one rather than rendering around it. Segments come back off disk on both
    routes (R-23), so neither reader may assume the shape this module wrote.
    """
    if not isinstance(segment, Mapping) or "start" not in segment or "end" not in segment:
        return None
    try:
        start, end = float(segment["start"]), float(segment["end"])
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    return start, end


def segment_words_are_placeable(segment: Any) -> bool:
    """Whether a segment's word grid is one a render can place frames inside.

    `segment_bounds` answers where a segment sits; this answers whether the
    *words* under it can be walked, and the two are separate questions because
    a render asks both. Interleaving speech with keyframes advances through
    `words` comparing each `end` against a frame's timestamp, so a word entry
    that is not a mapping, or one whose `end` is missing, unparseable or not
    finite, is not slow or wrong - it ends the walk on a `KeyError`, an
    `AttributeError` or a `TypeError` out of whatever command asked for the
    render. `words` absent entirely is placeable: the render then places the
    segment's frames against the segment's own bounds, which `segment_bounds`
    already answered for.

    Here rather than in either reader for the reason `segment_bounds` is here:
    segments come back off disk on the resume route and on the read-side one
    (R-23), and two readers asking whether a segment can be walked must not
    answer differently. What a word *says* is not this question - the render
    stringifies it, so any type survives - and only what places it is checked.
    """
    if not isinstance(segment, Mapping):
        return False
    words = segment.get("words")
    if words is None:
        return True
    if isinstance(words, str | bytes | bytearray) or not isinstance(words, Sequence):
        return False
    for word in words:
        if not isinstance(word, Mapping):
            return False
        try:
            end = float(word["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(end):
            return False
    return True


def select_transcript_window(
    segments: Iterable[Any],
    timestamp_sec: float,
    *,
    radius_sec: float = SALIENCE_WINDOW_RADIUS_SEC,
) -> str:
    """The speech surrounding a frame timestamp, or "" when there is none.

    "" is the emptiness predicate's answer, not a value to judge against: a
    zero-segment or whitespace-only transcript means the frame has no context
    to be salient *relative to*, so salience stays absent rather than being
    scored against nothing (D-003).

    A segment that cannot be placed - missing or non-numeric start/end, NaN,
    or not a mapping at all - is skipped, not defaulted: segments come back
    off disk on resume paths, and a corrupt one must neither leak into a
    window it never belonged to nor kill the run (ADR-0002).
    """
    low, high = timestamp_sec - radius_sec, timestamp_sec + radius_sec
    parts: list[str] = []
    accumulated = 0
    for segment in segments:
        if accumulated > _WINDOW_CHAR_BOUND:
            # Enough: the prompt applies its own authoritative cap after
            # redaction; accumulating an adversarial many-segment transcript
            # past several multiples of it is work with no reader (the same
            # bounded-work discipline as the redaction patterns).
            break
        bounds = segment_bounds(segment)
        if bounds is None:
            continue
        start, end = bounds
        if end >= low and start <= high:
            text = str(segment.get("text", "")).strip()
            if text:
                parts.append(text)
                accumulated += len(text)
    return " ".join(parts)
