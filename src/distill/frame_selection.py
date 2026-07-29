"""Keyframe candidate selection and extraction for Distill.

This module owns scene/fallback candidate timestamps, pHash dedupe, static
window safeguards, and PNG extraction. It does not OCR or render frames.

It does not own the shape of a **frame artifact** either: it produces one per
**keyframe** it keeps and knows only the fields it can answer for - where the
image is, when in the source it came from, and what its hash was. Everything
later added to a frame is added by the stage that learns it, onto the carrier
this module created (R-19).

It is where a run's **redaction** policy enters the frames, once: the policy is
recorded on the artifact at birth and travels with it through every later stage,
so `--no-redact-secrets` is honoured without each stage being told separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from .artifacts import FrameArtifact, RedactionState
from .capabilities import MISSING_TOOL_CODE, missing_tool_consequence
from .errors import DistillError, WarningRecord, warning
from .progress import ProgressCounter, ProgressReporter
from .run_command import CommandTimeouts, run

PHASH_DISTANCE_THRESHOLD = 10
MAX_CANDIDATE_SCHEDULE = 500_000
"""The most candidate timestamps a schedule may hold before the option tuple
that produced it is refused (D-009).

The gap-filling walk builds one candidate every `max_static_window_sec` across
the source, so its length is bounded by `duration / max_static_window_sec` -
finite, but nothing caps `max_duration_sec`, so a large cap paired with a narrow
window is a schedule bounded only by memory. There is no maximum accepted source
duration to derive a ceiling from, so the bound is placed on the candidate count
itself: a worst-case schedule of half a million keyframes is already far past
any real use (a two-hour source sampled every 14 milliseconds), and past it the
tuple is unphysical rather than merely large.
"""

CANDIDATE_TIMESTAMP_QUANTUM_SEC = 0.001
"""The smallest difference between two candidate timestamps that survives.

Candidate timestamps are rounded to the millisecond - it is what `ffmpeg -ss`
is given (`{:.3f}`) and what a **frame artifact** carries - so two candidates
closer together than this are the same seek. It is therefore also the smallest
`max_static_window_sec` a schedule can express, and `options` refuses anything
below it rather than quietly widening it to a spacing the operator did not ask
for (R-48).
"""
# Wall-clock ceiling for a single ffmpeg frame grab so a wedged decode cannot
# hang the whole run. Grabbing one frame is short even on a long source - the
# seek is not a scan - so here the total deadline is the meaningful limit and
# the idle timeout only catches a decode that stops talking sooner.
FRAME_EXTRACT_TIMEOUTS = CommandTimeouts(total_sec=120.0, idle_sec=60.0)


def scene_midpoint_candidates(video_path: Path, duration_sec: float) -> list[float]:
    try:
        from scenedetect import AdaptiveDetector, ContentDetector, detect
    except ImportError:
        return []
    try:
        scenes = detect(str(video_path), ContentDetector())
    except Exception:
        return []
    if not scenes:
        try:
            scenes = detect(str(video_path), AdaptiveDetector())
        except Exception:
            scenes = []
    candidates: list[float] = []
    for start, end in scenes:
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        midpoint = max(0.0, min(duration_sec, (start_sec + end_sec) / 2))
        candidates.append(midpoint)
    return candidates


def quantized_timestamp(seconds: float, ceiling_sec: float) -> float:
    """One candidate timestamp as the schedule carries it: to the millisecond.

    Inside `[0, ceiling_sec]` whatever it is handed, and past the ceiling never
    - not even when the nearest millisecond is: rounding a value that sits
    within half a quantum of the end of the source produced a seek half a
    millisecond beyond it, which is a **keyframe** ffmpeg has no frame for. Use
    `math.inf` for a walk the end of the source does not bound.
    """
    point = round(min(max(seconds, 0.0), ceiling_sec), 3)
    return point if point <= ceiling_sec else round(point - CANDIDATE_TIMESTAMP_QUANTUM_SEC, 3)


def _step_from(anchor: float, step_sec: float, ceiling_sec: float) -> float | None:
    """The next schedule timestamp after `anchor`, or `None` if there is none.

    Every loop that walks a schedule forward asks this, and it answers with a
    timestamp strictly greater than `anchor` or with nothing - which is what
    makes those loops end (R-48). Two ways to have nothing: the ceiling has
    been reached, or `anchor` is so large that a step of this size disappears
    into the float. Neither is worth a step that stands still, and neither is
    worth silently taking a *bigger* step than asked for: a window the operator
    named is refused at the boundary, not rounded up here.
    """
    point = quantized_timestamp(anchor + step_sec, ceiling_sec)
    return point if point > anchor else None


def ensure_static_window_is_expressible(max_static_window_sec: float) -> None:
    """Refuse a static window no schedule can express (R-48, finding 7).

    Below one **quantum** the gap-filling cursor rounds back onto the timestamp
    it started from, so the loop that fills toward a candidate never arrives.
    Refusing states the floor; widening the window to the quantum would answer
    an operator who asked for 0.0001s with a schedule ten times coarser and no
    way to tell.
    """
    if max_static_window_sec >= CANDIDATE_TIMESTAMP_QUANTUM_SEC:
        return
    raise DistillError(
        "E_BAD_OPTIONS",
        "frames",
        "max_static_window_sec must be a finite number "
        f"{CANDIDATE_TIMESTAMP_QUANTUM_SEC} or greater",
        {"max_static_window_sec": repr(max_static_window_sec)},
    )


def fixed_interval_candidates(duration_sec: float, interval_sec: float) -> list[float]:
    """Sample the source every `interval_sec`, from zero, staying inside it.

    The walk is unbounded rather than clamped to `duration_sec`: a step that
    lands on or past the end of the source ends the schedule, and clamping it
    to the end would sample there instead - a step of 90 seconds across a
    ten-second source would produce two samples where the interval asked for
    one.

    `interval_sec` is taken as given, with no floor of its own. This is the
    schedule a source no detector answered for gets - the static-slide case
    `max_static_window_sec` exists for - so a floor here would answer a
    validated 0.002s window with a schedule 500 times coarser and no way to
    tell, which is the widening `ensure_static_window_is_expressible` refuses
    at the other door. Termination does not need one either: `_step_from`
    answers with a strictly greater timestamp or with nothing, and the
    validated quantum floor keeps that step from rounding back onto its anchor
    (R-48).
    """
    if duration_sec <= 0:
        return []
    values = [0.0]
    while True:
        current = _step_from(values[-1], interval_sec, math.inf)
        if current is None or current >= duration_sec:
            break
        values.append(current)
    return values


def filtered_candidates(
    candidates: Iterable[float],
    duration_sec: float,
    min_interval_sec: float,
    max_static_window_sec: float,
) -> list[float]:
    """The timestamps worth seeking to, in order, each at least a quantum apart.

    Gap filling is what makes this more than a sort: a stretch longer than
    `max_static_window_sec` with no scene change in it is still sampled, so a
    slide left on screen for ten minutes is not one **keyframe**. That walk
    forward is also the only unbounded thing here, which is why every step it
    takes comes from `_step_from` (R-48).
    """
    ensure_static_window_is_expressible(max_static_window_sec)
    sorted_candidates = sorted({quantized_timestamp(c, duration_sec) for c in candidates})
    if not sorted_candidates:
        sorted_candidates = fixed_interval_candidates(duration_sec, max_static_window_sec)
        return sorted_candidates
    kept: list[float] = []
    last_candidate = -(10**9)
    for candidate in sorted_candidates:
        if not kept and candidate > max_static_window_sec:
            kept.append(0.0)
        while kept and candidate - kept[-1] > max_static_window_sec:
            fill = _step_from(kept[-1], max_static_window_sec, duration_sec)
            if fill is None:
                break
            kept.append(fill)
        if candidate - last_candidate >= min_interval_sec:
            kept.append(candidate)
            last_candidate = candidate
    while kept and duration_sec - kept[-1] > max_static_window_sec:
        fill = _step_from(kept[-1], max_static_window_sec, duration_sec)
        if fill is None:
            break
        kept.append(fill)
    return sorted(set(kept))


def extract_frame(
    video_path: Path, timestamp_sec: float, output_path: Path
) -> tuple[bool, list[WarningRecord]]:
    """Grab one frame, reporting whether it landed and what the attempt cost.

    Two answers rather than one warning, because they are independent: an
    invocation can produce the frame *and* record a **warning** (truncated
    capture, R-33), and a caller that read only a warning would throw the frame
    away or, worse, keep the warning to itself.
    """
    try:
        result = run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp_sec:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_path),
            ],
            stage="frames",
            total_timeout_sec=FRAME_EXTRACT_TIMEOUTS.total_sec,
            idle_timeout_sec=FRAME_EXTRACT_TIMEOUTS.idle_sec,
            check=False,
        )
    except DistillError as exc:
        # An absent tool is the capability table's decision, not this call
        # site's: ffmpeg is classified a **required capability**, so this call
        # raises rather than returning (ADR-0002, R-34). A wedged ffmpeg is a
        # different matter - the tool is installed, one keyframe was lost, and
        # that reduces the bundle without ending the run.
        if exc.code == MISSING_TOOL_CODE:
            return False, [missing_tool_consequence("frames", "ffmpeg", cause=exc)]
        return False, [
            warning(
                "frames",
                "frame_extract_timeout",
                f"ffmpeg timed out extracting frame at {timestamp_sec:.3f}s",
            )
        ]
    warnings = list(result.warnings)
    if result.returncode != 0:
        warnings.append(
            warning(
                "frames",
                "frame_extract_failed",
                f"could not extract frame at {timestamp_sec:.3f}s",
            )
        )
        return False, warnings
    return True, warnings


def phash(path: Path) -> str:
    import imagehash
    from PIL import Image

    with Image.open(path) as image:
        thumbnail = image.convert("RGB").resize((256, 256))
        return str(imagehash.phash(thumbnail))


def hamming_distance(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def select_keyframes(
    video_path: Path,
    frames_dir: Path,
    duration_sec: float,
    max_keyframes: int,
    min_interval_sec: float,
    max_static_window_sec: float,
    progress: ProgressCounter | ProgressReporter | None = None,
    redaction: RedactionState = RedactionState.NOT_APPLIED,
) -> tuple[list[FrameArtifact], list[WarningRecord]]:
    """Choose the **keyframes** worth interpreting and produce one artifact each.

    `redaction` is the run's policy and is recorded on every artifact this
    returns. Nothing here is **extracted text** - a timestamp and a pHash are
    Distill's own observations - so the policy has nothing to do yet; it is set
    here because this is where a frame begins, and a policy chosen once at the
    source is a policy no later stage can forget to pass on (D-020).
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[WarningRecord] = []
    candidates = filtered_candidates(
        scene_midpoint_candidates(video_path, duration_sec),
        duration_sec,
        min_interval_sec,
        max_static_window_sec,
    )
    if isinstance(progress, ProgressReporter):
        progress.complete(
            "scene_detection",
            detail={"candidate_count": len(candidates), "duration_sec": duration_sec},
        )
    frames: list[FrameArtifact] = []
    last_hash: str | None = None
    unexamined = 0
    for index, timestamp in enumerate(candidates):
        if len(frames) >= max_keyframes:
            unexamined = len(candidates) - index
            break
        path = frames_dir / f"frame_{len(frames) + 1:04d}.png"
        extracted, frame_warnings = extract_frame(video_path, timestamp, path)
        if isinstance(progress, ProgressReporter):
            progress.update(
                "frame_extraction",
                percent=((index + 1) / max(1, len(candidates))) * 100,
                detail={
                    "candidate_index": index + 1,
                    "candidate_count": len(candidates),
                    "kept_frames": len(frames),
                    "max_keyframes": max_keyframes,
                },
            )
        elif progress:
            progress.increment()
        warnings.extend(frame_warnings)
        if not extracted:
            continue
        try:
            frame_hash = phash(path)
        except Exception as exc:
            warnings.append(warning("frames", "phash_failed", f"could not hash frame: {exc}"))
            frame_hash = ""
        if (
            last_hash
            and frame_hash
            and hamming_distance(last_hash, frame_hash) < PHASH_DISTANCE_THRESHOLD
        ):
            path.unlink(missing_ok=True)
            continue
        last_hash = frame_hash or last_hash
        artifact = FrameArtifact(
            index=len(frames) + 1,
            timestamp_sec=timestamp,
            path=str(path),
            relative_path=f"frames/{path.name}",
            phash=frame_hash,
            source_candidate_index=index,
            redaction=redaction,
        )
        warnings.extend(dict(item) for item in artifact.warnings)
        frames.append(artifact)
    if unexamined:
        # The **bundle** covers less of the source than the schedule asked
        # for, and nothing downstream can tell: a run that stopped early and
        # one whose source only had this much in it publish the same frames.
        # Only when it happened - a run that spent its budget exactly is not
        # a run that lost anything.
        warnings.append(
            warning(
                "frames",
                "keyframe_budget_reached",
                f"stopped at max_keyframes={max_keyframes}; {unexamined} of "
                f"{len(candidates)} candidate timestamps were not examined",
            )
        )
    if isinstance(progress, ProgressReporter):
        progress.complete(
            "frame_extraction",
            detail={"candidate_count": len(candidates), "kept_frames": len(frames)},
        )
    return frames, warnings
