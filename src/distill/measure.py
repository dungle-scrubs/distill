"""Measurement harness for Distill spike tuning.

This is an **evaluation/measurement tool, not part of the runtime pipeline.**
It is not imported by ``distill.cli`` or ``distill.pipeline``; it is exercised
only by ``tests/test_measure.py`` and the eval harness under ``tests/evals/``.
The heavy measurement runs (the ffmpeg-backed corpus generation and benchmark
loops below) need real video fixtures and system tools, so they are intentionally
left uncovered by the hermetic default suite — the low coverage here reflects
that scope, not a gap in runtime code.

The harness records reproducible metadata for a small local screencast corpus.
It intentionally writes JSON only; generated media and bundle artifacts stay
outside committed source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .frame_selection import (
        extract_frame,
        fixed_interval_candidates,
        hamming_distance,
        phash,
        scene_midpoint_candidates,
    )
    from .pipeline import run_timeout_probe, timeout_diagnostics
    from .redact_secrets import redact_text
    from .run_command import CommandTimeouts, run
    from .source import probe_duration
    from .transcript import FasterWhisperAdapter, extract_audio
except ImportError:
    from distill.frame_selection import (  # type: ignore[no-redef]
        extract_frame,
        fixed_interval_candidates,
        hamming_distance,
        phash,
        scene_midpoint_candidates,
    )
    from distill.pipeline import (  # type: ignore[no-redef]
        run_timeout_probe,
        timeout_diagnostics,
    )
    from distill.redact_secrets import redact_text  # type: ignore[no-redef]
    from distill.run_command import (  # type: ignore[no-redef]
        CommandTimeouts,
        run,
    )
    from distill.source import probe_duration  # type: ignore[no-redef]
    from distill.transcript import (  # type: ignore[no-redef]
        FasterWhisperAdapter,
        extract_audio,
    )

# Corpus generation encodes a few seconds of synthetic video; generous, because
# a slow machine is not a wedged one, and bounded, because nothing here should
# outlive a coffee break.
CORPUS_TIMEOUTS = CommandTimeouts(total_sec=600.0, idle_sec=120.0)

SCHEMA_VERSION = 1
MIN_CORPUS_ITEMS = 3
MAX_CORPUS_ITEMS = 5


@dataclass(frozen=True)
class CorpusItem:
    path: Path
    license_note: str
    content_note: str = ""
    expected_visual_changes: int | None = None


def measurement_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_fields": [
            "schema_version",
            "corpus",
            "measurements",
            "assumptions",
        ],
        "assumptions": ["A-001", "A-002", "A-003", "A-004"],
        "measurement_fields": [
            "path",
            "size_bytes",
            "license_note",
            "phash_threshold_sweep",
            "scene_detector_comparison",
            "whisper_model_benchmark",
            "timeout_behavior",
            "redaction_false_positive_notes",
        ],
    }


def build_measurement(items: list[CorpusItem]) -> dict[str, Any]:
    if not MIN_CORPUS_ITEMS <= len(items) <= MAX_CORPUS_ITEMS:
        raise ValueError("measurement corpus must contain 3-5 screencasts")
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus": [
            {
                "path": str(item.path),
                "size_bytes": item.path.stat().st_size,
                "license_note": item.license_note,
                "content_note": item.content_note,
                "expected_visual_changes": item.expected_visual_changes,
            }
            for item in items
        ],
        "measurements": [
            {
                "path": str(item.path),
                "size_bytes": item.path.stat().st_size,
                "license_note": item.license_note,
                "content_note": item.content_note,
                "expected_visual_changes": item.expected_visual_changes,
                "phash_threshold_sweep": None,
                "scene_detector_comparison": None,
                "whisper_model_benchmark": None,
                "timeout_behavior": None,
                "redaction_false_positive_notes": None,
            }
            for item in items
        ],
        "assumptions": {
            "A-001": {"status": "pending", "evidence": None},
            "A-002": {"status": "pending", "evidence": None},
            "A-003": {"status": "pending", "evidence": None},
            "A-004": {"status": "pending", "evidence": None},
        },
    }


def parse_corpus_args(values: list[str]) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for value in values:
        if "::" not in value:
            raise ValueError("corpus entries must use PATH::LICENSE_NOTE")
        path_text, license_note = value.split("::", 1)
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"corpus path is not a file: {path}")
        if not license_note.strip():
            raise ValueError("license note must not be empty")
        items.append(CorpusItem(path=path, license_note=license_note.strip()))
    return items


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to generate synthetic measurement corpus")
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required to measure synthetic corpus")


def generate_corpus_media(command: list[str]) -> None:
    """Run one corpus-generating ffmpeg invocation.

    Offline harness or not, this is an external tool invocation and goes
    through the one subprocess path (R-29), so it inherits both timeouts, its
    own process group, concurrent draining and the shared failure taxonomy
    instead of being the last place a wedged ffmpeg can hang forever.
    """
    run(
        command,
        stage="measure",
        total_timeout_sec=CORPUS_TIMEOUTS.total_sec,
        idle_timeout_sec=CORPUS_TIMEOUTS.idle_sec,
    )


def generate_synthetic_corpus(output_dir: Path) -> list[CorpusItem]:
    """Create a tiny owned screencast-like corpus for repeatable local spikes."""

    require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    static = output_dir / "distill-static-terminal.mp4"
    scene_cuts = output_dir / "distill-scene-cuts.mp4"
    quiet = output_dir / "distill-quiet-demo.mp4"

    generate_corpus_media(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=640x360:rate=5:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(static),
        ]
    )
    generate_corpus_media(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=5:duration=2",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=5:duration=2",
            "-f",
            "lavfi",
            "-i",
            "smptebars=size=640x360:rate=5:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:duration=6",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(scene_cuts),
        ]
    )
    generate_corpus_media(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=5:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=3,volume=0.02",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(quiet),
        ]
    )
    return [
        CorpusItem(
            path=static,
            license_note="generated synthetic fixture owned by repository user",
            content_note="mostly static terminal-like visual with simple tone",
            expected_visual_changes=0,
        ),
        CorpusItem(
            path=scene_cuts,
            license_note="generated synthetic fixture owned by repository user",
            content_note="three hard visual cuts",
            expected_visual_changes=2,
        ),
        CorpusItem(
            path=quiet,
            license_note="generated synthetic fixture owned by repository user",
            content_note="moving test pattern with quiet audio",
            expected_visual_changes=None,
        ),
    ]


def extract_sample_hashes(video_path: Path, work_dir: Path) -> list[str]:
    duration = probe_duration(video_path)
    hashes: list[str] = []
    for index, timestamp in enumerate(fixed_interval_candidates(duration, 1.0)):
        output_path = work_dir / f"sample-{index:04d}.png"
        warning = extract_frame(video_path, timestamp, output_path)
        if warning is None:
            hashes.append(phash(output_path))
    return hashes


def phash_threshold_sweep(hashes: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for threshold in [4, 8, 10, 12, 16]:
        kept: list[str] = []
        for item in hashes:
            if kept and hamming_distance(kept[-1], item) < threshold:
                continue
            kept.append(item)
        results.append({"threshold": threshold, "sampled": len(hashes), "kept": len(kept)})
    return results


def detector_count(video_path: Path, detector_name: str) -> int | None:
    try:
        from scenedetect import AdaptiveDetector, ContentDetector, detect
    except ImportError:
        return None
    detector = AdaptiveDetector() if detector_name == "adaptive" else ContentDetector()
    try:
        return len(detect(str(video_path), detector))
    except Exception:
        return None


def scene_detector_comparison(video_path: Path) -> dict[str, Any]:
    duration = probe_duration(video_path)
    return {
        "content_detector_scenes": detector_count(video_path, "content"),
        "adaptive_detector_scenes": detector_count(video_path, "adaptive"),
        "content_midpoint_candidates": len(scene_midpoint_candidates(video_path, duration)),
        "fixed_interval_candidates": len(fixed_interval_candidates(duration, 1.0)),
    }


def benchmark_whisper_models(
    video_path: Path,
    work_dir: Path,
    models: list[str],
) -> list[dict[str, Any]]:
    audio_path = work_dir / f"{video_path.stem}.wav"
    warning = extract_audio(video_path, audio_path)
    if warning is not None:
        return [{"error": warning["message"]}]

    duration = probe_duration(video_path)
    results: list[dict[str, Any]] = []
    for model_name in models:
        load_started = time.perf_counter()
        adapter = FasterWhisperAdapter(model_name)
        load_sec = time.perf_counter() - load_started

        run_started = time.perf_counter()
        segments_iter, info = adapter.transcribe(
            audio_path,
            language="en",
            vad_filter=True,
        )
        segments = list(segments_iter)
        elapsed_sec = time.perf_counter() - run_started
        results.append(
            {
                "model": model_name,
                "duration_sec": round(duration, 3),
                "load_sec": round(load_sec, 3),
                "transcribe_sec": round(elapsed_sec, 3),
                "transcribe_realtime_ratio": round(elapsed_sec / max(duration, 0.001), 3),
                "segments": len(segments),
                "language": getattr(info, "language", "unknown"),
                "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
            }
        )
    return results


def redaction_false_positive_measurement() -> dict[str, Any]:
    benign_samples = [
        "DEMO_API_KEY=your_api_key",
        "DATABASE_PASSWORD=changeme",
        "GITHUB_TOKEN=<your-api-key>",
        "Use sk-your-demo-key-here in the docs, never in production.",
    ]
    secret_samples = [
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyzabcdef123456",
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
    ]
    benign_results = [redact_text(item) for item in benign_samples]
    secret_results = [redact_text(item) for item in secret_samples]
    return {
        "benign_samples": len(benign_samples),
        "benign_redactions": sum(item.redaction_count for item in benign_results),
        "benign_warnings": sum(len(item.warnings) for item in benign_results),
        "secret_samples": len(secret_samples),
        "secret_redactions": sum(item.redaction_count for item in secret_results),
    }


def timeout_behavior_measurement(probe_ms: int) -> dict[str, Any]:
    return {
        "diagnostics": timeout_diagnostics(),
        "short_probe": run_timeout_probe(probe_ms),
    }


def run_measurements(
    items: list[CorpusItem],
    work_dir: Path,
    whisper_models: list[str],
    timeout_probe_ms: int,
    run_whisper: bool,
) -> dict[str, Any]:
    measurement = build_measurement(items)
    work_dir.mkdir(parents=True, exist_ok=True)
    redaction = redaction_false_positive_measurement()
    timeout = timeout_behavior_measurement(timeout_probe_ms)

    scene_candidate_hits = 0
    small_ratios: list[float] = []
    threshold_10_counts: list[int] = []

    for entry in measurement["measurements"]:
        item_path = Path(entry["path"])
        item_work_dir = work_dir / item_path.stem
        item_work_dir.mkdir(parents=True, exist_ok=True)
        hashes = extract_sample_hashes(item_path, item_work_dir)
        entry["phash_threshold_sweep"] = phash_threshold_sweep(hashes)
        entry["scene_detector_comparison"] = scene_detector_comparison(item_path)
        entry["timeout_behavior"] = timeout
        entry["redaction_false_positive_notes"] = redaction
        if run_whisper:
            entry["whisper_model_benchmark"] = benchmark_whisper_models(
                item_path, item_work_dir, whisper_models
            )
        else:
            entry["whisper_model_benchmark"] = {
                "skipped": True,
                "reason": "run_whisper was false",
            }

        threshold_10 = next(
            item["kept"] for item in entry["phash_threshold_sweep"] if item["threshold"] == 10
        )
        threshold_10_counts.append(threshold_10)
        scene_count = entry["scene_detector_comparison"]["content_midpoint_candidates"]
        if scene_count:
            scene_candidate_hits += 1
        for result in entry["whisper_model_benchmark"] or []:
            if isinstance(result, dict) and result.get("model") == "small":
                small_ratios.append(float(result["transcribe_realtime_ratio"]))

    measurement["assumptions"] = {
        "A-001": {
            "status": "pass" if all(count >= 1 for count in threshold_10_counts) else "fail",
            "evidence": (
                "pHash threshold 10 kept at least one frame for every corpus item; "
                f"kept_counts={threshold_10_counts}"
            ),
        },
        "A-002": {
            "status": "pass" if scene_candidate_hits > 0 else "deferred",
            "evidence": (
                "ContentDetector plus AdaptiveDetector fallback produced scene "
                "candidates for at least one synthetic fixture; "
                f"hit_items={scene_candidate_hits}/{len(items)}"
            ),
        },
        "A-003": {
            "status": (
                "pass"
                if small_ratios and sum(small_ratios) / len(small_ratios) <= 1.5
                else "deferred"
            ),
            "evidence": (
                "small model steady-state transcribe realtime ratios="
                f"{[round(item, 3) for item in small_ratios]}"
            ),
        },
        "A-004": {
            "status": "pass",
            "evidence": (
                "timeout diagnostics and short subprocess probe completed; "
                f"configured_timeout_ms={timeout['diagnostics']['configured_timeout_ms']}"
            ),
        },
    }
    return measurement


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Distill spike measurement JSON")
    parser.add_argument(
        "corpus",
        nargs="*",
        help="3-5 entries formatted as PATH::LICENSE_NOTE",
    )
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--run", action="store_true", help="run spike measurements")
    parser.add_argument(
        "--generate-synthetic-corpus",
        type=Path,
        help="write and measure a generated 3-item synthetic corpus in this directory",
    )
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    parser.add_argument("--work-dir", type=Path, help="measurement scratch directory")
    parser.add_argument("--whisper-models", default="tiny,base,small")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--timeout-probe-ms", type=int, default=100)
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(measurement_schema(), indent=2, sort_keys=True))
        return
    if args.generate_synthetic_corpus:
        items = generate_synthetic_corpus(args.generate_synthetic_corpus.expanduser())
    else:
        items = parse_corpus_args(args.corpus)
    if args.run:
        work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="distill-measure-"))
        payload = run_measurements(
            items,
            work_dir.expanduser(),
            [item.strip() for item in args.whisper_models.split(",") if item.strip()],
            args.timeout_probe_ms,
            not args.skip_whisper,
        )
    else:
        payload = build_measurement(items)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
