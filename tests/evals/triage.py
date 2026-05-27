"""Survey cached Saccade frames and propose a stratified eval shortlist.

This is a developer utility, not a test. It reads every cached bundle's
``_ocr.json`` (so it needs no fresh OCR), measures each frame's mean luminance,
and buckets frames by background brightness and OCR yield. It then prints a
spread of candidates across those buckets so a human can pick a 16-frame eval
set that covers clean slides, hard dark slides, and text-free negatives.

Run:  uv run python tests/evals/triage.py [--cache DIR] [--limit N]

Selection is still a human call; this only surfaces the range.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat

DARK_LUMINANCE = 110.0  # matches saccade.ocr._DARK_BACKGROUND_THRESHOLD
LOW_TEXT_CHARS = 8
HIGH_TEXT_CHARS = 60


@dataclass(frozen=True)
class FrameSignal:
    bundle: str
    index: int
    path: Path
    luminance: float
    ocr_chars: int

    @property
    def background(self) -> str:
        return "dark" if self.luminance < DARK_LUMINANCE else "light"

    @property
    def text_band(self) -> str:
        if self.ocr_chars <= LOW_TEXT_CHARS:
            return "none"
        if self.ocr_chars <= HIGH_TEXT_CHARS:
            return "low"
        return "high"

    @property
    def bucket(self) -> str:
        return f"{self.background}/{self.text_band}"


def _luminance(path: Path) -> float:
    try:
        with Image.open(path) as image:
            return ImageStat.Stat(image.convert("L")).mean[0]
    except (OSError, ValueError):
        return -1.0


def collect(cache_dir: Path) -> list[FrameSignal]:
    signals: list[FrameSignal] = []
    for ocr_json in cache_dir.glob("*/g*/_ocr.json"):
        try:
            frames = json.loads(ocr_json.read_text()).get("frames", [])
        except (OSError, json.JSONDecodeError):
            continue
        bundle = ocr_json.parent.parent.name[:12]
        for frame in frames:
            path = ocr_json.parent / str(frame.get("relative_path", ""))
            if not path.exists():
                continue
            signals.append(
                FrameSignal(
                    bundle=bundle,
                    index=int(frame.get("index", 0)),
                    path=path,
                    luminance=round(_luminance(path), 1),
                    ocr_chars=len(str(frame.get("ocr_text", "")).strip()),
                )
            )
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "saccade",
        help="Saccade cache root",
    )
    parser.add_argument("--limit", type=int, default=4, help="candidates printed per bucket")
    args = parser.parse_args()

    signals = collect(args.cache)
    print(f"scanned {len(signals)} frames across cache: {args.cache}\n")

    buckets: dict[str, list[FrameSignal]] = {}
    for signal in signals:
        buckets.setdefault(signal.bucket, []).append(signal)

    order = ["dark/high", "dark/low", "light/high", "light/low", "dark/none", "light/none"]
    for bucket in order:
        items = buckets.get(bucket, [])
        if not items:
            continue
        print(f"== {bucket}  ({len(items)} frames) ==")
        for signal in sorted(items, key=lambda item: item.luminance)[: args.limit]:
            print(
                f"  lum={signal.luminance:6.1f}  ocr_chars={signal.ocr_chars:4d}  "
                f"{signal.bundle}  frame_{signal.index:04d}  {signal.path}"
            )
        print()


if __name__ == "__main__":
    main()
