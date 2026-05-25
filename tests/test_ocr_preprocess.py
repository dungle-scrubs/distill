from __future__ import annotations

from pathlib import Path

import pytest


def test_preprocess_inverts_dark_background_for_ocr(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image, ImageStat

    from saccade.ocr import preprocess_for_ocr

    # Dark background (20) with a brighter text-like band (200).
    image = Image.new("L", (40, 20), color=20)
    for y in range(8, 12):
        for x in range(4, 36):
            image.putpixel((x, y), 200)
    path = tmp_path / "dark.png"
    image.save(path)

    processed = preprocess_for_ocr(path)

    assert processed is not None
    assert processed.mode == "L"
    # Upscaled 2x and inverted so the background reads light for Tesseract.
    assert processed.size == (80, 40)
    assert ImageStat.Stat(processed).mean[0] > 128
