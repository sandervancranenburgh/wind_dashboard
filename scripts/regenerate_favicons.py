#!/usr/bin/env python3
"""Remove the source icon's exterior white matte and rebuild favicon sizes."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


ICON_SIZES = {
    "favicon-16x16.png": 16,
    "favicon-32x32.png": 32,
    "apple-touch-icon.png": 180,
    "icon-192x192.png": 192,
    "icon-512x512.png": 512,
}


def remove_exterior_white_matte(source: Image.Image) -> Image.Image:
    """Recover alpha only for near-white pixels connected to the image border."""
    source_rgba = np.asarray(source.convert("RGBA"), dtype=np.float32)
    rgb = source_rgba[:, :, :3]
    source_alpha = source_rgba[:, :, 3] / 255.0
    opacity_hint = np.max(255.0 - rgb, axis=2) / 255.0
    # A deliberately conservative dark core closes the rounded-square outline.
    # Flooding only its complement protects every enclosed white/cyan detail.
    dark_core = opacity_hint >= (100.0 / 255.0)
    passable = ~dark_core
    height, width = passable.shape
    exterior = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        queue.append((0, x))
        queue.append((height - 1, x))
    for y in range(1, height - 1):
        queue.append((y, 0))
        queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        if exterior[y, x] or not passable[y, x]:
            continue
        exterior[y, x] = True
        if y > 0:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x > 0:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))

    alpha = source_alpha.copy()
    # Only recover alpha from an opaque matte. Existing transparent and
    # antialiased pixels have already been cleaned and must survive reruns.
    matte_pixels = exterior & (source_alpha >= (254.5 / 255.0))
    alpha[matte_pixels] = opacity_hint[matte_pixels]
    cleaned_rgb = rgb.copy()
    nonzero = matte_pixels & (alpha > 0)
    for channel in range(3):
        values = cleaned_rgb[:, :, channel]
        values[nonzero] = (
            values[nonzero] - 255.0 * (1.0 - alpha[nonzero])
        ) / alpha[nonzero]
        cleaned_rgb[:, :, channel] = np.clip(values, 0.0, 255.0)
    cleaned_rgb[matte_pixels & (alpha == 0)] = 0.0

    rgba = np.dstack(
        [
            np.rint(cleaned_rgb).astype(np.uint8),
            np.rint(alpha * 255.0).astype(np.uint8),
        ]
    )
    return Image.fromarray(rgba, mode="RGBA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("next_day_wind_model/web_assets/icons/wingfoil-icon-master.png"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("next_day_wind_model/web_assets/icons"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = remove_exterior_white_matte(Image.open(args.source))
    cleaned.save(args.output_dir / "wingfoil-icon-master.png", optimize=True)

    rendered: dict[int, Image.Image] = {}
    for filename, size in ICON_SIZES.items():
        icon = cleaned.resize((size, size), Image.Resampling.LANCZOS)
        icon.save(args.output_dir / filename, optimize=True)
        rendered[size] = icon

    rendered[48] = cleaned.resize((48, 48), Image.Resampling.LANCZOS)
    rendered[48].save(
        args.output_dir / "favicon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48)],
    )


if __name__ == "__main__":
    main()
