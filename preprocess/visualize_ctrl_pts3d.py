#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize TraceAnything ctrl_pts3d arrays as PNG images.

ctrl_pts3d is not a camera image. It has shape [K, H, W, 3], where each pixel
stores a 3D control-point coordinate. This script turns each [H, W, 3] map into
an RGB visualization by mapping x/y/z -> R/G/B after robust normalization.

Default output is one montage PNG per .npy file, containing all K maps.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def find_ctrl_pts3d_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".npy":
            raise ValueError(f"Expected a .npy ctrl_pts3d file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("ctrl_pts3d/*.npy"))
    raise FileNotFoundError(input_path)


def output_path_for(npy_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    if input_root.is_file():
        rel = Path(npy_path.stem + suffix)
    else:
        rel_parent = npy_path.parent.parent.relative_to(input_root)
        rel = rel_parent / "ctrl_pts3d_images" / (npy_path.stem + suffix)
    return output_root / rel


def robust_min_max(arr: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)

    mins = []
    maxs = []
    for c in range(3):
        values = arr[..., c][finite[..., c]]
        if values.size == 0:
            mins.append(0.0)
            maxs.append(1.0)
            continue
        lo = float(np.percentile(values, low))
        hi = float(np.percentile(values, high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        mins.append(lo)
        maxs.append(hi)

    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def xyz_to_rgb(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3], got {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    mins, maxs = robust_min_max(arr, low=low, high=high)
    rgb = (arr - mins.reshape(1, 1, 3)) / (maxs - mins).reshape(1, 1, 3)
    rgb = np.where(np.isfinite(rgb), rgb, 0.0)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def make_montage(ctrl_pts3d: np.ndarray, low: float, high: float, gap: int) -> Image.Image:
    if ctrl_pts3d.ndim != 4 or ctrl_pts3d.shape[-1] != 3:
        raise ValueError(f"Expected ctrl_pts3d shape [K,H,W,3], got {ctrl_pts3d.shape}")

    k_count, height, width, _ = ctrl_pts3d.shape
    cols = min(5, k_count)
    rows = int(math.ceil(k_count / cols))
    canvas_w = cols * width + (cols - 1) * gap
    canvas_h = rows * height + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for k in range(k_count):
        row = k // cols
        col = k % cols
        x0 = col * (width + gap)
        y0 = row * (height + gap)
        tile = Image.fromarray(xyz_to_rgb(ctrl_pts3d[k], low=low, high=high))
        canvas.paste(tile, (x0, y0))
        draw.rectangle((x0, y0, x0 + 34, y0 + 16), fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + 2), f"k={k}", fill=(255, 255, 255))

    return canvas


def export_montage(
    npy_path: Path,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    low: float,
    high: float,
    gap: int,
) -> bool:
    out_path = output_path_for(npy_path, input_root=input_root, output_root=output_root, suffix=".png")
    if out_path.exists() and not overwrite:
        return False

    ctrl_pts3d = np.load(npy_path)
    image = make_montage(ctrl_pts3d, low=low, high=high, gap=gap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ctrl_pts3d .npy files as PNG montages.")
    parser.add_argument(
        "input",
        type=str,
        help="An extracted root containing ctrl_pts3d/*.npy files, or a single ctrl_pts3d .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output root for ctrl_pts3d visualization PNGs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    parser.add_argument("--low", type=float, default=1.0, help="Low percentile for coordinate normalization.")
    parser.add_argument("--high", type=float, default=99.0, help="High percentile for coordinate normalization.")
    parser.add_argument("--gap", type=int, default=4, help="Pixel gap between montage tiles.")
    parser.add_argument("--limit", type=int, default=0, help="Only visualize the first N files. Default: all files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve()
    files = find_ctrl_pts3d_files(input_path)
    if not files:
        raise FileNotFoundError(f"No ctrl_pts3d/*.npy files found under {input_path}")
    if args.limit > 0:
        files = files[: args.limit]

    written = 0
    for idx, npy_path in enumerate(files, start=1):
        if export_montage(
            npy_path=npy_path,
            input_root=input_path,
            output_root=output_root,
            overwrite=args.overwrite,
            low=args.low,
            high=args.high,
            gap=args.gap,
        ):
            written += 1
        if idx == 1 or idx % 100 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] written={written}", flush=True)

    print(f"[DONE] visualized {written} ctrl_pts3d file(s)")


if __name__ == "__main__":
    main()
