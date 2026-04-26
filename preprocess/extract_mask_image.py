#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build episode-level mask+image montages from TraceAnything trajectory outputs.

For each episode, this script reads:

    <chunk>/<view>/<episode>/images/{frame}.png
    <chunk>/<view>/<episode>/masks/{frame}.png

and writes one PNG:

    <chunk>/mask_image/<episode>.png

The output layout is one row per camera view, with frames placed left to right.
Masks are overlaid on images using a light-blue transparent color.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import os

DEFAULT_EPISODE_DIR = Path(
    "/sharedata/lsy/robotwin/open_microwave-demo_clean_collect_200-50/"
    "trajectories/chunk-000/observation.images.cam_high/episode_000000_0_455"
)

DEFAULT_VIEWS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

INPUT_DIRS = (
    # 'hanging_mug-demo_clean_collect_200-50',
    # 'turn_switch-demo_clean_collect_200-50',
    'move_stapler_pad-demo_clean_collect_200-50',
    'put_bottles_dustbin-piper_clean_50-50'
)

BASE_DIR = '/sharedata/lsy/robotwin'

# DEFAULT_MASK_COLOR = (135, 206, 250)
DEFAULT_MASK_COLOR = (255, 140, 105)

def parse_rgb(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB like '135,206,250', got: {text}")
    if any(part < 0 or part > 255 for part in parts):
        raise ValueError(f"RGB values should be in [0,255], got: {text}")
    return tuple(parts)


def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    mask_color: tuple[int, int, int] = DEFAULT_MASK_COLOR,
    mask_alpha: float = 0.38,
) -> np.ndarray:
    if mask.shape != image_rgb.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST)
        mask = np.asarray(mask_img) > 0

    out = image_rgb.astype(np.float32)
    color = np.asarray(mask_color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = out[mask] * (1.0 - mask_alpha) + color * mask_alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_cell(image: Image.Image, cell_width: int, cell_height: int) -> Image.Image:
    return image.convert("RGB").resize((cell_width, cell_height), Image.BILINEAR)


def make_mask_image_montage(
    episode_dir: Path,
    views: tuple[str, ...],
    output_dir: Path | None = None,
    out_file: Path | None = None,
    max_frames: int = 40,
    cell_width: int = 220,
    cell_height: int = 176,
    gap: int = 2,
    label_width: int = 150,
    mask_color: tuple[int, int, int] = DEFAULT_MASK_COLOR,
    mask_alpha: float = 0.38,
) -> Path:
    episode_dir = Path(episode_dir).resolve()
    chunk_dir = episode_dir.parent.parent
    episode_name = episode_dir.name

    rows: list[tuple[str, list[Image.Image]]] = []
    for view in views:
        view_dir = chunk_dir / view / episode_name
        images_dir = view_dir / "images"
        masks_dir = view_dir / "masks"
        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)
        if not masks_dir.is_dir():
            raise FileNotFoundError(masks_dir)

        image_paths = sorted(images_dir.glob("*.png"))
        mask_paths = sorted(masks_dir.glob("*.png"))
        frame_count = min(max_frames, len(image_paths), len(mask_paths))

        cells = []
        for frame_id in range(frame_count):
            image_path = images_dir / f"{frame_id:03d}.png"
            mask_path = masks_dir / f"{frame_id:03d}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            overlaid = overlay_mask_on_image(
                image_rgb=image_rgb,
                mask=mask,
                mask_color=mask_color,
                mask_alpha=mask_alpha,
            )
            cell = resize_cell(Image.fromarray(overlaid), cell_width, cell_height)
            draw = ImageDraw.Draw(cell)
            draw.rectangle((0, 0, 42, 18), fill=(255, 255, 255))
            draw.text((4, 2), f"{frame_id:02d}", fill=(0, 0, 0))
            cells.append(cell)

        rows.append((view, cells))

    if not rows or not any(cells for _, cells in rows):
        raise RuntimeError(f"No image/mask frames found for episode: {episode_dir}")

    ncols = max(len(cells) for _, cells in rows)
    width = label_width + ncols * cell_width + (ncols - 1) * gap
    height = len(rows) * cell_height + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row_id, (view, cells) in enumerate(rows):
        y = row_id * (cell_height + gap)
        draw.rectangle((0, y, label_width - 1, y + cell_height), fill=(245, 245, 245))
        short_view = view.replace("observation.images.", "")
        draw.text((10, y + 12), short_view, fill=(0, 0, 0))
        draw.text((10, y + 34), episode_name, fill=(70, 70, 70))
        for col_id, cell in enumerate(cells):
            x = label_width + col_id * (cell_width + gap)
            canvas.paste(cell, (x, y))

    if out_file is None:
        if output_dir is None:
            output_dir = chunk_dir / "mask_image"
        out_file = Path(output_dir) / f"{episode_name}.png"

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)
    print(f"Saved mask+image montage to: {out_file}")
    return out_file


def discover_episode_dirs(anchor_episode_dir: Path, first_view: str) -> list[Path]:
    anchor_episode_dir = Path(anchor_episode_dir).resolve()
    chunk_dir = anchor_episode_dir
    first_view_dir = chunk_dir / first_view
    if not first_view_dir.is_dir():
        raise FileNotFoundError(first_view_dir)
    return sorted(path for path in first_view_dir.iterdir() if path.is_dir())


def make_all_montages(
    anchor_episode_dir: Path,
    views: tuple[str, ...],
    output_dir: Path | None,
    skip_existing: bool,
    limit_episodes: int,
    **kwargs,
) -> None:
    chunk_dir = anchor_episode_dir / 'chunk-000'
    print(chunk_dir)
    episode_dirs = discover_episode_dirs(chunk_dir, views[0])
    if limit_episodes > 0:
        episode_dirs = episode_dirs[:limit_episodes]

    # chunk_dir = Path(anchor_episode_dir).resolve().parent.parent
    if output_dir is None:
        output_dir = chunk_dir / "mask_image"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for idx, episode_dir in enumerate(episode_dirs, start=1):
        out_file = output_dir / f"{episode_dir.name}.png"
        if skip_existing and out_file.exists():
            skipped += 1
            print(f"[{idx}/{len(episode_dirs)}] skip existing: {out_file}")
            continue

        print(f"[{idx}/{len(episode_dirs)}] rendering: {episode_dir.name}")
        make_mask_image_montage(
            episode_dir=episode_dir,
            views=views,
            output_dir=output_dir,
            out_file=out_file,
            **kwargs,
        )
        written += 1

    print(f"Done. written={written}, skipped={skipped}, output_dir={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create episode-level mask+image montages with light-blue mask overlays."
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default=str(DEFAULT_EPISODE_DIR),
        help="Path to one view's episode directory. Sibling view dirs are found from this chunk.",
    )
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default=list(INPUT_DIRS)
    )
    parser.add_argument("--views", type=str, nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--output-dir", type=str, default=None, help="Default: <chunk>/mask_image.")
    parser.add_argument("--out-file", type=str, default=None, help="Only for single-episode mode.")
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--cell-width", type=int, default=220)
    parser.add_argument("--cell-height", type=int, default=176)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--label-width", type=int, default=150)
    parser.add_argument("--mask-color", type=parse_rgb, default=DEFAULT_MASK_COLOR)
    parser.add_argument("--mask-alpha", type=float, default=0.38)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    views = tuple(args.views)
    kwargs = dict(
        max_frames=args.max_frames,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        gap=args.gap,
        label_width=args.label_width,
        mask_color=args.mask_color,
        mask_alpha=args.mask_alpha,
    )

    if args.all_episodes:
        for dir in args.input_dir:
            episode_dir = Path(BASE_DIR, dir, 'trajectories')
            make_all_montages(
                anchor_episode_dir=episode_dir,
                views=views,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                skip_existing=not args.overwrite,
                limit_episodes=args.limit_episodes,
                **kwargs,
            )
    else:
        
        make_mask_image_montage(
            episode_dir=Path(args.episode_dir),
            views=views,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            out_file=Path(args.out_file) if args.out_file else None,
            **kwargs,
        )


if __name__ == "__main__":
    main()
