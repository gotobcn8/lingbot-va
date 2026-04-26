#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export input images stored inside TraceAnything output.pt files.

The trajectory extraction script stores normalized TA input frames in:

    output.pt["views"][i]["img"]

This helper reads one output.pt, or recursively scans a directory for many
output.pt files, and writes those frames as PNGs for quick inspection.
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def find_output_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name != "output.pt":
            raise ValueError(f"Expected an output.pt file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("output.pt"))
    raise FileNotFoundError(input_path)


def tensor_to_uint8_image(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        arr = img
    elif torch.is_tensor(img):
        arr = img.detach().cpu().float().numpy()
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for image tensor, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim != 3:
        raise ValueError(f"Expected image with shape [1,3,H,W], [3,H,W], or [H,W,3], got {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {arr.shape}")

    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)

    min_val = float(arr.min())
    max_val = float(arr.max())
    if min_val >= -0.05 and max_val <= 1.05:
        arr = arr * 255.0
    else:
        arr = (arr + 1.0) * 127.5

    return np.clip(arr, 0, 255).astype(np.uint8)


def output_dir_for(output_pt: Path, input_root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return output_pt.parent / "images_from_output"

    if input_root.is_file():
        return output_root

    rel_parent = output_pt.parent.relative_to(input_root)
    return output_root / rel_parent / "images_from_output"


def iter_views(payload: dict) -> Iterable[dict]:
    views = payload.get("views")
    if views is None:
        raise KeyError("output.pt does not contain key 'views'")
    if not isinstance(views, (list, tuple)):
        raise TypeError(f"Expected payload['views'] to be a list/tuple, got {type(views)}")
    return views


def export_images(output_pt: Path, input_root: Path, output_root: Path | None, overwrite: bool) -> int:
    payload = torch.load(output_pt, map_location="cpu", weights_only=False)
    out_dir = output_dir_for(output_pt, input_root=input_root, output_root=output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = payload.get("frame_ids")
    views = iter_views(payload)
    count = 0
    for i, view in enumerate(views):
        if "img" not in view:
            raise KeyError(f"views[{i}] does not contain key 'img' in {output_pt}")

        frame_suffix = ""
        if frame_ids is not None and i < len(frame_ids):
            frame_suffix = f"_frame_{int(frame_ids[i]):06d}"
        out_path = out_dir / f"{i:03d}{frame_suffix}.png"
        if out_path.exists() and not overwrite:
            continue

        img_uint8 = tensor_to_uint8_image(view["img"])
        Image.fromarray(img_uint8).save(out_path)
        count += 1

    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export PNG images from TraceAnything output.pt payloads."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to one output.pt file or a directory recursively containing output.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory. For directory input, relative output.pt parents are preserved. "
            "Default: write images_from_output beside each output.pt."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve() if args.output_dir else None
    output_files = find_output_files(input_path)
    if not output_files:
        raise FileNotFoundError(f"No output.pt files found under {input_path}")

    total = 0
    for output_pt in output_files:
        count = export_images(
            output_pt=output_pt,
            input_root=input_path,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        total += count
        print(f"[OK] {output_pt}: exported {count} image(s)")

    print(f"[DONE] exported {total} image(s) from {len(output_files)} output.pt file(s)")


if __name__ == "__main__":
    main()
