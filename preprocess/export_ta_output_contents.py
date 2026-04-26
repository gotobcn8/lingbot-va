#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export TraceAnything output.pt contents into ordinary files.

For each output.pt this writes:

    ctrl_pts3d/{frame}.npy
    ctrl_conf/{frame}.npy
    fg_mask/{frame}.npy
    masks/{frame}.png
    pred_time/{frame}.txt
    view_img/{frame}.npy
    images/{frame}.png
    times.csv
    metadata.json

The .npy files preserve the original numeric tensors. PNG files are only for
quick inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PRED_ARRAY_KEYS = ("ctrl_pts3d", "ctrl_conf", "fg_mask")
LOCAL_PRED_ARRAY_KEYS = ("ctrl_pts3d_local", "ctrl_conf_local")


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_output_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name != "output.pt":
            raise ValueError(f"Expected an output.pt file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("output.pt"))
    raise FileNotFoundError(input_path)


def output_dir_for(output_pt: Path, input_root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return output_pt.parent / "output_extracted"
    if input_root.is_file():
        return output_root
    rel_parent = output_pt.parent.relative_to(input_root)
    return output_root / rel_parent


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def scalar_to_float(value: Any) -> float:
    arr = tensor_to_numpy(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, value: Any, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    ensure_dir(path.parent)
    np.save(path, tensor_to_numpy(value))
    return True


def image_to_uint8(img: Any) -> np.ndarray:
    arr = tensor_to_numpy(img)

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected image batch size 1, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError(f"Expected image shape [1,C,H,W], [C,H,W], [H,W,C], or [H,W], got {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 image channels, got shape {arr.shape}")

    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32, copy=False)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if min_val >= -0.05 and max_val <= 1.05:
        arr = arr * 255.0
    else:
        arr = (arr + 1.0) * 127.5

    return np.clip(arr, 0, 255).astype(np.uint8)


def mask_to_uint8(mask: Any) -> np.ndarray:
    arr = tensor_to_numpy(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected mask shape [H,W], got {arr.shape}")
    if arr.dtype == np.bool_:
        mask_bool = arr
    else:
        mask_bool = arr.astype(np.float32) > 0.5
    return (mask_bool.astype(np.uint8) * 255)


def save_png(path: Path, arr: np.ndarray, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)
    return True


def save_text(path: Path, text: str, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return True


def frame_stem(index: int, frame_ids: list[int] | None) -> str:
    stem = f"{index:03d}"
    if frame_ids is not None and index < len(frame_ids):
        stem += f"_frame_{int(frame_ids[index]):06d}"
    return stem


def describe_value(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__}


def json_ready(value: Any) -> Any:
    if torch.is_tensor(value):
        arr = tensor_to_numpy(value)
        if arr.size == 1:
            return scalar_to_float(value)
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return str(value)


def validate_payload(payload: dict[str, Any], output_pt: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "preds" not in payload:
        raise KeyError(f"{output_pt} does not contain key 'preds'")
    if "views" not in payload:
        raise KeyError(f"{output_pt} does not contain key 'views'")

    preds = payload["preds"]
    views = payload["views"]
    if not isinstance(preds, (list, tuple)):
        raise TypeError(f"Expected preds to be a list/tuple, got {type(preds)}")
    if not isinstance(views, (list, tuple)):
        raise TypeError(f"Expected views to be a list/tuple, got {type(views)}")
    if len(preds) != len(views):
        raise ValueError(f"{output_pt}: len(preds)={len(preds)} but len(views)={len(views)}")
    return list(preds), list(views)


def write_metadata(
    output_pt: Path,
    out_dir: Path,
    payload: dict[str, Any],
    preds: list[dict[str, Any]],
    views: list[dict[str, Any]],
    pred_array_keys: list[str],
    overwrite: bool,
) -> bool:
    pred0 = preds[0] if preds else {}
    view0 = views[0] if views else {}
    metadata = {
        "source_output_pt": str(output_pt),
        "num_frames": len(preds),
        "payload_keys": sorted(str(k) for k in payload.keys()),
        "exported_pred_array_keys": pred_array_keys,
        "exported_view_keys": ["img"],
        "frame_ids": json_ready(payload.get("frame_ids")),
        "video_path": json_ready(payload.get("video_path")),
        "view": json_ready(payload.get("view")),
        "first_pred": {str(k): describe_value(v) for k, v in pred0.items()},
        "first_view": {str(k): describe_value(v) for k, v in view0.items()},
    }
    return save_text(
        out_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        overwrite=overwrite,
    )


def export_one(
    output_pt: Path,
    input_root: Path,
    output_root: Path | None,
    overwrite: bool,
    include_local: bool,
) -> dict[str, int]:
    payload = torch_load_cpu(output_pt)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected {output_pt} to contain a dict, got {type(payload)}")

    preds, views = validate_payload(payload, output_pt)
    out_dir = output_dir_for(output_pt, input_root=input_root, output_root=output_root)
    ensure_dir(out_dir)

    frame_ids_value = payload.get("frame_ids")
    frame_ids = [int(x) for x in frame_ids_value] if frame_ids_value is not None else None

    pred_array_keys = list(PRED_ARRAY_KEYS)
    if include_local:
        pred_array_keys.extend(LOCAL_PRED_ARRAY_KEYS)

    counts = {
        "ctrl_pts3d": 0,
        "ctrl_conf": 0,
        "fg_mask_npy": 0,
        "fg_mask_png": 0,
        "pred_time": 0,
        "view_img_npy": 0,
        "view_img_png": 0,
        "metadata": 0,
    }

    times_rows: list[dict[str, Any]] = []
    for i, (pred, view) in enumerate(zip(preds, views)):
        stem = frame_stem(i, frame_ids)

        for key in pred_array_keys:
            if key not in pred:
                continue
            if save_npy(out_dir / key / f"{stem}.npy", pred[key], overwrite=overwrite):
                count_key = f"{key}_npy" if key == "fg_mask" else key
                counts[count_key] = counts.get(count_key, 0) + 1

        if "fg_mask" in pred:
            if save_png(out_dir / "masks" / f"{stem}.png", mask_to_uint8(pred["fg_mask"]), overwrite=overwrite):
                counts["fg_mask_png"] += 1

        pred_time = None
        if "time" in pred and pred["time"] is not None:
            pred_time = scalar_to_float(pred["time"])
            if save_text(out_dir / "pred_time" / f"{stem}.txt", f"{pred_time:.10f}\n", overwrite=overwrite):
                counts["pred_time"] += 1

        time_step = view.get("time_step")
        if "img" not in view:
            raise KeyError(f"{output_pt}: views[{i}] does not contain key 'img'")
        if save_npy(out_dir / "view_img" / f"{stem}.npy", view["img"], overwrite=overwrite):
            counts["view_img_npy"] += 1
        if save_png(out_dir / "images" / f"{stem}.png", image_to_uint8(view["img"]), overwrite=overwrite):
            counts["view_img_png"] += 1

        times_rows.append(
            {
                "index": i,
                "frame_id": "" if frame_ids is None or i >= len(frame_ids) else int(frame_ids[i]),
                "pred_time": "" if pred_time is None else f"{pred_time:.10f}",
                "view_time_step": "" if time_step is None else f"{float(time_step):.10f}",
            }
        )

    times_csv = out_dir / "times.csv"
    if overwrite or not times_csv.exists():
        with times_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "frame_id", "pred_time", "view_time_step"])
            writer.writeheader()
            writer.writerows(times_rows)

    if write_metadata(output_pt, out_dir, payload, preds, views, pred_array_keys, overwrite=overwrite):
        counts["metadata"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TraceAnything output.pt contents into folders."
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
        help="Output root. Default: write output_extracted beside each output.pt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist.",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also export ctrl_pts3d_local and ctrl_conf_local when present.",
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
    for idx, output_pt in enumerate(output_files, start=1):
        counts = export_one(
            output_pt=output_pt,
            input_root=input_path,
            output_root=output_root,
            overwrite=args.overwrite,
            include_local=args.include_local,
        )
        total += 1
        counts_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"[{idx}/{len(output_files)}] {output_pt} -> {counts_text}", flush=True)

    print(f"[DONE] exported {total} output.pt file(s)")


if __name__ == "__main__":
    main()
