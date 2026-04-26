#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract TraceAnything trajectory-field outputs for LeRobot-style datasets.

This script follows the dataset traversal used by extract_latents_from_ta.py:
it reads sampled frame_ids from latents/chunk-XXX/<view>/episode_XXXXXX.pth,
loads the matching videos/chunk-XXX/<view>/episode_XXXXXX.mp4 frames, runs
TraceAnything.forward(), and saves outputs in the same spirit as
TraceAnything/scripts/infer.py:

    trajectories/
      chunk-000/
        observation.images.cam_high/
          episode_000000/
            output.pt
            masks/000.png
            images/000.png

output.pt contains {"preds", "views", "frame_ids", "video_path", "view"}.
"""

import argparse
import multiprocessing as mp
import subprocess
import traceback
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import resize
from tqdm import tqdm

from traceanything import TraceAnything


IMAGE_KEYS = {
    "observation.images.cam_high": (256, 320),
    "observation.images.cam_left_wrist": (128, 160),
    "observation.images.cam_right_wrist": (128, 160),
}

DEFAULT_VIEWS = tuple(IMAGE_KEYS.keys())


def _pretty(msg: str) -> None:
    print(msg, flush=True)


def _get_state_dict(ckpt: dict) -> dict:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def _to_dict(x):
    return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x


def _load_cfg(cfg_path: Path):
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    return OmegaConf.load(cfg_path)


def _build_model_from_cfg(cfg, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    net_cfg = cfg.get("model", {}).get("net", None) or cfg.get("net", None)
    if net_cfg is None:
        raise KeyError("expect cfg.model.net or cfg.net in YAML")

    model = TraceAnything(
        encoder_args=_to_dict(net_cfg["encoder_args"]),
        decoder_args=_to_dict(net_cfg["decoder_args"]),
        head_args=_to_dict(net_cfg["head_args"]),
        targeting_mechanism=net_cfg.get("targeting_mechanism", "bspline_conf"),
        poly_degree=net_cfg.get("poly_degree", 10),
        whether_local=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _get_state_dict(ckpt)
    if all(k.startswith("net.") for k in sd.keys()):
        sd = {k[4:]: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def discover_task_roots(dataset_root: Path) -> List[Path]:
    """Support either one task directory or a parent containing task directories."""
    dataset_root = dataset_root.resolve()
    single_task_markers = ("latents", "videos", "meta")
    if all((dataset_root / marker).exists() for marker in single_task_markers):
        return [dataset_root]

    task_roots = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and all((child / marker).exists() for marker in single_task_markers):
            task_roots.append(child)
    if not task_roots:
        raise FileNotFoundError(
            f"No task directories found under {dataset_root}. "
            "Expected either a task root with latents/videos/meta or a parent dir containing task subdirectories."
        )
    return task_roots


def parse_devices(device: str = "", devices: str = "") -> List[str]:
    if devices.strip():
        return [item.strip() for item in devices.split(",") if item.strip()]
    if device.strip():
        return [device.strip()]
    return ["cuda:0"]


def iter_episode_latent_files(view_path: Path) -> Iterable[Path]:
    return sorted(path for path in view_path.iterdir() if path.is_file() and path.suffix == ".pth")


def get_video_info(video_path: Path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get video info for {video_path}: {result.stderr}")
    width, height = map(int, result.stdout.strip().split(","))
    return width, height


def read_video_frames_ffmpeg_single(video_path: Path, frame_ids: Sequence[int], video_size=None):
    if video_size is None:
        video_size = get_video_info(video_path)

    width, height = video_size
    frame_size = width * height * 3
    frames = []

    for fid in frame_ids:
        cmd = [
            "ffmpeg",
            "-hwaccel",
            "none",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{int(fid)})",
            "-vsync",
            "vfr",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_data = proc.stdout.read(frame_size)
        proc.wait()
        if proc.returncode != 0 or len(raw_data) != frame_size:
            stderr = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"Failed to read frame {fid} from {video_path}. FFmpeg error: {stderr}")
        frames.append(np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3)))

    return frames


def read_video_frames_ffmpeg_batch(video_path: Path, frame_ids: Sequence[int]):
    width, height = get_video_info(video_path)
    frame_size = width * height * 3
    sorted_ids = sorted(int(fid) for fid in frame_ids)
    min_frame = sorted_ids[0]
    max_frame = sorted_ids[-1]

    cmd = [
        "ffmpeg",
        "-hwaccel",
        "none",
        "-i",
        str(video_path),
        "-vf",
        f"select=between(n\\,{min_frame}\\,{max_frame}),setpts=PTS-STARTPTS",
        "-vsync",
        "vfr",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_size * 10)

    all_frames = []
    for _ in range(max_frame - min_frame + 1):
        raw_data = proc.stdout.read(frame_size)
        if len(raw_data) != frame_size:
            break
        all_frames.append(np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3)))

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"FFmpeg error for {video_path}: {stderr}")

    frame_map = {min_frame + i: frame for i, frame in enumerate(all_frames)}
    missing = [fid for fid in frame_ids if int(fid) not in frame_map]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested frames from {video_path}: {missing[:5]}")
    return [frame_map[int(fid)] for fid in frame_ids]


def read_video_frames_ffmpeg(video_path: Path, frame_ids: Sequence[int]):
    if not frame_ids:
        return []
    if len(frame_ids) > 10:
        return read_video_frames_ffmpeg_batch(video_path, frame_ids)
    return read_video_frames_ffmpeg_single(video_path, frame_ids)


def frames_rgb_to_tensor(frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
    tensors = []
    for frame in frames_rgb:
        x = torch.from_numpy(frame).float() / 255.0
        tensors.append(x.permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def build_frame_timesteps(num_frames: int) -> List[float]:
    if num_frames <= 1:
        return [0.0]
    return [i / (num_frames - 1) for i in range(num_frames)]


def maybe_downsample_ids_and_times(frame_ids: Sequence[int], max_views: int):
    frame_ids = [int(fid) for fid in frame_ids]
    timesteps = build_frame_timesteps(len(frame_ids))
    if max_views <= 0 or len(frame_ids) <= max_views:
        return frame_ids, timesteps
    if max_views == 1:
        return frame_ids[:1], timesteps[:1]
    stride = max(1, len(frame_ids) // (max_views - 1))
    keep_indices = list(range(0, len(frame_ids), stride))[:max_views]
    return [frame_ids[i] for i in keep_indices], [timesteps[i] for i in keep_indices]


def make_views(
    frames_rgb: Sequence[np.ndarray],
    view_name: str,
    device: torch.device,
    timesteps: Sequence[float],
):
    frames = frames_rgb_to_tensor(frames_rgb)
    target_size = IMAGE_KEYS.get(view_name)
    if target_size is not None:
        frames = resize(frames, target_size, antialias=True)

    frames = frames.mul(2.0).sub(1.0)
    if len(timesteps) != frames.shape[0]:
        raise ValueError(f"Expected {frames.shape[0]} timesteps, got {len(timesteps)}")
    return [
        {"img": frames[i : i + 1].to(device), "time_step": timesteps[i]}
        for i in range(frames.shape[0])
    ]


def tensor_to_image_uint8(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().squeeze(0)
    img_np = (img.permute(1, 2, 0).numpy() + 1.0) * 127.5
    return np.clip(img_np, 0, 255).astype(np.uint8)


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def _otsu_threshold_from_hist(hist: np.ndarray, bin_edges: np.ndarray) -> float | None:
    total = hist.sum()
    if total <= 0:
        return None
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w1 = np.cumsum(hist)
    w2 = total - w1
    sum_total = (hist * bin_centers).sum()
    sum_b = np.cumsum(hist * bin_centers)
    valid = (w1 > 0) & (w2 > 0)
    if not np.any(valid):
        return None
    m1 = sum_b[valid] / w1[valid]
    m2 = (sum_total - sum_b[valid]) / w2[valid]
    between = w1[valid] * w2[valid] * (m1 - m2) ** 2
    idx = np.argmax(between)
    return float(bin_centers[valid][idx])


def _smart_var_threshold(var_map_t: torch.Tensor) -> float:
    var_np = var_map_t.detach().float().cpu().numpy()
    v = np.log(var_np + 1e-9)
    hist, bin_edges = np.histogram(v, bins=256)
    thr_log = _otsu_threshold_from_hist(hist, bin_edges)
    if thr_log is None or not np.isfinite(thr_log):
        q65 = float(np.quantile(var_np, 0.65))
        q80 = float(np.quantile(var_np, 0.80))
        return 0.5 * (q65 + q80)
    thr_var = float(np.exp(thr_log))
    q40 = float(np.quantile(var_np, 0.40))
    q95 = float(np.quantile(var_np, 0.95))
    return max(q40, min(q95, thr_var))


def trim_evaluated_tracks(pred: dict) -> None:
    pred.pop("track_pts3d", None)
    pred.pop("track_conf", None)
    pred.pop("track_pts3d_local", None)
    pred.pop("track_conf_local", None)


def save_episode_outputs(
    out_dir: Path,
    preds: list,
    views: list,
    frame_ids: Sequence[int],
    video_path: Path,
    view_name: str,
    keep_evaluated_tracks: bool,
) -> None:
    masks_dir = out_dir / "masks"
    images_dir = out_dir / "images"
    masks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, pred in enumerate(preds):
        ctrl_pts3d = pred["ctrl_pts3d"]
        ctrl_pts3d_t = torch.from_numpy(ctrl_pts3d) if isinstance(ctrl_pts3d, np.ndarray) else ctrl_pts3d
        var_map = torch.var(ctrl_pts3d_t, dim=0, unbiased=False).mean(-1)
        thr = _smart_var_threshold(var_map)
        fg_mask = (~(var_map <= thr)).detach().cpu().numpy().astype(bool)
        Image.fromarray(fg_mask.astype(np.uint8) * 255).save(masks_dir / f"{i:03d}.png")
        pred["fg_mask"] = torch.from_numpy(fg_mask)

        img_uint8 = tensor_to_image_uint8(views[i]["img"])
        Image.fromarray(img_uint8).save(images_dir / f"{i:03d}.png")

        if not keep_evaluated_tracks:
            trim_evaluated_tracks(pred)

    payload = {
        "preds": to_cpu(preds),
        "views": to_cpu(views),
        "frame_ids": [int(fid) for fid in frame_ids],
        "video_path": str(video_path),
        "view": view_name,
    }
    torch.save(payload, out_dir / "output.pt")


def process_task(
    task_root: Path,
    ta_model: torch.nn.Module,
    device: torch.device,
    views_to_process: Sequence[str],
    output_root_name: str,
    skip_existing: bool,
    max_views: int,
    keep_evaluated_tracks: bool,
) -> None:
    video_root = task_root / "videos"
    latent_root = task_root / "latents"
    trajectory_root = task_root / output_root_name

    if not latent_root.exists():
        _pretty(f"[WARN] missing latents dir, skip task: {task_root}")
        return
    if not video_root.exists():
        _pretty(f"[WARN] missing videos dir, skip task: {task_root}")
        return

    chunks = sorted(path for path in latent_root.iterdir() if path.is_dir())
    for chunk_dir in tqdm(chunks, desc=f"{task_root.name}: chunks", leave=False):
        chunk_name = chunk_dir.name
        video_chunk_dir = video_root / chunk_name

        for view_name in views_to_process:
            view_path = chunk_dir / view_name
            if not view_path.exists():
                continue

            for eps_path in tqdm(
                iter_episode_latent_files(view_path),
                desc=f"{task_root.name}/{chunk_name}/{view_name}",
                leave=False,
            ):
                episode_stem = eps_path.stem
                out_dir = trajectory_root / chunk_name / view_name / episode_stem
                out_path = out_dir / "output.pt"
                if skip_existing and out_path.exists():
                    continue

                video_path = video_chunk_dir / view_name / f"{eps_path.name[:14]}.mp4"
                try:
                    content = torch.load(eps_path, map_location="cpu", weights_only=False)
                    frame_ids, timesteps = maybe_downsample_ids_and_times(
                        content["frame_ids"],
                        max_views=max_views,
                    )
                    frames_rgb = read_video_frames_ffmpeg(video_path, frame_ids)
                    if len(frames_rgb) != len(frame_ids):
                        raise RuntimeError(
                            f"Expected {len(frame_ids)} frames, got {len(frames_rgb)} from {video_path}"
                        )

                    views = make_views(
                        frames_rgb,
                        view_name=view_name,
                        device=device,
                        timesteps=timesteps,
                    )
                    with torch.no_grad():
                        preds = ta_model.forward(views)

                    save_episode_outputs(
                        out_dir=out_dir,
                        preds=preds,
                        views=views,
                        frame_ids=frame_ids,
                        video_path=video_path,
                        view_name=view_name,
                        keep_evaluated_tracks=keep_evaluated_tracks,
                    )
                except Exception as exc:
                    _pretty(f"[WARN] failed on {eps_path}: {exc}")

    _pretty(f"[OK] finished task: {task_root}")


def run_worker(
    worker_id: int,
    task_roots: Sequence[str],
    models_root: str,
    cfg_path: str,
    device: str,
    views_to_process: Sequence[str],
    output_root_name: str,
    skip_existing: bool,
    max_views: int,
    keep_evaluated_tracks: bool,
) -> None:
    if not task_roots:
        _pretty(f"[INFO] worker {worker_id} on {device} has no tasks assigned.")
        return

    try:
        _pretty(
            f"[INFO] worker {worker_id} starting on {device} "
            f"for task(s): {[Path(task_root).name for task_root in task_roots]}"
        )
        torch.set_grad_enabled(False)
        if device.startswith("cuda"):
            cuda_index = torch.device(device).index
            if cuda_index is not None:
                torch.cuda.set_device(cuda_index)

        device_obj = torch.device(device)
        cfg = _load_cfg(Path(cfg_path))
        model_ckpt = Path(models_root) / "trace_anything.pt"
        ta_model = _build_model_from_cfg(cfg, model_ckpt, device_obj)

        for task_root in task_roots:
            process_task(
                task_root=Path(task_root),
                ta_model=ta_model,
                device=device_obj,
                views_to_process=views_to_process,
                output_root_name=output_root_name,
                skip_existing=skip_existing,
                max_views=max_views,
                keep_evaluated_tracks=keep_evaluated_tracks,
            )
        _pretty(f"[INFO] worker {worker_id} finished on {device}")
    except Exception:
        _pretty(f"[ERROR] worker {worker_id} failed on {device}")
        _pretty(traceback.format_exc())
        raise


def launch_parallel(
    task_roots: Sequence[Path],
    models_root: Path,
    cfg_path: Path,
    devices: Sequence[str],
    views_to_process: Sequence[str],
    output_root_name: str,
    skip_existing: bool,
    max_views: int,
    keep_evaluated_tracks: bool,
    max_concurrent_tasks: int,
) -> None:
    if not task_roots:
        return

    max_workers = min(len(devices), max(1, max_concurrent_tasks))
    if max_workers == 1:
        run_worker(
            worker_id=0,
            task_roots=[str(path) for path in task_roots],
            models_root=str(models_root),
            cfg_path=str(cfg_path),
            device=devices[0],
            views_to_process=views_to_process,
            output_root_name=output_root_name,
            skip_existing=skip_existing,
            max_views=max_views,
            keep_evaluated_tracks=keep_evaluated_tracks,
        )
        return

    ctx = mp.get_context("spawn")
    for wave_start in range(0, len(task_roots), max_workers):
        wave_tasks = task_roots[wave_start : wave_start + max_workers]
        wave_index = wave_start // max_workers + 1
        total_waves = (len(task_roots) + max_workers - 1) // max_workers
        _pretty(
            f"[INFO] starting wave {wave_index}/{total_waves} with "
            f"{len(wave_tasks)} task(s): {[task.name for task in wave_tasks]}"
        )

        processes = []
        for worker_id, task_root in enumerate(wave_tasks):
            device = devices[worker_id % len(devices)]
            proc = ctx.Process(
                target=run_worker,
                args=(
                    worker_id,
                    [str(task_root)],
                    str(models_root),
                    str(cfg_path),
                    device,
                    views_to_process,
                    output_root_name,
                    skip_existing,
                    max_views,
                    keep_evaluated_tracks,
                ),
            )
            proc.start()
            processes.append(proc)

        failed = False
        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                failed = True
                _pretty(f"[ERROR] worker pid={proc.pid} exited with code {proc.exitcode}")
        if failed:
            raise RuntimeError(f"At least one worker failed during wave {wave_index}/{total_waves}.")


def parse_args():
    default_dataset_root = "/sharedata/lsy/robotwin/"
    task_name = 'hanging_mug-demo_clean_collect_200-50 move_stapler_pad-demo_clean_collect_200-50 put_bottles_dustbin-piper_clean_50-50 turn_switch-demo_clean_collect_200-50'
    default_model_root = "/root/.cache/models/trace-anything/"
    default_cfg_path = Path(__file__).resolve().parent / "traceanything" / "configs" / "eval.yaml"

    parser = argparse.ArgumentParser(
        description="Extract TraceAnything trajectory-field outputs for LeRobot video episodes."
    )
    parser.add_argument("--dataset-root", type=str, default=default_dataset_root)
    parser.add_argument("--ta-model-path", type=str, default=default_model_root)
    parser.add_argument("--cfg-path", type=str, default=str(default_cfg_path))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=str, default="")
    parser.add_argument("--task-names", type=str, nargs="*", default=None)
    parser.add_argument("--views", type=str, nargs="*", default=list(DEFAULT_VIEWS))
    parser.add_argument("--output-root-name", type=str, default="trajectories")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-views", type=int, default=40, help="0 disables episode frame downsampling.")
    parser.add_argument("--max-concurrent-tasks", type=int, default=0)
    parser.add_argument(
        "--keep-evaluated-tracks",
        action="store_true",
        help="Keep track_pts3d/track_conf lists in output.pt. This can be very large.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_roots = discover_task_roots(Path(args.dataset_root))
    args.task_names = ['hanging_mug-demo_clean_collect_200-50', 'move_stapler_pad-demo_clean_collect_200-50', 'put_bottles_dustbin-piper_clean_50-50', 'turn_switch-demo_clean_collect_200-50']
    if args.task_names:
        selected = set(args.task_names)
        task_roots = [path for path in task_roots if path.name in selected]
        if not task_roots:
            raise ValueError(f"No tasks matched --task-names: {sorted(selected)}")

    unknown_views = [view for view in args.views if view not in IMAGE_KEYS]
    if unknown_views:
        raise ValueError(f"Unknown --views values: {unknown_views}. Known views: {sorted(IMAGE_KEYS)}")

    devices = parse_devices(device=args.device, devices=args.devices)
    max_concurrent_tasks = args.max_concurrent_tasks or len(devices)
    _pretty(f"[INFO] found {len(task_roots)} task(s): {[path.name for path in task_roots]}")
    _pretty(f"[INFO] using devices: {devices}")
    _pretty(f"[INFO] views: {args.views}")
    _pretty(f"[INFO] output root name: {args.output_root_name}")

    launch_parallel(
        task_roots=task_roots,
        models_root=Path(args.ta_model_path),
        cfg_path=Path(args.cfg_path),
        devices=devices,
        views_to_process=args.views,
        output_root_name=args.output_root_name,
        skip_existing=not args.overwrite,
        max_views=args.max_views,
        keep_evaluated_tracks=args.keep_evaluated_tracks,
        max_concurrent_tasks=max_concurrent_tasks,
    )


if __name__ == "__main__":
    main()
