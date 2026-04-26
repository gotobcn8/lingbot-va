import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw


DEFAULT_EPISODE_DIR = Path(
    "/sharedata/lsy/robotwin/open_microwave-demo_clean_collect_200-50/"
    "trajectories/chunk-000/observation.images.cam_high/episode_000000_0_455"
)

DEFAULT_VIEWS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_axes_equal(ax):
    """
    让 3D 坐标轴比例一致，不然人会被拉伸
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_ctrl_pts3d(
    ctrl_pts3d,
    ctrl_conf=None,
    fg_mask=None,
    conf_thresh=None,
    max_points=20000,
    out_file="ctrl_pts3d_maps.png",
    elev=20,
    azim=-60,
    point_size=0.5,
):
    """
    ctrl_pts3d: [K, H, W, 3]
    ctrl_conf:  [K, H, W] or None
    fg_mask:    [H, W] or None
    """
    ctrl_pts3d = to_numpy(ctrl_pts3d)
    K, H, W, C = ctrl_pts3d.shape
    assert C == 3, f"Expected last dim = 3, got {C}"

    if ctrl_conf is not None:
        ctrl_conf = to_numpy(ctrl_conf)
        assert ctrl_conf.shape == (K, H, W)

    if fg_mask is not None:
        fg_mask = to_numpy(fg_mask).astype(bool)
        assert fg_mask.shape == (H, W)

    # 子图排列
    ncols = min(4, K)
    nrows = math.ceil(K / ncols)

    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))

    # 给每个 control point map 一个固定颜色
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(K)]

    for k in range(K):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")

        pts = ctrl_pts3d[k].reshape(-1, 3)  # [H*W, 3]

        valid = np.isfinite(pts).all(axis=1)

        if fg_mask is not None:
            valid &= fg_mask.reshape(-1)

        if ctrl_conf is not None and conf_thresh is not None:
            conf = ctrl_conf[k].reshape(-1)
            valid &= (conf >= conf_thresh)

        pts = pts[valid]

        # 点太多就随机采样，防止太卡
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]

        if len(pts) > 0:
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=point_size,
                c=[colors[k]],
                alpha=0.8,
                linewidths=0,
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Map {k}", fontsize=12)
        ax.set_axis_off()
        set_axes_equal(ax)

    plt.tight_layout()
    plt.savefig(out_file, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_file}")


def load_output(output_pt):
    data = torch.load(output_pt, map_location="cpu", weights_only=False)
    return data["preds"] if isinstance(data, dict) and "preds" in data else data


def resize_rgb_to_hw(rgb, height, width):
    if rgb.shape[:2] == (height, width):
        return rgb
    image = Image.fromarray(rgb)
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image)


def pred_to_points_and_colors(
    pred,
    image_rgb,
    ctrl_mode="mean",
    ctrl_index=0,
    use_mask=True,
    conf_thresh=None,
    max_points=3000,
    seed=0,
):
    """
    Return sampled 3D points and RGB colors from one TraceAnything prediction.

    ctrl_mode:
      mean  - average K control-point maps into one [H,W,3] surface
      index - use ctrl_pts3d[ctrl_index]
      all   - flatten all K maps; image colors are repeated for each K
    """
    ctrl_pts3d = to_numpy(pred["ctrl_pts3d"])
    if ctrl_pts3d.ndim != 4 or ctrl_pts3d.shape[-1] != 3:
        raise ValueError(f"Expected ctrl_pts3d [K,H,W,3], got {ctrl_pts3d.shape}")

    k_count, height, width, _ = ctrl_pts3d.shape
    image_rgb = resize_rgb_to_hw(image_rgb, height, width)

    if ctrl_mode == "mean":
        points_map = np.nanmean(ctrl_pts3d, axis=0)
        points = points_map.reshape(-1, 3)
        colors = image_rgb.reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        if use_mask and "fg_mask" in pred:
            valid &= to_numpy(pred["fg_mask"]).astype(bool).reshape(-1)
        if conf_thresh is not None and "ctrl_conf" in pred:
            conf = to_numpy(pred["ctrl_conf"]).mean(axis=0).reshape(-1)
            valid &= conf >= conf_thresh
    elif ctrl_mode == "index":
        if not 0 <= ctrl_index < k_count:
            raise ValueError(f"ctrl_index should be in [0, {k_count - 1}], got {ctrl_index}")
        points = ctrl_pts3d[ctrl_index].reshape(-1, 3)
        colors = image_rgb.reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        if use_mask and "fg_mask" in pred:
            valid &= to_numpy(pred["fg_mask"]).astype(bool).reshape(-1)
        if conf_thresh is not None and "ctrl_conf" in pred:
            conf = to_numpy(pred["ctrl_conf"])[ctrl_index].reshape(-1)
            valid &= conf >= conf_thresh
    elif ctrl_mode == "all":
        points = ctrl_pts3d.reshape(-1, 3)
        colors = np.tile(image_rgb.reshape(-1, 3), (k_count, 1))
        valid = np.isfinite(points).all(axis=1)
        if use_mask and "fg_mask" in pred:
            mask = np.tile(to_numpy(pred["fg_mask"]).astype(bool).reshape(-1), k_count)
            valid &= mask
        if conf_thresh is not None and "ctrl_conf" in pred:
            conf = to_numpy(pred["ctrl_conf"]).reshape(-1)
            valid &= conf >= conf_thresh
    else:
        raise ValueError(f"Unknown ctrl_mode: {ctrl_mode}")

    points = points[valid]
    colors = colors[valid]
    if len(points) == 0:
        return points, colors

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    return points, colors


def robust_axis_limits(points_list, low=1.0, high=99.0):
    points_list = [pts for pts in points_list if len(pts) > 0]
    if not points_list:
        return None

    pts = np.concatenate(points_list, axis=0)
    mins = np.percentile(pts, low, axis=0)
    maxs = np.percentile(pts, high, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def render_points_on_image(
    points,
    colors,
    image_rgb,
    cell_size=220,
    axis_limits=None,
    elev=18,
    azim=-70,
    point_size=0.35,
    background_alpha=0.62,
    dpi=100,
):
    background = Image.fromarray(image_rgb).convert("RGB").resize((cell_size, cell_size))
    white = Image.new("RGB", background.size, (255, 255, 255))
    background = Image.blend(white, background, background_alpha).convert("RGBA")

    fig = plt.figure(figsize=(cell_size / dpi, cell_size / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.patch.set_alpha(0.0)

    if len(points) > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=colors.astype(np.float32) / 255.0,
            alpha=0.95,
            linewidths=0,
            depthshade=False,
        )

    if axis_limits is not None:
        ax.set_xlim3d(*axis_limits[0])
        ax.set_ylim3d(*axis_limits[1])
        ax.set_zlim3d(*axis_limits[2])
    else:
        set_axes_equal(ax)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.grid(False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rendered = np.asarray(canvas.buffer_rgba())
    plt.close(fig)

    overlay = Image.fromarray(rendered, mode="RGBA")
    return Image.alpha_composite(background, overlay).convert("RGB")


def make_episode_view_montage(
    episode_dir,
    views=DEFAULT_VIEWS,
    out_file=None,
    output_dir=None,
    max_frames=40,
    cell_size=220,
    gap=2,
    label_width=150,
    ctrl_mode="mean",
    ctrl_index=0,
    use_mask=True,
    conf_thresh=None,
    max_points=3000,
    elev=18,
    azim=-70,
    point_size=0.35,
    background_alpha=0.62,
    seed=0,
):
    episode_dir = Path(episode_dir).resolve()
    chunk_dir = episode_dir.parent.parent
    episode_name = episode_dir.name

    view_dirs = [chunk_dir / view / episode_name for view in views]
    missing = [str(path) for path in view_dirs if not (path / "output.pt").is_file()]
    if missing:
        raise FileNotFoundError("Missing output.pt under:\n" + "\n".join(missing))

    rows = []
    for row_id, (view, view_dir) in enumerate(zip(views, view_dirs)):
        preds = load_output(view_dir / "output.pt")
        images_dir = view_dir / "images"
        frame_count = min(max_frames, len(preds))

        sampled = []
        images = []
        for frame_id in range(frame_count):
            image_path = images_dir / f"{frame_id:03d}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            points, colors = pred_to_points_and_colors(
                preds[frame_id],
                image_rgb,
                ctrl_mode=ctrl_mode,
                ctrl_index=ctrl_index,
                use_mask=use_mask,
                conf_thresh=conf_thresh,
                max_points=max_points,
                seed=seed + row_id * 1000 + frame_id,
            )
            sampled.append((points, colors))
            images.append(image_rgb)

        axis_limits = robust_axis_limits([points for points, _ in sampled])

        cells = []
        for frame_id, ((points, colors), image_rgb) in enumerate(zip(sampled, images)):
            cell = render_points_on_image(
                points,
                colors,
                image_rgb,
                cell_size=cell_size,
                axis_limits=axis_limits,
                elev=elev,
                azim=azim,
                point_size=point_size,
                background_alpha=background_alpha,
            )
            draw = ImageDraw.Draw(cell)
            draw.rectangle((0, 0, 42, 18), fill=(255, 255, 255))
            draw.text((4, 2), f"{frame_id:02d}", fill=(0, 0, 0))
            cells.append(cell)

        rows.append((view, cells))

    ncols = max(len(cells) for _, cells in rows)
    width = label_width + ncols * cell_size + (ncols - 1) * gap
    height = len(rows) * cell_size + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row_id, (view, cells) in enumerate(rows):
        y = row_id * (cell_size + gap)
        draw.rectangle((0, y, label_width - 1, y + cell_size), fill=(245, 245, 245))
        short_view = view.replace("observation.images.", "")
        draw.text((10, y + 12), short_view, fill=(0, 0, 0))
        draw.text((10, y + 34), episode_name, fill=(70, 70, 70))
        for col_id, cell in enumerate(cells):
            x = label_width + col_id * (cell_size + gap)
            canvas.paste(cell, (x, y))

    if out_file is None:
        if output_dir is None:
            output_dir = chunk_dir / "3d_control"
        out_file = Path(output_dir) / f"{episode_name}.png"
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)
    print(f"Saved episode montage to: {out_file}")
    return out_file


def discover_episode_dirs(anchor_episode_dir, first_view):
    anchor_episode_dir = Path(anchor_episode_dir).resolve()
    chunk_dir = anchor_episode_dir.parent.parent
    first_view_dir = chunk_dir / first_view
    if not first_view_dir.is_dir():
        raise FileNotFoundError(first_view_dir)
    return sorted(path for path in first_view_dir.iterdir() if path.is_dir())


def make_all_episode_montages(
    anchor_episode_dir,
    views=DEFAULT_VIEWS,
    output_dir=None,
    skip_existing=True,
    limit_episodes=0,
    **kwargs,
):
    episode_dirs = discover_episode_dirs(anchor_episode_dir, views[0])
    if limit_episodes > 0:
        episode_dirs = episode_dirs[:limit_episodes]

    chunk_dir = Path(anchor_episode_dir).resolve().parent.parent
    if output_dir is None:
        output_dir = chunk_dir / "3d_control"
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
        make_episode_view_montage(
            episode_dir=episode_dir,
            views=views,
            out_file=out_file,
            **kwargs,
        )
        written += 1

    print(f"Done. written={written}, skipped={skipped}, output_dir={output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one TraceAnything episode as a 3-row montage: one row per camera view."
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default=str(DEFAULT_EPISODE_DIR),
        help="Path to one view's episode directory. The script finds sibling view episode dirs from it.",
    )
    parser.add_argument("--views", type=str, nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--out-file", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for episode PNGs. Default: <chunk>/3d_control.",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Render every episode under the first view into output-dir/episode_name.png.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--cell-size", type=int, default=220)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--label-width", type=int, default=150)
    parser.add_argument("--ctrl-mode", choices=("mean", "index", "all"), default="mean")
    parser.add_argument("--ctrl-index", type=int, default=0)
    parser.add_argument("--conf-thresh", type=float, default=None)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-70.0)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--background-alpha", type=float, default=0.62)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    montage_kwargs = dict(
        max_frames=args.max_frames,
        cell_size=args.cell_size,
        gap=args.gap,
        label_width=args.label_width,
        ctrl_mode=args.ctrl_mode,
        ctrl_index=args.ctrl_index,
        use_mask=not args.no_mask,
        conf_thresh=args.conf_thresh,
        max_points=args.max_points,
        elev=args.elev,
        azim=args.azim,
        point_size=args.point_size,
        background_alpha=args.background_alpha,
        seed=args.seed,
    )

    if args.all_episodes:
        make_all_episode_montages(
            anchor_episode_dir=args.episode_dir,
            views=args.views,
            output_dir=args.output_dir,
            skip_existing=not args.overwrite,
            limit_episodes=args.limit_episodes,
            **montage_kwargs,
        )
    else:
        make_episode_view_montage(
            episode_dir=args.episode_dir,
            views=args.views,
            out_file=args.out_file,
            output_dir=args.output_dir,
            **montage_kwargs,
        )
