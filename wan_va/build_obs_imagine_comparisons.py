import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_VIEWS = (
    ("high", ("cam_high", "high")),
    ("left_wrist", ("cam_left_wrist", "left_wrist", "left")),
    ("right_wrist", ("cam_right_wrist", "right_wrist", "right")),
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(path):
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def image_files(directory):
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def discover_units(input_dir):
    input_dir = Path(input_dir)
    units = []

    if (input_dir / "obs").is_dir() and (input_dir / "imagine").is_dir():
        return [(input_dir.name, input_dir.name, input_dir)]

    direct_steps = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_dir() and path.name.startswith("step_")
            and (path / "obs").is_dir()
            and (path / "imagine").is_dir()
        ],
        key=natural_key,
    )
    if direct_steps:
        return [(input_dir.name, step.name, step) for step in direct_steps]

    for task_dir in sorted([path for path in input_dir.iterdir() if path.is_dir()], key=natural_key):
        if (task_dir / "obs").is_dir() and (task_dir / "imagine").is_dir():
            units.append((task_dir.name, task_dir.name, task_dir))
            continue

        for step_dir in sorted(task_dir.glob("step_*"), key=natural_key):
            if (step_dir / "obs").is_dir() and (step_dir / "imagine").is_dir():
                units.append((task_dir.name, step_dir.name, step_dir))

    return units


def open_rgb(path):
    return Image.open(path).convert("RGB")


def find_obs_images(obs_dir, aliases):
    matched = []
    for path in image_files(obs_dir):
        name = path.name.lower()
        if any(alias.lower() in name for alias in aliases):
            matched.append(open_rgb(path))
    return matched


def split_tshape(image):
    width, height = image.size
    top_h = height // 3
    high = image.crop((0, top_h, width, height))
    left = image.crop((0, 0, width // 2, top_h))
    right = image.crop((width // 2, 0, width, top_h))
    return {"high": high, "left_wrist": left, "right_wrist": right}


def split_horizontal(image):
    width, height = image.size
    part_w = width // 3
    return {
        "high": image.crop((0, 0, part_w, height)),
        "left_wrist": image.crop((part_w, 0, part_w * 2, height)),
        "right_wrist": image.crop((part_w * 2, 0, width, height)),
    }


def split_vertical(image):
    width, height = image.size
    part_h = height // 3
    return {
        "high": image.crop((0, 0, width, part_h)),
        "left_wrist": image.crop((0, part_h, width, part_h * 2)),
        "right_wrist": image.crop((0, part_h * 2, width, height)),
    }


def split_imagine_frame(image, layout):
    width, height = image.size
    if layout == "auto":
        if height > width * 1.1:
            layout = "tshape"
        elif width > height * 1.8:
            layout = "horizontal"
        else:
            layout = "horizontal"

    if layout == "tshape":
        return split_tshape(image)
    if layout == "vertical":
        return split_vertical(image)
    if layout == "horizontal":
        return split_horizontal(image)
    raise ValueError(f"Unknown imagine layout: {layout}")


def find_imagine_images(imagine_dir, view_name, aliases, layout):
    separate = []
    generic = []
    for path in image_files(imagine_dir):
        name = path.name.lower()
        if any(alias.lower() in name for alias in aliases):
            separate.append(open_rgb(path))
        else:
            generic.append(path)

    if separate:
        return separate

    frames = []
    for path in generic:
        split = split_imagine_frame(open_rgb(path), layout)
        if view_name in split:
            frames.append(split[view_name])
    return frames


def fit_height(image, target_h):
    width, height = image.size
    if height == target_h:
        return image
    target_w = max(1, round(width * target_h / height))
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def make_strip(images, row_h, gap, empty_text):
    if not images:
        canvas = Image.new("RGB", (row_h * 2, row_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, row_h // 2 - 6), empty_text, fill=(130, 130, 130))
        return canvas

    resized = [fit_height(image, row_h) for image in images]
    width = sum(image.width for image in resized) + gap * (len(resized) - 1)
    strip = Image.new("RGB", (width, row_h), "white")
    x = 0
    for image in resized:
        strip.paste(image, (x, 0))
        x += image.width + gap
    return strip


def draw_label(draw, box, text):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(245, 246, 248))
    draw.text((x0 + 12, y0 + max(8, (y1 - y0 - 12) // 2)), text, fill=(30, 34, 40))


def build_comparison(unit_dir, output_path, task_name, step_name, views, layout, row_h, gap, label_w):
    rows = []
    for view_name, aliases in views:
        obs_images = find_obs_images(unit_dir / "obs", aliases)
        imagine_images = find_imagine_images(unit_dir / "imagine", view_name, aliases, layout)
        rows.append((f"{view_name} obs", make_strip(obs_images, row_h, gap, "no obs")))
        rows.append((f"{view_name} imagine", make_strip(imagine_images, row_h, gap, "no imagine")))

    title_h = 36
    row_gap = gap
    content_w = max(strip.width for _, strip in rows)
    canvas_w = label_w + content_w
    canvas_h = title_h + len(rows) * row_h + (len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, canvas_w, title_h), fill=(26, 30, 36))
    draw.text((12, 11), f"{task_name} / {step_name}", fill="white")

    y = title_h
    for label, strip in rows:
        draw_label(draw, (0, y, label_w, y + row_h), label)
        canvas.paste(strip, (label_w, y))
        if strip.width < content_w:
            draw.rectangle((label_w + strip.width, y, canvas_w, y + row_h), fill="white")
        y += row_h + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_views(value):
    if not value:
        return DEFAULT_VIEWS
    views = []
    for item in value.split(","):
        name, _, alias_text = item.partition(":")
        aliases = tuple(alias_text.split("|")) if alias_text else (name,)
        views.append((name, aliases))
    return tuple(views)


def main():
    parser = argparse.ArgumentParser(
        description="Build 6-row obs/imagine comparison images for saved VA inference results."
    )
    parser.add_argument("input_dir", type=Path, help="directory containing task dirs or step dirs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where comparison images are written; default: input_dir/comparisons",
    )
    parser.add_argument(
        "--imagine-layout",
        choices=("auto", "tshape", "horizontal", "vertical"),
        default="auto",
        help="how to split a combined imagine frame into high/left/right views",
    )
    parser.add_argument("--row-height", type=int, default=180)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--label-width", type=int, default=150)
    parser.add_argument(
        "--views",
        default=None,
        help="custom views, e.g. 'high:cam_high,left_wrist:cam_left,right_wrist:cam_right'",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_dir / "comparisons")
    units = discover_units(args.input_dir)
    if not units:
        raise SystemExit(f"No obs/imagine step directories found under {args.input_dir}")

    views = parse_views(args.views)
    for task_name, step_name, unit_dir in units:
        output_path = output_dir / task_name / f"{step_name}.png"
        build_comparison(
            unit_dir=unit_dir,
            output_path=output_path,
            task_name=task_name,
            step_name=step_name,
            views=views,
            layout=args.imagine_layout,
            row_h=args.row_height,
            gap=args.gap,
            label_w=args.label_width,
        )
        print(output_path)


if __name__ == "__main__":
    main()
