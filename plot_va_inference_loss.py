import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


COLORS = {
    "latent_loss": "#2563eb",
    "action_losses": "#dc2626",
    "align_losses": "#059669",
}


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "plot"


def load_loss_data(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Loss JSON must be an object.")
    for metric, tasks in data.items():
        if not isinstance(tasks, dict):
            raise ValueError(f"Metric {metric!r} must contain a task object.")
    return data


def task_keys(data):
    keys = set()
    for tasks in data.values():
        keys.update(tasks.keys())
    return sorted(keys)


def plot_task(task, data, metrics, output_dir):
    task_name = os.path.basename(task.rstrip(os.sep)) or task
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = data.get(metric, {}).get(task, [])
        xs = list(range(1, len(values) + 1))
        color = COLORS.get(metric)

        ax.plot(
            xs,
            values,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=metric,
        )
        if values:
            mean = sum(values) / len(values)
            ax.axhline(
                mean,
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.82,
                label=f"mean={mean:.6g}",
            )

        ax.set_ylabel(metric)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.legend(loc="best", fontsize=9)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[-1].set_xlabel("sample index")
    fig.suptitle(task_name, fontsize=14, fontweight="bold")

    output_path = output_dir / f"{safe_filename(task_name)}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_summary(data, tasks, metrics, output_dir):
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = list(range(len(tasks)))
    width = 0.8 / max(1, len(metrics))

    for metric_index, metric in enumerate(metrics):
        offsets = [value + (metric_index - (len(metrics) - 1) / 2) * width for value in x]
        means = []
        labels = []
        for task in tasks:
            values = data.get(metric, {}).get(task, [])
            means.append(sum(values) / len(values) if values else 0.0)
            labels.append(os.path.basename(task.rstrip(os.sep)) or task)
        ax.bar(offsets, means, width=width, color=COLORS.get(metric), label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("mean loss")
    ax.set_title("VA inference loss means", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    output_path = output_dir / "summary_means.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot VA inference loss JSON.")
    parser.add_argument(
        "--input",
        default="va_inference_loss.json",
        help="Path to va_inference_loss.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="va_inference_loss_plots",
        help="Directory to save PNG figures.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_loss_data(input_path)
    metrics = list(data.keys())
    tasks = task_keys(data)

    saved = []
    for task in tasks:
        saved.append(plot_task(task, data, metrics, output_dir))
    saved.append(plot_summary(data, tasks, metrics, output_dir))

    print(f"Saved {len(saved)} plot(s) to {output_dir}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
