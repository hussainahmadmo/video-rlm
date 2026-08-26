#!/usr/bin/env python3
"""Plot matched low-frame FCFS-prep and priority-prep results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FRAMES = (2, 4, 8)
POLICIES = ("fcfs", "priority")
LABELS = {
    "fcfs": "Engine-only priority",
    "priority": "End-to-end priority",
}
COLORS = {"fcfs": "#D55E00", "priority": "#009E73"}
MARKERS = {"fcfs": "o", "priority": "s"}


def load_summaries(root: Path, frames: int, policy: str) -> list[dict]:
    paths = sorted(root.glob(f"frames{frames}/*/{policy}/summary.json"))
    if not paths:
        raise SystemExit(f"No summaries for frames={frames}, policy={policy}")
    return [json.loads(path.read_text()) for path in paths]


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    # Student-t critical value for n=3; all points in this experiment use 3 seeds.
    t_value = 4.303 if len(values) == 3 else 1.96
    error = t_value * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    metrics = (
        (
            "mean_end_to_end_ttft_s",
            "(a) Urgent end-to-end TTFT",
            "Mean TTFT (seconds)",
        ),
        (
            "mean_prep_queue_wait_s",
            "(b) Urgent media-preparation wait",
            "Mean queue wait (seconds)",
        ),
    )

    summaries = {
        (frames, policy): load_summaries(args.root, frames, policy)
        for frames in FRAMES
        for policy in POLICIES
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45), sharex=True)
    means_by_metric: dict[str, dict[str, list[float]]] = {}

    for ax, (field, title, ylabel) in zip(axes, metrics, strict=True):
        means_by_metric[field] = {}
        for policy in POLICIES:
            means: list[float] = []
            errors: list[float] = []
            for frames in FRAMES:
                values = [
                    float(summary["urgent"][field])
                    for summary in summaries[(frames, policy)]
                ]
                mean, error = mean_ci95(values)
                means.append(mean)
                errors.append(error)
            means_by_metric[field][policy] = means
            ax.errorbar(
                FRAMES,
                means,
                yerr=errors,
                color=COLORS[policy],
                marker=MARKERS[policy],
                markersize=6,
                linewidth=2,
                capsize=4,
                label=LABELS[policy],
            )
            for frames, mean in zip(FRAMES, means, strict=True):
                offset = 0.55 if policy == "fcfs" else -0.45
                ax.text(
                    frames,
                    max(0.08, mean + offset),
                    f"{mean:.1f}s",
                    color=COLORS[policy],
                    ha="center",
                    va="bottom" if policy == "fcfs" else "top",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Frames prepared per video request")
        ax.set_ylabel(ylabel)
        ax.set_xticks(FRAMES)
        ax.set_xlim(1.5, 8.5)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, linewidth=0.7)

    ttft = means_by_metric["mean_end_to_end_ttft_s"]
    for index, frames in enumerate(FRAMES):
        speedup = ttft["fcfs"][index] / ttft["priority"][index]
        low = ttft["priority"][index]
        high = ttft["fcfs"][index]
        label_y = max(3.0, low + 0.45 * (high - low))
        axes[0].text(
            frames,
            label_y,
            f"{speedup:.2f}$\\times$",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Priority benefit emerges when media preparation becomes congested",
        y=1.12,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Burst workload: 64 background + 16 urgent requests; 4 preparation workers; "
        "3 matched seeds per point. Error bars are 95% confidence intervals.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(args.output)
    print(pdf)


if __name__ == "__main__":
    main()
