#!/usr/bin/env python3
"""Plot accuracy across preparation worker counts and scheduling policies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


WORKERS = (2, 4, 8, 16)
POLICIES = ("fcfs", "priority", "priority_reserved")
LABELS = {
    "fcfs": "FCFS media preparation",
    "priority": "Priority media preparation",
    "priority_reserved": "Priority + reserved worker",
}
COLORS = {
    "fcfs": "#D55E00",
    "priority": "#009E73",
    "priority_reserved": "#0072B2",
}
MARKERS = {"fcfs": "o", "priority": "s", "priority_reserved": "^"}


def wilson95(correct: int, total: int) -> tuple[float, float, float]:
    """Return percentage estimate and asymmetric 95% Wilson errors."""
    z = 1.96
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    value = 100 * p
    return value, value - 100 * (center - half), 100 * (center + half) - value


def counts(suite: Path, workers: int, policy: str, workload: str) -> tuple[int, int]:
    correct = 0
    total = 0
    paths = list((suite / f"workers{workers}").glob(f"*/{policy}/summary.json"))
    if not paths:
        raise SystemExit(f"No summaries for workers={workers}, policy={policy}")
    for path in paths:
        row = json.loads(path.read_text())[workload]
        correct += int(row["correct"])
        total += int(row["requests"])
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    panels = (
        ("urgent", "Urgent-video accuracy", (45, 70)),
        ("background", "Background-video accuracy", (55, 75)),
    )

    for ax, (workload, title, ylim) in zip(axes, panels, strict=True):
        x_base = list(range(len(WORKERS)))
        for policy_index, policy in enumerate(POLICIES):
            estimates = []
            lower = []
            upper = []
            totals = []
            for workers in WORKERS:
                correct, total = counts(args.suite, workers, policy, workload)
                value, low_error, high_error = wilson95(correct, total)
                estimates.append(value)
                lower.append(low_error)
                upper.append(high_error)
                totals.append(total)
            ax.errorbar(
                [value + (policy_index - 1) * 0.08 for value in x_base],
                estimates,
                yerr=[lower, upper],
                color=COLORS[policy],
                marker=MARKERS[policy],
                markersize=7,
                linewidth=2.2,
                capsize=4,
                label=LABELS[policy],
            )

        sample_total = counts(args.suite, WORKERS[0], POLICIES[0], workload)[1]
        ax.set_title(f"{title} (n={sample_total} per point)")
        ax.set_xticks(x_base, [str(value) for value in WORKERS])
        ax.set_xlabel("Media-preparation workers")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=3,
        frameon=True,
    )
    fig.suptitle(
        "Priority-aware media preparation preserves answer accuracy\n"
        "Points are pooled accuracy; error bars are 95% Wilson confidence intervals",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.76))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
