#!/usr/bin/env python3
"""Plot urgent benefit and background cost of preparation priority scaling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def mean_ci95(samples: list[float]) -> tuple[float, float]:
    mean = float(np.mean(samples))
    if len(samples) < 2:
        return mean, 0.0
    # All current worker points have nine matched traces. The Student-t
    # multiplier for 8 degrees of freedom is 2.306.
    multiplier = 2.306 if len(samples) == 9 else 1.96
    error = multiplier * float(np.std(samples, ddof=1)) / math.sqrt(len(samples))
    return mean, error


def load_values(
    suite: Path,
) -> dict[int, dict[str, list[dict[str, object]]]]:
    values: dict[int, dict[str, list[dict[str, object]]]] = {}
    for workers in WORKERS:
        values[workers] = {}
        root = suite / f"workers{workers}"
        trace_names = {
            path.parent.parent.name
            for path in root.glob("*/fcfs/summary.json")
        }
        matched = [
            name
            for name in sorted(trace_names)
            if all((root / name / policy / "summary.json").is_file()
                   for policy in POLICIES)
        ]
        if not matched:
            raise SystemExit(f"No matched summaries for {workers} workers")
        for policy in POLICIES:
            values[workers][policy] = [
                json.loads((root / name / policy / "summary.json").read_text())
                for name in matched
            ]
    return values


def metric(
    values: dict[int, dict[str, list[dict[str, object]]]],
    workers: int,
    policy: str,
    section: str | None,
    field: str,
) -> list[float]:
    rows = values[workers][policy]
    if section is None:
        return [float(row[field]) for row in rows]
    return [float(row[section][field]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    values = load_values(args.suite)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panels = (
        ("urgent", "mean_end_to_end_ttft_s", "Urgent mean TTFT (seconds)"),
        ("background", "mean_end_to_end_ttft_s", "Background mean TTFT (seconds)"),
        (None, "throughput_qps", "Throughput (requests/second)"),
    )

    for ax, (section, field, ylabel) in zip(axes, panels, strict=True):
        for policy in POLICIES:
            points = [
                mean_ci95(metric(values, workers, policy, section, field))
                for workers in WORKERS
            ]
            means = [point[0] for point in points]
            errors = [point[1] for point in points]
            ax.errorbar(
                WORKERS,
                means,
                yerr=errors,
                color=COLORS[policy],
                marker=MARKERS[policy],
                markersize=7,
                linewidth=2.2,
                capsize=4,
                label=LABELS[policy],
            )
            for x, y in zip(WORKERS, means, strict=True):
                fmt = f"{y:.2f}" if section is None else f"{y:.1f}"
                offsets = {
                    "fcfs": (0, 8),
                    "priority": (-9, -15),
                    "priority_reserved": (9, 8),
                }
                ax.annotate(
                    fmt,
                    (x, y),
                    xytext=offsets[policy],
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=COLORS[policy],
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(WORKERS, [str(value) for value in WORKERS])
        ax.set_xlabel("Media-preparation workers")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title("Urgent requests: priority benefit")
    axes[1].set_title("Background requests: scheduling cost")
    axes[2].set_title("Aggregate throughput")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=3,
        frameon=True,
    )
    fig.suptitle(
        "More workers improve FCFS but do not eliminate priority inversion\n"
        "Means across 9 matched traces per point; error bars are 95% confidence intervals",
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.77))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
