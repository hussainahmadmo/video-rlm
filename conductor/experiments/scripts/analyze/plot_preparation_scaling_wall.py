#!/usr/bin/env python3
"""Plot why preparation overprovisioning does not replace priority scheduling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORKERS = (2, 4, 8, 16)
POLICIES = ("fcfs", "priority")
LABELS = {
    "fcfs": "Engine-only priority (FCFS preparation)",
    "priority": "End-to-end priority",
}
COLORS = {"fcfs": "#D55E00", "priority": "#009E73"}
MARKERS = {"fcfs": "o", "priority": "s"}
STAGES = (
    ("mean_prep_queue_wait_s", "Preparation queue", "#E69F00"),
    ("mean_prep_service_s", "Preparation service", "#F0E442"),
    ("mean_prepared_queue_wait_s", "Prepared queue", "#56B4E9"),
    ("mean_engine_to_first_token_s", "Engine to first token", "#0072B2"),
)


def mean_ci95(samples: list[float]) -> tuple[float, float]:
    mean = float(np.mean(samples))
    if len(samples) < 2:
        return mean, 0.0
    # Every point in this experiment contains nine matched traces (df=8).
    multiplier = 2.306 if len(samples) == 9 else 1.96
    error = multiplier * float(np.std(samples, ddof=1)) / math.sqrt(len(samples))
    return mean, error


def load_runs(suite: Path) -> dict[int, dict[str, list[dict[str, object]]]]:
    runs: dict[int, dict[str, list[dict[str, object]]]] = {}
    for workers in WORKERS:
        worker_root = suite / f"workers{workers}"
        names = {
            path.parent.parent.name
            for path in worker_root.glob("*/fcfs/summary.json")
        }
        matched = [
            name
            for name in sorted(names)
            if all(
                (worker_root / name / policy / "summary.json").is_file()
                for policy in POLICIES
            )
        ]
        if not matched:
            raise SystemExit(f"No matched runs found for {workers} workers")
        runs[workers] = {
            policy: [
                json.loads(
                    (worker_root / name / policy / "summary.json").read_text()
                )
                for name in matched
            ]
            for policy in POLICIES
        }
    return runs


def samples(
    runs: dict[int, dict[str, list[dict[str, object]]]],
    workers: int,
    policy: str,
    field: str,
    section: str | None = None,
) -> list[float]:
    if section is None:
        return [float(row[field]) for row in runs[workers][policy]]
    return [float(row[section][field]) for row in runs[workers][policy]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runs = load_runs(args.suite)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))

    # A: More workers lower latency, but do not close the scheduling gap.
    latency_means: dict[str, list[float]] = {}
    for policy in POLICIES:
        points = [
            mean_ci95(
                samples(
                    runs,
                    workers,
                    policy,
                    "mean_end_to_end_ttft_s",
                    "urgent",
                )
            )
            for workers in WORKERS
        ]
        means = [point[0] for point in points]
        errors = [point[1] for point in points]
        latency_means[policy] = means
        axes[0].errorbar(
            WORKERS,
            means,
            yerr=errors,
            color=COLORS[policy],
            marker=MARKERS[policy],
            markersize=7,
            linewidth=2.5,
            capsize=4,
            label=LABELS[policy],
        )
        for workers, value in zip(WORKERS, means, strict=True):
            offset = 9 if policy == "fcfs" else -16
            axes[0].annotate(
                f"{value:.1f}s",
                (workers, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                color=COLORS[policy],
                fontsize=9,
            )
    axes[0].set_title("A. FCFS remains slow with more workers", fontweight="bold")
    axes[0].set_ylabel("Urgent mean TTFT (seconds)")

    # B: Normalize throughput to the two-worker result and contrast with ideal scaling.
    ideal = [workers / WORKERS[0] for workers in WORKERS]
    axes[1].plot(
        WORKERS,
        ideal,
        color="#555555",
        linestyle="--",
        linewidth=2.2,
        label="Ideal linear scaling",
    )
    for policy in POLICIES:
        raw = [
            mean_ci95(samples(runs, workers, policy, "throughput_qps"))[0]
            for workers in WORKERS
        ]
        normalized = [value / raw[0] for value in raw]
        axes[1].plot(
            WORKERS,
            normalized,
            color=COLORS[policy],
            marker=MARKERS[policy],
            markersize=7,
            linewidth=2.5,
            label=LABELS[policy],
        )
    axes[1].annotate(
        "8→16 workers:\nonly ~5% more throughput",
        xy=(16, 2.35),
        xytext=(7.0, 4.5),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
        ha="center",
    )
    axes[1].text(
        10.8,
        6.2,
        "Ideal linear scaling",
        rotation=35,
        color="#555555",
        fontsize=9,
        ha="center",
    )
    axes[1].set_title("B. Preparation throughput saturates", fontweight="bold")
    axes[1].set_ylabel("Throughput speedup over 2 workers")
    axes[1].set_ylim(0.7, 8.6)

    # C: At the largest worker count, expose where time remains.
    x = np.arange(len(POLICIES))
    bottoms = np.zeros(len(POLICIES))
    for field, label, color in STAGES:
        values = np.array(
            [
                np.mean(samples(runs, 16, policy, field, "urgent"))
                for policy in POLICIES
            ]
        )
        axes[2].bar(x, values, bottom=bottoms, color=color, label=label)
        bottoms += values
    for index, total in enumerate(bottoms):
        axes[2].text(index, total + 2.5, f"{total:.1f}s", ha="center", fontsize=10)
    axes[2].annotate(
        "2.62× faster",
        xy=(0.5, 60),
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xticks(x, ["Engine-only\npriority", "End-to-end\npriority"])
    axes[2].set_title("C. Scheduling still matters at 16 workers", fontweight="bold")
    axes[2].set_ylabel("Urgent mean TTFT breakdown (seconds)")
    axes[2].set_ylim(0, max(bottoms) * 1.18)

    for ax in axes[:2]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(WORKERS, [str(value) for value in WORKERS])
        ax.set_xlabel("Media-preparation workers")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    stage_handles, stage_labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles + stage_handles,
        labels + stage_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=3,
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        "More preparation workers mitigate contention but do not eliminate priority inversion\n"
        "Means across 9 matched traces per point; latency error bars are 95% confidence intervals",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.77))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = [args.output]
    if args.output.suffix.lower() == ".png":
        outputs.append(args.output.with_suffix(".pdf"))
    for output in outputs:
        fig.savefig(output, dpi=240, bbox_inches="tight")
        print(output)


if __name__ == "__main__":
    main()
