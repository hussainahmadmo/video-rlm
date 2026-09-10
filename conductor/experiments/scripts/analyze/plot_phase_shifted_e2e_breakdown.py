#!/usr/bin/env python3
"""Plot where the phase-shifted FCFS-to-max-min E2E gain comes from."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = (
    ("Preparation queue", "prep_queue_wait_s", "#4C78A8"),
    ("Preparation service", "prep_service_s", "#F2CF5B"),
    ("Handoff queue", "prepared_queue_wait_s", "#E45756"),
    ("Inference", "vlm_service_s", "#72B7B2"),
)
POLICIES = ("FCFS", "Max–min")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fcfs", type=Path, nargs="+", required=True,
        help="Matched FCFS results.jsonl files, one per seed",
    )
    parser.add_argument(
        "--max-min", dest="max_min", type=Path, nargs="+", required=True,
        help="Matched max-min results.jsonl files in the same seed order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Where the End-to-End Improvement Comes From")
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("error") is None]
    if not rows:
        raise ValueError(f"no successful request rows in {path}")
    return rows


def seed_means(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    component_rows = []
    e2e_rows = []
    for path in paths:
        rows = load(path)
        component_rows.append([
            np.mean([float(row[field]) for row in rows])
            for _, field, _ in COMPONENTS
        ])
        e2e_rows.append(np.mean([float(row["end_to_end_s"]) for row in rows]))
    return np.asarray(component_rows), np.asarray(e2e_rows)


def main() -> None:
    args = parse_args()
    if len(args.fcfs) != len(args.max_min):
        raise SystemExit("--fcfs and --max-min must contain the same number of matched seeds")
    for path in [*args.fcfs, *args.max_min]:
        if not path.is_file():
            raise SystemExit(f"missing results file: {path}")

    fcfs_components, fcfs_e2e = seed_means(args.fcfs)
    mm_components, mm_e2e = seed_means(args.max_min)
    policy_components = np.stack((fcfs_components.mean(axis=0), mm_components.mean(axis=0)))
    component_savings = fcfs_components - mm_components
    e2e_savings = fcfs_e2e - mm_e2e

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    # Means are used here because component means add; medians and percentiles do not.
    bottoms = np.zeros(2)
    x = np.arange(2)
    for index, (label, _, color) in enumerate(COMPONENTS):
        values = policy_components[:, index]
        axes[0].bar(x, values, bottom=bottoms, width=0.62, label=label,
                    color=color, edgecolor="white", linewidth=0.7)
        bottoms += values
    axes[0].set_xticks(x, POLICIES)
    axes[0].set_ylabel("Mean end-to-end latency (s)")
    axes[0].set_title("(a) Mean E2E decomposition")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    for xpos, value in zip(x, (fcfs_e2e.mean(), mm_e2e.mean())):
        axes[0].text(xpos, value + max(fcfs_e2e.mean(), mm_e2e.mean()) * 0.02,
                     f"{value:.1f} s", ha="center", va="bottom", fontweight="bold")

    labels = [item[0] for item in COMPONENTS] + ["Total E2E"]
    savings = np.append(component_savings.mean(axis=0), e2e_savings.mean())
    colors = [item[2] for item in COMPONENTS] + ["#2E8540"]
    sx = np.arange(len(labels))
    axes[1].axhline(0, color="#444444", linewidth=0.9)
    axes[1].bar(sx, savings, color=colors, width=0.68)
    for seed in range(component_savings.shape[0]):
        seed_values = np.append(component_savings[seed], e2e_savings[seed])
        jitter = (seed - (component_savings.shape[0] - 1) / 2) * 0.07
        axes[1].scatter(sx + jitter, seed_values, color="black", s=20,
                        alpha=0.55, zorder=3)
    axes[1].set_xticks(sx, labels, rotation=23, ha="right")
    axes[1].set_ylabel("Seconds saved by max–min")
    axes[1].set_title("(b) FCFS minus max–min")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)
    for xpos, value in zip(sx, savings):
        offset = 0.7 if value >= 0 else -0.7
        axes[1].text(xpos, value + offset, f"{value:+.1f}", ha="center",
                     va="bottom" if value >= 0 else "top", fontsize=9,
                     fontweight="bold")

    fig.suptitle(args.title, fontsize=16, fontweight="bold")
    fig.text(
        0.5, -0.02,
        "Bars average matched seeds; black points show individual seeds. "
        "Positive savings mean max–min is faster.",
        ha="center", fontsize=9, color="#4B5563",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
