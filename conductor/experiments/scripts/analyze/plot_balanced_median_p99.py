#!/usr/bin/env python3
"""Plot median versus p99 E2E latency for the balanced workload suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = [
    ("fcfs", "FCFS", "s", "#D55E00"),
    ("tenant_round_robin", "Tenant RR", "D", "#8C78B8"),
    ("engine_tenant_fair", "Inference-only", "^", "#56B4E9"),
    ("prep_max_min", "Preparation-only", "o", "#E69F00"),
    ("max_min", "Independent max-min", "P", "#7CAE00"),
    ("cross_stage", "Cross-stage", "v", "#009E73"),
]

LOADS = [
    ("balanced-low", "(a) Low offered load: 0.2 requests/s"),
    ("balanced-high", "(b) High offered load: 0.8 requests/s"),
]

OFFSETS = {
    "balanced-low": {
        "fcfs": (-50, 10),
        "tenant_round_robin": (-65, -24),
        "engine_tenant_fair": (10, 17),
        "prep_max_min": (10, -2),
        "max_min": (10, -18),
        "cross_stage": (10, 16),
    },
    "balanced-high": {
        "fcfs": (-46, -21),
        "tenant_round_robin": (-78, 13),
        "engine_tenant_fair": (10, 16),
        "prep_max_min": (10, -14),
        "max_min": (-91, 14),
        "cross_stage": (10, 12),
    },
}


def latency_values(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    values = [float(row["end_to_end_s"]) for row in rows if not row.get("error")]
    if len(values) != 99:
        raise ValueError(f"expected 99 successful requests in {path}, found {len(values)}")
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
    csv_rows: list[dict[str, object]] = []

    for axis, (load, title) in zip(axes, LOADS):
        for policy, label, marker, color in POLICIES:
            repetitions = []
            for repetition in (1, 2, 3):
                path = (
                    args.root / load / f"repetition-{repetition}" / policy
                    / "results.jsonl"
                )
                values = latency_values(path)
                median = float(np.median(values))
                p99 = float(np.percentile(values, 99))
                repetitions.append((median, p99))
                csv_rows.append({
                    "load": load,
                    "policy": policy,
                    "repetition": repetition,
                    "median_e2e_s": median,
                    "p99_e2e_s": p99,
                })

            points = np.asarray(repetitions)
            x_mean, y_mean = points.mean(axis=0)
            x_std, y_std = points.std(axis=0, ddof=1)
            axis.scatter(
                points[:, 0], points[:, 1], marker=marker, s=65,
                color=color, alpha=0.24, linewidth=0, zorder=2,
            )
            axis.errorbar(
                x_mean, y_mean, xerr=x_std, yerr=y_std, fmt=marker,
                markersize=11, color=color, markeredgecolor="white",
                markeredgewidth=1.1, elinewidth=1.8, capsize=4,
                zorder=4, label=label,
            )
            dx, dy = OFFSETS[load][policy]
            axis.annotate(
                f"{label}\n({x_mean:.1f}, {y_mean:.1f})",
                xy=(x_mean, y_mean), xytext=(dx, dy),
                textcoords="offset points", fontsize=8.7, color=color,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.6},
            )

        axis.set_title(title, fontsize=13.5, fontweight="bold")
        axis.set_xlabel("Median end-to-end latency (s) ↓", fontsize=11.5)
        axis.set_ylabel("p99 end-to-end latency (s) ↓", fontsize=11.5)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.annotate(
            "better", xy=(0.04, 0.05), xytext=(0.18, 0.19),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#374151"},
            color="#374151", fontsize=9, fontweight="bold",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89),
        ncol=6, frameon=False, fontsize=9,
    )
    figure.suptitle(
        "Measured Typical and Tail Latency under Balanced Heterogeneous Requests",
        fontsize=18, fontweight="bold", y=0.985,
    )
    figure.text(
        0.5, 0.025,
        "Each tenant receives the same 8/32/128-frame request mix. "
        "Faint markers are repetitions; large markers are means; error bars are one standard deviation. Lower-left is better.",
        ha="center", fontsize=9.3, color="#4b5563",
    )
    figure.subplots_adjust(
        top=0.79, bottom=0.16, left=0.075, right=0.985, wspace=0.20
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".csv"))


if __name__ == "__main__":
    main()
