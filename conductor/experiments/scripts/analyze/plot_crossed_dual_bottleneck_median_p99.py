#!/usr/bin/env python3
"""Plot measured median versus p99 E2E latency for crossed contention."""

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

OFFSETS = {
    "fcfs": (-82, 16),
    "tenant_round_robin": (13, -25),
    "engine_tenant_fair": (12, 17),
    "prep_max_min": (-104, -23),
    "max_min": (12, 16),
    "cross_stage": (-8, -31),
}


def latency_values(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    errors = sum(bool(row.get("error")) for row in rows)
    values = [float(row["end_to_end_s"]) for row in rows if not row.get("error")]
    if errors or len(values) != 136:
        raise ValueError(
            f"expected 136 successful requests and no errors in {path}; "
            f"found {len(values)} successes and {errors} errors"
        )
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("cross_stage_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(9.8, 7.3))
    csv_rows: list[dict[str, object]] = []

    for policy, label, marker, color in POLICIES:
        root = args.cross_stage_root if policy == "cross_stage" else args.baseline_root
        repetitions = []
        for seed in (1, 2, 3):
            path = (
                root / f"crossed_dual_bottleneck-seed{seed}" / policy
                / "results.jsonl"
            )
            values = latency_values(path)
            median = float(np.median(values))
            p99 = float(np.percentile(values, 99))
            repetitions.append((median, p99))
            csv_rows.append({
                "policy": policy,
                "seed": seed,
                "median_e2e_s": median,
                "p99_e2e_s": p99,
                "result_suite": root.name,
            })

        points = np.asarray(repetitions)
        x_mean, y_mean = points.mean(axis=0)
        x_std, y_std = points.std(axis=0, ddof=1)
        axis.scatter(
            points[:, 0], points[:, 1], marker=marker, s=75,
            color=color, alpha=0.24, linewidth=0, zorder=2,
        )
        axis.errorbar(
            x_mean, y_mean, xerr=x_std, yerr=y_std, fmt=marker,
            markersize=12, color=color, markeredgecolor="white",
            markeredgewidth=1.1, elinewidth=1.8, capsize=4,
            zorder=4, label=label,
        )
        dx, dy = OFFSETS[policy]
        axis.annotate(
            f"{label}\n({x_mean:.1f}, {y_mean:.1f})",
            xy=(x_mean, y_mean), xytext=(dx, dy),
            textcoords="offset points", fontsize=9, color=color,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.6},
        )

    axis.set_xlabel("Median end-to-end latency (s) ↓", fontsize=12)
    axis.set_ylabel("p99 end-to-end latency (s) ↓", fontsize=12)
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.annotate(
        "better", xy=(0.04, 0.05), xytext=(0.17, 0.17),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        color="#374151", fontsize=10, fontweight="bold",
    )
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
        frameon=False, fontsize=9.5,
    )

    figure.suptitle(
        "Measured Typical and Tail Latency under Crossed CPU–GPU Contention",
        fontsize=17, fontweight="bold", y=0.985,
    )
    figure.text(
        0.5, 0.055,
        "A: 128 frames, short output; B: 32 frames, long output; "
        "C: low-rate, 8 frames, short output. Same 136-request trace; three runs per policy.",
        ha="center", fontsize=9.2, color="#4b5563",
    )
    figure.text(
        0.5, 0.025,
        "Faint markers are runs; large markers are means; error bars show one standard deviation. "
        "Lower-left is better.",
        ha="center", fontsize=9.2, color="#4b5563",
    )
    figure.subplots_adjust(top=0.78, bottom=0.16, left=0.12, right=0.97)

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
