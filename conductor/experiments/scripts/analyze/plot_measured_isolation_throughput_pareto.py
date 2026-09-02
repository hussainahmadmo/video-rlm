#!/usr/bin/env python3
"""Plot measured victim protection against throughput for headline runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


POLICY_ORDER = (
    "fcfs",
    "tenant_round_robin",
    "prep_max_min",
    "engine_tenant_fair",
    "max_min",
)
LABELS = {
    "fcfs": "FCFS",
    "tenant_round_robin": "Tenant RR",
    "prep_max_min": "Prep-only",
    "engine_tenant_fair": "Inference-only",
    "max_min": "Cross-stage (ours)",
}
COLORS = {
    "fcfs": "#D55E00",
    "tenant_round_robin": "#CC79A7",
    "prep_max_min": "#E69F00",
    "engine_tenant_fair": "#56B4E9",
    "max_min": "#009E73",
}
MARKERS = {
    "fcfs": "s",
    "tenant_round_robin": "D",
    "prep_max_min": "o",
    "engine_tenant_fair": "^",
    "max_min": "v",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(args.root.glob("aggressive_asymmetric-seed*/*/summary.json")):
        summary = json.loads(path.read_text())
        policy = str(summary.get("prep_policy") or path.parent.name)
        seed = path.parent.parent.name.rsplit("seed", 1)[-1]
        if policy in POLICY_ORDER and int(summary.get("errors", 0)) == 0:
            runs[seed][policy] = summary

    complete_seeds = [
        seed for seed, policies in sorted(runs.items())
        if all(policy in policies for policy in POLICY_ORDER)
    ]
    if not complete_seeds:
        raise SystemExit("no seed has all five completed policies")

    points = []
    for seed in complete_seeds:
        fcfs = runs[seed]["fcfs"]
        fcfs_victim_p95 = statistics.fmean(
            float(fcfs["tenants"][tenant]["p95_end_to_end_ttft_s"])
            for tenant in ("b", "c")
        )
        fcfs_throughput = float(fcfs["throughput_qps"])
        for policy in POLICY_ORDER:
            summary = runs[seed][policy]
            victim_p95 = statistics.fmean(
                float(summary["tenants"][tenant]["p95_end_to_end_ttft_s"])
                for tenant in ("b", "c")
            )
            points.append({
                "seed": int(seed),
                "policy": policy,
                "victim_p95_ttft_s": victim_p95,
                "victim_p95_reduction_percent": 100.0 * (1.0 - victim_p95 / fcfs_victim_p95),
                "throughput_qps": float(summary["throughput_qps"]),
                "normalized_throughput_percent": 100.0 * float(summary["throughput_qps"]) / fcfs_throughput,
            })

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axis = plt.subplots(figsize=(10.2, 7.2))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.72)
    label_offsets = {
        "fcfs": (-2, -0.8),
        "tenant_round_robin": (-19, -0.75),
        "prep_max_min": (-15, 0.7),
        "engine_tenant_fair": (3, 0.7),
        "max_min": (-20, -0.7),
    }
    report = {"complete_seeds": [int(seed) for seed in complete_seeds], "policies": {}}
    for policy in POLICY_ORDER:
        group = [row for row in points if row["policy"] == policy]
        xs = np.asarray([row["victim_p95_reduction_percent"] for row in group])
        ys = np.asarray([row["normalized_throughput_percent"] for row in group])
        axis.scatter(xs, ys, s=48, color=COLORS[policy], alpha=0.32, zorder=2)
        x_mean, y_mean = float(xs.mean()), float(ys.mean())
        axis.errorbar(
            x_mean,
            y_mean,
            xerr=float(xs.std(ddof=1)) if len(xs) > 1 else 0,
            yerr=float(ys.std(ddof=1)) if len(ys) > 1 else 0,
            fmt=MARKERS[policy],
            markersize=12,
            color=COLORS[policy],
            capsize=4,
            zorder=3,
        )
        dx, dy = label_offsets[policy]
        axis.annotate(
            LABELS[policy],
            (x_mean, y_mean),
            xytext=(x_mean + dx, y_mean + dy),
            fontsize=11,
            fontweight="bold" if policy == "max_min" else "normal",
            arrowprops=dict(arrowstyle="-", color=COLORS[policy], alpha=0.7),
        )
        report["policies"][policy] = {
            "runs": len(group),
            "mean_victim_p95_reduction_percent": x_mean,
            "mean_normalized_throughput_percent": y_mean,
            "mean_throughput_qps": statistics.fmean(row["throughput_qps"] for row in group),
        }

    axis.axhline(100, color="black", linestyle="--", linewidth=1.2, label="Matched FCFS throughput")
    axis.annotate(
        "Desired region",
        xy=(84, 103),
        xytext=(63, 104),
        color="#007A59",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#007A59", linewidth=1.4),
    )
    axis.set_xlim(-5, 90)
    axis.set_ylim(94, 105)
    axis.set_xlabel("Victim P95 TTFT reduction versus matched FCFS (%)")
    axis.set_ylabel("Normalized throughput (% of matched FCFS)")
    axis.set_title("Measured isolation benefit versus throughput", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")

    fig.text(
        0.5,
        0.78,
        "Workload: aggressor A sends 4× more 128-frame requests; victims B/C use 8–32 frames. "
        "Dots are matched seeds; markers are means; error bars show one standard deviation.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.065,
        "This x-axis measures observed victim isolation, not preparation/inference service fairness. "
        "The workload is preparation-dominated, so prep-only and cross-stage overlap.",
        ha="center",
        fontsize=9.8,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MEASURED A40 RESULT — Isolation–Efficiency Plane (3 Seeds)\n"
        "Preparation-aware scheduling protects victims without reducing throughput",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=230, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
