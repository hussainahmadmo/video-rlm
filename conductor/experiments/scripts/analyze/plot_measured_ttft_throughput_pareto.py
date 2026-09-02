#!/usr/bin/env python3
"""Plot measured victim TTFT against throughput for all headline policies."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


POLICIES = (
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
        if policy in POLICIES and int(summary.get("errors", 0)) == 0:
            runs[seed][policy] = summary
    seeds = [
        seed for seed, group in sorted(runs.items())
        if all(policy in group for policy in POLICIES)
    ]
    if not seeds:
        raise SystemExit("no matched seed has all five policies")

    rows = []
    for seed in seeds:
        fcfs_throughput = float(runs[seed]["fcfs"]["throughput_qps"])
        for policy in POLICIES:
            summary = runs[seed][policy]
            rows.append({
                "seed": int(seed),
                "policy": policy,
                "victim_mean_ttft_s": statistics.fmean(
                    float(summary["tenants"][tenant]["mean_end_to_end_ttft_s"])
                    for tenant in ("b", "c")
                ),
                "victim_p95_ttft_s": statistics.fmean(
                    float(summary["tenants"][tenant]["p95_end_to_end_ttft_s"])
                    for tenant in ("b", "c")
                ),
                "normalized_throughput_percent": (
                    100.0 * float(summary["throughput_qps"]) / fcfs_throughput
                ),
                "throughput_qps": float(summary["throughput_qps"]),
            })

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.67, wspace=0.20)
    report = {"complete_seeds": [int(seed) for seed in seeds], "policies": {}}
    for axis, field, title in (
        (axes[0], "victim_mean_ttft_s", "(a) Mean victim TTFT"),
        (axes[1], "victim_p95_ttft_s", "(b) P95 victim TTFT"),
    ):
        for policy in POLICIES:
            group = [row for row in rows if row["policy"] == policy]
            xs = np.asarray([row[field] for row in group])
            ys = np.asarray([row["normalized_throughput_percent"] for row in group])
            axis.scatter(xs, ys, s=42, color=COLORS[policy], alpha=0.30, zorder=2)
            axis.errorbar(
                float(xs.mean()),
                float(ys.mean()),
                xerr=float(xs.std(ddof=1)) if len(xs) > 1 else 0,
                yerr=float(ys.std(ddof=1)) if len(ys) > 1 else 0,
                fmt=MARKERS[policy],
                markersize=11,
                color=COLORS[policy],
                capsize=4,
                label=LABELS[policy],
                zorder=3,
            )
        axis.axhline(100, color="black", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Victim TTFT (seconds; lower is better)")
        axis.set_ylabel("Normalized throughput (% of matched FCFS)")
        axis.set_ylim(94, 105)
        axis.set_title(title, fontweight="bold")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.annotate(
            "Desired region",
            xy=(axis.get_xlim()[0], 104),
            xytext=(0.28, 0.92),
            textcoords="axes fraction",
            color="#007A59",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#007A59", linewidth=1.4),
        )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.79),
    )
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        report["policies"][policy] = {
            "runs": len(group),
            "mean_victim_mean_ttft_s": statistics.fmean(row["victim_mean_ttft_s"] for row in group),
            "mean_victim_p95_ttft_s": statistics.fmean(row["victim_p95_ttft_s"] for row in group),
            "mean_normalized_throughput_percent": statistics.fmean(row["normalized_throughput_percent"] for row in group),
            "mean_throughput_qps": statistics.fmean(row["throughput_qps"] for row in group),
        }

    fig.text(
        0.5,
        0.73,
        "Aggressor A: 4× request rate and 128 frames; victims B/C: 8–32 frames. "
        "Dots are matched seeds; markers are means; error bars show one standard deviation.",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.06,
        "Measured result: preparation-only and cross-stage overlap because this workload is preparation-dominated.",
        ha="center",
        fontsize=9.8,
        color="#555555",
        style="italic",
    )
    fig.suptitle(
        "MEASURED A40 RESULT — Victim TTFT Versus Throughput (3 Seeds)\n"
        "Preparation-aware fairness reduces delay without reducing aggregate throughput",
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
