#!/usr/bin/env python3
"""Plot tenant-latency gains and throughput cost for the aggressive workload."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


POLICY_ORDER = (
    "fcfs", "tenant_round_robin", "prep_max_min",
    "engine_tenant_fair", "max_min",
)
LABELS = {
    "fcfs": "FCFS",
    "tenant_round_robin": "Tenant RR",
    "prep_max_min": "Preparation-only",
    "engine_tenant_fair": "Inference-only",
    "max_min": "Cross-stage",
}
COLORS = {
    "fcfs": "#D55E00",
    "tenant_round_robin": "#009E73",
    "prep_max_min": "#CC79A7",
    "engine_tenant_fair": "#999999",
    "max_min": "#0072B2",
}


def load_runs(root: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(root.glob("aggressive_asymmetric-seed*/*/summary.json")):
        summary = json.loads(path.read_text())
        policy = str(summary.get("prep_policy") or path.parent.name)
        if policy in POLICY_ORDER:
            runs[policy].append(summary)
    missing = [policy for policy in ("fcfs", "max_min") if not runs[policy]]
    if missing:
        raise SystemExit(f"missing completed policies: {', '.join(missing)}")
    return runs


def values(runs: dict[str, list[dict]], policy: str, key: str) -> list[float]:
    if key == "throughput_qps":
        return [float(run[key]) for run in runs[policy]]
    tenant, metric = key.split(".", 1)
    return [float(run["tenants"][tenant][metric]) for run in runs[policy]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    runs = load_runs(args.root)
    policies = [policy for policy in POLICY_ORDER if runs.get(policy)]
    fig, axes = plt.subplots(
        1, 3, figsize=(12.8, 3.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]},
    )
    rng = np.random.default_rng(7)

    tenant_specs = (
        ("a", "Aggressor A"),
        ("b", "Victim B"),
        ("c", "Victim C"),
    )
    x = np.arange(len(tenant_specs), dtype=float)
    width = 0.72 / len(policies)

    for panel, (metric, title) in enumerate((
        ("mean_end_to_end_ttft_s", "Mean end-to-end TTFT"),
        ("p95_end_to_end_ttft_s", "P95 end-to-end TTFT"),
    )):
        axis = axes[panel]
        for index, policy in enumerate(policies):
            positions = x + (index - (len(policies) - 1) / 2) * width
            means = []
            all_samples = []
            for tenant, _ in tenant_specs:
                samples = values(runs, policy, f"{tenant}.{metric}")
                means.append(statistics.fmean(samples))
                all_samples.append(samples)
            axis.bar(
                positions, means, width=width, color=COLORS[policy],
                label=LABELS[policy], edgecolor="white", linewidth=0.6,
                zorder=2,
            )
            for position, samples in zip(positions, all_samples):
                jitter = rng.uniform(-0.035, 0.035, size=len(samples))
                axis.scatter(
                    np.full(len(samples), position) + jitter, samples,
                    s=15, color="black", alpha=0.65, zorder=3,
                )
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, [label for _, label in tenant_specs])
        axis.set_ylabel("Latency (s)")
        axis.grid(axis="y", alpha=0.25, zorder=0)
        for tenant_index, (tenant, _) in enumerate(tenant_specs[1:], start=1):
            fcfs_mean = statistics.fmean(values(
                runs, "fcfs", f"{tenant}.{metric}"
            ))
            cross_stage_mean = statistics.fmean(values(
                runs, "max_min", f"{tenant}.{metric}"
            ))
            speedup = fcfs_mean / cross_stage_mean
            axis.text(
                x[tenant_index]
                + (policies.index("max_min") - (len(policies) - 1) / 2) * width,
                cross_stage_mean + 3.0,
                f"{speedup:.1f}×", ha="center", va="bottom",
                color=COLORS["max_min"], fontsize=9, fontweight="bold",
            )

    fcfs_throughput = statistics.fmean(values(runs, "fcfs", "throughput_qps"))
    normalized = {
        policy: [100.0 * value / fcfs_throughput
                 for value in values(runs, policy, "throughput_qps")]
        for policy in policies
    }
    throughput_axis = axes[2]
    throughput_x = np.arange(len(policies), dtype=float)
    throughput_axis.bar(
        throughput_x,
        [statistics.fmean(normalized[policy]) for policy in policies],
        color=[COLORS[policy] for policy in policies], width=0.68,
        edgecolor="white", linewidth=0.6, zorder=2,
    )
    for position, policy in zip(throughput_x, policies):
        samples = normalized[policy]
        jitter = rng.uniform(-0.05, 0.05, size=len(samples))
        throughput_axis.scatter(
            np.full(len(samples), position) + jitter, samples,
            s=15, color="black", alpha=0.65, zorder=3,
        )
    throughput_axis.axhline(100.0, color="black", linestyle="--", linewidth=1)
    throughput_axis.set_title("Throughput cost", fontweight="bold")
    throughput_axis.set_xticks(
        throughput_x, [LABELS[policy] for policy in policies], rotation=18,
        ha="right",
    )
    throughput_axis.set_ylabel("Normalized throughput (%)")
    throughput_axis.set_ylim(94, 105)
    throughput_axis.grid(axis="y", alpha=0.25, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(policies), frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "PRELIMINARY MEASURED RESULT (3 seeds) — Preparation-aware fairness protects victims\n"
        "Aggressor A: 4× request rate and 128 frames; victims B/C: 8–32 frames",
        fontweight="bold", y=1.16,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    report = {
        "runs_per_policy": {policy: len(runs[policy]) for policy in policies},
        "mean_ttft_reduction_percent": {
            tenant: 100.0 * (
                1.0
                - statistics.fmean(values(
                    runs, "max_min", f"{tenant}.mean_end_to_end_ttft_s"
                )) / statistics.fmean(values(
                    runs, "fcfs", f"{tenant}.mean_end_to_end_ttft_s"
                ))
            )
            for tenant in ("b", "c")
        },
        "p95_ttft_reduction_percent": {
            tenant: 100.0 * (
                1.0
                - statistics.fmean(values(
                    runs, "max_min", f"{tenant}.p95_end_to_end_ttft_s"
                )) / statistics.fmean(values(
                    runs, "fcfs", f"{tenant}.p95_end_to_end_ttft_s"
                ))
            )
            for tenant in ("b", "c")
        },
        "mean_ttft_speedup_x": {
            tenant: statistics.fmean(values(
                runs, "fcfs", f"{tenant}.mean_end_to_end_ttft_s"
            )) / statistics.fmean(values(
                runs, "max_min", f"{tenant}.mean_end_to_end_ttft_s"
            ))
            for tenant in ("b", "c")
        },
        "p95_ttft_speedup_x": {
            tenant: statistics.fmean(values(
                runs, "fcfs", f"{tenant}.p95_end_to_end_ttft_s"
            )) / statistics.fmean(values(
                runs, "max_min", f"{tenant}.p95_end_to_end_ttft_s"
            ))
            for tenant in ("b", "c")
        },
        "throughput_change_percent": 100.0 * (
            statistics.fmean(values(runs, "max_min", "throughput_qps"))
            / fcfs_throughput - 1.0
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
