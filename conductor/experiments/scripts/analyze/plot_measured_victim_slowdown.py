#!/usr/bin/env python3
"""Plot measured victim slowdown using matched tenant-alone runs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


POLICY_ORDER = ("fcfs", "prep_max_min", "engine_tenant_fair", "max_min")
LABELS = {
    "fcfs": "FCFS",
    "prep_max_min": "Preparation-only",
    "engine_tenant_fair": "Inference-only",
    "max_min": "Cross-stage",
}
COLORS = {"b": "#0072B2", "c": "#E69F00"}


def load_contended(root: Path) -> dict[tuple[int, str], dict]:
    runs = {}
    for path in sorted(root.glob("aggressive_asymmetric-seed*/*/summary.json")):
        match = re.search(r"seed(\d+)", path.as_posix())
        if not match:
            continue
        summary = json.loads(path.read_text())
        policy = str(summary.get("prep_policy") or path.parent.name)
        if policy in POLICY_ORDER:
            runs[(int(match.group(1)), policy)] = summary
    return runs


def load_solo(root: Path) -> dict[tuple[int, str], dict]:
    runs = {}
    pattern = re.compile(r"victim-([bc])-solo-seed(\d+)")
    for path in sorted(root.glob("aggressive_asymmetric-victim-*-solo-seed*/fcfs/summary.json")):
        match = pattern.search(path.as_posix())
        if match:
            runs[(int(match.group(2)), match.group(1))] = json.loads(path.read_text())
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contended-root", type=Path, required=True)
    parser.add_argument("--solo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    contended = load_contended(args.contended_root)
    solo = load_solo(args.solo_root)
    seeds = sorted({seed for seed, _ in solo})
    if not seeds:
        raise SystemExit("no victim-alone summaries found")
    policies = [
        policy for policy in POLICY_ORDER
        if all((seed, policy) in contended for seed in seeds)
    ]
    if "fcfs" not in policies or "max_min" not in policies:
        raise SystemExit("matched FCFS and cross-stage runs are required")

    slowdowns: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for policy in policies:
        for tenant in ("b", "c"):
            for seed in seeds:
                contended_run = contended[(seed, policy)]
                solo_run = solo[(seed, tenant)]
                for metric in ("mean_end_to_end_s", "p95_end_to_end_s"):
                    numerator = float(contended_run["tenants"][tenant][metric])
                    denominator = float(solo_run["tenants"][tenant][metric])
                    slowdowns[metric][policy][tenant].append(
                        numerator / denominator
                    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    rng = np.random.default_rng(11)
    x = np.arange(len(policies), dtype=float)
    width = 0.34
    for axis, (metric, title) in zip(axes, (
        ("mean_end_to_end_s", "Mean victim slowdown"),
        ("p95_end_to_end_s", "P95 victim slowdown"),
    )):
        for tenant_index, tenant in enumerate(("b", "c")):
            positions = x + (-0.5 if tenant_index == 0 else 0.5) * width
            samples_by_policy = [slowdowns[metric][policy][tenant]
                                 for policy in policies]
            means = [statistics.fmean(samples) for samples in samples_by_policy]
            axis.bar(
                positions, means, width=width, color=COLORS[tenant],
                label=f"Victim {tenant.upper()}", edgecolor="white",
                linewidth=0.6, zorder=2,
            )
            for position, samples in zip(positions, samples_by_policy):
                jitter = rng.uniform(-0.035, 0.035, size=len(samples))
                axis.scatter(
                    np.full(len(samples), position) + jitter, samples,
                    color="black", alpha=0.65, s=16, zorder=3,
                )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, [LABELS[policy] for policy in policies])
        axis.set_ylabel("Slowdown vs. tenant alone (×)")
        axis.grid(axis="y", alpha=0.25, zorder=0)
        for tenant_index, tenant in enumerate(("b", "c")):
            cross_stage_mean = statistics.fmean(
                slowdowns[metric]["max_min"][tenant]
            )
            fcfs_mean = statistics.fmean(slowdowns[metric]["fcfs"][tenant])
            reduction_factor = fcfs_mean / cross_stage_mean
            position = x[-1] + (-0.5 if tenant_index == 0 else 0.5) * width
            axis.text(
                position, cross_stage_mean + 0.8,
                f"{reduction_factor:.1f}× reduction",
                ha="center", va="bottom", rotation=90,
                color=COLORS[tenant], fontsize=8, fontweight="bold",
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "Cross-stage max–min reduces measured victim slowdown",
        fontweight="bold", y=1.17,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    report = {
        metric: {
            policy: {
                tenant: {
                    "mean_slowdown_x": statistics.fmean(
                        slowdowns[metric][policy][tenant]
                    ),
                    "per_seed_slowdown_x": slowdowns[metric][policy][tenant],
                }
                for tenant in ("b", "c")
            }
            for policy in policies
        }
        for metric in ("mean_end_to_end_s", "p95_end_to_end_s")
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
