#!/usr/bin/env python3
"""Plot the completed portions of the two-GPU ASPLOS evaluation campaign."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "fcfs": "#D55E00",
    "priority": "#009E73",
    "priority_reserved": "#0072B2",
    "slo_adaptive": "#CC79A7",
    "shared_engine_only": "#D55E00",
    "shared_priority": "#009E73",
    "static_isolation": "#6B7280",
}
LABELS = {
    "fcfs": "Engine-only priority",
    "priority": "End-to-end priority",
    "priority_reserved": "+ reserved worker",
    "slo_adaptive": "SLO-adaptive",
    "shared_engine_only": "Shared, engine-only",
    "shared_priority": "Shared, end-to-end",
    "static_isolation": "Static isolation",
}
MARKERS = {
    "fcfs": "o",
    "priority": "s",
    "priority_reserved": "^",
}


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def load_runs(root: Path, phase: str) -> list[tuple[str, str, dict]]:
    return [
        (path.parents[1].name, path.parent.name, read_summary(path))
        for path in sorted((root / phase).glob("*/*/summary.json"))
    ]


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Return mean and a two-sided 95% Student-t confidence half-width."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    # Every plotted experimental point currently has five matched seeds.
    # Include nearby values so the helper remains useful for partial plots.
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
           6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    critical = t95.get(len(values), 1.96)
    return mean, critical * statistics.stdev(values) / math.sqrt(len(values))


def metric(rows: list[tuple[str, str, dict]], policy: str, field: str) -> list[float]:
    return [
        float(data["urgent"][field])
        for _, run_policy, data in rows
        if run_policy == policy
    ]


def save(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output = prefix.with_suffix(suffix)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    repeated = load_runs(args.root, "repeated_backlog_stress")
    fairness = load_runs(args.root, "fairness_aging")
    isolation = load_runs(args.root, "static_isolation")

    # Only plot isolation cases for which all three policies completed.
    isolation_by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for case, policy, data in isolation:
        isolation_by_case[case][policy] = data
    isolation_policies = {
        "shared_engine_only", "shared_priority", "static_isolation"
    }
    matched_isolation = {
        case: policies
        for case, policies in isolation_by_case.items()
        if isolation_policies <= policies.keys()
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))

    # (a) Backlog scaling, five matched seeds per point.
    ax = axes[0]
    backlogs = sorted({
        int(re.search(r"backlog(\d+)", case).group(1))
        for case, _, _ in repeated
    })
    for policy in ("fcfs", "priority", "priority_reserved"):
        means, errors = [], []
        for backlog in backlogs:
            values = [
                float(data["urgent"]["mean_end_to_end_ttft_s"])
                for case, run_policy, data in repeated
                if run_policy == policy and case.startswith(f"backlog{backlog}-")
            ]
            mean, error = mean_ci(values)
            means.append(mean)
            errors.append(error)
        ax.errorbar(
            backlogs, means, yerr=errors, color=COLORS[policy],
            marker=MARKERS[policy], markersize=6, linewidth=2.2,
            capsize=3, label=LABELS[policy],
        )
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(backlogs, [str(value) for value in backlogs])
    ax.set_xlabel("Low-priority background backlog")
    ax.set_ylabel("Urgent mean TTFT (seconds, log scale)")
    ax.set_title("(a) Repeated backlog stress", loc="left", fontweight="bold")
    ax.grid(alpha=0.22, which="both")
    fcfs_128 = statistics.mean([
        d["urgent"]["mean_end_to_end_ttft_s"] for c, p, d in repeated
        if c.startswith("backlog128-") and p == "fcfs"
    ])
    priority_128 = statistics.mean([
        d["urgent"]["mean_end_to_end_ttft_s"] for c, p, d in repeated
        if c.startswith("backlog128-") and p == "priority"
    ])
    ax.text(
        0.97, 0.05, f"At backlog 128:\n{fcfs_128 / priority_128:.1f}× faster",
        transform=ax.transAxes, ha="right", va="bottom", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#999999"},
    )

    # (b) Continuous arrival processes, five seeds per bar.
    ax = axes[1]
    arrival_types = ("continuous-poisson", "continuous-staggered")
    policies = ("fcfs", "priority", "priority_reserved", "slo_adaptive")
    x = np.arange(len(arrival_types))
    width = 0.19
    for index, policy in enumerate(policies):
        means, errors = [], []
        for arrival in arrival_types:
            values = [
                float(data["urgent"]["mean_end_to_end_ttft_s"])
                for case, run_policy, data in fairness
                if run_policy == policy and case.startswith(arrival + "-seed")
            ]
            mean, error = mean_ci(values)
            means.append(mean)
            errors.append(error)
        position = x + (index - 1.5) * width
        bars = ax.bar(
            position, means, width, yerr=errors, capsize=3,
            color=COLORS[policy], label=LABELS[policy],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in means],
                     padding=3, fontsize=7.5)
    ax.set_xticks(x, ["Random arrivals", "Staggered arrivals"])
    ax.set_ylabel("Urgent mean TTFT (seconds)")
    ax.set_title("(b) Continuous contention", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0, 190)

    # (c) Static isolation comparison, limited to matched completed traces.
    ax = axes[2]
    policies = ("shared_engine_only", "shared_priority", "static_isolation")
    means, errors = [], []
    for policy in policies:
        values = [
            float(run[policy]["urgent"]["mean_end_to_end_ttft_s"])
            for run in matched_isolation.values()
        ]
        mean, error = mean_ci(values)
        means.append(mean)
        errors.append(error)
    bars = ax.bar(
        np.arange(3), means, yerr=errors, capsize=4,
        color=[COLORS[policy] for policy in policies], width=0.68,
    )
    ax.bar_label(bars, labels=[f"{value:.1f}s" for value in means],
                 padding=4, fontsize=9, fontweight="semibold")
    ax.set_xticks(np.arange(3), ["Shared,\nengine-only", "Shared,\nend-to-end",
                                "Dedicated\nreplicas"])
    ax.set_ylabel("Urgent mean TTFT (seconds)")
    ax.set_title(
        f"(c) Static isolation (preliminary, n={len(matched_isolation)})",
        loc="left", fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(bottom=0)
    if matched_isolation:
        ax.text(
            0.98, 0.95,
            f"End-to-end is\n{means[2] / means[1]:.2f}× faster\nthan isolation",
            transform=ax.transAxes, ha="right", va="top", fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#999999"},
        )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(color=COLORS["fcfs"], label="Engine-only priority"),
            Patch(color=COLORS["priority"], label="End-to-end priority"),
            Patch(color=COLORS["priority_reserved"], label="+ reserved worker"),
            Patch(color=COLORS["slo_adaptive"], label="SLO-adaptive"),
            Patch(color=COLORS["static_isolation"], label="Static isolation"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5,
        frameon=False, columnspacing=1.2,
    )
    fig.suptitle(
        "Priority must cover media preparation, not only model execution",
        fontsize=16, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, -0.02,
        "Means across five matched seeds per point; error bars are 95% confidence intervals.\n"
        "Qwen2.5-VL-7B on NVIDIA A40 GPUs; native vLLM priority; four shared media-preparation workers.",
        ha="center", fontsize=9.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.82), w_pad=2.2)
    save(fig, args.output_prefix)


if __name__ == "__main__":
    main()
