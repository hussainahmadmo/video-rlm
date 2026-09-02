#!/usr/bin/env python3
"""Plot latency and throughput tradeoffs for multi-tenant scheduling policies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = (
    "fcfs", "engine_tenant_fair", "max_min", "priority", "sjf",
    "tenant_fair", "tenant_priority", "fair_slowdown",
)
LABELS = {
    "fcfs": "FCFS",
    "engine_tenant_fair": "Inference-only\nfairness",
    "priority": "Strict\npriority",
    "sjf": "Shortest\njob first",
    "max_min": "Max-min\nfairness",
    "tenant_fair": "Tenant\nfairness",
    "tenant_priority": "Tenant-fair\npriority",
    "fair_slowdown": "Fair\nslowdown",
}
COLORS = {
    "fcfs": "#D55E00",
    "engine_tenant_fair": "#999999",
    "priority": "#009E73",
    "sjf": "#E69F00",
    "max_min": "#56B4E9",
    "tenant_fair": "#7A5195",
    "tenant_priority": "#CC79A7",
    "fair_slowdown": "#0072B2",
}


def load_summaries(root: Path) -> dict[str, dict[str, dict]]:
    values: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(path.read_text())
        if int(summary.get("errors", 0)) != 0:
            continue
        trace = path.parents[1].name
        policy = path.parent.name
        if policy in POLICIES:
            results = [
                json.loads(line)
                for line in path.with_name("results.jsonl").read_text().splitlines()
                if line.strip()
            ]
            by_tenant: dict[str, list[float]] = defaultdict(list)
            for row in results:
                if not row.get("error") and row.get("estimated_slowdown"):
                    by_tenant[str(row.get("tenant", "default"))].append(
                        float(row["estimated_slowdown"])
                    )
            qualities = [
                1.0 / float(np.mean(slowdowns))
                for slowdowns in by_tenant.values() if slowdowns
            ]
            summary["tenant_quality_jain"] = (
                sum(qualities) ** 2
                / (len(qualities) * sum(value ** 2 for value in qualities))
                if qualities and any(qualities) else None
            )
            values[policy][trace] = summary
    return values


def get_metric(summary: dict, section: str | None, field: str) -> float:
    value = summary[field] if section is None else summary[section][field]
    return float(value)


def matched_fcfs_speedup(
    values: dict[str, dict[str, dict]],
    policy: str,
    section: str,
    field: str,
) -> float | None:
    matched = sorted(set(values.get("fcfs", {})) & set(values.get(policy, {})))
    if not matched:
        return None
    baseline = np.mean([
        get_metric(values["fcfs"][trace], section, field) for trace in matched
    ])
    candidate = np.mean([
        get_metric(values[policy][trace], section, field) for trace in matched
    ])
    return float(baseline / candidate) if candidate else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    values = load_summaries(args.root)
    if not values:
        raise SystemExit("No successful summaries found")

    completed = sum(len(rows) for rows in values.values())
    policies = [policy for policy in POLICIES if policy in values]
    trace_count = len(list((args.root / "traces").glob("*.jsonl")))
    expected = trace_count * len(policies) if trace_count else completed
    fig, axes_grid = plt.subplots(2, 3, figsize=(14.8, 8.2))
    axes = axes_grid.ravel()
    panels = (
        ("urgent", "mean_end_to_end_s", "Foreground mean E2E (s)", "seconds"),
        ("urgent", "p95_end_to_end_s", "Foreground p95 E2E (s)", "seconds"),
        (
            "urgent", "ttft_slo_attainment_percent",
            "Foreground TTFT SLO attainment (%)", "percent",
        ),
        (
            "background", "mean_end_to_end_s",
            "Background mean E2E (s)", "seconds",
        ),
        (None, "tenant_quality_jain", "Tenant Jain fairness", "fairness"),
        (None, "throughput_qps", "Throughput (requests/s)", "qps"),
    )
    x = np.arange(len(policies))

    for panel_index, (ax, (section, field, ylabel, value_format)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        for index, policy in enumerate(policies):
            rows = values.get(policy, {})
            samples = [
                get_metric(summary, section, field)
                for summary in rows.values()
            ]
            if not samples:
                continue
            mean = float(np.mean(samples))
            ax.bar(
                index,
                mean,
                width=0.68,
                color=COLORS[policy],
                alpha=0.82,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )
            offsets = np.linspace(-0.16, 0.16, len(samples)) if len(samples) > 1 else [0]
            ax.scatter(
                index + np.asarray(offsets),
                samples,
                s=30,
                facecolors="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
            if value_format == "seconds":
                value_label = f"{mean:.1f}s"
            elif value_format == "percent":
                value_label = f"{mean:.1f}%"
            elif value_format == "fairness":
                value_label = f"{mean:.2f}"
            else:
                value_label = f"{mean:.3f}"
            ax.annotate(
                value_label,
                (index, max(samples)),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
            if panel_index == 0 and policy != "fcfs":
                speedup = matched_fcfs_speedup(
                    values, policy, "urgent", "mean_end_to_end_s"
                )
                if speedup is not None:
                    ax.text(
                        index,
                        mean * 0.52,
                        f"{speedup:.2f}x",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                        fontweight="bold",
                        zorder=4,
                    )
        ax.set_xticks(x, [LABELS[policy] for policy in policies])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.tick_params(axis="x", labelsize=9)

    axes[0].set_title("(a) Foreground responsiveness", fontweight="bold")
    axes[1].set_title("(b) Foreground tail latency", fontweight="bold")
    axes[2].set_title("(c) Foreground SLO", fontweight="bold")
    axes[3].set_title("(d) Background cost", fontweight="bold")
    axes[4].set_title("(e) Tenant isolation", fontweight="bold")
    axes[5].set_title("(f) Aggregate efficiency", fontweight="bold")
    axes[4].set_ylim(0, 1.08)
    fig.suptitle(
        "Application priority, job cost, and tenant fairness are distinct objectives\n"
        f"{completed}/{expected} runs complete; bars are means and dots are independent seeds",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        "Equal tenant weights; mixed 8/32/128-frame requests; 4 preparation workers. "
        "Speedups in panel (a) use each policy's currently matched FCFS seeds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
