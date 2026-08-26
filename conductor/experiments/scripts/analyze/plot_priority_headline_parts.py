#!/usr/bin/env python3
"""Split the priority-results dashboard into publication-sized figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fcfs", "priority", "priority_reserved")
LABELS = {
    "fcfs": "Engine-only priority",
    "priority": "End-to-end priority",
    "priority_reserved": "End-to-end priority + reserved worker",
}
SHORT_LABELS = {
    "fcfs": "Engine-only",
    "priority": "End-to-end",
    "priority_reserved": "+ reserved worker",
}
COLORS = {
    "fcfs": "#D55E00",
    "priority": "#009E73",
    "priority_reserved": "#0072B2",
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    runs: dict[str, dict[str, dict]] = {}
    rows: dict[str, list[dict]] = {policy: [] for policy in POLICIES}
    for run_dir in sorted(args.suite.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "traces":
            continue
        if not all((run_dir / policy / "summary.json").is_file() for policy in POLICIES):
            continue
        runs[run_dir.name] = {}
        for policy in POLICIES:
            runs[run_dir.name][policy] = json.loads(
                (run_dir / policy / "summary.json").read_text()
            )
            rows[policy].extend(
                load_jsonl(run_dir / policy / "results.jsonl")
            )
    if not runs:
        raise SystemExit("No matched three-policy runs found")

    urgent = {
        policy: [
            row
            for row in rows[policy]
            if row.get("workload") == "urgent" and not row.get("error")
        ]
        for policy in POLICIES
    }

    # Part 1: headline mean and pooled p95 latency.
    means = [
        float(np.mean([row["end_to_end_ttft_s"] for row in urgent[policy]]))
        for policy in POLICIES
    ]
    p95s = [
        float(np.percentile([row["end_to_end_ttft_s"] for row in urgent[policy]], 95))
        for policy in POLICIES
    ]
    x = np.arange(len(POLICIES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    mean_bars = ax.bar(x - width / 2, means, width, color=[COLORS[p] for p in POLICIES])
    p95_bars = ax.bar(
        x + width / 2,
        p95s,
        width,
        color=[COLORS[p] for p in POLICIES],
        alpha=0.48,
        hatch="//",
    )
    ax.bar_label(mean_bars, labels=[f"{v:.1f}s" for v in means], padding=3, fontsize=8)
    ax.bar_label(p95_bars, labels=[f"{v:.1f}s" for v in p95s], padding=3, fontsize=8)
    ax.set_xticks(x, [SHORT_LABELS[p] for p in POLICIES])
    ax.set_ylabel("Urgent end-to-end TTFT (seconds)")
    ax.set_title(
        f"End-to-end priority reduces urgent mean and tail latency\n"
        f"{len(runs)} matched traces; {len(urgent['fcfs'])} urgent requests per policy",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        [mean_bars[0], p95_bars[0]],
        ["Mean", "P95 across urgent requests"],
        frameon=False,
    )
    fig.tight_layout()
    save(fig, args.output_dir, "headline_urgent_latency")

    # Part 2: SLO attainment.
    thresholds = (10, 20, 30, 60, 90, 120)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for policy in POLICIES:
        values = np.array([row["end_to_end_ttft_s"] for row in urgent[policy]])
        attainment = [100 * float(np.mean(values <= threshold)) for threshold in thresholds]
        ax.plot(
            thresholds,
            attainment,
            marker="o",
            linewidth=2,
            color=COLORS[policy],
            label=LABELS[policy],
        )
    ax.set_xlabel("Urgent TTFT objective (seconds)")
    ax.set_ylabel("Requests meeting objective (%)")
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 105)
    ax.set_title("End-to-end priority improves urgent SLO attainment", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, args.output_dir, "headline_slo_attainment")

    # Part 3: per-trace benefit distribution.
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for policy in ("priority", "priority_reserved"):
        speedups = np.array(
            [
                runs[name]["fcfs"]["urgent"]["mean_end_to_end_ttft_s"]
                / runs[name][policy]["urgent"]["mean_end_to_end_ttft_s"]
                for name in runs
            ]
        )
        ordered = np.sort(speedups)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        wins = 100 * float(np.mean(speedups > 1))
        median = float(np.median(speedups))
        ax.plot(
            ordered,
            cdf,
            linewidth=2.2,
            color=COLORS[policy],
            label=f"{LABELS[policy]} (wins={wins:.1f}%, median={median:.2f}$\\times$)",
        )
    ax.axvline(1, color="#333333", linestyle="--", linewidth=1, label="No benefit")
    ax.set_xlabel("Urgent mean-TTFT speedup over engine-only priority")
    ax.set_ylabel("Fraction of matched traces")
    ax.set_title("Priority benefit across matched workloads", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, args.output_dir, "headline_benefit_distribution")

    # Part 4: throughput, background latency, and accuracy cost.
    throughput = [
        float(np.mean([runs[name][policy]["throughput_qps"] for name in runs]))
        for policy in POLICIES
    ]
    background_ttft = [
        float(
            np.mean(
                [
                    runs[name][policy]["background"]["mean_end_to_end_ttft_s"]
                    for name in runs
                ]
            )
        )
        for policy in POLICIES
    ]
    accuracy = [
        100 * float(np.mean([bool(row.get("correct")) for row in urgent[policy]]))
        for policy in POLICIES
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for ax, metric, title, ylabel, fmt in (
        (axes[0], throughput, "Throughput", "Requests/second", ".3f"),
        (axes[1], background_ttft, "Background latency", "Mean TTFT (seconds)", ".1f"),
        (axes[2], accuracy, "Urgent accuracy", "Accuracy (%)", ".1f"),
    ):
        bars = ax.bar(x, metric, color=[COLORS[p] for p in POLICIES])
        ax.bar_label(bars, labels=[format(value, fmt) for value in metric], padding=3, fontsize=8)
        ax.set_xticks(x, [SHORT_LABELS[p] for p in POLICIES], rotation=18, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(bottom=0)
    fig.suptitle("Cost of urgent-request protection", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, args.output_dir, "headline_system_cost")


if __name__ == "__main__":
    main()
