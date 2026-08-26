#!/usr/bin/env python3
"""Plot urgent TTFT for matched FCFS, priority, and reserved-priority runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fcfs", "priority", "priority_reserved")
T975 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
    12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
    17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
    22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
    27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
}
LABELS = {
    "fcfs": "FCFS media prep",
    "priority": "Priority media prep",
    "priority_reserved": "Priority + reserved worker",
}
COLORS = {
    "fcfs": "#D55E00",
    "priority": "#009E73",
    "priority_reserved": "#0072B2",
}


def family(name: str) -> str:
    if name.startswith("burst_"):
        return "Burst"
    if name.startswith("staggered_"):
        return "Staggered"
    match = re.match(r"poisson_rate([0-9.]+)-", name)
    if match:
        return f"Poisson {match.group(1)} QPS"
    return "Other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    matched = 0

    for run in args.suite.iterdir():
        if not run.is_dir():
            continue
        summaries = [run / policy / "summary.json" for policy in POLICIES]
        if not all(path.is_file() for path in summaries):
            continue
        matched += 1
        group = family(run.name)
        for policy, path in zip(POLICIES, summaries, strict=True):
            data = json.loads(path.read_text())
            values[group][policy].append(
                float(data["urgent"]["mean_end_to_end_ttft_s"])
            )

    if not matched:
        raise SystemExit("No matched three-policy runs found")

    order = [
        name for name in (
            "Burst", "Staggered", "Poisson 0.25 QPS", "Poisson 0.5 QPS",
            "Poisson 0.75 QPS", "Poisson 1.0 QPS",
        ) if name in values
    ]
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6.5))

    for index, policy in enumerate(POLICIES):
        means = [np.mean(values[group][policy]) for group in order]
        errors = []
        for group in order:
            samples = values[group][policy]
            if len(samples) > 1:
                t_value = T975.get(len(samples), 1.96)
                errors.append(
                    t_value * np.std(samples, ddof=1) / math.sqrt(len(samples))
                )
            else:
                errors.append(0.0)
        bars = ax.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=LABELS[policy],
            color=COLORS[policy],
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    counts = [len(values[group]["fcfs"]) for group in order]
    ax.set_xticks(x, [f"{group}\n(n={count})" for group, count in zip(order, counts)])
    ax.set_ylabel("Urgent mean end-to-end TTFT (seconds)")
    ax.set_title(
        f"End-to-end media-preparation priority across arrival patterns\n"
        f"{matched} matched traces; bars are means, error bars are 95% CIs"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
