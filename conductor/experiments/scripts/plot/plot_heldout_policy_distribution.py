#!/usr/bin/env python3
"""Plot learned-policy selections for the clean held-out contention run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    2: "#2F6BFF",
    8: "#41B883",
    16: "#F3A712",
    32: "#E45756",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text().splitlines()
        if line.strip()
    ]
    counts = Counter(str(row["config_name"]) for row in rows)
    budgets = {
        name: int(next(row["vlm_budget"] for row in rows if row["config_name"] == name))
        for name in counts
    }
    ordered = sorted(counts, key=lambda name: (counts[name], name))
    values = [counts[name] for name in ordered]
    total = len(rows)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    bars = ax.barh(
        ordered,
        values,
        color=[COLORS[budgets[name]] for name in ordered],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, value in zip(bars, values):
        ax.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({100 * value / total:.1f}%)",
            va="center",
            fontsize=10,
        )

    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=f"Budget {budget}")
        for budget, color in COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2)
    ax.set_title(
        "Learned Adaptive Policy Choices on 138 Held-Out Queries",
        fontsize=16,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Selected policy")
    ax.set_xlim(0, max(values) * 1.22)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
