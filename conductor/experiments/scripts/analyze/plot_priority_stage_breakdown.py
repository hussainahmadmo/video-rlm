#!/usr/bin/env python3
"""Plot mean and p95 TTFT stage decomposition for priority experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

STAGES = (
    ("prep_queue_wait_s", "Preparation queue", "#4c78a8"),
    ("prep_service_s", "Preparation service", "#f58518"),
    ("prepared_queue_wait_s", "Prepared queue", "#eeca3b"),
    ("engine_to_first_token_s", "Engine to first token", "#54a24b"),
)


def load_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def p95(values):
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * 0.95)]) if ordered else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--workload", choices=["urgent", "background"], default="urgent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        parser.error("provide one --label for each --run")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, statistic in zip(axes, ("mean", "p95")):
        bottoms = [0.0] * len(args.run)
        for field, stage_label, color in STAGES:
            values = []
            for run in args.run:
                samples = [
                    float(row[field]) for row in load_rows(run / "results.jsonl")
                    if row.get("workload") == args.workload and row.get(field) is not None
                ]
                values.append(
                    sum(samples) / len(samples)
                    if statistic == "mean" and samples else p95(samples)
                )
            axis.bar(args.label, values, bottom=bottoms, label=stage_label, color=color)
            bottoms = [left + right for left, right in zip(bottoms, values)]
        for index, total in enumerate(bottoms):
            axis.text(index, total, f"{total:.2f}s", ha="center", va="bottom")
        axis.set_title(f"{statistic.upper()} stage components")
        axis.set_ylabel("Seconds")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.suptitle(f"{args.workload.title()} video TTFT decomposition")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
