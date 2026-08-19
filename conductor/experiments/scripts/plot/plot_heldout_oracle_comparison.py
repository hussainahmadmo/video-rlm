#!/usr/bin/env python3
"""Compute full held-out oracle choices and plot matched policy accuracy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


POLICY_ORDER = [
    "budget2",
    "budget8",
    "scan0.0039_k8_budget32",
    "w4_k8_budget16",
    "w8_k8_budget16",
    "w16_k8_budget16",
    "budget32",
    "scan0.0156_k8_budget32",
    "scan0.03125_k8_budget32",
    "scan0.125_k8_budget32",
    "k1",
    "k4",
    "k16",
    "scan05",
    "long_sparse_k16_w8_budget32",
    "temporal_uniform_k8_w16_budget32",
    "midscan_k12_w12_budget32",
    "local_neighbors_k16_w4_budget32",
]
RANK = {name: index for index, name in enumerate(POLICY_ORDER)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict) -> tuple[str, str]:
    return str(row.get("dataset") or "NExT-QA"), str(row.get("qid"))


def accuracy(path: Path, heldout: set[tuple[str, str]]) -> tuple[int, int]:
    rows = {key(row): row for row in load_jsonl(path) if key(row) in heldout}
    return sum(bool(row.get("correct")) for row in rows.values()), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--completion-results", type=Path, required=True)
    parser.add_argument("--budget2-results", type=Path, required=True)
    parser.add_argument("--budget32-results", type=Path, required=True)
    parser.add_argument("--adaptive-results", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--choices-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()

    heldout_rows = load_jsonl(args.heldout)
    heldout = {key(row) for row in heldout_rows}
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for source in (args.fixed_results, args.completion_results):
        for row in load_jsonl(source):
            if key(row) in heldout and row.get("config_name") in RANK:
                candidates[key(row)].append(row)

    oracle_rows = []
    for item in sorted(heldout):
        correct = [row for row in candidates[item] if row.get("correct")]
        choice = min(correct, key=lambda row: RANK[row["config_name"]]) if correct else None
        oracle_rows.append(
            {
                "dataset": item[0],
                "qid": item[1],
                "oracle_correct": choice is not None,
                "oracle_config": choice.get("config_name") if choice else None,
                "num_candidate_outcomes": len(candidates[item]),
            }
        )

    args.choices_output.write_text(
        "".join(json.dumps(row) + "\n" for row in oracle_rows)
    )
    oracle_correct = sum(row["oracle_correct"] for row in oracle_rows)
    oracle_counts = Counter(
        row["oracle_config"] for row in oracle_rows if row["oracle_config"]
    )

    measured = {}
    for name, path in (
        ("Fixed budget 2", args.budget2_results),
        ("Fixed budget 32", args.budget32_results),
        ("Learned adaptive", args.adaptive_results),
    ):
        correct, total = accuracy(path, heldout)
        measured[name] = {"correct": correct, "total": total, "accuracy_pct": 100 * correct / total}
    measured["Oracle"] = {
        "correct": oracle_correct,
        "total": len(heldout),
        "accuracy_pct": 100 * oracle_correct / len(heldout),
    }

    summary = {
        "heldout_examples": len(heldout),
        "methods": measured,
        "oracle_choice_counts": dict(oracle_counts),
        "oracle_unsolved": len(heldout) - oracle_correct,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    labels = list(measured)
    values = [measured[label]["accuracy_pct"] for label in labels]
    colors = ["#8B95A5", "#E45756", "#2F6BFF", "#41B883"]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.bar(labels, values, color=colors, width=0.66)
    for bar, label in zip(bars, labels):
        result = measured[label]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{result['accuracy_pct']:.2f}%\n({result['correct']}/{result['total']})",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    ax.set_ylim(0, max(values) + 12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy on 138 Strictly Held-Out VideoQA Queries", weight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.plot_output, dpi=240, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
