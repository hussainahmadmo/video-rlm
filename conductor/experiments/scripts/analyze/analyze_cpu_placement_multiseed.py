#!/usr/bin/env python3
"""Aggregate repeated CPU/GPU preparation-placement comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = (
    "cpu_only_same_pool",
    "cpu_only_equal_slots",
    "static_split_fcfs",
    "adaptive_placement_fcfs",
)
LABELS = {
    "cpu_only_same_pool": "CPU only\n(4 workers)",
    "cpu_only_equal_slots": "CPU only\n(6 workers)",
    "static_split_fcfs": "Static\nCPU/GPU",
    "adaptive_placement_fcfs": "Adaptive\nCPU/GPU",
}
COLORS = {
    "cpu_only_same_pool": "#9E9E9E",
    "cpu_only_equal_slots": "#5F6368",
    "static_split_fcfs": "#56B4E9",
    "adaptive_placement_fcfs": "#0072B2",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("seed*/comparison.json")):
        seed = int(path.parent.name[4:])
        if not (path.parent / "COMPLETE").is_file():
            raise SystemExit(f"incomplete seed: {path.parent}")
        variants = {row["variant"]: row for row in json.loads(path.read_text())}
        if set(variants) != set(POLICIES):
            raise SystemExit(f"unexpected variants in {path}: {sorted(variants)}")
        for policy in POLICIES:
            source = variants[policy]
            result = {
                "seed": seed,
                "policy": policy,
                "mean_e2e_s": float(source["all"]["end_to_end_s"]["mean"]),
                "median_e2e_s": float(source["all"]["end_to_end_s"]["median"]),
                "p95_e2e_s": float(source["all"]["end_to_end_s"]["p95"]),
                "throughput_qps": float(source["throughput_qps"]),
            }
            for frames in (1, 16, 128):
                result[f"median_{frames}_frames_s"] = float(
                    source["sizes"][f"{frames}_frames"]["end_to_end_s"]["median"]
                )
            rows.append(result)
    seeds = sorted({row["seed"] for row in rows})
    expected = {(seed, policy) for seed in seeds for policy in POLICIES}
    observed = {(row["seed"], row["policy"]) for row in rows}
    if len(seeds) < 3 or observed != expected:
        raise SystemExit(f"incomplete matrix; seeds={seeds}, missing={sorted(expected-observed)}")
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    fields = (
        "mean_e2e_s",
        "median_e2e_s",
        "p95_e2e_s",
        "throughput_qps",
        "median_1_frames_s",
        "median_16_frames_s",
        "median_128_frames_s",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    output = []
    for policy in POLICIES:
        group = sorted(grouped[policy], key=lambda row: row["seed"])
        item = {"policy": policy, "runs": len(group)}
        for field in fields:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_std"] = statistics.stdev(values)
        output.append(item)
    return output


def paired_changes(rows: list[dict]) -> dict:
    by_key = {(row["seed"], row["policy"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    output = {}
    comparisons = {
        "static_vs_equal_slot_cpu": ("static_split_fcfs", "cpu_only_equal_slots"),
        "adaptive_vs_same_pool_cpu": ("adaptive_placement_fcfs", "cpu_only_same_pool"),
        "adaptive_vs_equal_slot_cpu": ("adaptive_placement_fcfs", "cpu_only_equal_slots"),
        "adaptive_vs_static": ("adaptive_placement_fcfs", "static_split_fcfs"),
    }
    for name, (candidate, baseline) in comparisons.items():
        output[name] = {}
        for field in ("mean_e2e_s", "median_e2e_s", "p95_e2e_s"):
            values = [
                100.0
                * (
                    1.0
                    - by_key[(seed, candidate)][field]
                    / by_key[(seed, baseline)][field]
                )
                for seed in seeds
            ]
            output[name][f"{field}_reduction_pct_mean"] = statistics.fmean(values)
            output[name][f"{field}_reduction_pct_std"] = statistics.stdev(values)
        throughput = [
            100.0
            * (
                by_key[(seed, candidate)]["throughput_qps"]
                / by_key[(seed, baseline)]["throughput_qps"]
                - 1.0
            )
            for seed in seeds
        ]
        output[name]["throughput_gain_pct_mean"] = statistics.fmean(throughput)
        output[name]["throughput_gain_pct_std"] = statistics.stdev(throughput)
    return output


def plot(aggregates: list[dict], output: Path) -> None:
    lookup = {row["policy"]: row for row in aggregates}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))

    metrics = (("median_e2e_s", "Median"), ("p95_e2e_s", "p95"))
    x = np.arange(len(metrics), dtype=float)
    width = 0.19
    for index, policy in enumerate(POLICIES):
        means = [lookup[policy][f"{field}_mean"] for field, _ in metrics]
        stds = [lookup[policy][f"{field}_std"] for field, _ in metrics]
        axes[0].bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            color=COLORS[policy],
            label=LABELS[policy].replace("\n", " "),
        )
    axes[0].set_xticks(x, [label for _, label in metrics])
    axes[0].set_ylabel("End-to-end latency (s)")
    axes[0].set_title("(a) Overall latency", fontweight="bold", fontsize=9.5)

    frames = np.array([1, 16, 128])
    positions = np.arange(len(frames), dtype=float)
    for policy in POLICIES:
        means = [lookup[policy][f"median_{frame}_frames_s_mean"] for frame in frames]
        stds = [lookup[policy][f"median_{frame}_frames_s_std"] for frame in frames]
        axes[1].errorbar(
            positions,
            means,
            yerr=stds,
            marker="o",
            capsize=2,
            linewidth=1.5,
            color=COLORS[policy],
            label=LABELS[policy].replace("\n", " "),
        )
    axes[1].set_xticks(positions, [str(frame) for frame in frames])
    axes[1].set_xlabel("Input frames")
    axes[1].set_ylabel("Median end-to-end latency (s)")
    axes[1].set_title("(b) Latency by request size", fontweight="bold", fontsize=9.5)

    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = discover(args.input)
    aggregates = aggregate(runs)
    changes = paired_changes(runs)
    write_csv(args.output.with_name(args.output.name + "_runs.csv"), runs)
    write_csv(args.output.with_name(args.output.name + "_aggregate.csv"), aggregates)
    args.output.with_name(args.output.name + "_paired_changes.json").write_text(
        json.dumps(changes, indent=2) + "\n"
    )
    plot(aggregates, args.output)
    for row in aggregates:
        print(
            f"{row['policy']:26s} mean={row['mean_e2e_s_mean']:.2f}s "
            f"median={row['median_e2e_s_mean']:.2f}s p95={row['p95_e2e_s_mean']:.2f}s "
            f"throughput={row['throughput_qps_mean']:.3f} req/s"
        )
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    main()
