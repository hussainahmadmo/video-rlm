#!/usr/bin/env python3
"""Aggregate matched FCFS/priority trace trials and plot 95% confidence intervals."""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
        14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
        20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
        26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}


def ci95(values):
    n = len(values)
    avg = statistics.mean(values)
    if n < 2:
        return avg, None
    t = T975.get(n, 1.96)
    return avg, t * statistics.stdev(values) / math.sqrt(n)


def workload_key(name):
    return re.sub(r"[-_]seed\d+$", "", name)


def load_summaries(suite):
    records = []
    for summary_path in suite.glob("*/*/summary.json"):
        policy = summary_path.parent.name
        if policy not in ("fcfs", "priority"):
            continue
        data = json.loads(summary_path.read_text())
        records.append((workload_key(summary_path.parent.parent.name), policy, data))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path)
    args = p.parse_args()
    prefix = args.output_prefix or args.suite / "priority_trace_ci"

    metrics = {
        "urgent_mean_ttft_s": lambda d: d["urgent"]["mean_end_to_end_ttft_s"],
        "urgent_p95_ttft_s": lambda d: d["urgent"]["p95_end_to_end_ttft_s"],
        "urgent_mean_prep_wait_s": lambda d: d["urgent"]["mean_prep_queue_wait_s"],
        "background_mean_ttft_s": lambda d: d["background"]["mean_end_to_end_ttft_s"],
        "background_p95_ttft_s": lambda d: d["background"]["p95_end_to_end_ttft_s"],
        "throughput_qps": lambda d: d["throughput_qps"],
        "urgent_accuracy_percent": lambda d: d["urgent"]["accuracy_percent"],
        "background_accuracy_percent": lambda d: d["background"]["accuracy_percent"],
    }
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for workload, policy, data in load_summaries(args.suite):
        for metric, getter in metrics.items():
            value = getter(data)
            if value is not None:
                grouped[workload][policy][metric].append(float(value))

    rows = []
    for workload in sorted(grouped):
        if not all(policy in grouped[workload] for policy in ("fcfs", "priority")):
            continue
        row = {"workload": workload}
        for policy in ("fcfs", "priority"):
            for metric in metrics:
                values = grouped[workload][policy][metric]
                avg, half = ci95(values)
                row[f"{policy}_{metric}_mean"] = avg
                row[f"{policy}_{metric}_ci95"] = half
                row[f"{policy}_{metric}_n"] = len(values)
        f = row["fcfs_urgent_mean_ttft_s_mean"]
        q = row["priority_urgent_mean_ttft_s_mean"]
        row["urgent_ttft_reduction_percent"] = 100.0 * (1.0 - q / f)
        row["urgent_ttft_speedup"] = f / q
        rows.append(row)

    if not rows:
        raise SystemExit("no complete matched FCFS/priority trials found")

    csv_path = prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        x = list(range(len(rows)))
        labels = [row["workload"] for row in rows]
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6))
        for policy, label, color in (
            ("fcfs", "vLLM priority + FCFS media preparation", "#d95f02"),
            ("priority", "End-to-end priority", "#1b9e77"),
        ):
            means = [row[f"{policy}_urgent_mean_ttft_s_mean"] for row in rows]
            cis = [row[f"{policy}_urgent_mean_ttft_s_ci95"] or 0 for row in rows]
            ax.errorbar(x, means, yerr=cis, marker="o", capsize=4,
                        linewidth=2, label=label, color=color)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("Urgent mean end-to-end TTFT (s)")
        ax.set_xlabel("Arrival workload")
        ax.set_title("Cross-stage priority protects urgent video requests")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(prefix.with_suffix(".png"), dpi=200)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; wrote CSV only")

    print(json.dumps({
        "matched_workloads": len(rows),
        "csv": str(csv_path),
        "plot": str(prefix.with_suffix(".png")),
    }, indent=2))


if __name__ == "__main__":
    main()
