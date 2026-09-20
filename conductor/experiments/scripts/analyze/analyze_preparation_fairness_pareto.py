#!/usr/bin/env python3
"""Analyze the matched heterogeneous-placement fairness-slack sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def overlap(left: float, right: float, start: float, end: float) -> float:
    return max(0.0, min(right, end) - max(left, start))


def service_amount(row: dict, resource: str) -> float:
    if resource == "cpu":
        recorded = row.get("prep_cpu_worker_s")
        if finite(recorded):
            return float(recorded)
        return float(row["prep_service_s"]) if row.get("prep_backend") == "seek_cpu" else 0.0
    recorded = row.get("prep_gpu_reserved_lane_s")
    if finite(recorded):
        return float(recorded)
    return float(row["prep_service_s"]) * float(row.get("prep_gpu_lanes") or 0)


def joint_backlog_service(rows: list[dict], tenants: list[str], cpu_capacity: int,
                          gpu_capacity: int) -> dict:
    valid = [
        row for row in rows if not row.get("error")
        and all(finite(row.get(field)) for field in
                ("arrival_s", "prep_started_s", "prep_ready_s", "prep_service_s"))
    ]
    boundaries = sorted({
        float(row[field]) for row in valid
        for field in ("arrival_s", "prep_started_s", "prep_ready_s")
    })
    cpu = {tenant: 0.0 for tenant in tenants}
    gpu = {tenant: 0.0 for tenant in tenants}
    qualifying_s = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        if not all(any(
            row["tenant"] == tenant
            and float(row["arrival_s"]) <= midpoint < float(row["prep_ready_s"])
            for row in valid
        ) for tenant in tenants):
            continue
        qualifying_s += right - left
        for row in valid:
            start = float(row["prep_started_s"])
            end = float(row["prep_ready_s"])
            duration = end - start
            if duration <= 0:
                continue
            fraction = overlap(left, right, start, end) / duration
            tenant = str(row["tenant"])
            cpu[tenant] += fraction * service_amount(row, "cpu")
            gpu[tenant] += fraction * service_amount(row, "gpu")
    dominant = {
        tenant: max(cpu[tenant] / cpu_capacity, gpu[tenant] / gpu_capacity)
        for tenant in tenants
    }
    total = sum(dominant.values())
    if qualifying_s <= 0 or total <= 0:
        raise ValueError("no measurable service interval with every tenant backlogged")
    shares = {tenant: dominant[tenant] / total for tenant in tenants}
    gap = max(shares.values()) - min(shares.values())
    return {
        "qualifying_s": qualifying_s,
        "cpu_service": cpu,
        "gpu_lane_service": gpu,
        "dominant_service": dominant,
        "dominant_share": shares,
        "share_gap": gap,
        "service_balance": 1.0 - gap,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def policy_key(policy: str) -> tuple[int, float]:
    if policy == "adaptive_fcfs":
        return (1, math.inf)
    if not policy.startswith("fair_slack_"):
        raise ValueError(f"unexpected policy {policy}")
    return (0, float(policy[len("fair_slack_"):].replace("p", ".")))


def discover(roots: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for root in roots:
        for results_path in sorted(root.glob("seed*/**/results.jsonl")):
            policy = results_path.parent.name
            try:
                policy_key(policy)
                seed_name = results_path.parent.parent.name
                seed = int(seed_name[4:] if seed_name.startswith("seed") else seed_name)
            except ValueError:
                continue
            identity = (seed, policy)
            if identity in seen:
                raise SystemExit(f"duplicate result for seed/policy {identity}")
            seen.add(identity)
            rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
            summary = json.loads(results_path.with_name("summary.json").read_text())
            if len(rows) != 60 or int(summary.get("errors", 0)) != 0:
                raise SystemExit(f"invalid run: {results_path.parent}")
            tenants = sorted({str(row["tenant"]) for row in rows})
            if tenants != ["a", "b", "c"]:
                raise SystemExit(f"unexpected tenants in {results_path}: {tenants}")
            service = joint_backlog_service(
                rows, tenants, int(summary["prep_workers"]),
                int(summary["gpu_decoder_budget"]),
            )
            victim = [
                float(row["completion_s"]) - float(row["arrival_s"])
                for row in rows if row["tenant"] in ("b", "c")
            ]
            record = {
                "seed": seed,
                "policy": policy,
                "slack_s": math.inf if policy == "adaptive_fcfs" else policy_key(policy)[1],
                "throughput_qps": float(summary["throughput_qps"]),
                "service_balance": service["service_balance"],
                "service_gap": service["share_gap"],
                "joint_backlog_s": service["qualifying_s"],
                "bc_median_e2e_s": statistics.median(victim),
                "bc_p95_e2e_s": percentile(victim, 95),
            }
            for tenant in tenants:
                record[f"dominant_share_{tenant}"] = service["dominant_share"][tenant]
                record[f"cpu_service_{tenant}"] = service["cpu_service"][tenant]
                record[f"gpu_lane_service_{tenant}"] = service["gpu_lane_service"][tenant]
            records.append(record)
    seeds = sorted({int(row["seed"]) for row in records})
    policies = sorted({str(row["policy"]) for row in records}, key=policy_key)
    expected = {(seed, policy) for seed in seeds for policy in policies}
    observed = {(int(row["seed"]), str(row["policy"])) for row in records}
    if observed != expected or "adaptive_fcfs" not in policies:
        raise SystemExit(f"incomplete matrix; missing={sorted(expected - observed)}")
    fcfs = {int(row["seed"]): float(row["throughput_qps"])
            for row in records if row["policy"] == "adaptive_fcfs"}
    for row in records:
        row["throughput_pct_fcfs"] = 100.0 * float(row["throughput_qps"]) / fcfs[int(row["seed"])]
    return records


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[str(row["policy"])].append(row)
    fields = (
        "throughput_qps", "throughput_pct_fcfs", "service_balance",
        "service_gap", "joint_backlog_s", "bc_median_e2e_s", "bc_p95_e2e_s",
        "dominant_share_a", "dominant_share_b", "dominant_share_c",
    )
    output: list[dict] = []
    for policy in sorted(grouped, key=policy_key):
        group = grouped[policy]
        item = {
            "policy": policy,
            "slack_s": group[0]["slack_s"],
            "runs": len(group),
        }
        for field in fields:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    for row in output:
        x = float(row["throughput_pct_fcfs_mean"])
        y = float(row["service_balance_mean"])
        row["pareto"] = not any(
            other is not row
            and float(other["throughput_pct_fcfs_mean"]) >= x
            and float(other["service_balance_mean"]) >= y
            and (float(other["throughput_pct_fcfs_mean"]) > x
                 or float(other["service_balance_mean"]) > y)
            for other in output
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path) -> None:
    fair = [row for row in rows if row["policy"] != "adaptive_fcfs"]
    fcfs = next(row for row in rows if row["policy"] == "adaptive_fcfs")
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    x = [float(row["throughput_pct_fcfs_mean"]) for row in fair]
    y = [100.0 * float(row["service_balance_mean"]) for row in fair]
    colors = [float(row["bc_median_e2e_s_mean"]) for row in fair]
    ax.plot(x, y, color="#666666", linewidth=1.2, alpha=0.65, zorder=1)
    points = ax.scatter(x, y, c=colors, cmap="viridis_r", s=75,
                        edgecolor="white", linewidth=0.8, zorder=3)
    for row, px, py in zip(fair, x, y):
        ax.annotate(f"$\\delta$={float(row['slack_s']):g}", (px, py),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    fx = float(fcfs["throughput_pct_fcfs_mean"])
    fy = 100.0 * float(fcfs["service_balance_mean"])
    ax.scatter([fx], [fy], marker="s", s=80, color="#D55E00",
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.annotate("Adaptive FCFS", (fx, fy), xytext=(5, -13),
                textcoords="offset points", fontsize=8)
    ax.errorbar(
        [float(row["throughput_pct_fcfs_mean"]) for row in rows],
        [100.0 * float(row["service_balance_mean"]) for row in rows],
        xerr=[float(row["throughput_pct_fcfs_std"]) for row in rows],
        yerr=[100.0 * float(row["service_balance_std"]) for row in rows],
        fmt="none", ecolor="#777777", alpha=0.55, capsize=2, zorder=0,
    )
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("B/C median end-to-end latency (s)")
    ax.set_xlabel("Throughput (% of matched adaptive FCFS)")
    ax.set_ylabel("Preparation service balance (%)")
    ax.set_title("Efficiency–fairness tradeoff with identical CPU/GPU placement",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = discover(args.roots)
    aggregates = aggregate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output.with_name(args.output.name + "_runs.csv"), records)
    write_csv(args.output.with_name(args.output.name + "_aggregate.csv"), aggregates)
    args.output.with_name(args.output.name + "_aggregate.json").write_text(
        json.dumps(aggregates, indent=2, allow_nan=True) + "\n"
    )
    plot(aggregates, args.output)
    for row in aggregates:
        print(
            f"{row['policy']:18s} throughput={row['throughput_pct_fcfs_mean']:.1f}% "
            f"balance={100 * row['service_balance_mean']:.1f}% "
            f"B/C median={row['bc_median_e2e_s_mean']:.2f}s "
            f"pareto={row['pareto']}"
        )


if __name__ == "__main__":
    main()
