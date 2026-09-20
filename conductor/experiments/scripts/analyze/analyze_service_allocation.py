#!/usr/bin/env python3
"""Measure service shares only while every tenant is backlogged at a stage."""

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


POLICIES = ("fcfs", "prep_max_min", "engine_tenant_fair", "max_min")
POLICY_LABELS = {
    "fcfs": "FCFS",
    "prep_max_min": "Prep-only",
    "engine_tenant_fair": "Infer-only",
    "max_min": "Conductor",
}
TENANT_LABELS = {"a": "Tenant A", "b": "Tenant B", "c": "Tenant C"}
TENANT_COLORS = {"a": "#D55E00", "b": "#0072B2", "c": "#009E73"}
STAGES = {
    "prep": {
        "backlog_start": "arrival_s",
        "backlog_end": "prep_ready_s",
        "service_start": "prep_started_s",
        "service_end": "prep_ready_s",
        "service_amount": "prep_service_s",
    },
    "infer": {
        "backlog_start": "prep_ready_s",
        "backlog_end": "completion_s",
        "service_start": "vlm_submit_s",
        "service_end": "completion_s",
        "service_amount": "inference_accounted_service_s",
    },
}


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def overlap(left: float, right: float, start: float, end: float) -> float:
    return max(0.0, min(right, end) - max(left, start))


def measure_stage(rows: list[dict], stage: str, tenants: list[str]) -> dict:
    fields = STAGES[stage]
    valid = [
        row
        for row in rows
        if not row.get("error")
        and row.get("tenant") in tenants
        and all(finite(row.get(fields[key])) for key in fields)
    ]
    boundaries = sorted(
        {
            float(row[fields[key]])
            for row in valid
            for key in ("backlog_start", "backlog_end", "service_start", "service_end")
        }
    )
    received = {tenant: 0.0 for tenant in tenants}
    qualifying_s = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        all_backlogged = all(
            any(
                row["tenant"] == tenant
                and float(row[fields["backlog_start"]]) <= midpoint
                < float(row[fields["backlog_end"]])
                for row in valid
            )
            for tenant in tenants
        )
        if not all_backlogged:
            continue
        qualifying_s += right - left
        for row in valid:
            duration = float(row[fields["service_end"]]) - float(
                row[fields["service_start"]]
            )
            if duration <= 0:
                continue
            service_overlap = overlap(
                left,
                right,
                float(row[fields["service_start"]]),
                float(row[fields["service_end"]]),
            )
            received[str(row["tenant"])] += (
                service_overlap / duration * float(row[fields["service_amount"]])
            )

    total = sum(received.values())
    shares = {
        tenant: received[tenant] / total if total > 0 else math.nan for tenant in tenants
    }
    share_gap = max(shares.values()) - min(shares.values()) if total > 0 else math.nan
    l1_deviation = (
        0.5 * sum(abs(share - 1.0 / len(tenants)) for share in shares.values())
        if total > 0
        else math.nan
    )
    return {
        "qualifying_s": qualifying_s,
        "total_service": total,
        "received": received,
        "shares": shares,
        "share_gap": share_gap,
        "l1_deviation": l1_deviation,
    }


def discover_stage(root: Path, stage: str) -> list[dict]:
    records = []
    for results_path in sorted(root.glob("seed*/*/results.jsonl")):
        seed_dir, policy = results_path.parent.parent.name, results_path.parent.name
        if policy not in POLICIES:
            continue
        seed = int(seed_dir[4:])
        summary = json.loads(results_path.with_name("summary.json").read_text())
        if int(summary.get("errors", 0)) != 0:
            raise SystemExit(f"run contains errors: {results_path.parent}")
        rows = read_jsonl(results_path)
        tenants = sorted({str(row["tenant"]) for row in rows if not row.get("error")})
        if tenants != ["a", "b", "c"]:
            raise SystemExit(f"unexpected tenants in {results_path}: {tenants}")
        result = measure_stage(rows, stage, tenants)
        if result["qualifying_s"] <= 0 or result["total_service"] <= 0:
            raise SystemExit(f"no joint backlog service for {stage}: {results_path}")
        record = {
            "seed": seed,
            "policy": policy,
            "stage": stage,
            "qualifying_s": result["qualifying_s"],
            "total_service": result["total_service"],
            "share_gap": result["share_gap"],
            "l1_deviation": result["l1_deviation"],
            "throughput_qps": float(summary["throughput_qps"]),
        }
        for tenant in tenants:
            record[f"service_{tenant}"] = result["received"][tenant]
            record[f"share_{tenant}"] = result["shares"][tenant]
        records.append(record)
    expected_seeds = sorted({int(path.name[4:]) for path in root.glob("seed*")})
    expected = {(seed, policy, stage) for seed in expected_seeds for policy in POLICIES}
    observed = {(row["seed"], row["policy"], row["stage"]) for row in records}
    if observed != expected:
        raise SystemExit(f"incomplete matrix; missing={sorted(expected - observed)}")
    return records


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["policy"], row["stage"])].append(row)
    fields = (
        "qualifying_s",
        "total_service",
        "share_gap",
        "l1_deviation",
        "throughput_qps",
        "service_a",
        "service_b",
        "service_c",
        "share_a",
        "share_b",
        "share_c",
    )
    output = []
    for policy in POLICIES:
        for stage in STAGES:
            group = sorted(grouped[(policy, stage)], key=lambda row: row["seed"])
            item = {"policy": policy, "stage": stage, "runs": len(group)}
            for field in fields:
                values = [float(row[field]) for row in group]
                item[f"{field}_mean"] = statistics.fmean(values)
                item[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(aggregate_rows: list[dict], output: Path) -> None:
    lookup = {(row["policy"], row["stage"]): row for row in aggregate_rows}
    x = np.arange(len(POLICIES), dtype=float)
    width = 0.23
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.55))
    for column, stage in enumerate(STAGES):
        share_axis = axes[0, column]
        gap_axis = axes[1, column]
        for tenant_index, tenant in enumerate(("a", "b", "c")):
            means = [100.0 * float(lookup[(policy, stage)][f"share_{tenant}_mean"]) for policy in POLICIES]
            stds = [100.0 * float(lookup[(policy, stage)][f"share_{tenant}_std"]) for policy in POLICIES]
            positions = x + (tenant_index - 1) * width
            share_axis.bar(
                positions,
                means,
                width,
                yerr=stds,
                capsize=2,
                color=TENANT_COLORS[tenant],
                label=TENANT_LABELS[tenant],
                edgecolor="white",
                linewidth=0.5,
            )
        share_axis.axhline(100.0 / 3.0, color="#444444", linestyle=":", linewidth=1.1)
        share_axis.set_ylim(0, 100)
        share_axis.set_yticks((0, 25, 50, 75, 100))
        share_axis.set_ylabel("Service share (%)")
        share_axis.set_title(
            "(a) Preparation worker service" if stage == "prep" else "(b) Inference accounted service",
            fontsize=9.5,
            fontweight="bold",
        )
        gaps = [100.0 * float(lookup[(policy, stage)]["share_gap_mean"]) for policy in POLICIES]
        gap_stds = [100.0 * float(lookup[(policy, stage)]["share_gap_std"]) for policy in POLICIES]
        gap_axis.bar(x, gaps, yerr=gap_stds, capsize=2.5, color="#7A6FAC", width=0.62)
        gap_axis.set_ylim(0, 100)
        gap_axis.set_yticks((0, 25, 50, 75, 100))
        gap_axis.set_ylabel("Normalized share gap (pp)")
        gap_axis.set_title(
            "(c) Preparation allocation gap" if stage == "prep" else "(d) Inference allocation gap",
            fontsize=9.5,
            fontweight="bold",
        )
        for axis in (share_axis, gap_axis):
            axis.set_xticks(x, [POLICY_LABELS[policy] for policy in POLICIES], rotation=17, ha="right")
            axis.grid(axis="y", alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.0, w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--infer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = discover_stage(args.prep_root, "prep") + discover_stage(
        args.infer_root, "infer"
    )
    aggregates = aggregate(records)
    write_csv(args.output.with_name(args.output.name + "_runs.csv"), records)
    write_csv(args.output.with_name(args.output.name + "_aggregate.csv"), aggregates)
    plot(aggregates, args.output)
    for row in aggregates:
        shares = ", ".join(
            f"{tenant.upper()}={100 * float(row[f'share_{tenant}_mean']):.1f}%"
            for tenant in ("a", "b", "c")
        )
        print(
            f"{row['policy']:20s} {row['stage']:5s} {shares}; "
            f"gap={100 * float(row['share_gap_mean']):.1f}pp; "
            f"joint_backlog={float(row['qualifying_s_mean']):.1f}s"
        )


if __name__ == "__main__":
    main()
