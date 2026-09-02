#!/usr/bin/env python3
"""Summarize latency, slowdown, and tenant fairness for matched runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def jain(values: list[float]) -> float | None:
    if not values or not any(values):
        return None
    return sum(values) ** 2 / (len(values) * sum(value ** 2 for value in values))


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fmt(value, suffix="") -> str:
    return "--" if value is None else f"{float(value):.2f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urgent-e2e-slo-s", type=float, default=60.0)
    parser.add_argument("--background-e2e-slo-s", type=float, default=300.0)
    args = parser.parse_args()

    records = []
    for summary_path in sorted(args.root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        results = load_jsonl(summary_path.with_name("results.jsonl"))
        valid = [row for row in results if not row.get("error")]
        by_tenant = defaultdict(list)
        for row in valid:
            by_tenant[str(row.get("tenant", "default"))].append(row)
        tenant_slowdowns = []
        tenant_mean_completion = []
        for rows in by_tenant.values():
            slowdowns = [
                float(row["estimated_slowdown"])
                for row in rows if row.get("estimated_slowdown") is not None
            ]
            completions = [float(row["end_to_end_s"]) for row in rows]
            if slowdowns:
                tenant_slowdowns.append(sum(slowdowns) / len(slowdowns))
            if completions:
                tenant_mean_completion.append(sum(completions) / len(completions))
        all_slowdowns = [
            float(row["estimated_slowdown"])
            for row in valid if row.get("estimated_slowdown") is not None
        ]
        background_waits = [
            float(row["prep_queue_wait_s"])
            for row in valid if row["workload"] == "background"
        ]
        urgent = [row for row in valid if row["workload"] == "urgent"]
        background = [
            row for row in valid if row["workload"] == "background"
        ]
        urgent_arrival_start = min(
            (float(row["arrival_s"]) for row in urgent), default=None,
        )
        urgent_completion_end = max(
            (float(row["completion_s"]) for row in urgent), default=None,
        )
        background_progress = [
            row for row in background
            if urgent_arrival_start is not None
            and urgent_completion_end is not None
            and urgent_arrival_start <= float(row["prep_started_s"])
            <= urgent_completion_end
        ]
        urgent_e2e_attained = sum(
            float(row["end_to_end_s"]) <= args.urgent_e2e_slo_s
            for row in urgent
        )
        background_e2e_attained = sum(
            float(row["end_to_end_s"]) <= args.background_e2e_slo_s
            for row in background
        )
        wall_time_s = float(summary.get("wall_time_s") or 0.0)
        records.append({
            "trace": summary_path.parents[1].name,
            "policy": summary_path.parent.name,
            "requests": summary.get("total_requests"),
            "errors": summary.get("errors"),
            "throughput_qps": summary.get("throughput_qps"),
            "urgent_mean_e2e_s": summary["urgent"].get("mean_end_to_end_s"),
            "urgent_p95_e2e_s": summary["urgent"].get("p95_end_to_end_s"),
            "urgent_ttft_slo_attainment_percent": summary["urgent"].get(
                "ttft_slo_attainment_percent"
            ),
            "urgent_e2e_slo_attainment_percent": (
                100.0 * urgent_e2e_attained / len(urgent) if urgent else None
            ),
            "background_e2e_slo_attainment_percent": (
                100.0 * background_e2e_attained / len(background)
                if background else None
            ),
            "e2e_goodput_qps": (
                (urgent_e2e_attained + background_e2e_attained) / wall_time_s
                if wall_time_s else None
            ),
            "background_progress_during_urgent": len(background_progress),
            "background_max_prep_wait_s": max(background_waits) if background_waits else None,
            "p95_slowdown": percentile(all_slowdowns, 0.95),
            "max_slowdown": max(all_slowdowns) if all_slowdowns else None,
            # Equal service quality means equal inverse slowdown.
            "tenant_quality_jain": jain([
                1.0 / value for value in tenant_slowdowns if value > 0
            ]),
            "tenant_slowdown_ratio": (
                max(tenant_slowdowns) / min(tenant_slowdowns)
                if tenant_slowdowns and min(tenant_slowdowns) > 0 else None
            ),
            "tenant_completion_spread_s": (
                max(tenant_mean_completion) - min(tenant_mean_completion)
                if tenant_mean_completion else None
            ),
        })
    if not records:
        raise SystemExit("no completed runs found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    grouped = defaultdict(list)
    for record in records:
        grouped[record["policy"]].append(record)
    lines = [
        "# Multi-tenant scheduling validation",
        "",
        "| Policy | Runs | Urgent E2E | Urgent TTFT SLO | Urgent E2E SLO | E2E goodput | Tenant Jain | Worst/best slowdown | Background starts while urgent | Max background prep wait | Throughput |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in (
        "fcfs", "engine_tenant_fair", "max_min", "priority", "sjf",
        "tenant_fair", "tenant_priority", "fair_slowdown",
    ):
        rows = grouped.get(policy, [])
        if not rows:
            continue
        metric = lambda field: average([
            float(row[field]) for row in rows if row[field] is not None
        ])
        lines.append(
            f"| {policy} | {len(rows)} | {fmt(metric('urgent_mean_e2e_s'), 's')} "
            f"| {fmt(metric('urgent_ttft_slo_attainment_percent'), '%')} "
            f"| {fmt(metric('urgent_e2e_slo_attainment_percent'), '%')} "
            f"| {fmt(metric('e2e_goodput_qps'))} "
            f"| {fmt(metric('tenant_quality_jain'))} "
            f"| {fmt(metric('tenant_slowdown_ratio'), 'x')} "
            f"| {fmt(metric('background_progress_during_urgent'))} "
            f"| {fmt(metric('background_max_prep_wait_s'), 's')} "
            f"| {fmt(metric('throughput_qps'))} |"
        )
    md_path = args.output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "completed_runs": len(records),
        "csv": str(csv_path),
        "report": str(md_path),
    }, indent=2))


if __name__ == "__main__":
    main()
