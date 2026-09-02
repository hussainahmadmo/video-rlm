#!/usr/bin/env python3
"""Empirically evaluate cross-stage service isolation and progress.

The analysis intentionally avoids collapsing CPU preparation seconds and GPU
engine seconds into one synthetic unit.  It measures each stage separately and
only compares tenants while every tenant is backlogged at that stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


POLICY_ORDER = [
    "fcfs",
    "tenant_round_robin",
    "prep_max_min",
    "engine_tenant_fair",
    "max_min",
    "max_min_completion_only",
    "max_min_no_reconcile",
    "priority",
    "sjf",
    "tenant_fair",
    "tenant_priority",
    "fair_slowdown",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def interval_service(
    rows: Iterable[dict], tenant: str, start: str, end: str, t: float,
    service_field: str | None = None,
) -> float:
    total = 0.0
    for row in rows:
        if row.get("tenant") != tenant:
            continue
        left, right = row.get(start), row.get(end)
        if not valid_number(left) or not valid_number(right) or t <= left:
            continue
        elapsed = max(0.0, min(float(t), float(right)) - float(left))
        duration = float(right) - float(left)
        if service_field and valid_number(row.get(service_field)) and duration > 0:
            total += elapsed / duration * float(row[service_field])
        else:
            total += elapsed
    return total


def is_backlogged(rows: Iterable[dict], tenant: str, arrival: str, done: str, t: float) -> bool:
    return any(
        row.get("tenant") == tenant
        and valid_number(row.get(arrival))
        and valid_number(row.get(done))
        and float(row[arrival]) <= t < float(row[done])
        for row in rows
    )


def stage_leads(
    rows: list[dict],
    tenants: list[str],
    weights: dict[str, float],
    arrival: str,
    start: str,
    done: str,
    service_field: str | None = None,
) -> tuple[float, float, float]:
    """Return peak/mean service lead and peak dispatch-count lead.

    Values are evaluated only while every tenant has outstanding work at the
    stage. Service is profile-derived inference cost when present and occupied
    wall time otherwise, divided by tenant weight.
    """
    times = sorted(
        {
            float(row[key])
            for row in rows
            for key in (arrival, start, done)
            if valid_number(row.get(key))
        }
    )
    service_leads = []
    dispatches: Counter[str] = Counter()
    starts = sorted(
        (float(row[start]), str(row["tenant"]))
        for row in rows
        if row.get("tenant") in tenants and valid_number(row.get(start))
    )
    start_index = 0
    dispatch_leads = []

    for t in times:
        while start_index < len(starts) and starts[start_index][0] <= t:
            dispatches[starts[start_index][1]] += 1
            start_index += 1
        if not all(is_backlogged(rows, tenant, arrival, done, t) for tenant in tenants):
            continue
        services = [
            interval_service(
                rows, tenant, start, done, t, service_field
            ) / weights[tenant]
            for tenant in tenants
        ]
        service_leads.append(max(services) - min(services))
        counts = [dispatches[tenant] / weights[tenant] for tenant in tenants]
        dispatch_leads.append(max(counts) - min(counts))

    if not service_leads:
        return math.nan, math.nan, math.nan
    return max(service_leads), statistics.mean(service_leads), max(dispatch_leads)


def analyze_run(summary_path: Path, root: Path) -> dict:
    run_dir = summary_path.parent
    summary = json.loads(summary_path.read_text())
    rows = [row for row in read_jsonl(run_dir / "results.jsonl") if not row.get("error")]
    tenants = sorted({str(row["tenant"]) for row in rows if row.get("tenant") is not None})
    raw_weights = summary.get("tenant_weights") or {}
    weights = {tenant: float(raw_weights.get(tenant, 1.0)) for tenant in tenants}

    prep_peak, prep_mean, prep_dispatch = stage_leads(
        rows, tenants, weights, "arrival_s", "prep_started_s", "prep_ready_s"
    )
    engine_peak, engine_mean, engine_dispatch = stage_leads(
        rows, tenants, weights, "prep_ready_s", "vlm_submit_s", "completion_s",
        "inference_accounted_service_s",
    )

    wall = float(summary["wall_time_s"])
    prep_service = sum(float(row["prep_service_s"]) for row in rows if valid_number(row.get("prep_service_s")))
    engine_service = sum(
        float(row["completion_s"]) - float(row["vlm_submit_s"])
        for row in rows
        if valid_number(row.get("completion_s")) and valid_number(row.get("vlm_submit_s"))
    )
    prep_capacity = max(1, int(summary.get("prep_workers") or 1))
    engine_capacity = max(1, int(summary.get("vlm_concurrency_total") or 1))
    background = [row for row in rows if row.get("workload") == "background"]
    urgent = [row for row in rows if row.get("workload") == "urgent"]

    def mean_field(group: list[dict], field: str) -> float:
        values = [float(row[field]) for row in group if valid_number(row.get(field))]
        return statistics.mean(values) if values else math.nan

    tenant_slowdowns = [
        mean_field([row for row in rows if row.get("tenant") == tenant], "estimated_slowdown")
        for tenant in tenants
    ]
    tenant_slowdowns = [value for value in tenant_slowdowns if valid_number(value) and value > 0]

    policy = str(summary.get("prep_policy") or run_dir.name)
    if policy == "max_min":
        accounting_mode = summary.get("service_accounting_mode")
        if accounting_mode == "completion_only":
            policy = "max_min_completion_only"
        elif accounting_mode == "estimate_only" or (
            accounting_mode is None
            and summary.get("service_reconciliation") is False
        ):
            policy = "max_min_no_reconcile"

    return {
        "trace": summary_path.relative_to(root).parts[0],
        "policy": policy,
        "tenants": len(tenants),
        "requests": int(summary.get("total_requests", len(rows))),
        "errors": int(summary.get("errors", 0)),
        "completion_percent": 100.0 * len(rows) / max(1, int(summary.get("total_requests", len(rows)))),
        "prep_peak_service_lead_s": prep_peak,
        "prep_mean_service_lead_s": prep_mean,
        "prep_peak_dispatch_lead": prep_dispatch,
        "engine_peak_service_lead_s": engine_peak,
        "engine_mean_service_lead_s": engine_mean,
        "engine_peak_dispatch_lead": engine_dispatch,
        "worst_best_tenant_slowdown_ratio": (
            max(tenant_slowdowns) / min(tenant_slowdowns) if tenant_slowdowns else math.nan
        ),
        "urgent_mean_e2e_s": mean_field(urgent, "end_to_end_s"),
        "background_mean_e2e_s": mean_field(background, "end_to_end_s"),
        "mean_e2e_s": mean_field(rows, "end_to_end_s"),
        "max_background_prep_wait_s": max(
            (float(row["prep_queue_wait_s"]) for row in background if valid_number(row.get("prep_queue_wait_s"))),
            default=math.nan,
        ),
        "max_prep_wait_s": max(
            (float(row["prep_queue_wait_s"]) for row in rows if valid_number(row.get("prep_queue_wait_s"))),
            default=math.nan,
        ),
        "prep_utilization_percent": 100.0 * prep_service / (prep_capacity * wall),
        "engine_utilization_percent": 100.0 * engine_service / (engine_capacity * wall),
        "throughput_qps": float(summary["throughput_qps"]),
    }


def mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if valid_number(row.get(field))]
    return statistics.mean(values) if values else math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    fields = [
        "completion_percent",
        "prep_peak_service_lead_s",
        "prep_mean_service_lead_s",
        "prep_peak_dispatch_lead",
        "engine_peak_service_lead_s",
        "engine_mean_service_lead_s",
        "engine_peak_dispatch_lead",
        "worst_best_tenant_slowdown_ratio",
        "urgent_mean_e2e_s",
        "background_mean_e2e_s",
        "mean_e2e_s",
        "max_background_prep_wait_s",
        "max_prep_wait_s",
        "prep_utilization_percent",
        "engine_utilization_percent",
        "throughput_qps",
    ]
    result = []
    for policy in POLICY_ORDER:
        group = grouped.get(policy)
        if not group:
            continue
        item = {"policy": policy, "runs": len(group), "errors": sum(row["errors"] for row in group)}
        item.update({field: mean(group, field) for field in fields})
        result.append(item)
    return result


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Cross-stage empirical fairness",
        "",
        "Service lead is the largest gap in cumulative, weight-normalized occupied "
        "service seconds while every tenant is backlogged at that stage. CPU and "
        "engine service are deliberately reported separately.",
        "",
        "Engine dispatch lead counts admissions rather than concurrent residence "
        "time, because the API does not expose per-request GPU occupancy.",
        "",
        "| Policy | Runs | Complete | Prep service lead | Prep dispatch lead | "
        "Engine dispatch lead | Worst/best tenant slowdown | Mean E2E | Max prep wait | "
        "Prep util. | Engine util. | Throughput |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['runs']} | {row['completion_percent']:.1f}% | "
            f"{row['prep_peak_service_lead_s']:.1f}s | {row['prep_peak_dispatch_lead']:.1f} | "
            f"{row['engine_peak_dispatch_lead']:.1f} | {row['worst_best_tenant_slowdown_ratio']:.2f}x | "
            f"{row['mean_e2e_s']:.1f}s | {row['max_prep_wait_s']:.1f}s | "
            f"{row['prep_utilization_percent']:.1f}% | {row['engine_utilization_percent']:.1f}% | "
            f"{row['throughput_qps']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(path: Path, run_rows: list[dict], aggregate_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    policies = [row["policy"] for row in aggregate_rows]
    labels = [policy.replace("tenant_", "tenant\n").replace("fair_", "fair\n") for policy in policies]
    grouped = {policy: [row for row in run_rows if row["policy"] == policy] for policy in policies}
    panels = [
        ("prep_peak_service_lead_s", "(a) CPU-preparation service lead", "Seconds; lower is fairer"),
        ("prep_peak_dispatch_lead", "(b) Preparation dispatch lead", "Requests; lower is fairer"),
        ("worst_best_tenant_slowdown_ratio", "(c) Tenant slowdown spread", "Worst / best; lower is fairer"),
        ("mean_e2e_s", "(d) Mean end-to-end latency", "Seconds; lower is better"),
        ("max_prep_wait_s", "(e) Maximum preparation wait", "Seconds; lower is better"),
        ("throughput_qps", "(f) Aggregate throughput", "Requests/second"),
    ]
    colors = ["#777777", "#d55e00", "#e69f00", "#56b4e9", "#009e73", "#cc79a7"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    rng = np.random.default_rng(7)
    for ax, (field, title, ylabel) in zip(axes.flat, panels):
        means = [mean(grouped[policy], field) for policy in policies]
        ax.bar(range(len(policies)), means, color=colors[: len(policies)], alpha=0.82)
        for index, policy in enumerate(policies):
            values = [row[field] for row in grouped[policy] if valid_number(row.get(field))]
            jitter = rng.uniform(-0.10, 0.10, len(values))
            ax.scatter(index + jitter, values, color="black", s=18, alpha=0.75, zorder=3)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(policies)), labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Empirical service isolation, foreground latency, and background progress\n"
        "Bars are means across four matched traces; dots are individual traces",
        fontsize=16,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix (without extension)")
    args = parser.parse_args()

    summary_paths = sorted(args.root.glob("*/*/summary.json"))
    if not summary_paths:
        raise SystemExit(f"No summaries found below {args.root}")
    run_rows = [analyze_run(path, args.root) for path in summary_paths]
    aggregate_rows = aggregate(run_rows)
    write_csv(args.output.with_name(args.output.name + "_runs.csv"), run_rows)
    write_csv(args.output.with_suffix(".csv"), aggregate_rows)
    write_markdown(args.output.with_suffix(".md"), aggregate_rows)
    plot(args.output, run_rows, aggregate_rows)
    print(args.output.with_suffix(".md"))
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
