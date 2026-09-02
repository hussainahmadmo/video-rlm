#!/usr/bin/env python3
"""Analyze VTC-style multimodal fairness, isolation, and work conservation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def overlap(start: float, end: float, left: float, right: float) -> float:
    return max(0.0, min(end, right) - max(start, left))


def service_at(rows: list[dict], tenant: str, stage: str, time_s: float) -> float:
    if stage == "prep":
        bounds = ("prep_started_s", "prep_ready_s")
    else:
        bounds = ("vlm_submit_s", "completion_s")
    return sum(
        overlap(float(row[bounds[0]]), float(row[bounds[1]]), 0.0, time_s)
        for row in rows
        if str(row.get("tenant")) == tenant
        and row.get(bounds[0]) is not None and row.get(bounds[1]) is not None
    )


def service_rate(
    rows: list[dict], tenant: str, stage: str, time_s: float, window_s: float,
) -> float:
    if stage == "prep":
        bounds = ("prep_started_s", "prep_ready_s")
    else:
        bounds = ("vlm_submit_s", "completion_s")
    left = max(0.0, time_s - window_s)
    width = max(1e-9, time_s - left)
    return sum(
        overlap(float(row[bounds[0]]), float(row[bounds[1]]), left, time_s)
        for row in rows
        if str(row.get("tenant")) == tenant
        and row.get(bounds[0]) is not None and row.get(bounds[1]) is not None
    ) / width


def response_time(
    rows: list[dict], tenant: str, time_s: float, window_s: float,
) -> float | None:
    left = max(0.0, time_s - window_s / 2)
    right = time_s + window_s / 2
    values = [
        float(row["end_to_end_s"])
        for row in rows
        if str(row.get("tenant")) == tenant
        and left <= float(row["arrival_s"]) < right
        and row.get("end_to_end_s") is not None
    ]
    return mean(values)


def curve(rows: list[dict], wall_s: float, window_s: float) -> dict:
    tenants = sorted({str(row.get("tenant", "default")) for row in rows})
    samples = 61
    times = [wall_s * index / (samples - 1) for index in range(samples)]
    prep_lead = []
    engine_lead = []
    prep_rates = {tenant: [] for tenant in tenants}
    response = {tenant: [] for tenant in tenants}
    for time_s in times:
        prep = [service_at(rows, tenant, "prep", time_s) for tenant in tenants]
        engine = [service_at(rows, tenant, "engine", time_s) for tenant in tenants]
        prep_lead.append(max(prep) - min(prep) if prep else 0.0)
        engine_lead.append(max(engine) - min(engine) if engine else 0.0)
        for tenant in tenants:
            prep_rates[tenant].append(service_rate(
                rows, tenant, "prep", time_s, window_s,
            ))
            response[tenant].append(response_time(
                rows, tenant, time_s, window_s,
            ))
    return {
        "times": times,
        "tenants": tenants,
        "prep_lead": prep_lead,
        "engine_lead": engine_lead,
        "prep_rates": prep_rates,
        "response": response,
    }


def plot_scenario(
    scenario: str, runs: dict[str, tuple[list[dict], dict]], output: Path,
    window_s: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), constrained_layout=True)
    colors = {
        "fcfs": "#d55e00", "priority": "#cc79a7",
        "tenant_fair": "#0072b2", "tenant_priority": "#009e73",
        "engine_tenant_fair": "#999999", "max_min": "#56b4e9",
        "prep_max_min": "#cc79a7", "tenant_round_robin": "#f0e442",
        "max_min_completion_only": "#d55e00",
        "max_min_no_reconcile": "#e69f00",
    }
    curves = {}
    for policy, (rows, summary) in runs.items():
        curves[policy] = curve(rows, float(summary["wall_time_s"]), window_s)
        axes[0, 0].plot(
            curves[policy]["times"], curves[policy]["prep_lead"],
            label=policy, color=colors.get(policy), linewidth=2,
        )
        axes[0, 1].plot(
            curves[policy]["times"], curves[policy]["engine_lead"],
            label=policy, color=colors.get(policy), linewidth=2,
        )
    preferred = "max_min" if "max_min" in curves else next(iter(curves))
    selected = curves[preferred]
    for tenant in selected["tenants"]:
        axes[1, 0].plot(
            selected["times"], selected["prep_rates"][tenant],
            label=f"tenant {tenant}", linewidth=2,
        )
        values = selected["response"][tenant]
        axes[1, 1].plot(
            [time for time, value in zip(selected["times"], values) if value is not None],
            [value for value in values if value is not None],
            label=f"tenant {tenant}", linewidth=2,
        )
    axes[0, 0].set_title("Cumulative preparation-service difference")
    axes[0, 1].set_title("Cumulative engine-residence difference")
    axes[1, 0].set_title(f"{preferred}: preparation service rate")
    axes[1, 1].set_title(f"{preferred}: mean end-to-end latency")
    for axis in axes.flat:
        axis.set_xlabel("Elapsed time (s)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Max tenant lead (service-s)")
    axes[0, 1].set_ylabel("Max tenant lead (residence-s)")
    axes[1, 0].set_ylabel("CPU-seconds / wall-second")
    axes[1, 1].set_ylabel("Seconds")
    fig.suptitle(scenario.replace("_", " ").title(), fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    records = []
    plot_groups: dict[tuple[str, str], dict] = defaultdict(dict)
    for summary_path in sorted(args.root.glob("*/*/summary.json")):
        trace_name = summary_path.parents[1].name
        summary = json.loads(summary_path.read_text())
        policy = str(summary.get("prep_policy") or summary_path.parent.name)
        if policy == "max_min":
            accounting_mode = summary.get("service_accounting_mode")
            if accounting_mode == "completion_only":
                policy = "max_min_completion_only"
            elif accounting_mode == "estimate_only" or (
                accounting_mode is None
                and summary.get("service_reconciliation") is False
            ):
                policy = "max_min_no_reconcile"
        scenario, _, seed_text = trace_name.rpartition("-seed")
        all_rows = load_jsonl(summary_path.with_name("results.jsonl"))
        rows = [row for row in all_rows if not row.get("error")]
        if not rows:
            continue
        tenants = sorted({str(row.get("tenant", "default")) for row in rows})
        prep = {
            tenant: sum(float(row["prep_service_s"]) for row in rows
                        if str(row.get("tenant")) == tenant)
            for tenant in tenants
        }
        engine = {
            tenant: sum(float(row["completion_s"]) - float(row["vlm_submit_s"])
                        for row in rows if str(row.get("tenant")) == tenant)
            for tenant in tenants
        }
        e2e = [float(row["end_to_end_s"]) for row in rows]
        wall_s = float(summary["wall_time_s"])
        prep_workers = int(summary.get("prep_workers") or 1)
        stable_rows = [row for row in rows if str(row.get("tenant")) == "a"]
        thirds = [[], [], []]
        for row in stable_rows:
            phase = min(2, int(3 * float(row["arrival_s"]) / max(1e-9, wall_s)))
            thirds[phase].append(float(row["end_to_end_s"]))
        records.append({
            "scenario": scenario,
            "seed": seed_text,
            "policy": policy,
            "requests": len(rows),
            "errors": int(summary.get("errors") or 0),
            "throughput_qps": float(summary["throughput_qps"]),
            "prep_utilization_percent": 100 * sum(prep.values()) / (wall_s * prep_workers),
            "final_prep_service_lead_s": max(prep.values()) - min(prep.values()),
            "final_engine_residence_lead_s": max(engine.values()) - min(engine.values()),
            "mean_e2e_s": mean(e2e),
            "p95_e2e_s": percentile(e2e, 0.95),
            "tenant_a_first_third_e2e_s": mean(thirds[0]),
            "tenant_a_last_third_e2e_s": mean(thirds[2]),
        })
        plot_groups[(scenario, seed_text)][policy] = (rows, summary)
    if not records:
        raise SystemExit("no completed VTC-style runs found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["scenario"], record["policy"])].append(record)
    lines = [
        "# VTC-style multimodal experiments", "",
        "Service is reported separately for CPU preparation and inference-engine residence; the analysis does not invent a CPU-to-token conversion.", "",
        "| Scenario | Policy | Runs | Prep lead | Engine lead | Mean E2E | p95 E2E | Prep util. | Throughput |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (scenario, policy), rows in sorted(grouped.items()):
        avg = lambda key: statistics.fmean(float(row[key]) for row in rows)
        lines.append(
            f"| {scenario} | {policy} | {len(rows)} | "
            f"{avg('final_prep_service_lead_s'):.2f}s | "
            f"{avg('final_engine_residence_lead_s'):.2f}s | "
            f"{avg('mean_e2e_s'):.2f}s | {avg('p95_e2e_s'):.2f}s | "
            f"{avg('prep_utilization_percent'):.1f}% | "
            f"{avg('throughput_qps'):.4f} |"
        )
    md_path = args.output.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n")

    if not args.no_plots:
        for (scenario, seed), runs in plot_groups.items():
            plot_scenario(
                scenario, runs,
                args.output.parent / "figures" / f"{scenario}-seed{seed}",
                args.window_s,
            )
    print(json.dumps({
        "runs": len(records), "csv": str(csv_path), "report": str(md_path),
    }, indent=2))


if __name__ == "__main__":
    main()
