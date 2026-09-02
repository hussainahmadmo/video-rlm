#!/usr/bin/env python3
"""Aggregate and plot the complete multimodal-fairness experiment suite."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CROSS_ANALYZER = HERE / "analyze_cross_stage_fairness.py"
POLICY_ORDER = (
    "fcfs", "tenant_round_robin", "prep_max_min",
    "engine_tenant_fair", "max_min",
    "completion_only", "fixed_reservation", "profiled_reconciled",
)
LABELS = {
    "fcfs": "FCFS",
    "tenant_round_robin": "Tenant RR",
    "prep_max_min": "Preparation-only",
    "engine_tenant_fair": "Inference-only",
    "max_min": "Cross-stage",
    "completion_only": "Completion-only",
    "fixed_reservation": "Fixed reservation",
    "profiled_reconciled": "Profiled + reconciled",
}
COLORS = {
    "fcfs": "#D55E00",
    "tenant_round_robin": "#F0E442",
    "prep_max_min": "#CC79A7",
    "engine_tenant_fair": "#999999",
    "max_min": "#0072B2",
    "completion_only": "#D55E00",
    "fixed_reservation": "#E69F00",
    "profiled_reconciled": "#009E73",
}


def load_cross_analyzer():
    spec = importlib.util.spec_from_file_location("cross_stage_analysis", CROSS_ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def read_results(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_case_seed(name: str) -> tuple[str, str]:
    case, marker, seed = name.rpartition("-seed")
    return (case, seed) if marker else (name, "")


def analyze_run(path: Path, root: Path, cross) -> dict:
    relative = path.relative_to(root)
    phase, case_seed, variant = relative.parts[:3]
    case, seed = split_case_seed(case_seed)
    summary = json.loads(path.read_text())
    rows = [row for row in read_results(path.with_name("results.jsonl"))
            if not row.get("error")]
    cross_metrics = cross.analyze_run(path, root / phase)
    e2e = [float(row["end_to_end_s"]) for row in rows
           if finite(row.get("end_to_end_s"))]
    tenant_p95 = {}
    for tenant in sorted({str(row.get("tenant")) for row in rows}):
        values = [float(row["end_to_end_s"]) for row in rows
                  if str(row.get("tenant")) == tenant
                  and finite(row.get("end_to_end_s"))]
        tenant_p95[tenant] = percentile(values, 0.95)
    victims = [tenant_p95[tenant] for tenant in ("b", "c")
               if tenant in tenant_p95]
    return {
        "phase": phase,
        "case": case,
        "seed": seed,
        "variant": variant,
        "policy": str(summary.get("prep_policy") or variant),
        "accounting_mode": summary.get("service_accounting_mode"),
        "requests": int(summary.get("total_requests", len(rows))),
        "errors": int(summary.get("errors", 0)),
        "mean_e2e_s": mean(e2e),
        "p95_e2e_s": percentile(e2e, 0.95),
        "victim_mean_p95_e2e_s": mean(victims),
        "tenant_a_p95_e2e_s": tenant_p95.get("a", math.nan),
        "worst_tenant_p95_e2e_s": max(tenant_p95.values(), default=math.nan),
        "throughput_qps": float(summary.get("throughput_qps", math.nan)),
        "prep_peak_service_lead_s": cross_metrics["prep_peak_service_lead_s"],
        "prep_mean_service_lead_s": cross_metrics["prep_mean_service_lead_s"],
        "engine_peak_service_lead_s": cross_metrics["engine_peak_service_lead_s"],
        "engine_mean_service_lead_s": cross_metrics["engine_mean_service_lead_s"],
        "engine_peak_dispatch_lead": cross_metrics["engine_peak_dispatch_lead"],
        "prep_utilization_percent": cross_metrics["prep_utilization_percent"],
        "engine_utilization_percent": cross_metrics["engine_utilization_percent"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["case"], row["variant"])].append(row)
    numeric = (
        "mean_e2e_s", "p95_e2e_s", "victim_mean_p95_e2e_s",
        "tenant_a_p95_e2e_s",
        "worst_tenant_p95_e2e_s", "throughput_qps",
        "prep_peak_service_lead_s", "prep_mean_service_lead_s",
        "engine_peak_service_lead_s", "engine_mean_service_lead_s",
        "engine_peak_dispatch_lead", "prep_utilization_percent",
        "engine_utilization_percent",
    )
    result = []
    for key, group in sorted(grouped.items()):
        item = {
            "phase": key[0], "case": key[1], "variant": key[2],
            "runs": len(group), "errors": sum(row["errors"] for row in group),
        }
        for field in numeric:
            item[field] = mean([
                float(row[field]) for row in group if finite(row.get(field))
            ])
        result.append(item)
    return result


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Complete multimodal fairness suite", "",
        "CPU preparation and inference-admission service are kept as separate "
        "metrics. Service leads are evaluated only while all compared tenants "
        "are backlogged at the relevant stage.", "",
        "| Phase | Case | Variant | Runs | Errors | Victim P95 | Prep lead | "
        "Inference service lead | Throughput |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['case']} | {row['variant']} | "
            f"{row['runs']} | {row['errors']} | "
            f"{row['victim_mean_p95_e2e_s']:.2f}s | "
            f"{row['prep_peak_service_lead_s']:.2f}s | "
            f"{row['engine_peak_service_lead_s']:.2f}s | "
            f"{row['throughput_qps']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def ordered_variants(rows: list[dict]) -> list[str]:
    present = {row["variant"] for row in rows}
    return [variant for variant in POLICY_ORDER if variant in present] + sorted(
        present.difference(POLICY_ORDER)
    )


def plot_categorical(
    rows: list[dict], *, phase: str, field: str, ylabel: str,
    title: str, output: Path, case_order: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    selected = [row for row in rows if row["phase"] == phase and finite(row.get(field))]
    if not selected:
        return
    cases = case_order or sorted({row["case"] for row in selected})
    variants = ordered_variants(selected)
    lookup = {(row["case"], row["variant"]): float(row[field]) for row in selected}
    x = np.arange(len(cases), dtype=float)
    width = 0.78 / max(1, len(variants))
    fig, axis = plt.subplots(figsize=(max(7.2, 1.25 * len(cases)), 4.2), constrained_layout=True)
    for index, variant in enumerate(variants):
        positions = x + (index - (len(variants) - 1) / 2) * width
        heights = [lookup.get((case, variant), math.nan) for case in cases]
        axis.bar(
            positions, heights, width=width, label=LABELS.get(variant, variant),
            color=COLORS.get(variant), edgecolor="white", linewidth=0.5,
        )
    axis.set_xticks(x, [case.replace("_", " ") for case in cases])
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=min(4, len(variants)))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cross = load_cross_analyzer()
    paths = sorted(args.root.glob("*/*/*/summary.json"))
    if not paths:
        raise SystemExit(f"no completed suite runs below {args.root}")
    run_rows = [analyze_run(path, args.root, cross) for path in paths]
    aggregate_rows = aggregate(run_rows)
    write_csv(args.output.with_name(args.output.name + "_runs.csv"), run_rows)
    write_csv(args.output.with_suffix(".csv"), aggregate_rows)
    write_markdown(args.output.with_suffix(".md"), aggregate_rows)

    figures = args.root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_categorical(
        aggregate_rows, phase="stage", field="victim_mean_p95_e2e_s",
        ylabel="Mean victim P95 end-to-end latency (s)",
        title="Single-stage fairness fails when the bottleneck moves",
        output=figures / "stage_ablation",
        case_order=["prep_heavy", "inference_heavy", "mixed"],
    )
    plot_categorical(
        aggregate_rows, phase="fairness", field="prep_peak_service_lead_s",
        ylabel="Peak preparation-service lead (s; lower is fairer)",
        title="Service-aware max–min versus request-count scheduling",
        output=figures / "direct_service_fairness",
    )
    plot_categorical(
        aggregate_rows, phase="constant_rate", field="prep_peak_service_lead_s",
        ylabel="Peak preparation-service difference (worker-s)",
        title="Preparation fairness under unequal request rates",
        output=figures / "constant_rate_preparation_fairness",
    )
    plot_categorical(
        aggregate_rows, phase="constant_rate", field="engine_peak_service_lead_s",
        ylabel="Peak inference-service difference (s)",
        title="Inference-admission fairness under unequal request rates",
        output=figures / "constant_rate_inference_fairness",
    )
    plot_categorical(
        aggregate_rows, phase="stochastic", field="prep_peak_service_lead_s",
        ylabel="Peak preparation-service difference (worker-s)",
        title="Fairness with Poisson arrivals and heterogeneous videos",
        output=figures / "stochastic_preparation_fairness",
    )
    plot_categorical(
        aggregate_rows, phase="stochastic", field="engine_peak_service_lead_s",
        ylabel="Peak inference-service difference (s)",
        title="Inference fairness with Poisson heterogeneous workloads",
        output=figures / "stochastic_inference_fairness",
    )
    plot_categorical(
        aggregate_rows, phase="on_off", field="worst_tenant_p95_e2e_s",
        ylabel="Worst-tenant P95 end-to-end latency (s)",
        title="Intermittent and continuously backlogged tenants",
        output=figures / "on_off_latency",
        case_order=["on_off_two_tenant", "on_off_backlogged"],
    )
    plot_categorical(
        aggregate_rows, phase="work_conservation", field="throughput_qps",
        ylabel="Throughput (requests/s)",
        title="Work conservation with an underloaded tenant",
        output=figures / "work_conservation_throughput",
    )
    plot_categorical(
        aggregate_rows, phase="isolation", field="tenant_a_p95_e2e_s",
        ylabel="Victim tenant A P95 end-to-end latency (s)",
        title="Isolation from an aggressive tenant",
        output=figures / "noisy_neighbor_isolation",
    )
    plot_categorical(
        aggregate_rows, phase="load", field="victim_mean_p95_e2e_s",
        ylabel="Mean victim P95 end-to-end latency (s)",
        title="Sensitivity to offered-load scale",
        output=figures / "load_sensitivity",
        case_order=["load-0p5", "load-0p8", "load-1p0", "load-1p2", "load-1p5"],
    )
    plot_categorical(
        aggregate_rows, phase="heterogeneity", field="prep_peak_service_lead_s",
        ylabel="Peak preparation-service lead (s; lower is fairer)",
        title="Sensitivity to request-cost heterogeneity",
        output=figures / "heterogeneity_sensitivity",
        case_order=["heterogeneity-homogeneous", "heterogeneity-moderate", "heterogeneity-high"],
    )
    plot_categorical(
        aggregate_rows, phase="inflight", field="prep_peak_service_lead_s",
        ylabel="Peak preparation-service lead (s; lower is fairer)",
        title="In-flight accounting under increasing concurrency",
        output=figures / "inflight_accounting",
        case_order=["concurrency-1", "concurrency-2", "concurrency-4", "concurrency-8"],
    )
    print(json.dumps({
        "completed_runs": len(run_rows),
        "aggregate_csv": str(args.output.with_suffix('.csv')),
        "report": str(args.output.with_suffix('.md')),
        "figures": str(figures),
    }, indent=2))


if __name__ == "__main__":
    main()
