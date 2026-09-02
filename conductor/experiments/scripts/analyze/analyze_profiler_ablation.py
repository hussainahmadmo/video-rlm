#!/usr/bin/env python3
"""Compare matched frame-only and metadata-aware profiler runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def number(value: Any) -> float | None:
    return None if value is None else float(value)


def fmt(value: float | None, suffix: str = "") -> str:
    return "--" if value is None else f"{value:.2f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for summary_path in sorted(args.root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        results = load_jsonl(summary_path.with_name("results.jsonl"))
        valid = [row for row in results if not row.get("error")]
        configuration = summary.get("configuration") or {}
        prep_profile = configuration.get("prep_cost_profile") or {}
        engine_profile = configuration.get("engine_cost_profile") or {}
        variant = summary_path.parent.name
        policy = str(summary.get("prep_policy") or variant.split("_", 1)[0])
        profile = variant.removeprefix(policy + "_")

        tenant_completion: dict[str, list[float]] = defaultdict(list)
        tenant_slowdown: dict[str, list[float]] = defaultdict(list)
        for row in valid:
            tenant = str(row.get("tenant", "default"))
            if row.get("end_to_end_s") is not None:
                tenant_completion[tenant].append(float(row["end_to_end_s"]))
            if row.get("estimated_slowdown") is not None:
                tenant_slowdown[tenant].append(float(row["estimated_slowdown"]))

        completion_means = [mean(values) for values in tenant_completion.values()]
        completion_means = [value for value in completion_means if value is not None]
        slowdown_means = [mean(values) for values in tenant_slowdown.values()]
        slowdown_means = [value for value in slowdown_means if value is not None]

        records.append({
            "trace": summary_path.parents[1].name,
            "policy": policy,
            "profile": profile,
            "prep_cost_profiler": configuration.get("prep_cost_profiler"),
            "engine_cost_profiler": configuration.get("engine_cost_profiler"),
            "requests": summary.get("total_requests"),
            "errors": summary.get("errors"),
            "throughput_qps": summary.get("throughput_qps"),
            "foreground_mean_ttft_s": summary.get("urgent", {}).get("mean_end_to_end_ttft_s"),
            "foreground_mean_completion_s": summary.get("urgent", {}).get("mean_end_to_end_s"),
            "background_mean_completion_s": summary.get("background", {}).get("mean_end_to_end_s"),
            "foreground_accuracy_percent": summary.get("urgent", {}).get("accuracy_percent"),
            "prep_mae_s": prep_profile.get("mean_absolute_error_s"),
            "prep_mape_percent": prep_profile.get("mean_absolute_percentage_error_percent"),
            "engine_mae_s": engine_profile.get("mean_absolute_error_s"),
            "engine_mape_percent": engine_profile.get("mean_absolute_percentage_error_percent"),
            "tenant_completion_ratio": (
                max(completion_means) / min(completion_means)
                if completion_means and min(completion_means) > 0 else None
            ),
            "tenant_slowdown_ratio": (
                max(slowdown_means) / min(slowdown_means)
                if slowdown_means and min(slowdown_means) > 0 else None
            ),
        })

    if not records:
        raise SystemExit("no completed profiler-ablation runs found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["policy"]), str(record["profile"]))].append(record)

    fields = [
        "prep_mae_s", "prep_mape_percent", "engine_mae_s",
        "engine_mape_percent", "foreground_mean_ttft_s",
        "foreground_mean_completion_s", "background_mean_completion_s",
        "tenant_completion_ratio", "tenant_slowdown_ratio",
        "throughput_qps", "foreground_accuracy_percent",
    ]
    aggregate: list[dict[str, Any]] = []
    for (policy, profile), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "policy": policy,
            "profile": profile,
            "runs": len(rows),
            "errors": sum(int(row["errors"] or 0) for row in rows),
        }
        for field in fields:
            item[field] = mean([
                float(row[field]) for row in rows if row[field] is not None
            ])
        aggregate.append(item)

    aggregate_path = args.output.with_name(args.output.name + "_aggregate.csv")
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    lines = [
        "# Metadata-profiler ablation",
        "",
        "| Policy | Profile | Runs | Errors | Prep MAPE | Engine MAPE | Foreground completion | Background completion | Tenant completion ratio | Throughput |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['policy']} | {row['profile']} | {row['runs']} "
            f"| {row['errors']} | {fmt(number(row['prep_mape_percent']), '%')} "
            f"| {fmt(number(row['engine_mape_percent']), '%')} "
            f"| {fmt(number(row['foreground_mean_completion_s']), 's')} "
            f"| {fmt(number(row['background_mean_completion_s']), 's')} "
            f"| {fmt(number(row['tenant_completion_ratio']), 'x')} "
            f"| {fmt(number(row['throughput_qps']))} |"
        )
    report_path = args.output.with_suffix(".md")
    report_path.write_text("\n".join(lines) + "\n")

    policies = sorted({str(row["policy"]) for row in aggregate})
    profiles = ["frame_frame", "metadata_frame", "metadata_metadata"]
    labels = {
        "frame_frame": "Frame / frame",
        "metadata_frame": "Metadata / frame",
        "metadata_metadata": "Metadata / metadata",
    }
    colors = {
        "frame_frame": "#d55e00",
        "metadata_frame": "#009e73",
        "metadata_metadata": "#0072b2",
    }
    lookup = {
        (str(row["policy"]), str(row["profile"])): row
        for row in aggregate
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    width = 0.24
    x = list(range(len(policies)))
    panels = [
        ("prep_mape_percent", "Preparation prediction MAPE (%)"),
        ("foreground_mean_completion_s", "Foreground completion time (s)"),
        ("tenant_slowdown_ratio", "Tenant slowdown ratio (max/min)"),
    ]
    for offset, profile in enumerate(profiles):
        values_by_panel = []
        for metric, _ in panels:
            values_by_panel.append([
                lookup.get((policy, profile), {}).get(metric)
                for policy in policies
            ])
        positions = [value + (offset - 1) * width for value in x]
        for axis, values, (_, ylabel) in zip(axes, values_by_panel, panels):
            axis.bar(
                positions,
                [float(value) if value is not None else float("nan") for value in values],
                width=width,
                label=labels[profile],
                color=colors[profile],
            )
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xticks(x, [value.replace("_", "\n") for value in policies])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Matched cost-profiler ablation: where does metadata help?",
        fontweight="bold",
    )
    fig.tight_layout()
    plot_path = args.output.with_suffix(".pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "completed_runs": len(records),
        "raw_csv": str(csv_path),
        "aggregate_csv": str(aggregate_path),
        "report": str(report_path),
        "plot": str(plot_path),
    }, indent=2))


if __name__ == "__main__":
    main()
