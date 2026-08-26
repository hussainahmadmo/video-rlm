#!/usr/bin/env python3
"""Join latency and process-tree CPU metrics for the CPU-budget sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDS = (
    "cpu_budget",
    "cpu_set",
    "repeat",
    "order_index",
    "policy",
    "requests",
    "errors",
    "urgent_mean_ttft_s",
    "urgent_p95_ttft_s",
    "background_mean_ttft_s",
    "background_p95_ttft_s",
    "throughput_qps",
    "urgent_accuracy_percent",
    "mean_cpu_cores",
    "p95_cpu_cores",
    "max_cpu_cores",
    "mean_threads",
    "max_threads",
    "cpu_seconds_per_request",
    "summary_path",
)


def value(group: dict[str, Any], native_field: str, external_field: str) -> Any:
    return group.get(native_field, group.get(external_field))


def finite(values: list[Any]) -> list[float]:
    output = []
    for item in values:
        if item is None:
            continue
        number = float(item)
        if math.isfinite(number):
            output.append(number)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in sorted(args.root.glob("cpu*/repeat*/*/summary.json")):
        cpu_path = path.with_name("cpu_summary.json")
        if not cpu_path.is_file():
            continue
        summary = json.loads(path.read_text())
        cpu = json.loads(cpu_path.read_text())
        urgent = summary.get("urgent") or {}
        background = summary.get("background") or {}
        records.append(
            {
                "cpu_budget": cpu.get("configured_cpu_budget"),
                "cpu_set": cpu.get("cpu_set"),
                "repeat": cpu.get("repeat"),
                "order_index": cpu.get("order_index"),
                "policy": cpu.get("policy"),
                "requests": summary.get("total_requests"),
                "errors": summary.get("errors"),
                "urgent_mean_ttft_s": value(
                    urgent, "mean_ttft_s", "mean_end_to_end_ttft_s"
                ),
                "urgent_p95_ttft_s": value(
                    urgent, "p95_ttft_s", "p95_end_to_end_ttft_s"
                ),
                "background_mean_ttft_s": value(
                    background, "mean_ttft_s", "mean_end_to_end_ttft_s"
                ),
                "background_p95_ttft_s": value(
                    background, "p95_ttft_s", "p95_end_to_end_ttft_s"
                ),
                "throughput_qps": summary.get("throughput_qps"),
                "urgent_accuracy_percent": urgent.get("accuracy_percent"),
                "mean_cpu_cores": cpu.get("mean_cpu_cores"),
                "p95_cpu_cores": cpu.get("p95_cpu_cores"),
                "max_cpu_cores": cpu.get("max_cpu_cores"),
                "mean_threads": cpu.get("mean_threads"),
                "max_threads": cpu.get("max_threads"),
                "cpu_seconds_per_request": cpu.get("cpu_seconds_per_request"),
                "summary_path": str(path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["cpu_budget"]), str(record["policy"]))].append(record)

    aggregate_path = args.output.with_name(args.output.stem + "_aggregate.csv")
    aggregate_fields = (
        "cpu_budget",
        "policy",
        "runs",
        "urgent_mean_ttft_s",
        "urgent_p95_ttft_s",
        "throughput_qps",
        "mean_cpu_cores",
        "cpu_seconds_per_request",
        "max_threads",
    )
    aggregate: list[dict[str, Any]] = []
    for (budget, policy), items in sorted(grouped.items()):
        row: dict[str, Any] = {"cpu_budget": budget, "policy": policy, "runs": len(items)}
        for field in aggregate_fields[3:]:
            numbers = finite([item.get(field) for item in items])
            row[field] = statistics.mean(numbers) if numbers else None
        aggregate.append(row)
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)

    markdown_path = args.output.with_suffix(".md")
    with markdown_path.open("w") as handle:
        handle.write("# CPU-budget priority comparison\n\n")
        handle.write(
            "| CPUs | Policy | Runs | Urgent mean TTFT | Urgent p95 | "
            "Throughput | Mean cores | CPU-s/request | Max threads |\n"
        )
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in aggregate:
            def fmt(field: str, digits: int = 2) -> str:
                item = row.get(field)
                return "--" if item is None else f"{float(item):.{digits}f}"

            handle.write(
                f"| {row['cpu_budget']} | {row['policy']} | {row['runs']} | "
                f"{fmt('urgent_mean_ttft_s')} s | {fmt('urgent_p95_ttft_s')} s | "
                f"{fmt('throughput_qps', 3)} QPS | {fmt('mean_cpu_cores')} | "
                f"{fmt('cpu_seconds_per_request', 1)} | {fmt('max_threads', 0)} |\n"
            )

    print(f"runs={len(records)}")
    print(args.output)
    print(aggregate_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
