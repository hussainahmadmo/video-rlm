#!/usr/bin/env python3
"""Create one auditable CSV/Markdown report from priority experiment roots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


POLICY_ORDER = (
    "fcfs",
    "shared_engine_only",
    "priority",
    "shared_priority",
    "priority_reserved",
    "slo_adaptive",
    "static_isolation",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def metric(summary: dict[str, Any], workload: str, field: str) -> Any:
    return (summary.get(workload) or {}).get(field)


def run_record(root: Path, path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    results = load_jsonl(path.with_name("results.jsonl"))
    policy = path.parent.name
    run_id = str(path.parent.parent.relative_to(root))

    def maximum(workload: str, field: str) -> float | None:
        value = metric(summary, workload, f"max_{field}")
        if value is not None:
            return float(value)
        values = finite([
            row.get(field) for row in results if row.get("workload") == workload
        ])
        return max(values) if values else None

    return {
        "experiment_root": str(root),
        "run_id": run_id,
        "policy": policy,
        "model": summary.get("model") or "unknown",
        "decode_backend": summary.get("decode_backend") or "unknown",
        "prep_workers": summary.get("prep_workers"),
        "replicas": summary.get("replica_count", 1),
        "requests": summary.get("total_requests"),
        "errors": summary.get("errors"),
        "throughput_qps": summary.get("throughput_qps"),
        "urgent_mean_ttft_s": metric(summary, "urgent", "mean_end_to_end_ttft_s"),
        "urgent_p95_ttft_s": metric(summary, "urgent", "p95_end_to_end_ttft_s"),
        "urgent_max_ttft_s": maximum("urgent", "end_to_end_ttft_s"),
        "urgent_mean_prep_wait_s": metric(summary, "urgent", "mean_prep_queue_wait_s"),
        "urgent_slo_attainment_percent": metric(summary, "urgent", "ttft_slo_attainment_percent"),
        "urgent_accuracy_percent": metric(summary, "urgent", "accuracy_percent"),
        "background_mean_ttft_s": metric(summary, "background", "mean_end_to_end_ttft_s"),
        "background_p95_ttft_s": metric(summary, "background", "p95_end_to_end_ttft_s"),
        "background_max_ttft_s": maximum("background", "end_to_end_ttft_s"),
        "background_max_prep_wait_s": maximum("background", "prep_queue_wait_s"),
        "background_slo_attainment_percent": metric(summary, "background", "ttft_slo_attainment_percent"),
        "background_accuracy_percent": metric(summary, "background", "accuracy_percent"),
        "summary_path": str(path),
    }


def percent_change(new: Any, old: Any) -> float | None:
    if new is None or old in (None, 0):
        return None
    return 100.0 * (float(new) / float(old) - 1.0)


def pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(record["experiment_root"], record["run_id"])][record["policy"]] = record
    output = []
    for (root, run_id), policies in sorted(grouped.items()):
        baseline_name = "fcfs" if "fcfs" in policies else "shared_engine_only"
        if baseline_name not in policies:
            continue
        baseline = policies[baseline_name]
        for treatment_name in POLICY_ORDER:
            if treatment_name not in policies or treatment_name == baseline_name:
                continue
            treatment = policies[treatment_name]
            old = baseline["urgent_mean_ttft_s"]
            new = treatment["urgent_mean_ttft_s"]
            output.append({
                "experiment_root": root,
                "run_id": run_id,
                "baseline": baseline_name,
                "treatment": treatment_name,
                "urgent_ttft_speedup": (
                    float(old) / float(new) if old not in (None, 0) and new not in (None, 0) else None
                ),
                "urgent_ttft_reduction_percent": (
                    -percent_change(new, old) if percent_change(new, old) is not None else None
                ),
                "background_ttft_change_percent": percent_change(
                    treatment["background_mean_ttft_s"], baseline["background_mean_ttft_s"]
                ),
                "throughput_change_percent": percent_change(
                    treatment["throughput_qps"], baseline["throughput_qps"]
                ),
                "urgent_accuracy_change_points": (
                    float(treatment["urgent_accuracy_percent"]) - float(baseline["urgent_accuracy_percent"])
                    if treatment["urgent_accuracy_percent"] is not None
                    and baseline["urgent_accuracy_percent"] is not None else None
                ),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, suffix: str = "") -> str:
    return "--" if value is None else f"{float(value):.2f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for root in args.root:
        if not root.is_dir():
            raise SystemExit(f"experiment root does not exist: {root}")
        records.extend(run_record(root, path) for path in sorted(root.rglob("summary.json")))
    if not records:
        raise SystemExit("no summary.json files found")

    pair_rows = pairs(records)
    write_csv(args.output.with_suffix(".runs.csv"), records)
    write_csv(args.output.with_suffix(".pairs.csv"), pair_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[row["treatment"]].append(row)
    lines = [
        "# Paper evaluation report",
        "",
        f"Completed configurations: **{len(records)}**",
        f"Matched baseline/treatment pairs: **{len(pair_rows)}**",
        "",
        "| Treatment | Pairs | Geometric mean speedup | Mean urgent reduction | Mean background change | Mean throughput change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICY_ORDER:
        rows = grouped.get(policy, [])
        if not rows:
            continue
        speedups = finite([row["urgent_ttft_speedup"] for row in rows])
        geo = math.exp(statistics.mean(math.log(value) for value in speedups))
        urgent = finite([row["urgent_ttft_reduction_percent"] for row in rows])
        background = finite([row["background_ttft_change_percent"] for row in rows])
        throughput = finite([row["throughput_change_percent"] for row in rows])
        lines.append(
            f"| {policy} | {len(rows)} | {geo:.2f}x | "
            f"{fmt(statistics.mean(urgent), '%')} | "
            f"{fmt(statistics.mean(background), '%')} | "
            f"{fmt(statistics.mean(throughput), '%')} |"
        )
    lines.extend([
        "",
        "The CSV files retain per-run latency, maximum wait, SLO, accuracy, "
        "throughput, model, backend, worker, and replica settings. Aggregate "
        "means above should only be quoted when the supplied roots represent "
        "one intentionally matched experiment family.",
        "",
    ])
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({
        "runs": len(records),
        "pairs": len(pair_rows),
        "runs_csv": str(args.output.with_suffix('.runs.csv')),
        "pairs_csv": str(args.output.with_suffix('.pairs.csv')),
        "report": str(args.output.with_suffix('.md')),
    }, indent=2))


if __name__ == "__main__":
    main()
