#!/usr/bin/env python3
"""Create a cleaned mixed-priority result tree with selected QIDs removed.

The input tree is never modified. Each directory containing ``results.jsonl``
is mirrored below the output tree. Mixed-priority summaries are recomputed from
the retained request rows, including workload statistics, SLO attainment, error
counts, accuracy, and throughput.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "prep_queue_wait_s",
    "prep_service_s",
    "prepared_queue_wait_s",
    "engine_to_first_token_s",
    "end_to_end_ttft_s",
    "end_to_end_s",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row.get("correct")) for row in rows)
    result: dict[str, Any] = {
        "requests": len(rows),
        "errors": sum(row.get("error") is not None for row in rows),
        "correct": correct,
        "accuracy_percent": 100.0 * correct / len(rows) if rows else None,
    }
    for field in METRIC_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[f"mean_{field}"] = mean(values)
        result[f"p50_{field}"] = percentile(values, 0.50)
        result[f"p95_{field}"] = percentile(values, 0.95)
    slo_rows = [row for row in rows if row.get("ttft_slo_s") is not None]
    result["ttft_slo_requests"] = len(slo_rows)
    result["ttft_slo_attained"] = sum(
        bool(row.get("slo_attained")) for row in slo_rows
    )
    result["ttft_slo_attainment_percent"] = (
        100.0 * result["ttft_slo_attained"] / len(slo_rows)
        if slo_rows
        else None
    )
    return result


def row_qid(row: dict[str, Any]) -> str | None:
    value = row.get("qid") or row.get("question_id")
    return None if value is None else str(value)


def clean_run(
    run_dir: Path,
    input_root: Path,
    output_root: Path,
    excluded_qids: set[str],
) -> dict[str, Any]:
    rows = load_jsonl(run_dir / "results.jsonl")
    removed = [row for row in rows if row_qid(row) in excluded_qids]
    kept = [row for row in rows if row_qid(row) not in excluded_qids]
    relative = run_dir.relative_to(input_root)
    destination = output_root / relative
    write_jsonl(destination / "results.jsonl", kept)

    removed_request_ids = {
        str(row["request_id"]) for row in removed if row.get("request_id") is not None
    }
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        events = load_jsonl(events_path)
        events = [
            event
            for event in events
            if str(event.get("request_id")) not in removed_request_ids
        ]
        write_jsonl(destination / "events.jsonl", events)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in kept:
            by_workload[str(row.get("workload", "background"))].append(row)

        summary["total_requests"] = len(kept)
        summary["errors"] = sum(row.get("error") is not None for row in kept)
        wall_time = float(summary.get("wall_time_s") or 0.0)
        summary["throughput_qps"] = len(kept) / wall_time if wall_time else 0.0
        if "background" in summary:
            summary["background"] = summarize(by_workload["background"])
        if "urgent" in summary:
            summary["urgent"] = summarize(by_workload["urgent"])

        configuration = summary.get("configuration")
        if isinstance(configuration, dict):
            if configuration.get("trace_requests") is not None:
                configuration["trace_requests"] = len(kept)
            trace_workloads = configuration.get("trace_workloads")
            if isinstance(trace_workloads, dict):
                configuration["trace_workloads"] = {
                    workload: len(workload_rows)
                    for workload, workload_rows in sorted(by_workload.items())
                }

        summary["result_filter"] = {
            "excluded_qids": sorted(excluded_qids),
            "removed_requests": len(removed),
            "source_results": str(run_dir / "results.jsonl"),
            "raw_results_preserved": True,
        }
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    return {
        "run": str(relative),
        "input_requests": len(rows),
        "retained_requests": len(kept),
        "removed_requests": len(removed),
        "removed_workloads": dict(
            sorted(
                (workload, sum(row.get("workload") == workload for row in removed))
                for workload in {str(row.get("workload")) for row in removed}
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-qid", action="append", required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        parser.error("input and output roots must differ")
    if output_root.is_relative_to(input_root):
        parser.error("output root cannot be inside input root")

    runs = sorted(path.parent for path in input_root.rglob("results.jsonl"))
    if not runs:
        raise SystemExit(f"no results.jsonl files found below {input_root}")

    excluded_qids = set(args.exclude_qid)
    records = [
        clean_run(run, input_root, output_root, excluded_qids) for run in runs
    ]
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "excluded_qids": sorted(excluded_qids),
        "runs": len(records),
        "affected_runs": sum(record["removed_requests"] > 0 for record in records),
        "input_requests": sum(record["input_requests"] for record in records),
        "retained_requests": sum(record["retained_requests"] for record in records),
        "removed_requests": sum(record["removed_requests"] for record in records),
        "run_records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cleaning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "run_records"}, indent=2))


if __name__ == "__main__":
    main()
