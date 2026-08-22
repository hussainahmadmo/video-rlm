#!/usr/bin/env python3
"""Create METIS-style accuracy/delay and delay/load plots from dynamic runs.

Example:
  python plot_video_serving_tradeoff.py --output figures \
    --run fixed8=/path/fixed8_r0.25 --run fixed8=/path/fixed8_r0.5 \
    --run adaptive=/path/adaptive_r0.25 --run adaptive=/path/adaptive_r0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[round((len(values) - 1) * fraction)])


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be METHOD=/path/to/run")
    method, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not method or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return method, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics: list[dict[str, Any]] = []
    for method, run_dir in args.run:
        summary_path = run_dir / "summary.json"
        results_path = run_dir / "results.jsonl"
        if not summary_path.is_file() or not results_path.is_file():
            parser.error(f"run is missing summary/results: {run_dir}")
        summary = json.loads(summary_path.read_text())
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in load_jsonl(results_path):
            by_dataset[str(row.get("dataset") or "unknown")].append(row)
        for dataset, rows in sorted(by_dataset.items()):
            delays = [
                float(row.get("end_to_end_delay_s", row.get("latency_s", 0.0)) or 0.0)
                for row in rows
            ]
            correct = sum(row.get("correct") is True for row in rows)
            errors = sum(row.get("error") not in (None, False) for row in rows)
            metrics.append({
                "method": method,
                "run_dir": str(run_dir),
                "dataset": dataset,
                "arrival_rate_qps": summary.get("arrival_rate_qps"),
                "questions": len(rows),
                "correct": correct,
                "errors": errors,
                "accuracy_percent": 100.0 * correct / len(rows) if rows else 0.0,
                "error_percent": 100.0 * errors / len(rows) if rows else 0.0,
                "mean_end_to_end_delay_s": sum(delays) / len(delays) if delays else 0.0,
                "p50_end_to_end_delay_s": percentile(delays, 0.50),
                "p95_end_to_end_delay_s": percentile(delays, 0.95),
                "run_throughput_qps": summary.get("throughput_qps"),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    columns = list(metrics[0]) if metrics else []
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metrics)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"metrics.csv was written; install matplotlib for plots: {exc}")

    datasets = sorted({row["dataset"] for row in metrics})
    methods = sorted({row["method"] for row in metrics})
    colors = {method: f"C{index}" for index, method in enumerate(methods)}
    columns_per_row = min(3, max(1, len(datasets)))
    rows_per_figure = math.ceil(len(datasets) / columns_per_row)

    fig, axes = plt.subplots(rows_per_figure, columns_per_row, figsize=(5 * columns_per_row, 4 * rows_per_figure), squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        for method in methods:
            points = [row for row in metrics if row["dataset"] == dataset and row["method"] == method]
            axis.scatter(
                [row["mean_end_to_end_delay_s"] for row in points],
                [row["accuracy_percent"] for row in points],
                label=method, color=colors[method], s=45,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Mean end-to-end delay (s)")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(datasets):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.output / "quality_vs_delay.png", dpi=180)

    fig, axes = plt.subplots(rows_per_figure, columns_per_row, figsize=(5 * columns_per_row, 4 * rows_per_figure), squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        for method in methods:
            points = [
                row for row in metrics
                if row["dataset"] == dataset and row["method"] == method and row["arrival_rate_qps"] is not None
            ]
            points.sort(key=lambda row: float(row["arrival_rate_qps"]))
            if points:
                axis.plot(
                    [row["arrival_rate_qps"] for row in points],
                    [row["mean_end_to_end_delay_s"] for row in points],
                    marker="o", label=method, color=colors[method],
                )
        axis.set_title(dataset)
        axis.set_xlabel("Offered request rate (q/s)")
        axis.set_ylabel("Mean end-to-end delay (s)")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(datasets):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.output / "delay_under_load.png", dpi=180)
    print(f"wrote {args.output / 'metrics.csv'}")
    print(f"wrote {args.output / 'quality_vs_delay.png'}")
    print(f"wrote {args.output / 'delay_under_load.png'}")


if __name__ == "__main__":
    main()
