#!/usr/bin/env python3
"""Plot measured CPU and GPU service profiles with dual y axes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ORANGE = "#E69F00"
BLUE = "#56B4E9"


def read_cpu_profiles(root: Path, case_prefix: str) -> list[dict[str, Any]]:
    values: dict[int, list[float]] = defaultdict(list)
    for path in sorted(root.rglob("summary.json")):
        relative = path.relative_to(root)
        if case_prefix and not relative.parts[0].startswith(case_prefix):
            continue
        summary = json.loads(path.read_text())
        learned = (
            summary.get("configuration", {})
            .get("learned_prep_cost_s_by_frame_count", {})
        )
        for frames, service_s in learned.items():
            values[int(frames)].append(float(service_s))
    if not values:
        raise ValueError(f"no learned CPU preparation profiles under {root}")
    rows = []
    for frames, samples in sorted(values.items()):
        mean_service = statistics.mean(samples)
        rows.append({
            "frames": frames,
            "runs": len(samples),
            "service_s_mean": mean_service,
            "service_s_stdev": (
                statistics.stdev(samples) if len(samples) > 1 else 0.0
            ),
            "capacity_requests_per_minute_per_worker": 60.0 / mean_service,
        })
    return rows


def read_gpu_profile(
    aggregate_path: Path, raw_path: Path, output_tokens: int,
) -> list[dict[str, Any]]:
    aggregate = json.loads(aggregate_path.read_text())
    points = [
        point for point in aggregate.get("aggregated_points", [])
        if point.get("family") == "decode"
        and int(point.get("target", -1)) == output_tokens
    ]
    if not points:
        raise ValueError(
            f"no decode profile for {output_tokens} output tokens in {aggregate_path}"
        )

    samples: dict[int, list[float]] = defaultdict(list)
    if raw_path.exists():
        with raw_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("family") == "decode"
                    and int(row.get("target", -1)) == output_tokens
                ):
                    samples[int(row["concurrency"])].append(
                        float(row["amortized_service_s"])
                    )

    rows = []
    for point in sorted(points, key=lambda item: int(item["concurrency"])):
        concurrency = int(point["concurrency"])
        service_s = float(point["amortized_service_s"])
        repetitions = samples.get(concurrency, [])
        rows.append({
            "concurrency": concurrency,
            "output_tokens": output_tokens,
            "repetitions": len(repetitions),
            "amortized_service_s": service_s,
            "amortized_service_s_stdev": (
                statistics.stdev(repetitions) if len(repetitions) > 1 else 0.0
            ),
            "aggregate_output_tokens_per_second": output_tokens / service_s,
        })
    return rows


def style_axis(axis: Any) -> None:
    axis.grid(axis="y", alpha=0.3, linestyle="--")
    axis.spines["top"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--cpu-case-prefix", default="prep_cost")
    parser.add_argument("--gpu-profile", type=Path, required=True)
    parser.add_argument("--gpu-raw", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cpu = read_cpu_profiles(args.cpu_root, args.cpu_case_prefix)
    gpu = read_gpu_profile(args.gpu_profile, args.gpu_raw, args.output_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.18, top=0.72, wspace=0.40)

    cpu_axis = axes[0]
    cpu_cost_axis = cpu_axis.twinx()
    cpu_x = [row["frames"] for row in cpu]
    cpu_capacity = [row["capacity_requests_per_minute_per_worker"] for row in cpu]
    cpu_cost = [row["service_s_mean"] for row in cpu]
    cpu_cost_error = [row["service_s_stdev"] for row in cpu]
    cpu_axis.plot(
        cpu_x, cpu_capacity, color=ORANGE, marker="x", linewidth=2.2,
        markersize=7, label="Preparation capacity",
    )
    cpu_cost_axis.errorbar(
        cpu_x, cpu_cost, yerr=cpu_cost_error, color=BLUE, marker="x",
        linewidth=2.2, markersize=7, capsize=3, label="Preparation cost",
    )
    cpu_axis.set_xlabel("Sampled frames per request")
    cpu_axis.set_ylabel("Capacity (requests/min/worker)")
    cpu_cost_axis.set_ylabel("Preparation service (worker-s/request)")
    cpu_axis.set_title("(a) CPU preparation", pad=10, fontweight="bold")
    style_axis(cpu_axis)

    gpu_axis = axes[1]
    gpu_cost_axis = gpu_axis.twinx()
    gpu_x = [row["concurrency"] for row in gpu]
    gpu_throughput = [row["aggregate_output_tokens_per_second"] for row in gpu]
    gpu_cost = [row["amortized_service_s"] for row in gpu]
    gpu_cost_error = [row["amortized_service_s_stdev"] for row in gpu]
    gpu_axis.plot(
        gpu_x, gpu_throughput, color=ORANGE, marker="x", linewidth=2.2,
        markersize=7, label="Decode throughput",
    )
    gpu_cost_axis.errorbar(
        gpu_x, gpu_cost, yerr=gpu_cost_error, color=BLUE, marker="x",
        linewidth=2.2, markersize=7, capsize=3,
        label="Amortized service cost",
    )
    gpu_axis.set_xlabel("Concurrent decoding requests")
    gpu_axis.set_ylabel("Aggregate output throughput (tokens/s)")
    gpu_cost_axis.set_ylabel("Amortized service (s/request)")
    gpu_axis.set_title(
        f"(b) GPU decode ({args.output_tokens} output tokens)",
        pad=10,
        fontweight="bold",
    )
    style_axis(gpu_axis)

    handles = [cpu_axis.lines[0], cpu_cost_axis.lines[0]]
    labels = ["Throughput/capacity", "Profiled service cost"]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.90))
    fig.suptitle(
        "Platform Service Profiler: Preparation and Inference Cost",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.045,
        "Measured on the A40 platform; points are profile means/medians and "
        "error bars show variation across runs.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps({
        "cpu_preparation": cpu,
        "gpu_decode": gpu,
        "notes": {
            "cpu_capacity": "60 / mean worker-seconds per request",
            "gpu_throughput": "output tokens / amortized batch service seconds",
        },
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
