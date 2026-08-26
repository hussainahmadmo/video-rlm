#!/usr/bin/env python3
"""Plot a matched native-vLLM raw-video priority experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("uniform_priority", "trace_priority")
LABELS = {
    "uniform_priority": "Native vLLM, uniform priority",
    "trace_priority": "Native vLLM, request priority",
}
COLORS = {
    "uniform_priority": "#D55E00",
    "trace_priority": "#009E73",
}
MARKERS = {"uniform_priority": "o", "trace_priority": "s"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Run containing uniform_priority/ and trace_priority/",
    )
    parser.add_argument(
        "--trace",
        required=True,
        type=Path,
        help="Arrival trace used by both policies",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    trace_rows = load_jsonl(args.trace)
    trace_position = {
        str(row["request_id"]): index for index, row in enumerate(trace_rows)
    }
    if len(trace_position) != len(trace_rows):
        raise SystemExit(f"Duplicate request IDs in {args.trace}")

    rows_by_policy: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for policy in POLICIES:
        result_path = args.run / policy / "results.jsonl"
        summary_path = args.run / policy / "summary.json"
        if not result_path.is_file() or not summary_path.is_file():
            raise SystemExit(
                f"{policy} is incomplete: expected {result_path} and {summary_path}"
            )
        rows = load_jsonl(result_path)
        errors = [row for row in rows if row.get("error")]
        if errors:
            raise SystemExit(f"{policy} contains {len(errors)} failed requests")
        observed = {str(row["request_id"]) for row in rows}
        expected = set(trace_position)
        if observed != expected:
            raise SystemExit(
                f"{policy} request mismatch: missing={len(expected - observed)}, "
                f"extra={len(observed - expected)}"
            )
        rows_by_policy[policy] = sorted(
            rows, key=lambda row: trace_position[str(row["request_id"])]
        )
        summaries[policy] = load_json(summary_path)

    workloads = [
        str(row.get("workload") or str(row["request_id"]).split("-", 1)[0])
        for row in trace_rows
    ]
    urgent_positions = [i for i, workload in enumerate(workloads) if workload == "urgent"]
    background_positions = [
        i for i, workload in enumerate(workloads) if workload == "background"
    ]
    if not urgent_positions or not background_positions:
        raise SystemExit("Trace must contain background and urgent requests")
    boundary = min(urgent_positions) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.55))

    # Request-level behavior preserves the original trace order rather than
    # completion order, which makes the arrival boundary explicit.
    ax = axes[0]
    for policy in POLICIES:
        rows = rows_by_policy[policy]
        x = np.array([trace_position[str(row["request_id"])] for row in rows])
        y = np.array([float(row["ttft_s"]) for row in rows])
        ax.plot(
            x,
            y,
            label=LABELS[policy],
            color=COLORS[policy],
            marker=MARKERS[policy],
            markersize=2.8,
            linewidth=1.2,
            alpha=0.92,
        )
    ax.axvline(boundary, color="#555555", linestyle="--", linewidth=1)
    ax.axvspan(-0.5, boundary, color="#777777", alpha=0.045)
    ax.axvspan(boundary, len(trace_rows) - 0.5, color="#E69F00", alpha=0.08)
    ax.text(
        np.mean(background_positions),
        0.97,
        f"Background: {len(background_positions)} at $t=0$",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
    )
    urgent_arrival = float(trace_rows[urgent_positions[0]].get("arrival_s", 0.0))
    ax.text(
        np.mean(urgent_positions),
        0.97,
        f"Urgent: {len(urgent_positions)} at $t={urgent_arrival:g}$s",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )
    ax.set_title("(a) Request-level end-to-end TTFT", fontweight="bold")
    ax.set_xlabel("Request arrival order")
    ax.set_ylabel("TTFT (seconds)")
    ax.set_xlim(-1, len(trace_rows))
    ax.grid(alpha=0.22, linewidth=0.6)

    # The urgent ECDF exposes both the common case and tail behavior without
    # inventing a preparation-queue metric that native vLLM does not export.
    ax = axes[1]
    metric_lines = []
    for policy in POLICIES:
        urgent = [
            float(row["ttft_s"])
            for row in rows_by_policy[policy]
            if row["workload"] == "urgent"
        ]
        ordered = np.sort(urgent)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.step(
            ordered,
            cdf,
            where="post",
            label=LABELS[policy],
            color=COLORS[policy],
            linewidth=2.0,
        )
        mean_ttft = float(np.mean(urgent))
        median_ttft = float(np.median(urgent))
        metric_lines.append(
            f"{LABELS[policy]}: mean {mean_ttft:.1f}s, median {median_ttft:.1f}s"
        )
    uniform_mean = float(summaries["uniform_priority"]["urgent"]["mean_ttft_s"])
    priority_mean = float(summaries["trace_priority"]["urgent"]["mean_ttft_s"])
    speedup = uniform_mean / priority_mean
    reduction = 100.0 * (1.0 - priority_mean / uniform_mean)
    metric_lines.append(f"Mean improvement: {speedup:.2f}x ({reduction:.1f}% lower)")
    ax.text(
        0.98,
        0.04,
        "\n".join(metric_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.92},
    )
    ax.set_title("(b) Urgent-request TTFT distribution", fontweight="bold")
    ax.set_xlabel("Urgent TTFT (seconds)")
    ax.set_ylabel("Fraction of urgent requests completed")
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.22, linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Native vLLM priority with raw-video inputs",
        y=1.11,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "No client-side decoding or preparation workers; vLLM fetches and processes every raw video URL.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(args.output)
    print(pdf_path)


if __name__ == "__main__":
    main()
