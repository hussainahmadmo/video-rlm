#!/usr/bin/env python3
"""Plot per-request latency for one matched multimodal burst trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fcfs", "priority", "priority_reserved")
NATIVE_POLICIES = ("uniform_priority", "trace_priority")
LABELS = {
    "fcfs": "Engine-only priority",
    "priority": "End-to-end priority",
    "priority_reserved": "End-to-end priority + reserved worker",
}
NATIVE_LABELS = {
    "uniform_priority": "Native vLLM: uniform priority",
    "trace_priority": "Native vLLM: request priority",
}
COLORS = {
    "fcfs": "#D55E00",
    "priority": "#009E73",
    "priority_reserved": "#0072B2",
}
MARKERS = {"fcfs": "o", "priority": "s", "priority_reserved": "^"}
NATIVE_COLORS = {
    "uniform_priority": "#666666",
    "trace_priority": "#CC79A7",
}
NATIVE_MARKERS = {"uniform_priority": "D", "trace_priority": "X"}


def load_rows(path: Path) -> list[dict]:
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
        help="Matched run containing fcfs/, priority/, and priority_reserved/",
    )
    parser.add_argument(
        "--trace",
        required=True,
        type=Path,
        help=(
            "Arrival-trace JSONL used by the matched run. Its line order is "
            "used to break equal-arrival-time ties."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--native-run",
        type=Path,
        help=(
            "Optional native raw-video run containing uniform_priority/ and "
            "trace_priority/. When supplied, a third TTFT panel is added."
        ),
    )
    parser.add_argument(
        "--native-trace",
        type=Path,
        help="Arrival trace used by --native-run",
    )
    parser.add_argument("--background-count", type=int, default=64)
    parser.add_argument("--urgent-arrival-s", type=float, default=10.0)
    args = parser.parse_args()

    if bool(args.native_run) != bool(args.native_trace):
        raise SystemExit("--native-run and --native-trace must be supplied together")

    trace_rows = load_rows(args.trace)
    trace_position = {
        str(row["request_id"]): index for index, row in enumerate(trace_rows)
    }
    if len(trace_position) != len(trace_rows):
        raise SystemExit(f"Duplicate request IDs in trace: {args.trace}")

    by_policy: dict[str, list[dict]] = {}
    for policy in POLICIES:
        path = args.run / policy / "results.jsonl"
        if not path.is_file():
            raise SystemExit(f"Missing results: {path}")
        rows = [row for row in load_rows(path) if not row.get("error")]
        missing = [
            str(row["request_id"])
            for row in rows
            if str(row["request_id"]) not in trace_position
        ]
        if missing:
            raise SystemExit(
                f"Results contain request IDs absent from trace: {missing[:5]}"
            )
        by_policy[policy] = sorted(
            rows,
            key=lambda row: trace_position[str(row["request_id"])],
        )

    expected = {
        row["request_id"] for row in by_policy[POLICIES[0]]
    }
    for policy in POLICIES[1:]:
        observed = {row["request_id"] for row in by_policy[policy]}
        if observed != expected:
            raise SystemExit(f"Request mismatch for policy {policy}")

    native_by_policy: dict[str, list[dict]] = {}
    native_summaries: dict[str, dict] = {}
    native_trace_rows: list[dict] = []
    native_trace_position: dict[str, int] = {}
    if args.native_run:
        native_trace_rows = load_rows(args.native_trace)
        native_trace_position = {
            str(row["request_id"]): index
            for index, row in enumerate(native_trace_rows)
        }
        if len(native_trace_position) != len(native_trace_rows):
            raise SystemExit(f"Duplicate request IDs in native trace: {args.native_trace}")
        native_expected = set(native_trace_position)
        for policy in NATIVE_POLICIES:
            result_path = args.native_run / policy / "results.jsonl"
            summary_path = args.native_run / policy / "summary.json"
            if not result_path.is_file() or not summary_path.is_file():
                raise SystemExit(
                    f"Incomplete native policy {policy}: expected {result_path} "
                    f"and {summary_path}"
                )
            native_rows = load_rows(result_path)
            errors = [row for row in native_rows if row.get("error")]
            if errors:
                raise SystemExit(
                    f"Native policy {policy} contains {len(errors)} failed requests"
                )
            observed = {str(row["request_id"]) for row in native_rows}
            if observed != native_expected:
                raise SystemExit(
                    f"Native request mismatch for {policy}: "
                    f"missing={len(native_expected - observed)}, "
                    f"extra={len(observed - native_expected)}"
                )
            native_by_policy[policy] = sorted(
                native_rows,
                key=lambda row: native_trace_position[str(row["request_id"])],
            )
            native_summaries[policy] = json.loads(summary_path.read_text())

    panel_count = 3 if args.native_run else 2
    figure_width = 15.0 if args.native_run else 10.2
    fig, axes = plt.subplots(1, panel_count, figsize=(figure_width, 3.25))
    metrics = (
        ("end_to_end_ttft_s", "(a) End-to-end TTFT", "TTFT (seconds)"),
        (
            "prep_queue_wait_s",
            "(b) Media-preparation queue wait",
            "Queue wait (seconds)",
        ),
    )

    for ax, (field, title, ylabel) in zip(axes[:2], metrics, strict=True):
        for policy in POLICIES:
            rows = by_policy[policy]
            x = np.array(
                [
                    trace_position[str(row["request_id"])]
                    for row in rows
                ]
            )
            y = np.array([float(row[field]) for row in rows])
            ax.plot(
                x,
                y,
                color=COLORS[policy],
                marker=MARKERS[policy],
                markersize=2.7,
                linewidth=1.15,
                label=LABELS[policy],
                alpha=0.95,
            )

        boundary = args.background_count - 0.5
        ax.axvline(boundary, color="#555555", linestyle="--", linewidth=1)
        ax.axvspan(-0.5, boundary, color="#777777", alpha=0.045)
        ax.axvspan(boundary, args.background_count + 15.5, color="#E69F00", alpha=0.08)
        ax.text(
            args.background_count / 2,
            0.97,
            "Background: 64 requests\narrival at $t=0$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
        )
        ax.text(
            args.background_count + 7.5,
            0.97,
            f"Urgent: 16 requests\narrival at $t={args.urgent_arrival_s:g}$s",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Request arrival order")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1, args.background_count + 16)
        ax.grid(alpha=0.22, linewidth=0.6)

    axes[0].text(
        0.985,
        0.62,
        "Urgent mean TTFT\n"
        "227.1s $\\rightarrow$ 50.9s (4.46$\\times$)\n"
        "Reserved: 40.1s (5.67$\\times$)",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )

    if args.native_run:
        ax = axes[2]
        for policy in NATIVE_POLICIES:
            native_rows = native_by_policy[policy]
            x = np.array(
                [
                    native_trace_position[str(row["request_id"])]
                    for row in native_rows
                ]
            )
            y = np.array([float(row["ttft_s"]) for row in native_rows])
            ax.plot(
                x,
                y,
                color=NATIVE_COLORS[policy],
                marker=NATIVE_MARKERS[policy],
                markersize=2.7,
                linewidth=1.15,
                label=NATIVE_LABELS[policy],
                alpha=0.95,
            )

        native_workloads = [
            str(row.get("workload") or str(row["request_id"]).split("-", 1)[0])
            for row in native_trace_rows
        ]
        native_urgent = [
            index
            for index, workload in enumerate(native_workloads)
            if workload == "urgent"
        ]
        native_background = [
            index
            for index, workload in enumerate(native_workloads)
            if workload == "background"
        ]
        native_boundary = min(native_urgent) - 0.5
        native_urgent_arrival = float(
            native_trace_rows[native_urgent[0]].get("arrival_s", 0.0)
        )
        ax.axvline(native_boundary, color="#555555", linestyle="--", linewidth=1)
        ax.axvspan(-0.5, native_boundary, color="#777777", alpha=0.045)
        ax.axvspan(
            native_boundary,
            len(native_trace_rows) - 0.5,
            color="#E69F00",
            alpha=0.08,
        )
        ax.text(
            np.mean(native_background),
            0.97,
            f"Background: {len(native_background)} at $t=0$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.2,
        )
        ax.text(
            np.mean(native_urgent),
            0.97,
            f"Urgent: {len(native_urgent)} at $t={native_urgent_arrival:g}$s",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
        )
        uniform_mean = float(
            native_summaries["uniform_priority"]["urgent"]["mean_ttft_s"]
        )
        request_mean = float(
            native_summaries["trace_priority"]["urgent"]["mean_ttft_s"]
        )
        native_speedup = uniform_mean / request_mean
        native_reduction = 100.0 * (1.0 - request_mean / uniform_mean)
        ax.text(
            0.985,
            0.62,
            "Urgent mean TTFT\n"
            f"{uniform_mean:.1f}s $\\rightarrow$ {request_mean:.1f}s "
            f"({native_speedup:.2f}$\\times$; {native_reduction:.1f}\\% lower)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.1,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
        )
        ax.set_title(
            "(c) Native vLLM raw-video TTFT",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Request arrival order")
        ax.set_ylabel("TTFT (seconds)")
        ax.set_xlim(-1, len(native_trace_rows))
        ax.grid(alpha=0.22, linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    if args.native_run:
        native_handles, native_labels = axes[2].get_legend_handles_labels()
        handles.extend(native_handles)
        labels.extend(native_labels)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Priority must begin before engine admission",
        y=1.13,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(args.output)
    print(pdf)


if __name__ == "__main__":
    main()
