#!/usr/bin/env python3
"""Combine external end-to-end policies and native-vLLM CPU-limit results.

Every panel uses the same request IDs and trace-defined x positions. Native
failures are retained as censored timeout markers rather than silently
discarded. The fourth panel summarizes urgent means and timeout counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


EXTERNAL = (
    ("fcfs", "Engine-only priority", "#D55E00", "o"),
    ("priority", "End-to-end priority", "#009E73", "s"),
    (
        "priority_reserved",
        "End-to-end priority + reserved worker",
        "#0072B2",
        "^",
    ),
)
NATIVE_CONDITIONS = (
    ("unconstrained", "Unconstrained", "#666666", "D"),
    ("cpu16", "16 logical CPUs", "#56B4E9", "^"),
    ("cpu4", "4 logical CPUs", "#CC79A7", "o"),
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def native_path(root: Path, condition: str, policy: str) -> Path:
    if condition == "unconstrained":
        name = "uniform_priority" if policy == "uniform" else "trace_priority"
    else:
        name = "native_uniform" if policy == "uniform" else "native_priority"
    return root / name / "results.jsonl"


def validate_and_sort(
    path: Path, positions: dict[str, int], label: str
) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing {label} results: {path}")
    rows = load_jsonl(path)
    observed = {str(row["request_id"]) for row in rows}
    expected = set(positions)
    if observed != expected:
        raise SystemExit(
            f"Request mismatch for {label}: missing={len(expected-observed)}, "
            f"extra={len(observed-expected)}"
        )
    return sorted(rows, key=lambda row: positions[str(row["request_id"])])


def decorate_trace_axis(
    ax: plt.Axes, trace: list[dict], urgent_start: int, timeout_s: float
) -> None:
    boundary = urgent_start - 0.5
    ax.axvline(boundary, color="#555555", linestyle="--", linewidth=1)
    ax.axvspan(-0.5, boundary, color="#777777", alpha=0.045)
    ax.axvspan(boundary, len(trace) - 0.5, color="#E69F00", alpha=0.08)
    ax.text(
        urgent_start / 2,
        0.975,
        f"Background: {urgent_start} requests at $t=0$",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
    )
    ax.text(
        urgent_start + (len(trace) - urgent_start) / 2,
        0.975,
        f"Urgent: {len(trace)-urgent_start} requests at $t=10$s",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )
    ax.set_xlim(-1, len(trace))
    ax.set_ylim(1, timeout_s * 1.22)
    ax.set_yscale("log")
    ax.grid(alpha=0.22, linewidth=0.6, which="both")
    ax.set_xlabel("Matched request arrival order")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--external-run", required=True, type=Path)
    parser.add_argument("--native-unconstrained-run", required=True, type=Path)
    parser.add_argument("--native-cpu16-run", required=True, type=Path)
    parser.add_argument("--native-cpu4-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    trace = load_jsonl(args.trace)
    positions = {str(row["request_id"]): i for i, row in enumerate(trace)}
    if len(positions) != len(trace):
        raise SystemExit(f"Duplicate request IDs in {args.trace}")
    workload = {
        str(row["request_id"]): str(
            row.get("class") or row.get("workload") or "unknown"
        )
        for row in trace
    }
    urgent_positions = [
        positions[request_id]
        for request_id, kind in workload.items()
        if kind == "urgent"
    ]
    if not urgent_positions:
        raise SystemExit("Trace contains no urgent requests")
    urgent_start = min(urgent_positions)

    external: dict[str, list[dict]] = {}
    for policy, label, _, _ in EXTERNAL:
        external[policy] = validate_and_sort(
            args.external_run / policy / "results.jsonl",
            positions,
            label,
        )

    native_roots = {
        "unconstrained": args.native_unconstrained_run,
        "cpu16": args.native_cpu16_run,
        "cpu4": args.native_cpu4_run,
    }
    native: dict[tuple[str, str], list[dict]] = {}
    for policy in ("uniform", "priority"):
        for condition, label, _, _ in NATIVE_CONDITIONS:
            native[(condition, policy)] = validate_and_sort(
                native_path(native_roots[condition], condition, policy),
                positions,
                f"native {policy}, {label}",
            )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 8.0))
    ax_external, ax_uniform, ax_priority, ax_summary = axes.flat

    summary: list[tuple[str, float, int, str]] = []
    for policy, label, color, marker in EXTERNAL:
        rows = external[policy]
        x = np.array([positions[str(row["request_id"])] for row in rows])
        y = np.array([float(row["end_to_end_ttft_s"]) for row in rows])
        urgent = [
            float(row["end_to_end_ttft_s"])
            for row in rows
            if workload[str(row["request_id"])] == "urgent"
        ]
        urgent_mean = mean(urgent)
        summary.append((label, urgent_mean, 0, color))
        ax_external.plot(
            x,
            y,
            color=color,
            marker=marker,
            markersize=2.8,
            linewidth=1.1,
            alpha=0.92,
            label=f"{label}: urgent mean {urgent_mean:.1f}s",
        )
    decorate_trace_axis(ax_external, trace, urgent_start, args.timeout_s)
    ax_external.set_title(
        "(a) Explicit preparation scheduling (4 prep workers)",
        fontweight="bold",
    )
    ax_external.set_ylabel("End-to-end TTFT (seconds, log scale)")
    ax_external.legend(loc="upper left", fontsize=7.2, framealpha=0.94)

    for ax, policy, title in (
        (ax_uniform, "uniform", "(b) Native raw video: uniform priority"),
        (ax_priority, "priority", "(c) Native raw video: request priority"),
    ):
        for condition, condition_label, color, marker in NATIVE_CONDITIONS:
            rows = native[(condition, policy)]
            successful = [
                row
                for row in rows
                if not row.get("error") and row.get("ttft_s") is not None
            ]
            failed = [
                row
                for row in rows
                if row.get("error") or row.get("ttft_s") is None
            ]
            urgent = [
                float(row["ttft_s"])
                for row in successful
                if workload[str(row["request_id"])] == "urgent"
            ]
            urgent_mean = mean(urgent) if urgent else float("nan")
            combined_label = (
                f"Native {condition_label}, "
                f"{'uniform' if policy == 'uniform' else 'request priority'}"
            )
            summary.append((combined_label, urgent_mean, len(failed), color))
            x = np.array(
                [positions[str(row["request_id"])] for row in successful]
            )
            y = np.array([float(row["ttft_s"]) for row in successful])
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markersize=2.8,
                linewidth=1.05,
                alpha=0.88,
                label=(
                    f"{condition_label}: urgent mean {urgent_mean:.1f}s; "
                    f"timeouts {len(failed)}"
                ),
            )
            if failed:
                ax.scatter(
                    [positions[str(row["request_id"])] for row in failed],
                    [args.timeout_s] * len(failed),
                    color=color,
                    marker="X",
                    s=42,
                    linewidths=0.8,
                    edgecolors="black",
                    zorder=8,
                )
        decorate_trace_axis(ax, trace, urgent_start, args.timeout_s)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Native vLLM TTFT (seconds, log scale)")
        ax.legend(loc="upper left", fontsize=7.2, framealpha=0.94)

    # Keep the summary compact: three external policies, then paired native
    # uniform/request-priority bars for each CPU condition.
    selected = [summary[i] for i in (0, 1, 2, 3, 6, 4, 7, 5, 8)]
    names = [
        "Engine\nonly",
        "End-to-end\npriority",
        "Reserved\nworker",
        "Native U\nunconstr.",
        "Native P\nunconstr.",
        "Native U\n16 CPU",
        "Native P\n16 CPU",
        "Native U\n4 CPU",
        "Native P\n4 CPU",
    ]
    x = np.arange(len(selected))
    bars = ax_summary.bar(
        x,
        [item[1] for item in selected],
        color=[item[3] for item in selected],
        edgecolor="#333333",
        linewidth=0.55,
    )
    for bar, (_, value, failures, _) in zip(bars, selected, strict=True):
        text = f"{value:.1f}s"
        if failures:
            text += f"\n{failures} TO"
            bar.set_hatch("//")
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.06,
            text,
            ha="center",
            va="bottom",
            fontsize=7.3,
        )
    ax_summary.set_xticks(x, names, fontsize=7.2)
    ax_summary.set_yscale("log")
    ax_summary.set_ylim(10, 500)
    ax_summary.set_ylabel("Urgent mean TTFT (seconds, log scale)")
    ax_summary.set_title("(d) Urgent-request summary", fontweight="bold")
    ax_summary.grid(axis="y", alpha=0.22, which="both")
    ax_summary.text(
        0.99,
        0.98,
        "U = uniform; P = request priority; TO = all-request timeouts\n"
        "Native means exclude censored timeouts",
        transform=ax_summary.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#444444",
    )

    fig.suptitle(
        "Priority before engine admission: explicit scheduling and native raw video",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Same 79 request IDs, workload classes, frame budgets, and arrival order in every panel. "
        "X marks a request censored at the 1,800-second timeout.",
        ha="center",
        fontsize=8.4,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(args.output)
    print(pdf)


if __name__ == "__main__":
    main()
