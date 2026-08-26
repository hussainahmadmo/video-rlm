#!/usr/bin/env python3
"""Plot request-matched native-vLLM raw-video TTFT under CPU limits.

Failed requests are shown as censored timeout markers rather than removed.
The input trace defines a stable x position, so every condition compares the
same request at the same location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


POLICIES = (
    ("uniform", "Uniform engine priority"),
    ("priority", "Request priority"),
)
CONDITIONS = (
    ("unconstrained", "Unconstrained", "#666666", "D"),
    ("cpu16", "16 logical CPUs", "#0072B2", "^"),
    ("cpu4", "4 logical CPUs", "#D55E00", "o"),
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def result_path(root: Path, condition: str, policy: str) -> Path:
    if condition == "unconstrained":
        directory = "uniform_priority" if policy == "uniform" else "trace_priority"
        return root / directory / "results.jsonl"
    directory = "native_uniform" if policy == "uniform" else "native_priority"
    return root / directory / "results.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--unconstrained-run", required=True, type=Path)
    parser.add_argument("--cpu16-run", required=True, type=Path)
    parser.add_argument("--cpu4-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    roots = {
        "unconstrained": args.unconstrained_run,
        "cpu16": args.cpu16_run,
        "cpu4": args.cpu4_run,
    }
    trace = load_jsonl(args.trace)
    position = {str(row["request_id"]): i for i, row in enumerate(trace)}
    if len(position) != len(trace):
        raise SystemExit(f"Trace contains duplicate request IDs: {args.trace}")

    workload = {
        str(row["request_id"]): str(
            row.get("class") or row.get("workload") or "unknown"
        )
        for row in trace
    }
    urgent_positions = [
        position[request_id]
        for request_id, request_workload in workload.items()
        if request_workload == "urgent"
    ]
    if not urgent_positions:
        raise SystemExit("Trace contains no urgent requests")
    urgent_start = min(urgent_positions)

    data: dict[tuple[str, str], list[dict]] = {}
    expected = set(position)
    for policy, _ in POLICIES:
        for condition, _, _, _ in CONDITIONS:
            path = result_path(roots[condition], condition, policy)
            if not path.is_file():
                raise SystemExit(f"Missing results: {path}")
            rows = load_jsonl(path)
            observed = {str(row["request_id"]) for row in rows}
            if observed != expected:
                raise SystemExit(
                    f"Request mismatch for {condition}/{policy}: "
                    f"missing={len(expected-observed)}, extra={len(observed-expected)}"
                )
            data[(condition, policy)] = sorted(
                rows, key=lambda row: position[str(row["request_id"])]
            )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.15), sharey=True)
    for ax, (policy, policy_title) in zip(axes, POLICIES, strict=True):
        for condition, condition_label, color, marker in CONDITIONS:
            rows = data[(condition, policy)]
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
            urgent_ttft = [
                float(row["ttft_s"])
                for row in successful
                if workload[str(row["request_id"])] == "urgent"
            ]
            urgent_mean = mean(urgent_ttft) if urgent_ttft else float("nan")
            label = (
                f"{condition_label}: urgent mean {urgent_mean:.1f}s; "
                f"timeouts {len(failed)}"
            )

            x = np.array([position[str(row["request_id"])] for row in successful])
            y = np.array([float(row["ttft_s"]) for row in successful])
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markersize=3.0,
                linewidth=1.05,
                alpha=0.88,
                label=label,
            )
            if failed:
                failed_x = np.array(
                    [position[str(row["request_id"])] for row in failed]
                )
                failed_y = np.full(len(failed_x), args.timeout_s)
                ax.scatter(
                    failed_x,
                    failed_y,
                    color=color,
                    marker="X",
                    s=42,
                    linewidths=0.8,
                    edgecolors="black",
                    zorder=8,
                )

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
            fontsize=8,
        )
        ax.text(
            urgent_start + (len(trace) - urgent_start) / 2,
            0.975,
            f"Urgent: {len(trace)-urgent_start} requests at $t=10$s",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
        ax.set_title(policy_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Matched request arrival order")
        ax.set_yscale("log")
        ax.set_xlim(-1, len(trace))
        ax.set_ylim(1, args.timeout_s * 1.22)
        ax.grid(alpha=0.22, linewidth=0.6, which="both")
        ax.legend(loc="upper left", fontsize=7.4, framealpha=0.93)

    axes[0].set_ylabel("Native vLLM raw-video TTFT (seconds, log scale)")
    axes[1].text(
        0.99,
        0.02,
        f"X = request censored at {args.timeout_s:.0f}s timeout",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    fig.suptitle(
        "Native asynchronous video processing develops long tails under CPU limits",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "The same 79 requests appear at identical x positions; urgent means exclude censored timeouts.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
