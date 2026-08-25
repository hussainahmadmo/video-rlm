#!/usr/bin/env python3
"""Plot per-request latency for one matched multimodal burst trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fcfs", "priority", "priority_reserved")
LABELS = {
    "fcfs": "Engine-only priority",
    "priority": "End-to-end priority",
    "priority_reserved": "End-to-end priority + reserved worker",
}
COLORS = {
    "fcfs": "#D55E00",
    "priority": "#009E73",
    "priority_reserved": "#0072B2",
}
MARKERS = {"fcfs": "o", "priority": "s", "priority_reserved": "^"}


def request_position(request_id: str, background_count: int) -> int:
    workload, index = request_id.rsplit("-", 1)
    position = int(index)
    if workload == "urgent":
        position += background_count
    return position


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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--background-count", type=int, default=64)
    parser.add_argument("--urgent-arrival-s", type=float, default=10.0)
    args = parser.parse_args()

    by_policy: dict[str, list[dict]] = {}
    for policy in POLICIES:
        path = args.run / policy / "results.jsonl"
        if not path.is_file():
            raise SystemExit(f"Missing results: {path}")
        rows = [row for row in load_rows(path) if not row.get("error")]
        by_policy[policy] = sorted(
            rows,
            key=lambda row: request_position(
                str(row["request_id"]), args.background_count
            ),
        )

    expected = {
        row["request_id"] for row in by_policy[POLICIES[0]]
    }
    for policy in POLICIES[1:]:
        observed = {row["request_id"] for row in by_policy[policy]}
        if observed != expected:
            raise SystemExit(f"Request mismatch for policy {policy}")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.25), sharex=True)
    metrics = (
        ("end_to_end_ttft_s", "(a) End-to-end TTFT", "TTFT (seconds)"),
        (
            "prep_queue_wait_s",
            "(b) Media-preparation queue wait",
            "Queue wait (seconds)",
        ),
    )

    for ax, (field, title, ylabel) in zip(axes, metrics, strict=True):
        for policy in POLICIES:
            rows = by_policy[policy]
            x = np.array(
                [
                    request_position(str(row["request_id"]), args.background_count)
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

    handles, labels = axes[0].get_legend_handles_labels()
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
