#!/usr/bin/env python3
"""Plot frame-budget distributions from trace JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    counts: Counter[tuple[str, int]] = Counter()
    trace_count = 0
    row_count = 0

    for trace in sorted(args.trace_root.glob("*.jsonl")):
        trace_count += 1
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            workload = str(
                row.get("workload")
                or row.get("class")
                or row.get("request_type")
                or "unknown"
            )
            frames = row.get("frame_count") or row.get("video_frames") or row.get("frames")
            if frames is not None:
                counts[(workload, int(frames))] += 1

    if not counts:
        raise SystemExit(f"No frame counts found under {args.trace_root}")

    workloads = [name for name in ("background", "urgent") if any(k[0] == name for k in counts)]
    frames = sorted({frame_count for _, frame_count in counts})
    colors = {"background": "#D55E00", "urgent": "#0072B2"}
    x = np.arange(len(frames))
    width = 0.36

    fig, (ax_count, ax_share) = plt.subplots(1, 2, figsize=(14, 5.8))

    for index, workload in enumerate(workloads):
        offset = (index - (len(workloads) - 1) / 2) * width
        values = [counts[(workload, frame_count)] for frame_count in frames]
        total = sum(values)
        shares = [100 * value / total for value in values]

        bars = ax_count.bar(
            x + offset, values, width, label=workload.title(), color=colors[workload]
        )
        ax_count.bar_label(bars, fmt="%d", padding=3, fontsize=8)

        bars = ax_share.bar(
            x + offset, shares, width, label=workload.title(), color=colors[workload]
        )
        ax_share.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

    for ax in (ax_count, ax_share):
        ax.set_xticks(x, [str(value) for value in frames])
        ax.set_xlabel("Frames sampled per video request")
        ax.grid(axis="y", alpha=0.25)

    ax_count.set_ylabel("Request occurrences")
    ax_count.set_title("Raw occurrences")
    ax_count.legend()
    ax_share.set_ylabel("Share within workload (%)")
    ax_share.set_title("Normalized within each workload")
    ax_share.legend()

    fig.suptitle(
        f"Video frame budgets in active arrival traces\n"
        f"{trace_count} traces, {row_count:,} request occurrences"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
