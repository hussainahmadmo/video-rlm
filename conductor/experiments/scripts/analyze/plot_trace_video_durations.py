#!/usr/bin/env python3
"""Plot source-video durations represented in arrival traces."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    occurrences: list[tuple[str, int, str]] = []
    paths: set[str] = set()
    trace_count = 0

    for trace in sorted(args.trace_root.glob("*.jsonl")):
        trace_count += 1
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = row.get("video") or row.get("video_path") or row.get("path")
            frames = row.get("frame_count") or row.get("video_frames") or row.get("frames")
            workload = row.get("workload") or row.get("class") or row.get("request_type")
            if path and frames is not None and workload in {"background", "urgent"}:
                path = str(path)
                paths.add(path)
                occurrences.append((str(workload), int(frames), path))

    durations: dict[str, float] = {}
    failures = []
    for path in sorted(paths):
        try:
            durations[path] = probe_duration(path)
        except Exception as error:
            failures.append((path, str(error)))

    by_workload: dict[str, list[float]] = defaultdict(list)
    by_workload_frames: dict[tuple[str, int], list[float]] = defaultdict(list)
    for workload, frames, path in occurrences:
        if path not in durations:
            continue
        duration = durations[path]
        by_workload[workload].append(duration)
        by_workload_frames[(workload, frames)].append(duration)

    if not by_workload:
        raise SystemExit("No usable video durations found")

    colors = {"background": "#D55E00", "urgent": "#0072B2"}
    fig, (ax_cdf, ax_box) = plt.subplots(1, 2, figsize=(14, 5.8))

    for workload in ("background", "urgent"):
        values = np.sort(by_workload[workload])
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax_cdf.plot(
            values, cdf, linewidth=2.5, color=colors[workload],
            label=f"{workload.title()} (n={len(values):,})",
        )

    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("Source-video duration (seconds, log scale)")
    ax_cdf.set_ylabel("Cumulative share of requests")
    ax_cdf.set_title("Request-weighted duration distribution")
    ax_cdf.grid(alpha=0.25)
    ax_cdf.legend()

    frame_counts = sorted({frames for _, frames in by_workload_frames})
    positions = np.arange(len(frame_counts))
    width = 0.28
    for index, workload in enumerate(("background", "urgent")):
        data = [by_workload_frames[(workload, frames)] for frames in frame_counts]
        offset = (-0.5 if index == 0 else 0.5) * width
        boxes = ax_box.boxplot(
            data,
            positions=positions + offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(colors[workload])
            box.set_alpha(0.75)
        ax_box.plot([], [], color=colors[workload], linewidth=8,
                    label=workload.title())

    ax_box.set_yscale("log")
    ax_box.set_xticks(positions, [str(value) for value in frame_counts])
    ax_box.set_xlabel("Frames sampled per request")
    ax_box.set_ylabel("Source-video duration (seconds, log scale)")
    ax_box.set_title("Source duration versus frame budget")
    ax_box.grid(axis="y", alpha=0.25)
    ax_box.legend()

    fig.suptitle(
        f"Source-video lengths in active arrival traces\n"
        f"{trace_count} traces, {len(occurrences):,} occurrences, "
        f"{len(durations)} unique videos"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"output={args.output}")
    print(f"traces={trace_count}")
    print(f"occurrences={len(occurrences)}")
    print(f"unique_videos={len(durations)}")
    print(f"probe_failures={len(failures)}")


if __name__ == "__main__":
    main()
