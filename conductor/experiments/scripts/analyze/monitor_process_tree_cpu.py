#!/usr/bin/env python3
"""Sample CPU and thread use for dynamically changing process trees.

The monitor follows descendants created after startup (for example ffmpeg
workers), computes CPU use from /proc clock-tick deltas, and exits after a
designated workload PID terminates. Short-lived processes that start and exit
entirely between samples can be missed, so use a sub-second interval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[position]


def read_processes() -> dict[int, tuple[int, int, int]]:
    """Return PID -> (PPID, user+system ticks, thread count)."""
    processes: dict[int, tuple[int, int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # The command name can contain spaces and parentheses. Everything
            # after the final ')' starts with state, then PPID.
            text = (entry / "stat").read_text()
            fields = text[text.rfind(")") + 2 :].split()
            pid = int(entry.name)
            ppid = int(fields[1])
            ticks = int(fields[11]) + int(fields[12])
            threads = int(fields[17])
            processes[pid] = (ppid, ticks, threads)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return processes


def descendants(
    processes: dict[int, tuple[int, int, int]], roots: set[int]
) -> set[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for pid, (ppid, _, _) in processes.items():
        children[ppid].append(pid)
    selected: set[int] = set()
    stack = [pid for pid in roots if pid in processes]
    while stack:
        pid = stack.pop()
        if pid in selected:
            continue
        selected.add(pid)
        stack.extend(children.get(pid, ()))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", required=True, help="Comma-separated root PIDs")
    parser.add_argument("--until-pid", type=int, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=0.2)
    args = parser.parse_args()

    roots = {int(value) for value in args.roots.split(",") if value.strip()}
    if not roots or args.interval_s <= 0:
        parser.error("roots must be nonempty and interval must be positive")

    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    clock_ticks = os.sysconf("SC_CLK_TCK")
    started = time.monotonic()
    previous_time = started
    previous_ticks: dict[int, int] = {}
    cpu_cores: list[float] = []
    process_counts: list[float] = []
    thread_counts: list[float] = []
    cpu_seconds = 0.0
    observed_pids: set[int] = set()

    with args.samples.open("w") as handle:
        while True:
            now = time.monotonic()
            processes = read_processes()
            selected = descendants(processes, roots)
            observed_pids.update(selected)
            elapsed = max(now - previous_time, 1e-9)
            delta_ticks = 0
            for pid in selected:
                ticks = processes[pid][1]
                if pid in previous_ticks:
                    delta_ticks += max(0, ticks - previous_ticks[pid])
            sample_cpu_seconds = delta_ticks / clock_ticks
            cores = sample_cpu_seconds / elapsed
            threads = sum(processes[pid][2] for pid in selected)

            # The first sample establishes a tick baseline.
            if previous_ticks:
                cpu_seconds += sample_cpu_seconds
                cpu_cores.append(cores)
                process_counts.append(float(len(selected)))
                thread_counts.append(float(threads))
                handle.write(
                    json.dumps(
                        {
                            "elapsed_s": now - started,
                            "interval_s": elapsed,
                            "cpu_cores": cores,
                            "processes": len(selected),
                            "threads": threads,
                            "pids": sorted(selected),
                        }
                    )
                    + "\n"
                )
                handle.flush()

            previous_ticks = {pid: processes[pid][1] for pid in selected}
            previous_time = now
            if args.until_pid not in processes:
                break
            time.sleep(args.interval_s)

    wall_time = time.monotonic() - started
    summary = {
        "roots": sorted(roots),
        "until_pid": args.until_pid,
        "interval_s": args.interval_s,
        "samples": len(cpu_cores),
        "wall_time_s": wall_time,
        "cpu_seconds": cpu_seconds,
        "mean_cpu_cores": statistics.mean(cpu_cores) if cpu_cores else None,
        "p50_cpu_cores": percentile(cpu_cores, 0.50),
        "p95_cpu_cores": percentile(cpu_cores, 0.95),
        "max_cpu_cores": max(cpu_cores) if cpu_cores else None,
        "mean_processes": statistics.mean(process_counts) if process_counts else None,
        "max_processes": int(max(process_counts)) if process_counts else None,
        "mean_threads": statistics.mean(thread_counts) if thread_counts else None,
        "max_threads": int(max(thread_counts)) if thread_counts else None,
        "observed_pid_count": len(observed_pids),
        "measurement_note": (
            "Dynamic /proc sampling; processes shorter than the sampling interval "
            "may be undercounted."
        ),
    }
    # Ensure accidental NaN/Inf values never enter result JSON.
    for key, value in list(summary.items()):
        if isinstance(value, float) and not math.isfinite(value):
            summary[key] = None
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
