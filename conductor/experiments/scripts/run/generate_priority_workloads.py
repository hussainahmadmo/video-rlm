#!/usr/bin/env python3
"""Generate reproducible mixed-priority video-QA arrival traces."""

import argparse
import json
import random
from pathlib import Path


def load(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def duration(row):
    for key in ("duration_s", "video_duration_s", "duration"):
        if row.get(key) is not None:
            return float(row[key])
    return None


def frame_budget(row, rng):
    value = duration(row)
    if value is None:
        return rng.choice([8, 16, 32, 64, 128])
    if value <= 60:
        return rng.choice([8, 16, 32])
    if value <= 180:
        return rng.choice([16, 32, 64])
    return rng.choice([32, 64, 128])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pattern", choices=["burst", "staggered", "poisson", "replay"], required=True)
    p.add_argument("--replay-template", type=Path)
    p.add_argument("--background-count", type=int, default=64)
    p.add_argument("--urgent-count", type=int, default=16)
    p.add_argument("--rate-qps", type=float, default=0.5)
    p.add_argument("--interval-s", type=float, default=2.0)
    p.add_argument("--urgent-arrival-s", type=float, default=30.0)
    p.add_argument(
        "--urgent-pattern",
        choices=["burst", "staggered", "poisson"],
        default="burst",
        help="Arrival process for urgent requests after --urgent-arrival-s.",
    )
    p.add_argument("--urgent-interval-s", type=float, default=1.0)
    p.add_argument("--urgent-rate-qps", type=float, default=1.0)
    p.add_argument("--background-frames", type=int, default=128)
    p.add_argument("--urgent-frames", type=int, default=8)
    p.add_argument("--mixed-frames", action="store_true")
    p.add_argument(
        "--exclude-qid",
        action="append",
        default=[],
        help="Question ID to remove before sampling; may be repeated.",
    )
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    excluded = set(args.exclude_qid)
    source = [
        row for row in load(args.dataset)
        if str(row.get("qid") or row.get("question_id") or row.get("id"))
        not in excluded
    ]
    rng.shuffle(source)

    if args.pattern == "replay":
        if args.replay_template is None:
            p.error("--replay-template is required for replay")
        template = load(args.replay_template)
        if len(source) < len(template):
            raise SystemExit("dataset has fewer rows than replay template")
        trace = []
        for index, spec in enumerate(template):
            row = dict(source[index])
            workload = str(spec.get("class", spec.get("workload", "background")))
            row.update({
                "request_id": str(spec.get("request_id", f"{workload}-{index}")),
                "class": workload,
                "arrival_s": float(spec["arrival_s"]),
                "priority": int(spec.get("priority", 0 if workload == "urgent" else 10)),
                "frame_count": int(spec.get("frame_count", frame_budget(row, rng))),
            })
            trace.append(row)
    else:
        needed = args.background_count + args.urgent_count
        if len(source) < needed:
            raise SystemExit(f"need {needed} dataset rows, found {len(source)}")
        if args.pattern == "burst":
            arrivals = [0.0] * args.background_count
        elif args.pattern == "staggered":
            arrivals = [index * args.interval_s for index in range(args.background_count)]
        else:
            if args.rate_qps <= 0:
                p.error("--rate-qps must be positive")
            arrivals, now = [], 0.0
            for _ in range(args.background_count):
                now += rng.expovariate(args.rate_qps)
                arrivals.append(now)

        trace = []
        for index, (row, arrival) in enumerate(zip(source, arrivals)):
            row = dict(row)
            row.update({
                "request_id": f"background-{index}",
                "class": "background",
                "arrival_s": arrival,
                "priority": 10,
                "frame_count": frame_budget(row, rng) if args.mixed_frames else args.background_frames,
            })
            trace.append(row)
        if args.urgent_pattern == "burst":
            urgent_arrivals = [args.urgent_arrival_s] * args.urgent_count
        elif args.urgent_pattern == "staggered":
            if args.urgent_interval_s <= 0:
                p.error("--urgent-interval-s must be positive")
            urgent_arrivals = [
                args.urgent_arrival_s + index * args.urgent_interval_s
                for index in range(args.urgent_count)
            ]
        else:
            if args.urgent_rate_qps <= 0:
                p.error("--urgent-rate-qps must be positive")
            urgent_arrivals, now = [], args.urgent_arrival_s
            for _ in range(args.urgent_count):
                now += rng.expovariate(args.urgent_rate_qps)
                urgent_arrivals.append(now)

        offset = args.background_count
        urgent_rows = source[offset:offset + args.urgent_count]
        for index, (row, arrival) in enumerate(zip(urgent_rows, urgent_arrivals)):
            row = dict(row)
            row.update({
                "request_id": f"urgent-{index}",
                "class": "urgent",
                "arrival_s": arrival,
                "priority": 0,
                "frame_count": frame_budget(row, rng) if args.mixed_frames else args.urgent_frames,
            })
            trace.append(row)

    trace.sort(key=lambda row: (float(row["arrival_s"]), str(row["request_id"])))
    write(args.output, trace)
    counts = {}
    for row in trace:
        counts[str(row["frame_count"])] = counts.get(str(row["frame_count"]), 0) + 1
    print(json.dumps({"output": str(args.output), "requests": len(trace), "frame_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
