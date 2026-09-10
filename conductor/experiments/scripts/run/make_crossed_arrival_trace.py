#!/usr/bin/env python3
"""Generate matched fixed, Poisson, or bursty crossed CPU/GPU traces."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def fixed(rate: float, start: float, end: float) -> list[float]:
    if rate <= 0:
        return []
    values, now = [], start
    while now < end:
        values.append(now)
        now += 1.0 / rate
    return values


def poisson(rng: random.Random, rate: float, start: float, end: float) -> list[float]:
    values, now = [], start
    while rate > 0:
        now += rng.expovariate(rate)
        if now >= end:
            break
        values.append(now)
    return values


def bursty(
    rng: random.Random,
    rate: float,
    start: float,
    end: float,
    period: float,
    duty: float,
) -> list[float]:
    """Poisson arrivals in ON windows, preserving ``rate`` over full time."""
    values = []
    window = start
    while window < end:
        on_end = min(end, window + period * duty)
        values.extend(poisson(rng, rate / duty, window, on_end))
        window += period
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", choices=("fixed", "poisson", "bursty"), required=True)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--load", type=float, default=1.0,
                        help="Multiplier applied to aggressor A/B rates")
    parser.add_argument("--a-rate", type=float, default=0.25)
    parser.add_argument("--b-rate", type=float, default=0.80)
    parser.add_argument("--c-rate", type=float, default=0.10)
    parser.add_argument("--c-start-s", type=float, default=20.0)
    parser.add_argument("--burst-period-s", type=float, default=30.0)
    parser.add_argument("--burst-duty-cycle", type=float, default=0.25)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.load <= 0:
        parser.error("--duration-s and --load must be positive")
    if not 0 < args.burst_duty_cycle <= 1:
        parser.error("--burst-duty-cycle must be in (0, 1]")

    source_rows = [
        json.loads(line) for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if not source_rows:
        raise SystemExit(f"input contains no records: {args.input}")
    templates = {
        tenant: next(
            (row for row in source_rows if str(row.get("tenant")) == tenant),
            source_rows[0],
        )
        for tenant in ("a", "b", "c")
    }
    specs = {
        "a": {"rate": args.a_rate * args.load, "start": 0.0,
              "frames": 128, "tokens": 32,
              "prompt": "Briefly summarize the supplied video frames."},
        "b": {"rate": args.b_rate * args.load, "start": 0.0,
              "frames": 32, "tokens": 256,
              "prompt": "Describe the supplied video frames in chronological order with extensive detail."},
        "c": {"rate": args.c_rate, "start": min(args.c_start_s, args.duration_s / 3),
              "frames": 8, "tokens": 32,
              "prompt": "Briefly summarize the supplied video frames."},
    }

    rows = []
    for tenant_index, (tenant, spec) in enumerate(specs.items()):
        rng = random.Random(args.seed * 1009 + tenant_index)
        if args.pattern == "fixed":
            times = fixed(spec["rate"], spec["start"], args.duration_s)
        elif args.pattern == "poisson":
            times = poisson(rng, spec["rate"], spec["start"], args.duration_s)
        else:
            times = bursty(
                rng, spec["rate"], spec["start"], args.duration_s,
                args.burst_period_s, args.burst_duty_cycle,
            )
        for index, arrival in enumerate(times):
            row = dict(templates[tenant])
            row.update({
                "request_id": f"{tenant}-{index}", "tenant": tenant,
                "class": "background", "arrival_s": arrival, "priority": 10,
                "frame_count": spec["frames"], "max_tokens": spec["tokens"],
                "prompt_override": spec["prompt"],
                "scenario": f"crossed_{args.pattern}_load_{args.load:g}",
                "arrival_pattern": args.pattern, "trace_seed": args.seed,
                "load_multiplier": args.load,
            })
            rows.append(row)
    rows.sort(key=lambda row: (row["arrival_s"], row["tenant"], row["request_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    counts = {tenant: sum(row["tenant"] == tenant for row in rows) for tenant in specs}
    print(json.dumps({"output": str(args.output), "pattern": args.pattern,
                      "load": args.load, "seed": args.seed,
                      "requests": len(rows), "tenants": counts}, indent=2))


if __name__ == "__main__":
    main()
