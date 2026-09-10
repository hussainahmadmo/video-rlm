#!/usr/bin/env python3
"""Generate equal-demand tenant traces with transient heterogeneous bursts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TYPES = (
    ("prep_heavy", 128, 32, "Briefly summarize the supplied video frames."),
    ("inference_heavy", 32, 256,
     "Describe the supplied video frames in chronological order with extensive detail."),
    ("light", 8, 32, "Briefly summarize the supplied video frames."),
)
TENANTS = ("a", "b", "c")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bursts", type=int, default=3)
    parser.add_argument("--burst-width-s", type=float, default=30.0)
    parser.add_argument("--burst-spacing-s", type=float, default=70.0)
    parser.add_argument("--requests-per-tenant", type=int, default=33)
    args = parser.parse_args()
    if args.requests_per_tenant % args.bursts:
        parser.error("requests per tenant must be divisible by bursts")
    if args.requests_per_tenant % len(TYPES):
        parser.error("requests per tenant must be divisible by request types")

    source = next(
        (json.loads(line) for line in args.input.read_text().splitlines() if line.strip()),
        None,
    )
    if source is None:
        raise SystemExit(f"input contains no records: {args.input}")

    rng = random.Random(args.seed)
    per_burst = args.requests_per_tenant // args.bursts
    rows = []
    for tenant_index, tenant in enumerate(TENANTS):
        request_index = 0
        for burst_index in range(args.bursts):
            start = burst_index * args.burst_spacing_s
            # Conditional on the exact count, ordered uniform timestamps are
            # the arrival times of a homogeneous Poisson process in a window.
            arrivals = sorted(
                start + rng.random() * args.burst_width_s
                for _ in range(per_burst)
            )
            for arrival in arrivals:
                kind, frames, tokens, prompt = TYPES[
                    (request_index + tenant_index) % len(TYPES)
                ]
                row = dict(source)
                row.update({
                    "request_id": f"{tenant}-{request_index}",
                    "tenant": tenant,
                    "class": "background",
                    "arrival_s": arrival,
                    "priority": 10,
                    "frame_count": frames,
                    "max_tokens": tokens,
                    "prompt_override": prompt,
                    "request_type": kind,
                    "scenario": "balanced_transient_crossed_bursts",
                    "arrival_pattern": "conditioned_poisson_bursts",
                    "trace_seed": args.seed,
                })
                rows.append(row)
                request_index += 1

    rows.sort(key=lambda row: (row["arrival_s"], row["tenant"], row["request_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({
        "output": str(args.output), "seed": args.seed, "requests": len(rows),
        "tenants": {tenant: sum(r["tenant"] == tenant for r in rows) for tenant in TENANTS},
        "types_per_tenant": args.requests_per_tenant // len(TYPES),
        "burst_windows": [
            [i * args.burst_spacing_s, i * args.burst_spacing_s + args.burst_width_s]
            for i in range(args.bursts)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
