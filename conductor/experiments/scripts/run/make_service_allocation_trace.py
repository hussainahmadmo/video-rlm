#!/usr/bin/env python3
"""Generate a matched, continuously backlogged cross-stage service trace."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PREPARATION_TENANTS = {
    # Preparation-heavy, short generation.
    "a": (128, 32, "Briefly summarize the supplied video frames."),
    # Lighter preparation, long generation.
    "b": (
        16,
        256,
        "Describe the supplied video frames in chronological order with extensive detail.",
    ),
    # Light at both stages; enough queued requests keep it eligible initially.
    "c": (1, 32, "Briefly summarize the supplied video frames."),
}

INFERENCE_TENANTS = {
    # Homogeneous, cheap preparation isolates heterogeneous inference demand.
    "a": (
        1,
        256,
        "Describe the supplied video frames in chronological order with extensive detail.",
    ),
    "b": (1, 64, "Summarize the supplied video frames in several sentences."),
    "c": (1, 32, "Briefly summarize the supplied video frames."),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--requests-per-tenant", type=int, default=18)
    parser.add_argument("--arrival-window-s", type=float, default=1.0)
    parser.add_argument(
        "--workload", choices=("preparation", "inference"), default="preparation"
    )
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"video does not exist: {args.video}")
    if args.requests_per_tenant < 2:
        raise SystemExit("requests-per-tenant must be at least two")

    rng = random.Random(args.seed)
    rows = []
    tenants = PREPARATION_TENANTS if args.workload == "preparation" else INFERENCE_TENANTS
    for tenant, (frames, max_tokens, prompt) in tenants.items():
        for index in range(args.requests_per_tenant):
            # Independent jitter changes FCFS tie order across matched seeds,
            # while the short window ensures that all tenants build backlog.
            arrival = rng.uniform(0.0, args.arrival_window_s)
            rows.append(
                {
                    "request_id": f"{tenant}-{index}",
                    "qid": f"{tenant}-{index}",
                    "tenant": tenant,
                    "class": "background",
                    "modality": "video",
                    "video": str(args.video),
                    "frame_count": frames,
                    "max_tokens": max_tokens,
                    "arrival_s": arrival,
                    "priority": 0,
                    "prompt_override": prompt,
                    "scenario": f"{args.workload}_service_allocation",
                }
            )
    rows.sort(key=lambda row: (float(row["arrival_s"]), str(row["request_id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} requests to {args.output}")


if __name__ == "__main__":
    main()
