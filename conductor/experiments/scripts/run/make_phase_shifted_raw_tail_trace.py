#!/usr/bin/env python3
"""Generate rotating phase-shifted bursts for a raw E2E tail tradeoff."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


TENANTS = ("a", "b", "c")
REQUEST_TYPES = {
    "prep_heavy": (
        128,
        32,
        "Briefly summarize the supplied video frames.",
    ),
    "inference_heavy": (
        32,
        256,
        "Describe the supplied video frames in chronological order with extensive detail.",
    ),
    "light": (
        8,
        32,
        "Briefly summarize the supplied video frames.",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-spacing-s", type=float, default=450.0)
    parser.add_argument("--leader-window-s", type=float, default=5.0)
    parser.add_argument("--followers-window-s", type=float, default=5.0)
    parser.add_argument("--prep-heavy-requests", type=int, default=60)
    parser.add_argument("--inference-heavy-requests", type=int, default=20)
    parser.add_argument("--light-requests", type=int, default=20)
    args = parser.parse_args()

    if args.episodes % len(TENANTS):
        parser.error("episodes must be divisible by three so tenant roles balance")
    if args.episode_spacing_s <= args.leader_window_s + args.followers_window_s:
        parser.error("episode spacing must exceed the two arrival windows")

    sources = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if not sources:
        raise SystemExit(f"input contains no records: {args.input}")

    rng = random.Random(args.seed)
    rows: list[dict] = []
    sequence = 0

    for episode in range(args.episodes):
        start = episode * args.episode_spacing_s
        roles = {
            TENANTS[episode % 3]: ("prep_heavy", args.prep_heavy_requests),
            TENANTS[(episode + 1) % 3]: (
                "inference_heavy",
                args.inference_heavy_requests,
            ),
            TENANTS[(episode + 2) % 3]: ("light", args.light_requests),
        }

        for tenant, (request_type, count) in roles.items():
            frames, max_tokens, prompt = REQUEST_TYPES[request_type]
            if request_type == "prep_heavy":
                low = start
                high = start + args.leader_window_s
            else:
                low = start + args.leader_window_s
                high = low + args.followers_window_s

            for arrival in sorted(rng.uniform(low, high) for _ in range(count)):
                source = sources[(sequence + args.seed) % len(sources)]
                row = dict(source)
                row.update(
                    {
                        "request_id": f"phase-seed{args.seed}-{tenant}-{sequence}",
                        "tenant": tenant,
                        "class": "background",
                        "arrival_s": arrival,
                        "priority": 10,
                        "frame_count": frames,
                        "max_tokens": max_tokens,
                        "prompt_override": prompt,
                        "request_type": request_type,
                        "scenario": "phase_shifted_raw_tail_tradeoff",
                        "arrival_pattern": "phase_shifted_burst",
                        "trace_seed": args.seed,
                        "episode": episode + 1,
                    }
                )
                rows.append(row)
                sequence += 1

    rows.sort(key=lambda row: (row["arrival_s"], row["tenant"], row["request_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    tenant_counts = Counter(row["tenant"] for row in rows)
    type_counts = Counter(row["request_type"] for row in rows)
    horizon = (args.episodes - 1) * args.episode_spacing_s
    horizon += args.leader_window_s + args.followers_window_s
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seed": args.seed,
                "requests": len(rows),
                "tenants": dict(sorted(tenant_counts.items())),
                "request_types": dict(sorted(type_counts.items())),
                "episodes": args.episodes,
                "requests_per_episode": len(rows) // args.episodes,
                "episode_spacing_s": args.episode_spacing_s,
                "leader_window_s": args.leader_window_s,
                "followers_window_s": args.followers_window_s,
                "nominal_whole_trace_rate_qps": len(rows) / horizon,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
