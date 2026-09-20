#!/usr/bin/env python3
"""Generate a 60-request trace that backlogs CPU and GPU preparation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def requests(seed: int, video: str) -> list[dict]:
    rng = random.Random(seed)
    mixes = {
        # A arrives first and has enough medium and large work to occupy both
        # CPU workers and GPU preparation lanes under calibrated placement.
        "a": [16] * 10 + [128] * 10,
        # B and C arrive while A is in service and remain preparation-backlogged.
        "b": [1] * 5 + [16] * 10 + [128] * 5,
        "c": [1] * 5 + [16] * 10 + [128] * 5,
    }
    rows: list[dict] = []
    for tenant, sizes in mixes.items():
        rng.shuffle(sizes)
        base = 0.0 if tenant == "a" else 0.5
        for index, frame_count in enumerate(sizes):
            request_id = f"{tenant}-{index:02d}"
            rows.append(
                {
                    "request_id": request_id,
                    "qid": request_id,
                    "tenant": tenant,
                    "modality": "video",
                    "video": video,
                    "frame_count": frame_count,
                    "arrival_s": base + rng.uniform(0.0, 0.2),
                    "max_tokens": 32,
                    "priority": 0,
                    "prompt_override": "Briefly describe the video.",
                    "class": "background",
                }
            )
    rows.sort(key=lambda row: (row["arrival_s"], row["request_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    rows = requests(args.seed, args.video)
    if len(rows) != 60:
        raise RuntimeError("trace must contain exactly 60 requests")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
