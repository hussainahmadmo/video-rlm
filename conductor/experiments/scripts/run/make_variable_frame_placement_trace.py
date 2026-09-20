#!/usr/bin/env python3
"""Generate a matched variable-frame trace for preparation placement."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arrival-window-s", type=float, default=15.0)
    args = parser.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"video does not exist: {args.video}")
    if args.arrival_window_s <= 0:
        raise SystemExit("arrival window must be positive")

    rng = random.Random(args.seed)
    rows = []
    for tenant in ("a", "b", "c"):
        frame_counts = [1] * 5 + [16] * 10 + [128] * 5
        rng.shuffle(frame_counts)
        for index, frame_count in enumerate(frame_counts):
            rows.append(
                {
                    "request_id": f"{tenant}-{index}",
                    "qid": f"{tenant}-{index}",
                    "tenant": tenant,
                    "class": "background",
                    "modality": "video",
                    "video": str(args.video),
                    "frame_count": frame_count,
                    "max_tokens": 32,
                    "arrival_s": rng.uniform(0.0, args.arrival_window_s),
                    "priority": 0,
                    "prompt_override": "Briefly describe the video.",
                    "scenario": "variable_frame_placement",
                }
            )
    rows.sort(key=lambda row: (float(row["arrival_s"]), str(row["request_id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} requests to {args.output}")


if __name__ == "__main__":
    main()
