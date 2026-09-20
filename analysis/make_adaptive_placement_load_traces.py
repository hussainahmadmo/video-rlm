#!/usr/bin/env python3
"""Create matched low-load and burst traces for placement adaptation."""
import argparse
import json
from pathlib import Path


def write_trace(path: Path, interval_s: float) -> None:
    tenants = ("a", "b", "c")
    rows = []
    for index in range(60):
        tenant = tenants[index % len(tenants)]
        rows.append({
            "request_id": f"{path.stem}-{index}",
            "qid": f"{path.stem}-{index}",
            "tenant": tenant,
            "modality": "video",
            "video": "replaced-by-runner",
            "frame_count": 16,
            "arrival_s": index * interval_s,
            "max_tokens": 32,
            "priority": 0,
            "prompt_override": "Briefly describe the video.",
            "class": "background",
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_trace(args.output_dir / "low_load.jsonl", interval_s=2.0)
    write_trace(args.output_dir / "burst.jsonl", interval_s=0.05)


if __name__ == "__main__":
    main()
