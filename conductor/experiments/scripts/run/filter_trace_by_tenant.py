#!/usr/bin/env python3
"""Filter a JSONL arrival trace while preserving requests and arrival times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tenant", action="append", required=True)
    args = parser.parse_args()

    tenants = set(args.tenant)
    selected = []
    with args.input.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("tenant")) in tenants:
                selected.append(row)
    if not selected:
        raise SystemExit(f"no requests found for tenants {sorted(tenants)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "tenants": sorted(tenants),
        "requests": len(selected),
        "first_arrival_s": min(float(row["arrival_s"]) for row in selected),
        "last_arrival_s": max(float(row["arrival_s"]) for row in selected),
    }, indent=2))


if __name__ == "__main__":
    main()
