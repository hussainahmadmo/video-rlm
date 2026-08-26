#!/usr/bin/env python3
"""Summarize matched shared-versus-static-isolation experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


POLICIES = (
    "shared_engine_only",
    "shared_priority",
    "static_isolation",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def average(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    if not values:
        raise ValueError(f"no values for {'.'.join(path)}")
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()

    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for summary_path in sorted(args.run.glob("*/*/summary.json")):
        policy = summary_path.parent.name
        trace = summary_path.parent.parent.name
        if policy in POLICIES:
            traces.setdefault(trace, {})[policy] = load(summary_path)

    matched = {
        trace: policies
        for trace, policies in traces.items()
        if all(policy in policies for policy in POLICIES)
    }
    if not matched:
        raise SystemExit("no completed three-policy matched traces")

    print(f"run: {args.run}")
    print(f"matched traces: {len(matched)}")
    print()
    print(
        "policy                 urgent_mean  urgent_p95  background_mean  "
        "throughput"
    )

    aggregate: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        rows = [policies[policy] for policies in matched.values()]
        aggregate[policy] = {
            "urgent_mean": average(rows, ("urgent", "mean_end_to_end_ttft_s")),
            "urgent_p95": average(rows, ("urgent", "p95_end_to_end_ttft_s")),
            "background_mean": average(
                rows, ("background", "mean_end_to_end_ttft_s")
            ),
            "throughput": average(rows, ("throughput_qps",)),
        }
        item = aggregate[policy]
        print(
            f"{policy:23s} "
            f"{item['urgent_mean']:11.2f} "
            f"{item['urgent_p95']:11.2f} "
            f"{item['background_mean']:16.2f} "
            f"{item['throughput']:10.4f}"
        )

    baseline = aggregate["shared_engine_only"]
    print()
    for policy in ("shared_priority", "static_isolation"):
        item = aggregate[policy]
        speedup = baseline["urgent_mean"] / item["urgent_mean"]
        urgent_reduction = 100.0 * (
            1.0 - item["urgent_mean"] / baseline["urgent_mean"]
        )
        background_change = 100.0 * (
            item["background_mean"] / baseline["background_mean"] - 1.0
        )
        throughput_change = 100.0 * (
            item["throughput"] / baseline["throughput"] - 1.0
        )
        print(
            f"{policy}: urgent={speedup:.2f}x faster "
            f"({urgent_reduction:.1f}% lower), "
            f"background={background_change:+.1f}%, "
            f"throughput={throughput_change:+.1f}%"
        )

    isolation_errors = []
    for trace in matched:
        result_path = args.run / trace / "static_isolation" / "results.jsonl"
        summary = matched[trace]["static_isolation"]
        background_ports = set(summary.get("background_ports") or [])
        urgent_ports = set(summary.get("urgent_ports") or [])
        for line in result_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            port = row.get("replica_port")
            if port is None:
                continue
            expected = urgent_ports if row["workload"] == "urgent" else background_ports
            if port not in expected:
                isolation_errors.append(
                    f"{trace}:{row['request_id']} workload={row['workload']} "
                    f"port={port} expected={sorted(expected)}"
                )

    print()
    if isolation_errors:
        print(f"isolation routing violations: {len(isolation_errors)}")
        for message in isolation_errors[:10]:
            print(f"  {message}")
        raise SystemExit(1)
    print("isolation routing violations: 0")


if __name__ == "__main__":
    main()
