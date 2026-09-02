#!/usr/bin/env python3
"""Generate a matched multi-tenant video workload with cost heterogeneity."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_tenant_counts(
    values: list[str], tenants: list[str], default: int,
) -> dict[str, int]:
    """Parse repeated TENANT=COUNT overrides for a workload class."""
    counts = {tenant: default for tenant in tenants}
    for value in values:
        tenant, separator, count_text = value.partition("=")
        if not separator or tenant not in counts:
            raise ValueError(
                f"invalid tenant count {value!r}; expected one of "
                f"{', '.join(tenants)} as TENANT=COUNT"
            )
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid tenant count {value!r}; count must be an integer"
            ) from exc
        if count < 0:
            raise ValueError(
                f"invalid tenant count {value!r}; count cannot be negative"
            )
        counts[tenant] = count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tenants", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--background-per-tenant", type=int, default=12)
    parser.add_argument("--urgent-per-tenant", type=int, default=4)
    parser.add_argument(
        "--tenant-background-count", action="append", default=[],
        metavar="TENANT=COUNT",
        help="Override the background count for a tenant; may be repeated.",
    )
    parser.add_argument(
        "--tenant-urgent-count", action="append", default=[],
        metavar="TENANT=COUNT",
        help="Override the urgent count for a tenant; may be repeated.",
    )
    parser.add_argument("--urgent-arrival-s", type=float, default=5.0)
    parser.add_argument(
        "--uniform-class", action="store_true",
        help=("Emit one neutral request class with priority zero while "
              "preserving both arrival waves. This isolates tenant fairness "
              "from application priority."),
    )
    parser.add_argument(
        "--background-arrival-interval-s", type=float, default=0.0,
        help="Spacing between consecutive background arrivals per tenant.",
    )
    parser.add_argument(
        "--urgent-arrival-interval-s", type=float, default=0.0,
        help="Spacing between consecutive urgent arrivals per tenant.",
    )
    parser.add_argument(
        "--arrival-jitter-s", type=float, default=0.0,
        help="Deterministic per-seed uniform arrival jitter in either direction.",
    )
    parser.add_argument("--frame-budgets", nargs="+", type=int,
                        default=[8, 32, 128])
    parser.add_argument("--exclude-qid", action="append", default=[])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if len(set(args.tenants)) != len(args.tenants) or not args.tenants:
        parser.error("tenant names must be non-empty and unique")
    if min(args.background_per_tenant, args.urgent_per_tenant) < 0:
        parser.error("per-tenant request counts cannot be negative")
    if min(
        args.background_arrival_interval_s,
        args.urgent_arrival_interval_s,
        args.arrival_jitter_s,
    ) < 0:
        parser.error("arrival intervals and jitter cannot be negative")
    if not args.frame_budgets or min(args.frame_budgets) < 1:
        parser.error("frame budgets must be positive")

    try:
        background_counts = parse_tenant_counts(
            args.tenant_background_count,
            args.tenants,
            args.background_per_tenant,
        )
        urgent_counts = parse_tenant_counts(
            args.tenant_urgent_count,
            args.tenants,
            args.urgent_per_tenant,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not any(background_counts.values()) and not any(urgent_counts.values()):
        parser.error("the trace must contain at least one request")

    excluded = set(args.exclude_qid)
    rows = [
        row for row in load_jsonl(args.dataset)
        if str(row.get("qid") or row.get("question_id") or row.get("id"))
        not in excluded
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    needed = sum(background_counts.values()) + sum(urgent_counts.values())
    if len(rows) < needed:
        raise SystemExit(f"need {needed} dataset rows, found {len(rows)}")

    trace = []
    cursor = 0
    for workload, counts, arrival_s, interval_s, priority in (
        (
            "background", background_counts, 0.0,
            args.background_arrival_interval_s, 10,
        ),
        (
            "urgent", urgent_counts, args.urgent_arrival_s,
            args.urgent_arrival_interval_s, 0,
        ),
    ):
        # Every tenant receives the same multiset of frame budgets. This keeps
        # tenant demand matched while decoupling application class from cost.
        for tenant_index, tenant in enumerate(args.tenants):
            count = counts[tenant]
            budgets = [
                args.frame_budgets[index % len(args.frame_budgets)]
                for index in range(count)
            ]
            rng.shuffle(budgets)
            for local_index, frame_count in enumerate(budgets):
                row = dict(rows[cursor])
                cursor += 1
                row.update({
                    "request_id": (
                        f"{tenant}-{workload}-{local_index}"
                    ),
                    "tenant": tenant,
                    "class": "default" if args.uniform_class else workload,
                    "arrival_s": max(
                        0.0,
                        arrival_s
                        + local_index * interval_s
                        + rng.uniform(
                            -args.arrival_jitter_s,
                            args.arrival_jitter_s,
                        ),
                    ),
                    "priority": 0 if args.uniform_class else priority,
                    "frame_count": frame_count,
                })
                trace.append(row)

    trace.sort(key=lambda row: (
        float(row["arrival_s"]), str(row["tenant"]), str(row["request_id"]),
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in trace:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps({
        "output": str(args.output),
        "requests": len(trace),
        "tenants": args.tenants,
        "background_counts": background_counts,
        "urgent_counts": urgent_counts,
        "background_requests": sum(background_counts.values()),
        "urgent_requests": sum(urgent_counts.values()),
        "uniform_class": args.uniform_class,
        "background_arrival_interval_s": args.background_arrival_interval_s,
        "urgent_arrival_interval_s": args.urgent_arrival_interval_s,
        "frame_budgets": args.frame_budgets,
    }, indent=2))


if __name__ == "__main__":
    main()
