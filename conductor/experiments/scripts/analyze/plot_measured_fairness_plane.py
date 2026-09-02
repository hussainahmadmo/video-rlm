#!/usr/bin/env python3
"""Plot measured preparation and inference fairness for two-tenant runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


POLICY_ORDER = [
    "fcfs",
    "tenant_round_robin",
    "prep_max_min",
    "engine_tenant_fair",
    "max_min",
]
LABELS = {
    "fcfs": "FCFS",
    "tenant_round_robin": "Tenant RR",
    "prep_max_min": "Prep-only",
    "engine_tenant_fair": "Inference-only",
    "max_min": "Cross-stage (ours)",
}
COLORS = {
    "fcfs": "#D55E00",
    "tenant_round_robin": "#CC79A7",
    "prep_max_min": "#E69F00",
    "engine_tenant_fair": "#56B4E9",
    "max_min": "#009E73",
}
MARKERS = {
    "fcfs": "s",
    "tenant_round_robin": "D",
    "prep_max_min": "o",
    "engine_tenant_fair": "^",
    "max_min": "v",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def overlap(
    start: object, end: object, left: float, right: float,
    service: object = None,
) -> float:
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return 0.0
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        return 0.0
    amount = max(0.0, min(float(end), right) - max(float(start), left))
    duration = float(end) - float(start)
    if (
        isinstance(service, (int, float))
        and math.isfinite(float(service))
        and duration > 0
    ):
        return amount / duration * float(service)
    return amount


def fairness(a: float, b: float) -> float:
    total = a + b
    return math.nan if total <= 0 else 100.0 * (1.0 - abs(a - b) / total)


def analyze_run(summary_path: Path) -> dict | None:
    summary = json.loads(summary_path.read_text())
    results_path = summary_path.parent / "results.jsonl"
    if not results_path.exists() or int(summary.get("errors", 0)):
        return None
    rows = [row for row in read_jsonl(results_path) if not row.get("error")]
    tenants = sorted({str(row.get("tenant")) for row in rows})
    if tenants != ["a", "b"]:
        return None

    first = max(
        min(float(row["arrival_s"]) for row in rows if row["tenant"] == tenant)
        for tenant in tenants
    )
    last = min(
        max(float(row["arrival_s"]) for row in rows if row["tenant"] == tenant)
        for tenant in tenants
    )
    if last <= first:
        return None

    prep = {}
    inference = {}
    for tenant in tenants:
        tenant_rows = [row for row in rows if row["tenant"] == tenant]
        prep[tenant] = sum(
            overlap(row.get("prep_started_s"), row.get("prep_ready_s"), first, last)
            for row in tenant_rows
        )
        inference[tenant] = sum(
            overlap(
                row.get("vlm_submit_s"), row.get("completion_s"), first, last,
                row.get("inference_accounted_service_s"),
            )
            for row in tenant_rows
        )

    return {
        "policy": str(summary.get("prep_policy")),
        "prep_fairness": fairness(prep["a"], prep["b"]),
        "inference_fairness": fairness(inference["a"], inference["b"]),
        "window_start_s": first,
        "window_end_s": last,
        "prep_a_s": prep["a"],
        "prep_b_s": prep["b"],
        "inference_a_s": inference["a"],
        "inference_b_s": inference["b"],
        "summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row
        for path in sorted(args.root.rglob("summary.json"))
        if (row := analyze_run(path)) is not None
    ]
    if not rows:
        raise SystemExit(f"no complete two-tenant runs under {args.root}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axis = plt.subplots(figsize=(8.5, 7.4))
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.18, top=0.79)
    for policy in POLICY_ORDER:
        group = grouped.get(policy, [])
        if not group:
            continue
        xs = np.asarray([row["prep_fairness"] for row in group])
        ys = np.asarray([row["inference_fairness"] for row in group])
        axis.scatter(xs, ys, s=50, color=COLORS[policy], alpha=0.32)
        axis.errorbar(
            float(xs.mean()),
            float(ys.mean()),
            xerr=float(xs.std(ddof=1)) if len(xs) > 1 else 0,
            yerr=float(ys.std(ddof=1)) if len(ys) > 1 else 0,
            fmt=MARKERS[policy],
            markersize=11,
            color=COLORS[policy],
            capsize=4,
            label=f"{LABELS[policy]} (n={len(group)})",
        )

    axis.set_xlim(0, 102)
    axis.set_ylim(0, 102)
    axis.set_xlabel("Preparation-service fairness (%)")
    axis.set_ylabel("Inference-service fairness (%)")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title(
        "Simultaneous heterogeneous contention\nUpper-right is fairer at both stages",
        fontweight="bold",
    )
    fig.suptitle("Measured Cross-Stage Tenant Fairness Plane", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.075,
        r"Two-tenant score: $F_s=100\left(1-\frac{|S_A^s-S_B^s|}{S_A^s+S_B^s}\right)$. "
        "Dots are seeds; markers are means; error bars show one standard deviation.",
        ha="center",
        fontsize=9.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
