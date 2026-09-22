#!/usr/bin/env python3
"""Generate matched traces for the SJF arrival-distribution experiment.

For a workload and seed, every distribution receives the same ordered requests.
Only the interarrival gaps differ. Each finite trace is scaled to the same span,
so request count and realized offered rate are identical across distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


PATTERNS = ("poisson", "lognormal", "hyperexponential")
TENANTS = ("a", "b", "c")


def draw_gap(rng: random.Random, pattern: str, rate: float, scv: float) -> float:
    mean = 1.0 / rate
    if pattern == "poisson":
        return rng.expovariate(rate)
    if pattern == "lognormal":
        sigma2 = math.log1p(scv)
        mu = math.log(mean) - sigma2 / 2.0
        return rng.lognormvariate(mu, math.sqrt(sigma2))
    if pattern == "hyperexponential":
        p = 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))
        phase_rate = 2.0 * p / mean if rng.random() < p else 2.0 * (1.0 - p) / mean
        return rng.expovariate(phase_rate)
    raise ValueError(pattern)


def request_order(seed: int, requests_per_tenant: int) -> list[tuple[str, int]]:
    rng = random.Random(seed * 65537 + 17)
    rows: list[tuple[str, int]] = []
    for tenant_sequence in range(requests_per_tenant):
        block = list(TENANTS)
        rng.shuffle(block)
        rows.extend((tenant, tenant_sequence) for tenant in block)
    return rows


def frame_assignments(
    seed: int, workload: str, requests_per_tenant: int
) -> dict[str, list[int]]:
    if workload == "heavy":
        fixed = {"a": 128, "b": 16, "c": 1}
        return {tenant: [fixed[tenant]] * requests_per_tenant for tenant in TENANTS}
    if requests_per_tenant % 3:
        raise ValueError("mixed workload requires requests-per-tenant divisible by three")
    result = {}
    for tenant_index, tenant in enumerate(TENANTS):
        values = [1, 16, 128] * (requests_per_tenant // 3)
        random.Random(seed * 9173 + tenant_index).shuffle(values)
        result[tenant] = values
    return result


def arrival_times(
    pattern: str, seed: int, rate: float, scv: float, count: int
) -> tuple[list[float], list[float]]:
    rng = random.Random(seed * 7919 + PATTERNS.index(pattern) * 104729)
    gaps = [draw_gap(rng, pattern, rate, scv) for _ in range(count - 1)]
    target_span = (count - 1) / rate
    scale = target_span / sum(gaps)
    gaps = [gap * scale for gap in gaps]
    times = [0.0]
    for gap in gaps:
        times.append(times[-1] + gap)
    return times, gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--workload", choices=("heavy", "mixed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rate-qps", type=float, default=1.0)
    parser.add_argument("--scv", type=float, default=4.0)
    parser.add_argument("--requests-per-tenant", type=int, default=42)
    args = parser.parse_args()
    if args.rate_qps <= 0:
        parser.error("rate must be positive")
    if args.scv <= 1:
        parser.error("SCV must exceed one")
    if args.requests_per_tenant <= 0:
        parser.error("requests-per-tenant must be positive")
    if args.workload == "mixed" and args.requests_per_tenant % 3:
        parser.error("mixed workload requires requests-per-tenant divisible by three")

    args.output.mkdir(parents=True, exist_ok=True)
    order = request_order(args.seed, args.requests_per_tenant)
    frames = frame_assignments(args.seed, args.workload, args.requests_per_tenant)
    manifest: dict[str, object] = {
        "design": "matched_sjf_arrival_distributions",
        "seed": args.seed,
        "workload": args.workload,
        "requests": len(order),
        "requests_per_tenant": args.requests_per_tenant,
        "target_and_realized_span_rate_qps": args.rate_qps,
        "target_scv": {"poisson": 1.0, "lognormal": args.scv,
                       "hyperexponential": args.scv},
        "patterns": {},
    }
    for pattern in PATTERNS:
        times, gaps = arrival_times(pattern, args.seed, args.rate_qps, args.scv, len(order))
        rows = []
        for sequence, ((tenant, tenant_sequence), arrival_s) in enumerate(zip(order, times)):
            rows.append({
                "request_id": f"arrival-s{args.seed}-{args.workload}-{sequence:03d}",
                "qid": f"arrival-s{args.seed}-{args.workload}-{sequence:03d}",
                "tenant": tenant,
                "tenant_sequence": tenant_sequence,
                "modality": "video",
                "video": args.video,
                "frame_count": frames[tenant][tenant_sequence],
                "arrival_s": arrival_s,
                "max_tokens": 32,
                "priority": 0,
                "prompt_override": "Briefly describe the video.",
                "class": "background",
                "arrival_pattern": pattern,
                "interarrival_scv_target": 1.0 if pattern == "poisson" else args.scv,
                "target_aggregate_rate_qps": args.rate_qps,
                "trace_seed": args.seed,
                "tenant_frame_mode": "fixed" if args.workload == "heavy" else "mixed_equal",
            })
        path = args.output / f"{pattern}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        mean_gap = statistics.mean(gaps)
        manifest["patterns"][pattern] = {
            "path": str(path),
            "arrival_span_s": times[-1],
            "realized_span_rate_qps": (len(times) - 1) / times[-1],
            "sample_interarrival_scv": statistics.pvariance(gaps) / (mean_gap * mean_gap),
            "min_interarrival_s": min(gaps),
            "max_interarrival_s": max(gaps),
            "frame_counts": dict(Counter(row["frame_count"] for row in rows)),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
