#!/usr/bin/env python3
"""Generate VTC-style multi-tenant workloads for multimodal serving.

The scenarios mirror the empirical properties evaluated by VTC, but replace
text-token length with heterogeneous video preparation cost.  A trace is
policy-independent so every scheduler sees identical requests and arrivals.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable
from pathlib import Path


SCENARIOS = (
    "constant_overload",
    "work_conserving",
    "on_off_under_share",
    "on_off_backlogged",
    "on_off_prep_heavy",
    "on_off_two_tenant",
    "poisson_short_long",
    "poisson_mixed_cost",
    "poisson_heterogeneous_fairness",
    "noisy_neighbor_isolation",
    "cross_modality_noisy_neighbor",
    "aggressive_asymmetric",
    "rate_asymmetric_constant_cost",
    "heterogeneous_prep_cost",
    "heterogeneous_inference_cost",
    "simultaneous_cross_stage_heterogeneity",
    "rate_sweep_constant_cost",
    "work_conserving_two_tenant",
    "distribution_shift",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deterministic_arrivals(rate_qps: float, start: float, end: float) -> list[float]:
    if rate_qps <= 0 or end <= start:
        return []
    interval = 1.0 / rate_qps
    return [start + index * interval for index in range(math.ceil((end - start) / interval))
            if start + index * interval < end]


def poisson_arrivals(
    rng: random.Random, rate_qps: float, start: float, end: float,
) -> list[float]:
    arrivals = []
    now = start
    while rate_qps > 0:
        now += rng.expovariate(rate_qps)
        if now >= end:
            break
        arrivals.append(now)
    return arrivals


def on_off_arrivals(
    rate_qps: float,
    *,
    duration_s: float,
    period_s: float,
    duty_cycle: float,
) -> list[float]:
    arrivals = []
    start = 0.0
    while start < duration_s:
        arrivals.extend(deterministic_arrivals(
            rate_qps, start, min(duration_s, start + period_s * duty_cycle),
        ))
        start += period_s
    return arrivals


def phase_arrivals(
    phases: Iterable[tuple[float, float, float]],
) -> list[float]:
    arrivals = []
    for start, end, rate in phases:
        arrivals.extend(deterministic_arrivals(rate, start, end))
    return arrivals


def workload(
    scenario: str,
    *,
    duration_s: float,
    capacity_qps: float,
    rng: random.Random,
) -> dict[str, dict]:
    """Return per-tenant arrivals and frame-budget rules."""
    if scenario == "constant_overload":
        # Both tenants remain backlogged despite different offered rates.
        return {
            "a": {"arrivals": deterministic_arrivals(0.8 * capacity_qps, 0, duration_s), "frames": [32]},
            "b": {"arrivals": deterministic_arrivals(1.6 * capacity_qps, 0, duration_s), "frames": [32]},
        }
    if scenario == "work_conserving":
        # Two tenants use less than a third of capacity; the overloaded tenant
        # should consume the unused service without leaving workers idle.
        return {
            "a": {"arrivals": deterministic_arrivals(2 / 13 * capacity_qps, 0, duration_s), "frames": [32]},
            "b": {"arrivals": deterministic_arrivals(4 / 13 * capacity_qps, 0, duration_s), "frames": [32]},
            "c": {"arrivals": deterministic_arrivals(9 / 13 * capacity_qps, 0, duration_s), "frames": [32]},
        }
    if scenario == "work_conserving_two_tenant":
        # A requests less than its one-half share. B remains overloaded and
        # should consume A's unused capacity without delaying A.
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    0.25 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    1.0 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
        }
    if scenario == "on_off_under_share":
        return {
            "a": {"arrivals": on_off_arrivals(
                0.35 * capacity_qps, duration_s=duration_s,
                period_s=60.0, duty_cycle=0.5,
            ), "frames": [32]},
            "b": {"arrivals": deterministic_arrivals(1.25 * capacity_qps, 0, duration_s), "frames": [32]},
        }
    if scenario == "on_off_backlogged":
        return {
            "a": {"arrivals": on_off_arrivals(
                1.25 * capacity_qps, duration_s=duration_s,
                period_s=60.0, duty_cycle=0.5,
            ), "frames": [32]},
            "b": {"arrivals": deterministic_arrivals(1.5 * capacity_qps, 0, duration_s), "frames": [32]},
        }
    if scenario == "on_off_prep_heavy":
        # A preparation-heavy tenant arrives in periodic bursts while a
        # second tenant continuously submits cheaper work.  The aggregate
        # offered preparation demand exceeds capacity, making this the
        # cross-stage analogue of VTC's ON/OFF fairness experiment.
        return {
            "a": {"arrivals": on_off_arrivals(
                1.0 * capacity_qps, duration_s=duration_s,
                period_s=60.0, duty_cycle=0.5,
            ), "frames": [128]},
            "b": {"arrivals": deterministic_arrivals(
                1.0 * capacity_qps, 0, duration_s,
            ), "frames": [8, 32]},
        }
    if scenario == "on_off_two_tenant":
        # A alternates between under-share demand and idleness. B is always
        # overloaded. This tests work conservation and counter reactivation.
        return {
            "a": {
                "arrivals": on_off_arrivals(
                    0.25 * capacity_qps,
                    duration_s=duration_s,
                    period_s=60.0,
                    duty_cycle=0.5,
                ),
                "frames": [32],
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    1.0 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
        }
    if scenario == "poisson_short_long":
        return {
            "a": {"arrivals": poisson_arrivals(rng, 1.5 * capacity_qps, 0, duration_s), "frames": [8]},
            "b": {"arrivals": poisson_arrivals(rng, 0.75 * capacity_qps, 0, duration_s), "frames": [128]},
        }
    if scenario == "poisson_mixed_cost":
        return {
            "a": {"arrivals": poisson_arrivals(rng, 1.5 * capacity_qps, 0, duration_s), "frames": [8, 128]},
            "b": {"arrivals": poisson_arrivals(rng, 0.75 * capacity_qps, 0, duration_s), "frames": [32, 64]},
        }
    if scenario == "poisson_heterogeneous_fairness":
        # Both tenants remain backlogged under stochastic arrivals. A sends
        # smaller videos more frequently; B sends larger videos less often.
        return {
            "a": {
                "arrivals": poisson_arrivals(
                    rng, 1.5 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
            "b": {
                "arrivals": poisson_arrivals(
                    rng, 0.75 * capacity_qps, 0, duration_s
                ),
                "frames": [128],
            },
        }
    if scenario == "noisy_neighbor_isolation":
        third = duration_s / 3
        return {
            "a": {"arrivals": deterministic_arrivals(0.35 * capacity_qps, 0, duration_s), "frames": [32]},
            "b": {"arrivals": phase_arrivals([
                (0, third, 0.35 * capacity_qps),
                (third, 2 * third, 1.0 * capacity_qps),
                (2 * third, duration_s, 2.0 * capacity_qps),
            ]), "frames": [32]},
        }
    if scenario == "cross_modality_noisy_neighbor":
        # A continuously submits preparation-heavy video requests. B is a
        # low-rate text-only victim. The experiment compares B's TTFT alone,
        # under unconstrained decoder threading, and with bounded decoding.
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    1.0 * capacity_qps, 0, duration_s
                ),
                "frames": [128],
                "modality": "video",
                "max_tokens": 32,
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    0.25 * capacity_qps, 0, duration_s
                ),
                "frames": [0],
                "modality": "text",
                "max_tokens": 32,
            },
        }
    if scenario == "aggressive_asymmetric":
        # A offers four times the request rate of B and C while every A
        # request also carries the largest preparation budget. This stresses
        # isolation at both preparation and inference admission. B is the
        # cheap victim; C represents a normal mixed-cost tenant.
        normal_rate = 0.4 * capacity_qps
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    4.0 * normal_rate, 0, duration_s
                ),
                "frames": [128],
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    normal_rate, 0, duration_s
                ),
                "frames": [8],
            },
            "c": {
                "arrivals": deterministic_arrivals(
                    normal_rate, 0, duration_s
                ),
                "frames": [8, 32],
            },
        }
    if scenario == "rate_asymmetric_constant_cost":
        # Isolate offered-rate asymmetry with two continuously backlogged
        # tenants: A sends twice as many requests as B, while both submit the
        # same 32-frame request class. At 1.6x and 0.8x the reference capacity,
        # both offered rates exceed an unweighted tenant's one-half share.
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    1.6 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    0.8 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
        }
    if scenario == "heterogeneous_prep_cost":
        # Equal request rates but unequal CPU preparation costs. Both tenants
        # are offered more than an unweighted half-share so they remain
        # preparation-backlogged. Tenant round-robin equalizes request counts,
        # not worker-time, and therefore is not service-fair in this case.
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    0.8 * capacity_qps, 0, duration_s
                ),
                "frames": [128],
                "max_tokens": 32,
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    0.8 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
                "max_tokens": 32,
            },
        }
    if scenario == "heterogeneous_inference_cost":
        # Equal preparation costs and request rates, but unequal requested
        # generation lengths. The prompt override makes the long-output
        # tenant request actual generation work instead of merely setting a
        # larger upper bound that a multiple-choice answer would never use.
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    0.8 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
                "max_tokens": 128,
                "prompt_override": (
                    "Describe the supplied video frames in chronological "
                    "order. Produce a detailed response of approximately "
                    "100 tokens; do not answer with only a letter."
                ),
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    0.8 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
                "max_tokens": 8,
                "prompt_override": (
                    "Summarize the supplied video frames using exactly one "
                    "word."
                ),
            },
        }
    if scenario == "simultaneous_cross_stage_heterogeneity":
        # Equal-rate tenants stress opposite stages in the same run. Tenant A
        # is preparation-heavy but requests a short answer; tenant B is
        # preparation-light but requests a long answer. The offered rate is
        # intentionally above each tenant's nominal half-share. Validate that
        # both tenants remain backlogged at both stages in measured results.
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    0.8 * capacity_qps, 0, duration_s
                ),
                "frames": [128],
                "max_tokens": 8,
                "prompt_override": (
                    "Summarize the supplied video frames using exactly one "
                    "word."
                ),
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    0.8 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
                "max_tokens": 128,
                "prompt_override": (
                    "Describe the supplied video frames in chronological "
                    "order. Produce a detailed response of approximately "
                    "100 tokens; do not answer with only a letter."
                ),
            },
        }
    if scenario == "rate_sweep_constant_cost":
        # Here capacity_qps is the aggregate offered request rate selected by
        # the load sweep. A and B use identical requests with a 3:2 rate ratio.
        return {
            "a": {
                "arrivals": deterministic_arrivals(
                    0.6 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
            "b": {
                "arrivals": deterministic_arrivals(
                    0.4 * capacity_qps, 0, duration_s
                ),
                "frames": [32],
            },
        }
    if scenario == "distribution_shift":
        third = duration_s / 3
        return {
            "a": {"arrivals": phase_arrivals([
                (0, third / 2, 0.35 * capacity_qps),
                (third, 2 * third, 0.8 * capacity_qps),
                (2 * third, duration_s, 0.35 * capacity_qps),
            ]), "frames": [8, 32]},
            "b": {"arrivals": phase_arrivals([
                (0, third, 1.25 * capacity_qps),
                (third, 2 * third, 0.8 * capacity_qps),
                (2 * third, duration_s, 1.25 * capacity_qps),
            ]), "frames": [32, 128]},
        }
    raise ValueError(f"unsupported scenario: {scenario}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument(
        "--capacity-qps", type=float, default=0.22,
        help="Approximate measured system capacity used to scale offered load.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exclude-qid", action="append", default=[])
    args = parser.parse_args()
    if args.duration_s <= 0 or args.capacity_qps <= 0:
        parser.error("duration and capacity must be positive")

    excluded = set(args.exclude_qid)
    rows = [
        row for row in load_jsonl(args.dataset)
        if str(row.get("qid") or row.get("question_id") or row.get("id"))
        not in excluded
    ]
    if not rows:
        raise SystemExit("dataset contains no eligible requests")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    spec = workload(
        args.scenario,
        duration_s=args.duration_s,
        capacity_qps=args.capacity_qps,
        rng=rng,
    )

    trace = []
    cursor = 0
    for tenant, tenant_spec in spec.items():
        frame_budgets = tenant_spec["frames"]
        for index, arrival_s in enumerate(tenant_spec["arrivals"]):
            row = dict(rows[cursor % len(rows)])
            cursor += 1
            frame_count = frame_budgets[index % len(frame_budgets)]
            row.update({
                "request_id": f"{tenant}-{index}",
                "tenant": tenant,
                "class": "background",
                "arrival_s": float(arrival_s),
                "priority": 10,
                "frame_count": frame_count,
                "scenario": args.scenario,
            })
            modality = str(tenant_spec.get("modality", "video"))
            row["modality"] = modality
            if modality == "text":
                choices = "\n".join(
                    f"{chr(ord('A') + choice_index)}. {choice}"
                    for choice_index, choice in enumerate(
                        row.get("choices") or []
                    )
                )
                row["prompt_override"] = (
                    "Answer this text-only multiple-choice question.\n\n"
                    f"Question: {row['question']}\n\n{choices}\n\n"
                    "Return only the answer letter."
                )
            if tenant_spec.get("max_tokens") is not None:
                row["max_tokens"] = int(tenant_spec["max_tokens"])
            if tenant_spec.get("prompt_override") is not None:
                row["prompt_override"] = str(tenant_spec["prompt_override"])
            trace.append(row)
    trace.sort(key=lambda row: (
        float(row["arrival_s"]), str(row["tenant"]), str(row["request_id"]),
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in trace:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps({
        "scenario": args.scenario,
        "output": str(args.output),
        "requests": len(trace),
        "duration_s": args.duration_s,
        "capacity_qps": args.capacity_qps,
        "tenants": {
            tenant: sum(row["tenant"] == tenant for row in trace)
            for tenant in spec
        },
    }, indent=2))


if __name__ == "__main__":
    main()
