# Preparation noisy-neighbor fairness on L40S (2026-09-19)

## Scope

The result bundle is
`large_sweeps/noisy_neighbor_fairness_l40s_20260919/`. This is a one-video,
one-policy-order mechanism pilot on one L40S, not a statistical result.

Tenant A sends twelve 128-frame requests in a burst from 0 to 0.55 seconds.
Tenants B and C each send six later requests at two-second intervals,
alternating one and sixteen frames. All requests use CPU preparation and share
four preparation workers, four inference slots, and handoff capacity eight.
The tested order was inference-only fairness, two-stage max-min, preparation-only
fairness, and FCFS. All 96 measured requests completed without errors.

## Results

| Policy | Mean E2E (s) | Median E2E (s) | p95 E2E (s) | B/C median (s) | B/C p95 (s) | B/C mean prep wait (s) | Throughput (req/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 62.49 | 69.41 | 86.32 | 73.38 | 86.58 | 62.66 | 0.246 |
| Inference-only fairness | 62.69 | 69.24 | 86.97 | 74.01 | 87.06 | 63.35 | 0.244 |
| Preparation-only fairness | 53.07 | 49.00 | 98.61 | 40.16 | 63.36 | 29.30 | 0.235 |
| Two-stage max-min | 52.89 | 49.26 | 99.07 | 40.25 | 63.96 | 29.57 | 0.235 |

Relative to FCFS, two-stage max-min reduces B/C median E2E by 45.1%, B/C p95
by 26.1%, and B/C mean preparation waiting by 52.8%. Their median time to become
model-ready falls from 72.28 to 39.34 seconds (45.6%), and p95 model-ready time
falls from 85.16 to 62.74 seconds (26.3%). Overall mean and median fall by 15.4%
and 29.0%.

Inference-only fairness changes none of those outcomes materially: its B/C
median is 0.9% higher than FCFS and its B/C mean preparation wait is 1.1% higher.
This directly demonstrates the boundary described by the paper: a request still
blocked before model submission cannot benefit from inference admission policy.

The gain is not free. Tenant A's p95 rises from 69.44 to 101.77 seconds, overall
p95 rises by 14.8%, and throughput falls by 4.5%. Preparation-only and two-stage
policies are nearly identical here because preparation dominates and every
request uses the same 32-token output budget. This pilot supports a preparation
fairness benefit for late light tenants, not an across-the-board tail or
throughput improvement.

Preparation service shares are demand-weighted and should not be interpreted as
equal-share targets: A contributes far more work. The saved shares shift only
from 62.4/18.9/18.7% under FCFS to 60.0/20.2/19.8% under two-stage max-min. The
main evidence is when B/C receive service and become model-ready, rather than
aggregate service equality after the complete trace drains.

## Next validation

Repeat across at least three arrival seeds and rotate policy order. Vary the A
burst size, B/C arrival interval, and CPU-worker count to show the contention
boundary. Report B/C model-ready delay and E2E distributions alongside A's tail
and throughput so the isolation/efficiency tradeoff remains explicit.
