# Efficient fairness pilot on L40S (2026-09-20)

## Scope

This pilot evaluates whether Conductor can retain tenant protection while
avoiding inefficient one-lane GPU preparation.  The complete corrected result
bundle is `large_sweeps/efficient_fairness_l40s_20260920_26f15be/`.  All eight
measured runs completed their 60 requests without errors.

The efficient policy adds four mechanisms to the prior joint allocator:

- a minimum GPU width of two lanes;
- work-conserving borrowing up to four lanes unless another GPU-demanding
  tenant has received less normalized service;
- a five-second normalized-service tolerance in which predicted preparation
  completion breaks tenant-ordering ties; and
- active-frontier initialization when a tenant re-enters preparation.

The five-second tolerance was fixed before the measured runs.  It was not tuned
against these results.

## Workloads

Both workloads use one L40S, Qwen2.5-VL-7B, four CPU preparation workers, two
GPU preparation-job slots, four decoder lanes, four inference slots, and a
handoff capacity of eight.

- **Balanced:** 60 requests from three tenants over 15 seconds.  Every tenant
  sends five 1-frame, ten 16-frame, and five 128-frame requests.
- **Noisy neighbor:** tenant A bursts 40 128-frame requests.  After the burst
  starts, tenant B sends ten 1-frame requests and tenant C sends ten 16-frame
  requests.

The four policies use the same trace and capacities.  Static FCFS uses fixed
CPU-small/GPU-heavy placement and two GPU lanes.  Adaptive FCFS uses load-aware
placement and width without tenant fairness.  Current Full Conductor is the
previous strict least-service allocator.  Efficient Full Conductor is the new
bounded-lag policy above.

## Balanced results

| Policy | Mean E2E (s) | Median E2E (s) | p95 E2E (s) | Throughput (req/s) | GPU widths 1 / 2 / 4 |
|---|---:|---:|---:|---:|---:|
| Static FCFS | 30.82 | 27.05 | 78.67 | 0.568 | 0 / 15 / 0 |
| Adaptive FCFS | 30.49 | 25.45 | 86.95 | 0.528 | 0 / 4 / 11 |
| Current Full Conductor | 51.77 | 49.63 | 99.95 | 0.492 | 12 / 2 / 1 |
| Efficient Full Conductor | 45.09 | 45.70 | 84.26 | 0.549 | 0 / 9 / 6 |

Relative to current Full Conductor, the efficient policy lowers mean, median,
and p95 E2E by 12.9%, 7.9%, and 15.7%, respectively, and raises throughput by
11.7%.  It eliminates one-lane execution.  It remains 79.6% worse than adaptive
FCFS in median E2E, although its p95 is 3.1% lower and throughput is 4.0% higher.
Thus efficient widths recover part of the cost, but fair request ordering still
causes the balanced-workload median gap.

## Noisy-neighbor results

| Policy | Overall median / p95 E2E (s) | Throughput (req/s) | B 1-frame median / p95 (s) | C 16-frame median / p95 (s) |
|---|---:|---:|---:|---:|
| Static FCFS | 69.08 / 210.40 | 0.268 | 4.42 / 14.79 | 15.47 / 17.01 |
| Adaptive FCFS | 71.74 / 213.67 | 0.256 | 1.32 / 8.07 | 13.15 / 15.99 |
| Current Full Conductor | 76.06 / 214.48 | 0.256 | 1.34 / 6.31 | 12.55 / 13.54 |
| Efficient Full Conductor | 74.54 / 212.48 | 0.258 | 1.33 / 6.57 | 12.80 / 13.71 |

Relative to static FCFS, efficient Full Conductor lowers B's median and p95 by
70.0% and 55.6%, and lowers C's median and p95 by 17.3% and 19.4%.  The cost is
7.9% higher overall median, 1.0% higher overall p95, and 3.9% lower throughput.
Relative to adaptive FCFS, it lowers B and C p95 by 18.5% and 14.2%, while
overall p95 is 0.6% lower and throughput is 0.7% higher; overall median is 3.9%
higher.

The efficient policy therefore preserves the useful noisy-neighbor protection
and improves the prior Full Conductor implementation.  It does not establish a
universal latency win: balanced median latency remains substantially worse than
FCFS.  The next mechanism to revise is tenant request ordering, not decoder
width.

## Evidence boundary

These are fixed-order, one-video, one-trial pilots.  They do not support
statistical significance, broad hardware claims, or a claim that Conductor
improves every metric.  A paper result needs multiple seeds and videos, rotated
policy order, and a separately selected fairness-tolerance value.
