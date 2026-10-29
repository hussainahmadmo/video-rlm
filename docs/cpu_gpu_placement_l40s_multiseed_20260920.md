# CPU/GPU preparation placement on L40S (2026-09-20)

## Question

Does using heterogeneous CPU/GPU preparation capacity improve end-to-end
serving latency after matching the number of preparation slots, and does the
current adaptive policy improve on a calibrated static split?

## Isolation and workload

The comparison holds tenant selection, inference admission, and GPU width
fixed. All policies use FCFS at preparation and inference. GPU preparation uses
the independently implemented FlashCodec-inspired shared-chunk backend with a
fixed width of two decoder lanes.

Each of three tenants submits five one-frame, ten 16-frame, and five 128-frame
requests, for 60 requests per policy. Arrivals are independently randomized
over 15 seconds for each of three seeds. Policy order rotates across seeds.
Each seed starts a fresh model server and recalibrates complete preparation
time for every frame count on CPU and at GPU widths one, two, and four.

The four policies are:

- CPU-only with four preparation workers;
- CPU-only with six workers, matching the mixed policies' four CPU plus two GPU
  preparation-job slots;
- static CPU/GPU placement, sending one-/16-frame work to CPU and 128-frame work
  to GPU;
- adaptive CPU/GPU placement with four CPU and two GPU slots. It retains the
  calibrated static route unless another option improves predicted readiness
  by at least two seconds and 20%.

All runs use Qwen2.5-VL-7B on one L40S, one repeated source video, four
inference-submission slots, handoff capacity eight, and 32 output tokens. The
experiment runs in the pod exposed on port 40526 and is pinned to logical CPUs
0--7 on NUMA node 0. A separate placement-rate sweep runs in the port-32002 pod
on NUMA node 1; both pods share the same physical server, so this separation
reduces CPU interference but is not independent-host replication.

## Results

Values are means across three seeds; uncertainty is one sample standard
deviation.

| Policy | Mean E2E (s) | Median E2E (s) | p95 E2E (s) | Throughput (req/s) |
|---|---:|---:|---:|---:|
| CPU-only, 4 workers | 80.78 ± 8.25 | 76.66 ± 10.42 | 141.97 ± 5.07 | 0.336 ± 0.006 |
| CPU-only, 6 workers | 57.13 ± 5.79 | 54.41 ± 6.96 | 103.89 ± 1.88 | 0.434 ± 0.016 |
| Static CPU/GPU | 33.25 ± 2.64 | 30.40 ± 4.63 | 75.40 ± 3.85 | 0.595 ± 0.018 |
| Adaptive CPU/GPU | 33.16 ± 3.23 | 30.46 ± 4.56 | 76.39 ± 1.14 | 0.571 ± 0.006 |

Averaging paired per-seed ratios, static CPU/GPU placement lowers mean,
median, and p95 E2E latency by 41.7%, 44.2%, and 27.4% relative to the
six-worker equal-slot CPU baseline, while increasing throughput by 37.1%.
Adaptive placement lowers those latency metrics by 41.9%, 44.1%, and 26.5%
and increases throughput by 31.5%.

Both mixed policies route all 45 one-frame requests and all 90 16-frame
requests across the three seeds to CPU, and all 45 128-frame requests to GPU.
Adaptive placement therefore does not improve on the calibrated static split
in this balanced workload: relative to static, it changes mean E2E by -0.4%,
median by +0.2%, p95 by +1.4%, and throughput by -4.0%. The defensible result
is the benefit of heterogeneous CPU/GPU preparation capacity; a separate rate
sweep tests whether adaptive placement helps when contention changes the
fastest route.

## Claim boundary

This experiment can establish the benefit of adding and selecting heterogeneous
preparation capacity relative to matched CPU-only capacity. If static and
adaptive policies select the same routes, it does not establish an adaptive
routing advantage. It uses one source video and one L40S; the independent
arrival seeds and rotated order do not provide video or hardware generality.

Raw artifacts are stored in
`large_sweeps/cpu_placement_multiseed_l40s_20260920/`. Derived CSV files and the
paper figure are produced by
`conductor/experiments/scripts/analyze/analyze_cpu_placement_multiseed.py`.
