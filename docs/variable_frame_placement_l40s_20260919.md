# Variable-frame preparation placement on L40S (2026-09-19)

## Status and scope

The matched six-policy mechanism ablation completed on one NVIDIA L40S with
zero request errors. The local result bundle is
`large_sweeps/variable_frame_placement_l40s_20260919/`.

This is one trial on one repeated 313-second H.264 video in fixed policy order.
It is a mechanism diagnostic, not a statistically supported paper result.

## Workload and capacity

- Qwen2.5-VL-7B-Instruct, 32 output tokens per request.
- 60 requests, uniformly interleaved over 15 seconds.
- Three tenants with 20 requests each.
- Per tenant: five 1-frame, ten 16-frame, and five 128-frame requests.
- Four CPU preparation workers, two GPU preparation-job slots, four reserved
  decoder lanes, four inference slots, and handoff capacity eight.
- Fixed policies route requests below 32 frames to CPU and requests at or above
  32 frames to GPU with two lanes.
- Adaptive policies may consider CPU/GPU placement for every frame count and
  one, two, or four GPU lanes.

Isolated calibration measured preparation service in seconds as follows:

| Frames | CPU | GPU-1 | GPU-2 | GPU-4 |
|---:|---:|---:|---:|---:|
| 1 | 0.423 | 0.759 | 0.829 | 0.813 |
| 16 | 4.714 | 6.796 | 3.701 | 2.234 |
| 128 | 30.646 | 9.326 | 5.588 | 3.576 |

## Aggregate results

| Policy | Mean E2E (s) | Median E2E (s) | p95 E2E (s) | Throughput (req/s) |
|---|---:|---:|---:|---:|
| Fixed FCFS | 30.19 | 28.35 | 77.17 | 0.581 |
| Fixed fair | 43.60 | 45.88 | 81.38 | 0.562 |
| Placement-only FCFS | 42.53 | 46.59 | 80.29 | 0.503 |
| Width-only FCFS | 29.18 | 20.45 | 81.53 | 0.557 |
| Adaptive FCFS | 42.06 | 44.02 | 91.62 | 0.486 |
| Full Conductor | 39.58 | 42.62 | 78.79 | 0.518 |

Full Conductor improves over adaptive FCFS by 5.9% in mean E2E, 3.2% in
median E2E, and 14.0% in p95 E2E, while increasing throughput by 6.5%. It also
improves every aggregate metric over placement-only FCFS. These comparisons
support the value of coordinating tenant selection with an adaptive policy.

Full Conductor does not beat the calibrated fixed FCFS policy. Relative to
fixed FCFS, its mean and median are 31.1% and 50.4% higher, its p95 is 2.1%
higher, and its throughput is 10.8% lower. Width-only FCFS achieves a 27.9%
lower median than fixed FCFS but has a 5.7% higher p95 and 4.1% lower
throughput.

## Routing diagnosis

Fixed FCFS sends all 45 one-/16-frame requests to CPU and all 15 128-frame
requests to GPU. This mapping already follows the isolated crossover.

Adaptive FCFS sends 13 of 30 16-frame requests to GPU and 9 of 15 128-frame
requests to CPU. Full Conductor sends three 16-frame and two one-frame requests
to GPU, while sending six 128-frame requests to CPU. CPU preparation for
128-frame requests averages about 33 seconds, versus 9--15 seconds for the
observed GPU-routed requests. The adaptive choices therefore add CPU pressure
to a workload whose static size split already balances the backends well.

GPU service also changes under concurrent serving. The online GPU preparation
profile reports 24.9% MAPE for adaptive FCFS and 40.6% for full Conductor,
compared with roughly 8--9% for their CPU profiles. Isolated lane calibration
is therefore insufficient to predict GPU preparation service under concurrent
decoding and inference in this run.

## Claim boundary and next steps

Do not claim that adaptive placement beats static placement from this run.
The defensible result is narrower: tenant-aware coordination improves the
tested adaptive policy, while a calibrated static threshold remains better on
this workload.

Before repeating the comparison, update placement estimates to reflect
concurrent GPU service and add a conservative fallback that avoids routing
128-frame work to CPU unless the predicted GPU wait exceeds the measured CPU
penalty. Evaluate the revised policy over multiple seeds, videos, arrival
patterns, and policy-order rotations. Retain fixed FCFS as the primary strong
baseline.

## Conservative-routing rerun

The requested profiler and fallback changes were implemented and rerun on the
same L40S, video, trace construction, capacity, and fixed policy order. The
result bundle is
`large_sweeps/variable_frame_placement_l40s_conservative_20260919/`.

GPU cost prediction was already separated by decoder width. The revision adds
the dispatch-time load context (active GPU preparation jobs, reserved lanes,
and inference-occupancy bucket) to each width-specific online profile. It also
retains the isolated calibrated route unless the alternative improves predicted
readiness by at least two seconds and 20%. For this calibration, requests below
32 frames prefer CPU and requests at or above 32 frames prefer GPU.

| Policy | Mean E2E (s) | Median E2E (s) | p95 E2E (s) | Throughput (req/s) |
|---|---:|---:|---:|---:|
| Fixed FCFS | 32.71 | 29.12 | 77.75 | 0.576 |
| Fixed fair | 43.20 | 44.37 | 80.91 | 0.583 |
| Placement-only FCFS | 28.72 | 19.52 | 78.35 | 0.574 |
| Width-only FCFS | 30.11 | 20.98 | 88.66 | 0.541 |
| Adaptive FCFS | 29.47 | 24.17 | 85.23 | 0.575 |
| Full Conductor | 50.99 | 49.78 | 104.23 | 0.518 |

The routing correction worked: all adaptive policies sent all 15 one-frame and
all 30 sixteen-frame requests to CPU, and all 15 128-frame requests to GPU.
Placement-only FCFS improved by 32.5% in mean and 58.1% in median relative to
the earlier adaptive run, with roughly unchanged p95. In the revised run it is
12.2% lower in mean and 33.0% lower in median than fixed FCFS, but p95 and
throughput are effectively tied. Because this is a single repetition in fixed
order and the unchanged fixed FCFS control itself varied by 8.3% in mean across
the two runs, these differences are diagnostic rather than a paper claim.

Full Conductor became slower. It assigned one decoder lane to eight of the 15
large requests, compared with two under adaptive FCFS, and accumulated 40.25
seconds of mean preparation-queue wait versus 19.37 seconds for adaptive FCFS.
Its fair lane cap produced balanced reserved GPU-lane shares during the reported
overlap window (31.8%, 39.1%, and 29.1%), but the lower lane widths and request
order increased latency. This run therefore separates two conclusions:
the conservative placement rule fixes the observed backend misrouting, while
the current fair width allocator has a substantial latency cost and needs a
different policy or an explicit fairness/latency objective.

The width-specific online profiles remain sparse: the full policy observed only
eight, six, and one requests at widths one, two, and four. Their MAPE values were
16.9%, 36.7%, and 25.7%, respectively. Load-aware keys provide the intended
fallback hierarchy, but one run does not validate prediction accuracy for each
load context. Multi-seed, rotated-order experiments remain required.
