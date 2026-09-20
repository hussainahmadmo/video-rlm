# Stage-specific service allocation on L40S (2026-09-20)

This experiment measures tenant service allocation only during intervals in
which all three equal-weight tenants are backlogged at the stage being studied.
It compares FCFS, preparation-only least-service scheduling, inference-only
least-service scheduling, and full Conductor.

## Workloads and configuration

Both workloads use Qwen2.5-VL-7B-Instruct on one L40S, the same source video,
18 requests per tenant, arrivals independently jittered over the first second,
and three matched seeds. Policy order rotates across seeds.

- **Preparation contention:** tenants A/B/C request 128/16/1 frames and
  32/256/32 output tokens. The run uses four non-preemptive CPU preparation
  workers, four inference submission slots, and a handoff capacity of 16.
- **Inference contention:** all tenants request one frame; A/B/C request
  256/64/32 output tokens. The run uses eight CPU preparation workers, two
  inference submission slots, and a handoff capacity of 32.

The decoder is `seek_cpu`, FFmpeg and the decoder use one thread per job, and
generation ignores EOS so output demand follows the configured budget.

## Measurement

Preparation backlog spans arrival through model readiness. Preparation service
is observed worker occupancy. Inference backlog spans model readiness through
completion. Inference service is the runner's profiled/accounted service, spread
over each request's submission-to-completion interval when intersecting the
qualifying window. It is not physical GPU utilization.

For a stage, the analyzer integrates service only while every tenant has at
least one request backlogged. It reports each tenant's fraction of total service
in those intervals and the maximum-minus-minimum share gap. An equal allocation
has 33.3% per tenant and a zero-point gap. Qualifying-window duration differs by
policy because a policy can drain one tenant sooner.

## Results

Values are means across three matched seeds; uncertainty is one sample standard
deviation. Shares are ordered A/B/C.

| Policy | Preparation shares (%) | Preparation gap (pp) | Joint prep backlog (s) | Inference shares (%) | Inference gap (pp) | Joint inference backlog (s) |
|---|---:|---:|---:|---:|---:|---:|
| FCFS | 84.2 / 14.5 / 1.3 | 82.9 ± 0.5 | 153.5 ± 8.8 | 70.9 / 19.3 / 9.8 | 61.0 ± 0.4 | 68.3 ± 2.3 |
| Preparation-only | 41.8 / 35.0 / 23.2 | 26.6 ± 4.2 | 9.0 ± 1.1 | 71.1 / 19.0 / 9.9 | 61.2 ± 0.7 | 67.5 ± 1.1 |
| Inference-only | 84.2 / 14.5 / 1.3 | 82.9 ± 0.6 | 155.3 ± 8.5 | 36.3 / 31.1 / 32.6 | 5.3 ± 0.2 | 21.3 ± 0.1 |
| Conductor | 41.8 / 36.0 / 22.2 | 27.6 ± 4.8 | 9.5 ± 1.4 | 36.1 / 31.0 / 32.8 | 5.1 ± 0.8 | 21.2 ± 0.2 |

The result isolates the scheduler boundary. Preparation-only scheduling lowers
the preparation gap but leaves inference essentially unchanged. Inference-only
scheduling has the reverse behavior. Conductor lowers both gaps. Aggregate
throughput is also stable within each workload: 0.280--0.283 requests/s for the
preparation workload and 0.731--0.738 requests/s for the inference workload.

Preparation fairness cannot revoke a large request that is already running.
The light tenant also exhausts its preparation demand sooner, ending the joint
backlog interval. These non-preemptive service quanta explain why the fair
preparation policies improve the gap without reaching zero. The two workloads
are mechanism experiments on one video and one GPU; they do not measure physical
GPU allocation or establish a formal max-min guarantee.

## Artifacts

- Preparation raw results: `large_sweeps/service_allocation_l40s_20260920/`
- Inference raw results: `large_sweeps/inference_service_allocation_l40s_20260920/`
- Derived CSV and figures:
  `large_sweeps/service_allocation_cross_stage_l40s_20260920/`
- Trace generator:
  `conductor/experiments/scripts/run/make_service_allocation_trace.py`
- Launchers: `scripts/runpod/run_service_allocation_l40s.sh` and
  `scripts/runpod/run_inference_service_allocation_l40s.sh`
- Analyzer:
  `conductor/experiments/scripts/analyze/analyze_service_allocation.py`
