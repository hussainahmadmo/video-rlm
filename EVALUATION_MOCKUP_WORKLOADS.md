# Multimodal Fairness Evaluation Mockups

All figures in `analysis/figures/requested_workload_mockups/` are expected-result
mockups, not measured data. CPU preparation and inference-admission service use
different units and are never added together.

## Common methodology

- Run exactly the same request trace under every applicable scheduling policy.
- Use three or more random seeds and report the mean with 95% confidence intervals.
- Policies: FCFS, tenant round-robin, preparation-only max-min,
  inference-only/VTC-style admission, and full cross-stage max-min.
- Keep model, dataset, prompts, output-token limits, frame-selection method,
  worker counts, vLLM configuration, and random seeds fixed across policies.
- Report errors and completion rate in addition to successful-request metrics.
- Measure preparation service as observed occupied worker-time.
- Report inference as **inference-admission service** unless exact per-request GPU
  occupancy is available. vLLM continuous batching means admission counts and
  GPU seconds are not interchangeable.
- Compute tenant service gaps only during intervals in which every compared
  tenant is backlogged at the measured stage.

For two backlogged tenants, preparation service gap is

\[
D_{\mathrm{prep}}(t)=
\left|S_A^{\mathrm{prep}}(0,t)-S_B^{\mathrm{prep}}(0,t)\right|.
\]

Define inference-admission gap analogously using the chosen inference service
function. Smaller and bounded gaps indicate better service fairness.

## 1. Constant-rate, two-tenant fairness

**Question.** Does the scheduler prevent a higher-rate tenant from accumulating
more service when both tenants are continuously backlogged?

**Workload.** Use two separate heterogeneous traces. In the preparation-heavy
trace, both tenants have the same request rate and output limit, but A uses 128
frames and B uses 32. In the inference-heavy trace, both use 32 frames and the
same request rate, but A requests a roughly 100-token description while B
requests one word. Both tenants must remain backlogged at the measured stage.

Use the same trace in two matched stage configurations:

- Preparation-heavy: fewer preparation workers and ample inference capacity.
- Inference-heavy: ample preparation capacity and restricted vLLM concurrency.

**Axes.** X is elapsed time. Y is the raw service gap between A and B, in CPU
worker-seconds or inference-service units. Expected: preparation-only and
cross-stage bound the CPU gap; inference-only and cross-stage bound the
inference gap.

These heterogeneous traces expose why equal-turn tenant round-robin is not
equivalent to service fairness. Keep the original identical-cost, 2:1-rate
trace as a separate noisy-neighbor experiment; in that trace, round-robin is
expected to approximate fair service.

## 2. ON/OFF work conservation

**Question.** Does max-min immediately reallocate unused capacity rather than
reserving service for an idle tenant?

**Workload.** Run two matched experiments after separately measuring
preparation capacity and inference capacity. Requests use identical 32-frame
inputs. In each run, Tenant A offers service demand equal to 0.25 of the
measured stage capacity for 30 seconds and then sends nothing for 30 seconds.
Tenant B remains backlogged. Repeat for at least four cycles. Determine
backlog independently at the stage being measured.

**Axes.** X is time. The first two Y axes are preparation and inference service
rates as fractions of their separately measured stage capacities, averaged
over fixed windows. The third Y axis is windowed P95 end-to-end latency.
Expected: B receives approximately 0.75 capacity while A is ON and all capacity
while A is OFF; total utilization remains near one. A's latency remains stable
when it returns. B's latency can grow because its offered demand intentionally
exceeds capacity; that growth is not evidence of unfairness.

## 3. Heterogeneous video cost

**Question.** Does service-aware scheduling remain fair when equal request
counts have unequal preparation costs?

**Workload.** Both tenants remain preparation-backlogged. Hold codec,
resolution, prompt, and output limit fixed. Tenant A samples 128 frames and
Tenant B samples 32 frames. For codec and resolution sensitivity, repeat this
experiment while changing one characteristic at a time.

**Axes.** X is time. Y is received preparation service rate in one panel and
completed preparations in the other. Expected: tenants receive similar
worker-time, while the cheaper tenant completes more preparations. Do not claim
that a 4x frame ratio guarantees exactly 4x cost; use observed worker-time.

## 4. Constant underloaded/overloaded work conservation

**Question.** Can a backlogged tenant consume capacity unused by a tenant whose
demand is below its nominal fair share?

**Workload.** Identical 32-frame requests. Tenant A continuously offers 0.25 of
capacity and is not backlogged. Tenant B offers at least full capacity and
remains backlogged.

**Axes.** X is time. Y is received stage service rate and accumulated service.
Expected: A receives all requested service, B receives the remaining 0.75, and
total utilization remains one. The growing raw difference between A and B is
not unfair because A is not backlogged.

## Required companion metrics

For each policy and workload, also report:

- victim and worst-tenant P50/P95/P99 TTFT and end-to-end latency;
- slowdown relative to the same tenant running alone;
- throughput and completion rate;
- CPU and GPU utilization;
- preparation-queue and model-ready-queue lengths;
- scheduler decision overhead;
- profiler prediction error and reconciliation error where applicable.

These companion metrics distinguish improved fairness from lower utilization,
reduced throughput, or accidental throttling before requests reach inference.
