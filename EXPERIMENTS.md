# Experiment Results

## Highlight results

The central result is that vLLM engine-level priority does not protect an
urgent multimodal request while that request is still waiting for external
media preparation. Carrying the request priority through media preparation
substantially reduces urgent time to first token (TTFT).

### Status overview

| Experiment | Status |
| --- | --- |
| Full trace suite | **COMPLETE** |
| Background-backlog scaling | **COMPLETE** |
| Preparation-capacity reservation | **COMPLETE** |
| Fixed-load replica scaling | **COMPLETE** |
| Proportional-load replica scaling | **COMPLETE** |
| Static CPU+GPU isolation baseline | **QUEUED (runs after stress and fairness)** |
| Adaptive-policy validation | **COMPLETE (negative result)** |
| Preparation-worker scaling | **COMPLETE (108/108)** |
| Worker-scaling accuracy analysis | **COMPLETE** |
| External CPU decode ablation | **COMPLETE (36/36)** |
| External NVDEC matrix | **BLOCKED / PARTIAL (14/54)** |
| Native vLLM raw-video priority control | **COMPLETE (2/2 priority modes)** |
| Native vLLM backend controls | **NOT STARTED (0/6)** |
| Low-frame priority validation (2/4/8 frames) | **COMPLETE (18/18)** |
| Background/urgent frame-cost matrix | **COMPLETE (24/24)** |
| Repeated backlog stress (five seeds) | **RUNNING on two GPUs (0/75 at launch)** |
| Sustained-load fairness and aging | **QUEUED (runs after repeated stress)** |
| Representative-trace replay | **READY / REQUIRES TRACE TEMPLATE** |
| Second-model/machine portability subset | **READY / REQUIRES SERVED MODEL** |

### Remaining-paper evaluation launchers

The remaining experiments are collected behind
[`run_paper_evaluation_suite.sh`](run_paper_evaluation_suite.sh). Individual
resumable launchers are available for repeated backlog stress, the four
background/urgent cost regimes, sustained-load fairness, static CPU/GPU
isolation, representative-trace replay, and a compact second-model or
second-machine portability validation. Every launcher excludes the known
corrupt QID before sampling or copying traces, preserves matched traces across
policies, archives incomplete output directories, and skips completed
`summary.json` files.

The unified analyzer
[`analyze_paper_evaluation.py`](conductor/experiments/scripts/analyze/analyze_paper_evaluation.py)
writes per-run and matched-pair CSV files plus a Markdown summary. In addition
to urgent latency, it retains throughput, accuracy, background latency,
maximum background wait, and SLO attainment so urgent improvements cannot hide
starvation or quality regressions.

### Dataset and common settings

The experiments use the same source pool but not always the same subset of
videos. The pool contains 249 video-question examples: 50 each from EgoSchema,
NExT-QA, VRBench, and STAR, plus 49 from LVBench. Within every reported policy,
worker-count, replica-count, or decode-backend comparison, the compared runs
use the **exact same arrival trace, requests, videos, questions, frame budgets,
and priorities**. This matched design isolates the setting being changed.
Across different trace seeds and experiment families, the selected subset can
differ, so points in different figures should not be interpreted as repeated
measurements of one identical set of videos.

Unless a figure-specific block below says otherwise, the runner uses
Qwen/Qwen2.5-VL-7B-Instruct, 32 output tokens, maximum image side 448 pixels,
`max_pixels=100352`, a vLLM server configured with native priority scheduling,
lower numeric values as higher priority (`urgent=0`, `background=10`), four
vLLM request slots per replica, and a prepared-request queue depth of 32.
Mixed-frame traces use budgets of 8, 16, 32, 64, or 128 frames. Frame budgets
are sampled with a duration-aware rule: shorter videos draw from smaller
budgets and longer videos draw from larger budgets. Consequently, the duration
plots use a multivariate fit that controls for frame count rather than treating
duration and frame budget as independent by construction.

### Low-frame priority validation

**Status: COMPLETE (18/18 configurations)**

This experiment tests the claim that upstream priority inversion remains even
when video preparation is made deliberately small. It uses the exact same
videos and arrivals across all comparisons and overwrites every request's frame
budget with 2, 4, or 8 frames. The matrix contains three burst traces (urgent
arrival at 10 seconds, seeds 1--3), two media-preparation policies (FCFS and
priority), and three uniform frame budgets, for 18 configurations total.

All runs use 64 low-priority background requests and 16 high-priority urgent
requests per trace, four seek-based CPU preparation workers, one FFmpeg thread
per preparation task, four vLLM slots, queue depth 32, a 600-second decode
timeout, and matched Qwen/Qwen2.5-VL-7B-Instruct inference. The comparison will
report urgent preparation wait, end-to-end mean/p95 TTFT, background TTFT,
throughput, and accuracy.

| Frames/request | Policy | Urgent mean TTFT | Urgent p95 TTFT | Urgent prep wait | Background mean TTFT | Throughput |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | FCFS preparation | 1.64 s | 2.46 s | 0.72 s | 4.79 s | 6.276 QPS |
| 2 | Priority preparation | 1.59 s | 2.44 s | 0.72 s | 4.81 s | 6.304 QPS |
| 4 | FCFS preparation | 9.93 s | 11.67 s | 8.54 s | 9.26 s | 3.654 QPS |
| 4 | Priority preparation | 3.98 s | 5.96 s | 2.55 s | 10.59 s | 3.657 QPS |
| 8 | FCFS preparation | 27.11 s | 30.45 s | 24.57 s | 17.29 s | 1.938 QPS |
| 8 | Priority preparation | 6.54 s | 10.15 s | 3.88 s | 22.60 s | 1.920 QPS |

At two frames, both policies meet a five-second mean and p95 target and
priority changes urgent mean TTFT by only about 3%. Thus, the experiment does
not support the claim that two-frame requests are intrinsically slow. At four
frames, priority reduces urgent mean TTFT by 60.0% (2.50x); at eight frames it
reduces it by 75.9% (4.15x). Throughput is effectively unchanged within each
frame budget, and urgent accuracy is identical between policies (50.0%, 62.5%,
and 60.4% at 2, 4, and 8 frames). The result localizes the motivation: upstream
priority inversion is negligible when preparation remains below capacity, but
emerges rapidly once per-request preparation work is sufficient to build a
queue.

Results:
[`large_sweeps/low_frame_priority_validation_20260824_161125`](large_sweeps/low_frame_priority_validation_20260824_161125)

Launcher:
[`run_low_frame_priority_validation.sh`](run_low_frame_priority_validation.sh)

### Background/urgent frame-cost matrix

**Status: READY / NOT STARTED (0/24 configurations)**

This controlled 2x2 matrix separates background preparation cost from urgent
preparation cost. It evaluates background/urgent frame budgets of `128/8`
(heavy/light), `8/128` (light/heavy), `128/128` (heavy/heavy), and `8/8`
(light/light). Each condition uses the same three `burst_urgent10` traces,
seeds 1--3, and matched FCFS/priority preparation, yielding 24 runs.

All conditions hold constant the 64 background and 16 urgent questions,
arrival times, selected videos, four seek-based CPU preparation workers, one
FFmpeg thread per task, four vLLM slots, queue depth 32, and native vLLM
priority. Only the frame budget assigned to each request class changes. The
matrix will quantify whether priority benefit is caused by queued background
work, the urgent request's own service cost, or both.

Heavy-background/light-urgent and light/light behavior already appears in
earlier experiments, but those results come from different experiment
families. Rerunning every cell in one matrix avoids hardware/time confounds and
provides directly comparable effect sizes.

Launcher:
[`run_frame_cost_matrix.sh`](run_frame_cost_matrix.sh)

### Full trace suite

**Status: COMPLETE (legacy external-preparation characterization; excluded
from headline claims)**

This sweep predates the corrected native-vLLM per-request frame-budget path.
It remains useful for diagnosing how an explicit external FCFS queue scales,
but its maximum speedup is not used as a system headline or as evidence for
the native-vLLM integration.

The completed suite contains 120 matched traces per policy and 1,920 urgent
requests per policy. The table uses the derived results that exclude QID
`20520eff-abdf-4d4f-94ad-cc751a8960d0`.

| Media-preparation policy | Urgent mean TTFT | Urgent p95 TTFT | Mean throughput |
| --- | ---: | ---: | ---: |
| FCFS | 97.26 s | 285.64 s | 0.2660 QPS |
| Priority | 45.43 s | 95.74 s | 0.2657 QPS |
| Priority + one reserved worker | 41.40 s | 92.08 s | 0.2389 QPS |

Compared with FCFS, ordinary priority reduces urgent mean TTFT by **53.3%**
(2.14x) and p95 TTFT by **66.5%**, while preserving mean throughput. Adding a
reserved preparation worker reduces mean TTFT by **57.4%** (2.35x), with an
approximately 10% throughput cost across the full suite.

Urgent-request SLO attainment also improves:

| Policy | <=10 s | <=30 s | <=60 s |
| --- | ---: | ---: | ---: |
| FCFS | 5.3% | 20.2% | 45.4% |
| Priority | 7.0% | 33.8% | 72.7% |
| Priority + reserved worker | 10.6% | 38.8% | 78.2% |

Cleaned results:
[`large_sweeps/cleaned_priority_results_no_qid_20260823/runpod_core/priority_trace_parallel_full-20260823_074146`](large_sweeps/cleaned_priority_results_no_qid_20260823/runpod_core/priority_trace_parallel_full-20260823_074146)

Dashboard image:
[`priority_results_dashboard.png`](large_sweeps/imported_runpod_20260823/priority_trace_parallel_full-20260823_074146/priority_results_dashboard.png)

Arrival-pattern image:
[`priority_by_arrival_pattern.png`](large_sweeps/imported_runpod_20260823/priority_trace_parallel_full-20260823_074146/priority_by_arrival_pattern.png)

Confidence-interval image:
[`priority_trace_ci.png`](large_sweeps/imported_runpod_20260823/priority_trace_parallel_full-20260823_074146/priority_trace_ci.png)

Stage-breakdown image:
[`priority_stage_breakdown_all_traces.png`](large_sweeps/imported_runpod_20260823/priority_trace_parallel_full-20260823_074146/priority_stage_breakdown_all_traces.png)

**Settings for all four full-suite figures:** 120 matched traces per policy;
five seeds; 64 background and 16 urgent video questions per trace; burst,
2-second staggered, and Poisson background arrivals at 0.25, 0.5, 0.75, and
1.0 QPS; urgent arrivals at 1, 10, 30, or 60 seconds; mixed 8--128-frame
budgets; four preparation workers; four vLLM slots; queue depth 32; and FCFS,
priority, or priority plus one reserved preparation worker. Policies see the
same videos within each trace, while videos are reshuffled from the 249-example
pool between seeds. Error bars in the confidence-interval figure are computed
over matched trace-level results.

### Background-backlog scaling

**Status: COMPLETE (legacy external-preparation stress test; excluded from
headline claims)**

Priority keeps urgent mean TTFT near 25--28 seconds while FCFS latency grows
with the number of background videos.

| Background requests | FCFS urgent mean TTFT | Priority urgent mean TTFT | Speedup |
| ---: | ---: | ---: | ---: |
| 8 | 48.71 s | 26.75 s | 1.82x |
| 16 | 86.16 s | 25.31 s | 3.40x |
| 32 | 157.07 s | 28.37 s | 5.54x |
| 64 | 289.09 s | 26.30 s | 10.99x |
| 128 | 550.20 s | 26.89 s | excluded from corrected claims |

The backlog trend demonstrates upstream priority inversion in the external
preparation harness. Do not quote the 128-request point as the maximum gain of
the corrected native-vLLM system.

Cleaned results:
[`large_sweeps/cleaned_priority_results_no_qid_20260823/backlog_scaling`](large_sweeps/cleaned_priority_results_no_qid_20260823/backlog_scaling)

Image:
[`urgent_ttft_backlog_scaling.png`](large_sweeps/imported_runpod_all_20260823/priority_backlog_scaling_20260822_164745/urgent_ttft_backlog_scaling.png)

**Figure settings:** one matched burst workload per point; 8, 16, 32, 64, or
128 low-priority background requests arriving at 0 seconds; 16 high-priority
urgent requests arriving at 1 second; background videos sampled at 128 frames;
urgent videos sampled at 8 frames; four preparation workers; four vLLM slots;
queue depth 32; and the same requests/videos for FCFS and priority at each
backlog size. This is a controlled stress test rather than the mixed-duration,
mixed-frame full trace suite.

### Preparation-capacity reservation

**Status: COMPLETE**

All runs use four preparation workers. Reserving one worker is the best
observed trade-off.

| Background limit | Reserved workers | Urgent mean TTFT | Background mean TTFT | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 30.01 s | 73.90 s | 0.4191 QPS |
| 3 | 1 | 25.83 s | 71.00 s | 0.3994 QPS |
| 2 | 2 | 26.12 s | 107.91 s | 0.2985 QPS |
| 1 | 3 | 22.73 s | 229.88 s | 0.1611 QPS |

One reserved worker reduces urgent mean TTFT by 13.9% for a 4.7% throughput
cost. Reserving two or three workers imposes excessive background-latency and
throughput costs.

Cleaned results:
[`large_sweeps/cleaned_priority_results_no_qid_20260823/nature_reservation_ablation`](large_sweeps/cleaned_priority_results_no_qid_20260823/nature_reservation_ablation)

### Fixed-load replica scaling

**Status: COMPLETE**

Adding vLLM replicas alone does not remove the upstream preparation bottleneck.

| Replicas | FCFS urgent mean TTFT | Priority urgent mean TTFT | Reduction |
| ---: | ---: | ---: | ---: |
| 1 | 170.22 s | 38.94 s | 77.1% |
| 2 | 162.23 s | 37.24 s | 77.0% |
| 4 | 159.79 s | 32.13 s | 79.9% |

Cleaned results:
[`large_sweeps/cleaned_priority_results_no_qid_20260823/runpod_core/replica_scaling_pilot_20260823_182009`](large_sweeps/cleaned_priority_results_no_qid_20260823/runpod_core/replica_scaling_pilot_20260823_182009)

Image:
[`fixed_load_replica_scaling.png`](large_sweeps/imported_runpod_20260823/replica_scaling_pilot_20260823_182009/fixed_load_replica_scaling.png)

**Figure settings:** the exact `burst_urgent10-seed1` mixed-frame trace is used
at every point (64 background plus 16 urgent requests, 80 total); one, two, or
four A40-backed vLLM replicas; least-inflight replica routing; four shared
preparation workers; four vLLM slots per replica; queue depth 32; and matched
FCFS/priority runs. Offered load is fixed while serving capacity increases.

### Proportional-load replica scaling

**Status: COMPLETE**

When offered load grows with the replica count, preparation contention becomes
more pronounced and priority becomes increasingly valuable.

| Replicas | FCFS urgent mean TTFT | Priority urgent mean TTFT | Reduction |
| ---: | ---: | ---: | ---: |
| 1 | 22.74 s | 21.01 s | 7.6% |
| 2 | 86.47 s | 30.59 s | 64.6% |
| 4 | 226.19 s | 66.79 s | 70.5% |

Cleaned results:
[`large_sweeps/cleaned_priority_results_no_qid_20260823/runpod_core/proportional_replica_scaling_20260823_185015`](large_sweeps/cleaned_priority_results_no_qid_20260823/runpod_core/proportional_replica_scaling_20260823_185015)

Image:
[`proportional_load_replica_scaling.png`](large_sweeps/imported_runpod_20260823/proportional_replica_scaling_20260823_185015/proportional_load_replica_scaling.png)

**Figure settings:** the same 80-request base trace is copied once per replica,
giving 80, 160, and 320 requests for one, two, and four A40 replicas
respectively (64/16 background/urgent requests per copy); least-inflight
routing; four shared preparation workers; four vLLM slots per replica; queue
depth 32; mixed 8--128-frame budgets; and matched FCFS/priority runs. Unlike
the fixed-load graph, offered load grows in direct proportion to replica count.

### Static CPU+GPU isolation baseline

**Status: READY / NOT STARTED (0/27 configurations)**

This experiment answers whether urgent traffic should simply receive dedicated
resources. It compares three systems using the same total resources: two A40
vLLM replicas, four media-preparation workers, four request slots per replica,
and a total handoff depth of 32.

1. `shared_engine_only` uses four shared FCFS preparation workers and balances
   both request classes across both replicas.
2. `shared_priority` uses four shared priority-aware preparation workers and
   balances both classes across both replicas.
3. `static_isolation` dedicates three preparation workers, 24 handoff slots,
   and port 9000 to background traffic; it dedicates one preparation worker,
   eight handoff slots, and port 9001 to urgent traffic.

The launcher also accepts disjoint `BACKGROUND_CPUSET` and `URGENT_CPUSET`
values. When supplied, each preparation worker thread and the decoder
subprocesses it launches are pinned to the corresponding CPU set, providing a
physical CPU-core isolation control in addition to logical worker, handoff,
and GPU-replica isolation. CPU sets must be selected from the host's allowed
affinity and should be reported with the result.

The default matrix contains nine matched traces (burst, staggered, and Poisson
0.5 QPS, each with three seeds) and three policies, for 27 runs. The comparison
reports urgent mean/p95 TTFT, background mean TTFT, throughput, and verifies
that no request is routed to the other class's replica. It quantifies both the
latency benefit and the utilization cost of static isolation relative to the
work-conserving shared priority scheduler.

Launcher:
[`run_static_isolation_baseline.sh`](run_static_isolation_baseline.sh)

Analyzer:
[`analyze_static_isolation_baseline.py`](conductor/experiments/scripts/analyze/analyze_static_isolation_baseline.py)

### Adaptive-policy result

**Status: COMPLETE (negative result)**

The current SLO-adaptive implementation is a negative result and should not be
presented as an improvement.

| Policy | Urgent mean TTFT | Urgent p95 TTFT | Throughput |
| --- | ---: | ---: | ---: |
| Priority | 30.36 s | 46.87 s | 0.4435 QPS |
| Priority + reserved worker | 27.67 s | 44.11 s | 0.3978 QPS |
| SLO-adaptive | 41.59 s | 52.39 s | 0.4447 QPS |

Cleaned results:
[`large_sweeps/cleaned_priority_results_no_qid_20260823/nature_adaptive_validation`](large_sweeps/cleaned_priority_results_no_qid_20260823/nature_adaptive_validation)

### Preparation-worker scaling

**Status: COMPLETE (108/108 configurations)**

The completed worker-scaling suite contains 108 configurations: four worker
counts, nine matched traces, and three preparation policies. Increasing
parallelism improves every policy, but it does not make FCFS competitive with
priority-aware preparation.

| Workers | Policy | Urgent mean TTFT | Background mean TTFT | Throughput | Urgent speedup vs FCFS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | FCFS | 240.60 s | 232.97 s | 0.1524 QPS | -- |
| 2 | Priority | 67.54 s | 265.92 s | 0.1530 QPS | 3.56x |
| 2 | Priority + reserved worker | 60.57 s | 429.66 s | 0.0967 QPS | 3.97x |
| 4 | FCFS | 133.76 s | 121.67 s | 0.2490 QPS | -- |
| 4 | Priority | 50.32 s | 138.86 s | 0.2489 QPS | 2.66x |
| 4 | Priority + reserved worker | 45.86 s | 163.06 s | 0.2173 QPS | 2.92x |
| 8 | FCFS | 95.59 s | 84.44 s | 0.3407 QPS | -- |
| 8 | Priority | 39.34 s | 98.84 s | 0.3388 QPS | 2.43x |
| 8 | Priority + reserved worker | 37.60 s | 98.95 s | 0.3351 QPS | 2.54x |
| 16 | FCFS | 87.62 s | 79.10 s | 0.3579 QPS | -- |
| 16 | Priority | 33.48 s | 91.50 s | 0.3576 QPS | 2.62x |
| 16 | Priority + reserved worker | 32.71 s | 92.39 s | 0.3577 QPS | 2.68x |

At 16 workers, ordinary priority reduces urgent mean TTFT by **61.8%**
(2.62x), from 87.62 to 33.48 seconds, with effectively unchanged aggregate
throughput. Background mean TTFT increases by 15.7%. This is the best balanced
operating point: adding workers reduces contention, but priority remains
necessary to prevent head-of-line blocking. Reserved capacity is most useful
under severe contention; at two workers it produces the largest urgent gain
but imposes an 84% background-latency increase and a substantial throughput
cost.

#### Reviewer concern: would asynchronous preparation or more workers solve the problem?

The evaluated preparation path is already concurrent. The runner uses a
`ThreadPoolExecutor` with `--prep-workers` preparation tasks in flight, a
separate concurrent pool for vLLM submissions, and a dispatcher that continues
handling arrivals and completions while both pools execute. It does not submit
the entire workload immediately to a FIFO executor. Instead, it retains
unstarted requests in an explicit, reorderable pending heap and admits a new
task whenever a preparation slot becomes available. A bounded ready queue
separates preparation from inference admission.

Consequently, this experiment does not compare priority scheduling against a
purely serial preparation baseline. It asks whether increasing the concurrency
of an already parallel preparation tier removes the need for priority-aware
admission. The answer is no: increasing the pool from 2 to 16 workers improves
FCFS urgent mean TTFT from 240.60 to 87.62 seconds, but priority-aware admission
still lowers it to 33.48 seconds at 16 workers (2.62x faster) with essentially
the same throughput (0.3579 versus 0.3576 QPS). Parallelism increases capacity;
it does not specify which waiting request should receive that capacity next.

The implementation uses futures and thread pools rather than an `asyncio` HTTP
gateway, so the precise paper description is **a concurrent media-preparation
pool with asynchronous dispatch and completion handling**. Replacing threads
with coroutines would improve overlap for I/O-bound operations but would not
make CPU decoding, storage bandwidth, memory bandwidth, or decoder capacity
unbounded. Once offered preparation work exceeds any of these finite
resources, a queue forms and its admission order still matters. The current
system reorders queued work but intentionally does not preempt preparation
already executing.

Implementation:
[`run_mixed_end_to_end_priority.py`](conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py)

Worker-scaling results:
[`large_sweeps/prep_worker_scaling_2shard_20260823_184626`](large_sweeps/prep_worker_scaling_2shard_20260823_184626)

Latency/throughput image:
[`worker_scaling_priority_tradeoff.png`](large_sweeps/prep_worker_scaling_2shard_20260823_184626/worker_scaling_priority_tradeoff.png)

**Figure settings:** nine matched mixed-frame traces: burst at urgent=10
seconds, staggered at urgent=10 seconds, and Poisson 0.5 QPS at urgent=30
seconds, each with seeds 1--3; normally 64 background plus 16 urgent requests
per trace; 2, 4, 8, or 16 preparation workers; seek-based CPU preparation with
one FFmpeg thread per decode task; four vLLM slots; queue depth 32; and FCFS,
priority, or priority plus one reserved worker. The corrupt QID filter leaves
79 rather than 80 requests in affected traces. The two shards use separate A40
GPUs and disjoint NUMA CPU sets, and an entire trace remains on one shard for
all matched configurations.

Each latency/throughput point is the mean across nine matched traces. Its error
bar is a t-based 95% confidence interval over the nine trace-level means. These
bars characterize sensitivity to different arrival traces, not per-request
p95 latency, minima/maxima, or GPU measurement error. FCFS's large urgent-TTFT
intervals show that it is particularly sensitive to workload shape. Because
the policies use matched traces, formal significance testing should use the
paired per-trace difference (FCFS minus priority), rather than infer
significance solely from whether the displayed marginal intervals overlap.

### Accuracy under worker scaling

**Status: COMPLETE**

Scheduling changes latency without changing answer quality. All 108
worker-scaling configurations completed with zero request errors.

| Policy | Urgent correct | Urgent accuracy | Background correct | Background accuracy |
| --- | ---: | ---: | ---: | ---: |
| FCFS | 338/576 | 58.68% | 1523/2280 | 66.80% |
| Priority | 340/576 | 59.03% | 1519/2280 | 66.62% |
| Priority + reserved worker | 339/576 | 58.85% | 1519/2280 | 66.62% |

Accuracy image:
[`worker_scaling_accuracy.png`](large_sweeps/prep_worker_scaling_2shard_20260823_184626/worker_scaling_accuracy.png)

**Figure settings:** identical to the worker-scaling latency/throughput figure
above. Accuracy pools the completed answers across worker counts and the same
nine traces for each policy; it does not compare different video subsets
between policies. Each interval is a 95% Wilson binomial confidence interval.

Accuracy points are pooled correct/total proportions. Their error bars are 95%
Wilson binomial confidence intervals. The intervals overlap almost completely,
supporting the conclusion that the priority latency gains do not come from an
accuracy trade-off.

### External CPU decode ablation

**Status: COMPLETE (36/36 configurations)**

All 36 configurations in the CPU decode ablation completed. One-pass
sequential CPU decoding is a negative result for these long, sparsely sampled
videos.

| Policy | Decode method | Urgent mean TTFT | Throughput | Total errors |
| --- | --- | ---: | ---: | ---: |
| FCFS | Per-frame CPU seek | 132.54 s | 0.2519 QPS | 0 |
| FCFS | One-pass batch CPU | 732.22 s | 0.0371 QPS | 40 |
| Priority | Per-frame CPU seek | 50.56 s | 0.2508 QPS | 0 |
| Priority | One-pass batch CPU | 172.28 s | 0.0373 QPS | 39 |

The one-pass implementation decodes from the beginning of each video through
the final requested timestamp. It therefore reduces repeated seeks but performs
far more decoding for long videos. It lowers throughput by approximately 85%
and loses every matched urgent-TTFT comparison. Batching alone is not an
appropriate optimization for this workload.

Across 357 paired successful question/frame-count cases, a log-linear fit that
controls for frame count estimates that seek-based preparation scales as
`duration^0.16` and `frames^1.00`, whereas one-pass CPU decoding scales as
`duration^1.06` and `frames^0.21`. Thus, seeking is driven mainly by the number
of requested frames, while one-pass decoding grows approximately linearly with
video duration.

In plain language, targeted seeking jumps near the requested frames instead of
watching the entire video. At a fixed frame budget, doubling video duration
increases measured seek-preparation time by only about 12%; doubling the frame
budget approximately doubles it. One-pass decoding instead processes the video
from its beginning through the last requested timestamp. Doubling video
duration therefore increases its preparation time by approximately 2.1x.

This distinction explains the negative batching result. For long videos with
sparse frame sampling, batching frames into a single sequential pass performs
substantially more decoding than seeking directly to those frames. More CPU
workers can overlap additional requests, but they do not reduce this per-video
work and eventually contend for CPU and storage bandwidth. Decode optimization
and scheduling therefore address separate problems: the backend determines how
much preparation work is performed, while priority determines which waiting
request receives preparation capacity first.

CPU-ablation results:
[`large_sweeps/batched_cpu_ablation_2shard_20260823_184626`](large_sweeps/batched_cpu_ablation_2shard_20260823_184626)

Duration-scaling image:
[`cpu_decode_duration_scaling.png`](large_sweeps/batched_cpu_ablation_2shard_20260823_184626/cpu_decode_duration_scaling.png)

Combined CPU/NVDEC duration-scaling image:
[`decode_duration_scaling_with_nvdec.png`](large_sweeps/batched_cpu_ablation_2shard_20260823_184626/decode_duration_scaling_with_nvdec.png)

**CPU duration-figure settings:** the same nine traces and video questions used
by worker scaling; matched FCFS and priority runs; seek CPU versus one-pass
sequential CPU decode; four preparation workers; one FFmpeg thread per task;
four vLLM slots; queue depth 32; 600-second decode timeout; mixed 8--128-frame
budgets; and 357 unique successful `(question, frame-count)` pairs present for
both CPU backends. Each plotted service time is the median across available
matched repeats. Both CPU panels therefore contain the same paired cases.

**NVDEC panel settings:** one completed matched trace
(`burst_urgent10-seed1`), batch PyNvVideoCodec NVDEC, four preparation workers,
four vLLM slots, queue depth 32, 600-second decode timeout, and 77 unique
successful `(question, frame-count)` cases aggregated across FCFS and priority.
It uses videos from the same source pool and the same trace as the corresponding
CPU runs, but it is **not** matched to all 357 CPU cases: two timeouts and later
aborted NVDEC traces are excluded. The NVDEC scatter and fitted exponents are
therefore preliminary and can include survivor bias.

### Preliminary external NVDEC result

**Status: BLOCKED / PARTIAL (14/54 configurations)**

The NVDEC matrix is incomplete and must not yet be presented as a final
backend comparison. Fourteen of 54 configurations produced summaries. Only
`burst_urgent10-seed1` produced a complete matched three-backend comparison.

| Policy | Decode method | Urgent mean TTFT | Urgent p95 TTFT | Throughput | Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| FCFS | Per-frame CPU seek | 258.43 s | 284.88 s | 0.2622 QPS | 0 |
| FCFS | One-pass batch CPU | 1276.07 s | 1371.28 s | 0.0470 QPS | 4 |
| FCFS | Batch NVDEC | 362.87 s | 412.46 s | 0.0962 QPS | 2 |
| Priority | Per-frame CPU seek | 46.40 s | 73.20 s | 0.2621 QPS | 0 |
| Priority | One-pass batch CPU | 83.30 s | 86.57 s | 0.0491 QPS | 4 |
| Priority | Batch NVDEC | 76.53 s | 114.42 s | 0.0944 QPS | 2 |

On this single trace, NVDEC is approximately twice as fast as one-pass batch
CPU in throughput, but remains below per-frame CPU seeking. Two requests for
the same MKV video timed out after 600 seconds. The following two traces then
aborted inside PyNvVideoCodec on an FLV container with error 801 (`Resetting
the demuxer to original position failed`). The NVDEC experiment requires a
per-video CPU fallback before it can be resumed safely.

The combined duration plot includes a clearly marked exploratory NVDEC panel.
Across the 77 unique successful question/frame-count cases available from the
one completed trace, the same log-linear fit estimates `duration^0.41 ± 0.11`
and `frames^0.22`. This successful-only subset suggests that batch NVDEC is
less duration-sensitive than sequential CPU decoding (`duration^1.06`), but
more duration-sensitive than targeted CPU seeking (`duration^0.16`). Because
timeouts and aborted configurations are absent, this estimate is
survivor-biased and is not a final or directly matched backend comparison.

Partial NVDEC results:
[`large_sweeps/nvdec_ablation_2shard_20260823_184626`](large_sweeps/nvdec_ablation_2shard_20260823_184626)

## Data-quality note

The raw RunPod copy of video QID
`20520eff-abdf-4d4f-94ad-cc751a8960d0` was corrupt. Nature has a valid 180-second
copy, but no reruns were performed. Instead, this QID was removed consistently
from a derived result tree. The filter removed 382 occurrences across the
documented result sets: 370 background requests and 12 urgent requests. All 308
occurrences in the full trace and replica suite were background requests, so
its pooled urgent TTFT and SLO values are unchanged. The backlog table above
was recomputed after removing its affected urgent rows. Raw results and
transferred archives remain immutable.

Filtering utility:
[`conductor/experiments/scripts/analyze/filter_mixed_results_qid.py`](conductor/experiments/scripts/analyze/filter_mixed_results_qid.py)

Cleaned-result root:
[`large_sweeps/cleaned_priority_results_no_qid_20260823`](large_sweeps/cleaned_priority_results_no_qid_20260823)

The existing PNG files were generated before the derived filtering step. Their
full-suite values remain applicable because the excluded QID was a background
request in that suite; the cleaned backlog values differ slightly from its
existing image. Regenerate final publication figures from the cleaned summaries
before submission so every displayed value, request count, and throughput
matches the cleaned tables.

## Multi-tenant priority and fairness

### Matched equal-tenant validation

**Status: COMPLETE (24/24 runs; four independent seeds)**

This validation separates application priority from request cost and tenant
fairness. Every tenant receives the same number of requests and the same mixed
8/32/128-frame cost distribution. Each run uses four CPU preparation workers,
four vLLM slots, a bounded prepared queue of 16, CPU seeking with one FFmpeg
thread per preparation task, and equal tenant weights. All policies use the
same trace for a seed, and all 24 summaries report zero request errors.

| Policy | Foreground mean E2E | Foreground p95 E2E | E2E <=60 s | Peak CPU-service lead | Worst/best tenant slowdown | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FCFS | 176.12 s | 198.71 s | 0.0% | 201.0 s | 1.68x | 0.210 QPS |
| Strict priority | 37.18 s | 62.39 s | 85.4% | 181.3 s | 2.52x | 0.215 QPS |
| Shortest-job-first | 62.59 s | 192.49 s | 75.0% | 141.2 s | 1.27x | 0.213 QPS |
| Tenant fairness | 176.00 s | 199.55 s | 0.0% | 64.2 s | 1.15x | 0.226 QPS |
| Tenant-fair priority | 44.72 s | 74.21 s | 77.1% | 51.8 s | 1.19x | 0.230 QPS |
| Fair slowdown | 63.43 s | 189.12 s | 75.0% | 72.7 s | 1.15x | 0.216 QPS |

Tenant fairness alone does not protect foreground work: adding priority inside
each tenant reduces foreground mean completion latency from 176.00 to 44.72
seconds (3.94x faster, 74.6% lower) and increases throughput from 0.226 to
0.230 QPS. Compared with globally strict priority, hierarchical tenant
priority trades a 20.3% increase in foreground mean latency for substantially
stronger isolation: the largest cumulative CPU-service lead while all tenants
are backlogged falls from 181.3 to 51.8 seconds (71.4% lower), the preparation
dispatch lead falls from 12.0 to 5.25 requests (56.3% lower), and the
worst/best tenant-slowdown ratio falls from 2.52x to 1.19x (52.7% lower).

These are empirical rather than theorem-level guarantees. Every one of the 24
runs completes 100% of requests without error, so no observed foreground or
background request starves. Tenant-fair priority's maximum background
preparation wait is 160.6 seconds, 10.8% below globally strict priority, while
throughput is 6.9% higher. CPU preparation service is measured directly from
start and finish timestamps. Because the API does not expose per-request GPU
occupancy under concurrent batching, engine-side isolation is reported using
admission-count lead and per-tenant slowdown rather than incorrectly treating
concurrent residence time as additive GPU service.

The policy is VTC-inspired rather than a claim of a new fair-queueing
primitive: it selects the least-served tenant using weighted virtual service,
then applies application priority within that tenant at both preparation and
inference admission. Background aging prevents an old background request from
remaining behind newer foreground requests within its tenant. The paper's
systems claim is that this ordering and isolation must cover media preparation
as well as engine admission.

Results and generated report:
[`large_sweeps/multitenant_scheduling_20260826_104047`](large_sweeps/multitenant_scheduling_20260826_104047)

Figure:
[`large_sweeps/multitenant_scheduling_20260826_104047/figures/multitenant_scheduling_tradeoff.pdf`](large_sweeps/multitenant_scheduling_20260826_104047/figures/multitenant_scheduling_tradeoff.pdf)

Cross-stage empirical report and figure:
[`analysis/figures/cross_stage_fairness.md`](analysis/figures/cross_stage_fairness.md),
[`analysis/figures/cross_stage_fairness.pdf`](analysis/figures/cross_stage_fairness.pdf)

Reproducible analyzer:
[`conductor/experiments/scripts/analyze/analyze_cross_stage_fairness.py`](conductor/experiments/scripts/analyze/analyze_cross_stage_fairness.py)

### Sustained overload and noisy-neighbor validation

**Status: READY; not started because no vLLM ports are currently available and
Nature's NVIDIA driver is unavailable pending the requested reboot.**

The follow-up suite uses continuous arrivals rather than a single closed burst.
It includes an equal-tenant scenario and a noisy-neighbor scenario in which
tenant A submits 36 background requests while tenants B and C submit nine
each. All tenants still submit six foreground requests. The suite compares
FCFS, globally strict priority, tenant fairness, and tenant-fair priority over
three seeds (24 runs total). It reports foreground TTFT and completion SLO
attainment, completion goodput, tenant fairness, worst-tenant slowdown,
background progress while foreground requests are active, and maximum
background preparation wait.

Launchers:

- [`run_multitenant_sustained_validation.sh`](run_multitenant_sustained_validation.sh)
- [`run_multitenant_sustained_2shard.sh`](run_multitenant_sustained_2shard.sh)
- [`run_multitenant_sustained_with_servers.sh`](run_multitenant_sustained_with_servers.sh)

Once vLLM replicas are healthy on ports 9020 and 9021, start the complete
two-GPU suite with:

```bash
cd /dataheart/hussainahmad/video-rlm
nohup ./run_multitenant_sustained_2shard.sh \
  > logs/multitenant_sustained_2shard_launcher.log 2>&1 &
```

After a machine reboot, when no vLLM replicas are already running, the recovery
launcher can instead start both servers, run the suite, generate the report and
figure, and release the two GPUs automatically:

```bash
cd /dataheart/hussainahmad/video-rlm
nohup ./run_multitenant_sustained_with_servers.sh \
  > logs/multitenant_sustained_with_servers.log 2>&1 &
```

### VTC-style cross-stage empirical validation

**Status: READY; code and trace-generation smoke tests pass.**

This suite adapts the empirical tests from Virtual Token Counter (VTC) to a
multimodal pipeline. It does not equate a CPU second with a GPU token. Instead,
it maintains and reports separate per-tenant virtual-service counters for
media preparation and inference-engine residence. The matched baselines are:

- `fcfs`: no tenant-aware ordering at either stage;
- `engine_tenant_fair`: VTC-style tenant ordering only after requests become
  model-ready, while media preparation remains FCFS;
- `tenant_fair`: tenant virtual service controls admission at both media
  preparation and inference.

The nine scenarios test distinct empirical properties:

1. `constant_overload`: two continuously backlogged tenants with unequal
   request rates; tests whether service difference remains bounded.
2. `work_conserving`: two under-share tenants and one overloaded tenant; tests
   whether the overloaded tenant can consume otherwise idle capacity.
3. `on_off_under_share`: one intermittent tenant below its fair share and one
   continuously overloaded tenant; tests work conservation during OFF periods.
4. `on_off_backlogged`: an intermittent tenant injects enough work to remain
   backlogged during OFF periods; tests equal service between backlogged
   tenants.
5. `poisson_short_long`: stochastic arrivals with short-video versus
   long-video tenants; tests heterogeneous preparation costs.
6. `poisson_mixed_cost`: stochastic arrivals where each tenant has a different
   mixture of frame budgets; tests robustness when request count is a poor
   service metric.
7. `noisy_neighbor_isolation`: one tenant's offered load increases over three
   phases while another stays below share; tests latency isolation.
8. `distribution_shift`: ON/OFF, equal-overload, and unequal-load phases in one
   trace; tests counter behavior as the active demand distribution changes.
9. `on_off_prep_heavy`: tenant A periodically bursts 128-frame requests while
   tenant B continuously submits alternating 8- and 32-frame requests; tests
   whether engine-only fairness can protect tenants when the dominant and
   heterogeneous service demand occurs before engine admission.

The analyzer reports cumulative preparation-service difference,
inference-residence difference, rolling preparation service rate, rolling
end-to-end latency, preparation utilization, throughput, mean/p95 completion
latency, and noisy-neighbor isolation. The engine-residence value is explicitly
labeled as residence rather than GPU compute time because vLLM does not expose
per-request GPU occupancy in these result files.

Files:

- [`generate_vtc_multimodal_trace.py`](conductor/experiments/scripts/run/generate_vtc_multimodal_trace.py)
- [`run_vtc_multimodal_validation.sh`](run_vtc_multimodal_validation.sh)
- [`run_vtc_multimodal_2shard.sh`](run_vtc_multimodal_2shard.sh)
- [`analyze_vtc_multimodal.py`](conductor/experiments/scripts/analyze/analyze_vtc_multimodal.py)

With vLLM replicas ready on ports 9020 and 9021, run the full three-seed,
two-GPU matrix (72 runs) with:

```bash
cd /dataheart/hussainahmad/video-rlm
nohup ./run_vtc_multimodal_2shard.sh \
  > logs/vtc_multimodal_2shard_launcher.log 2>&1 &
```

For a faster paper-development pass, use one seed and 120-second traces:

```bash
cd /dataheart/hussainahmad/video-rlm
nohup env SEEDS=1 DURATION_S=120 ./run_vtc_multimodal_2shard.sh \
  > logs/vtc_multimodal_pilot.log 2>&1 &
```

#### Preparation-heavy ON/OFF validation

**Status: READY AND SMOKE-TESTED; blocked on the Nature NVIDIA driver reboot.**

This focused follow-up combines the ON/OFF and heterogeneous-cost cases rather
than testing them separately. Each matched run contains 61 requests: tenant A
submits 21 128-frame requests over three ON periods, while tenant B submits 40
continuous requests alternating between 8 and 32 frames. The corrupted EgoSchema
QID is excluded. It compares FCFS, engine-only tenant fairness, and cross-stage
tenant fairness over three seeds and preparation pools of two and four workers
(18 runs total). Every policy sees the same arrivals, videos, and frame budgets
within a worker point.

Launchers:

- [`run_prep_heavy_on_off_worker.sh`](run_prep_heavy_on_off_worker.sh)
- [`run_prep_heavy_on_off_2gpu.sh`](run_prep_heavy_on_off_2gpu.sh)
- [`run_prep_heavy_on_off_1gpu_with_server.sh`](run_prep_heavy_on_off_1gpu_with_server.sh)
- [`run_prep_heavy_on_off_with_servers.sh`](run_prep_heavy_on_off_with_servers.sh)

The two worker-count points execute sequentially even when two replicas are
available. Running them concurrently would introduce cross-experiment CPU and
memory-bandwidth contention and invalidate the worker-count comparison.

Once vLLM replicas are healthy on ports 9020 and 9021:

```bash
cd /dataheart/hussainahmad/video-rlm
nohup ./run_prep_heavy_on_off_2gpu.sh \
  > logs/prep_heavy_on_off_2gpu_launcher.log 2>&1 &
```

After the requested Nature reboot, if no replicas are already running, use
the self-contained launcher instead:

```bash
cd /dataheart/hussainahmad/video-rlm
nohup ./run_prep_heavy_on_off_with_servers.sh \
  > logs/prep_heavy_on_off_with_servers.log 2>&1 &
```

## Experiment status and remaining work

The decode experiments are split into two families so their conclusions are
not conflated:

1. **External media preparation:** compare per-frame CPU seeking, one-pass
   batched CPU decode, and one-pass PyNvVideoCodec NVDEC before requests enter
   vLLM. These runs test whether faster preparation reduces the upstream queue,
   while retaining FCFS versus priority as an independent factor.
2. **Native vLLM media processing:** send `video_url` inputs and select vLLM's
   own registered video backend using `media_io_kwargs.video`. This is the
   control for whether vLLM's frontend decoder alone removes the bottleneck.

### Important interpretation of native vLLM's asynchronous path

The current native-vLLM control is asynchronous: the client can submit many
raw-video requests concurrently, and vLLM fetches each `video_url` and performs
media loading and preprocessing inside its frontend rather than through the
runner's explicit four-worker preparation pool. Asynchrony allows fetching,
decoding, preprocessing, and inference to overlap, which can make native vLLM
appear faster than an externally bounded four-worker implementation. It does
**not** mean that media preparation has infinite capacity, that queued media
work is ordered by request priority, or that upstream contention has
disappeared. Async execution specifies how work overlaps; priority scheduling
specifies which waiting work should run first.

Asynchrony is a strong baseline. It can overlap URL fetching, decoding,
preprocessing, and GPU inference, avoid an unnecessarily serialized client
pipeline, and use otherwise idle CPU or I/O capacity. This explains why native
vLLM can deliver good aggregate throughput and competitive latency when the
host has sufficient resources. The limitation is not that asynchronous media
processing is inherently slow. The limitation is that concurrency alone does
not bound admission or decide which waiting media job should receive scarce
CPU/decoder capacity first during contention.

### Corrected native-vLLM bounded-priority comparison

**Status: COMPLETE; primary corrected result**

The corrected experiment adds per-request frame budgets to native vLLM and
verifies in server traces that requests actually use 8, 16, 32, 64, or 128
frames. Both configurations retain native vLLM engine priority and native
asynchronous raw-video processing. The control otherwise keeps native media
admission. The treatment places a four-job bounded priority queue before URL
fetching and decoding, allowing urgent requests to overtake queued background
media jobs while leaving active jobs non-preemptive.

| Metric | Native asynchronous vLLM | Bounded media priority | Change |
| --- | ---: | ---: | ---: |
| Successful requests | 79/79 | 79/79 | no errors |
| Urgent mean TTFT | 53.88 s | 32.47 s | **39.7% lower; 1.66x faster** |
| Urgent median TTFT | 31.31 s | 25.25 s | 19.4% lower |
| Urgent requests meeting 30-s SLO | 25.0% | 87.5% | +62.5 percentage points |
| Aggregate throughput | 0.1027 QPS | 0.1052 QPS | effectively unchanged |
| Background mean TTFT | 122.35 s | 352.24 s | 187.9% higher |
| Urgent accuracy | 56.25% | 56.25% | unchanged |

The mechanism improves urgent latency because bounded admission prevents an
unlimited set of background media operations from entering preparation ahead
of later urgent work, and priority ordering selects urgent queued jobs first.
The background-latency increase is the explicit cost of this strict policy:
background work yields while urgent work is present. Consequently, the result
supports end-to-end priority but also motivates aging, quotas, or a less
aggressive reservation policy for production fairness.

Results:
[`large_sweeps/corrected_mixed_frames_20260825_211302`](large_sweeps/corrected_mixed_frames_20260825_211302)

Consequently, native vLLM and external preparation are complementary controls,
not a direct apples-to-apples performance comparison. The external experiments
make preparation capacity and queue order observable and fixed. The native
control has zero client-side preparation workers and leaves vLLM's internal CPU
parallelism and media queue opaque. Final comparisons must therefore report CPU
utilization/concurrency, identical traces and frame budgets, randomized run
order, errors, throughput, and tail latency in addition to urgent mean TTFT.

The completed raw-video control used 63 background and 16 urgent requests (79
total), the same valid requests and frame budgets across both modes, native
`video_url` processing, and no client-side media preparation. Assigning trace
priorities reduced urgent mean TTFT from 51.11 seconds under uniform priority
to 30.33 seconds (40.7% lower, or 1.69x faster). Both runs completed without
request errors. This demonstrates that native asynchronous processing and
engine priority are useful, but it does not yet isolate whether vLLM applies
priority within its internal video-fetch/decode/preprocessing work. A matched
CPU profile and repeated, randomized trials are required before comparing this
number directly with the explicit end-to-end scheduler.

Raw native-control results:
[`large_sweeps/native_priority_control_20260825_132310`](large_sweeps/native_priority_control_20260825_132310)

### Resource-matched native versus external CPU control

**Status: READY (default matrix: 24 runs)**

This control pins the complete serving path to the same logical-CPU budget:
the vLLM API server and EngineCore, the native-video HTTP server when used,
the external preparation runner, and all dynamically created decoder
processes. The default matrix uses 16 and 64 logical CPUs, three repetitions,
and four policies: native uniform priority, native request priority, external
FCFS preparation, and external end-to-end priority. Policy order reverses on
alternating repetitions to reduce warm-cache and run-order bias. Every policy
uses the same filtered 79-request trace.

CPU use is sampled from dynamic `/proc` process trees every 200 ms, so newly
created FFmpeg workers are included rather than treating the configured four
preparation workers as total system CPU use. The report joins mean/p95 utilized
cores, maximum threads, and CPU-seconds per request with urgent/background
TTFT, throughput, errors, and accuracy. Processes that live for less than one
sampling interval may be slightly undercounted; the same monitor is used for
every policy.

Launcher:
[`run_cpu_budget_priority_comparison.sh`](run_cpu_budget_priority_comparison.sh)

Process-tree monitor:
[`monitor_process_tree_cpu.py`](conductor/experiments/scripts/analyze/monitor_process_tree_cpu.py)

Analyzer:
[`analyze_cpu_budget_priority.py`](conductor/experiments/scripts/analyze/analyze_cpu_budget_priority.py)

Nature currently has vLLM `0.17.0`, whose installed video-loader registry
contains `opencv`, `opencv_dynamic`, `molmo2`, and `openpangu`; it does **not**
contain the documented `pyav`, `torchcodec`, `pynvvideocodec`, or `deepstream`
loaders. PyNvVideoCodec is installed, which enables the external NVDEC
experiment, but it does not make native vLLM NVDEC available. The native NVDEC
matrix must therefore wait for a compatible vLLM environment. It also requires
CUDA MPS and a positive `--mm-ipc-gpu-memory-gb` server budget.

Suite status:

1. **Complete:** preparation-worker scaling at 2, 4, 8, and 16 workers,
   holding vLLM concurrency at four (108/108 configurations).
2. **Complete:** external per-frame CPU seek versus one-pass batched CPU decode
   (36/36 configurations).
3. **Blocked after partial completion:** external CPU versus PyNvVideoCodec
   NVDEC (14/54 summaries); add container-aware CPU fallback and resume only
   incomplete configurations.
4. **Not started:** native vLLM backend matrix (0/6); it remains capability
   gated by the installed loader registry and did not start because the NVDEC
   stage aborted.
5. **Partially complete:** worker-scaling latency/throughput and accuracy
   figures now include confidence intervals; final decode-backend figures wait
   for a valid NVDEC completion.

Launchers:

- [`run_prep_worker_scaling.sh`](run_prep_worker_scaling.sh)
- [`run_batched_cpu_ablation.sh`](run_batched_cpu_ablation.sh)
- [`run_nvdec_ablation.sh`](run_nvdec_ablation.sh)
- [`run_native_media_backend_matrix.sh`](run_native_media_backend_matrix.sh)
- [`run_remaining_priority_experiments.sh`](run_remaining_priority_experiments.sh)

Native capability inspector:
[`inspect_vllm_video_backends.py`](conductor/experiments/scripts/run/inspect_vllm_video_backends.py)

### Current execution status

The remaining-experiment chain was started on Nature on 2026-08-23. Earlier
single-shard attempts (`prep_worker_scaling_20260823_170401`,
`prep_worker_scaling_20260823_170527`, and
`prep_worker_scaling_20260823_173014`) are preserved but excluded from final
analysis. The clean run uses two NUMA-isolated shards: port 9000 and GPU 0 are
pinned to CPUs `0-15,32-47`, while port 9001 and GPU 1 are pinned to CPUs
`16-31,48-63`. Whole traces are assigned to one shard so every policy and
worker count for a trace uses the same hardware partition. Its output is:
[`large_sweeps/prep_worker_scaling_2shard_20260823_184626`](large_sweeps/prep_worker_scaling_2shard_20260823_184626)

Worker scaling and the batched-CPU ablation completed. Both shards subsequently
aborted during external NVDEC decoding because PyNvVideoCodec could not reset
the demuxer for an FLV input. The native-vLLM matrix therefore did not start.
Completed summaries remain valid; only the failed NVDEC configurations and the
native capability-gated controls remain.
