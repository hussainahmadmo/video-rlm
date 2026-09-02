# Research Question and Evaluation Plan

## Original Research Question

> How can a multimodal serving system preserve request priority across video
> preparation and model inference so that urgent requests achieve lower
> end-to-end latency?

This question felt too narrow because its solution was largely predetermined:
propagate request priorities across stages and schedule higher-priority
requests first. The expected result was similarly straightforward---requests
moved ahead in the queue should finish earlier.

Consequently, the evaluation risked appearing self-confirming:

> Assign urgent requests higher priority across the pipeline, then show that
> urgent requests experience lower latency.

## Reframed Research Question

> How should a multimodal serving system fairly allocate service among
> competing tenants across video preparation and inference when requests have
> heterogeneous and initially unknown costs?

We should avoid writing "preparation service + inference service" because CPU
preparation time and GPU inference service do not have a natural common unit.
Instead, the system maintains separate fairness accounts for the two stages.

### Initial Approach

Whenever capacity becomes available at either stage:

1. Select the backlogged tenant that has received the least service at that
   stage.
2. Dispatch that tenant's oldest eligible request.
3. Reserve the request's predicted cost using online profiling.
4. Reconcile the reservation with measured service after completion.

Thus, max--min controls allocation across tenants, while FCFS preserves
request order within each tenant.

## Proposed System: Full Cross-Stage Max--Min

### Video-Preparation Stage

- Maintain accumulated preparation service for every tenant.
- Select the least-served backlogged tenant.
- Run that tenant's oldest eligible request.
- Reserve its predicted cost at dispatch using the online profiler.
- Reconcile the reservation with measured preparation service after
  completion.

Preparation service includes video access, metadata parsing, frame fetching,
video decoding, frame sampling and conversion, and image preprocessing and
encoding.

### Inference-Admission Stage

- Maintain a separate inference-service counter for every tenant.
- Select the least-served tenant with a model-ready request.
- Admit that tenant's oldest ready request into vLLM.
- Reserve its predicted inference cost at admission.
- Reconcile the reservation with observed inference service after completion.

Once a request is admitted, native vLLM controls prefill, continuous batching,
token decoding, and GPU memory. The proposed mechanism therefore provides
fairness at the inference-admission boundary; it is not token-level scheduling
inside vLLM.

### Separate Per-Tenant Service Counters

For tenant $i$, maintain:

- $S_i^P$: video-preparation service, measured from preparation work such as
  fetching, decoding, sampling, and preprocessing;
- $S_i^I$: inference service consumed after model admission.

The counters remain separate because CPU preparation time and GPU inference
service have no natural common unit. At either stage, the scheduler selects:

$$
i^* = \arg\min_{i:Q_i\neq\emptyset} S_i.
$$

In words, select the backlogged tenant that has received the least service at
the current stage.

### In-Flight Service Accounting

In-flight accounting counts a request's expected service as soon as it starts
instead of waiting for it to finish. It is necessary because the system can
run multiple requests concurrently. Without it, a running request is absent
from its tenant's service counter, so that tenant can continue to appear least
served and receive several concurrent workers or inference slots.

At dispatch, reserve predicted cost $\widehat{c}(r)$:

$$
S_i \leftarrow S_i + \widehat{c}(r).
$$

At completion, reconcile the reservation with measured cost $c(r)$:

$$
S_i \leftarrow S_i + c(r) - \widehat{c}(r).
$$

The correct statement is:

> Without in-flight accounting, a tenant can occupy multiple concurrent
> workers before its running work is reflected in the fairness counters.

## Motivation Under Contention

> If every individual stage is accelerated, can queuing under contention still
> dominate end-to-end latency?

Yes. Even if every stage becomes faster, requests will still queue when the
arrival rate exceeds the capacity of video preparation or inference. Under
these conditions, end-to-end latency depends not only on stage efficiency but
also on which tenant receives service next.

An aggressive tenant can fill the queues and delay other tenants even when the
individual preparation and inference operations are highly optimized.
Therefore, stage-level acceleration does not eliminate the need for
inter-tenant scheduling and performance isolation.

## Comparison Policies

### Direct Baselines and Ablations

1. **Native vLLM/FCFS:** Handles requests approximately in arrival order and
   performs no tenant-aware service accounting.
2. **Tenant round-robin:** Gives each backlogged tenant a request turn but
   ignores heterogeneous request costs. Equal request counts do not imply
   equal service.
3. **VTC:** Accounts for each tenant's consumed inference service using
   weighted input and output tokens. It improves fairness inside LLM inference
   but does not account for video fetching, decoding, or preprocessing.
4. **Preparation-only max--min:** Applies tenant fairness during video
   preparation while leaving inference admission native.
5. **Inference-only max--min:** Leaves preparation in FCFS order while fairly
   controlling inference admission.
6. **Full cross-stage max--min:** Applies separate max--min accounting at both
   preparation and inference admission. This is the proposed system, not a
   baseline.

### Secondary Efficiency Comparisons

- **SJF:** Selects the request with the smallest predicted cost. It can improve
  mean latency but may starve expensive requests and provides no tenant
  fairness guarantee.
- **Autellix/PLAS:** Tracks attained LLM service for an entire agent program and
  favors programs with less accumulated service to improve program completion
  time.
- **Kairos:** Uses workflow dependencies and priorities to reduce multi-agent
  workflow latency.

Autellix and Kairos are not direct tenant-fairness baselines because they
optimize program or workflow completion rather than max--min tenant isolation.
Although Autellix and the proposed system both use attained-service
accounting, they use it at different abstraction levels and for different
objectives:

- Autellix asks: *Which agent program should run next to improve program
  completion time?*
- The proposed system asks: *Which backlogged tenant has received the least
  service at this stage?*

## Project-Meeting Guidance

The evaluation must answer three questions.

### Improvement

How much does cross-stage max--min improve fairness?

Measure:

- preparation-service disparity;
- inference-service disparity;
- victim slowdown;
- worst-tenant P95 TTFT;
- worst-tenant end-to-end latency.

Service disparity is the difference between the most- and least-served
continuously backlogged tenants at a stage; lower is fairer. Victim slowdown
is defined relative to the victim running without an aggressor:

$$
\text{victim slowdown} =
\frac{\text{victim P95 latency with aggressor}}
     {\text{victim P95 latency when run alone}}.
$$

Cross-stage max--min should reduce victim slowdown relative to FCFS and
single-stage fairness because it constrains the aggressor at both preparation
and inference admission.

### Cost

What does the fairness improvement cost?

Measure:

- aggregate throughput;
- overall mean and P95 latency;
- CPU and GPU utilization;
- scheduler and profiler overhead;
- cost-prediction error.

### Workloads

Under which conditions is the system useful?

Evaluate:

- low load versus overload;
- homogeneous versus heterogeneous request costs;
- balanced versus aggressive tenant arrival rates;
- preparation-heavy, inference-heavy, and mixed contention;
- different levels of preparation and inference concurrency.

The expected benefit is largest under overload with aggressive tenants and
heterogeneous request costs. Limited improvement is expected under low load,
homogeneous requests, or a single active tenant.

## Evaluation Plan

Present the evaluation as the following ordered sequence of figure mockups.

### Figure 1: Cross-Tenant Interference

- **X-axis:** aggressor request rate, normalized by system capacity;
- **Y-axis:** victim slowdown;
- **Policies:** FCFS, inference-only fairness, and cross-stage max--min;
- **Expected shape:** FCFS slowdown rises sharply under overload,
  inference-only provides partial protection, and cross-stage remains lowest;
- **Outcome:** establish that an aggressive tenant creates cross-stage victim
  interference.

### Figure 2: Headline Fairness Comparison

- **X-axis:** FCFS, round-robin, VTC, preparation-only, inference-only, and
  cross-stage;
- **Y-axis:** service disparity, victim slowdown, or worst-tenant P95 TTFT;
- **Expected shape:** cross-stage has the lowest bar;
- **Outcome:** show the overall tenant-isolation improvement.

### Figure 3: Why Both Stages Matter

- **X-axis:** preparation-heavy, inference-heavy, and mixed workloads;
- **Y-axis:** victim slowdown;
- **Policies:** preparation-only, inference-only, and cross-stage;
- **Expected shape:** each single-stage policy succeeds only when its stage is
  the bottleneck, while cross-stage remains effective across all three;
- **Outcome:** show that controlling only one stage is insufficient when the
  bottleneck moves.

### Figure 4: Max--Min Allocation Over Time

- **X-axis:** elapsed time;
- **Y-axis:** cumulative service received;
- **Lines:** one line per tenant, with separate preparation and inference
  panels;
- **Expected shape:** tenant curves diverge under FCFS and remain close under
  cross-stage max--min;
- **Outcome:** demonstrate that the mechanism actually approximates max--min
  allocation.

### Figure 5: Cost of Fairness

- **X-axis:** scheduling policy;
- **Y-axis:** normalized throughput, overall P95 latency, and utilization in
  separate panels;
- **Expected shape:** cross-stage remains close to native vLLM while providing
  substantially better isolation;
- **Outcome:** quantify the efficiency cost of fairness.

### Figure 6: In-Flight Accounting Ablation

- **X-axis:** preparation workers or inference concurrency;
- **Y-axis:** service disparity;
- **Policies:** completion-only accounting, fixed in-flight reservation, and
  profiled reservation with reconciliation;
- **Expected shape:** completion-only disparity grows with concurrency, while
  profiled in-flight accounting remains low;
- **Outcome:** establish why predicted dispatch-time accounting is necessary.

### Figure 7: Workload Sensitivity

- **X-axis:** request-cost heterogeneity;
- **Y-axis:** offered load;
- **Color:** reduction in victim slowdown or service disparity over FCFS;
- **Expected shape:** the largest gains occur at high load and high cost
  heterogeneity;
- **Outcome:** identify where the system is and is not beneficial.

Every final figure should specify:

- the question it answers;
- the x-axis and y-axis;
- the compared policies;
- the expected shape of the result;
- the intended conclusion.

The overall evaluation should establish:

> How much fairness improves, what efficiency cost it incurs, and under which
> workloads cross-stage scheduling is or is not beneficial.

## Terminology

"Tenant" and "client" refer to the same entity. Use **tenant** consistently
throughout the paper, except when describing prior work that explicitly uses
"client."
