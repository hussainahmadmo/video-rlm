# Project: Cross-Stage Fair Scheduling for Multimodal LLM Serving

Session handoff updated 2026-09-17. Target: **EuroSys 2027** (user decision).
System: **Conductor**. This file records project context not already detailed in
[AGENTS.md](AGENTS.md); consult that file for placement algorithms, exact pilot
configuration/results, accounting limitations, and evidence requirements.

## Goal and paper framing

Inference-side fairness is insufficient because requests compete for upstream
video preparation before becoming visible to the inference scheduler. The paper
applies service-aware tenant scheduling across the multimodal serving pipeline,
especially at this preparation boundary. Established fair-queueing ideas are
building blocks, not a claimed invention.

Logical scheduling path: request arrival → pending preparation queue → video
fetch/decode/frame sampling/preprocessing → model-ready queue → vLLM inference
→ response. Preparation is non-preemptive. Waiting to become model-ready is
outside the inference scheduler's direct control.

Fair admission controls access to service; faster preparation and CPU/GPU
placement can shorten execution. Their combination is an evaluation objective,
not a guarantee of improvement for every request type.

## Manuscript and important files

- Local Overleaf working tree: `69f58e9744973ad74e35062f/`.
- Entry point: `69f58e9744973ad74e35062f/main.tex`.
- Current title declarations differ: “Fairness Starts Before the GPU:
  Service-based Scheduling for Multimodal LLM Serving” and “Fairness Starts
  Before the GPU: Cross-Stage Service Scheduling for Multimodal LLM Serving.”
  Resolve the duplicate rather than assuming a final title was selected.
- Included sections: `fair_preparation_motivation.tex`,
  `scheduler_algorithm.tex`, `preparation_placement.tex`, and `evaluation.tex`.
  Bibliography: `references.bib`; figures: `figures/` in the same working tree.
- `paper/main.tex` in the user's example is a placeholder, not an existing entry
  point. The repository also contains paper drafts and export bundles under
  `paper/`; do not mistake them for the identified Overleaf entry point.
- The entry point still mentions MLSys 2027 in a template comment. The user's
  current EuroSys target takes precedence. No remote synchronization, submission
  format verification, manuscript edits, or Overleaf push occurred in this session.
- Implementation and placement evidence paths are mapped in `AGENTS.md`.

# Current Research State

## Decisions already made

The core scheduler keeps separate cumulative service counters for preparation
and inference. At each stage, select the eligible tenant with the least accounted
service, then its oldest eligible request. Reserve estimated service at dispatch
and reconcile against the stage's measured/profiled cost at completion. Normalize
by tenant weight where configured. Do not sum stage counters with different units.

The manuscript's `scheduler_algorithm.tex` specifies deterministic tenant-ID and
request-sequence ties, arrival-time initialization to the active service frontier,
and separate capacity checks. Preparation needs a free worker and room in the
bounded handoff, which counts running preparation plus model-ready work. Inference
needs a model-ready request and a free submission slot. The engine controls
execution after submission.

Preparation cost means worker occupancy. The manuscript's inference cost uses a
profiled service function and API-reported prompt/completion token counts, not
concurrent request wall time interpreted as physical GPU execution time. Verify
the concrete experiment's accounting mode before describing its results.

Keep this independent per-stage algorithm distinct from the optional joint
preparation allocator's CPU/GPU dominant-service heuristic. In particular, the
manuscript specifies frontier initialization; the joint allocator's historical
accounts currently do not reset on idle reactivation. “Cross-stage” does not by
itself establish a joint global optimization or a formal fairness guarantee.

Writing style: concise, technical systems-paper prose; no marketing language.
Distinguish design motivations, measured evidence, and untested hypotheses.

## Important terminology

- **Preparation:** media fetching, decoding, sampling, preprocessing, and
  constructing model-ready input; distinguish queue wait from execution time.
- **Cross-stage fairness:** fairness across preparation and inference.
- **Preparation-only fairness:** fair service admission before inference only.
- **FCFS:** arrival-order admission; do not call it resource-service fairness.
- **TTFT:** time to first token. Distinguish end-to-end TTFT from time measured
  only after submission to inference. **E2E:** end-to-end completion latency.
- **Victim:** the tenant whose responsiveness is measured under interference.
- **Decoder lanes:** reserved preparation parallelism, not measured hardware
  engine utilization. Placement-specific terminology is defined in `AGENTS.md`.

## Experiments completed

This session inspected code, documentation, and saved experiment artifacts; it
did not launch experiments or establish live host job status.

The profiled-light versus original-joint comparison has a saved complete status
for both 60-request policies. Exact configuration, routing counts, latency,
throughput, and source directory are recorded once in `AGENTS.md` under
“Verified saved comparison.” The older documentation describing this run as
queued is stale.

Other inspected saved allocation directories include
`large_sweeps/joint_allocation_comparison_gpu1_20260916/`,
`large_sweeps/mixed_fourworker_allocation_gpu1_20260916/`, and
`large_sweeps/flashstyle_only_allocation_gpu1_20260916/`. Their saved statuses
reported completion. Do not combine them as repeated trials: configurations and
backends differ. The Poisson placement sweep was located, but completion of its
full planned sweep was not established in this session.

## Results

The user provided these current project findings for the core fairness study:

- Cross-stage max-min scheduling substantially reduces victim tail TTFT under
  contention.
- Preparation-side scheduling accounts for much of that improvement.
- There is no meaningful throughput penalty in the evaluated fairness workload.
- Text requests can suffer severe latency inflation when competing with
  unrestricted video preparation.

These are user-stated research findings; the supporting core experiment IDs,
effect sizes, uncertainty, and exact workloads were not revalidated here. Attach
those sources before turning these statements into quantitative manuscript claims.
The throughput statement must not be transferred to the separate profiled-light
placement pilot, whose measured tradeoff is documented in `AGENTS.md`.

## Open questions

- Which exact saved runs and plots support each core fairness finding above?
  Which ablations establish the preparation-side contribution?
- Should placement/decoder allocation be a central evaluated contribution or an
  extension? This session did not settle its final manuscript prominence.
- How robust are placement gains to videos, arrival patterns, seeds, hardware,
  cost-model error, and inference interference?
- How should idle-tenant reactivation and GPU-path host consumption be handled
  in joint allocation? What fairness property is supportable by the actual units?
- What is the final paper title, and is the local Overleaf tree synchronized with
  the remote project? Neither was resolved here.

## Things tried that did not work / conclusions rejected

- The profiled-light pilot does not establish an across-the-board win: the light
  request improvement coexists with an overall p95/throughput tradeoff.
- Its bundled changes do not identify whether classification, reservation, or
  bypass individually caused the improvement.
- Reject interpreting scheduler occupancy as measured GPU utilization, reservation
  accounting as complete GPU fairness, or the heuristic as proven DRF.
- Reject a measured victory over the authors' FlashCodec: the tested backend is
  an independent inspired implementation. Other placement claim limits are in
  `AGENTS.md` and apply to manuscript writing.
- No additional failed implementation was produced in this session.

## Reviewer/advisor feedback

No external reviewer or advisor feedback was supplied in this conversation.
The user specified the EuroSys target, core framing, terminology, evidence-based
writing preferences, and the requirement for a durable session handoff. Do not
attribute those instructions to a reviewer.

## Files changed this session

- Created root `AGENTS.md`, then added project/Overleaf orientation and a link to
  this handoff; it contains the detailed placement record.
- Created root `paper_context.md` (no existing file was found).
- No scheduler code, experiment output, or manuscript source was changed.
  The workspace already contained extensive unrelated modifications and untracked
  artifacts; preserve them rather than attributing them to this session.

## Current TODO

0. **One-frame versus 128-frame contention experiment (planned):** compare
   tuned, bounded FCFS, inference-only fairness, preparation-only fairness, and
   Conductor's cross-stage fairness. Use a heavy tenant submitting 128-frame
   video requests and a light tenant submitting one-frame requests. Tune worker
   concurrency and decoder threads on a separate trace first; then keep decoder,
   worker capacity, CPU affinity, handoff capacity, and inference configuration
   identical across policies. Include a heavy burst followed by multiple light
   requests to test preparation head-of-line blocking, and an interleaved arrival
   trace. Measure per-tenant preparation waiting and accounted service, median
   and p95 end-to-end TTFT/E2E, and aggregate throughput. Repeat with multiple
   seeds and distinct videos, rotating policy order. Show measured timelines
   alongside the conceptual CPU/GPU cases; distinguish waiting behind queued
   work from unavoidable waiting for non-preemptive work already running. This
   TODO records the evaluation plan, not a completed result or a new run request.
1. Map each core fairness claim to exact experiment artifacts and manuscript
   figures, checking units, TTFT boundaries, workload, and baseline configurations.
2. Resolve duplicate title declarations and the stale venue comment in the local
   Overleaf source; verify current EuroSys requirements when doing formatting work.
3. Align manuscript descriptions with the implemented core scheduler versus the
   optional joint allocator, including their different reactivation behavior.
4. Decide placement's paper scope, retaining the measured tradeoffs and existing
   evidence limits. Use the validation plan in `AGENTS.md` for stronger claims.
5. Isolate the profiled-light mechanisms in future ablations and evaluate tails,
   throughput, tenant service, and inference interference together.
6. Verify remote Overleaf sync state before any later synchronization action.
   This handoff request does not itself request a push or new experiment run.

## 2026-09-18 preparation diagnosis update

See `docs/preparation_bottleneck_audit_20260918.md` for the exact Figure 1 sources,
corrected interpretation, and a completed local PyAV diagnostic. Figure 1 now
separates waiting/execution in seconds. A40 has substantial downstream handoff
backpressure; H100/B300 keep preparation slots nearly full under their eight-CPU
limit. These are different causes of waiting before preparation.

The actual decoder scans almost the entire 313-second video; the local diagnostic
counts 9,360 decoded frames for 128 selected frames. Conversion/resize/JPEG adds
measurable work. New instrumentation records substage wall times and decoded-frame
counts; an explicit `--cpu-decoder-threads` option enables controlled tuning. Saved
Figure 1 runs used the prior source hash and are not results for the instrumented
code. The local diagnostic is not a publishable target-host speedup comparison.
Full-pipeline tuned/matched-capacity experiments remain necessary. No remote sync.

## 2026-09-19 L40S variable-frame placement update

A six-policy placement/fairness ablation completed on one L40S with 60
interleaved requests at 1, 16, and 128 frames. Full Conductor improves mean,
median, p95, and throughput over adaptive FCFS, but it does not beat the
calibrated fixed FCFS policy. Fixed FCFS records 30.19/28.35/77.17 seconds
mean/median/p95 and 0.581 requests/s; full Conductor records
39.58/42.62/78.79 seconds and 0.518 requests/s. Adaptive policies route some
128-frame work to CPU and some 16-frame work to GPU; concurrent GPU service
also differs materially from isolated calibration. Do not use this run to
claim an adaptive-placement win. Full configuration, routing diagnosis, and
claim boundaries are in `docs/variable_frame_placement_l40s_20260919.md`; the
result bundle is `large_sweeps/variable_frame_placement_l40s_20260919/`.

### Conservative-routing rerun

The same L40S comparison was rerun after adding dispatch-load context to the
existing width-specific GPU profilers and a calibrated-route fallback requiring
a two-second and 20% predicted-readiness advantage before switching backends.
The bundle is
`large_sweeps/variable_frame_placement_l40s_conservative_20260919/`.

All revised adaptive variants route 1/16-frame work to CPU and 128-frame work to
GPU. Placement-only FCFS records 28.72/19.52/78.35 seconds mean/median/p95 and
0.574 requests/s, versus 32.71/29.12/77.75 and 0.576 for fixed FCFS in this run.
This single fixed-order trial does not support a robust speedup claim.

Full Conductor records 50.99/49.78/104.23 seconds and 0.518 requests/s. It uses
one lane for eight of 15 large requests and has 40.25 seconds mean preparation
queueing, compared with 19.37 seconds for adaptive FCFS. Its overlap-window GPU
lane shares are more balanced, but the current fair width cap causes a major
latency cost. Do not use the revised run as a full-Conductor latency win. The
detailed comparison and profiler limits are in
`docs/variable_frame_placement_l40s_20260919.md`.

### Minimum-two-lane ablation

The focused L40S comparison in
`large_sweeps/minimum_lane_ablation_l40s_20260919/` measures the proposed width
fix. Restricting full Conductor to two/four lanes reduces mean E2E from 51.97 to
43.18 seconds (16.9%), median from 49.65 to 45.68 seconds (8.0%), and p95 from
100.72 to 81.59 seconds (19.0%); throughput rises from 0.489 to 0.610 requests/s.
The min-two policy uses two lanes for 14 of 15 large requests and four for one.

Placement-only FCFS still records 27.91 seconds mean and 18.75 seconds median.
The width restriction fixes a real inefficiency but does not explain most of the
fair policy's latency cost. Mean preparation-queue waiting remains 33.11 seconds
under min-two versus 17.88 under placement-only FCFS, pointing to fair selection
and admission order as the next mechanism to isolate. Do not claim that the
minimum-width change makes full Conductor latency-competitive from this pilot.
