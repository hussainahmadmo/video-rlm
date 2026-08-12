# Conductor

Conductor is a research system for adaptive video question answering. The core idea is to use query semantics, video duration, resource state, and retrieval constraints to choose an execution policy instead of relying on one fixed sampling or retrieval configuration for every video.

## System Design

The current pipeline is:

```text
Question
  |
  v
Semantic Profiler
  |
  v
Temporal / context / precision requirement
  |
  v
Policy Compiler
  |
  v
Requested Policy
  |
  v
Resource Adapter
  |
  v
Final Execution Policy
  |
  v
CLIP + Retrieval + VLM
```

The semantic profiler predicts what kind of evidence a question needs. The policy compiler translates that semantic requirement, plus video metadata such as duration, into concrete execution knobs such as scan rate, retrieval budget, and VLM budget.

## Temporal Requirements

The temporal requirement describes where the evidence is expected to appear in the video:

- `local`: evidence is expected in a small portion of the video.
- `medium`: evidence needs broader coverage, but not exhaustive coverage.
- `global`: evidence may be spread across the full timeline, so the system should cover the video sparsely while staying within budget.

The profiler should not directly output a scan rate. It should output a semantic requirement. The compiler is responsible for translating that requirement into a concrete retrieval policy.

For the same semantic question, changing only the video duration can change the scan rate because longer videos increase the retrieval search space and decode cost.

Example:

```text
Question: What is the color of the car?

Short video: local policy may use higher scan_fps.
Medium video: medium policy may use moderate scan_fps.
Long video: global policy may use lower scan_fps to stay within budget.
```

Video duration does not determine the semantic requirement by itself. A longer video is not automatically more semantically complex, but it does create a larger retrieval space and higher decode cost.

## Baselines

Common baseline systems include:

- Fixed sampling.
- Uniform visual token budget.
- Sliding-window retrieval.
- Window scoring followed by top-k window selection.

A fixed retrieval policy can miss relevant evidence or waste latency because it applies the same retrieval behavior regardless of video duration, question type, or resource state.

## Retrieval Problem

Current CLIP retrieval roughly follows:

```text
Video -> Candidate Windows -> Sparse Frame Decode -> CLIP Score -> Rank Windows
```

Flat ranking scores the whole candidate set once:

```text
video
  -> fixed windows
  -> score every window with CLIP
  -> sort windows
  -> take top-k
```

This evaluates every candidate window, but cost grows linearly with video duration.

Examples:

- 35 second video: about 7 windows.
- 30 minute video: about 360 windows.
- 1 hour video: about 720 windows.

The key issue is that flat ranking scales with the number of candidate windows.

## Hierarchical Retrieval

Hierarchical retrieval turns retrieval into a two-stage search:

```text
Stage 1: coarse search
  -> divide the video into broad regions
  -> score representative windows

Stage 2: fine search
  -> select the best coarse regions
  -> split them into smaller child windows
  -> score those child windows
```

Example:

- Score 16 coarse regions.
- Select the best 4 coarse regions.
- Split each selected region into 4 child windows.
- Score 16 child windows.
- Total work: 16 coarse scores + 16 child scores = 32 scored windows.

If flat ranking generated 300 windows, hierarchical retrieval can reduce scoring work from 300 windows to about 32 windows. This only reduces decode latency if the implementation decodes only those selected windows.

## Decode Latency Findings

Dense decoding and sparse retrieval both contribute significantly to latency. The most important finding so far is that sparse retrieval latency is dominated by random access into compressed video, not by CLIP inference alone.

Increasing scan FPS increases candidate frames approximately linearly:

- 0.03125 fps -> 68 candidates.
- 0.0625 fps -> 136 candidates.
- 0.125 fps -> 271 candidates.

Decode latency also grows approximately linearly:

- 68 candidates -> about 7.5 seconds decode.
- 136 candidates -> about 14.5 seconds decode.
- 271 candidates -> about 30.4 seconds decode.

For long-video retrieval, latency is primarily a compressed-video access problem.

## CPU vs GPU Sparse Decode

GPU sparse retrieval was consistently slower than CPU sparse retrieval in early tests.

Example:

- CPU sparse decode: about 7.5 seconds.
- GPU sparse decode: about 30 seconds.

The bottleneck is not raw frame decoding throughput. Sparse retrieval requires many random accesses throughout the compressed stream. For each requested frame, the decoder must:

1. Seek to the nearest preceding keyframe.
2. Reconstruct dependent frames within the GOP.
3. Return the requested frame.

Moving decode to the GPU does not remove seek overhead or GOP reconstruction cost.

## GOP Reconstruction

GOP reconstruction happens inside the video decoder, such as FFmpeg or Decord.

When requesting a frame, the decoder cannot directly decode an arbitrary P-frame or B-frame. It must locate the nearest preceding I-frame, decode dependent intermediate frames, and reconstruct the target frame.

This is true for both CPU and GPU decode.

## Sparse vs Contiguous Decode

One experiment compared sparse retrieval of selected frames against decoding a full contiguous range covering those frames.

Results:

- Sparse retrieval, 8 frames: about 3.4 seconds.
- Merged contiguous range, about 993 frames: about 6.6 seconds.
- Merged and sampled, about 42 returned frames: about 8.5 seconds.

For the tested windows, contiguous decoding was slower because it decoded too much unnecessary video. The selected retrieval windows were far enough apart that contiguous decode introduced more extra work than it saved.

## Parallel Decode

Sparse retrieval contains parallelizable work. Multiple processes with independent `VideoReader` instances showed moderate speedups:

- 1 worker -> 7.84 seconds.
- 2 workers -> 5.86 seconds.
- 4 workers -> 5.04 seconds.
- 8 workers -> 4.83 seconds.
- 16 workers -> 6.67 seconds.
- 32 workers -> 7.57 seconds.

Best performance was around 4 to 8 workers, with about 1.6x speedup. Performance saturated after that, likely due to decoder contention, memory bandwidth limits, FFmpeg overhead, or synchronization costs.

## Storage Result

Copying videos from NFS storage to local disk did not significantly change sparse retrieval latency. Storage/network access is not the primary bottleneck. The dominant cost appears to be decoder seeking and GOP reconstruction.

## Seek Overhead

Cold seek measurements showed that decoding a small number of nearby frames has a large fixed cost:

- 1 frame -> about 0.63 seconds.
- 16 frames -> about 0.65 seconds.
- 32 frames -> about 0.88 seconds.
- 64 frames -> about 1.23 seconds.
- 128 frames -> about 1.35 seconds.
- 256 frames -> about 2.31 seconds.

Decoding 16 frames can cost approximately the same as decoding 1 frame. This suggests the fixed seek and reconstruction cost dominates for sparse access.

## Resolution Experiment

The decoder was modified to output 224x224 frames directly instead of decoding full 1080p frames and resizing later.

This reduced memory pressure and avoided large tensor allocations, but it produced little to no improvement in retrieval latency. The reason is that retrieval latency was dominated by random seeks and GOP reconstruction rather than pixel processing.

For a roughly 36 minute video:

- `scan_fps = 0.125`
- about 271 candidate frames examined
- CLIP encode: about 1.2 seconds
- sparse decode: about 26 seconds

Decode dominates overall retrieval latency.

## Optimization Directions

Promising directions:

- Reduce the number of candidate windows examined.
- Use hierarchical or coarse-to-fine retrieval.
- Parallelize candidate evaluation.
- Overlap decode and CLIP scoring.
- Reuse decoder state across nearby accesses.
- Explore GOP-aware retrieval.
- Benchmark sequential decoding that streams through the video once and keeps sampled frames.

Current evidence suggests that reducing random accesses may provide larger gains than accelerating decode kernels.

## Experiment Goals

Current experiment goals:

- Test more datasets.
- Compare VIMIO against fixed configs.
- Evaluate Egoschema, LVBench, and VRBench.
- Build latency graphs across short, medium, and long videos.
- Keep video resolution consistent across datasets.
- Fix timing instrumentation so stages can be distinguished.
- Plot per-category gains for each dataset.
- Evaluate how retrieval and evidence acquisition knobs affect accuracy, latency, and the accuracy-latency Pareto frontier.

## Knob Sensitivity

Scan FPS controls how densely the video is searched during retrieval.

Higher scan FPS:

- Examines more candidate moments.
- Increases retrieval latency.
- May improve retrieval quality.
- May improve answer accuracy.

Lower scan FPS:

- Reduces retrieval latency.
- Risks missing short-duration events.

Expected behavior:

- Very sparse retrieval, such as `scan_fps=0.0039`, has the lowest retrieval cost but may miss short events.
- Moderate retrieval, such as `scan_fps=0.0156` to `scan_fps=0.0625`, may be a good accuracy-latency tradeoff.
- Dense retrieval, such as `scan_fps=2.0`, may improve temporal localization but substantially increases retrieval latency.

## Oracle Configuration

Earlier dataset runs showed low diversity in oracle configurations. Each dataset often had a single configuration that was best under the current definition of best accuracy and lowest latency.

This needs more exploration because the main claim depends on showing that no single configuration works best across all situations.

## Current Direction

The current direction is shifting toward multiple low-fidelity drafters and one verifier.

The motivation is that low-fidelity video may be sufficient for many questions, while high-fidelity video is expensive. Not all questions require a high-fidelity KV cache. The system can use lower-fidelity multimodal tokens or KV caches for draft generation, then use a verifier to check the output.

Important cases:

- Same video, multiple questions.
- Different videos, multiple questions.
- Multiple low-fidelity KV caches with one high-fidelity verifier.

Core assumption:

- The full video KV cache can become larger than available HBM memory.

In that setting, keeping smaller KV caches on GPU may be more practical than keeping the full high-fidelity video KV cache.

## Harness Motivation

A harness is useful because the profiler and policy choices should not be recomputed from scratch every time. The harness can record what worked before and reuse plans or policy decisions.

Retrieval-only systems can miss evidence, especially on counting questions or questions where the correct answer depends on selecting the right windows. A harness can help compare retrieval-only behavior against adaptive policies and verifier-based approaches.

## Open Questions

- How should semantic requirements be mapped to retrieval policies?
- When does dense retrieval improve quality enough to justify latency?
- Which datasets show meaningful oracle policy diversity?
- Can hierarchical retrieval preserve accuracy while reducing decode work?
- Can sequential decode beat sparse random access for long videos?
- How much concurrency can VIMIO support?
- Which questions are evidence-limited versus composition-limited?
- When do low-fidelity drafters fail, and how should the verifier recover?

## Core insight - if we need more accuracy only VideoAgent type of papers work but they dont talk about end to end latency for 
differet queries once those summaries are created.

- once the summaries are created during the runtime I can control the policies. for different type of queries(the summary is important because retreival can not show gains and more
retrieval will never solve the accuracy problem.)

- problem with the approach of we can use agents to 
first create the summa

- the retreiver can use these summaries to control 


- using summaries and then doing an agentic search on such videos gives us gains in accuracy but tool use to get the accuracy gains from these summaries
is expensive.


(we assume that video agents already do this and gains come from such systems), but there is another problem that 

- background 
- - captioning and use tools agent loop.
    in each iteration the captioning model generatees and an input prompt for
    a window.

- We identify that agentic visual systems have coupled stages with very different execution properties: video captioning is reusable, batchable, and GPU-parallel, while agentic tool use is query-specific, latency-sensitive, and often sequential.

- We present VIMIO, a disaggregated execution system for long-video agentic QA. VIMIO separates video-level caption-memory construction from query-level agentic reasoning/tool use, allowing each stage to be scheduled, scaled, cached, and optimized independent

- VIMIO batches and parallelizes caption generation across GPUs to reduce time-to-memory-ready, then reuses the resulting caption cache across many questions and retrieval sweeps. Query-level agentic workers consume this cached memory and invoke tools only when additional evidence is needed.