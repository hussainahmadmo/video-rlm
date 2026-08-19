# Native vLLM Held-Out Runs

This directory contains native vLLM video-input runs for the 150-query held-out slice from:

`conductor/experiments/diverse_eval/agent_friendly_egoschema_nextqa_intentqa.jsonl`

## Clean Learned-Adaptive Run

File:

`results_learned_adaptive_150_clean.jsonl`

Summary:

- examples: 150
- errors: 0
- correct: 109 / 150
- accuracy: 72.67%
- wall time: 89.07s
- throughput: 1.68 q/s

Per dataset:

| Dataset | Correct | Accuracy | Avg latency |
|---|---:|---:|---:|
| EgoSchema | 31 / 50 | 62.0% | 3.05s |
| NExT-QA | 40 / 50 | 80.0% | 1.96s |
| IntentQA | 38 / 50 | 76.0% | 2.08s |

Important interpretation: in this native runner, the learned schedule uses `vlm_budget` as the native uniform frame count. The schedule selected `vlm_budget=32` for all 150 held-out examples, so this run is effectively native uniform 32-frame video input. The schedule did vary `chosen_config` between `scan0.03125_k8_budget32` and `budget32`, but native vLLM video input does not use the retrieval-specific knobs (`probe_fps`, `probe_topk`, `window_len_s`).

Schedule distribution:

| Field | Distribution |
|---|---|
| `vlm_budget` | 32 for 150 / 150 |
| `chosen_config` | `scan0.03125_k8_budget32`: 131, `budget32`: 19 |
| `probe_fps` | 0.03125: 131, 0.015625: 19 |
| `probe_topk` | 8 for 150 / 150 |
| `window_len_s` | 8 for 150 / 150 |

## 100-Example Uniform Baseline

File:

`results_uniform_2_8_32_100.jsonl`

This is the 100-example comparison run over EgoSchema + NExT-QA only.

| Config | Correct | Accuracy | Avg latency |
|---|---:|---:|---:|
| `native_uniform_2` | 71 / 100 | 71.0% | 2.56s |
| `native_uniform_8` | 70 / 100 | 70.0% | 2.82s |
| `native_uniform_32` | 69 / 100 | 69.0% | 2.98s |

