# Clean held-out contention evaluation

This evaluation removes every `(dataset, qid)` used to train the recovered
`k=15` stronger selector. The resulting workload contains 138 questions with
zero training overlap.

Both policies used the same Qwen2.5-VL-7B endpoint on port 9000 with
concurrency 32, temperature 0, and a 32-token output limit.

| Policy | Correct | Accuracy | Wall time | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Fixed budget 32 | 78/138 | 56.52% | 89.64 s | 1.5395 QPS |
| Learned adaptive | 80/138 | 57.97% | 31.72 s | 4.3506 QPS |

The learned policy provides 2.826x throughput (182.6% higher) and is 1.45
accuracy points higher on this run. Paired outcomes were: 70 both correct, 50
both wrong, 10 adaptive-only correct, and 8 budget-32-only correct.

The split is described in `split_manifest.json`. Exact model inputs and outputs
are in `adaptive_jobs.jsonl`, `budget32_jobs.jsonl`,
`adaptive_c32_results.jsonl`, and `budget32_c32_results.jsonl`.
