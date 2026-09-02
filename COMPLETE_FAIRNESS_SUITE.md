# Complete Multimodal Fairness Experiment Suite

The suite evaluates tenant fairness across video preparation and inference
admission. It runs matched policies sequentially against one existing vLLM
endpoint, skips completed summaries, preserves partial failed runs, and
generates aggregate CSV, Markdown, PNG, and PDF outputs.

## Requirements

- A healthy vLLM server on the selected port.
- The server must support the request `priority` field.
- The `vllm-mm` Python environment and evaluation dataset must exist.

Check the server:

```bash
curl -fsS -H 'Authorization: Bearer EMPTY' \
  http://127.0.0.1:9000/v1/models
```

## Dry Run

Print every generated command without running model requests:

```bash
cd /dataheart/hussainahmad/video-rlm

DRY_RUN=1 \
SEEDS="1" \
OUT="$PWD/large_sweeps/complete_fairness_dry" \
./run_complete_multimodal_fairness_suite.sh
```

## Recommended Phase-by-Phase Execution

Use the same `OUT` for every command. Completed runs are skipped when a phase
is resumed.

```bash
cd /dataheart/hussainahmad/video-rlm

export OUT="$PWD/large_sweeps/complete_multimodal_fairness_final"
export PORT=9000
export SEEDS="1 2 3"
```

Run the headline comparison and measured solo baselines first:

```bash
PHASES="headline solo" ./run_complete_multimodal_fairness_suite.sh
```

Run the stage-ablation matrix:

```bash
PHASES="stage" ./run_complete_multimodal_fairness_suite.sh
```

Run direct service-fairness and request-cost heterogeneity comparisons:

```bash
PHASES="fairness heterogeneity" ./run_complete_multimodal_fairness_suite.sh
```

Run offered-load sensitivity:

```bash
PHASES="load" ./run_complete_multimodal_fairness_suite.sh
```

Run the in-flight accounting ablation:

```bash
PHASES="inflight" ./run_complete_multimodal_fairness_suite.sh
```

Run every phase in one command:

```bash
PHASES="headline solo stage fairness load heterogeneity inflight" \
./run_complete_multimodal_fairness_suite.sh
```

The full default matrix contains 62 runs per seed, or 186 runs for three
seeds. Run it phase-by-phase unless the endpoint can remain available for an
extended period.

## Phases

| Phase | Purpose | Primary comparisons |
| --- | --- | --- |
| `headline` | Aggressive-tenant isolation | FCFS, preparation-only, inference-only, cross-stage |
| `solo` | Measured victim-slowdown denominators | Victims B and C alone |
| `stage` | Moving-bottleneck ablation | Preparation-heavy, inference-heavy, mixed |
| `fairness` | Direct service allocation | FCFS, tenant round-robin, max--min |
| `load` | Offered-load sensitivity | Five load scales and four policies |
| `heterogeneity` | Request-cost sensitivity | Homogeneous, moderate, high heterogeneity |
| `inflight` | Concurrency/accounting ablation | Completion-only, fixed reservation, profiled/reconciled |

The inference-only policy controls admission into vLLM. It is not a token-level
implementation of VTC inside continuous batching.

## Resume and Monitor

Rerun the same command with the same `OUT`. Any directory containing a
`summary.json` is skipped.

Count completed runs:

```bash
find "$OUT" -name summary.json | wc -l
```

Follow the latest logs:

```bash
find "logs/$(basename "$OUT")" -name '*.log' -printf '%T@ %p\n' \
  | sort -n | tail
```

Then run `tail -f` on the desired log path.

## Outputs

The suite writes:

```text
$OUT/suite_results_runs.csv      per-run measurements
$OUT/suite_results.csv           averages across seeds
$OUT/suite_results.md            concise aggregate report
$OUT/figures/                    paper-ready PNG and PDF figures
$OUT/<phase>/<case>/<variant>/   raw events, results, and summary
```

Regenerate aggregate results after interrupted or manually added runs:

```bash
/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python \
  conductor/experiments/scripts/analyze/analyze_complete_fairness_suite.py \
  --root "$OUT" --output "$OUT/suite_results"
```
