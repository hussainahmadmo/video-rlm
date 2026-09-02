# Multimodal Fairness Experiments

The launcher runs every scheduling policy sequentially against the same trace.
This prevents policies from competing with one another for the vLLM GPU.
The launcher itself is detached with `nohup`, so it continues after the shell
closes.

## Experiments

| Phase | Question |
|---|---|
| `headline` | Does cross-stage fairness reduce victim latency under an aggressive heterogeneous tenant? |
| `solo` | What is each victim tenant's no-contention latency for computing slowdown? |
| `stage` | Which policy works when preparation, inference, or both stages are bottlenecks? |
| `fairness` | Does service-aware scheduling outperform FCFS and request-count round-robin? |
| `constant_rate` | Do overloaded tenants receive similar preparation and inference service despite a 2:1 request-rate difference? |
| `stochastic` | Does fairness hold with Poisson arrivals and heterogeneous video costs? |
| `on_off` | Does the scheduler handle intermittent tenants without accumulating idle credit? |
| `work_conservation` | Can an overloaded tenant use capacity left unused by an underloaded tenant? |
| `isolation` | Is a low-rate tenant protected from an aggressive tenant? |
| `load` | How do fairness and latency change as offered load crosses capacity? |
| `heterogeneity` | How does the benefit change as video-frame costs become more heterogeneous? |
| `inflight` | Is reservation/reconciliation necessary when several requests execute concurrently? |

Every phase compares FCFS, tenant round-robin, preparation-only max-min,
inference-only tenant fairness, and full cross-stage max-min, except `inflight`,
which compares accounting variants of the proposed scheduler.

## Commands

First confirm that vLLM is available on port 9000:

```bash
curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:9000/v1/models
```

Run one experiment in the background:

```bash
cd /dataheart/hussainahmad/video-rlm
./launch_fairness_experiments.sh constant_rate
```

Run several selected experiments:

```bash
./launch_fairness_experiments.sh on_off work_conservation isolation
```

Run the complete set:

```bash
./launch_fairness_experiments.sh
```

Check progress:

```bash
./status_fairness_experiments.sh
```

For a quick one-seed validation before the full three-seed run:

```bash
SEEDS=1 OUT="$PWD/large_sweeps/fairness_experiments_smoke" \
STATE_DIR="$PWD/logs/fairness_experiments_smoke" \
./launch_fairness_experiments.sh constant_rate
```

The suite is resumable: rerunning with the same `OUT` skips runs that already
contain `summary.json`.
