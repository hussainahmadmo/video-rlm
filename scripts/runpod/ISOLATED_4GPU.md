# Isolated four-GPU fairness sweep

On a Pod with four GPUs and at least 64 vCPUs, generate and inspect a CPU plan
before starting anything:

```bash
ACTION=plan scripts/runpod/run_isolated_4gpu_fairness.sh
cat logs/isolated_4gpu/topology.tsv
```

The planner keeps SMT siblings together, assigns disjoint physical cores, and
prefers CPU cores on each GPU's NUMA node. Each experiment lane receives eight
logical CPUs for preparation, eight for its vLLM server, and one GPU.

Start the four isolated servers:

```bash
ACTION=start scripts/runpod/run_isolated_4gpu_fairness.sh
ACTION=status scripts/runpod/run_isolated_4gpu_fairness.sh
```

Create a fresh L40S token-service profile before the measured run. Then launch
the suite, providing the local trace template and profile explicitly:

```bash
ACTION=run \
SOURCE_TRACE=/workspace/data/source-trace.jsonl \
ENGINE_TOKEN_PROFILE=/workspace/profiles/l40s-token-service.json \
PATTERNS="fixed poisson bursty" \
LOADS="0.1 0.2 0.3 0.5" \
SEEDS="1 2 3" \
scripts/runpod/run_isolated_4gpu_fairness.sh
```

Experiment cells are distributed across the four lanes, while all policies
within one pattern/load/seed cell execute sequentially on the same lane. This
avoids comparing policies placed permanently on different GPUs.

Use `ACTION=stop` to stop only the vLLM processes recorded by this launcher.
