# Running the fairness experiments on the four-L40S Pod

1. Install the environment and inspect the topology:

```bash
VLLM_VERSION=0.17.0 bash scripts/runpod/install_runpod.sh
ACTION=plan scripts/runpod/run_isolated_4gpu_fairness.sh
cat logs/isolated_4gpu/topology.tsv
```

2. Start four isolated vLLM servers:

```bash
ACTION=start scripts/runpod/run_isolated_4gpu_fairness.sh
ACTION=status scripts/runpod/run_isolated_4gpu_fairness.sh
```

3. Profile L40S inference service on port 9000 and fit the profile:

```bash
mkdir -p /workspace/profiles/l40s
python conductor/experiments/scripts/run/profile_vllm_token_service.py \
  --port 9000 \
  --output /workspace/profiles/l40s/raw.jsonl \
  --concurrency "1 2 4" \
  --visual-frames "0 8 32 128" \
  --repeats 3

python conductor/experiments/scripts/analyze/analyze_vllm_token_service_profile.py \
  --input /workspace/profiles/l40s/raw.jsonl \
  --output /workspace/profiles/l40s/token_service_cost
```

The scheduler profile is `/workspace/profiles/l40s/token_service_cost.json`.

4. Run a short arrival-sweep smoke test:

```bash
ACTION=run \
SOURCE_TRACE=/workspace/data/source-trace.jsonl \
ENGINE_TOKEN_PROFILE=/workspace/profiles/l40s/token_service_cost.json \
PATTERNS=poisson LOADS=0.2 SEEDS=1 \
POLICIES="fcfs max_min age_aware_max_min" DURATION_S=20 \
scripts/runpod/run_isolated_4gpu_fairness.sh
```

5. Run the complete arrival-rate sweep:

```bash
ACTION=run \
SOURCE_TRACE=/workspace/data/source-trace.jsonl \
ENGINE_TOKEN_PROFILE=/workspace/profiles/l40s/token_service_cost.json \
PATTERNS="fixed poisson bursty" LOADS="0.1 0.2 0.3 0.5" SEEDS="1 2 3" \
scripts/runpod/run_isolated_4gpu_fairness.sh
```

6. Run balanced bursts and the age-threshold sweep:

```bash
SOURCE_TRACE=/workspace/data/source-trace.jsonl \
ENGINE_TOKEN_PROFILE=/workspace/profiles/l40s/token_service_cost.json \
EXPERIMENTS="balanced threshold" SEEDS="1 2 3" \
scripts/runpod/run_isolated_4gpu_tail_experiments.sh
```

Use a source trace whose media paths point to videos on the Pod's local disk.
Do not time video decoding directly from a network volume.
