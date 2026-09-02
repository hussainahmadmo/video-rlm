#!/usr/bin/env bash
set -euo pipefail

# Compare cost-only, tenant-fair, and fair+slowdown scheduling under the same
# closed-loop overload trace. Shard seeds across independent vLLM replicas.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
DATASET=${PRIORITY_DATASET:-$ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl}
PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SEEDS=${SEEDS:-"1 2 3 4"}
POLICIES=${POLICIES:-"fcfs priority sjf max_min tenant_fair tenant_priority fair_slowdown"}
PREP_WORKERS=${PREP_WORKERS:-4}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
BACKGROUND_PER_TENANT=${BACKGROUND_PER_TENANT:-12}
URGENT_PER_TENANT=${URGENT_PER_TENANT:-4}
EXCLUDE_QID=${EXCLUDE_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}
UNIFORM_CLASS=${UNIFORM_CLASS:-0}

GEN=$ROOT/conductor/experiments/scripts/run/generate_multitenant_overload_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_multitenant_scheduling.py
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/multitenant_scheduling_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_multitenant_scheduling.path"

test -f "$DATASET"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

generator_extra=()
if [[ "$UNIFORM_CLASS" == 1 ]]; then
  generator_extra+=(--uniform-class)
elif [[ "$UNIFORM_CLASS" != 0 ]]; then
  echo "UNIFORM_CLASS must be 0 or 1" >&2
  exit 2
fi

index=0
for seed in $SEEDS; do
  if (( index % NUM_SHARDS != SHARD_INDEX )); then
    index=$((index + 1))
    continue
  fi
  index=$((index + 1))
  name="equal-tenants-mixed-cost-seed${seed}"
  trace="$OUT/traces/$name.jsonl"
  if [[ ! -f "$trace" ]]; then
    "$PY" "$GEN" --dataset "$DATASET" --output "$trace" \
      --background-per-tenant "$BACKGROUND_PER_TENANT" \
      --urgent-per-tenant "$URGENT_PER_TENANT" \
      --frame-budgets 8 32 128 --urgent-arrival-s 5 \
      "${generator_extra[@]}" \
      --exclude-qid "$EXCLUDE_QID" --seed "$seed"
  fi

  for policy in $POLICIES; do
    output="$OUT/$name/$policy"
    log="$LOGROOT/$name-$policy.log"
    [[ -f "$output/summary.json" ]] && { echo "SKIP $name $policy"; continue; }
    [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
    echo "START shard=$SHARD_INDEX seed=$seed policy=$policy $(date -u +%FT%TZ)"
    VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$output" --port "$PORT" \
      --prep-policy "$policy" --tenant-weight a=1 --tenant-weight b=1 \
      --tenant-weight c=1 --decode-backend seek_cpu \
      --prep-workers "$PREP_WORKERS" --vlm-concurrency "$VLM_CONCURRENCY" \
      --prepared-queue-depth 16 --decode-timeout-s 600 \
      --request-timeout-s 1800 >"$log" 2>&1
    echo "DONE shard=$SHARD_INDEX seed=$seed policy=$policy $(date -u +%FT%TZ)"
  done
done

"$PY" "$ANALYZER" --root "$OUT" --output "$OUT/multitenant_results" || true
echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
