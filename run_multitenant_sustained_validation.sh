#!/usr/bin/env bash
set -euo pipefail

# Validate priority, tenant isolation, and background progress under sustained
# arrivals. Whole seed/scenario pairs are sharded across independent replicas.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
DATASET=${PRIORITY_DATASET:-$ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl}
PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SEEDS=${SEEDS:-"1 2 3"}
POLICIES=${POLICIES:-"fcfs priority tenant_fair tenant_priority"}
PREP_WORKERS=${PREP_WORKERS:-4}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
BACKGROUND_AGING_S=${BACKGROUND_AGING_S:-30}
EXCLUDE_QID=${EXCLUDE_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}

GEN=$ROOT/conductor/experiments/scripts/run/generate_multitenant_overload_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_multitenant_scheduling.py
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/multitenant_sustained_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_multitenant_sustained.path"

test -f "$DATASET"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

index=0
for seed in $SEEDS; do
  for scenario in equal noisy_neighbor; do
    if (( index % NUM_SHARDS != SHARD_INDEX )); then
      index=$((index + 1))
      continue
    fi
    index=$((index + 1))
    name="${scenario}-continuous-seed${seed}"
    trace="$OUT/traces/$name.jsonl"
    generator_args=(
      --dataset "$DATASET" --output "$trace" --seed "$seed"
      --background-per-tenant 18 --urgent-per-tenant 6
      --background-arrival-interval-s 0.75
      --urgent-arrival-s 5 --urgent-arrival-interval-s 1.5
      --arrival-jitter-s 0.1 --frame-budgets 8 32 128
      --exclude-qid "$EXCLUDE_QID"
    )
    if [[ "$scenario" == noisy_neighbor ]]; then
      generator_args+=(
        --tenant-background-count a=36
        --tenant-background-count b=9
        --tenant-background-count c=9
      )
    fi
    [[ -f "$trace" ]] || "$PY" "$GEN" "${generator_args[@]}"

    for policy in $POLICIES; do
      output="$OUT/$name/$policy"
      log="$LOGROOT/$name-$policy.log"
      [[ -f "$output/summary.json" ]] && {
        echo "SKIP $name $policy"
        continue
      }
      [[ ! -d "$output" ]] || mv \
        "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
      echo "START shard=$SHARD_INDEX scenario=$scenario seed=$seed policy=$policy $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
        --arrival-trace "$trace" --output "$output" --port "$PORT" \
        --prep-policy "$policy" --tenant-weight a=1 --tenant-weight b=1 \
        --tenant-weight c=1 --decode-backend seek_cpu \
        --prep-workers "$PREP_WORKERS" \
        --vlm-concurrency "$VLM_CONCURRENCY" \
        --prepared-queue-depth 16 \
        --urgent-ttft-slo-s 60 --background-ttft-slo-s 300 \
        --background-aging-s "$BACKGROUND_AGING_S" \
        --decode-timeout-s 600 --request-timeout-s 1800 \
        >"$log" 2>&1
      echo "DONE shard=$SHARD_INDEX scenario=$scenario seed=$seed policy=$policy $(date -u +%FT%TZ)"
    done
  done
done

"$PY" "$ANALYZER" --root "$OUT" --output "$OUT/multitenant_sustained_results" \
  --urgent-e2e-slo-s 60 --background-e2e-slo-s 300 || true
echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
