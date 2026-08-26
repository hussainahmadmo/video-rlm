#!/usr/bin/env bash
set -euo pipefail

# Sustained urgent arrivals expose background starvation and quantify whether
# reservation/aging policies bound it. This is deliberately separate from the
# latency headline so fairness costs cannot be hidden in an aggregate mean.

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${PRIORITY_DATASET:?Set PRIORITY_DATASET}"

PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SEEDS=${SEEDS:-"1 2 3 4 5"}
URGENT_PATTERNS=${URGENT_PATTERNS:-"staggered poisson"}
POLICIES=${POLICIES:-"fcfs priority priority_reserved slo_adaptive"}
BACKGROUND_COUNT=${BACKGROUND_COUNT:-32}
URGENT_COUNT=${URGENT_COUNT:-64}
BACKGROUND_FRAMES=${BACKGROUND_FRAMES:-64}
URGENT_FRAMES=${URGENT_FRAMES:-8}
BACKGROUND_AGING_S=${BACKGROUND_AGING_S:-120}
EXCLUDE_QID=${EXCLUDE_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}

ROOT=$VIDEO_RLM_ROOT
PY=$VLLM_PYTHON
GEN="$ROOT/conductor/experiments/scripts/run/generate_priority_workloads.py"
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${FAIRNESS_OUT:-"$ROOT/large_sweeps/fairness_aging_validation_$STAMP"}
LOGROOT=${FAIRNESS_LOGROOT:-"$ROOT/logs/$(basename "$OUT")"}
mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_fairness_aging_validation.path"

curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

index=0
for urgent_pattern in $URGENT_PATTERNS; do
  for seed in $SEEDS; do
    if (( index % NUM_SHARDS != SHARD_INDEX )); then
      index=$((index + 1))
      continue
    fi
    index=$((index + 1))
    name="continuous-${urgent_pattern}-seed${seed}"
    trace="$OUT/traces/$name.jsonl"
    generator_args=(
      --dataset "$PRIORITY_DATASET" --output "$trace"
      --pattern burst --background-count "$BACKGROUND_COUNT"
      --urgent-count "$URGENT_COUNT" --urgent-arrival-s 1
      --urgent-pattern "$urgent_pattern"
      --background-frames "$BACKGROUND_FRAMES"
      --urgent-frames "$URGENT_FRAMES"
      --exclude-qid "$EXCLUDE_QID" --seed "$seed"
    )
    if [[ "$urgent_pattern" == staggered ]]; then
      generator_args+=(--urgent-interval-s 0.5)
    else
      generator_args+=(--urgent-rate-qps 2.0)
    fi
    [[ -f "$trace" ]] || "$PY" "$GEN" "${generator_args[@]}"

    for policy in $POLICIES; do
      output="$OUT/$name/$policy"
      log="$LOGROOT/$name-$policy.log"
      [[ -f "$output/summary.json" ]] && { echo "SKIP $name $policy"; continue; }
      [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
      echo "START shard=$SHARD_INDEX $name $policy $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
        --arrival-trace "$trace" --output "$output" --port "$PORT" \
        --prep-policy "$policy" --background-prep-limit 3 \
        --urgent-prep-reserve 1 --background-aging-s "$BACKGROUND_AGING_S" \
        --decode-backend seek_cpu --prep-workers 4 --vlm-concurrency 4 \
        --prepared-queue-depth 32 --decode-timeout-s 600 \
        --request-timeout-s 1800 >"$log" 2>&1
      echo "DONE shard=$SHARD_INDEX $name $policy $(date -u +%FT%TZ)"
    done
  done
done

echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
