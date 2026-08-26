#!/usr/bin/env bash
set -euo pipefail

# Repeats the controlled heavy-background/light-urgent stress experiment.
# A condition is assigned wholly to one shard so every policy sees the same
# trace and vLLM replica. Existing summary.json files are resumed safely.

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${PRIORITY_DATASET:?Set PRIORITY_DATASET}"

PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SEEDS=${SEEDS:-"1 2 3 4 5"}
BACKLOGS=${BACKLOGS:-"8 16 32 64 128"}
POLICIES=${POLICIES:-"fcfs priority priority_reserved"}
BACKGROUND_FRAMES=${BACKGROUND_FRAMES:-128}
URGENT_FRAMES=${URGENT_FRAMES:-8}
URGENT_COUNT=${URGENT_COUNT:-16}
URGENT_ARRIVAL_S=${URGENT_ARRIVAL_S:-1}
EXCLUDE_QID=${EXCLUDE_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}

ROOT=$VIDEO_RLM_ROOT
PY=$VLLM_PYTHON
GEN="$ROOT/conductor/experiments/scripts/run/generate_priority_workloads.py"
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${BACKLOG_STRESS_OUT:-"$ROOT/large_sweeps/repeated_backlog_stress_$STAMP"}
LOGROOT=${BACKLOG_STRESS_LOGROOT:-"$ROOT/logs/$(basename "$OUT")"}

mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_repeated_backlog_stress.path"

if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "invalid shard $SHARD_INDEX/$NUM_SHARDS" >&2
  exit 2
fi
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

condition_index=0
for backlog in $BACKLOGS; do
  for seed in $SEEDS; do
    if (( condition_index % NUM_SHARDS != SHARD_INDEX )); then
      condition_index=$((condition_index + 1))
      continue
    fi
    condition_index=$((condition_index + 1))
    name="backlog${backlog}-seed${seed}"
    trace="$OUT/traces/$name.jsonl"
    if [[ ! -f "$trace" ]]; then
      "$PY" "$GEN" \
        --dataset "$PRIORITY_DATASET" --output "$trace" \
        --pattern burst --background-count "$backlog" \
        --urgent-count "$URGENT_COUNT" \
        --urgent-arrival-s "$URGENT_ARRIVAL_S" \
        --background-frames "$BACKGROUND_FRAMES" \
        --urgent-frames "$URGENT_FRAMES" \
        --exclude-qid "$EXCLUDE_QID" --seed "$seed"
    fi

    for policy in $POLICIES; do
      output="$OUT/$name/$policy"
      log="$LOGROOT/$name-$policy.log"
      if [[ -f "$output/summary.json" ]]; then
        echo "SKIP completed $name $policy"
        continue
      fi
      if [[ -d "$output" ]]; then
        mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
      fi
      echo "START shard=$SHARD_INDEX port=$PORT $name $policy $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
        --arrival-trace "$trace" --output "$output" --port "$PORT" \
        --prep-policy "$policy" --background-prep-limit 3 \
        --decode-backend seek_cpu --prep-workers 4 --vlm-concurrency 4 \
        --prepared-queue-depth 32 --decode-timeout-s 600 \
        --request-timeout-s 1800 >"$log" 2>&1
      echo "DONE shard=$SHARD_INDEX port=$PORT $name $policy $(date -u +%FT%TZ)"
    done
  done
done

echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
