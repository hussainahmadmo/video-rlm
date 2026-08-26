#!/usr/bin/env bash
set -euo pipefail

# Replays one representative arrival template with five independently sampled
# matched video/question assignments. The template must contain arrival_s and
# may also specify workload/class, priority, frame_count, and request_id.

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${PRIORITY_DATASET:?Set PRIORITY_DATASET}"
: "${REPLAY_TEMPLATE:?Set REPLAY_TEMPLATE to a JSONL arrival template}"

PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SEEDS=${SEEDS:-"1 2 3 4 5"}
POLICIES=${POLICIES:-"fcfs priority priority_reserved"}

ROOT=$VIDEO_RLM_ROOT
PY=$VLLM_PYTHON
GEN="$ROOT/conductor/experiments/scripts/run/generate_priority_workloads.py"
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${REPLAY_OUT:-"$ROOT/large_sweeps/replay_trace_validation_$STAMP"}
LOGROOT=${REPLAY_LOGROOT:-"$ROOT/logs/$(basename "$OUT")"}
mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_replay_trace_validation.path"

[[ -f "$REPLAY_TEMPLATE" ]] || { echo "missing template: $REPLAY_TEMPLATE" >&2; exit 1; }
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

index=0
for seed in $SEEDS; do
  if (( index % NUM_SHARDS != SHARD_INDEX )); then
    index=$((index + 1))
    continue
  fi
  index=$((index + 1))
  name="replay-seed${seed}"
  trace="$OUT/traces/$name.jsonl"
  [[ -f "$trace" ]] || "$PY" "$GEN" \
    --dataset "$PRIORITY_DATASET" --output "$trace" --pattern replay \
    --replay-template "$REPLAY_TEMPLATE" \
    --exclude-qid 20520eff-abdf-4d4f-94ad-cc751a8960d0 --seed "$seed"

  for policy in $POLICIES; do
    output="$OUT/$name/$policy"
    log="$LOGROOT/$name-$policy.log"
    [[ -f "$output/summary.json" ]] && { echo "SKIP $name $policy"; continue; }
    [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
    echo "START shard=$SHARD_INDEX $name $policy $(date -u +%FT%TZ)"
    VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$output" --port "$PORT" \
      --prep-policy "$policy" --background-prep-limit 3 \
      --decode-backend seek_cpu --prep-workers 4 --vlm-concurrency 4 \
      --prepared-queue-depth 32 --decode-timeout-s 600 \
      --request-timeout-s 1800 >"$log" 2>&1
    echo "DONE shard=$SHARD_INDEX $name $policy $(date -u +%FT%TZ)"
  done
done

echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
