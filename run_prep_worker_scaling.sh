#!/usr/bin/env bash
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT to the directory containing trace JSONL files}"
PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
WORKERS_LIST=${WORKERS_LIST:-"2 4 8 16"}
TRACE_NAMES=${TRACE_NAMES:-"burst_urgent10-seed1 burst_urgent10-seed2 burst_urgent10-seed3 staggered_urgent10-seed1 staggered_urgent10-seed2 staggered_urgent10-seed3 poisson_rate0.5-urgent30-seed1 poisson_rate0.5-urgent30-seed2 poisson_rate0.5-urgent30-seed3"}
POLICIES=${POLICIES:-"fcfs priority priority_reserved"}
RUNNER="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${PREP_WORKER_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/prep_worker_scaling_$STAMP"}
LOGROOT=${PREP_WORKER_LOGROOT:-"$VIDEO_RLM_ROOT/logs/$(basename "$OUT")"}
mkdir -p "$OUT/traces" "$LOGROOT"
echo "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_prep_worker_scaling.path"

if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "invalid shard $SHARD_INDEX/$NUM_SHARDS" >&2
  exit 2
fi

curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

trace_index=0
for trace_name in $TRACE_NAMES; do
  if (( trace_index % NUM_SHARDS != SHARD_INDEX )); then
    trace_index=$((trace_index + 1))
    continue
  fi
  trace_index=$((trace_index + 1))
  source_trace="$TRACE_ROOT/$trace_name.jsonl"
  clean_trace="$OUT/traces/$trace_name.jsonl"
  test -f "$source_trace" || { echo "missing trace: $source_trace" >&2; exit 1; }
  jq -c 'select((.qid // .question_id // "") != "20520eff-abdf-4d4f-94ad-cc751a8960d0")' \
    "$source_trace" > "$clean_trace"
done

for workers in $WORKERS_LIST; do
  trace_index=0
  for trace_name in $TRACE_NAMES; do
    if (( trace_index % NUM_SHARDS != SHARD_INDEX )); then
      trace_index=$((trace_index + 1))
      continue
    fi
    trace_index=$((trace_index + 1))
    trace="$OUT/traces/$trace_name.jsonl"
    for policy in $POLICIES; do
      name="workers${workers}/${trace_name}/${policy}"
      output="$OUT/$name"
      log="$LOGROOT/workers${workers}-${trace_name}-${policy}.log"
      if test -f "$output/summary.json"; then
        echo "SKIP $name"
        continue
      fi
      if test -d "$output"; then
        partial="${output}.partial_$(date +%Y%m%d_%H%M%S)"
        echo "ARCHIVE_PARTIAL $name -> $partial"
        mv "$output" "$partial"
      fi
      echo "START $name $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 "$VLLM_PYTHON" "$RUNNER" \
        --arrival-trace "$trace" \
        --output "$output" \
        --port "$PORT" \
        --prep-policy "$policy" \
        --background-prep-limit "$((workers - 1))" \
        --decode-backend seek_cpu \
        --prep-workers "$workers" \
        --vlm-concurrency 4 \
        --prepared-queue-depth 32 \
        --decode-timeout-s 600 \
        --request-timeout-s 1800 \
        >"$log" 2>&1
      echo "DONE $name $(date -u +%FT%TZ)"
    done
  done
done

echo "ALL_DONE"
echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
