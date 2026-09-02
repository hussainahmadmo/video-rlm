#!/usr/bin/env bash
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT to the directory containing trace JSONL files}"

PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
NVDEC_GPU_ID=${NVDEC_GPU_ID:-0}
TRACE_NAMES=${TRACE_NAMES:-"burst_urgent10-seed1 staggered_urgent10-seed1 poisson_rate0.5-urgent30-seed1"}
BACKENDS=${BACKENDS:-"seek_cpu indexed_nvdec"}
POLICIES=${POLICIES:-"fcfs priority priority_reserved"}
BAD_QID=${BAD_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}
RUNNER="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${INDEXED_NVDEC_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/indexed_nvdec_priority_$STAMP"}
LOGROOT=${INDEXED_NVDEC_LOGROOT:-"$VIDEO_RLM_ROOT/logs/$(basename "$OUT")"}

mkdir -p "$OUT/traces" "$LOGROOT"
echo "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_indexed_nvdec_priority.path"

curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "invalid shard $SHARD_INDEX/$NUM_SHARDS" >&2
  exit 2
fi

job_index=0
for trace_name in $TRACE_NAMES; do
  source_trace="$TRACE_ROOT/$trace_name.jsonl"
  clean_trace="$OUT/traces/$trace_name.jsonl"
  test -f "$source_trace" || { echo "missing trace: $source_trace" >&2; exit 1; }
  jq -c --arg qid "$BAD_QID" \
    'select(((.qid // .question_id // .id) | tostring) != $qid)' \
    "$source_trace" > "$clean_trace"

  for backend in $BACKENDS; do
    for policy in $POLICIES; do
      if (( job_index % NUM_SHARDS != SHARD_INDEX )); then
        job_index=$((job_index + 1))
        continue
      fi
      job_index=$((job_index + 1))
      name="$trace_name/${backend}_${policy}"
      summary="$OUT/$name/summary.json"
      if [[ -s $summary ]]; then
        echo "SKIP completed $name"
        continue
      fi
      echo "START $name $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 VIDEO_RLM_NVDEC_GPU_ID="$NVDEC_GPU_ID" \
        "$VLLM_PYTHON" "$RUNNER" \
          --arrival-trace "$clean_trace" \
          --output "$OUT/$name" \
          --port "$PORT" \
          --prep-policy "$policy" \
          --background-prep-limit 3 \
          --decode-backend "$backend" \
          --prep-workers 4 \
          --vlm-concurrency 4 \
          --prepared-queue-depth 32 \
          --decode-timeout-s 600 \
          --request-timeout-s 1800 \
          >"$LOGROOT/$trace_name-${backend}-${policy}.log" 2>&1
      echo "DONE $name $(date -u +%FT%TZ)"
    done
  done
done

echo "ALL_DONE"
echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
