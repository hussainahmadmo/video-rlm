#!/usr/bin/env bash
set -euo pipefail

# Run a compact matched subset on a second model, machine, or decoding backend.
# Point MODEL at the model already served by PORT; retain the same traces.

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT}"

PORT=${PORT:-9000}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
DECODE_BACKEND=${DECODE_BACKEND:-seek_cpu}
POLICIES=${POLICIES:-"fcfs priority priority_reserved"}
TRACE_NAMES=${TRACE_NAMES:-"burst_urgent10-seed1 staggered_urgent10-seed1 poisson_rate0.5-urgent30-seed1 poisson_rate1.0-urgent10-seed1"}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}

ROOT=$VIDEO_RLM_ROOT
PY=$VLLM_PYTHON
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
tag=$(printf '%s-%s' "$MODEL" "$DECODE_BACKEND" | tr '/: ' '___')
OUT=${PORTABILITY_OUT:-"$ROOT/large_sweeps/portability_${tag}_$STAMP"}
LOGROOT=${PORTABILITY_LOGROOT:-"$ROOT/logs/$(basename "$OUT")"}
mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_portability_validation.path"

{
  printf 'created_utc=%s\nmodel=%s\ndecode_backend=%s\nport=%s\n' \
    "$(date -u +%FT%TZ)" "$MODEL" "$DECODE_BACKEND" "$PORT"
  uname -a
  command -v lscpu >/dev/null && lscpu
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$OUT/system_manifest.txt"

curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

index=0
for trace_name in $TRACE_NAMES; do
  if (( index % NUM_SHARDS != SHARD_INDEX )); then
    index=$((index + 1))
    continue
  fi
  index=$((index + 1))
  source_trace="$TRACE_ROOT/$trace_name.jsonl"
  trace="$OUT/traces/$trace_name.jsonl"
  [[ -f "$source_trace" ]] || { echo "missing trace: $source_trace" >&2; exit 1; }
  jq -c 'select((.qid // .question_id // .id // "") != "20520eff-abdf-4d4f-94ad-cc751a8960d0")' \
    "$source_trace" > "$trace"
  for policy in $POLICIES; do
    output="$OUT/$trace_name/$policy"
    log="$LOGROOT/$trace_name-$policy.log"
    [[ -f "$output/summary.json" ]] && { echo "SKIP $trace_name $policy"; continue; }
    [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
    echo "START shard=$SHARD_INDEX model=$MODEL $trace_name $policy $(date -u +%FT%TZ)"
    VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$output" --port "$PORT" \
      --model "$MODEL" --decode-backend "$DECODE_BACKEND" \
      --prep-policy "$policy" --background-prep-limit 3 \
      --prep-workers 4 --vlm-concurrency 4 --prepared-queue-depth 32 \
      --decode-timeout-s 600 --request-timeout-s 1800 >"$log" 2>&1
    echo "DONE shard=$SHARD_INDEX $trace_name $policy $(date -u +%FT%TZ)"
  done
done

echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
