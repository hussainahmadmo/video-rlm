#!/usr/bin/env bash
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${BASE_TRACE:?Set BASE_TRACE}"
PORT=${PORT:-9000}
RUNNER="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
if [[ "$VIDEO_RLM_ROOT" == /workspace/* ]]; then
  OUT="$VIDEO_RLM_ROOT/conductor/experiments/large_sweeps/batched_cpu_ablation_$STAMP"
else
  OUT="$VIDEO_RLM_ROOT/large_sweeps/batched_cpu_ablation_$STAMP"
fi
LOGROOT="$VIDEO_RLM_ROOT/logs/batched_cpu_ablation_$STAMP"
mkdir -p "$OUT/traces" "$LOGROOT"
echo "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_batched_cpu_ablation.path"

# Remove the known corrupt background video while preserving all other arrivals.
CLEAN_TRACE="$OUT/traces/workload-clean.jsonl"
jq -c 'select(((.video_id // "") != "20520eff-abdf-4d4f-94ad-cc751a8960d0") and (((.video // "") | contains("20520eff-abdf-4d4f-94ad-cc751a8960d0")) | not))' \
  "$BASE_TRACE" > "$CLEAN_TRACE"
echo "Trace rows: source=$(wc -l < "$BASE_TRACE") clean=$(wc -l < "$CLEAN_TRACE")"

run_one() {
  local backend=$1
  local policy=$2
  local name="${backend}_${policy}"
  echo "START $name $(date -u +%FT%TZ)"
  VIDEO_RLM_FFMPEG_THREADS=1 "$VLLM_PYTHON" "$RUNNER" \
    --arrival-trace "$CLEAN_TRACE" \
    --output "$OUT/$name" \
    --port "$PORT" \
    --prep-policy "$policy" \
    --decode-backend "$backend" \
    --prep-workers 4 \
    --vlm-concurrency 4 \
    --prepared-queue-depth 32 \
    --decode-timeout-s 600 \
    --request-timeout-s 1800 \
    >"$LOGROOT/$name.log" 2>&1
  echo "DONE $name $(date -u +%FT%TZ)"
}

run_one seek_cpu fcfs
run_one seek_cpu priority
run_one batch_cpu fcfs
run_one batch_cpu priority

echo "ALL_DONE"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
