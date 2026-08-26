#!/usr/bin/env bash
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT to the directory containing source traces}"

PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
FRAME_PAIRS=${FRAME_PAIRS:-"128:8 8:128 128:128 8:8"}
TRACE_NAMES=${TRACE_NAMES:-"burst_urgent10-seed1 burst_urgent10-seed2 burst_urgent10-seed3"}
POLICIES=${POLICIES:-"fcfs priority"}
RUNNER="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${FRAME_COST_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/frame_cost_matrix_$STAMP"}
LOGROOT=${FRAME_COST_LOGROOT:-"$VIDEO_RLM_ROOT/logs/$(basename "$OUT")"}

mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_frame_cost_matrix.path"

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
  test -f "$source_trace" || {
    echo "missing trace: $source_trace" >&2
    exit 1
  }

  for pair in $FRAME_PAIRS; do
    background_frames=${pair%%:*}
    urgent_frames=${pair##*:}
    condition="background${background_frames}_urgent${urgent_frames}"
    trace="$OUT/traces/${trace_name}-${condition}.jsonl"

    jq -c \
      --argjson background_frames "$background_frames" \
      --argjson urgent_frames "$urgent_frames" '
        select((.qid // .question_id // "") !=
               "20520eff-abdf-4d4f-94ad-cc751a8960d0")
        | if (.class // .workload) == "urgent"
          then .frame_count = $urgent_frames
          else .frame_count = $background_frames
          end
      ' "$source_trace" > "$trace"

    for policy in $POLICIES; do
      name="$condition/$trace_name/$policy"
      output="$OUT/$name"
      log="$LOGROOT/$condition-$trace_name-$policy.log"

      if test -f "$output/summary.json"; then
        echo "SKIP completed $name"
        continue
      fi
      if test -d "$output"; then
        partial="${output}.partial_$(date +%Y%m%d_%H%M%S)"
        echo "ARCHIVE_PARTIAL $name -> $partial"
        mv "$output" "$partial"
      fi

      echo "START shard=$SHARD_INDEX port=$PORT $name $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 "$VLLM_PYTHON" "$RUNNER" \
        --arrival-trace "$trace" \
        --output "$output" \
        --port "$PORT" \
        --prep-policy "$policy" \
        --decode-backend seek_cpu \
        --prep-workers 4 \
        --vlm-concurrency 4 \
        --prepared-queue-depth 32 \
        --decode-timeout-s 600 \
        --request-timeout-s 1800 \
        >"$log" 2>&1
      echo "DONE shard=$SHARD_INDEX port=$PORT $name $(date -u +%FT%TZ)"
    done
  done
done

echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
