#!/usr/bin/env bash
# Run one fair native-vLLM decoder-backend ablation on all 249 questions.
# The target vLLM server must already be running on TASK_PORT. This script does
# not start, stop, or modify any server.

set -u
set -o pipefail

TASK_ROOT=${TASK_ROOT:-/workspace/video-rlm}
TASK_PYTHON=${TASK_PYTHON:-/workspace/vllm-mm/bin/python}
TASK_DATASET=${TASK_DATASET:-$TASK_ROOT/conductor/experiments/large_sweeps/native_vllm_metis249/dataset.jsonl}
TASK_PORT=${TASK_PORT:-9000}
TASK_RATE=${TASK_RATE:-1.0}
TASK_CONCURRENCY=${TASK_CONCURRENCY:-4}
TASK_CASE=${TASK_CASE:-opencv_default}
# Leave empty to preserve the server's default decoder backend.
TASK_VIDEO_BACKEND=${TASK_VIDEO_BACKEND:-}
TASK_VIDEO_BACKEND_KWARGS=${TASK_VIDEO_BACKEND_KWARGS:-\{\}}
TASK_STAMP=${TASK_STAMP:-$(date +%Y%m%d_%H%M%S)}
TASK_OUT=${TASK_OUT:-$TASK_ROOT/conductor/experiments/large_sweeps/native_backend_${TASK_CASE}_rate${TASK_RATE//./p}_$TASK_STAMP}
TASK_LOG=${TASK_LOG:-$TASK_ROOT/logs/native_backend_${TASK_CASE}_rate${TASK_RATE//./p}_$TASK_STAMP.log}

RUNNER=$TASK_ROOT/conductor/experiments/scripts/run/run_native_vllm_video_baseline.py
backend_args=()
if [[ -n $TASK_VIDEO_BACKEND ]]; then
  backend_args+=(--video-backend "$TASK_VIDEO_BACKEND")
fi

mkdir -p "$TASK_OUT"
echo "case=$TASK_CASE port=$TASK_PORT rate=$TASK_RATE backend=${TASK_VIDEO_BACKEND:-server_default}"
echo "output=$TASK_OUT"
echo "log=$TASK_LOG"

PYTHONUNBUFFERED=1 "$TASK_PYTHON" "$RUNNER" \
  --dataset "$TASK_DATASET" \
  --output "$TASK_OUT/results.jsonl" \
  --summary-output "$TASK_OUT/summary.json" \
  --frame-counts 8 \
  --ports "$TASK_PORT" \
  --concurrency "$TASK_CONCURRENCY" \
  --arrival-rate-qps "$TASK_RATE" \
  --video-root /workspace \
  --max-pixels 100352 \
  --max-tokens 32 \
  --request-timeout-s 300 \
  --video-backend-kwargs "$TASK_VIDEO_BACKEND_KWARGS" \
  "${backend_args[@]}" \
  >"$TASK_LOG" 2>&1

status=$?
if [[ $status -eq 0 ]]; then
  jq '{examples, errors_this_invocation, correct_this_invocation, throughput_qps, mean_end_to_end_delay_s, p50_end_to_end_delay_s, p95_end_to_end_delay_s, video_backend, video_backend_kwargs}' "$TASK_OUT/summary.json"
fi
exit "$status"
