#!/usr/bin/env bash
set -euo pipefail

# Matched native-vLLM comparison. Video fetch, decode, frame sampling, and
# inference stay inside vLLM; only native media admission changes.
ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
VLLM_BIN=${VLLM_BIN:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/vllm}
MODEL=${MODEL:-/dataheart/hussainahmad/hugging_face_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}
SERVED_MODEL=${SERVED_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
PORT=${PORT:-9000}
GPU=${GPU:-1}
MEDIA_THREADS=${MEDIA_THREADS:-8}
OPENCV_THREADS_PER_JOB=${OPENCV_THREADS_PER_JOB:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
SEEDS=${SEEDS:-"1 2 3 4"}
OUT=${OUT:-$ROOT/large_sweeps/native_vllm_max_min_final}
LOGROOT=${LOGROOT:-$ROOT/logs/native_vllm_max_min_final}

WRAPPER=$ROOT/conductor/experiments/scripts/run/run_vllm_with_media_priority.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_native_vllm_priority_burst.py
server_pid=""

mkdir -p "$OUT" "$LOGROOT"

stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}

wait_ready() {
  for _ in $(seq 1 240); do
    if ss -ltnp "( sport = :$PORT )" 2>/dev/null \
        | grep -Fq "pid=$server_pid,"; then
      if curl -fsS --max-time 3 -H 'Authorization: Bearer EMPTY' \
        "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        return 0
      fi
    fi
    kill -0 "$server_pid" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

start_experiment_server() {
  local policy=$1
  local server_log=$LOGROOT/server-$policy.log
  setsid env CUDA_VISIBLE_DEVICES="$GPU" \
    VLLM_MEDIA_LOADING_THREAD_COUNT="$MEDIA_THREADS" \
    "$PY" "$WRAPPER" \
      --media-preparation-policy "$policy" \
      --media-preparation-workers "$MEDIA_THREADS" \
      --opencv-threads-per-job "$OPENCV_THREADS_PER_JOB" \
      serve "$MODEL" --served-model-name "$SERVED_MODEL" \
      --host 127.0.0.1 --port "$PORT" --dtype auto --api-key EMPTY \
      --max-model-len 16384 --gpu-memory-utilization 0.85 \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --scheduling-policy priority \
      >"$server_log" 2>&1 < /dev/null &
  server_pid=$!
  echo "SERVER_START policy=$policy pid=$server_pid $(date -u +%FT%TZ)"
  if ! wait_ready; then
    tail -100 "$server_log" || true
    return 1
  fi
  echo "SERVER_READY policy=$policy $(date -u +%FT%TZ)"
}

restore_original_server() {
  local restore_log=$LOGROOT/server-restored.log
  setsid env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED_MODEL" --host 127.0.0.1 --port "$PORT" \
    --dtype auto --api-key EMPTY --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --scheduling-policy priority \
    >"$restore_log" 2>&1 < /dev/null &
  echo "RESTORED_SERVER_PID=$!"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_server
  restore_original_server
  exit "$status"
}
trap cleanup EXIT INT TERM

for policy in native decode_max_min; do
  start_experiment_server "$policy"
  label=native
  [[ "$policy" == decode_max_min ]] && label=max_min
  for seed in $SEEDS; do
    trace=$OUT/traces/equal-tenants-mixed-cost-seed${seed}.jsonl
    output=$OUT/equal-tenants-mixed-cost-seed${seed}/$label
    run_log=$LOGROOT/equal-tenants-mixed-cost-seed${seed}-$label.log
    if [[ -f "$output/summary.json" ]]; then
      echo "SKIP policy=$label seed=$seed"
      continue
    fi
    if [[ -d "$output" ]]; then
      mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
    fi
    echo "RUN_START policy=$label seed=$seed $(date -u +%FT%TZ)"
    "$PY" "$RUNNER" --arrival-trace "$trace" --output "$output" \
      --port "$PORT" --priority-mode uniform --server-media-policy "$label" \
      --video-map /dataheart/hussainahmad=http://127.0.0.1:8090 \
      --request-timeout-s 1800 >"$run_log" 2>&1
    echo "RUN_DONE policy=$label seed=$seed $(date -u +%FT%TZ)"
  done
  stop_server
done

echo "NATIVE_MAX_MIN_COMPARISON_DONE OUT=$OUT"
