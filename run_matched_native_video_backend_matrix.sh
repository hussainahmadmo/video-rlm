#!/usr/bin/env bash
# Run matched native vLLM video backends, one fresh server per backend.
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON to the NVDEC-capable environment}"
: "${NATIVE_DATASET:?Set NATIVE_DATASET to the matched compatible JSONL}"

GPU=${GPU:-0}
PORT=${PORT:-9020}
VIDEO_HTTP_PORT=${VIDEO_HTTP_PORT:-8090}
BACKENDS=${BACKENDS:-"opencv pynvvideocodec"}
RATES=${RATES:-"0.25 1.0"}
SEEDS=${SEEDS:-"1 2 3"}
FRAME_COUNT=${FRAME_COUNT:-8}
CONCURRENCY=${CONCURRENCY:-4}
MM_IPC_GPU_MEMORY_GB=${MM_IPC_GPU_MEMORY_GB:-1}
VLLM_BIN=${VLLM_BIN:-"$(dirname "$VLLM_PYTHON")/vllm"}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-"$VIDEO_RLM_ROOT/large_sweeps/native_backend_matched_$STAMP"}
LOGROOT=${LOGROOT:-"$VIDEO_RLM_ROOT/logs/$(basename "$OUT")"}

mkdir -p "$OUT" "$LOGROOT"
printf '%s\n' "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_native_backend_matched.path"

SERVER_PID=
cleanup_server() {
  if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
  fi
  SERVER_PID=
}
trap cleanup_server EXIT INT TERM

wait_for_server() {
  local backend=$1
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited during startup: backend=$backend" >&2
      tail -n 120 "$LOGROOT/vllm-$backend.log" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "server startup timed out: backend=$backend" >&2
  return 1
}

if ss -ltn | grep -q ":$PORT "; then
  echo "port $PORT is already occupied; stop that server before launching" >&2
  exit 1
fi

for backend in $BACKENDS; do
  echo "START_SERVER backend=$backend $(date -u +%FT%TZ)"
  env \
    GPU="$GPU" PORT="$PORT" VIDEO_BACKEND="$backend" \
    VLLM_BIN="$VLLM_BIN" MEDIA_LOADING_THREADS="$CONCURRENCY" \
    MM_IPC_GPU_MEMORY_GB="$MM_IPC_GPU_MEMORY_GB" \
    LOG_FILE="$LOGROOT/vllm-$backend.log" \
    "$VIDEO_RLM_ROOT/run_vllm_video_backend.sh" \
    >"$LOGROOT/vllm-$backend-launcher.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$backend"
  echo "SERVER_READY backend=$backend pid=$SERVER_PID $(date -u +%FT%TZ)"

  NATIVE_MEDIA_OUT="$OUT" \
  NATIVE_MEDIA_LOGROOT="$LOGROOT" \
  NATIVE_DATASET="$NATIVE_DATASET" \
  NATIVE_PORT="$PORT" \
  VIDEO_HTTP_ROOT=/dataheart/hussainahmad \
  VIDEO_HTTP_PORT="$VIDEO_HTTP_PORT" \
  NATIVE_BACKENDS="$backend" \
  NATIVE_RATES="$RATES" \
  NATIVE_SEEDS="$SEEDS" \
  NATIVE_FRAME_COUNT="$FRAME_COUNT" \
  NATIVE_CONCURRENCY="$CONCURRENCY" \
  NATIVE_NVDEC_READY=$([[ $backend == pynvvideocodec ]] && echo 1 || echo 0) \
    "$VIDEO_RLM_ROOT/run_native_media_backend_matrix.sh"

  failures=0
  while IFS= read -r summary; do
    errors=$(jq -r '.errors // 0' "$summary")
    if [[ $errors != 0 ]]; then
      echo "ERRORS backend=$backend count=$errors summary=$summary" >&2
      failures=$((failures + 1))
    fi
  done < <(find "$OUT" -type f -path "*/${backend}_*/summary.json" -print)
  (( failures == 0 )) || exit 1

  cleanup_server
  for _ in $(seq 1 60); do
    ss -ltn | grep -q ":$PORT " || break
    sleep 1
  done
  echo "DONE_BACKEND backend=$backend $(date -u +%FT%TZ)"
done

trap - EXIT INT TERM
echo "ALL_DONE OUT=$OUT LOGROOT=$LOGROOT"
