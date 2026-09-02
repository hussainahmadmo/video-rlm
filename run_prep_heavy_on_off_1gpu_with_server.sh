#!/usr/bin/env bash
set -euo pipefail

# Start one isolated vLLM replica, run the 2- and 4-worker points
# sequentially, generate the report, and release the GPU.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
VLLM=${VLLM_BIN:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/vllm}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-9020}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/prep_heavy_on_off_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_vtc_multimodal.py
mkdir -p "$OUT" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_prep_heavy_on_off.path"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver unavailable" >&2
  exit 1
fi
if ss -ltn | grep -q ":$PORT "; then
  echo "port $PORT is already in use" >&2
  exit 1
fi

server_pid=
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$server_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

nohup setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
  "$VLLM" serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --dtype auto --api-key EMPTY \
    --max-model-len 16384 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching --mm-processor-cache-gb 0 \
    --scheduling-policy priority \
    >"$LOGROOT/vllm-$PORT.log" 2>&1 &
server_pid=$!
echo "START_SERVER gpu=$GPU_ID port=$PORT pid=$server_pid"

for attempt in $(seq 1 120); do
  if curl -fsS --max-time 5 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "SERVER_READY attempt=$attempt $(date -u +%FT%TZ)"
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM exited during startup; inspect $LOGROOT/vllm-$PORT.log" >&2
    exit 1
  fi
  if ((attempt == 120)); then
    echo "vLLM startup timed out; inspect $LOGROOT/vllm-$PORT.log" >&2
    exit 1
  fi
  sleep 10
done

for workers in 2 4; do
  env PORT="$PORT" PREP_WORKERS="$workers" STAMP="$STAMP" \
    OUT="$OUT" LOGROOT="$LOGROOT" \
    "$ROOT/run_prep_heavy_on_off_worker.sh" \
    >"$LOGROOT/workers${workers}-launcher.log" 2>&1
done

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/video-rlm-matplotlib} \
  "$PY" "$ANALYZER" --root "$OUT" \
  --output "$OUT/prep_heavy_on_off_results"
echo "ALL_DONE $(date -u +%FT%TZ) OUT=$OUT"
