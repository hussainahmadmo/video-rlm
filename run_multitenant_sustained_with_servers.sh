#!/usr/bin/env bash
set -euo pipefail

# One-command recovery launcher: start two isolated vLLM replicas, wait until
# they are healthy, run the two-shard validation, and release both GPUs.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
VLLM=${VLLM_BIN:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/vllm}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
PORT0=${PORT0:-9020}
PORT1=${PORT1:-9021}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SERVER_LOGROOT=${SERVER_LOGROOT:-$ROOT/logs/multitenant_sustained_servers_$STAMP}
mkdir -p "$SERVER_LOGROOT"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver unavailable; reboot Nature before launching." >&2
  exit 1
fi
for port in "$PORT0" "$PORT1"; do
  if ss -ltn | grep -q ":$port "; then
    echo "port $port is already in use" >&2
    exit 1
  fi
done

server_pids=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if ((${#server_pids[@]})); then
    for pid in "${server_pids[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${server_pids[@]}"; do
      kill -KILL -- "-$pid" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

start_server() {
  local gpu=$1
  local port=$2
  nohup setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    "$VLLM" serve "$MODEL" \
      --host 0.0.0.0 --port "$port" --dtype auto --api-key EMPTY \
      --max-model-len 16384 --gpu-memory-utilization 0.85 \
      --no-enable-prefix-caching --mm-processor-cache-gb 0 \
      --scheduling-policy priority \
      >"$SERVER_LOGROOT/vllm-$port.log" 2>&1 &
  server_pids+=("$!")
  echo "START_SERVER gpu=$gpu port=$port pid=$!"
}

start_server 0 "$PORT0"
start_server 1 "$PORT1"

for attempt in $(seq 1 120); do
  ready=0
  for port in "$PORT0" "$PORT1"; do
    if curl -fsS --max-time 5 -H 'Authorization: Bearer EMPTY' \
      "http://127.0.0.1:$port/v1/models" >/dev/null; then
      ready=$((ready + 1))
    fi
  done
  if ((ready == 2)); then
    echo "SERVERS_READY attempt=$attempt $(date -u +%FT%TZ)"
    break
  fi
  for pid in "${server_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM exited during startup; inspect $SERVER_LOGROOT" >&2
      exit 1
    fi
  done
  if ((attempt == 120)); then
    echo "vLLM startup timed out; inspect $SERVER_LOGROOT" >&2
    exit 1
  fi
  sleep 10
done

env PORT0="$PORT0" PORT1="$PORT1" STAMP="$STAMP" \
  "$ROOT/run_multitenant_sustained_2shard.sh"
