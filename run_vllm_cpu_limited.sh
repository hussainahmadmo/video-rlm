#!/usr/bin/env bash
set -euo pipefail

# Launch one vLLM replica with a hard CPU-affinity limit. All descendants of
# this process, including EngineCore and native video-processing threads,
# inherit CPUSET. Thread-pool variables reduce oversubscription, while taskset
# is the mechanism that actually enforces the CPU limit.

VLLM_BIN=${VLLM_BIN:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/vllm}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
GPU=${GPU:-0}
PORT=${PORT:-9010}
CPUSET=${CPUSET:-0-3}
THREADS=${THREADS:-}
MEDIA_LOADING_THREADS=${MEDIA_LOADING_THREADS:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
VIDEO_FETCH_TIMEOUT=${VIDEO_FETCH_TIMEOUT:-1800}

command -v taskset >/dev/null || {
    echo "taskset is required" >&2
    exit 1
}
test -x "$VLLM_BIN" || {
    echo "vLLM executable not found: $VLLM_BIN" >&2
    exit 1
}

# Derive the number of permitted logical CPUs for library thread pools. This
# accepts CPU lists such as 0-3,8-11. An explicit THREADS value overrides it.
if test -z "$THREADS"; then
    THREADS=$(
        python3 - "$CPUSET" <<'PY'
import sys

cpus = set()
for item in sys.argv[1].split(","):
    item = item.strip()
    if not item:
        continue
    if "-" in item:
        start, end = map(int, item.split("-", 1))
        cpus.update(range(start, end + 1))
    else:
        cpus.add(int(item))
if not cpus:
    raise SystemExit("CPUSET is empty")
print(len(cpus))
PY
    )
fi

# vLLM's native media loader has a separate asynchronous thread pool. Keep it
# inside the same explicit CPU budget unless the caller requests another size.
if test -z "$MEDIA_LOADING_THREADS"; then
    MEDIA_LOADING_THREADS=$THREADS
fi

echo "model=$MODEL"
echo "gpu=$GPU"
echo "port=$PORT"
echo "cpuset=$CPUSET"
echo "thread_pool_limit=$THREADS"
echo "media_loading_threads=$MEDIA_LOADING_THREADS"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export VLLM_VIDEO_FETCH_TIMEOUT=$VIDEO_FETCH_TIMEOUT
export VLLM_MEDIA_LOADING_THREAD_COUNT=$MEDIA_LOADING_THREADS

# These variables limit common native CPU pools. They are not substitutes for
# taskset because a process may create additional asynchronous worker threads.
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export OPENCV_FOR_THREADS_NUM=$THREADS
export RAYON_NUM_THREADS=$THREADS
export TOKENIZERS_PARALLELISM=false

exec taskset --cpu-list "$CPUSET" \
    "$VLLM_BIN" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype auto \
    --api-key EMPTY \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --scheduling-policy priority
