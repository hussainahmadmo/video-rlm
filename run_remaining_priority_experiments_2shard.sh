#!/usr/bin/env bash
# Run two NUMA-isolated experiment shards against the two local A40 replicas.
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT}"

STAMP=$(date +%Y%m%d_%H%M%S)
PREP_WORKER_OUT=${PREP_WORKER_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/prep_worker_scaling_2shard_$STAMP"}
BATCH_CPU_OUT=${BATCH_CPU_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/batched_cpu_ablation_2shard_$STAMP"}
NVDEC_OUT=${NVDEC_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/nvdec_ablation_2shard_$STAMP"}
NATIVE_MEDIA_OUT=${NATIVE_MEDIA_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/native_media_backend_matrix_2shard_$STAMP"}

export PREP_WORKER_OUT BATCH_CPU_OUT NVDEC_OUT NATIVE_MEDIA_OUT
export PREP_WORKER_LOGROOT="$VIDEO_RLM_ROOT/logs/$(basename "$PREP_WORKER_OUT")"
export BATCH_CPU_LOGROOT="$VIDEO_RLM_ROOT/logs/$(basename "$BATCH_CPU_OUT")"
export NVDEC_LOGROOT="$VIDEO_RLM_ROOT/logs/$(basename "$NVDEC_OUT")"
export NATIVE_MEDIA_LOGROOT="$VIDEO_RLM_ROOT/logs/$(basename "$NATIVE_MEDIA_OUT")"

mkdir -p "$VIDEO_RLM_ROOT/logs"
echo "$PREP_WORKER_OUT" > "$VIDEO_RLM_ROOT/logs/latest_prep_worker_scaling.path"

launch_shard() {
  local shard=$1 port=$2 cpus=$3 gpu=$4
  local log="$VIDEO_RLM_ROOT/logs/remaining_priority_experiments_2shard_shard${shard}.log"
  taskset -c "$cpus" env \
    SHARD_INDEX="$shard" NUM_SHARDS=2 PORT="$port" NVDEC_GPU_ID="$gpu" \
    NATIVE_PORT="$port" VIDEO_HTTP_PORT="$((8088 + shard))" MASTER_LOG="$log" \
    "$VIDEO_RLM_ROOT/run_remaining_priority_experiments.sh" \
    >"$log.launcher" 2>&1 &
  SHARD_PID=$!
}

launch_shard 0 9000 0-15,32-47 0
PID0=$SHARD_PID
launch_shard 1 9001 16-31,48-63 1
PID1=$SHARD_PID
echo "SHARD0_PID=$PID0 SHARD1_PID=$PID1"
set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e
echo "TWO_SHARD_DONE status0=$STATUS0 status1=$STATUS1"
(( STATUS0 == 0 && STATUS1 == 0 ))
