#!/usr/bin/env bash
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT}"
PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
MASTER_LOG=${MASTER_LOG:-"$VIDEO_RLM_ROOT/logs/remaining_priority_experiments_shard${SHARD_INDEX}.log"}

run_stage() {
  local name=$1
  shift
  echo "START_STAGE $name $(date -u +%FT%TZ)" | tee -a "$MASTER_LOG"
  "$@" >>"$MASTER_LOG" 2>&1
  echo "DONE_STAGE $name $(date -u +%FT%TZ)" | tee -a "$MASTER_LOG"
}

export VIDEO_RLM_ROOT VLLM_PYTHON TRACE_ROOT PORT SHARD_INDEX NUM_SHARDS
run_stage prep_worker_scaling "$VIDEO_RLM_ROOT/run_prep_worker_scaling.sh"
run_stage batched_cpu_ablation "$VIDEO_RLM_ROOT/run_batched_cpu_ablation.sh"
run_stage nvdec_ablation "$VIDEO_RLM_ROOT/run_nvdec_ablation.sh"
# This native-vLLM control auto-skips loaders that the installed vLLM does not
# register. Native NVDEC additionally remains gated on NATIVE_NVDEC_READY=1.
run_stage native_media_backend_matrix "$VIDEO_RLM_ROOT/run_native_media_backend_matrix.sh"
echo "ALL_REMAINING_EXPERIMENTS_DONE $(date -u +%FT%TZ)" | tee -a "$MASTER_LOG"
