#!/usr/bin/env bash
# Same-server load sweep for native vLLM video input versus external prepared frames.
# Runs sequentially so two methods never compete for the same vLLM replica.

set -u
set -o pipefail

TASK_ROOT=${TASK_ROOT:-/workspace/video-rlm}
TASK_PYTHON=${TASK_PYTHON:-/workspace/vllm-mm/bin/python}
TASK_DATASET=${TASK_DATASET:-$TASK_ROOT/conductor/experiments/large_sweeps/native_vllm_metis249/dataset.jsonl}
TASK_PORT=${TASK_PORT:-9000}
TASK_CLIP_DEVICE=${TASK_CLIP_DEVICE:-cuda:3}
TASK_RATES=${TASK_RATES:-"0.10 0.25 0.50 0.75 1.00"}
TASK_STAMP=${TASK_STAMP:-$(date +%Y%m%d_%H%M%S)}
TASK_RUN_ROOT=${TASK_RUN_ROOT:-$TASK_ROOT/conductor/experiments/large_sweeps/native_vs_prepared_load_sweep_$TASK_STAMP}
TASK_LOG_ROOT=${TASK_LOG_ROOT:-$TASK_ROOT/logs/native_vs_prepared_load_sweep_$TASK_STAMP}
TASK_MASTER_LOG=$TASK_LOG_ROOT/master.log

NATIVE_RUNNER=$TASK_ROOT/conductor/experiments/scripts/run/run_native_vllm_video_baseline.py
PREPARED_RUNNER=$TASK_ROOT/conductor/experiments/scripts/run/run_dynamic_queue_aware_hybrid.py
SCHEDULE=$TASK_ROOT/conductor/experiments/large_sweeps/e2e_249/schedule_fixed_budget8.jsonl

mkdir -p "$TASK_RUN_ROOT" "$TASK_LOG_ROOT"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$TASK_MASTER_LOG"
}

rate_tag() {
  printf '%s' "$1" | tr '.' 'p'
}

run_with_traces() {
  local name=$1
  shift
  local trace_dir=$TASK_RUN_ROOT/$name/traces
  mkdir -p "$trace_dir"

  nvidia-smi dmon -s pucvmt -d 1 >"$trace_dir/gpu_dmon.log" 2>&1 &
  local gpu_pid=$!
  pidstat -rud -p ALL 1 >"$trace_dir/cpu_pidstat.log" 2>&1 &
  local cpu_pid=$!
  iostat -x 1 >"$trace_dir/disk_iostat.log" 2>&1 &
  local disk_pid=$!

  "$@"
  local status=$?
  kill "$gpu_pid" "$cpu_pid" "$disk_pid" 2>/dev/null || true
  wait "$gpu_pid" "$cpu_pid" "$disk_pid" 2>/dev/null || true
  return "$status"
}

run_native() {
  local rate=$1
  local tag
  tag=$(rate_tag "$rate")
  local out=$TASK_RUN_ROOT/native_uniform8_rate$tag
  local run_log=$TASK_LOG_ROOT/native_uniform8_rate$tag.log
  [[ -s $out/summary.json ]] && { log "SKIP native rate=$rate"; return; }
  mkdir -p "$out"
  log "START native_uniform8 rate=$rate port=$TASK_PORT"
  run_with_traces "native_uniform8_rate$tag" env PYTHONUNBUFFERED=1 "$TASK_PYTHON" "$NATIVE_RUNNER" \
    --dataset "$TASK_DATASET" \
    --output "$out/results.jsonl" \
    --summary-output "$out/summary.json" \
    --frame-counts 8 --ports "$TASK_PORT" --concurrency 4 \
    --arrival-rate-qps "$rate" --video-root /workspace \
    --max-pixels 100352 --max-tokens 32 --request-timeout-s 300 \
    >"$run_log" 2>&1
  local status=$?
  if [[ $status -eq 0 && -s $out/summary.json ]]; then
    log "DONE native_uniform8 rate=$rate"
    jq '{questions: .examples, errors: .errors_this_invocation, accuracy_percent: (100 * .correct_this_invocation / .completed_this_invocation), throughput_qps, mean_end_to_end_delay_s, p50_end_to_end_delay_s, p95_end_to_end_delay_s}' "$out/summary.json" | tee -a "$TASK_MASTER_LOG"
  else
    log "FAILED native_uniform8 rate=$rate exit=$status"
    tail -n 30 "$run_log" | tee -a "$TASK_MASTER_LOG"
  fi
  sleep 30
}

run_prepared() {
  local rate=$1
  local tag
  tag=$(rate_tag "$rate")
  local out=$TASK_RUN_ROOT/prepared_uniform8_rate$tag
  local run_log=$TASK_LOG_ROOT/prepared_uniform8_rate$tag.log
  [[ -s $out/summary.json ]] && { log "SKIP prepared rate=$rate"; return; }
  log "START prepared_uniform8 rate=$rate port=$TASK_PORT"
  run_with_traces "prepared_uniform8_rate$tag" env PYTHONUNBUFFERED=1 "$TASK_PYTHON" "$PREPARED_RUNNER" \
    --dataset "$TASK_DATASET" --output "$out" \
    --dispatch-policy fifo --config-policy fixed \
    --medium-schedule "$SCHEDULE" \
    --arrival-rate-qps "$rate" --ports "$TASK_PORT" \
    --cpu-workers 4 --vlm-concurrency 4 --prepared-queue-depth 32 \
    --short-input-mode prepared_frames --long-input-mode prepared_uniform \
    --short-max-s 1000000000000 --medium-max-s 1000000000001 \
    --short-frames 8 --long-frames 8 --video-root /workspace \
    --clip-device "$TASK_CLIP_DEVICE" --max-pixels 100352 --max-tokens 32 \
    --request-timeout-s 300 \
    >"$run_log" 2>&1
  local status=$?
  if [[ $status -eq 0 && -s $out/summary.json ]]; then
    log "DONE prepared_uniform8 rate=$rate"
    jq '{questions, errors, accuracy_percent, throughput_qps, mean_end_to_end_delay_s, p50_end_to_end_delay_s, p95_end_to_end_delay_s}' "$out/summary.json" | tee -a "$TASK_MASTER_LOG"
  else
    log "FAILED prepared_uniform8 rate=$rate exit=$status"
    tail -n 30 "$run_log" | tee -a "$TASK_MASTER_LOG"
  fi
  sleep 30
}

for rate in $TASK_RATES; do
  run_native "$rate"
  run_prepared "$rate"
done

log "ALL LOAD-SWEEP EXPERIMENTS FINISHED run_root=$TASK_RUN_ROOT"
