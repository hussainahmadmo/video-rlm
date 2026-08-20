#!/usr/bin/env bash

# Run the missing video-serving ablations sequentially.  Start this script with
# setsid (see the command printed below) so it survives terminal disconnects.

set -u
set -o pipefail

TASK_ROOT=${TASK_ROOT:-/workspace/video-rlm}
TASK_PYTHON=${TASK_PYTHON:-/workspace/vllm-mm/bin/python}
TASK_DATASET=${TASK_DATASET:-$TASK_ROOT/conductor/experiments/large_sweeps/native_vllm_metis249/dataset.jsonl}
TASK_E2E=${TASK_E2E:-$TASK_ROOT/conductor/experiments/large_sweeps/e2e_249}
TASK_PLAN=${TASK_PLAN:-$TASK_ROOT/conductor/experiments/large_sweeps/queue_aware_plan_249/dispatch_plan.jsonl}
TASK_NATIVE_OUT=${TASK_NATIVE_OUT:-$TASK_ROOT/conductor/experiments/large_sweeps/native_raw8_rate10_20260819_163241}
TASK_PORT=${TASK_PORT:-9000}
TASK_CLIP_DEVICE=${TASK_CLIP_DEVICE:-cuda:3}
TASK_RATE=${TASK_RATE:-1.0}
TASK_CPU_WORKERS=${TASK_CPU_WORKERS:-4}
TASK_VLM_CONCURRENCY=${TASK_VLM_CONCURRENCY:-4}
TASK_QUEUE_DEPTH=${TASK_QUEUE_DEPTH:-32}
TASK_STAMP=${TASK_STAMP:-$(date +%Y%m%d_%H%M%S)}

TASK_RUN_ROOT=${TASK_RUN_ROOT:-$TASK_ROOT/conductor/experiments/large_sweeps/overnight_ablations_$TASK_STAMP}
TASK_LOG_ROOT=${TASK_LOG_ROOT:-$TASK_ROOT/logs/overnight_ablations_$TASK_STAMP}
TASK_MASTER_LOG=$TASK_LOG_ROOT/master.log
TASK_RUNNER=$TASK_ROOT/conductor/experiments/scripts/run/run_dynamic_queue_aware_hybrid.py

mkdir -p "$TASK_RUN_ROOT" "$TASK_LOG_ROOT"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$TASK_MASTER_LOG"
}

require_file() {
  if [[ ! -f $1 ]]; then
    log "FATAL missing required file: $1"
    exit 2
  fi
}

require_file "$TASK_PYTHON"
require_file "$TASK_DATASET"
require_file "$TASK_PLAN"
require_file "$TASK_E2E/schedule_fixed_budget2.jsonl"
require_file "$TASK_E2E/schedule_fixed_budget8.jsonl"
require_file "$TASK_E2E/schedule_fixed_budget32.jsonl"
require_file "$TASK_RUNNER"

run_experiment() {
  local name=$1
  shift
  local out=$TASK_RUN_ROOT/$name
  local run_log=$TASK_LOG_ROOT/$name.log

  if [[ -s $out/summary.json ]]; then
    log "SKIP $name: summary already exists"
    return 0
  fi

  if [[ -e $out ]]; then
    out=${out}_retry_$(date +%H%M%S)
    run_log=${run_log%.log}_retry_$(date +%H%M%S).log
  fi

  log "START $name output=$out log=$run_log"
  env PYTHONUNBUFFERED=1 "$TASK_PYTHON" "$TASK_RUNNER" \
    --dataset "$TASK_DATASET" \
    --output "$out" \
    --dispatch-policy fifo \
    --arrival-rate-qps "$TASK_RATE" \
    --ports "$TASK_PORT" \
    --cpu-workers "$TASK_CPU_WORKERS" \
    --vlm-concurrency "$TASK_VLM_CONCURRENCY" \
    --prepared-queue-depth "$TASK_QUEUE_DEPTH" \
    --short-input-mode prepared_frames \
    --short-max-s 300 \
    --medium-max-s 1200 \
    --video-root /workspace \
    --clip-device "$TASK_CLIP_DEVICE" \
    --max-pixels 100352 \
    --max-tokens 32 \
    --request-timeout-s 300 \
    "$@" >"$run_log" 2>&1
  local status=$?

  if [[ $status -eq 0 && -s $out/summary.json ]]; then
    log "DONE $name"
    jq '{questions,errors,accuracy_percent,throughput_qps,mean_end_to_end_delay_s,p50_end_to_end_delay_s,p95_end_to_end_delay_s,selected_configs}' \
      "$out/summary.json" | tee -a "$TASK_MASTER_LOG"
  else
    log "FAILED $name exit=$status; continuing with the next experiment"
    tail -n 20 "$run_log" | tee -a "$TASK_MASTER_LOG"
  fi
  return 0
}

# Avoid overlapping with the currently running native raw-video baseline.
if [[ -d $TASK_NATIVE_OUT && ! -s $TASK_NATIVE_OUT/summary.json ]]; then
  log "WAIT native raw-8 baseline: $TASK_NATIVE_OUT"
  while pgrep -f -- "--output $TASK_NATIVE_OUT/results.jsonl" >/dev/null 2>&1; do
    completed=$(wc -l <"$TASK_NATIVE_OUT/results.jsonl" 2>/dev/null || printf '0')
    log "native raw-8 completed=$completed/249"
    sleep 60
  done
  log "Native process ended; allowing port/storage activity to settle"
  sleep 15
fi

# Pure external-decode baseline: route every duration through uniform JPEG
# preparation, so retrieval and codec refinement are disabled.
run_experiment external_uniform8 \
  --config-policy fixed \
  --medium-schedule "$TASK_E2E/schedule_fixed_budget8.jsonl" \
  --short-max-s 1000000000000 \
  --medium-max-s 1000000000001 \
  --short-frames 8 \
  --long-frames 8

# True load adaptation: baseline/rich use 8/32 as applicable and newly
# arriving work switches to budget 2 once live outstanding load reaches 4.
run_experiment adaptive_2_8_threshold4 \
  --dispatch-plan "$TASK_PLAN" \
  --config-policy load_adaptive \
  --adaptive-high-load 4 \
  --medium-schedule "$TASK_E2E/schedule_fixed_budget8.jsonl" \
  --medium-schedule-budget2 "$TASK_E2E/schedule_fixed_budget2.jsonl" \
  --medium-schedule-budget32 "$TASK_E2E/schedule_fixed_budget32.jsonl" \
  --adaptive-rich-short-frames 8 \
  --adaptive-cheap-short-frames 2 \
  --adaptive-rich-long-frames 8 \
  --adaptive-cheap-long-frames 2

run_experiment fixed_budget2 \
  --dispatch-plan "$TASK_PLAN" \
  --config-policy fixed \
  --medium-schedule "$TASK_E2E/schedule_fixed_budget2.jsonl" \
  --short-frames 2 \
  --long-frames 2

run_experiment fixed_budget32 \
  --dispatch-plan "$TASK_PLAN" \
  --config-policy fixed \
  --medium-schedule "$TASK_E2E/schedule_fixed_budget32.jsonl" \
  --short-frames 32 \
  --long-frames 32

log "ALL OVERNIGHT EXPERIMENTS FINISHED run_root=$TASK_RUN_ROOT"
