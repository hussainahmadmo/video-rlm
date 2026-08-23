#!/usr/bin/env bash
set -euo pipefail

ROOT=${VIDEO_RLM_ROOT:-/workspace/video-rlm}
PY=${VLLM_PYTHON:-/workspace/vllm-mm/bin/python}
GEN="$ROOT/conductor/experiments/scripts/run/generate_priority_workloads.py"
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
DATASET=${PRIORITY_DATASET:-"$ROOT/conductor/experiments/large_sweeps/native_vllm_metis249/dataset.jsonl"}
PORT=9001
MODE=pilot
REPLAY_TEMPLATE=
SHARD_INDEX=0
NUM_SHARDS=1
SUITE_ARG=
LOGROOT_ARG=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --full) MODE=full ;;
    --port) PORT=$2; shift ;;
    --dataset) DATASET=$2; shift ;;
    --replay-template) REPLAY_TEMPLATE=$2; shift ;;
    --shard-index) SHARD_INDEX=$2; shift ;;
    --num-shards) NUM_SHARDS=$2; shift ;;
    --suite) SUITE_ARG=$2; shift ;;
    --log-root) LOGROOT_ARG=$2; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$NUM_SHARDS" -lt 1 ] || [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
  echo "require 0 <= --shard-index < --num-shards" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$SUITE_ARG" ]; then
  SUITE=$SUITE_ARG
else
  SUITE="$ROOT/conductor/experiments/large_sweeps/priority_trace_suite_$MODE-$STAMP"
fi
if [ -n "$LOGROOT_ARG" ]; then
  LOGROOT=$LOGROOT_ARG
else
  LOGROOT="$ROOT/logs/priority_trace_suite_$MODE-$STAMP"
fi
TRACES="$SUITE/traces"
mkdir -p "$SUITE" "$LOGROOT" "$TRACES"
printf '%s\n' "$SUITE" > "$ROOT/logs/latest_priority_trace_suite.path"

curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/models" -H 'Authorization: Bearer EMPTY' >/dev/null

wait_for_idle() {
  while true; do
    counts=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/metrics" | awk '
      /^vllm:num_requests_running{/ {running += $NF}
      /^vllm:num_requests_waiting{/ {waiting += $NF}
      END {printf "%.0f %.0f", running, waiting}
    ')
    [ "$counts" = "0 0" ] && return
    echo "vLLM running/waiting: $counts"
    sleep 2
  done
}

run_trial() {
  local trace=$1 name=$2 policy=$3
  local out="$SUITE/$name/$policy"
  local log="$LOGROOT/$name-$policy-shard$SHARD_INDEX.log"

  if [ -f "$out/summary.json" ]; then
    echo "SKIP completed $name $policy"
    return
  fi

  wait_for_idle
  echo "START shard=$SHARD_INDEX port=$PORT $name $policy $(date -u +%FT%TZ)"
  "$PY" "$RUNNER" --arrival-trace "$trace" --output "$out" \
    --port "$PORT" --prep-policy "$policy" --prep-workers 4 \
    --background-prep-limit 3 \
    --vlm-concurrency 4 --prepared-queue-depth 32 \
    --request-timeout-s 1800 >"$log" 2>&1
  echo "DONE shard=$SHARD_INDEX port=$PORT $name $policy $(date -u +%FT%TZ)"
}

TRIAL_INDEX=0
generate_and_run() {
  local name=$1
  shift
  local this_index=$TRIAL_INDEX
  TRIAL_INDEX=$((TRIAL_INDEX + 1))

  if [ $((this_index % NUM_SHARDS)) -ne "$SHARD_INDEX" ]; then
    return
  fi

  local trace="$TRACES/$name.jsonl"
  if [ ! -f "$trace" ]; then
    "$PY" "$GEN" --dataset "$DATASET" --output "$trace" "$@"
  fi
  run_trial "$trace" "$name" fcfs
  run_trial "$trace" "$name" priority
  run_trial "$trace" "$name" priority_reserved
}

if [ "$MODE" = full ]; then
  SEEDS="1 2 3 4 5"
  URGENT_TIMES="1 10 30 60"
  RATES="0.25 0.5 0.75 1.0"
else
  SEEDS="1 2 3"
  URGENT_TIMES="10 30"
  RATES="0.5 1.0"
fi

for seed in $SEEDS; do
  for urgent_at in $URGENT_TIMES; do
    generate_and_run "burst_urgent$urgent_at-seed$seed" \
      --pattern burst --background-count 64 --urgent-count 16 \
      --urgent-arrival-s "$urgent_at" --mixed-frames --seed "$seed"
    generate_and_run "staggered_urgent$urgent_at-seed$seed" \
      --pattern staggered --background-count 64 --urgent-count 16 \
      --interval-s 2 --urgent-arrival-s "$urgent_at" --mixed-frames --seed "$seed"
    for rate in $RATES; do
      generate_and_run "poisson_rate$rate-urgent$urgent_at-seed$seed" \
        --pattern poisson --background-count 64 --urgent-count 16 \
        --rate-qps "$rate" --urgent-arrival-s "$urgent_at" --mixed-frames --seed "$seed"
    done
  done
done

if [ -n "$REPLAY_TEMPLATE" ]; then
  for seed in $SEEDS; do
    generate_and_run "replay_seed$seed" --pattern replay \
      --replay-template "$REPLAY_TEMPLATE" --seed "$seed"
  done
fi

echo "SHARD_DONE shard=$SHARD_INDEX port=$PORT suite=$SUITE"
