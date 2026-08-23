#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/video-rlm
PY=/workspace/vllm-mm/bin/python
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
SUITE=$(cat "$ROOT/logs/latest_priority_trace_suite.path")
TRACE="$SUITE/traces/burst_urgent10-seed1.jsonl"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/conductor/experiments/large_sweeps/replica_scaling_pilot_$STAMP"
LOGROOT="$ROOT/logs/replica_scaling_pilot_$STAMP"

mkdir -p "$OUT" "$LOGROOT"
echo "$OUT" > "$ROOT/logs/latest_replica_scaling_pilot.path"

run_one() {
  local replicas=$1
  local policy=$2
  shift 2
  local ports=("$@")
  local name="replicas${replicas}_${policy}"

  echo "START $name $(date -u +%FT%TZ)"

  "$PY" "$RUNNER" \
    --arrival-trace "$TRACE" \
    --output "$OUT/$name" \
    --ports "${ports[@]}" \
    --replica-routing least_inflight \
    --prep-policy "$policy" \
    --prep-workers 4 \
    --vlm-concurrency 4 \
    --prepared-queue-depth 32 \
    --request-timeout-s 1800 \
    >"$LOGROOT/$name.log" 2>&1

  echo "DONE $name $(date -u +%FT%TZ)"
}

run_one 1 fcfs     9000
run_one 1 priority 9000

run_one 2 fcfs     9000 9001
run_one 2 priority 9000 9001

run_one 4 fcfs     9000 9001 9002 9003
run_one 4 priority 9000 9001 9002 9003

echo "ALL_DONE"
echo "OUT=$OUT"
