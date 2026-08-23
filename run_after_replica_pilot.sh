#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/video-rlm
CURRENT=$(cat "$ROOT/logs/latest_replica_scaling_pilot.path")
CURRENT_LOG="$ROOT/logs/replica_scaling_pilot_launcher.log"

echo "Waiting for fixed-load replica pilot"
echo "CURRENT=$CURRENT"
while true; do
  DONE=$(find "$CURRENT" -type f -name summary.json | wc -l)
  echo "$(date -u +%FT%TZ) fixed-load completed=$DONE/6"
  (( DONE == 6 )) && break
  if ! pgrep -f '[r]un_replica_scaling_pilot.sh' >/dev/null; then
    echo "ERROR: fixed-load launcher stopped before 6/6"
    tail -n 50 "$CURRENT_LOG"
    exit 1
  fi
  sleep 30
done

while pgrep -f '[r]un_mixed_end_to_end_priority.py' >/dev/null; do
  echo "$(date -u +%FT%TZ) waiting for old runner to exit"
  sleep 10
done
while pgrep -f '[r]un_replica_scaling_pilot.sh' >/dev/null; do
  echo "$(date -u +%FT%TZ) waiting for old launcher to exit"
  sleep 10
done

while true; do
  BUSY=0
  for PORT in 9000 9001 9002 9003; do
    COUNTS=$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/metrics" | awk '
      /^vllm:num_requests_running{/ {r += $NF}
      /^vllm:num_requests_waiting{/ {w += $NF}
      END {print r+0, w+0}
    ') || { echo "ERROR: port $PORT unavailable"; exit 1; }
    read -r RUNNING WAITING <<< "$COUNTS"
    echo "port=$PORT running=$RUNNING waiting=$WAITING"
    if (( RUNNING > 0 || WAITING > 0 )); then BUSY=1; fi
  done
  (( BUSY == 0 )) && break
  sleep 10
done

SUITE=$(cat "$ROOT/logs/latest_priority_trace_suite.path")
TRACE="$SUITE/traces/poisson_rate0.5-urgent30-seed1.jsonl"
test -f "$TRACE" || { echo "ERROR: missing trace: $TRACE"; exit 1; }

echo "Starting proportional-load replica scaling"
exec env \
  VIDEO_RLM_ROOT="$ROOT" \
  VLLM_PYTHON=/workspace/vllm-mm/bin/python \
  BASE_TRACE="$TRACE" \
  AVAILABLE_PORTS="9000 9001 9002 9003" \
  REPLICAS_LIST="1 2 4" \
  "$ROOT/run_proportional_replica_scaling.sh"
