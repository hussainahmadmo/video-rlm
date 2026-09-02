#!/usr/bin/env bash
set -euo pipefail

# Launch one fairness phase, several named phases, or the complete publication
# experiment set in one detached process. Runs within the suite are sequential
# so scheduling policies do not contend for the same vLLM server.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PORT=${PORT:-9000}
SEEDS=${SEEDS:-"1 2 3"}
OUT=${OUT:-$ROOT/large_sweeps/fairness_experiments_final}
STATE_DIR=${STATE_DIR:-$ROOT/logs/fairness_experiments_final}
LOG=${LOG:-$STATE_DIR/launcher.log}
PID_FILE=${PID_FILE:-$STATE_DIR/launcher.pid}
CONFIG_FILE=${CONFIG_FILE:-$STATE_DIR/launcher.config}

ALL_PHASES="headline solo stage fairness constant_rate stochastic on_off work_conservation isolation load heterogeneity inflight"
PHASES=${PHASES:-${*:-$ALL_PHASES}}

valid_phase() {
  [[ " $ALL_PHASES " == *" $1 "* ]]
}

for phase in $PHASES; do
  if ! valid_phase "$phase"; then
    echo "Unknown phase: $phase" >&2
    echo "Valid phases: $ALL_PHASES" >&2
    exit 2
  fi
done

if [[ -f "$PID_FILE" ]]; then
  old_pid=$(<"$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "A fairness suite is already running (PID $old_pid)."
    echo "Check it with: ./status_fairness_experiments.sh"
    exit 1
  fi
fi

if ! curl -fsS --max-time 5 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo "vLLM is not ready on port $PORT; no experiment was started." >&2
  exit 1
fi

mkdir -p "$STATE_DIR" "$OUT"
{
  printf 'OUT=%q\n' "$OUT"
  printf 'PORT=%q\n' "$PORT"
  printf 'SEEDS=%q\n' "$SEEDS"
  printf 'PHASES=%q\n' "$PHASES"
  printf 'STARTED_AT=%q\n' "$(date -u +%FT%TZ)"
} >"$CONFIG_FILE"

nohup env \
  VIDEO_RLM_ROOT="$ROOT" \
  OUT="$OUT" \
  LOGROOT="$STATE_DIR/runs" \
  PORT="$PORT" \
  SEEDS="$SEEDS" \
  PHASES="$PHASES" \
  "$ROOT/run_complete_multimodal_fairness_suite.sh" \
  >"$LOG" 2>&1 </dev/null &

pid=$!
printf '%s\n' "$pid" >"$PID_FILE"

echo "Started fairness experiments in the background."
echo "PID: $pid"
echo "Phases: $PHASES"
echo "Seeds: $SEEDS"
echo "Results: $OUT"
echo "Log: $LOG"
echo "Status: $ROOT/status_fairness_experiments.sh"
