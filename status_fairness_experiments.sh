#!/usr/bin/env bash
set -euo pipefail

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
STATE_DIR=${STATE_DIR:-$ROOT/logs/fairness_experiments_final}
PID_FILE=${PID_FILE:-$STATE_DIR/launcher.pid}
CONFIG_FILE=${CONFIG_FILE:-$STATE_DIR/launcher.config}
LOG=${LOG:-$STATE_DIR/launcher.log}

if [[ -f "$CONFIG_FILE" ]]; then
  # This file is generated locally by launch_fairness_experiments.sh using
  # shell-escaped values.
  source "$CONFIG_FILE"
else
  OUT=${OUT:-$ROOT/large_sweeps/fairness_experiments_final}
fi

if [[ -f "$PID_FILE" ]]; then
  pid=$(<"$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "State: RUNNING"
    ps -p "$pid" -o pid=,etime=,cmd=
  else
    echo "State: STOPPED (last PID $pid)"
  fi
else
  echo "State: NOT STARTED"
fi

completed=0
if [[ -d "$OUT" ]]; then
  completed=$(find "$OUT" -type f -name summary.json | wc -l)
fi
echo "Completed runs: $completed"
echo "Results: $OUT"

if [[ -f "$LOG" ]]; then
  echo "Latest launcher output:"
  tail -n 15 "$LOG"
else
  echo "Launcher log does not exist yet: $LOG"
fi

