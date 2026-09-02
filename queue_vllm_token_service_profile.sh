#!/usr/bin/env bash
set -euo pipefail

WAIT_PID=${WAIT_PID:-}
POLL_S=${POLL_S:-30}
ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}

if [[ -n "$WAIT_PID" ]]; then
  echo "WAITING_FOR_PID pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_S"
  done
fi

# Avoid overlap if the original launcher exited just before a child runner.
while pgrep -f '[r]un_mixed_end_to_end_priority.py' >/dev/null; do
  echo "WAITING_FOR_ACTIVE_FAIRNESS_CASE"
  sleep "$POLL_S"
done

cd "$ROOT"
exec ./run_vllm_token_service_profile.sh
