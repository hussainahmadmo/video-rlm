#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/video-rlm
MODE=pilot
if [ "$#" -gt 0 ]; then
  if [ "$1" = "--full" ]; then
    MODE=full
  else
    echo "usage: $0 [--full]" >&2
    exit 2
  fi
fi

for port in 9000 9001 9002 9003; do
  curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" \
    -H 'Authorization: Bearer EMPTY' >/dev/null || {
      echo "port $port is not ready" >&2
      exit 1
    }
done

stamp=$(date +%Y%m%d_%H%M%S)
suite="$ROOT/conductor/experiments/large_sweeps/priority_trace_parallel_$MODE-$stamp"
logroot="$ROOT/logs/priority_trace_parallel_$MODE-$stamp"
mkdir -p "$suite" "$logroot"
printf '%s\n' "$suite" > "$ROOT/logs/latest_priority_trace_suite.path"
: > "$logroot/worker_pids.txt"

for shard in 0 1 2 3; do
  port=$((9000 + shard))
  mode_arg=
  [ "$MODE" = full ] && mode_arg=--full

  nohup "$ROOT/run_priority_trace_suite.sh" $mode_arg \
    --port "$port" \
    --shard-index "$shard" \
    --num-shards 4 \
    --suite "$suite" \
    --log-root "$logroot" \
    > "$logroot/shard$shard-launcher.log" 2>&1 &

  pid=$!
  printf '%s\t%s\t%s\n' "$shard" "$port" "$pid" | tee -a "$logroot/worker_pids.txt"
done

echo "MODE=$MODE"
echo "SUITE=$suite"
echo "LOGROOT=$logroot"
echo "Monitor:"
echo "tail -f $logroot/shard*-launcher.log"
