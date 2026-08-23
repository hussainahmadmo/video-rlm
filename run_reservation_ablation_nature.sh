#!/usr/bin/env bash
set -u

ROOT=/dataheart/hussainahmad/video-rlm
PY=/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"

OUT=$(cat "$ROOT/logs/latest_reservation_ablation.path")
TRACE_ROOT="$OUT/traces"
LOGROOT="${OUT/\/large_sweeps\//\/logs\/}"

mkdir -p "$LOGROOT"

run_shard() {
  local SHARD=$1
  local PORT=$2
  local INDEX=0

  for SEED in 1 2 3; do
    for BASE in \
      burst_urgent10 \
      staggered_urgent10 \
      poisson_rate0.5-urgent30 \
      poisson_rate1.0-urgent10
    do
      for LIMIT in 4 3 2 1; do
        if [ $((INDEX % 2)) -eq "$SHARD" ]; then
          NAME="${BASE}-seed${SEED}-limit${LIMIT}"
          TRACE="$TRACE_ROOT/${BASE}-seed${SEED}.jsonl"
          RUN_OUT="$OUT/$NAME"
          LOG="$LOGROOT/$NAME.log"

          echo "START $NAME port=$PORT $(date -u +%FT%TZ)"

          "$PY" "$RUNNER" \
            --arrival-trace "$TRACE" \
            --output "$RUN_OUT" \
            --port "$PORT" \
            --prep-policy priority_reserved \
            --background-prep-limit "$LIMIT" \
            --prep-workers 4 \
            --vlm-concurrency 4 \
            --prepared-queue-depth 32 \
            --request-timeout-s 1800 \
            < /dev/null > "$LOG" 2>&1

          STATUS=$?
          echo "DONE $NAME status=$STATUS $(date -u +%FT%TZ)"
        fi

        INDEX=$((INDEX + 1))
      done
    done
  done

  echo "SHARD_DONE shard=$SHARD port=$PORT"
}

run_shard 0 9000 > "$LOGROOT/shard0-launcher.log" 2>&1 &
PID0=$!

run_shard 1 9001 > "$LOGROOT/shard1-launcher.log" 2>&1 &
PID1=$!

echo "Shard 0 PID=$PID0"
echo "Shard 1 PID=$PID1"

wait "$PID0"
STATUS0=$?

wait "$PID1"
STATUS1=$?

echo "ALL DONE shard0=$STATUS0 shard1=$STATUS1"
