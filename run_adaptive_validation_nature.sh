#!/usr/bin/env bash
set -u

ROOT=/dataheart/hussainahmad/video-rlm
PY=/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python
RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"

PREVIOUS_OUT=$(cat "$ROOT/logs/latest_reservation_ablation.path")
TRACE_ROOT="$PREVIOUS_OUT/traces"
OUT=$(cat "$ROOT/logs/latest_adaptive_validation.path")
LOGROOT="$ROOT/logs/$(basename "$OUT")"

mkdir -p "$OUT" "$LOGROOT"

run_shard() {
  local SHARD=$1
  local PORT=$2
  local TRACE_INDEX=0

  for SEED in 1 2 3; do
    for BASE in \
      burst_urgent10 \
      staggered_urgent10 \
      poisson_rate0.5-urgent30 \
      poisson_rate1.0-urgent10
    do
      if [ $((TRACE_INDEX % 2)) -eq "$SHARD" ]; then
        TRACE="$TRACE_ROOT/${BASE}-seed${SEED}.jsonl"

        for POLICY in priority priority_reserved slo_adaptive; do
          NAME="${BASE}-seed${SEED}-${POLICY}"
          RUN_OUT="$OUT/$NAME"
          LOG="$LOGROOT/$NAME.log"

          POLICY_ARGS=()

          if [ "$POLICY" = "priority_reserved" ]; then
            POLICY_ARGS=(
              --background-prep-limit 3
            )
          elif [ "$POLICY" = "slo_adaptive" ]; then
            POLICY_ARGS=(
              --urgent-prep-reserve 1
              --urgent-ttft-slo-s 30
              --background-ttft-slo-s 300
              --background-aging-s 120
              --prep-fixed-cost-s 0.25
              --prep-seconds-per-frame 0.12
              --prep-cost-ewma-alpha 0.2
            )
          fi

          echo "START $NAME port=$PORT $(date -u +%FT%TZ)"

          "$PY" "$RUNNER" \
            --arrival-trace "$TRACE" \
            --output "$RUN_OUT" \
            --port "$PORT" \
            --prep-policy "$POLICY" \
            "${POLICY_ARGS[@]}" \
            --prep-workers 4 \
            --vlm-concurrency 4 \
            --prepared-queue-depth 32 \
            --request-timeout-s 1800 \
            < /dev/null > "$LOG" 2>&1

          STATUS=$?
          echo "DONE $NAME status=$STATUS $(date -u +%FT%TZ)"
        done
      fi

      TRACE_INDEX=$((TRACE_INDEX + 1))
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
echo "OUT=$OUT"

wait "$PID0"
STATUS0=$?

wait "$PID1"
STATUS1=$?

SUMMARIES=$(find "$OUT" -type f -name summary.json | wc -l)

echo \
  "ADAPTIVE_VALIDATION_DONE summaries=$SUMMARIES/36 shard0=$STATUS0 shard1=$STATUS1"
