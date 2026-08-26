#!/usr/bin/env bash
set -euo pipefail

# Run the remaining self-contained ASPLOS evaluation on two vLLM replicas.
# Phases are deliberately serialized so one experiment cannot contaminate
# another, while stress and fairness conditions are sharded across both GPUs.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
TRACE_ROOT=${TRACE_ROOT:-$ROOT/conductor/experiments/large_sweeps/nature_priority_pilot/traces}
PRIORITY_DATASET=${PRIORITY_DATASET:-$ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl}
PORT0=${PORT0:-9000}
PORT1=${PORT1:-9001}
STAMP=${ASPLOS_EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${ASPLOS_EVAL_OUT:-$ROOT/large_sweeps/asplos_remaining_2gpu_$STAMP}
LOGROOT=${ASPLOS_EVAL_LOGROOT:-$ROOT/logs/$(basename "$OUT")}

mkdir -p "$OUT" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_asplos_remaining_2gpu.path"

for required in \
  "$PY" \
  "$PRIORITY_DATASET" \
  "$ROOT/run_repeated_backlog_stress.sh" \
  "$ROOT/run_fairness_aging_validation.sh" \
  "$ROOT/run_static_isolation_baseline.sh"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done
[[ -d "$TRACE_ROOT" ]] || { echo "missing trace directory: $TRACE_ROOT" >&2; exit 1; }

wait_for_servers() {
  while true; do
    local ready=0 port
    for port in "$PORT0" "$PORT1"; do
      if curl -fsS --max-time 5 -H 'Authorization: Bearer EMPTY' \
        "http://127.0.0.1:$port/v1/models" >/dev/null; then
        ready=$((ready + 1))
      fi
    done
    if (( ready == 2 )); then
      echo "SERVERS_READY ports=$PORT0,$PORT1 $(date -u +%FT%TZ)"
      return
    fi
    echo "WAITING_FOR_SERVERS ready=$ready/2 ports=$PORT0,$PORT1 $(date -u +%FT%TZ)"
    sleep 30
  done
}

run_two_shards() {
  local phase=$1 launcher=$2 output=$3 phase_logs=$4
  mkdir -p "$output" "$phase_logs"
  echo "START_PHASE $phase $(date -u +%FT%TZ)"

  local -a output_env
  case "$phase" in
    stress)
      output_env=(
        "BACKLOG_STRESS_OUT=$output"
        "BACKLOG_STRESS_LOGROOT=$phase_logs"
      )
      ;;
    fairness)
      output_env=(
        "FAIRNESS_OUT=$output"
        "FAIRNESS_LOGROOT=$phase_logs"
      )
      ;;
    *)
      echo "unsupported sharded phase: $phase" >&2
      return 2
      ;;
  esac

  env "${output_env[@]}" VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" \
  TRACE_ROOT="$TRACE_ROOT" PRIORITY_DATASET="$PRIORITY_DATASET" \
  PORT="$PORT0" SHARD_INDEX=0 NUM_SHARDS=2 \
  "$launcher" >"$phase_logs/shard0-launcher.log" 2>&1 &
  local pid0=$!

  env "${output_env[@]}" VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" \
  TRACE_ROOT="$TRACE_ROOT" PRIORITY_DATASET="$PRIORITY_DATASET" \
  PORT="$PORT1" SHARD_INDEX=1 NUM_SHARDS=2 \
  "$launcher" >"$phase_logs/shard1-launcher.log" 2>&1 &
  local pid1=$!

  local status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  if (( status != 0 )); then
    echo "FAILED_PHASE $phase status=$status $(date -u +%FT%TZ)" >&2
    return "$status"
  fi
  echo "DONE_PHASE $phase $(date -u +%FT%TZ)"
}

wait_for_servers

run_two_shards stress "$ROOT/run_repeated_backlog_stress.sh" \
  "$OUT/repeated_backlog_stress" "$LOGROOT/repeated_backlog_stress"

wait_for_servers
run_two_shards fairness "$ROOT/run_fairness_aging_validation.sh" \
  "$OUT/fairness_aging" "$LOGROOT/fairness_aging"

wait_for_servers
echo "START_PHASE static_isolation $(date -u +%FT%TZ)"
VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" TRACE_ROOT="$TRACE_ROOT" \
PORTS="$PORT0 $PORT1" BACKGROUND_PORTS="$PORT0" URGENT_PORTS="$PORT1" \
STATIC_ISOLATION_OUT="$OUT/static_isolation" \
STATIC_ISOLATION_LOGROOT="$LOGROOT/static_isolation" \
  "$ROOT/run_static_isolation_baseline.sh" \
  >"$LOGROOT/static_isolation-launcher.log" 2>&1
echo "DONE_PHASE static_isolation $(date -u +%FT%TZ)"

"$PY" "$ROOT/conductor/experiments/scripts/analyze/analyze_paper_evaluation.py" \
  --root "$OUT" --output "$OUT/paper_evaluation_report" \
  >"$LOGROOT/analyze.log" 2>&1 || true

echo "ALL_DONE OUT=$OUT LOGROOT=$LOGROOT $(date -u +%FT%TZ)"
