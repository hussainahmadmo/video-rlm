#!/usr/bin/env bash
set -euo pipefail

ROOT=${VIDEO_RLM_ROOT:-/workspace/video-rlm}
PY=${VLLM_PYTHON:-/workspace/venvs/vllm-mm/bin/python}
PORT=${PORT:-9000}
VIDEO=${VIDEO:-/workspace/data/kN88RP3XWUU.mp4}
OUT=${OUT:-/workspace/results/service_allocation_l40s}
LOGROOT=${LOGROOT:-$OUT/logs}
SEEDS=${SEEDS:-"1 2 3"}
POLICIES=${POLICIES:-"fcfs prep_max_min engine_tenant_fair max_min"}
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_service_allocation_trace.py

mkdir -p "$OUT/traces" "$LOGROOT"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

for seed in $SEEDS; do
  trace="$OUT/traces/seed${seed}.jsonl"
  [[ -f "$trace" ]] || "$PY" "$GENERATOR" \
    --video "$VIDEO" --output "$trace" --seed "$seed" --workload preparation

  # Rotate policy order across seeds to reduce fixed-order thermal bias.
  case "$seed" in
    1) ordered="fcfs prep_max_min engine_tenant_fair max_min" ;;
    2) ordered="prep_max_min engine_tenant_fair max_min fcfs" ;;
    *) ordered="engine_tenant_fair max_min fcfs prep_max_min" ;;
  esac
  for policy in $ordered; do
    [[ " $POLICIES " == *" $policy "* ]] || continue
    run_out="$OUT/seed${seed}/$policy"
    log="$LOGROOT/seed${seed}-${policy}.log"
    if [[ -f "$run_out/summary.json" ]] \
        && jq -e '.total_requests == 54 and .errors == 0' "$run_out/summary.json" >/dev/null; then
      echo "SKIP seed=$seed policy=$policy"
      continue
    fi
    [[ ! -d "$run_out" ]] || mv "$run_out" "${run_out}.partial_$(date -u +%Y%m%d_%H%M%S)"
    echo "START seed=$seed policy=$policy $(date -u +%FT%TZ)"
    VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$run_out" --port "$PORT" \
      --prep-policy "$policy" --tenant-weight a=1 --tenant-weight b=1 \
      --tenant-weight c=1 --decode-backend seek_cpu --cpu-decoder-threads 1 \
      --prep-workers 4 --vlm-concurrency 4 --prepared-queue-depth 16 \
      --ignore-eos --decode-timeout-s 600 --request-timeout-s 1800 \
      >"$log" 2>&1
    echo "DONE seed=$seed policy=$policy $(date -u +%FT%TZ)"
  done
done

echo "SERVICE_ALLOCATION_COMPLETE OUT=$OUT"
