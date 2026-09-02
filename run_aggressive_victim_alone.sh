#!/usr/bin/env bash
set -euo pipefail

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
SOURCE=${SOURCE:-$ROOT/large_sweeps/aggressive_asymmetric_cross_stage_final}
OUT=${OUT:-$ROOT/large_sweeps/aggressive_asymmetric_victim_alone}
LOGROOT=${LOGROOT:-$ROOT/logs/aggressive_asymmetric_victim_alone}
PORT=${PORT:-9000}
SEEDS=${SEEDS:-"1 2 3"}
TENANTS=${TENANTS:-"b c"}

FILTER=$ROOT/conductor/experiments/scripts/run/filter_trace_by_tenant.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py

mkdir -p "$OUT/traces" "$LOGROOT"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

for seed in $SEEDS; do
  source_trace="$SOURCE/traces/aggressive_asymmetric-seed${seed}.jsonl"
  test -f "$source_trace"
  for tenant in $TENANTS; do
    name="aggressive_asymmetric-victim-${tenant}-solo-seed${seed}"
    trace="$OUT/traces/$name.jsonl"
    "$PY" "$FILTER" --input "$source_trace" --output "$trace" \
      --tenant "$tenant"
    output="$OUT/$name/fcfs"
    log="$LOGROOT/$name-fcfs.log"
    if [[ -f "$output/summary.json" ]]; then
      echo "SKIP tenant=$tenant seed=$seed policy=fcfs"
      continue
    fi
    if [[ -d "$output" ]]; then
      mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
    fi
    echo "START tenant=$tenant seed=$seed policy=fcfs $(date -u +%FT%TZ)"
    VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$output" --port "$PORT" \
      --prep-policy fcfs \
      --tenant-weight a=1 --tenant-weight b=1 --tenant-weight c=1 \
      --decode-backend seek_cpu --prep-workers 4 --vlm-concurrency 4 \
      --prepared-queue-depth 16 \
      --decode-timeout-s 600 --request-timeout-s 1800 \
      >"$log" 2>&1
    echo "DONE tenant=$tenant seed=$seed policy=fcfs $(date -u +%FT%TZ)"
  done
done

echo "VICTIM_ALONE_DONE OUT=$OUT"
