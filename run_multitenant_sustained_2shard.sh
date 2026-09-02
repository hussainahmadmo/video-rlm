#!/usr/bin/env bash
set -euo pipefail

# Start the sustained multi-tenant validation on two already-running vLLM
# replicas. This parent waits for both shards and regenerates the final report.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
PORT0=${PORT0:-9020}
PORT1=${PORT1:-9021}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/multitenant_sustained_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_multitenant_scheduling.py
PLOTTER=$ROOT/conductor/experiments/scripts/analyze/plot_multitenant_scheduling.py
mkdir -p "$OUT" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_multitenant_sustained.path"

for port in "$PORT0" "$PORT1"; do
  curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$port/v1/models" >/dev/null
done

echo "START_2SHARD $(date -u +%FT%TZ) OUT=$OUT"
env PORT="$PORT0" SHARD_INDEX=0 NUM_SHARDS=2 STAMP="$STAMP" \
  OUT="$OUT" LOGROOT="$LOGROOT" \
  "$ROOT/run_multitenant_sustained_validation.sh" \
  >"$LOGROOT/shard0-launcher.log" 2>&1 &
pid0=$!
env PORT="$PORT1" SHARD_INDEX=1 NUM_SHARDS=2 STAMP="$STAMP" \
  OUT="$OUT" LOGROOT="$LOGROOT" \
  "$ROOT/run_multitenant_sustained_validation.sh" \
  >"$LOGROOT/shard1-launcher.log" 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if (( status != 0 )); then
  echo "SHARD_FAILURE; inspect $LOGROOT/shard*-launcher.log" >&2
  exit "$status"
fi

"$PY" "$ANALYZER" --root "$OUT" \
  --output "$OUT/multitenant_sustained_results" \
  --urgent-e2e-slo-s 60 --background-e2e-slo-s 300
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/video-rlm-matplotlib} \
  "$PY" "$PLOTTER" --root "$OUT" \
  --output "$OUT/figures/multitenant_sustained_tradeoff"
echo "ALL_DONE $(date -u +%FT%TZ) OUT=$OUT"
