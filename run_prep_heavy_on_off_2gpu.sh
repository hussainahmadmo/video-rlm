#!/usr/bin/env bash
set -euo pipefail

# Matched preparation-heavy ON/OFF validation on two independent replicas.
# Worker points run sequentially so their CPU preparation pools cannot
# interfere with one another on the shared host.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
PORT0=${PORT0:-9020}
PORT1=${PORT1:-9021}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/prep_heavy_on_off_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_vtc_multimodal.py
mkdir -p "$OUT" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_prep_heavy_on_off.path"

for port in "$PORT0" "$PORT1"; do
  curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$port/v1/models" >/dev/null
done

echo "START_2GPU $(date -u +%FT%TZ) OUT=$OUT"
env PORT="$PORT0" PREP_WORKERS=2 STAMP="$STAMP" OUT="$OUT" LOGROOT="$LOGROOT" \
  "$ROOT/run_prep_heavy_on_off_worker.sh" \
  >"$LOGROOT/workers2-launcher.log" 2>&1
env PORT="$PORT1" PREP_WORKERS=4 STAMP="$STAMP" OUT="$OUT" LOGROOT="$LOGROOT" \
  "$ROOT/run_prep_heavy_on_off_worker.sh" \
  >"$LOGROOT/workers4-launcher.log" 2>&1

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/video-rlm-matplotlib} \
  "$PY" "$ANALYZER" --root "$OUT" \
  --output "$OUT/prep_heavy_on_off_results"
echo "ALL_DONE $(date -u +%FT%TZ) OUT=$OUT"
