#!/usr/bin/env bash
set -euo pipefail

# Run the VTC-style suite on two independent, already-running vLLM replicas.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
PORT0=${PORT0:-9020}
PORT1=${PORT1:-9021}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/vtc_multimodal_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_vtc_multimodal.py
mkdir -p "$OUT" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_vtc_multimodal.path"

for port in "$PORT0" "$PORT1"; do
  curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$port/v1/models" >/dev/null
done

echo "START_2SHARD $(date -u +%FT%TZ) OUT=$OUT"
env PORT="$PORT0" SHARD_INDEX=0 NUM_SHARDS=2 STAMP="$STAMP" \
  OUT="$OUT" LOGROOT="$LOGROOT" \
  "$ROOT/run_vtc_multimodal_validation.sh" \
  >"$LOGROOT/shard0-launcher.log" 2>&1 &
pid0=$!
env PORT="$PORT1" SHARD_INDEX=1 NUM_SHARDS=2 STAMP="$STAMP" \
  OUT="$OUT" LOGROOT="$LOGROOT" \
  "$ROOT/run_vtc_multimodal_validation.sh" \
  >"$LOGROOT/shard1-launcher.log" 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if (( status != 0 )); then
  echo "SHARD_FAILURE; inspect $LOGROOT/shard*-launcher.log" >&2
  exit "$status"
fi

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/video-rlm-matplotlib} \
  "$PY" "$ANALYZER" --root "$OUT" \
  --output "$OUT/vtc_multimodal_results"
echo "ALL_DONE $(date -u +%FT%TZ) OUT=$OUT"
