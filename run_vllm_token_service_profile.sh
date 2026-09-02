#!/usr/bin/env bash
set -euo pipefail

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-9000}
OUT=${OUT:-$ROOT/large_sweeps/vllm_token_service_profile}
REPEATS=${REPEATS:-3}
INPUT_LENGTHS=${INPUT_LENGTHS:-"64 256 1024"}
OUTPUT_LENGTHS=${OUTPUT_LENGTHS:-"8 32 128 256"}
VISUAL_FRAMES=${VISUAL_FRAMES:-"0 1 4 8 16"}
CONCURRENCY=${CONCURRENCY:-"1 4 8"}

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$HOST"
export no_proxy="${no_proxy:+$no_proxy,}$HOST"

PROFILE=$ROOT/conductor/experiments/scripts/run/profile_vllm_token_service.py
ANALYZE=$ROOT/conductor/experiments/scripts/analyze/analyze_vllm_token_service_profile.py
RAW=$OUT/token_service_profile.jsonl
FIGURE=$OUT/token_service_cost

mkdir -p "$OUT"
curl --noproxy '*' -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://$HOST:$PORT/v1/models" >/dev/null

"$PY" "$PROFILE" \
  --host "$HOST" \
  --port "$PORT" \
  --output "$RAW" \
  --input-lengths "$INPUT_LENGTHS" \
  --output-lengths "$OUTPUT_LENGTHS" \
  --visual-frames "$VISUAL_FRAMES" \
  --concurrency "$CONCURRENCY" \
  --repeats "$REPEATS"

"$PY" "$ANALYZE" --input "$RAW" --output "$FIGURE"
echo "VLLM_TOKEN_SERVICE_PROFILE_DONE OUT=$OUT"
