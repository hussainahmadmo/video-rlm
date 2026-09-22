#!/usr/bin/env bash
set -euo pipefail

ROOT=${VIDEO_RLM_ROOT:-/workspace/video-rlm-joint-controller}
PY=${VLLM_PYTHON:-/workspace/venv/bin/python}
TRACE_ROOT=${TRACE_ROOT:-$ROOT/data/azure_peak_segments}
LANE_PROFILE=${LANE_PROFILE:-$TRACE_ROOT/lane_profile_interpolated.json}
PORT=${PORT:-8000}
CPUSET=${CPUSET:-0-7}
OUT_ROOT=${OUT_ROOT:-/workspace/results/azure_peak_segments_$(date -u +%Y%m%d_%H%M%S)}
LAUNCHER=$ROOT/scripts/runpod/run_end_to_end_components_l40s.sh

forward="full_fcfs full_tenant_round_robin full_prep_sjf full_prep_sjf_aging full_conductor_strict_fifo"
reverse="full_conductor_strict_fifo full_prep_sjf_aging full_prep_sjf full_tenant_round_robin full_fcfs"
rotate="full_prep_sjf full_conductor_strict_fifo full_fcfs full_prep_sjf_aging full_tenant_round_robin"
orders=("$forward" "$reverse" "$rotate")

for required in "$PY" "$LANE_PROFILE" "$LAUNCHER" "$TRACE_ROOT/manifest.json"; do
  test -e "$required" || { echo "missing: $required" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
{
  echo "design=azure_peak_fixed_time_segments"
  echo "source_manifest=$TRACE_ROOT/manifest.json"
  echo "segment_duration_s=100"
  echo "placement=joint_predicted_readiness"
  echo "capacity=fixed"
  echo "profile=piecewise_linear_from_1_16_128_calibration"
  echo "policies=fcfs tenant_round_robin prep_sjf prep_sjf_aging conductor_strict_fifo"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT_ROOT/manifest.txt"

for segment in 0 1 2; do
  trace=$TRACE_ROOT/segment${segment}.jsonl
  test -s "$trace" || { echo "missing trace: $trace" >&2; exit 2; }
  variants=${orders[$segment]}
  priority=${variants%% *}
  output=$OUT_ROOT/segment${segment}
  echo "START segment=$segment requests=$(wc -l <"$trace") output=$output"
  env VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" PORT="$PORT" CPUSET="$CPUSET" \
    LANE_PROFILE="$LANE_PROFILE" SOURCE_TRACE="$trace" SEEDS="$segment" \
    VARIANTS="$variants" PRIORITY_VARIANT="$priority" \
    FRAME_COUNTS="trace-derived-1-to-128" OUT="$output" bash "$LAUNCHER"
  echo "DONE segment=$segment"
done

echo "completed=$(date -u +%FT%TZ)" >>"$OUT_ROOT/manifest.txt"
touch "$OUT_ROOT/COMPLETE"
echo "AZURE_PEAK_SEGMENTS_COMPLETE output=$OUT_ROOT"
