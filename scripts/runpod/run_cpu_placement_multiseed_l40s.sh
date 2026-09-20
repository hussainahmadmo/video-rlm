#!/usr/bin/env bash
set -euo pipefail

ROOT=${VIDEO_RLM_ROOT:-/workspace/video-rlm}
PY=${VLLM_PYTHON:-/workspace/venvs/vllm-mm/bin/python}
VIDEO=${VIDEO:-/workspace/data/kN88RP3XWUU.mp4}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the pinned Qwen2.5-VL-7B snapshot}
OUT=${OUT:-/workspace/results/cpu_placement_multiseed_l40s_20260920}
CPUSET=${CPUSET:-0-7}
SEEDS=${SEEDS:-"1 2 3"}
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_variable_frame_placement_trace.py
RUNNER=$ROOT/analysis/run_joint_allocation_comparison.py

test -x "$PY"
test -f "$VIDEO"
test -d "$MODEL_PATH"
test -f "$GENERATOR"
test -f "$RUNNER"
test ! -e "$OUT/COMPLETE"
mkdir -p "$OUT/traces" "$OUT/logs"

for seed in $SEEDS; do
  trace="$OUT/traces/seed${seed}.jsonl"
  run_out="$OUT/seed${seed}"
  log="$OUT/logs/seed${seed}.log"
  if [[ -f "$run_out/COMPLETE" ]]; then
    echo "SKIP seed=$seed"
    continue
  fi
  "$PY" "$GENERATOR" --video "$VIDEO" --output "$trace" --seed "$seed"
  [[ ! -e "$run_out" ]] || mv "$run_out" "${run_out}.partial_$(date -u +%Y%m%d_%H%M%S)"
  echo "START seed=$seed rotation=$(( (seed - 1) % 4 )) $(date -u +%FT%TZ)"
  VIDEO_RLM_ROOT="$ROOT" VIDEO_RLM_MODEL_SNAPSHOT="$MODEL_PATH" \
    VIDEO_RLM_SOURCE_VIDEO="$VIDEO" VIDEO_RLM_NVDEC_GPU_ID=0 \
    MPLCONFIGDIR="$OUT/matplotlib" \
    taskset -c "$CPUSET" "$PY" "$RUNNER" \
      --output "$run_out" --arrival-trace "$trace" --video "$VIDEO" \
      --model-snapshot "$MODEL_PATH" --gpu 0 --cpu-workers 4 \
      --max-cpu-workers 6 --gpu-backend flashstyle_nvdec \
      --cpu-placement-ablation --adaptive-all-frame-counts \
      --conservative-placement --policy-order-rotation "$(( (seed - 1) % 4 ))" \
      >"$log" 2>&1
  echo "DONE seed=$seed $(date -u +%FT%TZ)"
done

touch "$OUT/COMPLETE"
echo "CPU_PLACEMENT_MULTISEED_COMPLETE OUT=$OUT"
