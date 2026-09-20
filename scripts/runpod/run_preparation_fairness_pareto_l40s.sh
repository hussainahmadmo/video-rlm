#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEO_RLM_ROOT:-/workspace/video-rlm}"
PYTHON="${VLLM_PYTHON:-/workspace/venvs/vllm-mm/bin/python}"
OUTPUT="${OUTPUT:?set OUTPUT to a fresh result directory}"
VIDEO="${VIDEO:-/workspace/data/kN88RP3XWUU.mp4}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the local Qwen2.5-VL-7B snapshot}"
CPUSET="${CPUSET:?set CPUSET to CPUs local to this pod GPU}"
SEEDS="${SEEDS:-0 1 2}"
GPU="${GPU:-0}"

test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}"
cp "$0" "${OUTPUT}/launcher.sh"

for seed in ${SEEDS}; do
  trace="${OUTPUT}/seed${seed}_trace.jsonl"
  run="${OUTPUT}/seed${seed}"
  "${PYTHON}" "${ROOT}/conductor/experiments/scripts/run/make_heterogeneous_noisy_neighbor_trace.py" \
    --output "${trace}" --video "${VIDEO}" --seed "${seed}"
  taskset -c "${CPUSET}" "${PYTHON}" "${ROOT}/analysis/run_joint_allocation_comparison.py" \
    --output "${run}" \
    --arrival-trace "${trace}" \
    --fairness-slack-ablation \
    --fairness-slacks 0 1 2 5 10 20 \
    --adaptive-all-frame-counts \
    --conservative-placement \
    --cpu-workers 4 \
    --max-cpu-workers 6 \
    --gpu-backend flashstyle_nvdec \
    --gpu-lane-budget 4 \
    --gpu-prep-jobs 2 \
    --gpu-widths 1 2 4 \
    --gpu "${GPU}" \
    --video "${VIDEO}" \
    --model-snapshot "${MODEL_PATH}" \
    --policy-order-rotation "$((seed % 7))"
done

touch "${OUTPUT}/COMPLETE"
