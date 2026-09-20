#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/video-rlm}"
PYTHON="${PYTHON:-/workspace/venvs/vllm-mm/bin/python}"
VIDEO="${VIDEO:?Set VIDEO to the local MP4 path on the L40S pod}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the pinned Qwen2.5-VL-7B snapshot}"
GPU="${GPU:-0}"
CPUSET="${CPUSET:-0-7}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results/adaptive_placement_load_sweep_l40s_${RUN_TAG}}"
TRACE_DIR="${OUTPUT_ROOT}/traces"

test -x "${PYTHON}"
test -f "${VIDEO}"
test -d "${MODEL_PATH}"
test ! -e "${OUTPUT_ROOT}"

GPU_NAME="$(nvidia-smi --id="${GPU}" --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *L40S* ]]; then
  echo "Refusing to run: GPU ${GPU} is ${GPU_NAME}, expected NVIDIA L40S." >&2
  exit 2
fi

mkdir -p "${TRACE_DIR}" "${OUTPUT_ROOT}/metadata"
nvidia-smi -L > "${OUTPUT_ROOT}/metadata/gpus.txt"
lscpu > "${OUTPUT_ROOT}/metadata/lscpu.txt"
ffmpeg -version > "${OUTPUT_ROOT}/metadata/ffmpeg.txt"
"${PYTHON}" -m pip freeze > "${OUTPUT_ROOT}/metadata/python-packages.txt"
sha256sum "${VIDEO}" > "${OUTPUT_ROOT}/metadata/video.sha256"
"${PYTHON}" "${ROOT}/analysis/make_adaptive_placement_load_traces.py" \
  --output-dir "${TRACE_DIR}"

export VIDEO_RLM_ROOT="${ROOT}"
export VIDEO_RLM_MODEL_SNAPSHOT="${MODEL_PATH}"
export VIDEO_RLM_SOURCE_VIDEO="${VIDEO}"
export VIDEO_RLM_NVDEC_GPU_ID=0
export MPLCONFIGDIR="${OUTPUT_ROOT}/metadata/matplotlib"

for workload in low_load burst; do
  taskset -c "${CPUSET}" "${PYTHON}" \
    "${ROOT}/analysis/run_joint_allocation_comparison.py" \
    --output "${OUTPUT_ROOT}/${workload}" \
    --video "${VIDEO}" \
    --model-snapshot "${MODEL_PATH}" \
    --arrival-trace "${TRACE_DIR}/${workload}.jsonl" \
    --gpu "${GPU}" \
    --cpu-workers 4 \
    --max-cpu-workers 6 \
    --gpu-backend flashstyle_nvdec \
    --cpu-placement-ablation \
    --adaptive-all-frame-counts \
    --conservative-placement
done

printf 'Completed low-load and burst placement comparisons.\n' > "${OUTPUT_ROOT}/COMPLETE"
