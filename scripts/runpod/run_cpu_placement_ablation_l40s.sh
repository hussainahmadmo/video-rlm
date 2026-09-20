#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/video-rlm}"
PYTHON="${PYTHON:-/workspace/venvs/vllm-mm/bin/python}"
VIDEO="${VIDEO:?Set VIDEO to the local MP4 path on the L40S pod}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the pinned Qwen2.5-VL-7B snapshot}"
GPU="${GPU:-0}"
CPUSET="${CPUSET:-0-7}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-/workspace/results/cpu_placement_ablation_l40s_${RUN_TAG}}"

test -x "${PYTHON}"
test -f "${VIDEO}"
test -d "${MODEL_PATH}"
test ! -e "${OUTPUT}"

GPU_NAME="$(nvidia-smi --id="${GPU}" --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *L40S* ]]; then
  echo "Refusing to run: GPU ${GPU} is ${GPU_NAME}, expected NVIDIA L40S." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")" "${OUTPUT}.metadata"
nvidia-smi -L > "${OUTPUT}.metadata/gpus.txt"
lscpu > "${OUTPUT}.metadata/lscpu.txt"
ffmpeg -version > "${OUTPUT}.metadata/ffmpeg.txt"
"${PYTHON}" -m pip freeze > "${OUTPUT}.metadata/python-packages.txt"
sha256sum "${VIDEO}" > "${OUTPUT}.metadata/video.sha256"

export VIDEO_RLM_ROOT="${ROOT}"
export VIDEO_RLM_MODEL_SNAPSHOT="${MODEL_PATH}"
export VIDEO_RLM_SOURCE_VIDEO="${VIDEO}"
export VIDEO_RLM_NVDEC_GPU_ID=0
export MPLCONFIGDIR="${OUTPUT}.metadata/matplotlib"

exec taskset -c "${CPUSET}" "${PYTHON}" \
  "${ROOT}/analysis/run_joint_allocation_comparison.py" \
  --output "${OUTPUT}" \
  --video "${VIDEO}" \
  --model-snapshot "${MODEL_PATH}" \
  --gpu "${GPU}" \
  --cpu-workers 4 \
  --max-cpu-workers 6 \
  --gpu-backend flashstyle_nvdec \
  --variable-frame-workload \
  --cpu-placement-ablation \
  --adaptive-all-frame-counts \
  --conservative-placement
