#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-mm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
INSTALL_YOLO="${INSTALL_YOLO:-0}"
INSTALL_ZSH="${INSTALL_ZSH:-1}"
INSTALL_ZSH_AUTOCOMPLETE="${INSTALL_ZSH_AUTOCOMPLETE:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="${REPO_ROOT}/scripts/runpod/requirements-runpod.txt"

echo "[runpod] repo: ${REPO_ROOT}"
echo "[runpod] env: ${ENV_NAME}"

if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    echo "[runpod] installing system packages"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      ca-certificates \
      cmake \
      ffmpeg \
      git \
      git-lfs \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      ninja-build \
      pkg-config \
      zsh
    git lfs install || true
  else
    echo "[runpod] non-root user; skipping apt package install"
  fi
else
  echo "[runpod] apt-get not found; skipping system package install"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[runpod] creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${ENV_NAME}"
else
  echo "[runpod] conda not found; using current Python environment"
fi

python -m pip install --upgrade pip setuptools wheel

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  echo "[runpod] PyTorch is not installed. Install a CUDA PyTorch build for your RunPod image, then rerun this script."
  echo "[runpod] Recommended base image: RunPod PyTorch image with CUDA already installed."
  exit 1
fi

echo "[runpod] installing Python requirements"
python -m pip install -r "${REQ_FILE}"

if [ -d "${REPO_ROOT}/decord/python" ]; then
  echo "[runpod] installing local decord package in editable mode"
  python -m pip install -e "${REPO_ROOT}/decord/python" || {
    echo "[runpod] local decord editable install failed; the pip decord package remains installed"
  }
fi

if [ "${INSTALL_VLLM}" = "1" ]; then
  echo "[runpod] installing vLLM"
  python -m pip install vllm
else
  echo "[runpod] skipping vLLM install because INSTALL_VLLM=${INSTALL_VLLM}"
fi

if [ "${INSTALL_YOLO}" = "1" ]; then
  echo "[runpod] installing optional YOLO dependencies"
  python -m pip install ultralytics
else
  echo "[runpod] skipping YOLO deps because INSTALL_YOLO=${INSTALL_YOLO}"
fi

if [ "${INSTALL_ZSH}" = "1" ]; then
  echo "[runpod] installing zsh setup"
  mkdir -p "${HOME}/.zsh"
  if [ -f "${HOME}/.zshrc" ] && ! grep -q "VIMIO / video-rlm RunPod zsh setup" "${HOME}/.zshrc"; then
    cp "${HOME}/.zshrc" "${HOME}/.zshrc.before-vimio"
  fi
  cp "${REPO_ROOT}/scripts/runpod/runpod.zshrc" "${HOME}/.zshrc"

  if [ "${INSTALL_ZSH_AUTOCOMPLETE}" = "1" ]; then
    if [ ! -d "${HOME}/.zsh/zsh-autocomplete" ]; then
      git clone --depth 1 https://github.com/marlonrichert/zsh-autocomplete.git "${HOME}/.zsh/zsh-autocomplete" || {
        echo "[runpod] zsh-autocomplete clone failed; continuing without it"
      }
    fi
  fi
else
  echo "[runpod] skipping zsh setup because INSTALL_ZSH=${INSTALL_ZSH}"
fi

python - <<'PY'
import importlib.util
required = [
    "numpy",
    "openai",
    "PIL",
    "requests",
    "torch",
    "transformers",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing required packages: {missing}")
print("[runpod] required Python imports OK")
PY

cat <<EOF

[runpod] install complete.

To use this environment later:
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}
  cd ${REPO_ROOT}

To use zsh:
  zsh

To add secrets without committing them:
  printf 'export HF_TOKEN=...\nexport WANDB_API_KEY=...\n' > ~/.vimio_secrets
  chmod 600 ~/.vimio_secrets

Verify caption caches:
  find conductor/experiments/self_improving/data/video_agent_caption_cache* -type f | wc -l
  du -sh conductor/experiments/self_improving/data/video_agent_caption_cache*

Start vLLM servers, for example:
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9000 --model Qwen/Qwen2.5-VL-7B-Instruct
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9001 --model Qwen/Qwen2.5-VL-7B-Instruct

EOF
