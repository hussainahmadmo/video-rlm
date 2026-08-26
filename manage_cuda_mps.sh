#!/usr/bin/env bash
set -euo pipefail

# Manage an isolated CUDA MPS control daemon for one physical GPU. The printed
# environment variables must also be present when launching vLLM/NVDEC.

ACTION=${1:-status}
GPU=${GPU:-0}
ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
MPS_ROOT=${MPS_ROOT:-$ROOT/logs/cuda_mps/gpu$GPU}
PIPE_DIR=${CUDA_MPS_PIPE_DIRECTORY:-$MPS_ROOT/pipe}
MPS_LOG_DIR=${CUDA_MPS_LOG_DIRECTORY:-$MPS_ROOT/log}

command -v nvidia-cuda-mps-control >/dev/null || {
    echo "nvidia-cuda-mps-control is not installed" >&2
    exit 1
}

print_environment() {
    printf 'export CUDA_MPS_PIPE_DIRECTORY=%q\n' "$PIPE_DIR"
    printf 'export CUDA_MPS_LOG_DIRECTORY=%q\n' "$MPS_LOG_DIR"
}

case "$ACTION" in
    start)
        mkdir -p "$PIPE_DIR" "$MPS_LOG_DIR"
        export CUDA_VISIBLE_DEVICES=$GPU
        export CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR
        export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR
        if printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null 2>&1; then
            echo "CUDA MPS is already running for GPU $GPU"
        else
            nvidia-cuda-mps-control -d
            sleep 1
            printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null
            echo "CUDA MPS started for GPU $GPU"
        fi
        print_environment
        ;;
    status)
        export CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR
        export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR
        if printf 'get_server_list\n' | nvidia-cuda-mps-control; then
            echo "CUDA MPS is reachable for GPU $GPU"
            print_environment
        else
            echo "CUDA MPS is not running for GPU $GPU" >&2
            exit 1
        fi
        ;;
    stop)
        export CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR
        export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR
        printf 'quit\n' | nvidia-cuda-mps-control
        echo "CUDA MPS stopped for GPU $GPU"
        ;;
    *)
        echo "Usage: GPU=<index> $0 {start|status|stop}" >&2
        exit 2
        ;;
esac
