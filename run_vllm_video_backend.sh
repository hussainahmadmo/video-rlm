#!/usr/bin/env bash
set -euo pipefail

# Start one vLLM server with a selectable native video-decoding backend.
# Use the same vLLM environment and server settings for CPU-vs-NVDEC tests so
# that the decoding backend is the controlled variable.

VLLM_BIN=${VLLM_BIN:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm-nvdec/bin/vllm}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
VIDEO_BACKEND=${VIDEO_BACKEND:-opencv}
GPU=${GPU:-0}
PORT=${PORT:-9000}
MEDIA_LOADING_THREADS=${MEDIA_LOADING_THREADS:-8}
NUM_FFMPEG_THREADS=${NUM_FFMPEG_THREADS:-1}
TORCHCODEC_SEEK_MODE=${TORCHCODEC_SEEK_MODE:-exact}
MM_IPC_GPU_MEMORY_GB=${MM_IPC_GPU_MEMORY_GB:-3}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
VIDEO_FETCH_TIMEOUT=${VIDEO_FETCH_TIMEOUT:-1800}
LOG_DIR=${LOG_DIR:-/dataheart/hussainahmad/video-rlm/logs/vllm_video_backends}
MPS_ROOT=${MPS_ROOT:-/dataheart/hussainahmad/video-rlm/logs/cuda_mps/gpu$GPU}

test -x "$VLLM_BIN" || {
    echo "vLLM executable not found: $VLLM_BIN" >&2
    exit 1
}
[[ $MEDIA_LOADING_THREADS =~ ^[1-9][0-9]*$ ]] || {
    echo "MEDIA_LOADING_THREADS must be a positive integer" >&2
    exit 1
}

case "$VIDEO_BACKEND" in
    opencv|pyav)
        MEDIA_IO_KWARGS=$(printf '{"video":{"backend":"%s"}}' "$VIDEO_BACKEND")
        ;;
    torchcodec)
        [[ $NUM_FFMPEG_THREADS =~ ^[0-9]+$ ]] || {
            echo "NUM_FFMPEG_THREADS must be a non-negative integer" >&2
            exit 1
        }
        case "$TORCHCODEC_SEEK_MODE" in
            exact|approximate) ;;
            *) echo "TORCHCODEC_SEEK_MODE must be exact or approximate" >&2; exit 1 ;;
        esac
        MEDIA_IO_KWARGS=$(printf \
            '{"video":{"backend":"torchcodec","num_ffmpeg_threads":%s,"seek_mode":"%s"}}' \
            "$NUM_FFMPEG_THREADS" "$TORCHCODEC_SEEK_MODE")
        ;;
    pynvvideocodec)
        # PyNvVideoCodec runs in the API process while EngineCore runs in a
        # separate CUDA process, so CUDA MPS must already be active.
        command -v nvidia-cuda-mps-control >/dev/null || {
            echo "nvidia-cuda-mps-control is required for pynvvideocodec" >&2
            exit 1
        }
        export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-$MPS_ROOT/pipe}
        export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-$MPS_ROOT/log}
        printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || {
            echo "CUDA MPS is not reachable through $CUDA_MPS_PIPE_DIRECTORY" >&2
            echo "Run: GPU=$GPU ./manage_cuda_mps.sh start" >&2
            exit 1
        }
        awk -v value="$MM_IPC_GPU_MEMORY_GB" \
            'BEGIN { exit !(value + 0 > 0) }' || {
            echo "MM_IPC_GPU_MEMORY_GB must be greater than zero for NVDEC" >&2
            exit 1
        }
        MEDIA_IO_KWARGS='{"video":{"backend":"pynvvideocodec"}}'
        ;;
    *)
        echo "Unsupported VIDEO_BACKEND=$VIDEO_BACKEND" >&2
        echo "Choose: opencv, pyav, torchcodec, or pynvvideocodec" >&2
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-$LOG_DIR/vllm-$PORT-$VIDEO_BACKEND.log}

echo "vllm=$VLLM_BIN"
echo "model=$MODEL"
echo "gpu=$GPU port=$PORT"
echo "video_backend=$VIDEO_BACKEND"
echo "media_loading_threads=$MEDIA_LOADING_THREADS"
echo "media_io_kwargs=$MEDIA_IO_KWARGS"
echo "log=$LOG_FILE"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export VLLM_VIDEO_FETCH_TIMEOUT=$VIDEO_FETCH_TIMEOUT
export VLLM_MEDIA_LOADING_THREAD_COUNT=$MEDIA_LOADING_THREADS

ARGS=(
    serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --dtype auto
    --api-key EMPTY
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --no-enable-prefix-caching
    --mm-processor-cache-gb 0
    --scheduling-policy priority
    --media-io-kwargs "$MEDIA_IO_KWARGS"
)

if [[ $VIDEO_BACKEND == pynvvideocodec ]]; then
    ARGS+=(--mm-ipc-gpu-memory-gb "$MM_IPC_GPU_MEMORY_GB")
fi

exec "$VLLM_BIN" "${ARGS[@]}" >>"$LOG_FILE" 2>&1
