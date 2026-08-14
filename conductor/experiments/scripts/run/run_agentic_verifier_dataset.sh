#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-$ROOT/conductor/experiments/diverse_eval/reasoning/star_100_available.jsonl}"
SWEEP_RESULTS="${SWEEP_RESULTS:-$DATASET}"
OUT_DIR="${OUT_DIR:-$ROOT/conductor/experiments/large_sweeps/agentic_verifier_$(date +%Y%m%d_%H%M%S)}"

NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEXES="${SHARD_INDEXES:-0}"
PORTS="${PORTS:-9000}"
CUDA_DEVICES="${CUDA_DEVICES:-}"

CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-$ROOT/conductor/experiments/self_improving/data/video_agent_caption_cache_reasoning}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
CONTEXT_COVERAGE="${CONTEXT_COVERAGE:-all}"
CONTEXT_SEGMENT_S="${CONTEXT_SEGMENT_S:-4}"
CONTEXT_FRAMES_PER_SEGMENT="${CONTEXT_FRAMES_PER_SEGMENT:-2}"

ENABLE_OBJECT_TOOLS="${ENABLE_OBJECT_TOOLS:-1}"
DETECTOR_MODEL="${DETECTOR_MODEL:-yolov8x-worldv2.pt}"
DETECTOR_CONF="${DETECTOR_CONF:-0.15}"

mkdir -p "$OUT_DIR"

export PYTHONPATH="${PYTHONPATH:-$ROOT/decord/python:$ROOT}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

IFS=',' read -r -a SHARD_ARRAY <<< "$SHARD_INDEXES"
IFS=',' read -r -a PORT_ARRAY <<< "$PORTS"

if [ -n "$CUDA_DEVICES" ]; then
  IFS=',' read -r -a CUDA_ARRAY <<< "$CUDA_DEVICES"
else
  CUDA_ARRAY=()
fi

echo "[agentic-verifier] root=$ROOT"
echo "[agentic-verifier] dataset=$DATASET"
echo "[agentic-verifier] sweep_results=$SWEEP_RESULTS"
echo "[agentic-verifier] out_dir=$OUT_DIR"
echo "[agentic-verifier] num_shards=$NUM_SHARDS shard_indexes=$SHARD_INDEXES ports=$PORTS"
echo "[agentic-verifier] detector=${DETECTOR_MODEL} enable_object_tools=${ENABLE_OBJECT_TOOLS}"

pids=()
for pos in "${!SHARD_ARRAY[@]}"; do
  shard_index="${SHARD_ARRAY[$pos]}"

  if [ "$pos" -ge "${#PORT_ARRAY[@]}" ]; then
    echo "Missing port for shard position $pos" >&2
    exit 2
  fi

  port="${PORT_ARRAY[$pos]}"

  if [ "${#CUDA_ARRAY[@]}" -gt 0 ]; then
    if [ "$pos" -ge "${#CUDA_ARRAY[@]}" ]; then
      echo "Missing CUDA device for shard position $pos" >&2
      exit 2
    fi
    cuda_device="${CUDA_ARRAY[$pos]}"
  else
    cuda_device="$pos"
  fi

  output="$OUT_DIR/results.part${shard_index}.jsonl"
  log="$OUT_DIR/run.part${shard_index}.log"

  args=(
    "$PYTHON_BIN" "$ROOT/conductor/control_plane/run_video_agent.py"
    --force-agentic
    --input "$DATASET"
    --sweep-results "$SWEEP_RESULTS"
    --output "$output"
    --base-url "http://localhost:${port}/v1"
    --caption-base-url "http://localhost:${port}/v1"
    --caption-cache-dir "$CAPTION_CACHE_DIR"
    --context-coverage "$CONTEXT_COVERAGE"
    --context-segment-s "$CONTEXT_SEGMENT_S"
    --context-frames-per-segment "$CONTEXT_FRAMES_PER_SEGMENT"
    --max-rounds "$MAX_ROUNDS"
    --num-shards "$NUM_SHARDS"
    --shard-index "$shard_index"
    --siglip-model "$SIGLIP_MODEL"
    --clip-device "cuda:${cuda_device}"
  )

  if [ "$ENABLE_OBJECT_TOOLS" = "1" ]; then
    args+=(
      --enable-object-tools
      --detector-model "$DETECTOR_MODEL"
      --detector-device "cuda:${cuda_device}"
      --detector-conf "$DETECTOR_CONF"
    )
  fi

  nohup "${args[@]}" > "$log" 2>&1 &
  pids+=("$!")
  echo "[start] shard=$shard_index port=$port cuda=$cuda_device pid=${pids[-1]} output=$output log=$log"
done

echo "[agentic-verifier] launched ${#pids[@]} workers"
