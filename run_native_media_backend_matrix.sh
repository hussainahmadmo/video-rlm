#!/usr/bin/env bash
# Compare vLLM's own video loaders. This is intentionally separate from the
# external seek_cpu/batch_cpu/batch_nvdec preparation experiments.
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"

DATASET=${NATIVE_DATASET:-"$VIDEO_RLM_ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl"}
PORT=${NATIVE_PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
VIDEO_HTTP_ROOT=${VIDEO_HTTP_ROOT:-/dataheart/hussainahmad}
VIDEO_HTTP_PORT=${VIDEO_HTTP_PORT:-8088}
BACKENDS=${NATIVE_BACKENDS:-"opencv pyav torchcodec pynvvideocodec deepstream"}
RATES=${NATIVE_RATES:-"0.25 1.0"}
SEEDS=${NATIVE_SEEDS:-"1 2 3"}
FRAME_COUNT=${NATIVE_FRAME_COUNT:-8}
CONCURRENCY=${NATIVE_CONCURRENCY:-4}
NATIVE_NVDEC_READY=${NATIVE_NVDEC_READY:-0}
BAD_QID=${BAD_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}

RUNNER="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/run_native_vllm_video_baseline.py"
INSPECTOR="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/inspect_vllm_video_backends.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${NATIVE_MEDIA_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/native_media_backend_matrix_$STAMP"}
LOGROOT=${NATIVE_MEDIA_LOGROOT:-"$VIDEO_RLM_ROOT/logs/$(basename "$OUT")"}
mkdir -p "$OUT" "$LOGROOT"
echo "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_native_media_backend_matrix.path"

test -f "$DATASET"
test -f "$RUNNER"
if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "invalid shard $SHARD_INDEX/$NUM_SHARDS" >&2
  exit 2
fi

CAPABILITIES="$OUT/backend_capabilities.shard${SHARD_INDEX}.json"
FILTERED_DATASET="$OUT/dataset.filtered.shard${SHARD_INDEX}.jsonl"
"$VLLM_PYTHON" "$INSPECTOR" | tee "$CAPABILITIES"

jq -c --arg qid "$BAD_QID" \
  'select(((.qid // .question_id // .id) | tostring) != $qid)' \
  "$DATASET" > "$FILTERED_DATASET"

curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

HTTP_PID=
if ! curl -fsS --max-time 3 "http://127.0.0.1:$VIDEO_HTTP_PORT/" >/dev/null; then
  "$VLLM_PYTHON" -m http.server "$VIDEO_HTTP_PORT" \
    --bind 127.0.0.1 --directory "$VIDEO_HTTP_ROOT" \
    >"$LOGROOT/video_http.log" 2>&1 &
  HTTP_PID=$!
  trap 'test -z "${HTTP_PID:-}" || kill "$HTTP_PID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 20); do
    curl -fsS --max-time 3 "http://127.0.0.1:$VIDEO_HTTP_PORT/" >/dev/null && break
    sleep 0.5
  done
fi

backend_registered() {
  jq -e --arg backend "$1" \
    '.registered_video_backends | index($backend) != null' \
    "$CAPABILITIES" >/dev/null
}

job_index=0
for backend in $BACKENDS; do
  if ! backend_registered "$backend"; then
    echo "SKIP backend=$backend: not registered by this vLLM build" | tee -a "$LOGROOT/master.log"
    continue
  fi
  if [[ $backend == pynvvideocodec || $backend == deepstream ]]; then
    if [[ $NATIVE_NVDEC_READY != 1 ]]; then
      echo "SKIP backend=$backend: set NATIVE_NVDEC_READY=1 only after starting vLLM with its documented MPS/VRAM prerequisites" \
        | tee -a "$LOGROOT/master.log"
      continue
    fi
  fi

  backend_kwargs='{}'
  [[ $backend == torchcodec ]] && backend_kwargs='{"seek_mode":"approximate","num_ffmpeg_threads":1}'
  [[ $backend == pynvvideocodec ]] && backend_kwargs='{"hw_decoders":2}'

  for rate in $RATES; do
    for seed in $SEEDS; do
      if (( job_index % NUM_SHARDS != SHARD_INDEX )); then
        job_index=$((job_index + 1))
        continue
      fi
      job_index=$((job_index + 1))
      tag="${backend}_rate${rate//./p}_seed${seed}"
      run_out="$OUT/$tag"
      echo "START $tag $(date -u +%FT%TZ)" | tee -a "$LOGROOT/master.log"
      "$VLLM_PYTHON" "$RUNNER" \
        --dataset "$FILTERED_DATASET" \
        --output "$run_out/results.jsonl" \
        --summary-output "$run_out/summary.json" \
        --frame-counts "$FRAME_COUNT" \
        --ports "$PORT" \
        --concurrency "$CONCURRENCY" \
        --arrival-rate-qps "$rate" \
        --arrival-order seeded_random \
        --arrival-seed "$seed" \
        --video-map "$VIDEO_HTTP_ROOT=http://127.0.0.1:$VIDEO_HTTP_PORT" \
        --video-backend "$backend" \
        --video-backend-kwargs "$backend_kwargs" \
        --max-pixels 100352 \
        --max-tokens 32 \
        --request-timeout-s 1800 \
        >"$LOGROOT/$tag.log" 2>&1
      echo "DONE $tag $(date -u +%FT%TZ)" | tee -a "$LOGROOT/master.log"
    done
  done
done

echo "ALL_DONE OUT=$OUT" | tee -a "$LOGROOT/master.log"
echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT" | tee -a "$LOGROOT/master.log"
