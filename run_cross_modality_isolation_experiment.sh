#!/usr/bin/env bash
set -euo pipefail

# Measure whether preparation-heavy video traffic slows a low-rate text tenant.
# A priority-aware vLLM endpoint must already be healthy at PORT.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
DATASET=${PRIORITY_DATASET:-$ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl}
PORT=${PORT:-9000}
SEEDS=${SEEDS:-"1 2 3"}
DURATION_S=${DURATION_S:-180}
CAPACITY_QPS=${CAPACITY_QPS:-0.22}
PREP_WORKERS=${PREP_WORKERS:-8}
OUT=${OUT:-$ROOT/large_sweeps/cross_modality_isolation}
LOGROOT=${LOGROOT:-$ROOT/logs/cross_modality_isolation}

GEN=$ROOT/conductor/experiments/scripts/run/generate_vtc_multimodal_trace.py
FILTER=$ROOT/conductor/experiments/scripts/run/filter_trace_by_tenant.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py

mkdir -p "$OUT/traces" "$LOGROOT"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

run_case() {
  local seed=$1 name=$2 trace=$3 policy=$4 ffmpeg_threads=$5
  local output="$OUT/seed${seed}/$name"
  local log="$LOGROOT/seed${seed}-${name}.log"
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP seed=$seed case=$name"
    return
  fi
  if [[ -d "$output" ]]; then
    mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "START seed=$seed case=$name $(date -u +%FT%TZ)"
  VIDEO_RLM_FFMPEG_THREADS=$ffmpeg_threads "$PY" "$RUNNER" \
    --arrival-trace "$trace" --output "$output" --port "$PORT" \
    --prep-policy "$policy" --tenant-weight a=1 --tenant-weight b=1 \
    --prep-workers "$PREP_WORKERS" --urgent-prep-reserve 0 \
    --vlm-concurrency 4 --prepared-queue-depth 16 \
    --decode-backend seek_cpu --decode-timeout-s 600 \
    --request-timeout-s 1800 >"$log" 2>&1
  echo "DONE seed=$seed case=$name $(date -u +%FT%TZ)"
}

for seed in $SEEDS; do
  mixed="$OUT/traces/mixed-seed${seed}.jsonl"
  text="$OUT/traces/text-solo-seed${seed}.jsonl"
  if [[ ! -f "$mixed" ]]; then
    "$PY" "$GEN" --dataset "$DATASET" --output "$mixed" \
      --scenario cross_modality_noisy_neighbor --duration-s "$DURATION_S" \
      --capacity-qps "$CAPACITY_QPS" --seed "$seed"
  fi
  if [[ ! -f "$text" ]]; then
    "$PY" "$FILTER" --input "$mixed" --output "$text" --tenant b
  fi

  run_case "$seed" text_solo "$text" fcfs 1
  run_case "$seed" video_contended_unbounded "$mixed" fcfs 0
  run_case "$seed" video_contended_bounded "$mixed" fcfs 1
  run_case "$seed" video_contended_cross_stage "$mixed" max_min 1
done

echo "CROSS_MODALITY_ISOLATION_DONE OUT=$OUT"
