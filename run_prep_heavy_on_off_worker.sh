#!/usr/bin/env bash
set -euo pipefail

# Run one preparation-worker point of the preparation-heavy ON/OFF study.
# A two-GPU wrapper invokes this once per independent vLLM replica.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
DATASET=${PRIORITY_DATASET:-$ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl}
PORT=${PORT:-9020}
PREP_WORKERS=${PREP_WORKERS:-4}
SEEDS=${SEEDS:-"1 2 3"}
POLICIES=${POLICIES:-"fcfs engine_tenant_fair tenant_fair"}
DURATION_S=${DURATION_S:-180}
CAPACITY_QPS=${CAPACITY_QPS:-0.22}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-16}
DECODE_BACKEND=${DECODE_BACKEND:-seek_cpu}
EXCLUDE_QID=${EXCLUDE_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}

GEN=$ROOT/conductor/experiments/scripts/run/generate_vtc_multimodal_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/prep_heavy_on_off_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
mkdir -p "$OUT/traces" "$LOGROOT"

test -f "$DATASET"
test -f "$GEN"
test -f "$RUNNER"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

for seed in $SEEDS; do
  trace="$OUT/traces/on_off_prep_heavy-w${PREP_WORKERS}-seed${seed}.jsonl"
  if [[ ! -f "$trace" ]]; then
    "$PY" "$GEN" \
      --dataset "$DATASET" --output "$trace" \
      --scenario on_off_prep_heavy \
      --duration-s "$DURATION_S" --capacity-qps "$CAPACITY_QPS" \
      --seed "$seed" --exclude-qid "$EXCLUDE_QID"
  fi

  name="on_off_prep_heavy-w${PREP_WORKERS}-seed${seed}"
  for policy in $POLICIES; do
    output="$OUT/$name/$policy"
    log="$LOGROOT/$name-$policy.log"
    if [[ -f "$output/summary.json" ]]; then
      echo "SKIP workers=$PREP_WORKERS seed=$seed policy=$policy"
      continue
    fi
    if [[ -d "$output" ]]; then
      mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
    fi
    echo "START workers=$PREP_WORKERS seed=$seed policy=$policy $(date -u +%FT%TZ)"
    VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$output" --port "$PORT" \
      --prep-policy "$policy" \
      --tenant-weight a=1 --tenant-weight b=1 \
      --decode-backend "$DECODE_BACKEND" \
      --prep-workers "$PREP_WORKERS" \
      --vlm-concurrency "$VLM_CONCURRENCY" \
      --prepared-queue-depth "$PREPARED_QUEUE_DEPTH" \
      --decode-timeout-s 600 --request-timeout-s 1800 \
      >"$log" 2>&1
    echo "DONE workers=$PREP_WORKERS seed=$seed policy=$policy $(date -u +%FT%TZ)"
  done
done

echo "WORKER_POINT_DONE workers=$PREP_WORKERS port=$PORT OUT=$OUT"
