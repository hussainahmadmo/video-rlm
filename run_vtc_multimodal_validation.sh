#!/usr/bin/env bash
set -euo pipefail

# Empirical VTC-style validation for a two-stage video-MLLM serving pipeline.
# Each scenario/seed is assigned to one shard; policies run sequentially so
# matched policies never contend for the same vLLM replica.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
DATASET=${PRIORITY_DATASET:-$ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl}
PORT=${PORT:-9020}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SEEDS=${SEEDS:-"1 2 3"}
SCENARIOS=${SCENARIOS:-"constant_overload work_conserving on_off_under_share on_off_backlogged poisson_short_long poisson_mixed_cost noisy_neighbor_isolation distribution_shift"}
POLICIES=${POLICIES:-"fcfs engine_tenant_fair max_min"}
VARIANT_SUFFIX=${VARIANT_SUFFIX:-}
DISABLE_SERVICE_RECONCILIATION=${DISABLE_SERVICE_RECONCILIATION:-0}
DURATION_S=${DURATION_S:-180}
CAPACITY_QPS=${CAPACITY_QPS:-0.22}
PREP_WORKERS=${PREP_WORKERS:-4}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-16}
DECODE_BACKEND=${DECODE_BACKEND:-seek_cpu}
EXCLUDE_QID=${EXCLUDE_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}

GEN=$ROOT/conductor/experiments/scripts/run/generate_vtc_multimodal_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_vtc_multimodal.py
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/vtc_multimodal_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_vtc_multimodal.path"

test -f "$DATASET"
test -f "$GEN"
test -f "$RUNNER"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

runner_extra=()
if [[ "$DISABLE_SERVICE_RECONCILIATION" == 1 ]]; then
  runner_extra+=(--no-service-reconciliation)
elif [[ "$DISABLE_SERVICE_RECONCILIATION" != 0 ]]; then
  echo "DISABLE_SERVICE_RECONCILIATION must be 0 or 1" >&2
  exit 2
fi

index=0
for seed in $SEEDS; do
  for scenario in $SCENARIOS; do
    if (( index % NUM_SHARDS != SHARD_INDEX )); then
      index=$((index + 1))
      continue
    fi
    index=$((index + 1))
    name="${scenario}-seed${seed}"
    trace="$OUT/traces/$name.jsonl"
    [[ -f "$trace" ]] || "$PY" "$GEN" \
      --dataset "$DATASET" --output "$trace" --scenario "$scenario" \
      --duration-s "$DURATION_S" --capacity-qps "$CAPACITY_QPS" \
      --seed "$seed" --exclude-qid "$EXCLUDE_QID"

    for policy in $POLICIES; do
      variant="${policy}${VARIANT_SUFFIX}"
      output="$OUT/$name/$variant"
      log="$LOGROOT/$name-$variant.log"
      if [[ -f "$output/summary.json" ]]; then
        echo "SKIP shard=$SHARD_INDEX scenario=$scenario seed=$seed policy=$policy"
        continue
      fi
      [[ ! -d "$output" ]] || mv \
        "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
      echo "START shard=$SHARD_INDEX scenario=$scenario seed=$seed policy=$policy $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
        --arrival-trace "$trace" --output "$output" --port "$PORT" \
        --prep-policy "$policy" \
        --tenant-weight a=1 --tenant-weight b=1 --tenant-weight c=1 \
        --decode-backend "$DECODE_BACKEND" \
        --prep-workers "$PREP_WORKERS" \
        --vlm-concurrency "$VLM_CONCURRENCY" \
        --prepared-queue-depth "$PREPARED_QUEUE_DEPTH" \
        --decode-timeout-s 600 --request-timeout-s 1800 \
        "${runner_extra[@]}" \
        >"$log" 2>&1
      echo "DONE shard=$SHARD_INDEX scenario=$scenario seed=$seed policy=$policy $(date -u +%FT%TZ)"
    done
  done
done

"$PY" "$ANALYZER" --root "$OUT" --output "$OUT/vtc_multimodal_results" \
  --no-plots || true
echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
