#!/usr/bin/env bash
set -euo pipefail

# Matched ablation of frame-only and metadata-aware online cost profiling.
# Shard seeds across independent vLLM replicas by setting SHARD_INDEX,
# NUM_SHARDS, and PORT. All variants use identical traces and scheduling
# parameters; only the preparation/engine profiler changes.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SEEDS=${SEEDS:-"1 2 3 4"}
POLICIES=${POLICIES:-"tenant_priority fair_slowdown"}
PROFILES=${PROFILES:-"frame_frame metadata_frame metadata_metadata"}
PREP_WORKERS=${PREP_WORKERS:-4}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
TRACE_ROOT=${TRACE_ROOT:-$ROOT/large_sweeps/multitenant_scheduling_20260826_104047/traces}

RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
INDEXER=$ROOT/conductor/experiments/scripts/run/build_video_metadata_index.py
ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_profiler_ablation.py
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/profiler_ablation_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
METADATA_INDEX=${METADATA_INDEX:-$OUT/video_metadata.jsonl}

mkdir -p "$OUT" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_profiler_ablation.path"

test -f "$RUNNER"
test -f "$INDEXER"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

if [[ ! -f "$METADATA_INDEX" ]]; then
  "$PY" "$INDEXER" --trace-root "$TRACE_ROOT" \
    --output "$METADATA_INDEX" --workers 16
fi

index=0
for seed in $SEEDS; do
  if (( index % NUM_SHARDS != SHARD_INDEX )); then
    index=$((index + 1))
    continue
  fi
  index=$((index + 1))
  name="equal-tenants-mixed-cost-seed${seed}"
  trace="$TRACE_ROOT/$name.jsonl"
  test -f "$trace"

  # Alternate order across seeds to reduce systematic warm-state bias.
  profile_order=$PROFILES
  if (( seed % 2 == 0 )); then
    profile_order=$(printf '%s\n' $PROFILES | tac | tr '\n' ' ')
  fi

  for policy in $POLICIES; do
    for profile in $profile_order; do
      case "$profile" in
        frame_frame)
          prep_profiler=frame_ewma
          engine_profiler=frame_ewma
          ;;
        metadata_frame)
          prep_profiler=metadata_ewma
          engine_profiler=frame_ewma
          ;;
        metadata_metadata)
          prep_profiler=metadata_ewma
          engine_profiler=metadata_ewma
          ;;
        *)
          echo "Unknown profile variant: $profile" >&2
          exit 2
          ;;
      esac

      variant="${policy}_${profile}"
      output="$OUT/$name/$variant"
      log="$LOGROOT/$name-$variant.log"
      if [[ -f "$output/summary.json" ]]; then
        echo "SKIP shard=$SHARD_INDEX seed=$seed variant=$variant"
        continue
      fi
      if [[ -d "$output" ]]; then
        mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
      fi

      echo "START shard=$SHARD_INDEX seed=$seed variant=$variant $(date -u +%FT%TZ)"
      VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
        --arrival-trace "$trace" --output "$output" --port "$PORT" \
        --prep-policy "$policy" \
        --prep-cost-profiler "$prep_profiler" \
        --engine-cost-profiler "$engine_profiler" \
        --video-metadata-index "$METADATA_INDEX" \
        --tenant-weight a=1 --tenant-weight b=1 --tenant-weight c=1 \
        --decode-backend seek_cpu --prep-workers "$PREP_WORKERS" \
        --vlm-concurrency "$VLM_CONCURRENCY" --prepared-queue-depth 16 \
        --decode-timeout-s 600 --request-timeout-s 1800 \
        >"$log" 2>&1
      echo "DONE shard=$SHARD_INDEX seed=$seed variant=$variant $(date -u +%FT%TZ)"
    done
  done
done

"$PY" "$ANALYZER" --root "$OUT" --output "$OUT/profiler_ablation" || true
echo "SHARD_DONE shard=$SHARD_INDEX/$NUM_SHARDS port=$PORT OUT=$OUT"
