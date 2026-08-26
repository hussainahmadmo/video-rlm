#!/usr/bin/env bash
set -euo pipefail

# Entry point for the remaining paper experiments. Select stages with STAGES;
# each stage is resumable and may be sharded across independent vLLM servers.
# Example: STAGES="stress frame_cost fairness" ./run_paper_evaluation_suite.sh

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT}"
: "${PRIORITY_DATASET:?Set PRIORITY_DATASET}"

ROOT=$VIDEO_RLM_ROOT
STAGES=${STAGES:-"stress frame_cost fairness isolation replay portability"}
PORT=${PORT:-9000}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
STAMP=${PAPER_EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}
EVALROOT=${PAPER_EVAL_OUT:-"$ROOT/large_sweeps/paper_evaluation_$STAMP"}
LOGROOT=${PAPER_EVAL_LOGROOT:-"$ROOT/logs/$(basename "$EVALROOT")"}
mkdir -p "$EVALROOT" "$LOGROOT"
printf '%s\n' "$EVALROOT" > "$ROOT/logs/latest_paper_evaluation.path"

run_stage() {
  local stage=$1
  shift
  echo "START_STAGE $stage $(date -u +%FT%TZ)"
  "$@"
  echo "DONE_STAGE $stage $(date -u +%FT%TZ)"
}

for stage in $STAGES; do
  case "$stage" in
    stress)
      BACKLOG_STRESS_OUT="$EVALROOT/repeated_backlog_stress" \
      BACKLOG_STRESS_LOGROOT="$LOGROOT/repeated_backlog_stress" \
      PORT="$PORT" SHARD_INDEX="$SHARD_INDEX" NUM_SHARDS="$NUM_SHARDS" \
        run_stage stress "$ROOT/run_repeated_backlog_stress.sh"
      ;;
    frame_cost)
      FRAME_COST_OUT="$EVALROOT/frame_cost_matrix" \
      FRAME_COST_LOGROOT="$LOGROOT/frame_cost_matrix" \
      PORT="$PORT" SHARD_INDEX="$SHARD_INDEX" NUM_SHARDS="$NUM_SHARDS" \
        run_stage frame_cost "$ROOT/run_frame_cost_matrix.sh"
      ;;
    fairness)
      FAIRNESS_OUT="$EVALROOT/fairness_aging" \
      FAIRNESS_LOGROOT="$LOGROOT/fairness_aging" \
      PORT="$PORT" SHARD_INDEX="$SHARD_INDEX" NUM_SHARDS="$NUM_SHARDS" \
        run_stage fairness "$ROOT/run_fairness_aging_validation.sh"
      ;;
    isolation)
      if (( SHARD_INDEX == 0 )); then
        STATIC_ISOLATION_OUT="$EVALROOT/static_isolation" \
        STATIC_ISOLATION_LOGROOT="$LOGROOT/static_isolation" \
          run_stage isolation "$ROOT/run_static_isolation_baseline.sh"
      else
        echo "SKIP isolation on shard $SHARD_INDEX (run once on shard 0)"
      fi
      ;;
    replay)
      if [[ -z "${REPLAY_TEMPLATE:-}" ]]; then
        echo "SKIP replay: set REPLAY_TEMPLATE to a representative JSONL arrival trace"
      else
        REPLAY_OUT="$EVALROOT/replay_trace" \
        REPLAY_LOGROOT="$LOGROOT/replay_trace" \
        PORT="$PORT" SHARD_INDEX="$SHARD_INDEX" NUM_SHARDS="$NUM_SHARDS" \
          run_stage replay "$ROOT/run_replay_trace_validation.sh"
      fi
      ;;
    portability)
      if [[ -z "${PORTABILITY_MODEL:-}" ]]; then
        echo "SKIP portability: set PORTABILITY_MODEL to the served second model"
      else
        PORTABILITY_OUT="$EVALROOT/portability" \
        PORTABILITY_LOGROOT="$LOGROOT/portability" \
        MODEL="$PORTABILITY_MODEL" PORT="$PORT" \
        SHARD_INDEX="$SHARD_INDEX" NUM_SHARDS="$NUM_SHARDS" \
          run_stage portability "$ROOT/run_portability_validation.sh"
      fi
      ;;
    backend)
      run_stage backend "$ROOT/run_native_media_backend_matrix.sh"
      ;;
    *)
      echo "unknown stage: $stage" >&2
      exit 2
      ;;
  esac
done

"$VLLM_PYTHON" \
  "$ROOT/conductor/experiments/scripts/analyze/analyze_paper_evaluation.py" \
  --root "$EVALROOT" --output "$EVALROOT/paper_evaluation_report" || true

echo "ALL_SELECTED_STAGES_DONE OUT=$EVALROOT LOGROOT=$LOGROOT"
