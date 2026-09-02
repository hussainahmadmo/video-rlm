#!/usr/bin/env bash
set -euo pipefail

# Complete evaluation for unweighted, stage-wise max-min fairness.
# A vLLM server configured with --scheduling-policy priority must already be
# healthy at PORT. Each phase is resumable because completed summaries skip.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
PORT=${PORT:-9000}
SEEDS=${SEEDS:-"1 2 3 4"}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/max_min_fairness_$STAMP}
PHASES=${PHASES:-"primary robustness reconciliation"}

PRIMARY=$ROOT/run_multitenant_scheduling_validation.sh
ROBUST=$ROOT/run_vtc_multimodal_validation.sh
CROSS_ANALYZER=$ROOT/conductor/experiments/scripts/analyze/analyze_cross_stage_fairness.py

mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_max_min_fairness.path"

test -x "$PRIMARY"
test -x "$ROBUST"
curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:$PORT/v1/models" >/dev/null

contains_phase() {
  [[ " $PHASES " == *" $1 "* ]]
}

if contains_phase primary; then
  env VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" PORT="$PORT" \
    SEEDS="$SEEDS" POLICIES="fcfs engine_tenant_fair max_min" \
    UNIFORM_CLASS=1 \
    OUT="$OUT/primary" LOGROOT="$ROOT/logs/$(basename "$OUT")/primary" \
    "$PRIMARY"
  "$PY" "$CROSS_ANALYZER" --root "$OUT/primary" \
    --output "$OUT/primary/cross_stage_fairness"
fi

if contains_phase robustness; then
  env VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" PORT="$PORT" \
    SEEDS="$SEEDS" POLICIES="fcfs engine_tenant_fair max_min" \
    SCENARIOS="constant_overload work_conserving on_off_prep_heavy poisson_short_long distribution_shift" \
    OUT="$OUT/robustness" LOGROOT="$ROOT/logs/$(basename "$OUT")/robustness" \
    "$ROBUST"
  "$PY" "$CROSS_ANALYZER" --root "$OUT/robustness" \
    --output "$OUT/robustness/cross_stage_fairness"
fi

if contains_phase reconciliation; then
  # Reuse matched heterogeneous traces. The standard reconciled max-min runs
  # are produced first, followed by the estimate-only accounting ablation.
  for disabled in 0 1; do
    suffix=""
    [[ "$disabled" == 0 ]] || suffix="_no_reconcile"
    env VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" PORT="$PORT" \
      SEEDS="$SEEDS" POLICIES="max_min" \
      SCENARIOS="on_off_prep_heavy poisson_mixed_cost" \
      DISABLE_SERVICE_RECONCILIATION="$disabled" VARIANT_SUFFIX="$suffix" \
      OUT="$OUT/reconciliation" \
      LOGROOT="$ROOT/logs/$(basename "$OUT")/reconciliation" \
      "$ROBUST"
  done
  "$PY" "$CROSS_ANALYZER" --root "$OUT/reconciliation" \
    --output "$OUT/reconciliation/cross_stage_fairness"
fi

echo "MAX_MIN_EVALUATION_DONE OUT=$OUT"
