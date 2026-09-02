#!/usr/bin/env bash
set -euo pipefail

# Resumable end-to-end evaluation suite for cross-stage multimodal fairness.
# A priority-aware vLLM endpoint must already be healthy at PORT. Policies run
# sequentially so matched experiments never contend with each other.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
DATASET=${PRIORITY_DATASET:-$ROOT/large_sweeps/native_vllm_metis249/dataset_nature.jsonl}
PORT=${PORT:-9000}
SEEDS=${SEEDS:-"1 2 3"}
PHASES=${PHASES:-"headline solo stage fairness load heterogeneity inflight"}
DRY_RUN=${DRY_RUN:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/complete_multimodal_fairness_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}
DECODE_BACKEND=${DECODE_BACKEND:-seek_cpu}
EXCLUDE_QID=${EXCLUDE_QID:-20520eff-abdf-4d4f-94ad-cc751a8960d0}
ENGINE_TOKEN_PROFILE=${ENGINE_TOKEN_PROFILE:-$ROOT/large_sweeps/vllm_token_service_profile/token_service_cost.json}

HEADLINE_DURATION_S=${HEADLINE_DURATION_S:-90}
STAGE_DURATION_S=${STAGE_DURATION_S:-45}
LOAD_DURATION_S=${LOAD_DURATION_S:-60}
MOCKUP_DURATION_S=${MOCKUP_DURATION_S:-180}
ON_OFF_DURATION_S=${ON_OFF_DURATION_S:-240}
CAPACITY_QPS=${CAPACITY_QPS:-0.22}

GEN_VTC=$ROOT/conductor/experiments/scripts/run/generate_vtc_multimodal_trace.py
GEN_EQUAL=$ROOT/conductor/experiments/scripts/run/generate_multitenant_overload_trace.py
FILTER=$ROOT/conductor/experiments/scripts/run/filter_trace_by_tenant.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
ANALYZE_CROSS=$ROOT/conductor/experiments/scripts/analyze/analyze_cross_stage_fairness.py
ANALYZE_SUITE=$ROOT/conductor/experiments/scripts/analyze/analyze_complete_fairness_suite.py
PLOT_GAIN=$ROOT/conductor/experiments/scripts/analyze/plot_aggressive_cross_stage_gains.py
PLOT_SLOWDOWN=$ROOT/conductor/experiments/scripts/analyze/plot_measured_victim_slowdown.py
PLOT_FAIRNESS_PLANE=$ROOT/conductor/experiments/scripts/analyze/plot_measured_fairness_plane.py

mkdir -p "$OUT/traces" "$OUT/figures" "$LOGROOT"
printf '%s\n' "$OUT" > "$ROOT/logs/latest_complete_multimodal_fairness.path"

for required in "$DATASET" "$GEN_VTC" "$GEN_EQUAL" "$FILTER" "$RUNNER"; do
  test -f "$required"
done
ENGINE_PROFILE_ARGS=()
if [[ -n "$ENGINE_TOKEN_PROFILE" ]]; then
  test -f "$ENGINE_TOKEN_PROFILE"
  ENGINE_PROFILE_ARGS=(--engine-token-profile "$ENGINE_TOKEN_PROFILE")
fi
if [[ "$DRY_RUN" != 1 ]]; then
  curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$PORT/v1/models" >/dev/null
fi

contains_phase() {
  [[ " $PHASES " == *" $1 "* ]]
}

run_command() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

generate_vtc_trace() {
  local trace=$1
  local scenario=$2
  local duration=$3
  local capacity=$4
  local seed=$5
  [[ -f "$trace" ]] && return
  run_command "$PY" "$GEN_VTC" \
    --dataset "$DATASET" --output "$trace" --scenario "$scenario" \
    --duration-s "$duration" --capacity-qps "$capacity" --seed "$seed" \
    --exclude-qid "$EXCLUDE_QID"
}

generate_equal_trace() {
  local trace=$1
  local seed=$2
  shift 2
  [[ -f "$trace" ]] && return
  run_command "$PY" "$GEN_EQUAL" \
    --dataset "$DATASET" --output "$trace" --seed "$seed" \
    --background-per-tenant 12 --urgent-per-tenant 0 --uniform-class \
    --frame-budgets "$@" --exclude-qid "$EXCLUDE_QID"
}

run_case() {
  local phase=$1
  local case_name=$2
  local seed=$3
  local policy=$4
  local variant=$5
  local trace=$6
  local prep_workers=$7
  local vlm_concurrency=$8
  shift 8

  local run_name="${case_name}-seed${seed}"
  local output="$OUT/$phase/$run_name/$variant"
  local log="$LOGROOT/$phase/$run_name-$variant.log"
  mkdir -p "$(dirname "$log")"
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP phase=$phase case=$case_name seed=$seed variant=$variant"
    return
  fi
  if [[ "$DRY_RUN" != 1 && -d "$output" ]]; then
    mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "START phase=$phase case=$case_name seed=$seed variant=$variant $(date -u +%FT%TZ)"
  if [[ "$DRY_RUN" == 1 ]]; then
    run_command env VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$output" --port "$PORT" \
      --prep-policy "$policy" --tenant-weight a=1 --tenant-weight b=1 \
      --tenant-weight c=1 --decode-backend "$DECODE_BACKEND" \
      --prep-workers "$prep_workers" --vlm-concurrency "$vlm_concurrency" \
      --prepared-queue-depth 16 --decode-timeout-s 600 \
      --request-timeout-s 1800 "${ENGINE_PROFILE_ARGS[@]}" "$@"
  else
    VIDEO_RLM_FFMPEG_THREADS=1 "$PY" "$RUNNER" \
      --arrival-trace "$trace" --output "$output" --port "$PORT" \
      --prep-policy "$policy" --tenant-weight a=1 --tenant-weight b=1 \
      --tenant-weight c=1 --decode-backend "$DECODE_BACKEND" \
      --prep-workers "$prep_workers" --vlm-concurrency "$vlm_concurrency" \
      --prepared-queue-depth 16 --decode-timeout-s 600 \
      --request-timeout-s 1800 "${ENGINE_PROFILE_ARGS[@]}" "$@" >"$log" 2>&1
  fi
  echo "DONE phase=$phase case=$case_name seed=$seed variant=$variant $(date -u +%FT%TZ)"
}

analyze_phase() {
  local phase=$1
  [[ "$DRY_RUN" == 1 ]] && return
  [[ -d "$OUT/$phase" ]] || return
  "$PY" "$ANALYZE_CROSS" --root "$OUT/$phase" \
    --output "$OUT/$phase/cross_stage_fairness" || true
}

DIRECT_POLICIES=(
  fcfs tenant_round_robin prep_max_min engine_tenant_fair max_min
)
ALL_FAIRNESS_POLICIES=(
  fcfs tenant_round_robin prep_max_min engine_tenant_fair max_min
)

if contains_phase headline; then
  for seed in $SEEDS; do
    trace="$OUT/traces/aggressive_asymmetric-seed${seed}.jsonl"
    generate_vtc_trace "$trace" aggressive_asymmetric \
      "$HEADLINE_DURATION_S" "$CAPACITY_QPS" "$seed"
    for policy in "${DIRECT_POLICIES[@]}"; do
      run_case headline aggressive_asymmetric "$seed" "$policy" "$policy" \
        "$trace" 4 4
    done
  done
  analyze_phase headline
fi

if contains_phase solo; then
  for seed in $SEEDS; do
    source_trace="$OUT/traces/aggressive_asymmetric-seed${seed}.jsonl"
    generate_vtc_trace "$source_trace" aggressive_asymmetric \
      "$HEADLINE_DURATION_S" "$CAPACITY_QPS" "$seed"
    for tenant in b c; do
      name="aggressive_asymmetric-victim-${tenant}-solo"
      trace="$OUT/traces/${name}-seed${seed}.jsonl"
      if [[ ! -f "$trace" ]]; then
        run_command "$PY" "$FILTER" --input "$source_trace" \
          --output "$trace" --tenant "$tenant"
      fi
      run_case solo "$name" "$seed" fcfs fcfs "$trace" 4 4
    done
  done
fi

if contains_phase stage; then
  for seed in $SEEDS; do
    trace="$OUT/traces/stage-asymmetric-seed${seed}.jsonl"
    generate_vtc_trace "$trace" aggressive_asymmetric \
      "$STAGE_DURATION_S" "$CAPACITY_QPS" "$seed"
    for spec in prep_heavy:1:8 inference_heavy:8:1 mixed:4:4; do
      IFS=: read -r case_name prep_workers vlm_concurrency <<<"$spec"
      for policy in "${DIRECT_POLICIES[@]}"; do
        run_case stage "$case_name" "$seed" "$policy" "$policy" \
          "$trace" "$prep_workers" "$vlm_concurrency"
      done
    done
  done
  analyze_phase stage
fi

if contains_phase fairness; then
  for seed in $SEEDS; do
    trace="$OUT/traces/equal-backlogged-seed${seed}.jsonl"
    generate_equal_trace "$trace" "$seed" 8 32 128
    for policy in fcfs tenant_round_robin max_min; do
      run_case fairness equal_backlogged "$seed" "$policy" "$policy" \
        "$trace" 4 4
    done
  done
  analyze_phase fairness
fi

if contains_phase constant_rate; then
  for seed in $SEEDS; do
    trace="$OUT/traces/constant-rate-two-tenant-seed${seed}.jsonl"
    generate_vtc_trace "$trace" rate_asymmetric_constant_cost \
      "$MOCKUP_DURATION_S" "$CAPACITY_QPS" "$seed"
    for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
      run_case constant_rate two_tenant_2_to_1 "$seed" "$policy" \
        "$policy" "$trace" 4 4
    done
  done
  analyze_phase constant_rate
fi

if contains_phase stochastic; then
  for seed in $SEEDS; do
    trace="$OUT/traces/poisson-heterogeneous-seed${seed}.jsonl"
    generate_vtc_trace "$trace" poisson_heterogeneous_fairness \
      "$MOCKUP_DURATION_S" "$CAPACITY_QPS" "$seed"
    for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
      run_case stochastic poisson_heterogeneous "$seed" "$policy" \
        "$policy" "$trace" 4 4
    done
  done
  analyze_phase stochastic
fi

if contains_phase stage_heterogeneity; then
  for seed in $SEEDS; do
    for spec in \
      heterogeneous_prep_cost:prep_cost:2:8 \
      heterogeneous_inference_cost:inference_cost:8:2; do
      IFS=: read -r scenario case_name prep_workers vlm_concurrency <<<"$spec"
      trace="$OUT/traces/${scenario}-seed${seed}.jsonl"
      generate_vtc_trace "$trace" "$scenario" \
        "$MOCKUP_DURATION_S" "$CAPACITY_QPS" "$seed"
      for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
        run_case stage_heterogeneity "$case_name" "$seed" "$policy" \
          "$policy" "$trace" "$prep_workers" "$vlm_concurrency"
      done
    done
  done
  analyze_phase stage_heterogeneity
fi

if contains_phase fairness_plane; then
  for seed in $SEEDS; do
    trace="$OUT/traces/simultaneous-cross-stage-heterogeneity-seed${seed}.jsonl"
    generate_vtc_trace "$trace" simultaneous_cross_stage_heterogeneity \
      "$MOCKUP_DURATION_S" "$CAPACITY_QPS" "$seed"
    for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
      run_case fairness_plane simultaneous_cross_stage_heterogeneity "$seed" \
        "$policy" "$policy" "$trace" 4 4
    done
  done
  analyze_phase fairness_plane
fi

if contains_phase on_off; then
  for seed in $SEEDS; do
    for scenario in on_off_two_tenant on_off_backlogged; do
      trace="$OUT/traces/${scenario}-seed${seed}.jsonl"
      generate_vtc_trace "$trace" "$scenario" \
        "$ON_OFF_DURATION_S" "$CAPACITY_QPS" "$seed"
      for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
        run_case on_off "$scenario" "$seed" "$policy" "$policy" \
          "$trace" 4 4
      done
    done
  done
  analyze_phase on_off
fi

if contains_phase work_conservation; then
  for seed in $SEEDS; do
    trace="$OUT/traces/work-conserving-two-tenant-seed${seed}.jsonl"
    generate_vtc_trace "$trace" work_conserving_two_tenant \
      "$MOCKUP_DURATION_S" "$CAPACITY_QPS" "$seed"
    for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
      run_case work_conservation two_tenant_underloaded_overloaded "$seed" \
        "$policy" "$policy" "$trace" 4 4
    done
  done
  analyze_phase work_conservation
fi

if contains_phase isolation; then
  for seed in $SEEDS; do
    trace="$OUT/traces/noisy-neighbor-isolation-seed${seed}.jsonl"
    generate_vtc_trace "$trace" noisy_neighbor_isolation \
      "$MOCKUP_DURATION_S" "$CAPACITY_QPS" "$seed"
    for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
      run_case isolation noisy_neighbor "$seed" "$policy" "$policy" \
        "$trace" 4 4
    done
  done
  analyze_phase isolation
fi

if contains_phase load; then
  for load_spec in 0p5:0.11 0p8:0.176 1p0:0.22 1p2:0.264 1p5:0.33; do
    IFS=: read -r load_label load_capacity <<<"$load_spec"
    for seed in $SEEDS; do
      trace="$OUT/traces/load-${load_label}-seed${seed}.jsonl"
      generate_vtc_trace "$trace" rate_sweep_constant_cost \
        "$LOAD_DURATION_S" "$load_capacity" "$seed"
      for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
        run_case load "load-${load_label}" "$seed" "$policy" "$policy" \
          "$trace" 4 4
      done
    done
  done
  analyze_phase load
fi

if contains_phase heterogeneity; then
  for hetero_spec in homogeneous:32 moderate:8,32 high:8,32,128; do
    IFS=: read -r hetero_label frame_csv <<<"$hetero_spec"
    IFS=, read -ra frames <<<"$frame_csv"
    for seed in $SEEDS; do
      trace="$OUT/traces/heterogeneity-${hetero_label}-seed${seed}.jsonl"
      generate_equal_trace "$trace" "$seed" "${frames[@]}"
      for policy in "${ALL_FAIRNESS_POLICIES[@]}"; do
        run_case heterogeneity "heterogeneity-${hetero_label}" "$seed" \
          "$policy" "$policy" "$trace" 4 4
      done
    done
  done
  analyze_phase heterogeneity
fi

if contains_phase inflight; then
  for concurrency in 1 2 4 8; do
    for seed in $SEEDS; do
      trace="$OUT/traces/inflight-seed${seed}.jsonl"
      generate_equal_trace "$trace" "$seed" 8 32 128
      run_case inflight "concurrency-${concurrency}" "$seed" max_min \
        completion_only "$trace" "$concurrency" "$concurrency" \
        --completion-only-accounting
      run_case inflight "concurrency-${concurrency}" "$seed" max_min \
        fixed_reservation "$trace" "$concurrency" "$concurrency" \
        --cost-profiler-min-samples 100000 --prep-fixed-cost-s 4 \
        --prep-seconds-per-frame 0.000000001 --vlm-fixed-cost-s 4
      run_case inflight "concurrency-${concurrency}" "$seed" max_min \
        profiled_reconciled "$trace" "$concurrency" "$concurrency"
    done
  done
  analyze_phase inflight
fi

if [[ "$DRY_RUN" != 1 ]]; then
  "$PY" "$ANALYZE_SUITE" --root "$OUT" --output "$OUT/suite_results"
  if contains_phase headline; then
    "$PY" "$PLOT_GAIN" --root "$OUT/headline" \
      --output "$OUT/figures/headline_tenant_latency" || true
  fi
  if contains_phase headline && contains_phase solo; then
    "$PY" "$PLOT_SLOWDOWN" --contended-root "$OUT/headline" \
      --solo-root "$OUT/solo" \
      --output "$OUT/figures/measured_victim_slowdown" || true
  fi
  if contains_phase fairness_plane; then
    "$PY" "$PLOT_FAIRNESS_PLANE" --root "$OUT/fairness_plane" \
      --output "$OUT/figures/measured_cpu_inference_fairness_plane" || true
  fi
fi

echo "COMPLETE_MULTIMODAL_FAIRNESS_SUITE_DONE OUT=$OUT"
