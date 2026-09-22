#!/usr/bin/env bash
set -euo pipefail

# End-to-end component ablation on one L40S. All variants reuse byte-identical
# arrival traces. Full Conductor runs first, followed by its matched ablations.

ROOT=${VIDEO_RLM_ROOT:-/workspace/video-rlm}
PY=${VLLM_PYTHON:-/workspace/venv/bin/python}
VIDEO=${VIDEO:-/workspace/data/kN88RP3XWUU.mp4}
PORT=${PORT:-8000}
CPUSET=${CPUSET:-0-7}
SEEDS=${SEEDS:-"1"}
RATE=${RATE:-1.0}
DURATION_S=${DURATION_S:-120}
ARRIVAL_PATTERN=${ARRIVAL_PATTERN:-poisson}
STEP_AFTER_REQUESTS=${STEP_AFTER_REQUESTS:-25}
STEP_DURATION_S=${STEP_DURATION_S:-60}
STEP_MULTIPLIER=${STEP_MULTIPLIER:-2}
TRACE_MODE=${TRACE_MODE:-fixed}
REQUESTS_PER_TENANT=${REQUESTS_PER_TENANT:-}
FRAME_COUNTS=${FRAME_COUNTS:-"1 16 128"}
SOURCE_TRACE=${SOURCE_TRACE:-}
LANE_PROFILE=${LANE_PROFILE:-/workspace/data/l40s_lane_profile.json}
OUT=${OUT:-/workspace/results/end_to_end_components_l40s_$(date -u +%Y%m%d_%H%M%S)}
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_component_poisson_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py
VARIANTS=${VARIANTS:-"full_conductor full_conductor_strict_fifo cpu_fcfs cpu_fair hybrid_fcfs hybrid_fair full_fcfs"}
PRIORITY_VARIANT=${PRIORITY_VARIANT:-full_conductor}

for required in "$PY" "$VIDEO" "$LANE_PROFILE" "$GENERATOR" "$RUNNER"; do
  test -e "$required" || { echo "missing: $required" >&2; exit 2; }
done
if [[ -n "$SOURCE_TRACE" ]]; then
  test -f "$SOURCE_TRACE" || { echo "missing source trace: $SOURCE_TRACE" >&2; exit 2; }
fi
test ! -e "$OUT"
curl -fsS --max-time 5 -H 'Authorization: Bearer EMPTY' \
  "http://127.0.0.1:${PORT}/v1/models" >/dev/null
if pgrep -f '[r]un_mixed_end_to_end_priority.py' >/dev/null; then
  echo "another experiment runner is active" >&2
  pgrep -af '[r]un_mixed_end_to_end_priority.py' >&2
  exit 1
fi

mkdir -p "$OUT"/{traces,logs,tmp,metadata}
nvidia-smi -L >"$OUT/metadata/gpus.txt"
lscpu >"$OUT/metadata/lscpu.txt"
ffmpeg -version >"$OUT/metadata/ffmpeg.txt"
"$PY" -m pip freeze >"$OUT/metadata/python-packages.txt"
sha256sum "$VIDEO" "$LANE_PROFILE" >"$OUT/metadata/input.sha256"
{
  echo "design=end_to_end_component_ablation"
  echo "rate_qps=$RATE"
  echo "duration_s=$DURATION_S"
  echo "arrival_pattern=$ARRIVAL_PATTERN"
  echo "step_after_requests=$STEP_AFTER_REQUESTS"
  echo "step_duration_s=$STEP_DURATION_S"
  echo "step_multiplier=$STEP_MULTIPLIER"
  echo "tenant_frame_mode=$TRACE_MODE"
  echo "requests_per_tenant=${REQUESTS_PER_TENANT:-duration_bounded}"
  echo "frame_counts=$FRAME_COUNTS"
  echo "source_trace=${SOURCE_TRACE:-generated}"
  echo "seeds=$SEEDS"
  echo "variants=$VARIANTS"
  echo "placement=joint_predicted_readiness_for_hybrid_variants"
  echo "cpu_workers=4"
  echo "gpu_prep_jobs=2"
  echo "gpu_decoder_lanes=4"
  echo "within_tenant_selection=resource_feasible_class; bounded_sjf_for_full_conductor"
  echo "inference_slots=4"
  echo "handoff_depth=8"
  echo "inference_admission=fcfs"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT/manifest.txt"

common=(
  --port "$PORT" --tenant-weight a=1 --tenant-weight b=1 --tenant-weight c=1
  --decode-backend seek_cpu --prep-workers 4 --vlm-concurrency 4
  --prepared-queue-depth 8 --urgent-prep-reserve 0
  --decode-timeout-s 600 --request-timeout-s 1800 --ignore-eos
  --prep-cost-profiler frame_ewma
)
joint=(
  --prep-placement joint --gpu-prep-backend flashstyle_nvdec
  --gpu-prep-limit 2 --gpu-decoder-budget 4 --joint-gpu-widths 1 2 4
  --joint-lane-profile "$LANE_PROFILE" --joint-bypass-heavy
)

run_variant() {
  local seed=$1 variant=$2 trace=$3 output log expected
  output=$OUT/seed${seed}/$variant
  log=$OUT/logs/seed${seed}-${variant}.log
  expected=$(wc -l <"$trace")
  mkdir -p "$(dirname "$output")"
  local -a args
  case "$variant" in
    cpu_fcfs)
      args=(--prep-policy fcfs --prep-placement fixed)
      ;;
    cpu_fair)
      args=(--prep-policy prep_max_min --prep-placement fixed)
      ;;
    hybrid_fcfs)
      args=(--prep-policy fcfs "${joint[@]}" --joint-allocation-order fcfs --joint-fixed-lanes 2)
      ;;
    hybrid_fair)
      args=(--prep-policy prep_max_min "${joint[@]}" --joint-allocation-order fair --joint-fixed-lanes 2)
      ;;
    full_fcfs)
      args=(--prep-policy fcfs "${joint[@]}" --joint-allocation-order fcfs)
      ;;
    full_tenant_round_robin)
      args=(--prep-policy tenant_round_robin "${joint[@]}" --joint-allocation-order fcfs)
      ;;
    full_prep_sjf)
      args=(--prep-policy prep_sjf "${joint[@]}" --joint-allocation-order fcfs)
      ;;
    full_prep_sjf_aging)
      args=(--prep-policy prep_sjf_aging "${joint[@]}" --joint-allocation-order fcfs
        --sjf-aging-s 30)
      ;;
    full_inference_fair)
      args=(--prep-policy engine_tenant_fair "${joint[@]}" --joint-allocation-order fcfs)
      ;;
    full_cross_stage_fair)
      args=(--prep-policy max_min "${joint[@]}" --joint-allocation-order fair)
      ;;
    full_conductor)
      args=(--prep-policy prep_max_min "${joint[@]}" --joint-allocation-order fair
        --joint-within-tenant-sjf
        --joint-within-tenant-bypass-age-s 30
        --joint-within-tenant-bypass-limit 4)
      ;;
    full_conductor_strict_fifo)
      args=(--prep-policy prep_max_min "${joint[@]}" --joint-allocation-order fair)
      ;;
    eager_conductor)
      args=(--prep-policy prep_max_min "${joint[@]}" --joint-allocation-order fair
        --prep-workers 8 --prepared-queue-depth 32)
      ;;
    adaptive_capacity_conductor)
      args=(--prep-policy prep_max_min "${joint[@]}" --joint-allocation-order fair
        --prep-workers 8 --prepared-queue-depth 32
        --capacity-adaptation online
        --capacity-initial-prep-workers 4 --capacity-min-prep-workers 1
        --capacity-initial-handoff 8 --capacity-min-handoff 4
        --capacity-control-interval-s 10
        --capacity-worker-step 1 --capacity-handoff-step 4
        --capacity-hysteresis-epochs 2 --capacity-cooldown-epochs 1)
      ;;
    joint_adaptive_capacity_conductor)
      args=(--prep-policy prep_max_min "${joint[@]}" --joint-allocation-order fair
        --prep-workers 8 --prepared-queue-depth 32
        --capacity-adaptation online --capacity-adapt-gpu
        --capacity-initial-prep-workers 4 --capacity-min-prep-workers 1
        --capacity-initial-handoff 8 --capacity-min-handoff 4
        --capacity-initial-gpu-jobs 1 --capacity-min-gpu-jobs 1
        --capacity-initial-gpu-lanes 2 --capacity-min-gpu-lanes 1
        --capacity-control-interval-s 10
        --capacity-worker-step 1 --capacity-handoff-step 4
        --capacity-hysteresis-epochs 2 --capacity-cooldown-epochs 1)
      ;;
    joint_adaptive_capacity_fcfs)
      args=(--prep-policy fcfs "${joint[@]}" --joint-allocation-order fcfs
        --prep-workers 8 --prepared-queue-depth 32
        --capacity-adaptation online --capacity-adapt-gpu
        --capacity-initial-prep-workers 4 --capacity-min-prep-workers 1
        --capacity-initial-handoff 8 --capacity-min-handoff 4
        --capacity-initial-gpu-jobs 1 --capacity-min-gpu-jobs 1
        --capacity-initial-gpu-lanes 2 --capacity-min-gpu-lanes 1
        --capacity-control-interval-s 10
        --capacity-worker-step 1 --capacity-handoff-step 4
        --capacity-hysteresis-epochs 2 --capacity-cooldown-epochs 1)
      ;;
    *) echo "unknown variant: $variant" >&2; return 2 ;;
  esac
  echo "RUN seed=$seed variant=$variant requests=$expected"
  taskset -c "$CPUSET" env CUDA_VISIBLE_DEVICES=0 VIDEO_RLM_NVDEC_GPU_ID=0 \
    VIDEO_RLM_FLASH_GPU_IDS=0 VIDEO_RLM_FLASH_WORKERS_PER_GPU=4 \
    VIDEO_RLM_FFMPEG_THREADS=1 TMPDIR="$OUT/tmp" \
    "$PY" "$RUNNER" --arrival-trace "$trace" --output "$output" \
    "${common[@]}" "${args[@]}" >"$log" 2>&1
  "$PY" - "$output/summary.json" "$expected" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
assert summary["total_requests"] == expected, summary
assert summary["errors"] == 0, summary
PY
  echo "DONE seed=$seed variant=$variant"
}

read -r -a seed_list <<<"$SEEDS"
read -r -a variant_list <<<"$VARIANTS"
if [[ -n "$SOURCE_TRACE" && ${#seed_list[@]} -ne 1 ]]; then
  echo "SOURCE_TRACE requires exactly one entry in SEEDS" >&2
  exit 2
fi
for seed in "${seed_list[@]}"; do
  trace=$OUT/traces/seed${seed}.jsonl
  if [[ -n "$SOURCE_TRACE" ]]; then
    cp "$SOURCE_TRACE" "$trace"
    sha256sum "$SOURCE_TRACE" >"$OUT/metadata/source-trace.sha256"
    continue
  fi
  read -r -a frame_count_list <<<"$FRAME_COUNTS"
  generator_args=(--output "$trace" --video "$VIDEO" \
    --aggregate-rate-qps "$RATE" --duration-s "$DURATION_S" --seed "$seed" \
    --arrival-pattern "$ARRIVAL_PATTERN" --tenant-frame-mode "$TRACE_MODE" \
    --frame-counts "${frame_count_list[@]}" \
    --step-after-requests "$STEP_AFTER_REQUESTS" \
    --step-duration-s "$STEP_DURATION_S" --step-multiplier "$STEP_MULTIPLIER")
  if [[ -n "$REQUESTS_PER_TENANT" ]]; then
    generator_args+=(--requests-per-tenant "$REQUESTS_PER_TENANT")
  fi
  "$PY" "$GENERATOR" "${generator_args[@]}"
done

# Run the requested headline configuration first when it is part of this suite.
for variant in "${variant_list[@]}"; do
  if [[ "$variant" == "$PRIORITY_VARIANT" ]]; then
    for seed in "${seed_list[@]}"; do
      run_variant "$seed" "$PRIORITY_VARIANT" "$OUT/traces/seed${seed}.jsonl"
    done
    break
  fi
done
for variant in "${variant_list[@]}"; do
  [[ "$variant" == "$PRIORITY_VARIANT" ]] && continue
  for seed in "${seed_list[@]}"; do
    run_variant "$seed" "$variant" "$OUT/traces/seed${seed}.jsonl"
  done
done

echo "completed=$(date -u +%FT%TZ)" >>"$OUT/manifest.txt"
echo "COMPONENT_ABLATION_COMPLETE output=$OUT"
