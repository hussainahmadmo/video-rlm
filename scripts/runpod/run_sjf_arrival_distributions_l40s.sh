#!/usr/bin/env bash
set -euo pipefail

# Run matched arrival-distribution comparisons. Each argument is
# WORKLOAD:PATTERN:SEED:ORDER. Workloads are heavy or mixed; patterns are
# poisson, lognormal, or hyperexponential; orders are forward/reverse/rotate.

ROOT=${VIDEO_RLM_ROOT:-/workspace/video-rlm-joint-controller}
PY=${VLLM_PYTHON:-/workspace/venv/bin/python}
VIDEO=${VIDEO:-$ROOT/data/kN88RP3XWUU.mp4}
LANE_PROFILE=${LANE_PROFILE:-$ROOT/data/l40s_lane_profile.json}
PORT=${PORT:-8000}
CPUSET=${CPUSET:-0-7}
RATE=${RATE:-1.0}
SCV=${SCV:-4.0}
REQUESTS_PER_TENANT=${REQUESTS_PER_TENANT:-42}
POLICY_MODE=${POLICY_MODE:-sjf}
OUT_ROOT=${OUT_ROOT:-/workspace/results/sjf_arrival_distributions_$(date -u +%Y%m%d_%H%M%S)}
GENERATOR=$ROOT/analysis/make_sjf_arrival_distribution_traces.py
LAUNCHER=$ROOT/scripts/runpod/run_end_to_end_components_l40s.sh

if (( $# == 0 )); then
  echo "usage: $0 {heavy|mixed}:{poisson|lognormal|hyperexponential}:SEED:{forward|reverse|rotate} [...]" >&2
  exit 2
fi
for required in "$PY" "$VIDEO" "$LANE_PROFILE" "$GENERATOR" "$LAUNCHER"; do
  test -e "$required" || { echo "missing: $required" >&2; exit 2; }
done

case "$POLICY_MODE" in
  sjf)
    forward="full_fcfs full_prep_sjf full_prep_sjf_aging"
    reverse="full_prep_sjf_aging full_prep_sjf full_fcfs"
    rotate="full_prep_sjf full_fcfs full_prep_sjf_aging"
    policies="fcfs prep_sjf prep_sjf_aging"
    ;;
  all)
    forward="full_fcfs full_tenant_round_robin full_prep_sjf full_prep_sjf_aging full_conductor_strict_fifo"
    reverse="full_conductor_strict_fifo full_prep_sjf_aging full_prep_sjf full_tenant_round_robin full_fcfs"
    rotate="full_prep_sjf full_conductor_strict_fifo full_fcfs full_prep_sjf_aging full_tenant_round_robin"
    policies="fcfs tenant_round_robin prep_sjf prep_sjf_aging conductor_strict_fifo"
    ;;
  *) echo "unknown POLICY_MODE: $POLICY_MODE" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT"/{traces,logs}
{
  echo "design=matched_sjf_arrival_distributions"
  echo "rate_qps=$RATE"
  echo "bursty_scv=$SCV"
  echo "requests_per_tenant=$REQUESTS_PER_TENANT"
  echo "frame_counts=1 16 128"
  echo "placement=joint_predicted_readiness"
  echo "capacity=fixed"
  echo "policies=$policies"
  echo "jobs=$*"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT_ROOT/manifest.txt"

for specification in "$@"; do
  IFS=: read -r workload pattern seed order <<<"$specification"
  case "$workload" in heavy|mixed) ;; *) echo "unknown workload: $workload" >&2; exit 2 ;; esac
  case "$pattern" in poisson|lognormal|hyperexponential) ;; *) echo "unknown pattern: $pattern" >&2; exit 2 ;; esac
  case "$order" in
    forward) variants=$forward ;;
    reverse) variants=$reverse ;;
    rotate) variants=$rotate ;;
    *) echo "unknown order: $order" >&2; exit 2 ;;
  esac

  trace_root=$OUT_ROOT/traces/${workload}_seed${seed}
  if [[ ! -f "$trace_root/$pattern.jsonl" ]]; then
    "$PY" "$GENERATOR" --output "$trace_root" --video "$VIDEO" \
      --workload "$workload" --seed "$seed" --rate-qps "$RATE" \
      --scv "$SCV" --requests-per-tenant "$REQUESTS_PER_TENANT" \
      >"$OUT_ROOT/logs/generate-${workload}-seed${seed}.log"
  fi

  priority=${variants%% *}
  output=$OUT_ROOT/${workload}_${pattern}_seed${seed}_${order}
  echo "START workload=$workload pattern=$pattern seed=$seed order=$order output=$output"
  env VIDEO_RLM_ROOT="$ROOT" VLLM_PYTHON="$PY" VIDEO="$VIDEO" \
    LANE_PROFILE="$LANE_PROFILE" PORT="$PORT" CPUSET="$CPUSET" \
    SEEDS="$seed" VARIANTS="$variants" PRIORITY_VARIANT="$priority" \
    SOURCE_TRACE="$trace_root/$pattern.jsonl" OUT="$output" \
    bash "$LAUNCHER"
  echo "DONE workload=$workload pattern=$pattern seed=$seed order=$order"
done

echo "completed=$(date -u +%FT%TZ)" >>"$OUT_ROOT/manifest.txt"
touch "$OUT_ROOT/COMPLETE"
echo "SJF_ARRIVAL_DISTRIBUTIONS_COMPLETE output=$OUT_ROOT"
