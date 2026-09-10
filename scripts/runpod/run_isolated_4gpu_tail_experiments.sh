#!/usr/bin/env bash
set -euo pipefail

# Run the balanced transient-burst experiment and its matched age-threshold
# sweep using servers/topology created by run_isolated_4gpu_fairness.sh.

ROOT=${VIDEO_RLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${VLLM_PYTHON:-$(command -v python)}
SOURCE_TRACE=${SOURCE_TRACE:-}
ENGINE_TOKEN_PROFILE=${ENGINE_TOKEN_PROFILE:-}
TOPOLOGY_FILE=${TOPOLOGY_FILE:-$ROOT/logs/isolated_4gpu/topology.tsv}
EXPERIMENTS=${EXPERIMENTS:-"balanced threshold"}
SEEDS=${SEEDS:-"1 2 3"}
POLICIES=${POLICIES:-"fcfs tenant_round_robin engine_tenant_fair prep_max_min max_min age_aware_max_min cross_stage"}
THRESHOLD_PAIRS=${THRESHOLD_PAIRS:-"45:90 60:105 75:120 75:150 90:150"}
PREP_WORKERS=${PREP_WORKERS:-8}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/isolated_4gpu_tail_$STAMP}
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_balanced_transient_burst_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py

for command in taskset curl jq; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
for required in "$PY" "$SOURCE_TRACE" "$ENGINE_TOKEN_PROFILE" \
  "$TOPOLOGY_FILE" "$GENERATOR" "$RUNNER"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done

mapfile -t topology_rows < <(tail -n +2 "$TOPOLOGY_FILE")
(( ${#topology_rows[@]} >= 3 )) || {
  echo "at least three isolated GPU lanes are required" >&2
  exit 2
}

healthy() {
  curl -fsS --max-time 3 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}

for row in "${topology_rows[@]}"; do
  IFS=$'\t' read -r gpu port _ <<<"$row"
  healthy "$port" || { echo "vLLM is not healthy for GPU $gpu on port $port" >&2; exit 1; }
done

mkdir -p "$OUT/traces" "$OUT/logs" "$OUT/tmp"
cp "$TOPOLOGY_FILE" "$OUT/topology.tsv"
{
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "experiments=$EXPERIMENTS"; echo "seeds=$SEEDS"
  echo "policies=$POLICIES"; echo "threshold_pairs=$THRESHOLD_PAIRS"
  echo "prep_workers=$PREP_WORKERS"; echo "vlm_concurrency=$VLM_CONCURRENCY"
  echo "source_trace=$SOURCE_TRACE"; echo "engine_token_profile=$ENGINE_TOKEN_PROFILE"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT/manifest.txt"

contains_experiment() {
  [[ " $EXPERIMENTS " == *" $1 "* ]]
}

run_one() {
  local lane=$1 policy=$2 trace=$3 output=$4 log=$5 soft=$6 hard=$7
  local row gpu port prep_cpus engine_cpus node
  row=${topology_rows[$lane]}
  IFS=$'\t' read -r gpu port prep_cpus engine_cpus node <<<"$row"
  if [[ -f "$output/summary.json" ]] \
    && jq -e '.total_requests > 0 and .errors == 0' "$output/summary.json" >/dev/null; then
    echo "SKIP lane=$lane policy=$policy output=$output"
    return 0
  fi
  [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT/tmp/lane${lane}" "$(dirname "$log")"
  echo "RUN lane=$lane gpu=$gpu cpus=$prep_cpus policy=$policy trace=$(basename "$trace")"
  taskset -c "$prep_cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
    VIDEO_RLM_FFMPEG_THREADS=1 TMPDIR="$OUT/tmp/lane${lane}" \
    "$PY" "$RUNNER" --arrival-trace "$trace" --output "$output" \
    --port "$port" --prep-policy "$policy" \
    --tenant-weight a=1 --tenant-weight b=1 --tenant-weight c=1 \
    --decode-backend seek_cpu --prep-workers "$PREP_WORKERS" \
    --vlm-concurrency "$VLM_CONCURRENCY" \
    --prepared-queue-depth "$PREPARED_QUEUE_DEPTH" \
    --decode-timeout-s 600 --request-timeout-s 1800 --ignore-eos \
    --engine-token-profile "$ENGINE_TOKEN_PROFILE" \
    --age-soft-threshold-s "$soft" --age-hard-threshold-s "$hard" \
    >"$log" 2>&1
  jq -e '.total_requests > 0 and .errors == 0' "$output/summary.json" >/dev/null
  echo "DONE lane=$lane policy=$policy output=$output"
}

read -r -a seed_list <<<"$SEEDS"
read -r -a policy_list <<<"$POLICIES"
(( ${#seed_list[@]} <= ${#topology_rows[@]} )) || {
  echo "SEEDS contains more simultaneous seeds than available lanes" >&2
  exit 2
}

for seed in "${seed_list[@]}"; do
  trace=$OUT/traces/balanced-burst-seed${seed}.jsonl
  "$PY" "$GENERATOR" --input "$SOURCE_TRACE" --output "$trace" --seed "$seed"
done

run_seed() {
  local lane=$1
  local seed=$2
  local trace=$OUT/traces/balanced-burst-seed${seed}.jsonl
  local offset step policy pair soft hard tag
  if contains_experiment balanced; then
    offset=$(((seed + lane) % ${#policy_list[@]}))
    for ((step=0; step<${#policy_list[@]}; step++)); do
      policy=${policy_list[$(((offset + step) % ${#policy_list[@]}))]}
      run_one "$lane" "$policy" "$trace" \
        "$OUT/balanced/seed${seed}/$policy" \
        "$OUT/logs/balanced-seed${seed}-${policy}.log" 75 150
    done
  fi
  if contains_experiment threshold; then
    for pair in $THRESHOLD_PAIRS; do
      soft=${pair%%:*}
      hard=${pair##*:}
      tag=soft${soft}_hard${hard}
      run_one "$lane" age_aware_max_min "$trace" \
        "$OUT/threshold/seed${seed}/$tag" \
        "$OUT/logs/threshold-seed${seed}-${tag}.log" "$soft" "$hard"
    done
  fi
}

pids=()
for ((lane=0; lane<${#seed_list[@]}; lane++)); do
  run_seed "$lane" "${seed_list[$lane]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
((status == 0)) || { echo "one or more tail-experiment lanes failed" >&2; exit 1; }

echo "completed=$(date -u +%FT%TZ)" >>"$OUT/manifest.txt"
echo "ISOLATED_4GPU_TAIL_EXPERIMENTS_COMPLETE output=$OUT"
