#!/usr/bin/env bash
set -euo pipefail

# Sustained open-loop Poisson sweep. Each rate/seed/role cell runs every policy
# sequentially on one isolated GPU/CPU lane; independent cells run concurrently.

ROOT=${VIDEO_RLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${VLLM_PYTHON:-$(command -v python)}
SOURCE_TRACE=${SOURCE_TRACE:-}
ENGINE_TOKEN_PROFILE=${ENGINE_TOKEN_PROFILE:-}
TOPOLOGY_FILE=${TOPOLOGY_FILE:-$ROOT/logs/isolated_4gpu/topology.tsv}
ARRIVAL_RATES=${ARRIVAL_RATES:-"0.37 0.42 0.46"}
DURATION_S=${DURATION_S:-300}
SEEDS=${SEEDS:-"1"}
ROLE_OFFSETS=${ROLE_OFFSETS:-"0 1 2"}
POLICIES=${POLICIES:-"fcfs tenant_round_robin prep_max_min engine_tenant_fair max_min age_aware_max_min"}
PREP_WORKERS=${PREP_WORKERS:-8}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
DECODE_BACKEND=${DECODE_BACKEND:-seek_cpu}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/sustained_crossed_poisson_4gpu_$STAMP}
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_sustained_crossed_poisson_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py

for command in taskset curl jq pgrep wc; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
for required in "$PY" "$SOURCE_TRACE" "$ENGINE_TOKEN_PROFILE" \
  "$TOPOLOGY_FILE" "$GENERATOR" "$RUNNER"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
case "$DECODE_BACKEND" in
  seek_cpu|batch_cpu|batch_nvdec|indexed_nvdec) ;;
  *) echo "unsupported DECODE_BACKEND: $DECODE_BACKEND" >&2; exit 2 ;;
esac
if [[ "$DECODE_BACKEND" == *nvdec ]]; then
  "$PY" -c 'import PyNvVideoCodec' >/dev/null || {
    echo "DECODE_BACKEND=$DECODE_BACKEND requires PyNvVideoCodec" >&2
    exit 2
  }
fi
if pgrep -f '[r]un_mixed_end_to_end_priority.py' >/dev/null; then
  echo "another fairness experiment is already running; wait for it first" >&2
  pgrep -af '[r]un_mixed_end_to_end_priority.py' >&2
  exit 1
fi

mapfile -t topology_rows < <(tail -n +2 "$TOPOLOGY_FILE")
(( ${#topology_rows[@]} > 0 )) || { echo "no GPU lanes in $TOPOLOGY_FILE" >&2; exit 2; }
for row in "${topology_rows[@]}"; do
  IFS=$'\t' read -r gpu port _ <<<"$row"
  curl -fsS --max-time 3 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$port/v1/models" >/dev/null || {
      echo "vLLM is not healthy for GPU $gpu on port $port" >&2
      exit 1
    }
done

read -r -a rate_list <<<"$ARRIVAL_RATES"
read -r -a seed_list <<<"$SEEDS"
read -r -a role_list <<<"$ROLE_OFFSETS"
read -r -a policy_list <<<"$POLICIES"
mkdir -p "$OUT/traces" "$OUT/logs" "$OUT/tmp"
cp "$TOPOLOGY_FILE" "$OUT/topology.tsv"
{
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "design=sustained_equal_rate_crossed_poisson"
  echo "arrival_rates_qps=$ARRIVAL_RATES"
  echo "duration_s=$DURATION_S"
  echo "seeds=$SEEDS"
  echo "role_offsets=$ROLE_OFFSETS"
  echo "policies=$POLICIES"
  echo "prep_workers=$PREP_WORKERS"
  echo "vlm_concurrency=$VLM_CONCURRENCY"
  echo "decode_backend=$DECODE_BACKEND"
  echo "source_trace=$SOURCE_TRACE"
  echo "engine_token_profile=$ENGINE_TOKEN_PROFILE"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT/manifest.txt"
printf '%s\n' "$OUT" >"$ROOT/logs/latest_sustained_crossed_poisson_4gpu.path"

cell_rates=()
cell_rate_tags=()
cell_seeds=()
cell_roles=()
for rate in "${rate_list[@]}"; do
  rate_tag=${rate//./p}
  for seed in "${seed_list[@]}"; do
    for role in "${role_list[@]}"; do
      trace=$OUT/traces/rate${rate_tag}-seed${seed}-role${role}.jsonl
      "$PY" "$GENERATOR" --input "$SOURCE_TRACE" --output "$trace" \
        --aggregate-rate-qps "$rate" --duration-s "$DURATION_S" \
        --seed "$seed" --role-offset "$role"
      cell_rates+=("$rate")
      cell_rate_tags+=("$rate_tag")
      cell_seeds+=("$seed")
      cell_roles+=("$role")
    done
  done
done

run_policy() {
  local lane=$1 rate=$2 rate_tag=$3 seed=$4 role=$5 policy=$6
  local row gpu port prep_cpus engine_cpus node trace output log expected
  row=${topology_rows[$lane]}
  IFS=$'\t' read -r gpu port prep_cpus engine_cpus node <<<"$row"
  trace=$OUT/traces/rate${rate_tag}-seed${seed}-role${role}.jsonl
  output=$OUT/rate${rate_tag}/seed${seed}/role${role}/$policy
  log=$OUT/logs/rate${rate_tag}-seed${seed}-role${role}-${policy}-lane${lane}.log
  expected=$(wc -l <"$trace")

  if [[ -f "$output/summary.json" ]] && jq -e --argjson n "$expected" \
    '.total_requests == $n and .errors == 0' "$output/summary.json" >/dev/null; then
    echo "SKIP lane=$lane gpu=$gpu rate=$rate seed=$seed role=$role policy=$policy"
    return
  fi
  [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT/tmp/lane${lane}" "$(dirname "$output")"
  echo "RUN lane=$lane gpu=$gpu port=$port cpus=$prep_cpus rate=$rate seed=$seed role=$role policy=$policy requests=$expected"
  taskset -c "$prep_cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
    VIDEO_RLM_NVDEC_GPU_ID=0 \
    VIDEO_RLM_FFMPEG_THREADS=1 TMPDIR="$OUT/tmp/lane${lane}" \
    "$PY" "$RUNNER" --arrival-trace "$trace" --output "$output" \
    --port "$port" --prep-policy "$policy" \
    --tenant-weight a=1 --tenant-weight b=1 --tenant-weight c=1 \
    --decode-backend "$DECODE_BACKEND" --prep-workers "$PREP_WORKERS" \
    --vlm-concurrency "$VLM_CONCURRENCY" \
    --prepared-queue-depth "$PREPARED_QUEUE_DEPTH" \
    --decode-timeout-s 600 --request-timeout-s 1800 --ignore-eos \
    --engine-token-profile "$ENGINE_TOKEN_PROFILE" >"$log" 2>&1
  jq -e --argjson n "$expected" \
    '.total_requests == $n and .errors == 0' "$output/summary.json" >/dev/null
  echo "DONE lane=$lane gpu=$gpu rate=$rate seed=$seed role=$role policy=$policy"
}

run_lane() {
  local lane=$1 index rate rate_tag seed role offset step policy
  for ((index=lane; index<${#cell_rates[@]}; index+=${#topology_rows[@]})); do
    rate=${cell_rates[$index]}
    rate_tag=${cell_rate_tags[$index]}
    seed=${cell_seeds[$index]}
    role=${cell_roles[$index]}
    offset=$((index % ${#policy_list[@]}))
    for ((step=0; step<${#policy_list[@]}; step++)); do
      policy=${policy_list[$(((offset + step) % ${#policy_list[@]}))]}
      run_policy "$lane" "$rate" "$rate_tag" "$seed" "$role" "$policy"
    done
  done
}

pids=()
for ((lane=0; lane<${#topology_rows[@]}; lane++)); do
  run_lane "$lane" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
((status == 0)) || { echo "one or more sustained-Poisson lanes failed" >&2; exit 1; }

echo "completed=$(date -u +%FT%TZ)" >>"$OUT/manifest.txt"
echo "SUSTAINED_CROSSED_POISSON_SUITE_COMPLETE output=$OUT"
