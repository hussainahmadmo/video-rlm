#!/usr/bin/env bash
set -euo pipefail

# Equal-count crossed-resource experiment. A matched seed/role cell runs every
# policy sequentially on one isolated GPU/CPU lane; independent cells use the
# four lanes concurrently.

ROOT=${VIDEO_RLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${VLLM_PYTHON:-$(command -v python)}
SOURCE_TRACE=${SOURCE_TRACE:-}
ENGINE_TOKEN_PROFILE=${ENGINE_TOKEN_PROFILE:-}
TOPOLOGY_FILE=${TOPOLOGY_FILE:-$ROOT/logs/isolated_4gpu/topology.tsv}
SEEDS=${SEEDS:-"1 2 3"}
ROLE_OFFSETS=${ROLE_OFFSETS:-"0 1 2"}
POLICIES=${POLICIES:-"fcfs tenant_round_robin prep_max_min engine_tenant_fair max_min age_aware_max_min"}
REQUESTS_PER_TENANT=${REQUESTS_PER_TENANT:-40}
ARRIVAL_WINDOW_S=${ARRIVAL_WINDOW_S:-15}
PREP_WORKERS=${PREP_WORKERS:-8}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
DECODE_BACKEND=${DECODE_BACKEND:-seek_cpu}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/simultaneous_crossed_4gpu_$STAMP}
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_simultaneous_crossed_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py

for command in taskset curl jq pgrep; do
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

read -r -a seed_list <<<"$SEEDS"
read -r -a role_list <<<"$ROLE_OFFSETS"
read -r -a policy_list <<<"$POLICIES"
mkdir -p "$OUT/traces" "$OUT/logs" "$OUT/tmp"
cp "$TOPOLOGY_FILE" "$OUT/topology.tsv"
{
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "design=simultaneous_equal_count_crossed"
  echo "seeds=$SEEDS"
  echo "role_offsets=$ROLE_OFFSETS"
  echo "policies=$POLICIES"
  echo "requests_per_tenant=$REQUESTS_PER_TENANT"
  echo "arrival_window_s=$ARRIVAL_WINDOW_S"
  echo "prep_workers=$PREP_WORKERS"
  echo "vlm_concurrency=$VLM_CONCURRENCY"
  echo "decode_backend=$DECODE_BACKEND"
  echo "source_trace=$SOURCE_TRACE"
  echo "engine_token_profile=$ENGINE_TOKEN_PROFILE"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT/manifest.txt"

cell_seeds=()
cell_roles=()
for seed in "${seed_list[@]}"; do
  for role in "${role_list[@]}"; do
    trace=$OUT/traces/simultaneous-seed${seed}-role${role}.jsonl
    "$PY" "$GENERATOR" --input "$SOURCE_TRACE" --output "$trace" \
      --seed "$seed" --role-offset "$role" \
      --requests-per-tenant "$REQUESTS_PER_TENANT" \
      --arrival-window-s "$ARRIVAL_WINDOW_S"
    cell_seeds+=("$seed")
    cell_roles+=("$role")
  done
done

run_policy() {
  local lane=$1 seed=$2 role=$3 policy=$4
  local row gpu port prep_cpus engine_cpus node trace output log expected
  row=${topology_rows[$lane]}
  IFS=$'\t' read -r gpu port prep_cpus engine_cpus node <<<"$row"
  trace=$OUT/traces/simultaneous-seed${seed}-role${role}.jsonl
  output=$OUT/seed${seed}/role${role}/$policy
  log=$OUT/logs/seed${seed}-role${role}-${policy}-lane${lane}.log
  expected=$((3 * REQUESTS_PER_TENANT))

  if [[ -f "$output/summary.json" ]] && jq -e --argjson n "$expected" \
    '.total_requests == $n and .errors == 0' "$output/summary.json" >/dev/null; then
    echo "SKIP lane=$lane gpu=$gpu seed=$seed role=$role policy=$policy"
    return
  fi
  [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT/tmp/lane${lane}" "$(dirname "$output")"
  echo "RUN lane=$lane gpu=$gpu port=$port cpus=$prep_cpus seed=$seed role=$role policy=$policy"
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
  echo "DONE lane=$lane gpu=$gpu seed=$seed role=$role policy=$policy"
}

run_lane() {
  local lane=$1 index seed role offset step policy
  for ((index=lane; index<${#cell_seeds[@]}; index+=${#topology_rows[@]})); do
    seed=${cell_seeds[$index]}
    role=${cell_roles[$index]}
    offset=$((index % ${#policy_list[@]}))
    for ((step=0; step<${#policy_list[@]}; step++)); do
      policy=${policy_list[$(((offset + step) % ${#policy_list[@]}))]}
      run_policy "$lane" "$seed" "$role" "$policy"
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
((status == 0)) || { echo "one or more simultaneous-crossed lanes failed" >&2; exit 1; }

echo "completed=$(date -u +%FT%TZ)" >>"$OUT/manifest.txt"
echo "SIMULTANEOUS_CROSSED_SUITE_COMPLETE output=$OUT"
