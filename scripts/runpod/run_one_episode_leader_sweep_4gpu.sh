#!/usr/bin/env bash
set -euo pipefail

# Fast phase-shifted burst sweep: each trace contains one heavy-leader/light-
# follower episode. Three role offsets restore tenant balance across traces.
# Every policy pair stays on the same isolated GPU/CPU lane.

ROOT=${VIDEO_RLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${VLLM_PYTHON:-$(command -v python)}
SOURCE_TRACE=${SOURCE_TRACE:-}
ENGINE_TOKEN_PROFILE=${ENGINE_TOKEN_PROFILE:-}
TOPOLOGY_FILE=${TOPOLOGY_FILE:-$ROOT/logs/isolated_4gpu/topology.tsv}
LEADER_BURSTS=${LEADER_BURSTS:-"20 40 60 80 100"}
SEEDS=${SEEDS:-"1"}
ROLE_OFFSETS=${ROLE_OFFSETS:-"0 1 2"}
POLICIES=${POLICIES:-"fcfs max_min"}
FOLLOWER_INFERENCE_REQUESTS=${FOLLOWER_INFERENCE_REQUESTS:-0}
FOLLOWER_LIGHT_REQUESTS=${FOLLOWER_LIGHT_REQUESTS:-20}
PREP_WORKERS=${PREP_WORKERS:-8}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/one_episode_leader_sweep_$STAMP}
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_phase_shifted_raw_tail_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py

for command in taskset curl jq pgrep; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
for required in "$PY" "$SOURCE_TRACE" "$ENGINE_TOKEN_PROFILE" \
  "$TOPOLOGY_FILE" "$GENERATOR" "$RUNNER"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
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

read -r -a burst_list <<<"$LEADER_BURSTS"
read -r -a seed_list <<<"$SEEDS"
read -r -a role_list <<<"$ROLE_OFFSETS"
read -r -a policy_list <<<"$POLICIES"
mkdir -p "$OUT/traces" "$OUT/logs" "$OUT/tmp"
cp "$TOPOLOGY_FILE" "$OUT/topology.tsv"
{
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "design=one_episode_per_trace_matched_role_rotation"
  echo "leader_bursts=$LEADER_BURSTS"
  echo "seeds=$SEEDS"
  echo "role_offsets=$ROLE_OFFSETS"
  echo "policies=$POLICIES"
  echo "follower_inference_requests=$FOLLOWER_INFERENCE_REQUESTS"
  echo "follower_light_requests=$FOLLOWER_LIGHT_REQUESTS"
  echo "source_trace=$SOURCE_TRACE"
  echo "engine_token_profile=$ENGINE_TOKEN_PROFILE"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT/manifest.txt"

cell_seeds=()
cell_bursts=()
cell_roles=()
for seed in "${seed_list[@]}"; do
  for burst in "${burst_list[@]}"; do
    for role in "${role_list[@]}"; do
      trace=$OUT/traces/leader${burst}-seed${seed}-role${role}.jsonl
      "$PY" "$GENERATOR" --input "$SOURCE_TRACE" --output "$trace" \
        --seed "$seed" --episodes 1 --role-offset "$role" \
        --allow-unbalanced-episodes --prep-heavy-requests "$burst" \
        --inference-heavy-requests "$FOLLOWER_INFERENCE_REQUESTS" \
        --light-requests "$FOLLOWER_LIGHT_REQUESTS"
      cell_seeds+=("$seed")
      cell_bursts+=("$burst")
      cell_roles+=("$role")
    done
  done
done

run_policy() {
  local lane=$1 seed=$2 burst=$3 role=$4 policy=$5
  local row gpu port prep_cpus engine_cpus node trace output log expected
  row=${topology_rows[$lane]}
  IFS=$'\t' read -r gpu port prep_cpus engine_cpus node <<<"$row"
  trace=$OUT/traces/leader${burst}-seed${seed}-role${role}.jsonl
  output=$OUT/leader${burst}/seed${seed}/role${role}/$policy
  log=$OUT/logs/leader${burst}-seed${seed}-role${role}-${policy}-lane${lane}.log
  expected=$((burst + FOLLOWER_INFERENCE_REQUESTS + FOLLOWER_LIGHT_REQUESTS))

  if [[ -f "$output/summary.json" ]] && jq -e --argjson n "$expected" \
    '.total_requests == $n and .errors == 0' "$output/summary.json" >/dev/null; then
    echo "SKIP lane=$lane gpu=$gpu leader=$burst seed=$seed role=$role policy=$policy"
    return
  fi
  [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT/tmp/lane${lane}" "$(dirname "$output")"
  echo "RUN lane=$lane gpu=$gpu port=$port cpus=$prep_cpus leader=$burst seed=$seed role=$role policy=$policy"
  taskset -c "$prep_cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
    VIDEO_RLM_FFMPEG_THREADS=1 TMPDIR="$OUT/tmp/lane${lane}" \
    "$PY" "$RUNNER" --arrival-trace "$trace" --output "$output" \
    --port "$port" --prep-policy "$policy" \
    --tenant-weight a=1 --tenant-weight b=1 --tenant-weight c=1 \
    --decode-backend seek_cpu --prep-workers "$PREP_WORKERS" \
    --vlm-concurrency "$VLM_CONCURRENCY" \
    --prepared-queue-depth "$PREPARED_QUEUE_DEPTH" \
    --decode-timeout-s 600 --request-timeout-s 1800 --ignore-eos \
    --engine-token-profile "$ENGINE_TOKEN_PROFILE" >"$log" 2>&1
  jq -e --argjson n "$expected" \
    '.total_requests == $n and .errors == 0' "$output/summary.json" >/dev/null
  echo "DONE lane=$lane gpu=$gpu leader=$burst seed=$seed role=$role policy=$policy"
}

run_lane() {
  local lane=$1 index seed burst role offset step policy
  for ((index=lane; index<${#cell_seeds[@]}; index+=${#topology_rows[@]})); do
    seed=${cell_seeds[$index]}
    burst=${cell_bursts[$index]}
    role=${cell_roles[$index]}
    offset=$((index % ${#policy_list[@]}))
    for ((step=0; step<${#policy_list[@]}; step++)); do
      policy=${policy_list[$(((offset + step) % ${#policy_list[@]}))]}
      run_policy "$lane" "$seed" "$burst" "$role" "$policy"
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
((status == 0)) || { echo "one or more one-episode lanes failed" >&2; exit 1; }

echo "completed=$(date -u +%FT%TZ)" >>"$OUT/manifest.txt"
echo "ONE_EPISODE_LEADER_SWEEP_COMPLETE output=$OUT"
