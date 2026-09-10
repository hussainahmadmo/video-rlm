#!/usr/bin/env bash
set -euo pipefail

# Sweep preparation-heavy leader burst sizes. Each cell runs every policy
# sequentially on one isolated lane; cells run concurrently across GPU lanes.

ROOT=${VIDEO_RLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${VLLM_PYTHON:-$(command -v python)}
SOURCE_TRACE=${SOURCE_TRACE:-}
ENGINE_TOKEN_PROFILE=${ENGINE_TOKEN_PROFILE:-}
TOPOLOGY_FILE=${TOPOLOGY_FILE:-$ROOT/logs/isolated_4gpu/topology.tsv}
LEADER_BURSTS=${LEADER_BURSTS:-"20 40 60 80 100"}
SEEDS=${SEEDS:-"1"}
POLICIES=${POLICIES:-"fcfs max_min"}
FOLLOWER_INFERENCE_REQUESTS=${FOLLOWER_INFERENCE_REQUESTS:-20}
FOLLOWER_LIGHT_REQUESTS=${FOLLOWER_LIGHT_REQUESTS:-20}
EPISODE_SPACING_S=${EPISODE_SPACING_S:-450}
PREP_WORKERS=${PREP_WORKERS:-8}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/phase_shifted_leader_sweep_$STAMP}
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
read -r -a policy_list <<<"$POLICIES"
mkdir -p "$OUT/traces" "$OUT/logs" "$OUT/tmp"
cp "$TOPOLOGY_FILE" "$OUT/topology.tsv"
{
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "leader_bursts=$LEADER_BURSTS"
  echo "seeds=$SEEDS"
  echo "policies=$POLICIES"
  echo "follower_inference_requests=$FOLLOWER_INFERENCE_REQUESTS"
  echo "follower_light_requests=$FOLLOWER_LIGHT_REQUESTS"
  echo "episode_spacing_s=$EPISODE_SPACING_S"
  echo "source_trace=$SOURCE_TRACE"
  echo "engine_token_profile=$ENGINE_TOKEN_PROFILE"
  echo "started=$(date -u +%FT%TZ)"
} >"$OUT/manifest.txt"

cell_seeds=()
cell_bursts=()
for seed in "${seed_list[@]}"; do
  for burst in "${burst_list[@]}"; do
    trace=$OUT/traces/leader${burst}-seed${seed}.jsonl
    "$PY" "$GENERATOR" --input "$SOURCE_TRACE" --output "$trace" \
      --seed "$seed" --prep-heavy-requests "$burst" \
      --inference-heavy-requests "$FOLLOWER_INFERENCE_REQUESTS" \
      --light-requests "$FOLLOWER_LIGHT_REQUESTS" \
      --episode-spacing-s "$EPISODE_SPACING_S"
    cell_seeds+=("$seed")
    cell_bursts+=("$burst")
  done
done

run_policy() {
  local lane=$1 seed=$2 burst=$3 policy=$4
  local row gpu port prep_cpus engine_cpus node trace output log expected
  row=${topology_rows[$lane]}
  IFS=$'\t' read -r gpu port prep_cpus engine_cpus node <<<"$row"
  trace=$OUT/traces/leader${burst}-seed${seed}.jsonl
  output=$OUT/leader${burst}/seed${seed}/$policy
  log=$OUT/logs/leader${burst}-seed${seed}-${policy}-lane${lane}.log
  expected=$((3 * (burst + FOLLOWER_INFERENCE_REQUESTS + FOLLOWER_LIGHT_REQUESTS)))

  if [[ -f "$output/summary.json" ]] && jq -e --argjson n "$expected" \
    '.total_requests == $n and .errors == 0' "$output/summary.json" >/dev/null; then
    echo "SKIP lane=$lane gpu=$gpu leader=$burst seed=$seed policy=$policy"
    return
  fi
  [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT/tmp/lane${lane}" "$(dirname "$output")"
  echo "RUN lane=$lane gpu=$gpu port=$port cpus=$prep_cpus leader=$burst seed=$seed policy=$policy"
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
  echo "DONE lane=$lane gpu=$gpu leader=$burst seed=$seed policy=$policy"
}

run_lane() {
  local lane=$1 index seed burst offset step policy
  for ((index=lane; index<${#cell_seeds[@]}; index+=${#topology_rows[@]})); do
    seed=${cell_seeds[$index]}
    burst=${cell_bursts[$index]}
    offset=$((index % ${#policy_list[@]}))
    for ((step=0; step<${#policy_list[@]}; step++)); do
      policy=${policy_list[$(((offset + step) % ${#policy_list[@]}))]}
      run_policy "$lane" "$seed" "$burst" "$policy"
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
((status == 0)) || { echo "one or more sweep lanes failed" >&2; exit 1; }

echo "completed=$(date -u +%FT%TZ)" >>"$OUT/manifest.txt"
echo "PHASE_SHIFTED_LEADER_SWEEP_COMPLETE output=$OUT"
