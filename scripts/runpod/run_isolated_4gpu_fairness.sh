#!/usr/bin/env bash
set -euo pipefail

# Four independent one-GPU experiment lanes on one Pod. Each lane owns whole
# physical CPU cores, one vLLM server, and one GPU. Experiment cells are spread
# across lanes; every policy for a cell runs sequentially on the same lane.

ROOT=${VIDEO_RLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${VLLM_PYTHON:-$(command -v python)}
VLLM=${VLLM_BIN:-$(command -v vllm || true)}
ACTION=${ACTION:-plan}
GPU_COUNT=${GPU_COUNT:-4}
PREP_CORES=${PREP_CORES:-8}
ENGINE_CORES=${ENGINE_CORES:-8}
BASE_PORT=${BASE_PORT:-9000}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
PREP_WORKERS=${PREP_WORKERS:-8}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
PATTERNS=${PATTERNS:-"fixed poisson bursty"}
LOADS=${LOADS:-"0.1 0.2 0.3 0.5"}
SEEDS=${SEEDS:-"1 2 3"}
POLICIES=${POLICIES:-"fcfs tenant_round_robin engine_tenant_fair prep_max_min max_min age_aware_max_min cross_stage"}
DURATION_S=${DURATION_S:-120}
SOURCE_TRACE=${SOURCE_TRACE:-}
ENGINE_TOKEN_PROFILE=${ENGINE_TOKEN_PROFILE:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/isolated_4gpu_fairness_$STAMP}
STATE_ROOT=${STATE_ROOT:-$ROOT/logs/isolated_4gpu}
TOPOLOGY_FILE=${TOPOLOGY_FILE:-$STATE_ROOT/topology.tsv}
SERVER_LOGROOT=${SERVER_LOGROOT:-$STATE_ROOT/server_logs}
SERVER_PIDROOT=${SERVER_PIDROOT:-$STATE_ROOT/server_pids}
PLANNER=$ROOT/scripts/runpod/plan_isolated_gpu_cpus.py
GENERATOR=$ROOT/conductor/experiments/scripts/run/make_crossed_arrival_trace.py
RUNNER=$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py

usage() {
  cat <<'EOF'
Usage:
  ACTION=plan  scripts/runpod/run_isolated_4gpu_fairness.sh
  ACTION=start scripts/runpod/run_isolated_4gpu_fairness.sh
  ACTION=run SOURCE_TRACE=... ENGINE_TOKEN_PROFILE=... scripts/runpod/run_isolated_4gpu_fairness.sh
  ACTION=all SOURCE_TRACE=... ENGINE_TOKEN_PROFILE=... scripts/runpod/run_isolated_4gpu_fairness.sh
  ACTION=status scripts/runpod/run_isolated_4gpu_fairness.sh
  ACTION=stop scripts/runpod/run_isolated_4gpu_fairness.sh

ACTION defaults to plan. The planner keeps SMT siblings together and prefers
CPUs on each GPU's NUMA node. Inspect topology.tsv before starting servers.
EOF
}

for command in taskset nvidia-smi curl jq; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
[[ -x "$PY" ]] || { echo "Python not found: $PY" >&2; exit 2; }
[[ -f "$PLANNER" && -f "$GENERATOR" && -f "$RUNNER" ]] || {
  echo "repository scripts are missing below $ROOT" >&2
  exit 2
}
[[ "$GPU_COUNT" == 4 ]] || {
  echo "this launcher intentionally requires GPU_COUNT=4" >&2
  exit 2
}
[[ "$PREP_WORKERS" -le "$PREP_CORES" ]] || {
  echo "PREP_WORKERS cannot exceed PREP_CORES" >&2
  exit 2
}

mkdir -p "$STATE_ROOT" "$SERVER_LOGROOT" "$SERVER_PIDROOT"

make_plan() {
  "$PY" "$PLANNER" --gpu-count "$GPU_COUNT" \
    --prep-cores "$PREP_CORES" --engine-cores "$ENGINE_CORES" \
    --base-port "$BASE_PORT" --output "$TOPOLOGY_FILE"
  echo
  echo "Topology details:"
  nvidia-smi topo -m
  echo
  echo "CPU plan written to $TOPOLOGY_FILE"
}

ensure_plan() {
  [[ -f "$TOPOLOGY_FILE" ]] || make_plan
  [[ $(awk 'END {print NR-1}' "$TOPOLOGY_FILE") -eq "$GPU_COUNT" ]] || {
    echo "invalid topology plan: $TOPOLOGY_FILE" >&2
    exit 2
  }
}

healthy() {
  curl -fsS --max-time 3 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}

start_servers() {
  ensure_plan
  [[ -n "$VLLM" && -x "$VLLM" ]] || { echo "vLLM not found: $VLLM" >&2; exit 2; }
  while IFS=$'\t' read -r gpu port _ engine_cpus _; do
    [[ "$gpu" == gpu ]] && continue
    pid_file=$SERVER_PIDROOT/gpu${gpu}.pid
    log=$SERVER_LOGROOT/gpu${gpu}.log
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null && healthy "$port"; then
      echo "REUSE_SERVER gpu=$gpu port=$port pid=$(<"$pid_file")"
      continue
    fi
    if healthy "$port"; then
      echo "port $port is occupied by an unmanaged healthy server" >&2
      echo "stop it or set a different BASE_PORT" >&2
      exit 1
    fi
    rm -f "$pid_file"
    echo "START_SERVER gpu=$gpu port=$port cpus=$engine_cpus log=$log"
    nohup setsid taskset -c "$engine_cpus" \
      env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      "$VLLM" serve "$MODEL" \
      --served-model-name "$SERVED_MODEL" --host 127.0.0.1 --port "$port" \
      --api-key EMPTY --dtype auto --tensor-parallel-size 1 \
      --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
      --max-num-seqs "$VLM_CONCURRENCY" --max-num-batched-tokens 4096 \
      --limit-mm-per-prompt '{"image":128,"video":0}' \
      --no-enable-prefix-caching --mm-processor-cache-gb 0 \
      --scheduling-policy priority </dev/null >"$log" 2>&1 &
    echo $! >"$pid_file"
  done <"$TOPOLOGY_FILE"

  for _ in $(seq 1 300); do
    ready=0
    while IFS=$'\t' read -r gpu port _; do
      [[ "$gpu" == gpu ]] && continue
      healthy "$port" && ready=$((ready + 1))
    done <"$TOPOLOGY_FILE"
    ((ready == GPU_COUNT)) && {
      echo "SERVERS_READY count=$ready"
      return 0
    }
    sleep 2
  done
  echo "server startup timed out; inspect $SERVER_LOGROOT" >&2
  return 1
}

server_status() {
  ensure_plan
  while IFS=$'\t' read -r gpu port prep_cpus engine_cpus node; do
    [[ "$gpu" == gpu ]] && continue
    pid_file=$SERVER_PIDROOT/gpu${gpu}.pid
    pid=missing
    [[ -f "$pid_file" ]] && pid=$(<"$pid_file")
    state=down
    healthy "$port" && state=healthy
    echo "gpu=$gpu port=$port state=$state pid=$pid prep=$prep_cpus engine=$engine_cpus numa=$node"
    if [[ "$pid" != missing ]] && kill -0 "$pid" 2>/dev/null; then
      taskset -pc "$pid"
    fi
  done <"$TOPOLOGY_FILE"
}

stop_servers() {
  ensure_plan
  while IFS=$'\t' read -r gpu _; do
    [[ "$gpu" == gpu ]] && continue
    pid_file=$SERVER_PIDROOT/gpu${gpu}.pid
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      args=$(ps -o args= -p "$pid")
      if [[ "$args" == *"vllm"* && "$args" == *"serve"* ]]; then
        echo "STOP_SERVER gpu=$gpu pid=$pid"
        kill "$pid"
      else
        echo "refusing to stop unexpected process pid=$pid args=$args" >&2
      fi
    fi
    rm -f "$pid_file"
  done <"$TOPOLOGY_FILE"
}

run_suite() {
  ensure_plan
  [[ -f "$SOURCE_TRACE" ]] || { echo "SOURCE_TRACE does not exist: $SOURCE_TRACE" >&2; exit 2; }
  [[ -f "$ENGINE_TOKEN_PROFILE" ]] || {
    echo "ENGINE_TOKEN_PROFILE does not exist: $ENGINE_TOKEN_PROFILE" >&2
    echo "generate a new profile on this L40S setup before running the suite" >&2
    exit 2
  }
  [[ $(server_status | grep -c 'state=healthy') -eq "$GPU_COUNT" ]] || {
    echo "all four managed vLLM servers must be healthy" >&2
    exit 1
  }

  mkdir -p "$OUT/traces" "$OUT/logs" "$OUT/tmp"
  cp "$TOPOLOGY_FILE" "$OUT/topology.tsv"
  nvidia-smi topo -m >"$OUT/nvidia-topology.txt"
  lscpu -e=CPU,CORE,SOCKET,NODE >"$OUT/cpu-topology.txt"
  {
    echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "patterns=$PATTERNS"; echo "loads=$LOADS"; echo "seeds=$SEEDS"
    echo "policies=$POLICIES"; echo "duration_s=$DURATION_S"
    echo "prep_workers=$PREP_WORKERS"; echo "vlm_concurrency=$VLM_CONCURRENCY"
    echo "prep_physical_cores=$PREP_CORES"; echo "engine_physical_cores=$ENGINE_CORES"
    echo "source_trace=$SOURCE_TRACE"; echo "engine_token_profile=$ENGINE_TOKEN_PROFILE"
    echo "started=$(date -u +%FT%TZ)"
  } >"$OUT/manifest.txt"

  cells=()
  cell_index=0
  for pattern in $PATTERNS; do
    for load in $LOADS; do
      load_tag=${load/./p}
      for seed in $SEEDS; do
        trace=$OUT/traces/${pattern}-load${load_tag}-seed${seed}.jsonl
        "$PY" "$GENERATOR" --input "$SOURCE_TRACE" --output "$trace" \
          --pattern "$pattern" --load "$load" --duration-s "$DURATION_S" --seed "$seed"
        cells+=("$cell_index|$pattern|$load|$load_tag|$seed|$trace")
        cell_index=$((cell_index + 1))
      done
    done
  done

  mapfile -t topology_rows < <(tail -n +2 "$TOPOLOGY_FILE")
  read -r -a policy_list <<<"$POLICIES"

  run_lane() {
    local lane=$1 row gpu port prep_cpus engine_cpus node
    row=${topology_rows[$lane]}
    IFS=$'\t' read -r gpu port prep_cpus engine_cpus node <<<"$row"
    mkdir -p "$OUT/tmp/lane${lane}"
    for cell in "${cells[@]}"; do
      IFS='|' read -r index pattern load load_tag seed trace <<<"$cell"
      ((index % GPU_COUNT == lane)) || continue
      base=$OUT/${pattern}-load${load_tag}-seed${seed}
      mkdir -p "$base"
      offset=$(((seed + lane) % ${#policy_list[@]}))
      for ((step=0; step<${#policy_list[@]}; step++)); do
        policy=${policy_list[$(((offset + step) % ${#policy_list[@]}))]}
        output=$base/$policy
        log=$OUT/logs/${pattern}-load${load_tag}-seed${seed}-${policy}-lane${lane}.log
        if [[ -f "$output/summary.json" ]] \
          && jq -e '.total_requests > 0 and .errors == 0' "$output/summary.json" >/dev/null; then
          echo "SKIP lane=$lane pattern=$pattern load=$load seed=$seed policy=$policy"
          continue
        fi
        [[ ! -d "$output" ]] || mv "$output" "${output}.partial_$(date +%Y%m%d_%H%M%S)"
        echo "RUN lane=$lane gpu=$gpu cpus=$prep_cpus pattern=$pattern load=$load seed=$seed policy=$policy"
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
        jq -e '.total_requests > 0 and .errors == 0' "$output/summary.json" >/dev/null
        echo "DONE lane=$lane pattern=$pattern load=$load seed=$seed policy=$policy"
      done
    done
  }

  pids=()
  for ((lane=0; lane<GPU_COUNT; lane++)); do
    run_lane "$lane" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  ((status == 0)) || { echo "one or more lanes failed; inspect $OUT/logs" >&2; return 1; }
  echo "completed=$(date -u +%FT%TZ)" >>"$OUT/manifest.txt"
  echo "ISOLATED_4GPU_SUITE_COMPLETE output=$OUT"
}

case "$ACTION" in
  plan) make_plan ;;
  start) start_servers ;;
  run) run_suite ;;
  all) start_servers; run_suite ;;
  status) server_status ;;
  stop) stop_servers ;;
  help|-h|--help) usage ;;
  *) echo "unknown ACTION=$ACTION" >&2; usage >&2; exit 2 ;;
esac
