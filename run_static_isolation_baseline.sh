#!/usr/bin/env bash
set -euo pipefail

# Compare three systems with the same total resources:
#   1. shared_engine_only: shared 4-worker preparation, FCFS admission,
#      and two shared vLLM replicas;
#   2. shared_priority: shared 4-worker priority preparation and two shared
#      replicas; and
#   3. static_isolation: three background preparation slots plus one urgent
#      slot, with one vLLM replica dedicated to each workload class.
#
# Required environment:
#   VIDEO_RLM_ROOT, VLLM_PYTHON, TRACE_ROOT
# Optional environment:
#   PORTS="9000 9001"
#   BACKGROUND_PORTS="9000"
#   URGENT_PORTS="9001"
#   TRACE_NAMES="..."
#   POLICIES="shared_engine_only shared_priority static_isolation"
#   STATIC_ISOLATION_OUT, STATIC_ISOLATION_LOGROOT

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${TRACE_ROOT:?Set TRACE_ROOT to the directory containing source traces}"

PORTS=${PORTS:-"9000 9001"}
BACKGROUND_PORTS=${BACKGROUND_PORTS:-"9000"}
URGENT_PORTS=${URGENT_PORTS:-"9001"}
BACKGROUND_PREP_WORKERS=${BACKGROUND_PREP_WORKERS:-3}
URGENT_PREP_WORKERS=${URGENT_PREP_WORKERS:-1}
PREP_WORKERS=$((BACKGROUND_PREP_WORKERS + URGENT_PREP_WORKERS))
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
BACKGROUND_PREPARED_QUEUE_DEPTH=${BACKGROUND_PREPARED_QUEUE_DEPTH:-24}
URGENT_PREPARED_QUEUE_DEPTH=${URGENT_PREPARED_QUEUE_DEPTH:-8}
BACKGROUND_CPUSET=${BACKGROUND_CPUSET:-}
URGENT_CPUSET=${URGENT_CPUSET:-}
POLICIES=${POLICIES:-"shared_engine_only shared_priority static_isolation"}
TRACE_NAMES=${TRACE_NAMES:-"
burst_urgent10-seed1 burst_urgent10-seed2 burst_urgent10-seed3
staggered_urgent10-seed1 staggered_urgent10-seed2 staggered_urgent10-seed3
poisson_rate0.5-urgent30-seed1 poisson_rate0.5-urgent30-seed2
poisson_rate0.5-urgent30-seed3
"}

RUNNER="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${STATIC_ISOLATION_OUT:-"$VIDEO_RLM_ROOT/large_sweeps/static_isolation_baseline_$STAMP"}
LOGROOT=${STATIC_ISOLATION_LOGROOT:-"$VIDEO_RLM_ROOT/logs/$(basename "$OUT")"}

mkdir -p "$OUT/traces" "$LOGROOT"
printf '%s\n' "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_static_isolation_baseline.path"

read -r -a port_array <<<"$PORTS"
read -r -a background_port_array <<<"$BACKGROUND_PORTS"
read -r -a urgent_port_array <<<"$URGENT_PORTS"

if (( ${#port_array[@]} < 2 )); then
  echo "static isolation requires at least two vLLM replicas" >&2
  exit 2
fi

for port in "${port_array[@]}"; do
  curl -fsS --max-time 10 -H 'Authorization: Bearer EMPTY' \
    "http://127.0.0.1:$port/v1/models" >/dev/null || {
      echo "vLLM port $port is unavailable" >&2
      exit 1
    }
done

wait_for_idle() {
  while true; do
    local busy=0
    local port counts
    for port in "${port_array[@]}"; do
      counts=$(curl -fsS --max-time 5 "http://127.0.0.1:$port/metrics" | awk '
        /^vllm:num_requests_running{/ {running += $NF}
        /^vllm:num_requests_waiting{/ {waiting += $NF}
        END {printf "%.0f %.0f", running, waiting}
      ')
      if [[ "$counts" != "0 0" ]]; then
        echo "port=$port running/waiting=$counts"
        busy=1
      fi
    done
    (( busy == 0 )) && return
    sleep 2
  done
}

run_one() {
  local trace_name=$1
  local policy_name=$2
  local trace="$OUT/traces/$trace_name.jsonl"
  local output="$OUT/$trace_name/$policy_name"
  local log="$LOGROOT/$trace_name-$policy_name.log"

  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP completed $trace_name $policy_name"
    return
  fi
  if [[ -d "$output" ]]; then
    local partial="${output}.partial_$(date +%Y%m%d_%H%M%S)"
    echo "ARCHIVE_PARTIAL $output -> $partial"
    mv "$output" "$partial"
  fi

  wait_for_idle
  echo "START $trace_name $policy_name $(date -u +%FT%TZ)"

  common_args=(
    --arrival-trace "$trace"
    --output "$output"
    --ports "${port_array[@]}"
    --decode-backend seek_cpu
    --prep-workers "$PREP_WORKERS"
    --vlm-concurrency "$VLM_CONCURRENCY"
    --prepared-queue-depth "$PREPARED_QUEUE_DEPTH"
    --decode-timeout-s 600
    --request-timeout-s 1800
  )

  case "$policy_name" in
    shared_engine_only)
      policy_args=(
        --prep-policy fcfs
        --replica-routing least_inflight
      )
      ;;
    shared_priority)
      policy_args=(
        --prep-policy priority
        --replica-routing least_inflight
      )
      ;;
    static_isolation)
      policy_args=(
        --prep-policy static_isolation
        --background-prep-workers "$BACKGROUND_PREP_WORKERS"
        --urgent-prep-workers "$URGENT_PREP_WORKERS"
        --background-prepared-queue-depth "$BACKGROUND_PREPARED_QUEUE_DEPTH"
        --urgent-prepared-queue-depth "$URGENT_PREPARED_QUEUE_DEPTH"
        --replica-routing workload_isolated
        --background-ports "${background_port_array[@]}"
        --urgent-ports "${urgent_port_array[@]}"
      )
      if [[ -n "$BACKGROUND_CPUSET" || -n "$URGENT_CPUSET" ]]; then
        if [[ -z "$BACKGROUND_CPUSET" || -z "$URGENT_CPUSET" ]]; then
          echo "set both BACKGROUND_CPUSET and URGENT_CPUSET, or neither" >&2
          exit 2
        fi
        policy_args+=(
          --background-cpu-set "$BACKGROUND_CPUSET"
          --urgent-cpu-set "$URGENT_CPUSET"
        )
      fi
      ;;
    *)
      echo "unknown policy: $policy_name" >&2
      exit 2
      ;;
  esac

  VIDEO_RLM_FFMPEG_THREADS=1 \
    "$VLLM_PYTHON" "$RUNNER" \
      "${common_args[@]}" "${policy_args[@]}" \
      >"$log" 2>&1

  echo "DONE $trace_name $policy_name $(date -u +%FT%TZ)"
}

for trace_name in $TRACE_NAMES; do
  source_trace="$TRACE_ROOT/$trace_name.jsonl"
  if [[ ! -f "$source_trace" ]]; then
    echo "missing trace: $source_trace" >&2
    exit 1
  fi

  # Keep every policy matched and exclude the known corrupt RunPod video.
  jq -c '
    select((.qid // .question_id // .id // "") !=
           "20520eff-abdf-4d4f-94ad-cc751a8960d0")
  ' "$source_trace" > "$OUT/traces/$trace_name.jsonl"

  for policy in $POLICIES; do
    run_one "$trace_name" "$policy"
  done
done

echo "ALL_DONE"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
