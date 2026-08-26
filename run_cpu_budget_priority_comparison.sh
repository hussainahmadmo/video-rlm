#!/usr/bin/env bash
set -euo pipefail

# Resource-matched native-versus-external priority experiment.
#
# This launcher starts a fresh vLLM server for every CPU budget, pins vLLM,
# its EngineCore, the video HTTP server, and each workload runner to the same
# logical-CPU set, and dynamically profiles their complete process trees.
# Stop any server already occupying GPU_ID before launching this suite.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
GPU_ID=${GPU_ID:-0}
SERVER_PORT=${SERVER_PORT:-9020}
VIDEO_PORT=${VIDEO_PORT:-8095}
CPU_BUDGETS=${CPU_BUDGETS:-"16 64"}
REPEATS=${REPEATS:-3}
POLICIES=${POLICIES:-"native_uniform native_priority external_fcfs external_priority"}
TRACE=${TRACE:-$ROOT/large_sweeps/asplos_remaining_2gpu_20260825_002540/static_isolation/traces/burst_urgent10-seed1.jsonl}
VIDEO_ROOT=${VIDEO_ROOT:-/dataheart/hussainahmad}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
PREP_WORKERS=${PREP_WORKERS:-4}
VLM_CONCURRENCY=${VLM_CONCURRENCY:-4}
PREPARED_QUEUE_DEPTH=${PREPARED_QUEUE_DEPTH:-32}
COOLDOWN_S=${COOLDOWN_S:-10}
MONITOR_INTERVAL_S=${MONITOR_INTERVAL_S:-0.2}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/cpu_budget_priority_comparison_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/$(basename "$OUT")}

SERVER_LAUNCHER="$ROOT/run_vllm_cpu_limited.sh"
NATIVE_RUNNER="$ROOT/conductor/experiments/scripts/run/run_native_vllm_priority_burst.py"
EXTERNAL_RUNNER="$ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
MONITOR="$ROOT/conductor/experiments/scripts/analyze/monitor_process_tree_cpu.py"
ANALYZER="$ROOT/conductor/experiments/scripts/analyze/analyze_cpu_budget_priority.py"

for command in curl jq ss taskset; do
    command -v "$command" >/dev/null || {
        echo "Required command is missing: $command" >&2
        exit 1
    }
done
for path in "$PY" "$SERVER_LAUNCHER" "$NATIVE_RUNNER" "$EXTERNAL_RUNNER" "$MONITOR"; do
    test -e "$path" || { echo "Required path is missing: $path" >&2; exit 1; }
done
test -f "$TRACE" || { echo "Trace missing: $TRACE" >&2; exit 1; }
if ss -ltn | grep -qE ":($SERVER_PORT|$VIDEO_PORT)[[:space:]]"; then
    echo "Port $SERVER_PORT or $VIDEO_PORT is already in use" >&2
    exit 1
fi

mkdir -p "$OUT/traces" "$LOGROOT"
echo "$OUT" > "$ROOT/logs/latest_cpu_budget_priority_comparison.path"

# Apply the known corrupt-video exclusion once, before every policy sees the
# trace. If the QID is absent, jq simply reproduces the original trace.
jq -c '
  select((.qid // .question_id // .id // "") !=
         "20520eff-abdf-4d4f-94ad-cc751a8960d0")
' "$TRACE" > "$OUT/traces/workload.jsonl"
TRACE="$OUT/traces/workload.jsonl"

cpuset_for_budget() {
    local budget=$1
    local override="CPUSET_$budget"
    if [[ -n ${!override:-} ]]; then
        printf '%s\n' "${!override}"
        return
    fi
    case "$budget" in
        4) printf '0-3\n' ;;
        8) printf '0-7\n' ;;
        16) printf '0-15\n' ;;
        # On this 2x16-core host, 32-47 are SMT siblings of 0-15.
        32) printf '0-15,32-47\n' ;;
        64) printf '0-63\n' ;;
        *)
            echo "No default CPU set for budget=$budget; set CPUSET_$budget" >&2
            return 2
            ;;
    esac
}

VLLM_LAUNCH_PID=
VIDEO_PID=
cleanup_services() {
    local pid
    for pid in "$VLLM_LAUNCH_PID" "$VIDEO_PID"; do
        if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    VLLM_LAUNCH_PID=
    VIDEO_PID=
}
trap cleanup_services EXIT INT TERM

wait_for_server() {
    local deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if curl -sf --max-time 5 "http://127.0.0.1:$SERVER_PORT/health" >/dev/null; then
            return
        fi
        if [[ -n $VLLM_LAUNCH_PID ]] && ! kill -0 "$VLLM_LAUNCH_PID" 2>/dev/null; then
            echo "vLLM exited during startup" >&2
            return 1
        fi
        sleep 10
    done
    echo "Timed out waiting for vLLM" >&2
    return 1
}

wait_for_idle() {
    while true; do
        local counts
        counts=$(curl -sf --max-time 5 "http://127.0.0.1:$SERVER_PORT/metrics" | awk '
          /^vllm:num_requests_running{/ {running += $NF}
          /^vllm:num_requests_waiting{/ {waiting += $NF}
          END {printf "%.0f %.0f", running, waiting}
        ')
        [[ $counts == "0 0" ]] && return
        echo "vLLM running/waiting=$counts"
        sleep 2
    done
}

run_one() {
    local budget=$1 cpuset=$2 repeat=$3 policy=$4 order_index=$5
    local run_dir="$OUT/cpu$budget/repeat$repeat/$policy"
    local run_log="$LOGROOT/cpu$budget-repeat$repeat-$policy.log"
    local cpu_samples="$LOGROOT/cpu$budget-repeat$repeat-$policy-cpu.jsonl"
    local cpu_summary="$run_dir/cpu_summary.json"
    local cpu_summary_tmp="$LOGROOT/cpu$budget-repeat$repeat-$policy-cpu-summary.json"

    if [[ -f $run_dir/summary.json && -f $cpu_summary ]]; then
        echo "SKIP completed cpu=$budget repeat=$repeat policy=$policy"
        return
    fi
    if [[ -d $run_dir ]]; then
        mv "$run_dir" "${run_dir}.partial_$(date +%Y%m%d_%H%M%S)"
    fi
    # Both workload runners intentionally require their output directory not
    # to exist. Create only its parent; write monitor output under LOGROOT
    # until the runner has created run_dir.
    mkdir -p "$(dirname "$run_dir")"
    wait_for_idle
    echo "START cpu=$budget repeat=$repeat order=$order_index policy=$policy $(date -u +%FT%TZ)"

    local -a command
    case "$policy" in
        native_uniform|native_priority)
            local mode=uniform
            [[ $policy == native_priority ]] && mode=trace
            command=(
                "$PY" "$NATIVE_RUNNER"
                --arrival-trace "$TRACE"
                --output "$run_dir"
                --port "$SERVER_PORT"
                --priority-mode "$mode"
                --video-root "$VIDEO_ROOT"
                --video-base-url "http://127.0.0.1:$VIDEO_PORT"
                --request-timeout-s 1800
            )
            ;;
        external_fcfs|external_priority|external_reserved)
            local prep_policy=fcfs
            [[ $policy == external_priority ]] && prep_policy=priority
            [[ $policy == external_reserved ]] && prep_policy=priority_reserved
            command=(
                "$PY" "$EXTERNAL_RUNNER"
                --arrival-trace "$TRACE"
                --output "$run_dir"
                --port "$SERVER_PORT"
                --prep-policy "$prep_policy"
                --prep-workers "$PREP_WORKERS"
                --vlm-concurrency "$VLM_CONCURRENCY"
                --prepared-queue-depth "$PREPARED_QUEUE_DEPTH"
                --decode-backend seek_cpu
                --decode-timeout-s 600
                --request-timeout-s 1800
            )
            ;;
        *) echo "Unknown policy: $policy" >&2; return 2 ;;
    esac

    taskset --cpu-list "$cpuset" env \
        VIDEO_RLM_FFMPEG_THREADS=1 \
        OMP_NUM_THREADS="$budget" \
        MKL_NUM_THREADS="$budget" \
        OPENBLAS_NUM_THREADS="$budget" \
        NUMEXPR_NUM_THREADS="$budget" \
        "${command[@]}" > "$run_log" 2>&1 &
    local workload_pid=$!

    local roots="$API_PID,$workload_pid"
    if [[ $policy == native_* ]]; then
        roots="$roots,$VIDEO_PID"
    fi
    "$PY" "$MONITOR" \
        --roots "$roots" \
        --until-pid "$workload_pid" \
        --samples "$cpu_samples" \
        --summary "$cpu_summary_tmp" \
        --interval-s "$MONITOR_INTERVAL_S" &
    local monitor_pid=$!

    local status=0
    wait "$workload_pid" || status=$?
    wait "$monitor_pid" || true
    if (( status != 0 )); then
        echo "FAILED cpu=$budget repeat=$repeat policy=$policy status=$status" >&2
        tail -50 "$run_log" >&2 || true
        return "$status"
    fi

    "$PY" - "$run_dir/summary.json" "$cpu_summary_tmp" "$budget" "$cpuset" \
        "$repeat" "$order_index" "$policy" <<'PY'
import json, pathlib, sys
summary_path, cpu_path = map(pathlib.Path, sys.argv[1:3])
summary = json.loads(summary_path.read_text())
cpu = json.loads(cpu_path.read_text())
cpu["configured_cpu_budget"] = int(sys.argv[3])
cpu["cpu_set"] = sys.argv[4]
cpu["repeat"] = int(sys.argv[5])
cpu["order_index"] = int(sys.argv[6])
cpu["policy"] = sys.argv[7]
requests = int(summary.get("total_requests") or 0)
cpu["cpu_seconds_per_request"] = (
    cpu["cpu_seconds"] / requests if requests else None
)
cpu_path.write_text(json.dumps(cpu, indent=2) + "\n")
PY
    cp "$cpu_summary_tmp" "$cpu_summary"

    echo "DONE cpu=$budget repeat=$repeat policy=$policy $(date -u +%FT%TZ)"
    sleep "$COOLDOWN_S"
}

read -r -a policy_array <<< "$POLICIES"
for budget in $CPU_BUDGETS; do
    cpuset=$(cpuset_for_budget "$budget")
    budget_log="$LOGROOT/cpu$budget"
    mkdir -p "$budget_log"

    echo "START_SERVICES cpu=$budget cpuset=$cpuset $(date -u +%FT%TZ)"
    taskset --cpu-list "$cpuset" "$PY" -m http.server "$VIDEO_PORT" \
        --bind 127.0.0.1 --directory "$VIDEO_ROOT" \
        > "$budget_log/video-http.log" 2>&1 &
    VIDEO_PID=$!
    sleep 2
    curl -sf --max-time 5 "http://127.0.0.1:$VIDEO_PORT/" >/dev/null || {
        echo "Video server failed for cpu=$budget" >&2
        exit 1
    }

    env CPUSET="$cpuset" THREADS="$budget" GPU="$GPU_ID" \
        PORT="$SERVER_PORT" MODEL="$MODEL" \
        "$SERVER_LAUNCHER" > "$budget_log/vllm.log" 2>&1 &
    VLLM_LAUNCH_PID=$!
    wait_for_server || { tail -100 "$budget_log/vllm.log" >&2; exit 1; }
    API_PID=$(ss -ltnp | sed -n "/:$SERVER_PORT /s/.*pid=\([0-9][0-9]*\).*/\1/p" | head -1)
    [[ -n $API_PID ]] || { echo "Could not resolve vLLM API PID" >&2; exit 1; }
    taskset -pc "$API_PID" > "$budget_log/affinity.txt"

    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        if (( repeat % 2 == 1 )); then
            ordered=("${policy_array[@]}")
        else
            ordered=()
            for ((i=${#policy_array[@]}-1; i>=0; i--)); do
                ordered+=("${policy_array[i]}")
            done
        fi
        order_index=0
        for policy in "${ordered[@]}"; do
            order_index=$((order_index + 1))
            run_one "$budget" "$cpuset" "$repeat" "$policy" "$order_index"
        done
    done

    cleanup_services
    sleep 10
done

if [[ -f $ANALYZER ]]; then
    "$PY" "$ANALYZER" --root "$OUT" --output "$OUT/cpu_budget_results.csv"
fi

echo "ALL_DONE"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
