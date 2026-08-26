#!/usr/bin/env bash
set -euo pipefail

# Profile native vLLM raw-video serving while running both priority modes.
# Required: an already-running vLLM server and HTTP video server.
# Override TRACE, PORT, VIDEO_ROOT, VIDEO_BASE_URL, or PY as needed.

ROOT=${VIDEO_RLM_ROOT:-/dataheart/hussainahmad/video-rlm}
PY=${VLLM_PYTHON:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}
RUNNER="$ROOT/conductor/experiments/scripts/run/run_native_vllm_priority_burst.py"
TRACE=${TRACE:-$ROOT/large_sweeps/asplos_remaining_2gpu_20260825_002540/static_isolation/traces/burst_urgent10-seed1.jsonl}
PORT=${PORT:-9000}
VIDEO_ROOT=${VIDEO_ROOT:-/dataheart/hussainahmad}
VIDEO_BASE_URL=${VIDEO_BASE_URL:-http://127.0.0.1:8089}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$ROOT/large_sweeps/native_priority_cpu_profile_$STAMP}
LOGROOT=${LOGROOT:-$ROOT/logs/native_priority_cpu_profile_$STAMP}

for command in curl jq mpstat pidstat ps ss; do
    command -v "$command" >/dev/null || {
        echo "Required command is missing: $command" >&2
        exit 1
    }
done
test -x "$PY" || { echo "Python not executable: $PY" >&2; exit 1; }
test -f "$RUNNER" || { echo "Runner missing: $RUNNER" >&2; exit 1; }
test -f "$TRACE" || { echo "Trace missing: $TRACE" >&2; exit 1; }
curl -sf --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "vLLM is unavailable on port $PORT" >&2
    exit 1
}
curl -sf --max-time 10 "$VIDEO_BASE_URL/" >/dev/null || {
    echo "Video HTTP server is unavailable at $VIDEO_BASE_URL" >&2
    exit 1
}

API_PID=$(ss -ltnp | sed -n "/:$PORT /s/.*pid=\([0-9][0-9]*\).*/\1/p" | head -1)
test -n "$API_PID" || {
    echo "Could not resolve the vLLM API PID listening on port $PORT" >&2
    exit 1
}

# Read Linux's process-child list recursively. Unlike parsing pstree output,
# this includes child processes but excludes individual thread IDs.
collect_process_tree() {
    local parent_pid=$1
    local child_pid
    echo "$parent_pid"
    if test -r "/proc/$parent_pid/task/$parent_pid/children"; then
        for child_pid in $(cat "/proc/$parent_pid/task/$parent_pid/children"); do
            collect_process_tree "$child_pid"
        done
    fi
}

PROFILE_PIDS=$(collect_process_tree "$API_PID" | sort -nu | paste -sd, -)
test -n "$PROFILE_PIDS" || PROFILE_PIDS=$API_PID

mkdir -p "$OUT" "$LOGROOT"
echo "$OUT" > "$ROOT/logs/latest_native_priority_cpu_profile.path"

{
    echo "timestamp=$STAMP"
    echo "root=$ROOT"
    echo "trace=$TRACE"
    echo "port=$PORT"
    echo "video_root=$VIDEO_ROOT"
    echo "video_base_url=$VIDEO_BASE_URL"
    echo "api_pid=$API_PID"
    echo "profile_pids=$PROFILE_PIDS"
    echo "logical_cpus=$(nproc)"
    echo
    lscpu
    echo
    ps -o pid,ppid,nlwp,psr,pcpu,pmem,etime,cmd -p "$PROFILE_PIDS"
    echo
    grep -E 'Threads|Cpus_allowed_list' "/proc/$API_PID/status"
    echo
    command -v pstree >/dev/null && pstree -ap "$API_PID"
} > "$LOGROOT/system_and_process_metadata.log"

PIDSTAT_PID=
MPSTAT_PID=

stop_monitors() {
    for monitor_pid in "$PIDSTAT_PID" "$MPSTAT_PID"; do
        if test -n "$monitor_pid" && kill -0 "$monitor_pid" 2>/dev/null; then
            kill -TERM "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
        fi
    done
    PIDSTAT_PID=
    MPSTAT_PID=
}
trap stop_monitors EXIT

run_policy() {
    local label=$1
    local priority_mode=$2

    echo "START $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    pidstat -h -u -r -w -p "$PROFILE_PIDS" 1 \
        > "$LOGROOT/$label-pidstat.log" 2>&1 &
    PIDSTAT_PID=$!
    mpstat -P ALL 1 > "$LOGROOT/$label-mpstat.log" 2>&1 &
    MPSTAT_PID=$!

    "$PY" "$RUNNER" \
        --arrival-trace "$TRACE" \
        --output "$OUT/$label" \
        --port "$PORT" \
        --priority-mode "$priority_mode" \
        --video-root "$VIDEO_ROOT" \
        --video-base-url "$VIDEO_BASE_URL" \
        --request-timeout-s 1800 \
        > "$LOGROOT/$label-workload.log" 2>&1

    stop_monitors
    echo "DONE $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# Reverse the earlier experiment's order to expose potential warm-cache bias.
run_policy trace_priority trace
sleep 10
run_policy uniform_priority uniform

for policy in trace_priority uniform_priority; do
    jq -c '{priority_mode,total_requests,errors,wall_time_s,throughput_qps,urgent,background}' \
        "$OUT/$policy/summary.json"
done | tee "$LOGROOT/result_summaries.jsonl"

echo "ALL DONE"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
