#!/usr/bin/env bash
set -euo pipefail

: "${VIDEO_RLM_ROOT:?Set VIDEO_RLM_ROOT}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON}"
: "${BASE_TRACE:?Set BASE_TRACE}"
: "${AVAILABLE_PORTS:?Set AVAILABLE_PORTS, e.g. '9000 9001'}"

REPLICAS_LIST=${REPLICAS_LIST:-"1 2"}
RUNNER="$VIDEO_RLM_ROOT/conductor/experiments/scripts/run/run_mixed_end_to_end_priority.py"
STAMP=$(date +%Y%m%d_%H%M%S)

if [[ "$VIDEO_RLM_ROOT" == /workspace/* ]]; then
  OUT="$VIDEO_RLM_ROOT/conductor/experiments/large_sweeps/proportional_replica_scaling_$STAMP"
else
  OUT="$VIDEO_RLM_ROOT/large_sweeps/proportional_replica_scaling_$STAMP"
fi

LOGROOT="$VIDEO_RLM_ROOT/logs/proportional_replica_scaling_$STAMP"
TRACE_ROOT="$OUT/traces"
mkdir -p "$OUT" "$LOGROOT" "$TRACE_ROOT"
echo "$OUT" > "$VIDEO_RLM_ROOT/logs/latest_proportional_replica_scaling.path"
read -r -a PORT_POOL <<< "$AVAILABLE_PORTS"

make_scaled_trace() {
  local replicas=$1
  local destination="$TRACE_ROOT/requests_$((80 * replicas)).jsonl"
  "$VLLM_PYTHON" - "$BASE_TRACE" "$destination" "$replicas" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
replicas = int(sys.argv[3])
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
if len(rows) != 80:
    raise SystemExit(f"Expected an 80-request base trace, found {len(rows)}: {source}")
with destination.open("w") as handle:
    for copy_index in range(replicas):
        for row_index, original in enumerate(rows):
            row = dict(original)
            original_id = str(row.get("request_id") or row.get("id") or f"request-{row_index}")
            row["request_id"] = f"{original_id}-replica-copy{copy_index}"
            handle.write(json.dumps(row) + "\n")
print(f"Created {destination}: {len(rows) * replicas} requests")
PY
}

run_one() {
  local replicas=$1
  local policy=$2
  local trace="$TRACE_ROOT/requests_$((80 * replicas)).jsonl"
  local name="replicas${replicas}_${policy}"
  local ports=("${PORT_POOL[@]:0:$replicas}")
  if (( ${#ports[@]} != replicas )); then
    echo "Need $replicas ports, but only have: ${PORT_POOL[*]}" >&2
    exit 1
  fi
  echo "START $name requests=$((80 * replicas)) ports=${ports[*]} $(date -u +%FT%TZ)"
  "$VLLM_PYTHON" "$RUNNER" \
    --arrival-trace "$trace" \
    --output "$OUT/$name" \
    --ports "${ports[@]}" \
    --replica-routing least_inflight \
    --prep-policy "$policy" \
    --prep-workers 4 \
    --vlm-concurrency 4 \
    --prepared-queue-depth 32 \
    --request-timeout-s 1800 \
    >"$LOGROOT/$name.log" 2>&1
  echo "DONE $name $(date -u +%FT%TZ)"
}

for replicas in $REPLICAS_LIST; do
  make_scaled_trace "$replicas"
  run_one "$replicas" fcfs
  run_one "$replicas" priority
done

echo "ALL_DONE"
echo "OUT=$OUT"
echo "LOGROOT=$LOGROOT"
