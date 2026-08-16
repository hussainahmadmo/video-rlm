#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}"

DATASET="${DATASET:-$ROOT/conductor/experiments/diverse_eval/agent_friendly_egoschema_nextqa_intentqa.jsonl}"
OUT_DIR="${OUT_DIR:-$ROOT/conductor/experiments/large_sweeps/resource_aware_fixed_$(date +%Y%m%d_%H%M%S)}"
PORTS="${PORTS:-9000}"
CLIP_DEVICE="${CLIP_DEVICE:-cpu}"
DECORD_CTX="${DECORD_CTX:-cpu}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
FORCE_TIER="${FORCE_TIER:-}"
SELECTOR_MODEL="${SELECTOR_MODEL:-}"
NO_HARD_QUERY_UPGRADE="${NO_HARD_QUERY_UPGRADE:-0}"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$ROOT/decord/python:$ROOT"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CLIP_DEVICE
export DECORD_CTX
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

SCHEDULE="$OUT_DIR/resource_aware_schedule.jsonl"
SCHEDULER_ARGS=(
  --dataset "$DATASET"
  --output "$SCHEDULE"
)

if [ -n "$FORCE_TIER" ]; then
  SCHEDULER_ARGS+=(--force-tier "$FORCE_TIER")
fi

if [ -n "$SELECTOR_MODEL" ]; then
  SCHEDULER_ARGS+=(--selector-model "$SELECTOR_MODEL")
fi

if [ "$NO_HARD_QUERY_UPGRADE" = "1" ]; then
  SCHEDULER_ARGS+=(--no-hard-query-upgrade)
fi

echo "[schedule] dataset=$DATASET"
echo "[schedule] output=$SCHEDULE"
"$PYTHON_BIN" "$ROOT/conductor/experiments/scripts/build/build_resource_aware_schedule.py" \
  "${SCHEDULER_ARGS[@]}"

IFS=',' read -r -a PORT_ARRAY <<< "$PORTS"

if [ "${#PORT_ARRAY[@]}" -eq 1 ]; then
  export VLM_BASE_URL="http://localhost:${PORT_ARRAY[0]}/v1"
  EXTRA_ARGS=()
  if [ -n "$MAX_EXAMPLES" ]; then
    EXTRA_ARGS+=(--max_examples "$MAX_EXAMPLES")
  fi

  "$PYTHON_BIN" "$ROOT/conductor/experiments/scripts/run/run_synthesis_grid_ultra.py" \
    --dataset "$DATASET" \
    --output "$OUT_DIR/results.jsonl" \
    --config_file "$ROOT/conductor/experiments/configs/oracle_candidate_limited.json" \
    --profiler_json "$SCHEDULE" \
    "${EXTRA_ARGS[@]}"

  echo "[done] $OUT_DIR/results.jsonl"
  exit 0
fi

echo "[split] ports=${PORTS}"
"$PYTHON_BIN" - "$DATASET" "$SCHEDULE" "$OUT_DIR" "${#PORT_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
schedule = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
parts = int(sys.argv[4])

rows = [
    json.loads(line)
    for line in dataset.read_text().splitlines()
    if line.strip()
]
schedule_rows = [
    json.loads(line)
    for line in schedule.read_text().splitlines()
    if line.strip()
]
schedule_by_qid = {
    str(row.get("qid") or row.get("id")): row
    for row in schedule_rows
}

for idx in range(parts):
    part_rows = rows[idx::parts]
    part_schedule = [
        schedule_by_qid[str(row.get("qid") or row.get("id"))]
        for row in part_rows
    ]
    dataset_path = out_dir / f"dataset.part{idx}.jsonl"
    schedule_path = out_dir / f"schedule.part{idx}.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in part_rows)
    )
    schedule_path.write_text(
        "".join(json.dumps(row) + "\n" for row in part_schedule)
    )
    print(
        f"[split] {dataset_path} rows={len(part_rows)} "
        f"schedule={schedule_path}"
    )
PY

pids=()
for idx in "${!PORT_ARRAY[@]}"; do
  port="${PORT_ARRAY[$idx]}"
  export VLM_BASE_URL="http://localhost:${port}/v1"
  dataset_part="$OUT_DIR/dataset.part${idx}.jsonl"
  schedule_part="$OUT_DIR/schedule.part${idx}.jsonl"
  output_part="$OUT_DIR/results.part${idx}.jsonl"

  (
    "$PYTHON_BIN" "$ROOT/conductor/experiments/scripts/run/run_synthesis_grid_ultra.py" \
      --dataset "$dataset_part" \
      --output "$output_part" \
      --config_file "$ROOT/conductor/experiments/configs/oracle_candidate_limited.json" \
      --profiler_json "$schedule_part"
  ) > "$OUT_DIR/run.part${idx}.log" 2>&1 &
  pids+=("$!")
  echo "[start] part=$idx port=$port pid=${pids[-1]} output=$output_part"
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON_BIN" - "$OUT_DIR" "${#PORT_ARRAY[@]}" <<'PY'
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
parts = int(sys.argv[2])
merged = out_dir / "results.jsonl"
with merged.open("w") as out:
    for idx in range(parts):
        part = out_dir / f"results.part{idx}.jsonl"
        if part.exists():
            out.write(part.read_text())
print(f"[done] {merged}")
PY
