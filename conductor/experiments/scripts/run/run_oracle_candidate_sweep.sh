#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/dataheart/hussainahmad/miniconda3/envs/vllm-mm/bin/python}"

DATASET="${DATASET:-$ROOT/conductor/experiments/diverse_eval/all_datasets_diverse_available.jsonl}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/conductor/experiments/configs/oracle_candidate_limited.json}"
OUT_DIR="${OUT_DIR:-$ROOT/conductor/experiments/large_sweeps/oracle_candidate_$(date +%Y%m%d_%H%M%S)}"
CLIP_DEVICE="${CLIP_DEVICE:-cpu}"
PORTS="${PORTS:-9000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

mkdir -p "$OUT_DIR"

echo "[audit] dataset=$DATASET"
echo "[audit] config_file=$CONFIG_FILE"
echo "[audit] out_dir=$OUT_DIR"

"$PYTHON_BIN" "$ROOT/conductor/experiments/scripts/build/audit_sweep_inputs.py" \
  --dataset "$DATASET" \
  --config_file "$CONFIG_FILE"

export PYTHONPATH="$ROOT/decord/python:$ROOT"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CLIP_DEVICE
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

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
    --config_file "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}"

  echo "[done] $OUT_DIR/results.jsonl"
  exit 0
fi

echo "[split] ports=${PORTS}"
"$PYTHON_BIN" - "$DATASET" "$OUT_DIR" "${#PORT_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
parts = int(sys.argv[3])

rows = [
    json.loads(line)
    for line in dataset.read_text().splitlines()
    if line.strip()
]

for idx in range(parts):
    part_rows = rows[idx::parts]
    path = out_dir / f"dataset.part{idx}.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in part_rows)
    )
    print(f"[split] {path} rows={len(part_rows)}")
PY

pids=()
for idx in "${!PORT_ARRAY[@]}"; do
  port="${PORT_ARRAY[$idx]}"
  export VLM_BASE_URL="http://localhost:${port}/v1"
  dataset_part="$OUT_DIR/dataset.part${idx}.jsonl"
  output_part="$OUT_DIR/results.part${idx}.jsonl"

  (
    "$PYTHON_BIN" "$ROOT/conductor/experiments/scripts/run/run_synthesis_grid_ultra.py" \
      --dataset "$dataset_part" \
      --output "$output_part" \
      --config_file "$CONFIG_FILE"
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
