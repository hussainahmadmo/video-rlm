#!/usr/bin/env bash
# Run a matched end-to-end load sweep. Each job owns one vLLM port/GPU while
# --concurrency controls request-level contention within that one server.

set -u
set -o pipefail

TASK_ROOT=${TASK_ROOT:-/workspace/video-rlm}
TASK_PYTHON=${TASK_PYTHON:-/workspace/vllm-mm/bin/python}
TASK_DATASET=${TASK_DATASET:-$TASK_ROOT/conductor/experiments/large_sweeps/native_vllm_metis249/dataset.jsonl}
TASK_SCHEDULE_ROOT=${TASK_SCHEDULE_ROOT:-$TASK_ROOT/conductor/experiments/large_sweeps/e2e_249}
TASK_SWEEP_ROOT=${TASK_SWEEP_ROOT:-$TASK_ROOT/conductor/experiments/large_sweeps/e2e_load_249_1gpu_$(date +%Y%m%d_%H%M%S)}

TASK_PORTS=${TASK_PORTS:-9000,9001,9002,9003}
TASK_CONCURRENCIES=${TASK_CONCURRENCIES:-"1 2 4 8 16 32"}
TASK_DECODE_WORKERS=${TASK_DECODE_WORKERS:-2}
TASK_ANSWER_FRAME_WORKERS=${TASK_ANSWER_FRAME_WORKERS:-1}
TASK_DECODE_AHEAD_BATCHES=${TASK_DECODE_AHEAD_BATCHES:-2}
TASK_DECODE_TIMEOUT_S=${TASK_DECODE_TIMEOUT_S:-60}
TASK_NATIVE_REQUEST_TIMEOUT_S=${TASK_NATIVE_REQUEST_TIMEOUT_S:-600}

IFS=',' read -r -a PORTS <<< "$TASK_PORTS"
read -r -a CONCURRENCIES <<< "$TASK_CONCURRENCIES"
METHODS=(
  native_uniform_2
  native_uniform_8
  native_uniform_32
  retrieval_budget2
  retrieval_budget8
  retrieval_budget32
  learned_adaptive
)

mkdir -p "$TASK_SWEEP_ROOT"

if [[ ! -s "$TASK_DATASET" ]]; then
  echo "missing dataset: $TASK_DATASET" >&2
  exit 2
fi

for schedule in \
  "$TASK_SCHEDULE_ROOT/schedule_fixed_budget2.jsonl" \
  "$TASK_SCHEDULE_ROOT/schedule_fixed_budget8.jsonl" \
  "$TASK_SCHEDULE_ROOT/schedule_fixed_budget32.jsonl" \
  "$TASK_SCHEDULE_ROOT/schedule_learned_adaptive.jsonl"
do
  if [[ ! -s "$schedule" ]]; then
    echo "missing schedule: $schedule" >&2
    exit 2
  fi
done

for port in "${PORTS[@]}"; do
  if ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "vLLM health check failed on port ${port}" >&2
    exit 2
  fi
done

if ! curl -fsS -I "http://127.0.0.1:8088/datasets/" >/dev/null; then
  echo "video HTTP server check failed on port 8088" >&2
  exit 2
fi

schedule_for() {
  case "$1" in
    retrieval_budget2)  printf '%s\n' "$TASK_SCHEDULE_ROOT/schedule_fixed_budget2.jsonl" ;;
    retrieval_budget8)  printf '%s\n' "$TASK_SCHEDULE_ROOT/schedule_fixed_budget8.jsonl" ;;
    retrieval_budget32) printf '%s\n' "$TASK_SCHEDULE_ROOT/schedule_fixed_budget32.jsonl" ;;
    learned_adaptive)   printf '%s\n' "$TASK_SCHEDULE_ROOT/schedule_learned_adaptive.jsonl" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local method=$1
  local concurrency=$2
  local port=$3
  local output_dir="$TASK_SWEEP_ROOT/$method/c${concurrency}"
  local started ended status

  mkdir -p "$output_dir"
  started=$(date +%s)

  echo "[start] method=${method} concurrency=${concurrency} port=${port}" | tee "$output_dir/run.log"

  if [[ "$method" == native_uniform_* ]]; then
    local frames=${method##*_}
    "$TASK_PYTHON" "$TASK_ROOT/conductor/experiments/scripts/run/run_native_vllm_video_baseline.py" \
      --dataset "$TASK_DATASET" \
      --output "$output_dir/results.jsonl" \
      --summary-output "$output_dir/summary.json" \
      --frame-counts "$frames" \
      --ports "$port" \
      --concurrency "$concurrency" \
      --video-root /workspace \
      --max-pixels 100352 \
      --max-tokens 32 \
      --request-timeout-s "$TASK_NATIVE_REQUEST_TIMEOUT_S" >> "$output_dir/run.log" 2>&1
    status=$?
  else
    local schedule
    schedule=$(schedule_for "$method")
    "$TASK_PYTHON" "$TASK_ROOT/conductor/experiments/scripts/run/run_batched_clip_streaming_vlm.py" \
      --dataset "$TASK_DATASET" \
      --schedule "$schedule" \
      --prepared-output "$output_dir/prepared.jsonl" \
      --results-output "$output_dir/results.jsonl" \
      --ports "$port" \
      --concurrency "$concurrency" \
      --decode-workers "$TASK_DECODE_WORKERS" \
      --answer-frame-workers "$TASK_ANSWER_FRAME_WORKERS" \
      --decode-ahead-batches "$TASK_DECODE_AHEAD_BATCHES" \
      --decode-timeout-s "$TASK_DECODE_TIMEOUT_S" >> "$output_dir/run.log" 2>&1
    status=$?
  fi

  ended=$(date +%s)
  printf '%s\n' "$((ended - started))" > "$output_dir/wall_seconds.txt"
  printf 'method=%s\nconcurrency=%s\nport=%s\nexit_status=%s\nstarted_epoch=%s\nended_epoch=%s\n' \
    "$method" "$concurrency" "$port" "$status" "$started" "$ended" > "$output_dir/run_status.txt"
  echo "[finish] method=${method} concurrency=${concurrency} port=${port} exit=${status} wall=$((ended-started))s" | tee -a "$output_dir/run.log"
  return "$status"
}

{
  echo "dataset=$TASK_DATASET"
  echo "ports=$TASK_PORTS"
  echo "concurrencies=${CONCURRENCIES[*]}"
  echo "decode_workers=$TASK_DECODE_WORKERS"
  echo "answer_frame_workers=$TASK_ANSWER_FRAME_WORKERS"
  echo "decode_ahead_batches=$TASK_DECODE_AHEAD_BATCHES"
  echo "decode_timeout_s=$TASK_DECODE_TIMEOUT_S"
  echo "native_request_timeout_s=$TASK_NATIVE_REQUEST_TIMEOUT_S"
  printf 'methods=%s\n' "${METHODS[*]}"
} > "$TASK_SWEEP_ROOT/manifest.txt"

overall_status=0
for concurrency in "${CONCURRENCIES[@]}"; do
  echo "[load-level] concurrency=$concurrency"
  for ((batch_start=0; batch_start<${#METHODS[@]}; batch_start+=${#PORTS[@]})); do
    pids=()
    labels=()
    for ((slot=0; slot<${#PORTS[@]}; slot++)); do
      method_index=$((batch_start + slot))
      [[ $method_index -lt ${#METHODS[@]} ]] || continue
      method=${METHODS[$method_index]}
      port=${PORTS[$slot]}
      run_one "$method" "$concurrency" "$port" &
      pids+=("$!")
      labels+=("$method@${port}")
    done
    for ((job=0; job<${#pids[@]}; job++)); do
      if ! wait "${pids[$job]}"; then
        echo "[failed] ${labels[$job]} c=$concurrency" >&2
        overall_status=1
      fi
    done
  done
done

exit "$overall_status"
