#!/usr/bin/env python3
"""Measure end-to-end priority across external video preparation and vLLM.

Background videos arrive first and urgent videos arrive later.  The preparation
dispatcher can be FCFS or priority ordered.  Prepared requests retain the same
priority when submitted to a vLLM server using ``--scheduling-policy priority``.

Unlike submitting every preparation task to ThreadPoolExecutor immediately,
this runner keeps an explicit pending heap and admits at most ``--prep-workers``
tasks.  A newly arrived urgent request can therefore overtake background work
that has not begun preparation.  Running decode work is intentionally
non-preemptive.
"""

from __future__ import annotations

import argparse
import base64
import heapq
import importlib.util
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[4]
CODEC_PATH = (
    ROOT
    / "conductor/experiments/scripts/run/run_codec_guided_vllm_baseline.py"
)
WRITE_LOCK = threading.Lock()


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def qid(row: dict[str, Any]) -> str:
    value = row.get("qid") or row.get("question_id") or row.get("id")
    if value is None:
        raise ValueError("row has no qid/question_id/id")
    return str(value)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_label(text: str | None, choice_count: int) -> str | None:
    if not text:
        return None
    valid = [chr(ord("A") + index) for index in range(choice_count)]
    normalized = text.strip().upper()
    if normalized in valid:
        return normalized
    match = re.search(r"\b(" + "|".join(valid) + r")\b", normalized)
    return match.group(1) if match else None


def make_codec_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        max_tokens=args.max_tokens,
        max_pixels=args.max_pixels,
        window_s=8.0,
        decode_max_side=args.decode_max_side,
        index_timeout_s=args.index_timeout_s,
        decode_timeout_s=args.decode_timeout_s,
        request_timeout_s=args.request_timeout_s,
        refine_with_clip=False,
        clip_device="cpu",
        refine_candidates=1,
        refine_regions=1,
        refine_radius_s=1.0,
        probe_max_side=args.decode_max_side,
    )


def prepare_uniform(
    job: dict[str, Any], codec: Any, codec_args: SimpleNamespace
) -> dict[str, Any]:
    """Decode uniformly sampled JPEGs outside vLLM."""
    row = job["row"]
    video = Path(row["video"])
    started = time.perf_counter()
    duration_s = codec.probe_duration(video, codec_args.index_timeout_s)
    timestamps = codec.temporal_anchors(duration_s, job["frame_count"])
    content: list[dict[str, Any]] = []
    for timestamp in timestamps:
        jpeg = codec.decode_jpeg(
            video,
            timestamp,
            codec_args.decode_max_side,
            codec_args.decode_timeout_s,
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(jpeg).decode("ascii")
                },
            }
        )
    content.append({"type": "text", "text": codec.make_prompt(row)})
    return {
        "content": content,
        "duration_s": duration_s,
        "timestamps": timestamps,
        "decode_service_s": time.perf_counter() - started,
    }


def call_vllm(
    job: dict[str, Any],
    prepared: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
) -> dict[str, Any]:
    """Stream one prepared request so time to first token is observable."""
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=args.request_timeout_s,
        max_retries=0,
    )
    submitted_at = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    error: str | None = None
    try:
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prepared["content"]}],
            temperature=0.0,
            max_tokens=args.max_tokens,
            stream=True,
            extra_body={
                "priority": job["priority"],
                "mm_processor_kwargs": {"max_pixels": args.max_pixels},
            },
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(delta)
    except Exception as exc:
        error = repr(exc)
    completed_at = time.perf_counter()
    return {
        "text": "".join(pieces),
        "error": error,
        "ttft_from_vllm_submit_s": (
            None if first_token_at is None else first_token_at - submitted_at
        ),
        "vlm_service_s": completed_at - submitted_at,
        "first_token_offset_s": (
            None if first_token_at is None else first_token_at - submitted_at
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "prep_queue_wait_s",
        "prep_service_s",
        "prepared_queue_wait_s",
        "engine_to_first_token_s",
        "end_to_end_ttft_s",
        "end_to_end_s",
    )
    summary: dict[str, Any] = {
        "requests": len(rows),
        "errors": sum(row.get("error") is not None for row in rows),
        "correct": sum(bool(row.get("correct")) for row in rows),
        "accuracy_percent": (
            100.0 * sum(bool(row.get("correct")) for row in rows) / len(rows)
            if rows
            else None
        ),
    }
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[f"mean_{field}"] = mean(values)
        summary[f"p50_{field}"] = percentile(values, 0.50)
        summary[f"p95_{field}"] = percentile(values, 0.95)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-dataset", type=Path)
    parser.add_argument("--urgent-dataset", type=Path)
    parser.add_argument("--arrival-trace", type=Path, help="Per-request JSONL arrival trace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--prep-policy", choices=["fcfs", "priority"], default="priority")
    parser.add_argument("--background-requests", type=int, default=32)
    parser.add_argument("--urgent-requests", type=int, default=16)
    parser.add_argument("--background-arrival-s", type=float, default=0.0)
    parser.add_argument("--urgent-arrival-s", type=float, default=1.0)
    parser.add_argument("--background-frames", type=int, default=128)
    parser.add_argument("--urgent-frames", type=int, default=8)
    parser.add_argument("--background-priority", type=int, default=10)
    parser.add_argument("--urgent-priority", type=int, default=0)
    parser.add_argument("--prep-workers", type=int, default=4)
    parser.add_argument("--vlm-concurrency", type=int, default=4)
    parser.add_argument("--prepared-queue-depth", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--decode-max-side", type=int, default=448)
    parser.add_argument("--index-timeout-s", type=float, default=120.0)
    parser.add_argument("--decode-timeout-s", type=float, default=60.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    if min(args.prep_workers, args.vlm_concurrency, args.prepared_queue_depth) < 1:
        parser.error("worker and queue counts must be positive")
    if args.arrival_trace is None:
        if args.background_dataset is None or args.urgent_dataset is None:
            parser.error("provide --arrival-trace, or both dataset arguments")
        if min(args.background_requests, args.urgent_requests) < 1:
            parser.error("request counts must be positive")
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")

    background_rows: list[dict[str, Any]] = []
    urgent_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    if args.arrival_trace is not None:
        trace_rows = load_jsonl(args.arrival_trace)
        if not trace_rows:
            raise SystemExit("arrival trace is empty")
    else:
        background_rows = load_jsonl(args.background_dataset)[: args.background_requests]
        urgent_rows = load_jsonl(args.urgent_dataset)[: args.urgent_requests]
        if len(background_rows) != args.background_requests:
            raise SystemExit("background dataset has fewer rows than requested")
        if len(urgent_rows) != args.urgent_requests:
            raise SystemExit("urgent dataset has fewer rows than requested")

    codec = import_path("mixed_priority_codec", CODEC_PATH)
    codec_args = make_codec_args(args)
    base_url = f"http://127.0.0.1:{args.port}/v1"
    readiness_client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=min(args.request_timeout_s, 30.0),
        max_retries=0,
    )
    readiness_client.models.list()

    args.output.mkdir(parents=True)
    results_path = args.output / "results.jsonl"
    events_path = args.output / "events.jsonl"
    started = time.perf_counter()

    arrivals: list[tuple[float, int, dict[str, Any]]] = []

    def add_job(row, workload, arrival_s, frames, priority, request_id):
        nonlocal_sequence = len(arrivals)
        job = {
            "request_id": request_id, "workload": workload, "row": row,
            "arrival_s": float(arrival_s), "frame_count": int(frames),
            "priority": int(priority), "sequence": nonlocal_sequence,
        }
        if job["arrival_s"] < 0 or job["frame_count"] < 1:
            raise SystemExit(f"invalid trace job: {request_id}")
        heapq.heappush(arrivals, (job["arrival_s"], nonlocal_sequence, job))

    if trace_rows:
        for index, trace_row in enumerate(trace_rows):
            row = dict(trace_row)
            workload = str(row.pop("class", row.pop("workload", "background")))
            arrival_s = row.pop("arrival_s", 0.0)
            frames = row.pop("frame_count", 8)
            priority = row.pop("priority", 0)
            request_id = str(row.pop("request_id", f"{workload}-{index}"))
            add_job(row, workload, arrival_s, frames, priority, request_id)
    else:
        for workload, rows, arrival_s, frames, priority in (
            ("background", background_rows, args.background_arrival_s, args.background_frames, args.background_priority),
            ("urgent", urgent_rows, args.urgent_arrival_s, args.urgent_frames, args.urgent_priority),
        ):
            for index, row in enumerate(rows):
                add_job(row, workload, arrival_s, frames, priority, f"{workload}-{index}")

    pending: list[tuple[int, int, dict[str, Any]]] = []
    ready: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    prep_futures: dict[Future, dict[str, Any]] = {}
    vlm_futures: dict[Future, tuple[dict[str, Any], dict[str, Any]]] = {}
    completed: list[dict[str, Any]] = []

    def elapsed() -> float:
        return time.perf_counter() - started

    def scheduling_key(job: dict[str, Any]) -> int:
        return job["priority"] if args.prep_policy == "priority" else 0

    def event(name: str, job: dict[str, Any], **extra: Any) -> None:
        append_jsonl(
            events_path,
            {
                "event": name,
                "time_s": elapsed(),
                "request_id": job["request_id"],
                "workload": job["workload"],
                "priority": job["priority"],
                "frame_count": job["frame_count"],
                **extra,
            },
        )

    with ThreadPoolExecutor(max_workers=args.prep_workers) as prep_pool, ThreadPoolExecutor(
        max_workers=args.vlm_concurrency
    ) as vlm_pool:
        while arrivals or pending or prep_futures or ready or vlm_futures:
            now = elapsed()
            while arrivals and arrivals[0][0] <= now:
                _, _, job = heapq.heappop(arrivals)
                heapq.heappush(
                    pending, (scheduling_key(job), job["sequence"], job)
                )
                event("arrival", job, pending_depth=len(pending))

            while (
                pending
                and len(prep_futures) < args.prep_workers
                and len(ready) + len(prep_futures) < args.prepared_queue_depth
            ):
                _, _, job = heapq.heappop(pending)
                job["prep_started_s"] = elapsed()
                event("prep_start", job, pending_depth=len(pending))
                prep_futures[prep_pool.submit(prepare_uniform, job, codec, codec_args)] = job

            while ready and len(vlm_futures) < args.vlm_concurrency:
                _, _, job, prepared = heapq.heappop(ready)
                job["vlm_submit_s"] = elapsed()
                event("vlm_submit", job, ready_depth=len(ready))
                vlm_futures[
                    vlm_pool.submit(call_vllm, job, prepared, args, base_url)
                ] = (job, prepared)

            futures = list(prep_futures) + list(vlm_futures)
            if not futures:
                if arrivals:
                    time.sleep(max(0.0, min(0.05, arrivals[0][0] - elapsed())))
                continue

            done, _ = wait(
                futures, timeout=0.05, return_when=FIRST_COMPLETED
            )
            for future in done:
                if future in prep_futures:
                    job = prep_futures.pop(future)
                    job["prep_ready_s"] = elapsed()
                    try:
                        prepared = future.result()
                    except Exception as exc:
                        result = {
                            "request_id": job["request_id"],
                            "workload": job["workload"],
                            "qid": qid(job["row"]),
                            "priority": job["priority"],
                            "frame_count": job["frame_count"],
                            "arrival_s": job["arrival_s"],
                            "prep_started_s": job["prep_started_s"],
                            "prep_ready_s": job["prep_ready_s"],
                            "error": repr(exc),
                            "correct": False,
                            "prep_queue_wait_s": job["prep_started_s"] - job["arrival_s"],
                            "prep_service_s": job["prep_ready_s"] - job["prep_started_s"],
                            "prepared_queue_wait_s": None,
                            "engine_to_first_token_s": None,
                            "end_to_end_ttft_s": None,
                            "end_to_end_s": job["prep_ready_s"] - job["arrival_s"],
                        }
                        completed.append(result)
                        append_jsonl(results_path, result)
                        event("prep_error", job, error=repr(exc))
                        continue
                    event("prep_ready", job, ready_depth=len(ready) + 1)
                    heapq.heappush(
                        ready,
                        (
                            scheduling_key(job),
                            job["sequence"],
                            job,
                            prepared,
                        ),
                    )
                else:
                    job, prepared = vlm_futures.pop(future)
                    job["completion_s"] = elapsed()
                    response = future.result()
                    first_token_s = (
                        None
                        if response["first_token_offset_s"] is None
                        else job["vlm_submit_s"]
                        + response["first_token_offset_s"]
                    )
                    row = job["row"]
                    prediction = parse_label(
                        response["text"], len(row.get("choices") or [])
                    )
                    gold = row.get("answer_label")
                    if gold is None and row.get("answer_idx") is not None:
                        gold = chr(ord("A") + int(row["answer_idx"]))
                    result = {
                        "request_id": job["request_id"],
                        "workload": job["workload"],
                        "qid": qid(row),
                        "video": row.get("video"),
                        "priority": job["priority"],
                        "frame_count": job["frame_count"],
                        "arrival_s": job["arrival_s"],
                        "prep_started_s": job["prep_started_s"],
                        "prep_ready_s": job["prep_ready_s"],
                        "vlm_submit_s": job["vlm_submit_s"],
                        "first_token_s": first_token_s,
                        "completion_s": job["completion_s"],
                        "prep_queue_wait_s": job["prep_started_s"] - job["arrival_s"],
                        "prep_service_s": job["prep_ready_s"] - job["prep_started_s"],
                        "prepared_queue_wait_s": job["vlm_submit_s"] - job["prep_ready_s"],
                        "engine_to_first_token_s": response["ttft_from_vllm_submit_s"],
                        "end_to_end_ttft_s": (
                            None
                            if first_token_s is None
                            else first_token_s - job["arrival_s"]
                        ),
                        "end_to_end_s": job["completion_s"] - job["arrival_s"],
                        "duration_s": prepared["duration_s"],
                        "selected_timestamps_s": prepared["timestamps"],
                        "prediction_label": prediction,
                        "prediction_text": response["text"],
                        "answer_label": gold,
                        "correct": prediction == gold,
                        "error": response["error"],
                    }
                    completed.append(result)
                    append_jsonl(results_path, result)
                    event("completion", job, error=response["error"])
                    print(
                        f"[done] {job['request_id']} priority={job['priority']} "
                        f"prep_wait={result['prep_queue_wait_s']:.2f} "
                        f"ttft={result['end_to_end_ttft_s']} "
                        f"error={response['error'] is not None}",
                        flush=True,
                    )

    wall_s = elapsed()
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_workload[row["workload"]].append(row)
    summary = {
        "method": "mixed_end_to_end_video_priority",
        "prep_policy": args.prep_policy,
        "server_requirement": "--scheduling-policy priority",
        "port": args.port,
        "wall_time_s": wall_s,
        "total_requests": len(completed),
        "errors": sum(row.get("error") is not None for row in completed),
        "throughput_qps": len(completed) / wall_s if wall_s else 0.0,
        "prep_workers": args.prep_workers,
        "vlm_concurrency": args.vlm_concurrency,
        "prepared_queue_depth": args.prepared_queue_depth,
        "background": summarize(by_workload["background"]),
        "urgent": summarize(by_workload["urgent"]),
        "configuration": {
            "arrival_trace": str(args.arrival_trace) if args.arrival_trace else None,
            "trace_driven": args.arrival_trace is not None,
            "trace_requests": len(trace_rows) if trace_rows else None,
            "trace_workloads": ({
                name: sum(1 for row in trace_rows if row.get("class", row.get("workload", "background")) == name)
                for name in sorted({str(row.get("class", row.get("workload", "background"))) for row in trace_rows})
            } if trace_rows else None),
            "background_requests": args.background_requests,
            "urgent_requests": args.urgent_requests,
            "background_arrival_s": args.background_arrival_s,
            "urgent_arrival_s": args.urgent_arrival_s,
            "background_frames": args.background_frames,
            "urgent_frames": args.urgent_frames,
            "background_priority": args.background_priority,
            "urgent_priority": args.urgent_priority,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
