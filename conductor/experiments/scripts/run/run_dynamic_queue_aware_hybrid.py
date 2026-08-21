#!/usr/bin/env python3
"""Run native, retrieval, and codec video QA jobs through one shared queue.

Unlike ``run_duration_aware_hybrid.py``, this is one event loop. CPU-heavy
preparation (retrieval and codec frame selection) is bounded separately from
VLM requests. A request is selected again whenever a CPU or VLM slot becomes
available, so decisions use the *current* prepared/decode/VLM queue sizes.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import random
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[4]
NATIVE_PATH = ROOT / "conductor/experiments/scripts/run/run_native_vllm_video_baseline.py"
CODEC_PATH = ROOT / "conductor/experiments/scripts/run/run_codec_guided_vllm_baseline.py"
PREP_PATH = ROOT / "conductor/experiments/scripts/build/build_prepared_vlm_jobs.py"
SEND_PATH = ROOT / "conductor/experiments/scripts/run/run_prepared_vlm_jobs.py"


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


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def qid(row: dict[str, Any]) -> str:
    value = row.get("qid") or row.get("question_id") or row.get("id")
    if value is None:
        raise ValueError("row has no qid")
    return str(value)


def duration(row: dict[str, Any], native) -> float:
    for key in ("_measured_duration_s", "duration_s"):
        try:
            value = float(row[key])
            if value > 0:
                return value
        except (KeyError, TypeError, ValueError):
            pass
    return float(native.video_duration_s(row))


def route_for(value: float, short_max: float, medium_max: float) -> str:
    if value <= short_max:
        return "short_native"
    if value <= medium_max:
        return "medium_retrieval"
    return "long_codec_refined"


def cost(route: str, value: float) -> float:
    if route == "short_native":
        return 0.25
    if route == "medium_retrieval":
        return 1.0 + min(value / 1200.0, 1.0)
    return 2.0 + min(value / 3600.0, 2.0)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile without adding a NumPy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def make_codec_args(args):
    return SimpleNamespace(
        model=args.model, max_tokens=args.max_tokens, max_pixels=args.max_pixels,
        window_s=args.codec_window_s, decode_max_side=args.codec_decode_max_side,
        index_timeout_s=args.codec_index_timeout_s,
        decode_timeout_s=args.codec_decode_timeout_s,
        request_timeout_s=args.request_timeout_s, refine_with_clip=True,
        clip_device=args.clip_device, refine_candidates=args.codec_refine_candidates,
        refine_regions=args.codec_refine_regions,
        refine_radius_s=args.codec_refine_radius_s,
        probe_max_side=args.codec_probe_max_side,
    )


def prepare_codec(codec, row, frame_count: int, codec_args) -> dict[str, Any]:
    """CPU/GPU preparation only; the VLM call happens in the shared VLM queue."""
    started = time.perf_counter()
    video = Path(row["video"])
    duration_s = codec.probe_duration(video, codec_args.index_timeout_s)
    activity, _ = codec.scan_packet_activity(video, codec_args.window_s, codec_args.index_timeout_s)
    probes = codec.select_timestamps(activity, duration_s, codec_args.window_s, codec_args.refine_candidates)
    t0 = time.perf_counter()
    probe_jpegs = [codec.decode_jpeg(video, ts, codec_args.probe_max_side, codec_args.decode_timeout_s) for ts in probes]
    probe_decode_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = codec.score_probes(probe_jpegs, row, codec_args.clip_device)
    clip_score_s = time.perf_counter() - t0
    centers = codec.choose_refinement_centers(probes, scores, codec_args.refine_regions, codec_args.refine_radius_s)
    timestamps = codec.dense_local_timestamps(centers, duration_s, frame_count, codec_args.refine_radius_s)
    t0 = time.perf_counter()
    content = []
    for ts in timestamps:
        jpeg = codec.decode_jpeg(video, ts, codec_args.decode_max_side, codec_args.decode_timeout_s)
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    final_decode_s = time.perf_counter() - t0
    content.append({"type": "text", "text": codec.make_prompt(row)})
    return {
        "kind": "codec", "row": row, "content": content,
        "frame_count": frame_count,
        "duration_s": duration_s, "timestamps": timestamps, "probes": probes,
        "scores": scores, "centers": centers, "packet_windows": len(activity),
        "probe_decode_s": probe_decode_s, "clip_score_s": clip_score_s,
        "final_decode_s": final_decode_s, "prepare_latency_s": time.perf_counter() - started,
    }


def prepare_uniform_external(codec, row, frame_count: int, codec_args) -> dict[str, Any]:
    """Decode a short video outside vLLM into uniformly spaced JPEG frames."""
    started = time.perf_counter()
    video = Path(row["video"])
    duration_s = codec.probe_duration(video, codec_args.index_timeout_s)
    timestamps = codec.temporal_anchors(duration_s, frame_count)
    t0 = time.perf_counter()
    content = []
    for ts in timestamps:
        jpeg = codec.decode_jpeg(
            video, ts, codec_args.decode_max_side, codec_args.decode_timeout_s
        )
        content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            },
        })
    final_decode_s = time.perf_counter() - t0
    content.append({"type": "text", "text": codec.make_prompt(row)})
    return {
        "kind": "external_uniform", "row": row, "content": content,
        "frame_count": frame_count, "duration_s": duration_s,
        "timestamps": timestamps, "probes": [], "scores": [], "centers": [],
        "packet_windows": 0, "probe_decode_s": 0.0, "clip_score_s": 0.0,
        "final_decode_s": final_decode_s,
        "prepare_latency_s": time.perf_counter() - started,
        "config_name": f"prepared_uniform_{frame_count}",
        "method": "external_uniform_frames",
        "retrieval_mode": "none",
    }


def finish_codec(codec, prepared: dict[str, Any], client: OpenAI, base_url: str, codec_args) -> dict[str, Any]:
    row = prepared["row"]
    started = time.perf_counter()
    text = None
    error = None
    try:
        response = client.chat.completions.create(
            model=codec_args.model, messages=[{"role": "user", "content": prepared["content"]}],
            temperature=0.0, max_tokens=codec_args.max_tokens,
            extra_body={"mm_processor_kwargs": {"max_pixels": codec_args.max_pixels}},
        )
        text = response.choices[0].message.content
    except Exception as exc:
        error = repr(exc)
    choices = row.get("choices") or []
    prediction = codec.parse_label(text, len(choices))
    gold = row.get("answer_label")
    if gold is None and row.get("answer_idx") is not None:
        gold = chr(ord("A") + int(row["answer_idx"]))
    return {
        "qid": qid(row), "video_id": row.get("video_id"), "video": row.get("video"),
        "dataset": row.get("dataset"), "question": row.get("question"), "choices": choices,
        "answer_idx": row.get("answer_idx"), "answer_label": gold, "answer": row.get("answer"),
        "config_name": prepared.get("config_name", f"codec_refined_{prepared['frame_count']}"),
        "method": prepared.get("method", "codec_guided_query_refined_frames"),
        "retrieval_mode": prepared.get("retrieval_mode", "codec_siglip_coarse_to_fine"),
        "uniform_frame_count": prepared["frame_count"],
        "max_pixels": codec_args.max_pixels, "duration_s": prepared["duration_s"],
        "selected_timestamps_s": prepared["timestamps"], "probe_timestamps_s": prepared["probes"],
        "probe_scores": prepared["scores"], "refinement_centers_s": prepared["centers"],
        "codec_packet_windows": prepared["packet_windows"], "probe_decode_s": prepared["probe_decode_s"],
        "clip_score_s": prepared["clip_score_s"], "final_decode_s": prepared["final_decode_s"],
        "prepare_latency_s": prepared["prepare_latency_s"], "prediction_label": prediction,
        "prediction_text": text, "correct": prediction == gold,
        "latency_s": prepared["prepare_latency_s"] + (time.perf_counter() - started),
        "num_vlm_calls": 0 if error else 1, "base_url": base_url, "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--medium-schedule", type=Path, required=True)
    parser.add_argument(
        "--medium-schedule-budget2", type=Path,
        help="Fixed budget2 retrieval schedule; required for --config-policy load_adaptive.",
    )
    parser.add_argument(
        "--medium-schedule-budget32", type=Path,
        help="Fixed budget32 retrieval schedule; required for --config-policy load_adaptive.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dispatch-plan", type=Path)
    parser.add_argument(
        "--dispatch-policy",
        choices=["queue_aware", "fifo", "shortest_estimated"],
        default="queue_aware",
        help=(
            "queue_aware prioritizes costly preparation when the prepared "
            "queue is empty; fifo is the same implementation/configuration "
            "but preserves dataset arrival order as a scheduling control; "
            "shortest_estimated prefers the smallest duration-derived "
            "preparation-cost estimate."
        ),
    )
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--vlm-concurrency", type=int, default=1)
    parser.add_argument("--prepared-queue-depth", type=int, default=8)
    parser.add_argument(
        "--max-pending", type=int, default=0,
        help=(
            "Maximum requests visible to the preparation dispatcher. Zero preserves "
            "the legacy unbounded pending list; a positive value applies backpressure "
            "before preparation admission."
        ),
    )
    parser.add_argument(
        "--resource-policy", choices=["fixed", "stage_adaptive"], default="fixed",
        help=(
            "fixed uses configured worker limits throughout the run; "
            "stage_adaptive regulates preparation admission from observed stage "
            "service times and queue occupancy while preserving a safe VLM floor."
        ),
    )
    parser.add_argument(
        "--min-vlm-concurrency", type=int,
        help=(
            "Minimum effective VLM admission limit for stage_adaptive. Defaults "
            "to --vlm-concurrency, so adaptation never throttles configured GPU serving."
        ),
    )
    parser.add_argument(
        "--controller-interval-s", type=float, default=1.0,
        help="Minimum interval between stage-adaptive control decisions.",
    )
    parser.add_argument(
        "--target-ready-queue", type=int,
        help=(
            "Target number of prepared jobs. Defaults to half of "
            "--prepared-queue-depth when stage_adaptive is enabled."
        ),
    )
    parser.add_argument(
        "--warmup-prepared", type=int, default=0,
        help=(
            "Do not submit to vLLM until this many prepared jobs are queued. "
            "Use a positive value only for saturation experiments; zero starts serving immediately."
        ),
    )
    parser.add_argument(
        "--arrival-rate-qps", type=float,
        help=(
            "Open-loop offered request rate. Omit for a burst arrival at t=0. "
            "Use this, rather than client concurrency alone, for delay-under-load sweeps."
        ),
    )
    parser.add_argument(
        "--arrival-order", choices=["dataset", "seeded_random", "long_short"], default="dataset",
        help=(
            "dataset preserves the legacy order; seeded_random uses --arrival-seed; "
            "long_short alternates highest and lowest estimated preparation cost. "
            "Use --dispatch-policy fifo to preserve the selected arrival trace."
        ),
    )
    parser.add_argument(
        "--arrival-seed", type=int, default=0,
        help="Seed used by --arrival-order seeded_random.",
    )
    parser.add_argument("--short-max-s", type=float, default=300.0)
    parser.add_argument("--medium-max-s", type=float, default=1200.0)
    parser.add_argument("--short-frames", type=int, default=8)
    parser.add_argument(
        "--short-input-mode", choices=["native_video", "prepared_frames"],
        default="native_video",
        help="native_video is the vLLM baseline; prepared_frames externally decodes JPEGs before the VLM queue.",
    )
    parser.add_argument("--long-frames", type=int, default=8)
    parser.add_argument(
        "--long-input-mode", choices=["codec_refined", "prepared_uniform"],
        default="codec_refined",
        help=(
            "codec_refined uses codec-guided temporal refinement for long videos; "
            "prepared_uniform uses the same-count uniformly sampled external frames."
        ),
    )
    parser.add_argument(
        "--config-policy", choices=["fixed", "load_adaptive"], default="fixed",
        help="fixed reproduces supplied budgets; load_adaptive chooses rich, baseline, or cheap actions from live queue pressure.",
    )
    parser.add_argument("--adaptive-rich-short-frames", type=int, default=8)
    parser.add_argument("--adaptive-cheap-short-frames", type=int, default=2)
    parser.add_argument("--adaptive-rich-long-frames", type=int, default=8)
    parser.add_argument("--adaptive-cheap-long-frames", type=int, default=2)
    parser.add_argument("--adaptive-high-load", type=int, default=2)
    parser.add_argument("--video-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--video-base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--clip-image-batch-size", type=int, default=16)
    parser.add_argument("--codec-window-s", type=float, default=8.0)
    parser.add_argument("--codec-refine-candidates", type=int, default=8)
    parser.add_argument("--codec-refine-regions", type=int, default=2)
    parser.add_argument("--codec-refine-radius-s", type=float, default=16.0)
    parser.add_argument("--codec-probe-max-side", type=int, default=224)
    parser.add_argument("--codec-decode-max-side", type=int, default=448)
    parser.add_argument("--codec-index-timeout-s", type=float, default=120.0)
    parser.add_argument("--codec-decode-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    if min(args.cpu_workers, args.vlm_concurrency, args.prepared_queue_depth) < 1:
        parser.error("worker and queue sizes must be positive")
    if args.min_vlm_concurrency is not None and not 1 <= args.min_vlm_concurrency <= args.vlm_concurrency:
        parser.error("--min-vlm-concurrency must be between one and --vlm-concurrency")
    if args.max_pending < 0:
        parser.error("--max-pending must be non-negative")
    if args.controller_interval_s <= 0:
        parser.error("--controller-interval-s must be positive")
    if args.target_ready_queue is not None and not 0 <= args.target_ready_queue <= args.prepared_queue_depth:
        parser.error("--target-ready-queue must be between zero and --prepared-queue-depth")
    if not 0 <= args.warmup_prepared <= args.prepared_queue_depth:
        parser.error("--warmup-prepared must be between zero and --prepared-queue-depth")
    if args.adaptive_high_load < 1:
        parser.error("--adaptive-high-load must be positive")
    if args.arrival_rate_qps is not None and args.arrival_rate_qps <= 0:
        parser.error("--arrival-rate-qps must be positive")
    if args.config_policy == "load_adaptive" and (
        args.medium_schedule_budget2 is None or args.medium_schedule_budget32 is None
    ):
        parser.error("load_adaptive requires --medium-schedule-budget2 and --medium-schedule-budget32")
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    os.environ["CLIP_DEVICE"] = args.clip_device
    native = import_path("dynamic_native", NATIVE_PATH)
    codec = import_path("dynamic_codec", CODEC_PATH)
    prep = import_path("dynamic_prep", PREP_PATH)
    send = import_path("dynamic_send", SEND_PATH)
    retrieval_runner = prep.import_runner()
    retrieval_lock = threading.Lock()
    # Retrieval retains its own lock because its current helper combines decode
    # and CLIP scoring. Codec decode is intentionally *not* protected here:
    # ``score_probes`` has its own small model lock, so several CPU workers can
    # extract codec frames concurrently while only SigLIP scoring serializes.
    native_args = SimpleNamespace(
        video_mappings=[(args.video_root.resolve(), args.video_base_url)], max_pixels=args.max_pixels,
        model=args.model, max_tokens=args.max_tokens, request_timeout_s=args.request_timeout_s,
    )
    codec_args = make_codec_args(args)
    schedules = {"budget8": {qid(row): row for row in load_jsonl(args.medium_schedule)}}
    if args.config_policy == "load_adaptive":
        schedules["budget2"] = {qid(row): row for row in load_jsonl(args.medium_schedule_budget2)}
        schedules["budget32"] = {qid(row): row for row in load_jsonl(args.medium_schedule_budget32)}
    plan = {str(row["qid"]): row for row in load_jsonl(args.dispatch_plan)} if args.dispatch_plan else {}
    rows = load_jsonl(args.dataset)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    items = []
    for row in rows:
        value = duration(row, native)
        route = route_for(value, args.short_max_s, args.medium_max_s)
        decision = plan.get(qid(row))
        if decision and decision.get("route") != route:
            raise SystemExit(f"plan route mismatch for {qid(row)}")
        items.append({"row": row, "route": route, "duration_s": value, "cost": cost(route, value), "rank": int((decision or {}).get("dispatch_rank", 10**9)), "decision": decision})
    if args.arrival_order == "seeded_random":
        random.Random(args.arrival_seed).shuffle(items)
    elif args.arrival_order == "long_short":
        ordered = sorted(items, key=lambda item: (item["duration_s"], qid(item["row"])))
        low, high = deque(ordered), deque(reversed(ordered))
        trace_items = []
        while low:
            trace_items.append(high.popleft())
            if low:
                low.pop()
            if low:
                trace_items.append(low.popleft())
                if high:
                    high.pop()
        items = trace_items
    elif args.dispatch_policy == "queue_aware":
        # Preserve the legacy queue-aware order for the default dataset trace.
        items.sort(key=lambda item: item["rank"])
    ports = [part.strip() for part in args.ports.split(",") if part.strip()]
    urls = [f"http://127.0.0.1:{port}/v1" for port in ports]
    for url in urls:
        native.check_server(url)
    clients = {url: OpenAI(base_url=url, api_key="EMPTY", timeout=args.request_timeout_s, max_retries=0) for url in urls}
    args.output.mkdir(parents=True)
    results_path = args.output / "results.jsonl"
    decisions_path = args.output / "dispatch_events.jsonl"
    resource_events_path = args.output / "resource_events.jsonl"
    arrival_trace_path = args.output / "arrival_trace.jsonl"
    write_lock = threading.Lock()
    for index, item in enumerate(items):
        append_jsonl(arrival_trace_path, {
            "arrival_index": index,
            "qid": qid(item["row"]),
            "route": item["route"],
            "duration_s": item["duration_s"],
            "estimated_prepare_cost": item["cost"],
        }, write_lock)
    pending: list[dict[str, Any]] = []
    arrivals: deque[dict[str, Any]] = deque(items)
    ready: deque[dict[str, Any]] = deque()
    prep_futures: dict[Future, dict[str, Any]] = {}
    vlm_futures: dict[Future, dict[str, Any]] = {}
    inflight_by_url = Counter()
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    vlm_admission_open = args.warmup_prepared == 0
    target_ready_queue = args.target_ready_queue if args.target_ready_queue is not None else max(1, args.prepared_queue_depth // 2)
    min_vlm_concurrency = (
        args.min_vlm_concurrency
        if args.min_vlm_concurrency is not None
        else args.vlm_concurrency
    )
    effective_prep_limit = args.cpu_workers
    effective_vlm_limit = args.vlm_concurrency
    prep_ewma_s: float | None = None
    vlm_ewma_s: float | None = None
    last_controller_s = -float("inf")

    def elapsed() -> float:
        return time.perf_counter() - started

    def observe_stage(stage: str, value: float) -> None:
        """Maintain an EWMA service-time profile for preparation and VLM stages."""
        nonlocal prep_ewma_s, vlm_ewma_s
        if value <= 0:
            return
        alpha = 0.2
        if stage == "prep":
            prep_ewma_s = value if prep_ewma_s is None else alpha * value + (1 - alpha) * prep_ewma_s
        else:
            vlm_ewma_s = value if vlm_ewma_s is None else alpha * value + (1 - alpha) * vlm_ewma_s

    def update_resource_limits() -> None:
        """Adapt admission limits, not physical thread-pool sizes, from live state."""
        nonlocal effective_prep_limit, effective_vlm_limit, last_controller_s
        now = elapsed()
        if args.resource_policy != "stage_adaptive" or now - last_controller_s < args.controller_interval_s:
            return
        last_controller_s = now
        old_limits = (effective_prep_limit, effective_vlm_limit)
        # Keep all configured VLM capacity available by default. The first
        # controller version reduced VLM admission to match the slower prep
        # stage, which created an avoidable ready-queue bottleneck. Adaptation
        # instead controls how aggressively preparation is admitted.
        effective_vlm_limit = args.vlm_concurrency
        if prep_ewma_s is None or vlm_ewma_s is None:
            effective_prep_limit = args.cpu_workers
        else:
            ratio = max(0.1, prep_ewma_s / vlm_ewma_s)
            if len(ready) >= target_ready_queue:
                # Prepared work is accumulating, so reserve CPU capacity by
                # reducing new preparation admission, never inference service.
                effective_prep_limit = max(1, args.cpu_workers - 1)
            elif not ready and (pending or arrivals):
                # The preparation stage is starving inference; use the full
                # preparation pool to replenish the ready queue.
                effective_prep_limit = args.cpu_workers
            else:
                # Keep a small amount of headroom when stages are balanced.
                effective_prep_limit = max(1, min(args.cpu_workers, round(ratio)))
            effective_vlm_limit = max(min_vlm_concurrency, effective_vlm_limit)
        if old_limits != (effective_prep_limit, effective_vlm_limit):
            event = {
                "time_s": now,
                "policy": args.resource_policy,
                "prep_limit": effective_prep_limit,
                "vlm_limit": effective_vlm_limit,
                "prep_ewma_s": prep_ewma_s,
                "vlm_ewma_s": vlm_ewma_s,
                "ready_queue": len(ready),
                "pending_queue": len(pending),
            }
            append_jsonl(resource_events_path, event, write_lock)
            print("[resource] " + json.dumps(event), flush=True)

    def release_arrivals() -> None:
        """Move due open-loop requests into the dispatcher-visible pending queue."""
        now = elapsed()
        while arrivals and (args.max_pending == 0 or len(pending) < args.max_pending):
            index = int(arrivals[0]["arrival_index"])
            scheduled = 0.0 if args.arrival_rate_qps is None else index / args.arrival_rate_qps
            if now < scheduled:
                break
            item = arrivals.popleft()
            item["arrival_s"] = scheduled
            pending.append(item)

    for index, item in enumerate(arrivals):
        item["arrival_index"] = index

    def select_pending() -> dict[str, Any]:
        if args.dispatch_policy == "fifo":
            item = pending.pop(0)
            item["live_rule"] = "fifo_arrival_order"
            return item
        if args.dispatch_policy == "shortest_estimated":
            item = min(
                pending,
                key=lambda value: (value["cost"], value["duration_s"], value["rank"]),
            )
            pending.remove(item)
            item["live_rule"] = "shortest_estimated_prepare_cost"
            return item
        # Empty ready queue: hide a costly preparation behind future VLM work.
        candidates = [item for item in pending if item["route"] != "short_native"]
        if not ready and not prep_futures and candidates:
            item = max(candidates, key=lambda value: (value["cost"], -value["rank"]))
            rule = "prepared_empty_start_expensive_job"
        elif len(ready) >= args.prepared_queue_depth:
            item = min(pending, key=lambda value: (value["cost"], value["rank"]))
            rule = "prepared_full_prefer_cheap_job"
        else:
            item = min(pending, key=lambda value: value["rank"])
            rule = "dispatch_plan_priority"
        pending.remove(item)
        item["live_rule"] = rule
        return item

    def select_config(item: dict[str, Any]) -> None:
        """Select a budget from the live shared queue state at dispatch time."""
        queue_load = len(prep_futures) + len(ready) + len(vlm_futures)
        item["queue_load_at_dispatch"] = queue_load
        if args.config_policy == "fixed":
            if item["route"] == "short_native":
                prefix = "native_uniform" if args.short_input_mode == "native_video" else "prepared_uniform"
                item["selected_config"] = f"{prefix}_{args.short_frames}"
                item["frame_count"] = args.short_frames
            elif item["route"] == "medium_retrieval":
                item["selected_config"] = "budget8"
                item["medium_schedule_key"] = "budget8"
            else:
                prefix = "codec_refined" if args.long_input_mode == "codec_refined" else "prepared_uniform"
                item["selected_config"] = f"{prefix}_{args.long_frames}"
                item["frame_count"] = args.long_frames
            item["config_decision_rule"] = "fixed_config"
            return

        tier = "cheap" if queue_load >= args.adaptive_high_load else "rich" if queue_load == 0 else "baseline"
        item["config_decision_rule"] = f"load_adaptive_{tier}"
        if item["route"] == "short_native":
            frames = args.adaptive_cheap_short_frames if tier == "cheap" else args.adaptive_rich_short_frames
            prefix = "native_uniform" if args.short_input_mode == "native_video" else "prepared_uniform"
            item["selected_config"] = f"{prefix}_{frames}"
            item["frame_count"] = frames
        elif item["route"] == "medium_retrieval":
            key = {"cheap": "budget2", "baseline": "budget8", "rich": "budget32"}[tier]
            item["selected_config"] = key
            item["medium_schedule_key"] = key
        else:
            frames = args.adaptive_cheap_long_frames if tier == "cheap" else args.adaptive_rich_long_frames
            prefix = "codec_refined" if args.long_input_mode == "codec_refined" else "prepared_uniform"
            item["selected_config"] = f"{prefix}_{frames}"
            item["frame_count"] = frames

    def prepare_medium(item):
        row = item["row"]
        schedule_row = schedules[item["medium_schedule_key"]].get(qid(row))
        if schedule_row is None:
            raise RuntimeError(f"missing medium schedule qid={qid(row)}")
        with retrieval_lock:
            config = prep.build_config_from_schedule(schedule_row)
            config = prep.apply_existing_policy_rewrites(retrieval_runner, row, config)
            if config.get("method") != "clip_oneshot":
                raise NotImplementedError(f"dynamic runner supports clip_oneshot only: {config.get('method')}")
            job = prep.prepare_clip_oneshot_job(retrieval_runner, row, config)
        return {"kind": "medium", "job": job}

    def prepare_item(item):
        if item["route"] == "short_native":
            return prepare_uniform_external(codec, item["row"], item["frame_count"], codec_args)
        if item["route"] == "medium_retrieval":
            return prepare_medium(item)
        if args.long_input_mode == "prepared_uniform":
            return prepare_uniform_external(codec, item["row"], item["frame_count"], codec_args)
        return prepare_codec(codec, item["row"], item["frame_count"], codec_args)

    def choose_url() -> str:
        return min(urls, key=lambda url: (inflight_by_url[url], url))

    def submit_vlm(item, payload):
        url = choose_url()
        item["vlm_submit_s"] = elapsed()
        inflight_by_url[url] += 1
        if payload["kind"] == "native":
            future = vlm_pool.submit(native.run_one, item["row"], item["frame_count"], clients[url], url, native_args)
        elif payload["kind"] == "medium":
            future = vlm_pool.submit(send.run_one, payload["job"], url, args.max_tokens)
        else:
            future = vlm_pool.submit(finish_codec, codec, payload, clients[url], url, codec_args)
        vlm_futures[future] = {"item": item, "url": url, "kind": payload["kind"]}

    def record_result(item, result):
        completion_s = elapsed()
        arrival_s = float(item.get("arrival_s", 0.0))
        dispatch_s = float(item.get("dispatch_s", completion_s))
        prep_started_s = float(item.get("prep_started_s", dispatch_s))
        prep_ready_s = float(item.get("prep_ready_s", completion_s))
        vlm_submit_s = item.get("vlm_submit_s")
        result["arrival_s"] = arrival_s
        result["dispatch_s"] = dispatch_s
        result["prep_started_s"] = prep_started_s
        result["prep_ready_s"] = prep_ready_s
        result["vlm_submit_s"] = vlm_submit_s
        result["completion_s"] = completion_s
        result["queue_wait_before_prepare_s"] = max(0.0, dispatch_s - arrival_s)
        result["preparation_wall_s"] = max(0.0, prep_ready_s - prep_started_s)
        result["prepared_queue_wait_s"] = max(
            0.0, (float(vlm_submit_s) if vlm_submit_s is not None else completion_s) - prep_ready_s
        )
        result["vlm_service_wall_s"] = (
            max(0.0, completion_s - float(vlm_submit_s)) if vlm_submit_s is not None else None
        )
        if result["vlm_service_wall_s"] is not None:
            observe_stage("vlm", result["vlm_service_wall_s"])
        result["end_to_end_delay_s"] = max(0.0, completion_s - arrival_s)
        result["hybrid_route"] = item["route"]
        result["dispatch_rank"] = item["rank"]
        result["dispatch_decision_rule"] = item.get("live_rule")
        result["selected_config"] = item.get("selected_config")
        result["config_decision_rule"] = item.get("config_decision_rule")
        result["queue_load_at_dispatch"] = item.get("queue_load_at_dispatch")
        result["estimated_video_duration_s"] = item["duration_s"]
        result["estimated_prepare_cost"] = item["cost"]
        append_jsonl(results_path, result, write_lock)
        completed.append(result)
        print(f"[done] qid={result['qid']} route={item['route']} correct={result.get('correct')} error={result.get('error') is not None}", flush=True)

    arrival_description = "burst" if args.arrival_rate_qps is None else f"{args.arrival_rate_qps:g}qps"
    print(f"questions={len(rows)} arrival_rate={arrival_description} cpu_workers={args.cpu_workers} vlm_concurrency={args.vlm_concurrency} prepared_queue_depth={args.prepared_queue_depth} warmup_prepared={args.warmup_prepared} ports={','.join(ports)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.cpu_workers) as prep_pool, ThreadPoolExecutor(max_workers=args.vlm_concurrency) as vlm_pool:
        while arrivals or pending or prep_futures or ready or vlm_futures:
            release_arrivals()
            update_resource_limits()
            while pending and len(prep_futures) < effective_prep_limit and len(ready) < args.prepared_queue_depth:
                item = select_pending()
                item["dispatch_s"] = elapsed()
                select_config(item)
                event = {"qid": qid(item["row"]), "route": item["route"], "dispatch_rank": item["rank"], "decision_rule": item["live_rule"], "selected_config": item["selected_config"], "config_decision_rule": item["config_decision_rule"], "queue_load_at_dispatch": item["queue_load_at_dispatch"], "pending_after": len(pending), "decode_inflight": len(prep_futures), "prepared_queue": len(ready), "vlm_inflight": len(vlm_futures)}
                append_jsonl(decisions_path, event, write_lock)
                print("[dispatch] " + json.dumps(event), flush=True)
                if item["route"] == "short_native" and args.short_input_mode == "native_video":
                    item["prep_started_s"] = item["dispatch_s"]
                    item["prep_ready_s"] = item["dispatch_s"]
                    ready.append((item, {"kind": "native"}))
                else:
                    item["prep_started_s"] = elapsed()
                    prep_futures[prep_pool.submit(prepare_item, item)] = item
            if not vlm_admission_open and (
                len(ready) >= args.warmup_prepared
                or (not arrivals and not pending and not prep_futures)
            ):
                vlm_admission_open = True
                print(f"[warmup-complete] prepared_queue={len(ready)}", flush=True)
            while vlm_admission_open and ready and len(vlm_futures) < effective_vlm_limit:
                item, payload = ready.popleft()
                if payload["kind"] == "error":
                    record_result(item, {"qid": qid(item["row"]), "dataset": item["row"].get("dataset"), "correct": False, "error": payload["error"], "latency_s": 0.0})
                else:
                    submit_vlm(item, payload)
            all_futures = list(prep_futures) + list(vlm_futures)
            if not all_futures:
                if arrivals:
                    next_index = int(arrivals[0]["arrival_index"])
                    next_arrival_s = next_index / args.arrival_rate_qps if args.arrival_rate_qps else 0.0
                    time.sleep(max(0.0, min(0.2, next_arrival_s - elapsed())))
                continue
            done, _ = wait(all_futures, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                if future in prep_futures:
                    item = prep_futures.pop(future)
                    item["prep_ready_s"] = elapsed()
                    observe_stage("prep", item["prep_ready_s"] - item["prep_started_s"])
                    try:
                        ready.append((item, future.result()))
                    except Exception as exc:
                        ready.append((item, {"kind": "error", "error": repr(exc)}))
                else:
                    meta = vlm_futures.pop(future)
                    inflight_by_url[meta["url"]] -= 1
                    item = meta["item"]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"qid": qid(item["row"]), "dataset": item["row"].get("dataset"), "correct": False, "error": repr(exc), "latency_s": 0.0}
                    record_result(item, result)
    wall_s = elapsed()
    delays = [float(row["end_to_end_delay_s"]) for row in completed]
    summary = {"method": "dynamic_queue_aware_duration_hybrid", "dispatch_policy": args.dispatch_policy, "config_policy": args.config_policy, "arrival_rate_qps": args.arrival_rate_qps, "dataset": str(args.dataset.resolve()), "questions": len(completed), "correct": sum(bool(row.get("correct")) for row in completed), "errors": sum(row.get("error") is not None for row in completed), "accuracy_percent": 100 * sum(bool(row.get("correct")) for row in completed) / len(completed) if completed else 0.0, "wall_time_s": wall_s, "throughput_qps": len(completed) / wall_s if wall_s else 0.0, "mean_end_to_end_delay_s": sum(delays) / len(delays) if delays else 0.0, "p50_end_to_end_delay_s": percentile(delays, 0.50), "p95_end_to_end_delay_s": percentile(delays, 0.95), "routes": dict(Counter(row.get("hybrid_route") for row in completed)), "selected_configs": dict(Counter(row.get("selected_config") for row in completed)), "config_decision_rules": dict(Counter(row.get("config_decision_rule") for row in completed)), "short_input_mode": args.short_input_mode, "long_input_mode": args.long_input_mode, "cpu_workers": args.cpu_workers, "vlm_concurrency": args.vlm_concurrency, "prepared_queue_depth": args.prepared_queue_depth, "warmup_prepared": args.warmup_prepared, "ports": ports}
    summary["arrival_order"] = args.arrival_order
    summary["arrival_seed"] = args.arrival_seed if args.arrival_order == "seeded_random" else None
    summary["resource_policy"] = args.resource_policy
    summary["controller_interval_s"] = args.controller_interval_s
    summary["target_ready_queue"] = target_ready_queue
    summary["min_vlm_concurrency"] = min_vlm_concurrency
    summary["effective_prep_limit_final"] = effective_prep_limit
    summary["effective_vlm_limit_final"] = effective_vlm_limit
    summary["stage_profile"] = {
        "prep_ewma_s": prep_ewma_s,
        "vlm_ewma_s": vlm_ewma_s,
        "suggested_prepare_workers_per_vlm": (
            prep_ewma_s / vlm_ewma_s if prep_ewma_s is not None and vlm_ewma_s else None
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
