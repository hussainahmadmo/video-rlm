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
        "config_name": f"codec_refined_{prepared['frame_count']}", "method": "codec_guided_query_refined_frames",
        "retrieval_mode": "codec_siglip_coarse_to_fine", "uniform_frame_count": prepared["frame_count"],
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dispatch-plan", type=Path)
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--vlm-concurrency", type=int, default=1)
    parser.add_argument("--prepared-queue-depth", type=int, default=8)
    parser.add_argument("--short-max-s", type=float, default=300.0)
    parser.add_argument("--medium-max-s", type=float, default=1200.0)
    parser.add_argument("--short-frames", type=int, default=8)
    parser.add_argument("--long-frames", type=int, default=8)
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
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    os.environ["CLIP_DEVICE"] = args.clip_device
    native = import_path("dynamic_native", NATIVE_PATH)
    codec = import_path("dynamic_codec", CODEC_PATH)
    prep = import_path("dynamic_prep", PREP_PATH)
    send = import_path("dynamic_send", SEND_PATH)
    retrieval_runner = prep.import_runner()
    retrieval_lock = threading.Lock()
    # The retrieval CLIP model and codec SigLIP may share the same GPU as
    # vLLM. Serialize their bounded scoring/preparation phases so increasing
    # CPU workers never creates overlapping vision-model peaks and an OOM.
    vision_prepare_lock = threading.Lock()
    native_args = SimpleNamespace(
        video_mappings=[(args.video_root.resolve(), args.video_base_url)], max_pixels=args.max_pixels,
        model=args.model, max_tokens=args.max_tokens, request_timeout_s=args.request_timeout_s,
    )
    codec_args = make_codec_args(args)
    schedule = {qid(row): row for row in load_jsonl(args.medium_schedule)}
    plan = {str(row["qid"]): row for row in load_jsonl(args.dispatch_plan)} if args.dispatch_plan else {}
    rows = load_jsonl(args.dataset)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    pending = []
    for row in rows:
        value = duration(row, native)
        route = route_for(value, args.short_max_s, args.medium_max_s)
        decision = plan.get(qid(row))
        if decision and decision.get("route") != route:
            raise SystemExit(f"plan route mismatch for {qid(row)}")
        pending.append({"row": row, "route": route, "duration_s": value, "cost": cost(route, value), "rank": int((decision or {}).get("dispatch_rank", 10**9)), "decision": decision})
    pending.sort(key=lambda item: item["rank"])
    ports = [part.strip() for part in args.ports.split(",") if part.strip()]
    urls = [f"http://127.0.0.1:{port}/v1" for port in ports]
    for url in urls:
        native.check_server(url)
    clients = {url: OpenAI(base_url=url, api_key="EMPTY", timeout=args.request_timeout_s, max_retries=0) for url in urls}
    args.output.mkdir(parents=True)
    results_path = args.output / "results.jsonl"
    decisions_path = args.output / "dispatch_events.jsonl"
    write_lock = threading.Lock()
    ready: deque[dict[str, Any]] = deque()
    prep_futures: dict[Future, dict[str, Any]] = {}
    vlm_futures: dict[Future, dict[str, Any]] = {}
    inflight_by_url = Counter()
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()

    def select_pending() -> dict[str, Any]:
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

    def prepare_medium(item):
        row = item["row"]
        schedule_row = schedule.get(qid(row))
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
        with vision_prepare_lock:
            if item["route"] == "medium_retrieval":
                return prepare_medium(item)
            return prepare_codec(codec, item["row"], args.long_frames, codec_args)

    def choose_url() -> str:
        return min(urls, key=lambda url: (inflight_by_url[url], url))

    def submit_vlm(item, payload):
        url = choose_url()
        inflight_by_url[url] += 1
        if payload["kind"] == "native":
            future = vlm_pool.submit(native.run_one, item["row"], args.short_frames, clients[url], url, native_args)
        elif payload["kind"] == "medium":
            future = vlm_pool.submit(send.run_one, payload["job"], url, args.max_tokens)
        else:
            future = vlm_pool.submit(finish_codec, codec, payload, clients[url], url, codec_args)
        vlm_futures[future] = {"item": item, "url": url, "kind": payload["kind"]}

    def record_result(item, result):
        result["hybrid_route"] = item["route"]
        result["dispatch_rank"] = item["rank"]
        result["dispatch_decision_rule"] = item.get("live_rule")
        append_jsonl(results_path, result, write_lock)
        completed.append(result)
        print(f"[done] qid={result['qid']} route={item['route']} correct={result.get('correct')} error={result.get('error') is not None}", flush=True)

    print(f"questions={len(rows)} cpu_workers={args.cpu_workers} vlm_concurrency={args.vlm_concurrency} prepared_queue_depth={args.prepared_queue_depth} ports={','.join(ports)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.cpu_workers) as prep_pool, ThreadPoolExecutor(max_workers=args.vlm_concurrency) as vlm_pool:
        while pending or prep_futures or ready or vlm_futures:
            while pending and len(prep_futures) < args.cpu_workers and len(ready) < args.prepared_queue_depth:
                item = select_pending()
                event = {"qid": qid(item["row"]), "route": item["route"], "dispatch_rank": item["rank"], "decision_rule": item["live_rule"], "pending_after": len(pending), "decode_inflight": len(prep_futures), "prepared_queue": len(ready), "vlm_inflight": len(vlm_futures)}
                append_jsonl(decisions_path, event, write_lock)
                print("[dispatch] " + json.dumps(event), flush=True)
                if item["route"] == "short_native":
                    ready.append((item, {"kind": "native"}))
                else:
                    prep_futures[prep_pool.submit(prepare_item, item)] = item
            while ready and len(vlm_futures) < args.vlm_concurrency:
                item, payload = ready.popleft()
                if payload["kind"] == "error":
                    record_result(item, {"qid": qid(item["row"]), "dataset": item["row"].get("dataset"), "correct": False, "error": payload["error"], "latency_s": 0.0})
                else:
                    submit_vlm(item, payload)
            all_futures = list(prep_futures) + list(vlm_futures)
            if not all_futures:
                continue
            done, _ = wait(all_futures, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                if future in prep_futures:
                    item = prep_futures.pop(future)
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
    wall_s = time.perf_counter() - started
    summary = {"method": "dynamic_queue_aware_duration_hybrid", "dataset": str(args.dataset.resolve()), "questions": len(completed), "correct": sum(bool(row.get("correct")) for row in completed), "errors": sum(row.get("error") is not None for row in completed), "accuracy_percent": 100 * sum(bool(row.get("correct")) for row in completed) / len(completed) if completed else 0.0, "wall_time_s": wall_s, "throughput_qps": len(completed) / wall_s if wall_s else 0.0, "routes": dict(Counter(row.get("hybrid_route") for row in completed)), "cpu_workers": args.cpu_workers, "vlm_concurrency": args.vlm_concurrency, "prepared_queue_depth": args.prepared_queue_depth, "ports": ports}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
