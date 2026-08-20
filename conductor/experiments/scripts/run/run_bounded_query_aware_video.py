#!/usr/bin/env python3
"""Bounded, query-aware video serving with disaggregated preparation stages.

This is an experimental serving runner.  Each raw video passes through bounded
queues: probe decode -> timed CLIP microbatch -> final-frame decode -> vLLM.
Only final frames are sent to vLLM.  The runner intentionally has no
persistent cache: CLIP batching happens only across requests currently in
flight.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import queue
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
CODEC_PATH = ROOT / "conductor/experiments/scripts/run/run_codec_guided_vllm_baseline.py"


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


def qid(row: dict[str, Any]) -> str:
    value = row.get("qid") or row.get("question_id") or row.get("id")
    if value is None:
        raise ValueError("row has no qid")
    return str(value)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


class ClipMicroBatcher:
    """Own one SigLIP model and score probe images from several requests at once."""

    def __init__(self, codec, device: str, max_images: int, wait_ms: float):
        self.codec = codec
        self.device = device
        self.max_images = max_images
        self.wait_s = wait_ms / 1000.0
        self.inbox: queue.Queue[tuple[dict[str, Any], Future] | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="clip-microbatch", daemon=True)
        self.thread.start()

    def submit(self, probe: dict[str, Any]) -> Future:
        future: Future = Future()
        self.inbox.put((probe, future))
        return future

    def close(self) -> None:
        self.inbox.put(None)
        self.thread.join()

    def _run(self) -> None:
        # Import here so decoder workers never contend while the model loads.
        import torch
        from PIL import Image

        model, processor = self.codec.load_clip_refiner(self.device)
        while True:
            first = self.inbox.get()
            if first is None:
                return
            batch = [first]
            images_in_batch = len(first[0]["probe_jpegs"])
            deadline = time.monotonic() + self.wait_s
            while images_in_batch < self.max_images:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = self.inbox.get(timeout=remaining)
                except queue.Empty:
                    break
                if next_item is None:
                    self.inbox.put(None)
                    break
                probe, future = next_item
                if images_in_batch and images_in_batch + len(probe["probe_jpegs"]) > self.max_images:
                    self.inbox.put(next_item)
                    break
                batch.append(next_item)
                images_in_batch += len(probe["probe_jpegs"])
            try:
                started = time.perf_counter()
                images, queries, image_spans, query_spans = [], [], [], []
                for probe, _ in batch:
                    image_start = len(images)
                    images.extend(Image.open(io.BytesIO(raw)).convert("RGB") for raw in probe["probe_jpegs"])
                    image_spans.append((image_start, len(images)))
                    query_start = len(queries)
                    queries.extend(self.codec.clip_queries(probe["row"]))
                    query_spans.append((query_start, len(queries)))
                with self.codec.CLIP_LOCK, torch.no_grad():
                    image_inputs = processor(images=images, return_tensors="pt")
                    dtype = next(model.parameters()).dtype
                    image_inputs = {
                        key: value.to(device=self.device, dtype=dtype) if value.is_floating_point() else value.to(self.device)
                        for key, value in image_inputs.items()
                    }
                    text_inputs = processor(text=queries, return_tensors="pt", padding=True, truncation=True, max_length=64)
                    text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
                    image_features = self.codec.feature_tensor(model.get_image_features(**image_inputs))
                    text_features = self.codec.feature_tensor(model.get_text_features(**text_inputs))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    matrix = (image_features @ text_features.T).float().cpu()
                elapsed = time.perf_counter() - started
                for (probe, future), (i0, i1), (t0, t1) in zip(batch, image_spans, query_spans):
                    scored = dict(probe)
                    scored["scores"] = matrix[i0:i1, t0:t1].max(dim=1).values.tolist()
                    scored["clip_score_s"] = elapsed
                    scored["clip_batch_requests"] = len(batch)
                    scored["clip_batch_images"] = len(images)
                    future.set_result(scored)
            except Exception as exc:
                for _, future in batch:
                    future.set_exception(exc)


def make_codec_args(args):
    return SimpleNamespace(
        model=args.model, max_tokens=args.max_tokens, max_pixels=args.max_pixels,
        window_s=8.0, decode_max_side=args.decode_max_side,
        index_timeout_s=args.index_timeout_s, decode_timeout_s=args.decode_timeout_s,
        refine_radius_s=args.refine_radius_s, probe_max_side=args.probe_max_side,
    )


def probe_video(codec, row: dict[str, Any], args) -> dict[str, Any]:
    started = time.perf_counter()
    video = Path(row["video"])
    duration_s = codec.probe_duration(video, args.index_timeout_s)
    timestamps = codec.temporal_anchors(duration_s, args.probe_frames)
    decode_started = time.perf_counter()
    jpegs = [codec.decode_jpeg(video, stamp, args.probe_max_side, args.decode_timeout_s) for stamp in timestamps]
    return {
        "row": row, "duration_s": duration_s, "probe_timestamps": timestamps,
        "probe_jpegs": jpegs, "probe_decode_s": time.perf_counter() - decode_started,
        "probe_wall_s": time.perf_counter() - started,
    }


def refine_and_decode(codec, scored: dict[str, Any], args) -> dict[str, Any]:
    started = time.perf_counter()
    centers = codec.choose_refinement_centers(
        scored["probe_timestamps"], scored["scores"], args.refine_regions, args.refine_radius_s
    )
    local_count = max(0, args.final_frames - args.global_anchor_frames)
    local = codec.dense_local_timestamps(centers, scored["duration_s"], local_count, args.refine_radius_s)
    anchors = codec.temporal_anchors(scored["duration_s"], args.global_anchor_frames)
    # Preserve temporal order but guarantee the advertised final-frame budget
    # when an anchor happens to coincide with a local refinement timestamp.
    timestamps: list[float] = []
    for stamp in anchors + local + codec.temporal_anchors(scored["duration_s"], args.final_frames * 3):
        if all(abs(stamp - chosen) > 1e-6 for chosen in timestamps):
            timestamps.append(stamp)
        if len(timestamps) == args.final_frames:
            break
    timestamps.sort()
    video = Path(scored["row"]["video"])
    images = []
    for stamp in timestamps:
        jpeg = codec.decode_jpeg(video, stamp, args.decode_max_side, args.decode_timeout_s)
        images.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    images.append({"type": "text", "text": codec.make_prompt(scored["row"])})
    prepared = dict(scored)
    prepared.update({
        "content": images, "selected_timestamps": timestamps, "centers": centers,
        "final_frames": args.final_frames, "global_anchor_frames": args.global_anchor_frames,
        "final_decode_s": time.perf_counter() - started,
        "prepare_latency_s": scored["probe_wall_s"] + scored.get("clip_score_s", 0.0) + (time.perf_counter() - started),
    })
    return prepared


def run_vlm(codec, prepared: dict[str, Any], client: OpenAI, url: str, codec_args) -> dict[str, Any]:
    row = prepared["row"]
    started = time.perf_counter()
    text, error = None, None
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
        "answer_label": gold, "prediction_label": prediction, "prediction_text": text,
        "correct": prediction == gold, "error": error, "latency_s": time.perf_counter() - started,
        "config_name": f"query_refined_{prepared['final_frames']}",
        "method": "bounded_batched_clip_coarse_to_fine",
        "duration_s": prepared["duration_s"], "probe_timestamps_s": prepared["probe_timestamps"],
        "probe_scores": prepared["scores"], "refinement_centers_s": prepared["centers"],
        "selected_timestamps_s": prepared["selected_timestamps"],
        "uniform_frame_count": prepared["final_frames"], "global_anchor_frames": prepared["global_anchor_frames"],
        "probe_decode_s": prepared["probe_decode_s"], "clip_score_s": prepared.get("clip_score_s", 0.0),
        "clip_batch_requests": prepared.get("clip_batch_requests", 1),
        "clip_batch_images": prepared.get("clip_batch_images", len(prepared["probe_jpegs"])),
        "final_decode_s": prepared["final_decode_s"], "prepare_latency_s": prepared["prepare_latency_s"],
        "base_url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--arrival-rate-qps", type=float, help="Omit for a burst.")
    parser.add_argument("--cpu-workers", type=int, default=4, help="Total probe/final decode workers.")
    parser.add_argument("--vlm-concurrency", type=int, default=4)
    parser.add_argument("--max-pending", type=int, default=32)
    parser.add_argument("--ready-queue-depth", type=int, default=16)
    parser.add_argument("--probe-frames", type=int, default=32)
    parser.add_argument("--final-frames", type=int, default=8)
    parser.add_argument("--global-anchor-frames", type=int, default=2)
    parser.add_argument("--refine-regions", type=int, default=3)
    parser.add_argument("--refine-radius-s", type=float, default=10.0)
    parser.add_argument("--clip-batch-images", type=int, default=64)
    parser.add_argument("--clip-batch-wait-ms", type=float, default=10.0)
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--probe-max-side", type=int, default=224)
    parser.add_argument("--decode-max-side", type=int, default=448)
    parser.add_argument("--index-timeout-s", type=float, default=120.0)
    parser.add_argument("--decode-timeout-s", type=float, default=60.0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    if min(args.cpu_workers, args.vlm_concurrency, args.max_pending, args.ready_queue_depth, args.probe_frames, args.final_frames) < 1:
        parser.error("worker, queue, and frame counts must be positive")
    if not 0 <= args.global_anchor_frames <= args.final_frames:
        parser.error("--global-anchor-frames must be in [0, --final-frames]")
    if args.arrival_rate_qps is not None and args.arrival_rate_qps <= 0:
        parser.error("--arrival-rate-qps must be positive")
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")

    codec = import_path("bounded_codec", CODEC_PATH)
    codec_args = make_codec_args(args)
    urls = [f"http://127.0.0.1:{part.strip()}/v1" for part in args.ports.split(",") if part.strip()]
    for url in urls:
        # The codec runner uses the same OpenAI-compatible readiness endpoint.
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(f"{url}/models", headers={"Authorization": "Bearer EMPTY"}), timeout=10):
            pass
    clients = {url: OpenAI(base_url=url, api_key="EMPTY", timeout=args.request_timeout_s, max_retries=0) for url in urls}
    rows = load_jsonl(args.dataset)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    args.output.mkdir(parents=True)
    result_path = args.output / "results.jsonl"
    trace_path = args.output / "stage_events.jsonl"
    write_lock = threading.Lock()

    def write(path: Path, value: dict[str, Any]) -> None:
        with write_lock, path.open("a") as handle:
            handle.write(json.dumps(value) + "\n")

    started = time.perf_counter()
    arrivals = deque(enumerate(rows))
    pending: deque[dict[str, Any]] = deque()
    probe_futures: dict[Future, dict[str, Any]] = {}
    score_futures: dict[Future, dict[str, Any]] = {}
    final_futures: dict[Future, dict[str, Any]] = {}
    ready: deque[tuple[dict[str, Any], dict[str, Any]]] = deque()
    vlm_futures: dict[Future, tuple[dict[str, Any], str]] = {}
    inflight = Counter()
    completed: list[dict[str, Any]] = []
    batcher = ClipMicroBatcher(codec, args.clip_device, args.clip_batch_images, args.clip_batch_wait_ms)

    def now() -> float:
        return time.perf_counter() - started

    def release() -> None:
        while arrivals and len(pending) < args.max_pending:
            index, row = arrivals[0]
            arrival_s = 0.0 if args.arrival_rate_qps is None else index / args.arrival_rate_qps
            if now() < arrival_s:
                return
            arrivals.popleft()
            pending.append({"row": row, "arrival_s": arrival_s})

    def choose_url() -> str:
        return min(urls, key=lambda value: (inflight[value], value))

    with ThreadPoolExecutor(max_workers=args.cpu_workers) as decode_pool, ThreadPoolExecutor(max_workers=args.vlm_concurrency) as vlm_pool:
        try:
            while arrivals or pending or probe_futures or score_futures or final_futures or ready or vlm_futures:
                release()
                while (
                    pending
                    and len(probe_futures) + len(final_futures) < args.cpu_workers
                    and len(ready) + len(final_futures) < args.ready_queue_depth
                ):
                    item = pending.popleft()
                    item["probe_started_s"] = now()
                    future = decode_pool.submit(probe_video, codec, item["row"], args)
                    probe_futures[future] = item
                    write(trace_path, {"event": "probe_admit", "qid": qid(item["row"]), "time_s": now(), "pending": len(pending)})
                while ready and len(vlm_futures) < args.vlm_concurrency:
                    item, prepared = ready.popleft()
                    url = choose_url()
                    item["vlm_submit_s"] = now()
                    inflight[url] += 1
                    vlm_futures[vlm_pool.submit(run_vlm, codec, prepared, clients[url], url, codec_args)] = (item, url)
                all_futures = list(probe_futures) + list(score_futures) + list(final_futures) + list(vlm_futures)
                if not all_futures:
                    if arrivals:
                        index, _ = arrivals[0]
                        target = 0.0 if args.arrival_rate_qps is None else index / args.arrival_rate_qps
                        time.sleep(max(0.0, min(0.05, target - now())))
                    continue
                done, _ = wait(all_futures, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in probe_futures:
                        item = probe_futures.pop(future)
                        try:
                            scored_future = batcher.submit(future.result())
                            score_futures[scored_future] = item
                        except Exception as exc:
                            failed: Future = Future()
                            failed.set_exception(exc)
                            score_futures[failed] = item
                    elif future in score_futures:
                        item = score_futures.pop(future)
                        try:
                            scored = future.result()
                            item["final_started_s"] = now()
                            final_futures[decode_pool.submit(refine_and_decode, codec, scored, args)] = item
                        except Exception as exc:
                            ready.append((item, {"error": repr(exc)}))
                    elif future in final_futures:
                        item = final_futures.pop(future)
                        try:
                            ready.append((item, future.result()))
                        except Exception as exc:
                            ready.append((item, {"error": repr(exc)}))
                    else:
                        item, url = vlm_futures.pop(future)
                        inflight[url] -= 1
                        completion = now()
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {"qid": qid(item["row"]), "correct": False, "error": repr(exc)}
                        result.update({
                            "arrival_s": item["arrival_s"], "completion_s": completion,
                            "end_to_end_delay_s": completion - item["arrival_s"],
                            "queue_wait_before_probe_s": item["probe_started_s"] - item["arrival_s"],
                            "queue_wait_before_vlm_s": item["vlm_submit_s"] - item.get("final_started_s", item["vlm_submit_s"]),
                        })
                        write(result_path, result)
                        completed.append(result)
                        print(f"[done] qid={result['qid']} correct={result.get('correct')} error={result.get('error') is not None}", flush=True)
                # Materialize preparation errors without occupying vLLM slots.
                while ready and "error" in ready[0][1]:
                    item, payload = ready.popleft()
                    completion = now()
                    result = {"qid": qid(item["row"]), "correct": False, "error": payload["error"], "arrival_s": item["arrival_s"], "completion_s": completion, "end_to_end_delay_s": completion - item["arrival_s"]}
                    write(result_path, result)
                    completed.append(result)
        finally:
            batcher.close()

    wall = now()
    delays = [float(result["end_to_end_delay_s"]) for result in completed]
    summary = {
        "method": "bounded_batched_clip_coarse_to_fine", "questions": len(completed),
        "correct": sum(bool(result.get("correct")) for result in completed),
        "errors": sum(result.get("error") is not None for result in completed),
        "accuracy_percent": 100 * sum(bool(result.get("correct")) for result in completed) / len(completed) if completed else 0.0,
        "wall_time_s": wall, "throughput_qps": len(completed) / wall if wall else 0.0,
        "mean_end_to_end_delay_s": sum(delays) / len(delays) if delays else 0.0,
        "p50_end_to_end_delay_s": percentile(delays, .5), "p95_end_to_end_delay_s": percentile(delays, .95),
        "arrival_rate_qps": args.arrival_rate_qps, "cpu_workers": args.cpu_workers,
        "vlm_concurrency": args.vlm_concurrency, "max_pending": args.max_pending,
        "ready_queue_depth": args.ready_queue_depth, "probe_frames": args.probe_frames,
        "final_frames": args.final_frames, "global_anchor_frames": args.global_anchor_frames,
        "refine_regions": args.refine_regions, "clip_batch_images": args.clip_batch_images,
        "clip_batch_wait_ms": args.clip_batch_wait_ms, "clip_device": args.clip_device, "ports": urls,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
