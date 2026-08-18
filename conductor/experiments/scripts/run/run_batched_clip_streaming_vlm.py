#!/usr/bin/env python
import argparse
import importlib.util
import json
import queue
import sys
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
BUILD_PREPARED_PATH = (
    ROOT / "conductor/experiments/scripts/build/build_prepared_vlm_jobs.py"
)
RUN_PREPARED_PATH = (
    ROOT / "conductor/experiments/scripts/run/run_prepared_vlm_jobs.py"
)


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def qid(row):
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def completed_keys(path):
    keys = set()
    if not Path(path).exists():
        return keys
    for row in load_jsonl(path):
        keys.add((qid(row), row.get("config_name")))
    return keys


def load_skip_qids(values, path):
    skip_qids = set(values or [])
    if path:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            skip_qids.add(line)
    return skip_qids


def log_result(result):
    print(
        f"[done] qid={result.get('qid')} "
        f"config={result.get('config_name')} "
        f"pred={result.get('prediction_label')} "
        f"gold={result.get('answer_label')} "
        f"correct={result.get('correct')} "
        f"latency={float(result.get('latency_s') or 0.0):.2f}s",
        flush=True,
    )


def port_from_base_url(base_url):
    return base_url.rsplit(":", 1)[-1].split("/", 1)[0]


def frames_to_numpy(frames):
    if hasattr(frames, "cpu"):
        return frames.cpu().numpy()
    if hasattr(frames, "asnumpy"):
        return frames.asnumpy()
    return np.asarray(frames)


def make_scan_indices(runner, video_path, scan_fps, *, private_reader=False):
    # Decord VideoReader objects are not safe to use from multiple threads.
    # Parallel decode uses one reader per task instead of the shared cache.
    vr = (
        runner.DecordVideo(video_path)
        if private_reader
        else runner.get_vr(video_path)
    )
    fps = float(vr.get_avg_fps())
    nframes = len(vr)
    step = max(1, int(round(fps / float(scan_fps))))
    idxs = list(range(0, nframes, step))
    return vr, fps, nframes, idxs


def windows_from_times(times, duration_s, window_len_s):
    windows = []
    half = float(window_len_s) / 2.0
    for ts in times:
        windows.append(
            [
                max(0.0, float(ts) - half),
                min(float(duration_s), float(ts) + half),
            ]
        )
    return windows


def select_non_overlapping_topk(scores, windows, k):
    order = np.argsort(scores)[::-1]
    selected = []
    selected_windows = []
    for idx in order:
        window = windows[idx]
        overlap = False
        for chosen in selected_windows:
            inter = max(
                0.0,
                min(window[1], chosen[1]) - max(window[0], chosen[0]),
            )
            union = max(window[1], chosen[1]) - min(window[0], chosen[0])
            if union > 0 and inter / union > 0.5:
                overlap = True
                break
        if not overlap:
            selected.append(idx)
            selected_windows.append(window)
        if len(selected) == k:
            break

    return [
        {
            "window": windows[idx],
            "score": float(scores[idx]),
        }
        for idx in selected
    ]


def decode_scan_row(runner, row, *, private_reader):
    item = row["item"]
    config = row["config"]
    vr, fps, nframes, scan_idxs = make_scan_indices(
        runner,
        item["video"],
        config["scan_fps"],
        private_reader=private_reader,
    )
    duration_s = nframes / fps
    times = [idx / fps for idx in scan_idxs]
    images = []
    decode_s = 0.0
    copy_s = 0.0
    pil_s = 0.0

    print(
        f"[BATCH-SCAN] qid={qid(item)} frames={len(scan_idxs)} "
        f"fps={config['scan_fps']}",
        flush=True,
    )

    for start in range(0, len(scan_idxs), 256):
        chunk = scan_idxs[start:start + 256]
        if not chunk:
            continue
        t0 = time.time()
        frames = vr.get_batch(chunk)
        decode_s += time.time() - t0

        t0 = time.time()
        cpu_frames = frames_to_numpy(frames)
        copy_s += time.time() - t0

        t0 = time.time()
        images.extend(
            Image.fromarray(frame).convert("RGB")
            for frame in cpu_frames
        )
        pil_s += time.time() - t0

    return {
        "images": images,
        "times": times,
        "duration_s": duration_s,
        "windows": windows_from_times(
            times,
            duration_s,
            config["window_len_s"],
        ),
        "decode_s": decode_s,
        "copy_s": copy_s,
        "pil_s": pil_s,
    }


@torch.no_grad()
def batched_clip_retrieve(
    runner,
    batch,
    *,
    image_batch_size,
    decode_workers=1,
):
    model, processor, device = runner.get_retrieval_model()
    scan_specs = []
    all_images = []
    processor_s = 0.0
    image_encode_s = 0.0

    # Image scan frames depend only on the video and scan rate, not the
    # question. Decode and encode each unique visual scan once per batch.
    unique_scan_rows = {}
    row_scan_keys = []
    for row in batch:
        key = (
            str(row["item"]["video"]),
            float(row["config"]["scan_fps"]),
        )
        row_scan_keys.append(key)
        unique_scan_rows.setdefault(key, row)

    unique_items = list(unique_scan_rows.items())
    decode_started = time.time()
    if decode_workers > 1:
        with ThreadPoolExecutor(max_workers=decode_workers) as decode_pool:
            decoded_unique = list(
                decode_pool.map(
                    lambda entry: (
                        entry[0],
                        decode_scan_row(
                            runner,
                            entry[1],
                            private_reader=True,
                        ),
                    ),
                    unique_items,
                )
            )
    else:
        decoded_unique = [
            (
                key,
                decode_scan_row(runner, row, private_reader=False),
            )
            for key, row in unique_items
        ]
    decode_wall_s = time.time() - decode_started
    decoded_by_key = dict(decoded_unique)

    raw_decode_s = sum(row["decode_s"] for _, row in decoded_unique)
    raw_copy_s = sum(row["copy_s"] for _, row in decoded_unique)
    raw_pil_s = sum(row["pil_s"] for _, row in decoded_unique)
    raw_total_s = raw_decode_s + raw_copy_s + raw_pil_s
    wall_scale = decode_wall_s / raw_total_s if raw_total_s else 0.0
    decode_s = raw_decode_s * wall_scale
    copy_s = raw_copy_s * wall_scale
    pil_s = raw_pil_s * wall_scale

    ranges_by_key = {}
    for key, decoded in decoded_unique:
        start_offset = len(all_images)
        all_images.extend(decoded["images"])
        ranges_by_key[key] = (start_offset, len(all_images))

    for row, key in zip(batch, row_scan_keys):
        decoded = decoded_by_key[key]
        start_offset, end_offset = ranges_by_key[key]
        scan_specs.append(
            {
                "start": start_offset,
                "end": end_offset,
                "times": decoded["times"],
                "windows": windows_from_times(
                    decoded["times"],
                    decoded["duration_s"],
                    row["config"]["window_len_s"],
                ),
            }
        )

    image_feats = []
    for start in range(0, len(all_images), image_batch_size):
        images = all_images[start:start + image_batch_size]
        t0 = time.time()
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        processor_s += time.time() - t0

        runner.sync_device()
        t0 = time.time()
        feats = runner.extract_feature_tensor(
            model.get_image_features(**inputs)
        )
        runner.sync_device()
        image_encode_s += time.time() - t0
        feats = feats / feats.norm(dim=-1, keepdim=True)
        image_feats.append(feats)

    if image_feats:
        image_feats = torch.cat(image_feats, dim=0)
    else:
        image_feats = torch.empty((0, 768), device=device)

    all_queries = []
    query_ranges = []
    for row in batch:
        query = runner.build_clip_query(row["item"], config=row["config"])
        if isinstance(query, (list, tuple)):
            queries = [str(value) for value in query]
        else:
            queries = [str(query)]
        query_ranges.append((len(all_queries), len(all_queries) + len(queries)))
        all_queries.extend(queries)

    t0 = time.time()
    text_inputs = processor(
        text=all_queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    text_processor_s = time.time() - t0

    runner.sync_device()
    t0 = time.time()
    text_feats = runner.extract_feature_tensor(
        model.get_text_features(**text_inputs)
    )
    runner.sync_device()
    text_encode_s = time.time() - t0
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    retrievals = []
    for row, scan, query_range in zip(batch, scan_specs, query_ranges):
        start, end = scan["start"], scan["end"]
        q_start, q_end = query_range
        feats = image_feats[start:end]
        q_feats = text_feats[q_start:q_end]
        scores = (feats @ q_feats.T).max(dim=1).values.cpu().numpy()
        top_windows = select_non_overlapping_topk(
            scores,
            scan["windows"],
            int(row["config"]["clip_topk"]),
        )
        retrievals.append(
            {
                "top_windows": top_windows,
                "candidate_windows_examined": len(scan["windows"]),
            }
        )

    batch_latency = {
        "clip_decode_s": decode_s,
        "clip_gpu_cpu_copy_s": copy_s,
        "clip_pil_s": pil_s,
        "clip_processor_s": processor_s + text_processor_s,
        "clip_image_encode_s": image_encode_s,
        "clip_text_s": text_encode_s,
        "clip_total_s": (
            decode_s
            + copy_s
            + pil_s
            + processor_s
            + text_processor_s
            + image_encode_s
            + text_encode_s
        ),
        "batched_clip_questions": float(len(batch)),
        "batched_clip_images": float(len(all_images)),
        "batched_clip_unique_scans": float(len(unique_scan_rows)),
        "scan_reuse_hits": float(len(batch) - len(unique_scan_rows)),
        "decode_workers": float(decode_workers),
        "clip_decode_wall_s": decode_wall_s,
        "clip_decode_worker_s": raw_decode_s,
    }
    return retrievals, batch_latency


def sample_answer_frames_private(runner, video_path, windows, frame_allocations, *, max_side=768, jpeg_quality=85):
    started = time.time()
    open_started = time.time()
    vr = runner.DecordVideo(video_path)
    open_s = time.time() - open_started
    fps = float(vr.get_avg_fps())
    nframes = len(vr)
    indices, timestamps = [], []
    for (start_s, end_s), count in zip(windows, frame_allocations):
        if count <= 0:
            continue
        times = ([(start_s + end_s) / 2.0] if count == 1 else np.linspace(start_s, end_s, count).tolist())
        for timestamp in times:
            index = max(0, min(nframes - 1, int(round(timestamp * fps))))
            indices.append(index)
            timestamps.append(index / fps)
    unique_indices, unique_timestamps, seen = [], [], set()
    for index, timestamp in zip(indices, timestamps):
        if index not in seen:
            seen.add(index)
            unique_indices.append(index)
            unique_timestamps.append(timestamp)
    if not unique_indices:
        raise RuntimeError("No answer frames selected")
    t0 = time.time()
    frames = vr.get_batch(unique_indices)
    answer_decode_s = time.time() - t0
    t0 = time.time()
    cpu_frames = frames_to_numpy(frames)
    copy_s = time.time() - t0
    images, pil_s, jpeg_s = [], 0.0, 0.0
    for array in cpu_frames:
        t0 = time.time()
        image = Image.fromarray(array).convert("RGB")
        pil_s += time.time() - t0
        t0 = time.time()
        images.append(runner.pil_to_data_url(image, max_side=max_side, jpeg_quality=jpeg_quality))
        jpeg_s += time.time() - t0
    frame_extract_s = time.time() - started
    return ({
        "video_path": video_path,
        "timestamps": unique_timestamps,
        "frame_indices": unique_indices,
        "images": images,
    }, {
        "answer_video_open_s": open_s,
        "answer_decode_s": answer_decode_s,
        "gpu_cpu_copy_s": copy_s,
        "pil_s": pil_s,
        "jpeg_base64_s": jpeg_s,
        "frame_extract_s": frame_extract_s,
    })


def build_job_from_batched_retrieval(runner, item, config, retrieval, batch_latency, *, answer_frame_workers=1):
    latency_breakdown = {}
    for key, value in batch_latency.items():
        if key == "decode_workers":
            latency_breakdown[key] = value
        else:
            latency_breakdown[key] = value / max(1.0, batch_latency["batched_clip_questions"])
    latency_breakdown["answer_frame_workers"] = float(answer_frame_workers)

    start = time.time()
    top_windows = retrieval["top_windows"]
    selected_windows = runner.apply_profiler_window_hints(
        item=item,
        top_windows=top_windows,
        config=config,
    )
    if not selected_windows:
        raise RuntimeError("No windows selected")

    total_frames = int(config["vlm_budget"])
    base = total_frames // len(selected_windows)
    extra = total_frames % len(selected_windows)
    frame_allocations = [
        base + (1 if idx < extra else 0)
        for idx in range(len(selected_windows))
    ]

    frames, frame_latency = sample_answer_frames_private(
        runner, item["video"], selected_windows, frame_allocations
    )
    latency_breakdown.update(frame_latency)
    if len(frames["images"]) > 64:
        raise RuntimeError(f"Too many images: {len(frames['images'])}")

    evidence_hint = runner.build_profiler_evidence_context(
        selected_windows=selected_windows,
        top_windows=top_windows,
        config=config,
    )
    if config.get("answer_with_confidence"):
        prompt = runner.build_answer_prompt_with_confidence(
            item["question"],
            item["choices"],
            extra_context=evidence_hint,
        )
    else:
        prompt = runner.build_answer_prompt(
            item["question"],
            item["choices"],
            extra_context=evidence_hint,
        )

    clip_retrieval_s = float(latency_breakdown.get("clip_total_s", 0.0))
    answer_frame_pack_s = float(latency_breakdown.get("frame_extract_s", 0.0))
    prep_latency_s = clip_retrieval_s + (time.time() - start)
    stage_latency_s = {
        "clip_retrieval_s": clip_retrieval_s,
        "answer_frame_pack_s": answer_frame_pack_s,
        "vlm_generation_s": 0.0,
        "other_s": max(0.0, prep_latency_s - clip_retrieval_s - answer_frame_pack_s),
        "total_s": prep_latency_s,
    }

    return {
        "qid": item["qid"],
        "video_id": item.get("video_id"),
        "video": item.get("video"),
        "dataset": item.get("dataset"),
        "question": item["question"],
        "choices": item["choices"],
        "answer_idx": item.get("answer_idx"),
        "answer_label": item.get("answer_label"),
        "answer": item.get("answer"),
        "question_category": item.get("question_category"),
        "topic_category": item.get("topic_category"),
        "vimio_profile": item.get("vimio_profile"),
        "duration_s": item.get("duration_s"),
        "duration_bucket": item.get("duration_bucket"),
        "lvb_duration_bucket": item.get("lvb_duration_bucket"),
        "config_name": config["name"],
        "method": config["method"],
        "scan_fps": config.get("scan_fps"),
        "clip_topk": config.get("clip_topk"),
        "window_len_s": config.get("window_len_s"),
        "vlm_budget": config.get("vlm_budget"),
        "evidence_type": config.get("evidence_type"),
        "expand_neighbors": config.get("expand_neighbors"),
        "preserve_order": config.get("preserve_order"),
        "include_uniform_anchors": config.get("include_uniform_anchors"),
        "answer_with_confidence": config.get("answer_with_confidence"),
        "scheduler_reason": config.get("scheduler_reason"),
        "scheduler_query_class": config.get("scheduler_query_class"),
        "scheduler_gpu_state": config.get("scheduler_gpu_state"),
        "prompt": prompt,
        "frames": frames,
        "prep_latency_s": prep_latency_s,
        "latency_breakdown": latency_breakdown,
        "stage_latency_s": stage_latency_s,
        "decord_ctx": runner.DECORD_CTX,
        "clip_device": runner.CLIP_DEVICE,
        "retrieval_effort": {
            "candidate_windows": retrieval["candidate_windows_examined"],
            "selected_windows": len(selected_windows),
            "selected_frames": len(frames["frame_indices"]),
        },
        "evidence": {
            "clip_top_windows": top_windows,
            "selected_windows": selected_windows,
            "selected_frame_indices": frames["frame_indices"],
            "selected_timestamps": frames["timestamps"],
        },
        "prepare_error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--schedule",
        action="append",
        required=True,
        help=(
            "Schedule JSONL. Repeat this option to evaluate multiple policies "
            "in one persistent process with one retrieval-model load."
        ),
    )
    parser.add_argument("--prepared-output", required=True)
    parser.add_argument("--results-output", required=True)
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--queue-depth", type=int, default=18)
    parser.add_argument("--prepared-queue-depth", type=int, default=None)
    parser.add_argument("--prep-batch-size", type=int, default=32)
    parser.add_argument("--clip-image-batch-size", type=int, default=128)
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=1,
        help=(
            "Number of independent CPU video-decode workers used per "
            "preparation batch. Values above one use private Decord readers."
        ),
    )
    parser.add_argument("--answer-frame-workers", type=int, default=1, help="Independent CPU workers for selected answer-frame extraction.")
    parser.add_argument("--decode-ahead-batches", type=int, default=1, help="Bounded compact retrieval batches prepared ahead of answer-frame packing.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-qid",
        action="append",
        default=[],
        help="Question id to skip before video decode. Can be repeated.",
    )
    parser.add_argument(
        "--skip-qids-file",
        default=None,
        help="Optional newline-delimited qid file to skip before video decode.",
    )
    args = parser.parse_args()
    if args.decode_workers < 1:
        parser.error("--decode-workers must be at least 1")
    if args.answer_frame_workers < 1:
        parser.error("--answer-frame-workers must be at least 1")
    if args.decode_ahead_batches < 1:
        parser.error("--decode-ahead-batches must be at least 1")

    prep = import_from_path("build_prepared_vlm_jobs", BUILD_PREPARED_PATH)
    send = import_from_path("run_prepared_vlm_jobs", RUN_PREPARED_PATH)
    runner = prep.import_runner()

    examples = load_jsonl(args.dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    schedules = []
    for schedule_path in args.schedule:
        schedule_rows = load_jsonl(schedule_path)
        schedule_by_qid = {qid(row): row for row in schedule_rows}
        if len(schedule_by_qid) != len(schedule_rows):
            raise SystemExit(
                f"duplicate qids in schedule: {schedule_path}"
            )
        schedules.append((schedule_path, schedule_by_qid))

    result_done = set()
    prepared_done = set()
    if not args.no_resume:
        result_done = completed_keys(args.results_output)
        prepared_done = completed_keys(args.prepared_output)

    skip_qids = load_skip_qids(args.skip_qid, args.skip_qids_file)
    if skip_qids:
        print(f"skip_qids={len(skip_qids)}", flush=True)

    planned = []
    planned_keys = set()
    for schedule_path, schedule_by_qid in schedules:
        for idx, item in enumerate(examples, start=1):
            item_qid = qid(item)
            if item_qid in skip_qids:
                print(
                    f"[{idx}/{len(examples)}] skip requested qid={item_qid}",
                    flush=True,
                )
                continue
            schedule_row = schedule_by_qid.get(item_qid)
            if schedule_row is None:
                print(
                    f"[skip] missing schedule={schedule_path} qid={item_qid}",
                    flush=True,
                )
                continue
            config = prep.build_config_from_schedule(schedule_row)
            config = prep.apply_existing_policy_rewrites(runner, item, config)
            if config.get("method") != "clip_oneshot":
                print(
                    f"[skip] unsupported method qid={item_qid} "
                    f"method={config.get('method')}",
                    flush=True,
                )
                continue
            key = (item_qid, config["name"])
            if key in planned_keys:
                raise SystemExit(
                    "duplicate question/config across schedules: "
                    f"qid={item_qid} config={config['name']}"
                )
            planned_keys.add(key)
            if key in result_done:
                print(
                    f"[{idx}/{len(examples)}] skip done "
                    f"qid={item_qid} config={config['name']}"
                )
                continue
            planned.append(
                {
                    "idx": idx,
                    "item": item,
                    "config": config,
                    "schedule": schedule_path,
                }
            )

    ports = [port.strip() for port in args.ports.split(",") if port.strip()]
    base_urls = [f"http://localhost:{port}/v1" for port in ports]
    concurrency = args.concurrency or max(1, len(base_urls) * 2)

    print(
        f"examples={len(examples)} schedules={len(schedules)} "
        f"planned={len(planned)} "
        f"ports={','.join(ports)} concurrency={concurrency} "
        f"queue_depth={args.queue_depth} "
        f"prepared_queue_depth={args.prepared_queue_depth or args.queue_depth} "
        f"prep_batch_size={args.prep_batch_size} "
        f"clip_image_batch_size={args.clip_image_batch_size} "
        f"decode_workers={args.decode_workers} "
        f"answer_frame_workers={args.answer_frame_workers} "
        f"decode_ahead_batches={args.decode_ahead_batches}",
        flush=True,
    )

    prepared_queue = queue.Queue(
        maxsize=args.prepared_queue_depth or args.queue_depth
    )
    sentinel = object()
    prep_errors = []
    send_futures = {}
    total_started = 0
    total_completed = 0
    total_prepared = 0
    started = time.time()

    retrieval_queue = queue.Queue(maxsize=args.decode_ahead_batches)
    retrieval_sentinel = object()
    pipeline_stop = threading.Event()

    def bounded_put(target_queue, value):
        while not pipeline_stop.is_set():
            try:
                target_queue.put(value, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def retrieve_batches():
        try:
            batch_start = 0
            while batch_start < len(planned):
                if pipeline_stop.is_set():
                    break
                batch_end = min(
                    len(planned),
                    batch_start + args.prep_batch_size,
                )
                hard_end = min(
                    len(planned),
                    batch_start + 2 * args.prep_batch_size,
                )
                # Treat prep_batch_size as a soft boundary: keep adjacent
                # questions for one video together, up to a bounded 2x size.
                while (
                    batch_end < hard_end
                    and planned[batch_end]["item"].get("video")
                    == planned[batch_end - 1]["item"].get("video")
                ):
                    batch_end += 1
                batch = planned[batch_start:batch_end]
                print(
                    f"[batch-retrieve] {batch_start + 1}-{batch_end} "
                    f"size={len(batch)}",
                    flush=True,
                )
                retrievals, batch_latency = batched_clip_retrieve(
                    runner, batch,
                    image_batch_size=args.clip_image_batch_size,
                    decode_workers=args.decode_workers,
                )
                if not bounded_put(retrieval_queue, (batch, retrievals, batch_latency)):
                    break
                batch_start = batch_end
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            prep_errors.append(exc)
            pipeline_stop.set()
        finally:
            bounded_put(retrieval_queue, retrieval_sentinel)

    def prepare_jobs():
        try:
            with ThreadPoolExecutor(max_workers=args.answer_frame_workers) as answer_pool:
                while not pipeline_stop.is_set():
                    payload = retrieval_queue.get()
                    if payload is retrieval_sentinel:
                        break
                    batch, retrievals, batch_latency = payload
                    futures = {
                        answer_pool.submit(
                            build_job_from_batched_retrieval,
                            runner, row["item"], row["config"], retrieval, batch_latency,
                            answer_frame_workers=args.answer_frame_workers,
                        ): row
                        for row, retrieval in zip(batch, retrievals)
                    }
                    for future in as_completed(futures):
                        row = futures[future]
                        item, config = row["item"], row["config"]
                        key = (qid(item), config["name"])
                        try:
                            job = future.result()
                        except Exception as exc:
                            import traceback
                            traceback.print_exc()
                            job = {
                                "qid": item.get("qid"), "video_id": item.get("video_id"),
                                "video": item.get("video"), "dataset": item.get("dataset"),
                                "question": item.get("question"), "choices": item.get("choices"),
                                "answer_idx": item.get("answer_idx"), "answer_label": item.get("answer_label"),
                                "answer": item.get("answer"), "config_name": config.get("name"),
                                "method": config.get("method"), "prepare_error": repr(exc),
                                "latency_breakdown": {"answer_frame_workers": float(args.answer_frame_workers)},
                            }
                        if key not in prepared_done:
                            append_jsonl(args.prepared_output, job)
                            prepared_done.add(key)
                        if not bounded_put(prepared_queue, job):
                            break
                        print(f"[prepared] qid={qid(job)} config={job.get('config_name')} queue={prepared_queue.qsize()}", flush=True)
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            prep_errors.append(exc)
            pipeline_stop.set()
        finally:
            prepared_queue.put(sentinel)

    def submit_job(send_pool, job):
        nonlocal total_started
        base_url = base_urls[total_started % len(base_urls)]
        future = send_pool.submit(
            send.run_one,
            job,
            base_url,
            args.max_tokens,
        )
        send_futures[future] = job
        total_started += 1
        print(
            f"[send] qid={qid(job)} "
            f"port={port_from_base_url(base_url)} "
            f"inflight={len(send_futures)} "
            f"prepared_queue={prepared_queue.qsize()}",
            flush=True,
        )

    def drain_done(done_futures):
        nonlocal total_completed
        for future in list(done_futures):
            send_futures.pop(future, None)
            result = future.result()
            append_jsonl(args.results_output, result)
            total_completed += 1
            log_result(result)

    retrieval_thread = threading.Thread(target=retrieve_batches, daemon=True)
    prep_thread = threading.Thread(target=prepare_jobs, daemon=True)
    retrieval_thread.start()
    prep_thread.start()

    with ThreadPoolExecutor(max_workers=concurrency) as send_pool:
        prep_finished = False
        while not prep_finished or send_futures:
            while not prep_finished and len(send_futures) < args.queue_depth:
                try:
                    timeout = 0.1 if not send_futures else 0.0
                    job = prepared_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if job is sentinel:
                    prep_finished = True
                    break
                total_prepared += 1
                submit_job(send_pool, job)

            done_now = [
                future for future in send_futures
                if future.done()
            ]
            if done_now:
                drain_done(done_now)
                continue

            if send_futures:
                done_now, _ = wait(
                    list(send_futures),
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                if done_now:
                    drain_done(done_now)
            elif not prep_finished:
                time.sleep(0.05)

    prep_thread.join()
    pipeline_stop.set()
    retrieval_thread.join()
    if prep_errors:
        raise prep_errors[0]

    print(
        f"prepared={total_prepared} sent={total_started} "
        f"completed={total_completed} elapsed={time.time() - started:.2f}s",
        flush=True,
    )
    print(f"prepared_output={args.prepared_output}")
    print(f"results_output={args.results_output}")


if __name__ == "__main__":
    main()
