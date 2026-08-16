#!/usr/bin/env python
import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def port_from_base_url(base_url):
    return base_url.rsplit(":", 1)[-1].split("/", 1)[0]


def make_scan_indices(runner, video_path, scan_fps):
    vr = runner.get_vr(video_path)
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


@torch.no_grad()
def batched_clip_retrieve(runner, batch, *, image_batch_size):
    model, processor, device = runner.get_retrieval_model()
    scan_specs = []
    all_images = []
    decode_s = 0.0
    copy_s = 0.0
    pil_s = 0.0
    processor_s = 0.0
    image_encode_s = 0.0

    for row in batch:
        item = row["item"]
        config = row["config"]
        vr, fps, nframes, scan_idxs = make_scan_indices(
            runner,
            item["video"],
            config["scan_fps"],
        )
        duration_s = nframes / fps
        start_offset = len(all_images)
        times = [idx / fps for idx in scan_idxs]

        print(
            f"[BATCH-SCAN] qid={qid(item)} frames={len(scan_idxs)} "
            f"fps={config['scan_fps']}",
            flush=True,
        )

        for start in range(0, len(scan_idxs), 256):
            chunk = scan_idxs[start:start + 256]
            if not chunk:
                continue
            runner.sync_device()
            t0 = time.time()
            frames = vr.get_batch(chunk)
            runner.sync_device()
            decode_s += time.time() - t0

            t0 = time.time()
            cpu_frames = frames.cpu().numpy()
            runner.sync_device()
            copy_s += time.time() - t0

            t0 = time.time()
            all_images.extend(
                Image.fromarray(frame).convert("RGB")
                for frame in cpu_frames
            )
            pil_s += time.time() - t0

        scan_specs.append(
            {
                "start": start_offset,
                "end": len(all_images),
                "times": times,
                "windows": windows_from_times(
                    times,
                    duration_s,
                    config["window_len_s"],
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
    }
    return retrievals, batch_latency


def build_job_from_batched_retrieval(runner, item, config, retrieval, batch_latency):
    runner.LATENCY_STATS.clear()
    for key, value in batch_latency.items():
        runner.LATENCY_STATS[key] = value / max(1.0, batch_latency["batched_clip_questions"])

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

    frames = runner.sample_frames_from_windows(
        video_path=item["video"],
        windows=selected_windows,
        frame_allocations=frame_allocations,
    )
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

    prep_latency_s = float(runner.LATENCY_STATS.get("clip_total_s", 0.0)) + (
        time.time() - start
    )
    stage_latency_s = runner.build_stage_latency_summary(prep_latency_s)
    stage_latency_s["vlm_generation_s"] = 0.0
    stage_latency_s["total_s"] = prep_latency_s
    stage_latency_s["other_s"] = max(
        0.0,
        prep_latency_s
        - stage_latency_s.get("clip_retrieval_s", 0.0)
        - stage_latency_s.get("answer_frame_pack_s", 0.0),
    )

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
        "latency_breakdown": dict(runner.LATENCY_STATS),
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
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--prepared-output", required=True)
    parser.add_argument("--results-output", required=True)
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--queue-depth", type=int, default=18)
    parser.add_argument("--prep-batch-size", type=int, default=32)
    parser.add_argument("--clip-image-batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    prep = import_from_path("build_prepared_vlm_jobs", BUILD_PREPARED_PATH)
    send = import_from_path("run_prepared_vlm_jobs", RUN_PREPARED_PATH)
    runner = prep.import_runner()

    examples = load_jsonl(args.dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    schedule_by_qid = {
        qid(row): row
        for row in load_jsonl(args.schedule)
    }

    result_done = set()
    prepared_done = set()
    if not args.no_resume:
        result_done = completed_keys(args.results_output)
        prepared_done = completed_keys(args.prepared_output)

    planned = []
    for idx, item in enumerate(examples, start=1):
        schedule_row = schedule_by_qid.get(qid(item))
        if schedule_row is None:
            print(f"[skip] missing schedule qid={qid(item)}", flush=True)
            continue
        config = prep.build_config_from_schedule(schedule_row)
        config = prep.apply_existing_policy_rewrites(runner, item, config)
        if config.get("method") != "clip_oneshot":
            print(
                f"[skip] unsupported method qid={qid(item)} "
                f"method={config.get('method')}",
                flush=True,
            )
            continue
        key = (qid(item), config["name"])
        if key in result_done:
            print(f"[{idx}/{len(examples)}] skip done qid={qid(item)}")
            continue
        planned.append({"idx": idx, "item": item, "config": config})

    ports = [port.strip() for port in args.ports.split(",") if port.strip()]
    base_urls = [f"http://localhost:{port}/v1" for port in ports]
    concurrency = args.concurrency or max(1, len(base_urls) * 2)

    print(
        f"examples={len(examples)} planned={len(planned)} "
        f"ports={','.join(ports)} concurrency={concurrency} "
        f"queue_depth={args.queue_depth} "
        f"prep_batch_size={args.prep_batch_size} "
        f"clip_image_batch_size={args.clip_image_batch_size}",
        flush=True,
    )

    send_futures = {}
    total_started = 0
    total_completed = 0
    total_prepared = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as send_pool:
        for batch_start in range(0, len(planned), args.prep_batch_size):
            batch = planned[batch_start:batch_start + args.prep_batch_size]
            print(
                f"[batch-prep] {batch_start + 1}-"
                f"{batch_start + len(batch)} size={len(batch)}",
                flush=True,
            )
            retrievals, batch_latency = batched_clip_retrieve(
                runner,
                batch,
                image_batch_size=args.clip_image_batch_size,
            )

            for row, retrieval in zip(batch, retrievals):
                item = row["item"]
                config = row["config"]
                key = (qid(item), config["name"])
                try:
                    job = build_job_from_batched_retrieval(
                        runner,
                        item,
                        config,
                        retrieval,
                        batch_latency,
                    )
                except Exception as exc:
                    import traceback

                    traceback.print_exc()
                    job = {
                        "qid": item.get("qid"),
                        "video_id": item.get("video_id"),
                        "video": item.get("video"),
                        "dataset": item.get("dataset"),
                        "question": item.get("question"),
                        "choices": item.get("choices"),
                        "answer_idx": item.get("answer_idx"),
                        "answer_label": item.get("answer_label"),
                        "answer": item.get("answer"),
                        "config_name": config.get("name"),
                        "method": config.get("method"),
                        "prepare_error": repr(exc),
                        "latency_breakdown": dict(runner.LATENCY_STATS),
                    }

                if key not in prepared_done:
                    append_jsonl(args.prepared_output, job)
                    prepared_done.add(key)
                total_prepared += 1

                while len(send_futures) >= args.queue_depth:
                    for future in as_completed(list(send_futures), timeout=None):
                        send_job = send_futures.pop(future)
                        result = future.result()
                        append_jsonl(args.results_output, result)
                        total_completed += 1
                        print(
                            f"[done] qid={result.get('qid')} "
                            f"config={result.get('config_name')} "
                            f"pred={result.get('prediction_label')} "
                            f"gold={result.get('answer_label')} "
                            f"correct={result.get('correct')} "
                            f"latency={float(result.get('latency_s') or 0.0):.2f}s",
                            flush=True,
                        )
                        break

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
                    f"inflight={len(send_futures)}",
                    flush=True,
                )

            done_now = [
                future for future in send_futures
                if future.done()
            ]
            for future in done_now:
                result = future.result()
                send_futures.pop(future)
                append_jsonl(args.results_output, result)
                total_completed += 1
                print(
                    f"[done] qid={result.get('qid')} "
                    f"config={result.get('config_name')} "
                    f"pred={result.get('prediction_label')} "
                    f"gold={result.get('answer_label')} "
                    f"correct={result.get('correct')} "
                    f"latency={float(result.get('latency_s') or 0.0):.2f}s",
                    flush=True,
                )

        for future in as_completed(send_futures):
            result = future.result()
            append_jsonl(args.results_output, result)
            total_completed += 1
            print(
                f"[done] qid={result.get('qid')} "
                f"config={result.get('config_name')} "
                f"pred={result.get('prediction_label')} "
                f"gold={result.get('answer_label')} "
                f"correct={result.get('correct')} "
                f"latency={float(result.get('latency_s') or 0.0):.2f}s",
                flush=True,
            )

    print(
        f"prepared={total_prepared} sent={total_started} "
        f"completed={total_completed} elapsed={time.time() - started:.2f}s",
        flush=True,
    )
    print(f"prepared_output={args.prepared_output}")
    print(f"results_output={args.results_output}")


if __name__ == "__main__":
    main()
