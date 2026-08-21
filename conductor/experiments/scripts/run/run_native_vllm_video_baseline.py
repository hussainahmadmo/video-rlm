#!/usr/bin/env python3
"""Evaluate native vLLM uniform video sampling without semantic retrieval."""

import argparse
import json
import math
import random
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from openai import OpenAI

LOCK = threading.Lock()


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[round((len(values) - 1) * fraction)])


def load_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def qid(row):
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def parse_label(text, choice_count):
    if not text:
        return None
    valid = [chr(ord("A") + i) for i in range(choice_count)]
    normalized = text.strip().upper()
    if normalized in valid:
        return normalized
    match = re.search(
        r"(?:FINAL\s+ANSWER|ANSWER)\s*[:=]\s*("
        + "|".join(valid) + r")\b",
        normalized,
    )
    if not match:
        match = re.search(r"\b(" + "|".join(valid) + r")\b", normalized)
    return match.group(1) if match else None


def prompt(row):
    choices = "\n".join(
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(row.get("choices") or [])
    )
    return (
        "Answer the multiple-choice question using the complete video.\n\n"
        f"Question: {row['question']}\n\n{choices}\n\n"
        "Return only the answer letter."
    )


def make_video_url(row, video_mappings):
    value = str(row.get("video") or row.get("video_url") or "")
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).resolve()
    for root, base_url in video_mappings:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        encoded = "/".join(urllib.parse.quote(part) for part in relative.parts)
        return f"{base_url.rstrip('/')}/{encoded}"
    roots = ", ".join(str(root) for root, _ in video_mappings)
    raise ValueError(f"video is outside configured roots ({roots}): {path}")


def check_server(base_url):
    with urllib.request.urlopen(
        base_url.removesuffix("/v1") + "/health", timeout=10
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"unhealthy vLLM server: {base_url}")


def video_duration_s(row):
    for field in ("_measured_duration_s", "duration_s"):
        if row.get(field) is not None:
            return float(row[field])
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(row["video"]),
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def run_one(
    row, frame_count, client, base_url, args, *,
    config_name=None, schedule_row=None,
):
    started = time.perf_counter()
    url = None
    duration_s = None
    sampling_fps = None
    try:
        url = make_video_url(row, args.video_mappings)
        duration_s = video_duration_s(row)
        sampling_fps = frame_count / max(duration_s, 1e-6)
        video_io_kwargs = {
            # Qwen2/2.5-VL is FPS-driven and ignores num_frames.
            "fps": sampling_fps,
            "min_frames": frame_count,
            "max_frames": frame_count,
            "max_duration": max(1, math.ceil(duration_s)),
        }
        if args.video_backend is not None:
            video_io_kwargs["backend"] = args.video_backend
        video_io_kwargs.update(args.video_backend_kwargs)
        extra_body = {"media_io_kwargs": {"video": video_io_kwargs}}
        if args.max_pixels is not None:
            extra_body["mm_processor_kwargs"] = {
                "max_pixels": args.max_pixels,
            }
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": url}},
                {"type": "text", "text": prompt(row)},
            ]}],
            temperature=0.0,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        prediction_text = response.choices[0].message.content
        error = None
    except Exception as exc:
        prediction_text = None
        error = repr(exc)
    latency_s = time.perf_counter() - started
    choices = row.get("choices") or []
    prediction = parse_label(prediction_text, len(choices))
    gold = row.get("answer_label")
    if gold is None and row.get("answer_idx") is not None:
        gold = chr(ord("A") + int(row["answer_idx"]))
    return {
        "qid": qid(row), "video_id": row.get("video_id"),
        "video": row.get("video"), "video_url": url,
        "dataset": row.get("dataset"), "question": row.get("question"),
        "choices": choices, "answer_idx": row.get("answer_idx"),
        "answer_label": gold, "answer": row.get("answer"),
        "config_name": config_name or f"native_uniform_{frame_count}",
        "method": (
            "native_vllm_learned_adaptive_video"
            if schedule_row is not None
            else "native_vllm_uniform_video"
        ),
        "retrieval_mode": "none",
        "source_policy_config": (
            schedule_row.get("chosen_config") if schedule_row else None
        ),
        "uniform_frame_count": frame_count,
        "video_backend": args.video_backend or "server_default",
        "video_backend_kwargs": args.video_backend_kwargs,
        "max_pixels": args.max_pixels,
        "duration_s": duration_s, "sampling_fps": sampling_fps,
        "prediction_label": prediction, "prediction_text": prediction_text,
        "correct": prediction == gold, "latency_s": latency_s,
        "num_vlm_calls": 0 if error else 1, "base_url": base_url,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--ports", default="9000,9001,9002,9003")
    parser.add_argument("--frame-counts", default="2,8,32")
    parser.add_argument(
        "--schedule", type=Path,
        help=(
            "Learned schedule JSONL joined by qid. Each query uses its "
            "scheduled vlm_budget as native vLLM's uniform frame count."
        ),
    )
    parser.add_argument("--schedule-budget-field", default="vlm_budget")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--arrival-rate-qps", type=float,
        help=(
            "Open-loop offered request rate. Omit for burst arrivals at t=0. "
            "Use for fair delay-under-load comparisons with the dynamic pipeline."
        ),
    )
    parser.add_argument(
        "--arrival-order", choices=["dataset", "seeded_random", "long_short"], default="dataset",
        help=(
            "dataset preserves file order; seeded_random uses --arrival-seed; "
            "long_short alternates longest and shortest videos."
        ),
    )
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--request-timeout-s", type=float, default=600.0,
        help=(
            "Maximum wall time for one native vLLM request, including video "
            "fetch/decode and generation (default: 600). A timeout is "
            "recorded as an error so the resumable run can continue."
        ),
    )
    parser.add_argument(
        "--max-pixels", type=int, default=100352,
        help=(
            "Maximum pixels per native video frame before VLM tokenization "
            "(default: 100352 = 128 * 28 * 28). Set explicitly for "
            "reproducible visual-token bounds."
        ),
    )
    parser.add_argument(
        "--video-root", type=Path, default=Path("/workspace/datasets")
    )
    parser.add_argument(
        "--video-base-url", default="http://127.0.0.1:8088"
    )
    parser.add_argument(
        "--video-backend",
        choices=["opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"],
        help=(
            "vLLM video decoder backend. Omit to use the server default "
            "(currently OpenCV in this setup)."
        ),
    )
    parser.add_argument(
        "--video-backend-kwargs", default="{}",
        help=(
            "JSON object forwarded inside media_io_kwargs.video, for example "
            "'{\"seek_mode\": \"approximate\", \"num_ffmpeg_threads\": 4}'."
        ),
    )
    parser.add_argument(
        "--video-map", action="append", default=[],
        help="LOCAL_ROOT=HTTP_BASE; repeat for multiple roots",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--repeat-qid",
        help=(
            "Repeat exactly one dataset record; cloned requests receive unique qids. "
            "Useful for controlled single-video stress tests."
        ),
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        help="Number of unique clones to create with --repeat-qid.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    frame_counts = []
    if args.schedule is None:
        frame_counts = [
            int(value) for value in args.frame_counts.split(",")
            if value.strip()
        ]
        if not frame_counts or any(value < 1 for value in frame_counts):
            parser.error("--frame-counts must contain positive integers")
        if len(set(frame_counts)) != len(frame_counts):
            parser.error("--frame-counts contains duplicates")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.arrival_rate_qps is not None and args.arrival_rate_qps <= 0:
        parser.error("--arrival-rate-qps must be positive")
    if args.request_timeout_s <= 0:
        parser.error("--request-timeout-s must be positive")
    if args.max_pixels is not None and args.max_pixels < 28 * 28:
        parser.error("--max-pixels must be at least 784")
    try:
        args.video_backend_kwargs = json.loads(args.video_backend_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --video-backend-kwargs JSON: {exc}")
    if not isinstance(args.video_backend_kwargs, dict):
        parser.error("--video-backend-kwargs must decode to a JSON object")
    if "backend" in args.video_backend_kwargs:
        parser.error("set --video-backend instead of backend in --video-backend-kwargs")
    video_mappings = []
    for value in args.video_map:
        if "=" not in value:
            parser.error("--video-map requires LOCAL_ROOT=HTTP_BASE")
        root, base_url = value.split("=", 1)
        video_mappings.append((Path(root).resolve(), base_url))
    if not video_mappings:
        video_mappings = [(args.video_root.resolve(), args.video_base_url)]
    args.video_mappings = sorted(
        video_mappings, key=lambda item: len(str(item[0])), reverse=True
    )

    rows = load_jsonl(args.dataset)
    if args.repeat_qid is not None:
        if args.repeat_count is None or args.repeat_count < 1:
            parser.error("--repeat-qid requires --repeat-count >= 1")
        matches = [row for row in rows if qid(row) == args.repeat_qid]
        if len(matches) != 1:
            parser.error(
                f"--repeat-qid expected one matching record, found {len(matches)}: "
                f"{args.repeat_qid}"
            )
        source_row = matches[0]
        rows = []
        for index in range(args.repeat_count):
            clone = dict(source_row)
            clone["qid"] = f"{args.repeat_qid}__repeat_{index:04d}"
            rows.append(clone)
    elif args.repeat_count is not None:
        parser.error("--repeat-count requires --repeat-qid")
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    ports = [value.strip() for value in args.ports.split(",") if value.strip()]
    base_urls = [f"http://127.0.0.1:{port}/v1" for port in ports]
    for base_url in base_urls:
        check_server(base_url)
    clients = {
        # A bad or pathological video must release this worker after one
        # bounded attempt; it is written as an error and the run continues.
        url: OpenAI(
            base_url=url,
            api_key="EMPTY",
            timeout=args.request_timeout_s,
            max_retries=0,
        )
        for url in base_urls
    }

    completed = set()
    if args.output.exists() and not args.no_resume:
        completed = {
            (qid(row), row.get("config_name"))
            for row in load_jsonl(args.output)
        }
    if args.schedule is not None:
        schedule_by_qid = {}
        for schedule_row in load_jsonl(args.schedule):
            key = qid(schedule_row)
            if key in schedule_by_qid:
                parser.error(f"duplicate qid in --schedule: {key}")
            schedule_by_qid[key] = schedule_row
        missing = [qid(row) for row in rows if qid(row) not in schedule_by_qid]
        if missing:
            parser.error(
                f"--schedule is missing {len(missing)} dataset qids; "
                f"first={missing[0]}"
            )
        planned = []
        selected_counts = []
        for row in rows:
            schedule_row = schedule_by_qid[qid(row)]
            try:
                count = int(schedule_row[args.schedule_budget_field])
            except (KeyError, TypeError, ValueError) as exc:
                parser.error(
                    f"invalid {args.schedule_budget_field!r} for "
                    f"qid={qid(row)}: {exc}"
                )
            if count < 1:
                parser.error(f"non-positive scheduled budget for qid={qid(row)}")
            selected_counts.append(count)
            config_name = "native_learned_adaptive"
            if (qid(row), config_name) not in completed:
                planned.append((row, count, config_name, schedule_row))
        frame_counts = sorted(set(selected_counts))
    else:
        planned = [
            (row, count, f"native_uniform_{count}", None)
            for count in frame_counts
            for row in rows
            if (qid(row), f"native_uniform_{count}") not in completed
        ]
    if args.arrival_order == "seeded_random":
        random.Random(args.arrival_seed).shuffle(planned)
    elif args.arrival_order == "long_short":
        ordered = sorted(planned, key=lambda entry: (video_duration_s(entry[0]), qid(entry[0])))
        low, high = deque(ordered), deque(reversed(ordered))
        trace_planned = []
        while low:
            trace_planned.append(high.popleft())
            if low:
                low.pop()
            if low:
                trace_planned.append(low.popleft())
                if high:
                    high.pop()
        planned = trace_planned
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrival_trace_path = args.output.parent / "arrival_trace.jsonl"
    with arrival_trace_path.open("w") as trace_handle:
        for index, (row, count, config_name, _) in enumerate(planned):
            trace_handle.write(json.dumps({
                "arrival_index": index,
                "qid": qid(row),
                "duration_s": video_duration_s(row),
                "config_name": config_name,
                "frame_count": count,
            }) + "\n")
    arrival_description = "burst" if args.arrival_rate_qps is None else f"{args.arrival_rate_qps:g}qps"
    print(
        f"examples={len(rows)} frame_counts={frame_counts} "
        f"planned={len(planned)} concurrency={args.concurrency} "
        f"arrival_rate={arrival_description} ports={','.join(ports)}",
        flush=True,
    )

    started = time.perf_counter()
    new_results = []
    next_job = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        while next_job < len(planned) or futures:
            now_s = time.perf_counter() - started
            while next_job < len(planned) and len(futures) < args.concurrency:
                arrival_s = (
                    0.0 if args.arrival_rate_qps is None
                    else next_job / args.arrival_rate_qps
                )
                if now_s < arrival_s:
                    break
                row, count, config_name, schedule_row = planned[next_job]
                next_job += 1
                # Video affinity lets each server reuse multimodal preprocessing.
                route_key = str(row.get("video") or row.get("video_url") or qid(row))
                base_url = base_urls[hash(route_key) % len(base_urls)]
                submit_s = time.perf_counter() - started
                future = pool.submit(
                    run_one, row, count, clients[base_url], base_url, args,
                    config_name=config_name, schedule_row=schedule_row,
                )
                futures[future] = (arrival_s, submit_s)
                now_s = time.perf_counter() - started

            if not futures:
                if next_job < len(planned):
                    next_arrival_s = next_job / args.arrival_rate_qps if args.arrival_rate_qps else 0.0
                    time.sleep(max(0.0, min(0.2, next_arrival_s - (time.perf_counter() - started))))
                continue

            timeout_s = 0.2
            if next_job < len(planned) and args.arrival_rate_qps is not None:
                timeout_s = max(0.0, min(timeout_s, next_job / args.arrival_rate_qps - (time.perf_counter() - started)))
            done, _ = wait(list(futures), timeout=timeout_s, return_when=FIRST_COMPLETED)
            for future in done:
                arrival_s, submit_s = futures.pop(future)
                result = future.result()
                completion_s = time.perf_counter() - started
                result["arrival_s"] = arrival_s
                result["dispatch_s"] = submit_s
                result["prep_started_s"] = submit_s
                result["prep_ready_s"] = submit_s
                result["vlm_submit_s"] = submit_s
                result["completion_s"] = completion_s
                result["queue_wait_before_prepare_s"] = max(0.0, submit_s - arrival_s)
                result["preparation_wall_s"] = 0.0
                result["prepared_queue_wait_s"] = 0.0
                result["vlm_service_wall_s"] = max(0.0, completion_s - submit_s)
                result["end_to_end_delay_s"] = max(0.0, completion_s - arrival_s)
                append_jsonl(args.output, result)
                new_results.append(result)
                print(
                    f"[done] qid={result['qid']} config={result['config_name']} "
                    f"pred={result['prediction_label']} gold={result['answer_label']} "
                    f"correct={result['correct']} latency={result['latency_s']:.2f}s "
                    f"e2e={result['end_to_end_delay_s']:.2f}s error={result['error'] is not None}",
                    flush=True,
                )

    wall_s = time.perf_counter() - started
    delays = [float(row["end_to_end_delay_s"]) for row in new_results]
    summary = {
        "dataset": str(args.dataset.resolve()),
        "output": str(args.output.resolve()),
        "method": (
            "native_vllm_learned_adaptive_video"
            if args.schedule is not None
            else "native_vllm_uniform_video"
        ),
        "schedule": str(args.schedule.resolve()) if args.schedule else None,
        "frame_counts": frame_counts,
        "examples": len(rows),
        "planned_this_invocation": len(planned),
        "completed_this_invocation": len(new_results),
        "errors_this_invocation": sum(r["error"] is not None for r in new_results),
        "correct_this_invocation": sum(bool(r["correct"]) for r in new_results),
        "wall_time_s": wall_s,
        "throughput_qps": len(new_results) / wall_s if wall_s else 0.0,
        "arrival_rate_qps": args.arrival_rate_qps,
        "arrival_order": args.arrival_order,
        "arrival_seed": args.arrival_seed if args.arrival_order == "seeded_random" else None,
        "mean_end_to_end_delay_s": sum(delays) / len(delays) if delays else 0.0,
        "p50_end_to_end_delay_s": percentile(delays, 0.50),
        "p95_end_to_end_delay_s": percentile(delays, 0.95),
        "ports": ports, "concurrency": args.concurrency,
        "request_timeout_s": args.request_timeout_s,
        "video_backend": args.video_backend or "server_default",
        "video_backend_kwargs": args.video_backend_kwargs,
        "video_root": str(args.video_root.resolve()),
        "video_base_url": args.video_base_url,
    }
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
