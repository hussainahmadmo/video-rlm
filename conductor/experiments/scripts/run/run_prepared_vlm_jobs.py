#!/usr/bin/env python
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
_WRITE_LOCK = threading.Lock()


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with _WRITE_LOCK:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def qid(row):
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def parse_mcq_label(response, num_choices=None):
    if response is None:
        return None

    if num_choices is None:
        valid = ["A", "B", "C", "D", "E"]
    else:
        valid = [chr(ord("A") + idx) for idx in range(num_choices)]

    text = response.strip().upper()
    if text in valid:
        return text

    final_pattern = (
        r"(?:FINAL\s+ANSWER|ANSWER)\s*[:=]\s*("
        + "|".join(valid)
        + r")\b"
    )
    match = re.search(final_pattern, text)
    if match:
        return match.group(1)

    if "CHOICE CHECKS" in text or "EVIDENCE TIMELINE" in text:
        return None

    match = re.search(r"\b(" + "|".join(valid) + r")\b", text)
    if match:
        return match.group(1)
    return None


def parse_confidence(response):
    if response is None:
        return None

    match = re.search(
        r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)",
        response.strip(),
        re.IGNORECASE,
    )
    if not match:
        return None

    try:
        confidence = float(match.group(1))
    except Exception:
        return None

    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def make_content(job):
    frames = job["frames"]
    content = []
    for idx, img_url in enumerate(frames["images"]):
        timestamp = frames["timestamps"][idx]
        content.append(
            {
                "type": "text",
                "text": f"Frame {idx} at {timestamp:.1f}s:",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": img_url,
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": job["prompt"],
        }
    )
    return content


def call_vlm(job, base_url, max_tokens):
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )

    t0 = time.time()
    content = make_content(job)
    prepare_s = time.time() - t0

    t_request = time.time()
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    request_s = time.time() - t_request
    call_s = time.time() - t0

    return response.choices[0].message.content, {
        "vlm_prepare_s": prepare_s,
        "vlm_request_s": request_s,
        "vlm_call_s": call_s,
        "vlm_call_count": 1.0,
    }


def run_one(job, base_url, max_tokens):
    if job.get("prepare_error"):
        return {
            **base_result_fields(job),
            "prediction_label": None,
            "prediction_text": None,
            "prediction_confidence": None,
            "correct": False,
            "latency_s": job.get("prep_latency_s"),
            "num_vlm_calls": 0,
            "error": job.get("prepare_error"),
        }

    try:
        text, vlm_stats = call_vlm(job, base_url, max_tokens=max_tokens)
        error = None
    except Exception as exc:
        text = None
        vlm_stats = {}
        error = repr(exc)

    label = parse_mcq_label(text, num_choices=len(job.get("choices") or []))
    confidence = parse_confidence(text)
    gold = job.get("answer_label")

    latency_breakdown = dict(job.get("latency_breakdown") or {})
    latency_breakdown.update(vlm_stats)

    prep_latency_s = float(job.get("prep_latency_s") or 0.0)
    vlm_call_s = float(vlm_stats.get("vlm_call_s") or 0.0)
    latency_s = prep_latency_s + vlm_call_s

    stage_latency_s = dict(job.get("stage_latency_s") or {})
    stage_latency_s["vlm_generation_s"] = vlm_call_s
    stage_latency_s["total_s"] = latency_s
    measured = (
        float(stage_latency_s.get("clip_retrieval_s") or 0.0)
        + float(stage_latency_s.get("answer_frame_pack_s") or 0.0)
        + vlm_call_s
    )
    stage_latency_s["other_s"] = max(0.0, latency_s - measured)

    return {
        **base_result_fields(job),
        "prediction_label": label,
        "prediction_text": text,
        "prediction_confidence": confidence,
        "correct": label == gold,
        "latency_s": latency_s,
        "prep_latency_s": prep_latency_s,
        "num_vlm_calls": int(vlm_stats.get("vlm_call_count") or 0),
        "latency_breakdown": latency_breakdown,
        "stage_latency_s": stage_latency_s,
        "error": error,
    }


def base_result_fields(job):
    return {
        "qid": job.get("qid"),
        "video_id": job.get("video_id"),
        "video": job.get("video"),
        "dataset": job.get("dataset"),
        "question": job.get("question"),
        "choices": job.get("choices"),
        "answer_idx": job.get("answer_idx"),
        "answer_label": job.get("answer_label"),
        "answer": job.get("answer"),
        "question_category": job.get("question_category"),
        "topic_category": job.get("topic_category"),
        "vimio_profile": job.get("vimio_profile"),
        "duration_s": job.get("duration_s"),
        "duration_bucket": job.get("duration_bucket"),
        "lvb_duration_bucket": job.get("lvb_duration_bucket"),
        "scan_fps": job.get("scan_fps"),
        "clip_topk": job.get("clip_topk"),
        "window_len_s": job.get("window_len_s"),
        "config_name": job.get("config_name"),
        "method": job.get("method"),
        "vlm_budget": job.get("vlm_budget"),
        "evidence_type": job.get("evidence_type"),
        "expand_neighbors": job.get("expand_neighbors"),
        "preserve_order": job.get("preserve_order"),
        "include_uniform_anchors": job.get("include_uniform_anchors"),
        "decord_ctx": job.get("decord_ctx"),
        "clip_device": job.get("clip_device"),
        "evidence": job.get("evidence"),
        "retrieval_effort": job.get("retrieval_effort"),
        "fallback_used": False,
        "scheduler_reason": job.get("scheduler_reason"),
        "scheduler_query_class": job.get("scheduler_query_class"),
        "scheduler_gpu_state": job.get("scheduler_gpu_state"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    jobs = load_jsonl(args.jobs)
    if args.max_examples is not None:
        jobs = jobs[: args.max_examples]

    output = Path(args.output)
    done = set()
    if output.exists() and not args.no_resume:
        for row in load_jsonl(output):
            done.add((qid(row), row.get("config_name")))

    jobs = [
        row for row in jobs
        if (qid(row), row.get("config_name")) not in done
    ]

    ports = [
        port.strip()
        for port in args.ports.split(",")
        if port.strip()
    ]
    base_urls = [f"http://localhost:{port}/v1" for port in ports]
    concurrency = args.concurrency or max(1, len(base_urls) * 2)

    print(
        f"jobs={len(jobs)} ports={','.join(ports)} concurrency={concurrency}",
        flush=True,
    )

    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for idx, job in enumerate(jobs):
            base_url = base_urls[idx % len(base_urls)]
            fut = pool.submit(run_one, job, base_url, args.max_tokens)
            futures[fut] = job

        for fut in as_completed(futures):
            row = fut.result()
            append_jsonl(output, row)
            completed += 1
            print(
                f"[{completed}/{len(jobs)}] "
                f"qid={row.get('qid')} "
                f"config={row.get('config_name')} "
                f"pred={row.get('prediction_label')} "
                f"gold={row.get('answer_label')} "
                f"correct={row.get('correct')} "
                f"latency={float(row.get('latency_s') or 0.0):.2f}s",
                flush=True,
            )

    print(
        f"wrote: {output} elapsed={time.time() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
