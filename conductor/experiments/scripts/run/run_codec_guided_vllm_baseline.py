#!/usr/bin/env python3
"""Codec-guided temporal-frame baseline for vLLM video QA.

This is deliberately not a reproduction of CodecSight. It scans compressed
packet metadata (no pixel decode), selects temporally diverse high-activity
windows, decodes only one JPEG per selected timestamp, and sends those images
to vLLM. It is intended to measure the end-to-end alternative to passing a
raw video URL to vLLM's native video loader.
"""

import argparse
import base64
import csv
import io
import json
import math
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


LOCK = threading.Lock()
CLIP_LOCK = threading.Lock()
CLIP_MODEL = None
CLIP_PROCESSOR = None


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
    valid = [chr(ord("A") + index) for index in range(choice_count)]
    text = text.strip().upper()
    if text in valid:
        return text
    match = re.search(r"\b(" + "|".join(valid) + r")\b", text)
    return match.group(1) if match else None


def make_prompt(row):
    choices = "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(row.get("choices") or [])
    )
    return (
        "Answer the multiple-choice question using the supplied video frames. "
        "The frames are in chronological order.\n\n"
        f"Question: {row['question']}\n\n{choices}\n\n"
        "Return only the answer letter."
    )


def probe_duration(video_path, timeout_s):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        check=True, capture_output=True, text=True, timeout=timeout_s,
    )
    return float(result.stdout.strip())


def scan_packet_activity(video_path, window_s, timeout_s):
    """Return compressed-byte/keyframe statistics without decoding pixels."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_packets", "-show_entries", "packet=pts_time,size,flags",
        "-of", "csv=p=0", str(video_path),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    activity = defaultdict(float)
    keyframes = []
    deadline = time.monotonic() + timeout_s
    try:
        reader = csv.reader(process.stdout)
        for fields in reader:
            if time.monotonic() > deadline:
                raise TimeoutError(f"codec packet scan exceeded {timeout_s}s")
            if len(fields) < 2 or fields[0] in {"N/A", ""}:
                continue
            try:
                timestamp = float(fields[0])
                size = int(fields[1])
            except ValueError:
                continue
            bucket = max(0, int(timestamp // window_s))
            activity[bucket] += size
            if len(fields) > 2 and "K" in fields[2]:
                keyframes.append(timestamp)
        status = process.wait(timeout=max(1, deadline - time.monotonic()))
        if status:
            error = process.stderr.read().strip()
            raise RuntimeError(f"ffprobe packet scan failed: {error}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    return dict(activity), keyframes


def temporal_anchors(duration_s, count):
    # Container duration can extend beyond the last decodable display frame.
    # Leave a conservative one-second tail margin (or 10% for tiny clips).
    end_s = max(0.0, duration_s - min(1.0, duration_s / 10.0))
    if count == 1:
        return [end_s / 2]
    return [end_s * index / (count - 1) for index in range(count)]


def select_timestamps(activity, duration_s, window_s, frame_count):
    """Keep global anchors, then add separated high compressed-activity bins."""
    n_windows = max(1, math.ceil(duration_s / window_s))
    safe_end_s = max(0.0, duration_s - min(1.0, duration_s / 10.0))
    anchor_count = min(frame_count, 3)
    selected = temporal_anchors(duration_s, anchor_count)
    candidates = sorted(
        range(n_windows), key=lambda index: activity.get(index, 0.0), reverse=True
    )
    min_separation = max(window_s, duration_s / max(frame_count, 1) / 2)
    for index in candidates:
        if len(selected) >= frame_count:
            break
        timestamp = min(safe_end_s, (index + 0.5) * window_s)
        if all(abs(timestamp - chosen) >= min_separation for chosen in selected):
            selected.append(timestamp)
    # Empty/static videos still receive an evenly spaced fixed budget.
    for timestamp in temporal_anchors(duration_s, frame_count):
        if len(selected) >= frame_count:
            break
        if all(abs(timestamp - chosen) > 0.001 for chosen in selected):
            selected.append(timestamp)
    return sorted(selected[:frame_count])


def decode_jpeg(video_path, timestamp_s, max_side, timeout_s):
    """Fast input seek: FFmpeg decodes only from the prior keyframe onward."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp_s:.3f}",
        "-i", str(video_path), "-frames:v", "1",
        "-vf", f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, timeout=timeout_s,
    )
    if not result.stdout:
        raise RuntimeError(f"empty JPEG at timestamp={timestamp_s:.3f}")
    return result.stdout


def config_name(args, frame_count):
    prefix = "codec_refined" if args.refine_with_clip else "codec_activity"
    return f"{prefix}_{frame_count}"


def clip_queries(row):
    """Use answer-aware probes when the multiple-choice options are short."""
    question = str(row["question"]).strip()
    choices = row.get("choices") or []
    if choices:
        return [f"{question}\nCandidate answer: {choice}" for choice in choices]
    return [question]


def load_clip_refiner(device):
    global CLIP_MODEL, CLIP_PROCESSOR
    with CLIP_LOCK:
        if CLIP_MODEL is None:
            print(f"Loading SigLIP refinement model on {device}", flush=True)
            CLIP_PROCESSOR = AutoProcessor.from_pretrained(
                "google/siglip-base-patch16-224"
            )
            # Keep the optional refiner small enough to coexist with a vLLM
            # server. CPU retains the default precision; CUDA uses FP16.
            model_kwargs = {}
            if str(device).startswith("cuda"):
                model_kwargs["torch_dtype"] = torch.float16
            CLIP_MODEL = AutoModel.from_pretrained(
                "google/siglip-base-patch16-224", **model_kwargs
            ).to(device)
            CLIP_MODEL.eval()
    return CLIP_MODEL, CLIP_PROCESSOR


def feature_tensor(output):
    """Normalize Transformers' model-specific embedding return wrappers."""
    if isinstance(output, torch.Tensor):
        return output
    for attribute in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(output, "last_hidden_state", None)
    if isinstance(value, torch.Tensor):
        return value[:, 0]
    raise TypeError(f"no embedding tensor in {type(output).__name__}")


@torch.no_grad()
def score_probes(probe_jpegs, row, device):
    """Return question-conditioned relevance scores for a bounded probe set."""
    model, processor = load_clip_refiner(device)
    images = [Image.open(io.BytesIO(value)).convert("RGB") for value in probe_jpegs]
    queries = clip_queries(row)
    # The shared model may run on CUDA; serialize only this small scoring step
    # so concurrent request workers never race on its processor/model state.
    with CLIP_LOCK:
        image_inputs = processor(images=images, return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        image_inputs = {
            key: (
                value.to(device=device, dtype=model_dtype)
                if value.is_floating_point() else value.to(device)
            )
            for key, value in image_inputs.items()
        }
        text_inputs = processor(
            text=queries, return_tensors="pt", padding=True,
            truncation=True, max_length=64,
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        image_feats = feature_tensor(model.get_image_features(**image_inputs))
        text_feats = feature_tensor(model.get_text_features(**text_inputs))
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        scores = (image_feats @ text_feats.T).max(dim=1).values.cpu().tolist()
    return [float(score) for score in scores]


def choose_refinement_centers(timestamps, scores, regions, radius_s):
    chosen = []
    for index in sorted(range(len(scores)), key=lambda item: scores[item], reverse=True):
        timestamp = timestamps[index]
        if all(abs(timestamp - existing) >= 2.0 * radius_s for existing in chosen):
            chosen.append(timestamp)
        if len(chosen) >= regions:
            break
    return sorted(chosen)


def dense_local_timestamps(centers, duration_s, frame_count, radius_s):
    """Spend the final VLM budget only near relevant coarse windows."""
    if not centers:
        return temporal_anchors(duration_s, frame_count)
    per_region = math.ceil(frame_count / len(centers))
    selected = []
    safe_end_s = max(0.0, duration_s - min(1.0, duration_s / 10.0))
    for center in centers:
        start = max(0.0, center - radius_s)
        end = min(safe_end_s, center + radius_s)
        for index in range(per_region):
            if per_region == 1:
                timestamp = center
            else:
                timestamp = start + (end - start) * index / (per_region - 1)
            if all(abs(timestamp - existing) > 0.001 for existing in selected):
                selected.append(timestamp)
            if len(selected) >= frame_count:
                return sorted(selected)
    # A pathological tiny/overlapping region may leave a slot; retain global
    # anchors rather than exceeding the final frame budget.
    for timestamp in temporal_anchors(duration_s, frame_count):
        if len(selected) >= frame_count:
            break
        if all(abs(timestamp - existing) > 0.001 for existing in selected):
            selected.append(timestamp)
    return sorted(selected[:frame_count])


def run_one(row, frame_count, client, base_url, args):
    started = time.perf_counter()
    duration_s = None
    timestamps = []
    probe_timestamps = []
    refinement_centers = []
    probe_scores = []
    packet_windows = 0
    probe_decode_s = 0.0
    clip_score_s = 0.0
    final_decode_s = 0.0
    error = None
    prediction_text = None
    try:
        video_path = Path(row["video"])
        duration_s = probe_duration(video_path, args.index_timeout_s)
        activity, _ = scan_packet_activity(
            video_path, args.window_s, args.index_timeout_s,
        )
        packet_windows = len(activity)
        if args.refine_with_clip:
            probe_timestamps = select_timestamps(
                activity, duration_s, args.window_s, args.refine_candidates,
            )
            t0 = time.perf_counter()
            probe_jpegs = [
                decode_jpeg(
                    video_path, timestamp, args.probe_max_side,
                    args.decode_timeout_s,
                )
                for timestamp in probe_timestamps
            ]
            probe_decode_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            probe_scores = score_probes(probe_jpegs, row, args.clip_device)
            clip_score_s = time.perf_counter() - t0
            refinement_centers = choose_refinement_centers(
                probe_timestamps, probe_scores, args.refine_regions,
                args.refine_radius_s,
            )
            timestamps = dense_local_timestamps(
                refinement_centers, duration_s, frame_count,
                args.refine_radius_s,
            )
        else:
            timestamps = select_timestamps(
                activity, duration_s, args.window_s, frame_count,
            )
        content = []
        t0 = time.perf_counter()
        for timestamp in timestamps:
            jpeg = decode_jpeg(
                video_path, timestamp, args.decode_max_side,
                args.decode_timeout_s,
            )
            data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        final_decode_s = time.perf_counter() - t0
        content.append({"type": "text", "text": make_prompt(row)})
        extra_body = {"mm_processor_kwargs": {"max_pixels": args.max_pixels}}
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        prediction_text = response.choices[0].message.content
    except Exception as exc:
        error = repr(exc)
    latency_s = time.perf_counter() - started
    choices = row.get("choices") or []
    prediction = parse_label(prediction_text, len(choices))
    gold = row.get("answer_label")
    if gold is None and row.get("answer_idx") is not None:
        gold = chr(ord("A") + int(row["answer_idx"]))
    return {
        "qid": qid(row), "video_id": row.get("video_id"),
        "video": row.get("video"), "dataset": row.get("dataset"),
        "question": row.get("question"), "choices": choices,
        "answer_idx": row.get("answer_idx"), "answer_label": gold,
        "answer": row.get("answer"),
        "config_name": config_name(args, frame_count),
        "method": (
            "codec_guided_query_refined_frames"
            if args.refine_with_clip else "codec_guided_temporal_frames"
        ),
        "retrieval_mode": (
            "codec_siglip_coarse_to_fine"
            if args.refine_with_clip else "codec_packet_activity"
        ),
        "uniform_frame_count": frame_count, "max_pixels": args.max_pixels,
        "duration_s": duration_s, "selected_timestamps_s": timestamps,
        "codec_window_s": args.window_s, "codec_packet_windows": packet_windows,
        "probe_timestamps_s": probe_timestamps,
        "probe_scores": probe_scores,
        "refinement_centers_s": refinement_centers,
        "probe_decode_s": probe_decode_s, "clip_score_s": clip_score_s,
        "final_decode_s": final_decode_s,
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
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--window-s", type=float, default=8.0)
    parser.add_argument("--decode-max-side", type=int, default=448)
    parser.add_argument("--index-timeout-s", type=float, default=120.0)
    parser.add_argument("--decode-timeout-s", type=float, default=30.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--refine-with-clip", action="store_true",
        help="Use bounded SigLIP question scoring before local frame refinement.",
    )
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--refine-candidates", type=int, default=8)
    parser.add_argument("--refine-regions", type=int, default=2)
    parser.add_argument("--refine-radius-s", type=float, default=16.0)
    parser.add_argument("--probe-max-side", type=int, default=224)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.window_s <= 0:
        parser.error("--concurrency and --window-s must be positive")
    if min(args.index_timeout_s, args.decode_timeout_s, args.request_timeout_s) <= 0:
        parser.error("all timeout values must be positive")
    if args.refine_candidates < 2 or args.refine_regions < 1:
        parser.error("--refine-candidates must be at least 2 and regions at least 1")
    if args.refine_regions > args.refine_candidates or args.refine_radius_s <= 0:
        parser.error("invalid refinement regions or radius")
    frame_counts = [int(value) for value in args.frame_counts.split(",") if value]
    if not frame_counts or any(value < 1 for value in frame_counts):
        parser.error("--frame-counts must contain positive integers")
    rows = load_jsonl(args.dataset)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    ports = [value.strip() for value in args.ports.split(",") if value.strip()]
    base_urls = [f"http://127.0.0.1:{port}/v1" for port in ports]
    clients = {
        url: OpenAI(base_url=url, api_key="EMPTY", timeout=args.request_timeout_s,
                    max_retries=0)
        for url in base_urls
    }
    completed = set()
    if args.output.exists() and not args.no_resume:
        completed = {(qid(row), row.get("config_name")) for row in load_jsonl(args.output)}
    planned = [
        (row, count) for count in frame_counts for row in rows
        if (qid(row), config_name(args, count)) not in completed
    ]
    print(
        f"examples={len(rows)} frame_counts={frame_counts} planned={len(planned)} "
        f"concurrency={args.concurrency} ports={','.join(ports)} "
        f"window_s={args.window_s} refine_with_clip={args.refine_with_clip}",
    )
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_one, row, count,
                        clients[base_urls[hash(str(row.get('video') or qid(row))) % len(base_urls)]],
                        base_urls[hash(str(row.get('video') or qid(row))) % len(base_urls)], args): (row, count)
            for row, count in planned
        }
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(args.output, result)
            results.append(result)
            print(
                f"[done] qid={result['qid']} config={result['config_name']} "
                f"pred={result['prediction_label']} gold={result['answer_label']} "
                f"correct={result['correct']} latency={result['latency_s']:.2f}s "
                f"error={result['error'] is not None}", flush=True,
            )
    wall_s = time.perf_counter() - started
    summary = {
        "dataset": str(args.dataset.resolve()), "output": str(args.output.resolve()),
        "method": (
            "codec_guided_query_refined_frames"
            if args.refine_with_clip else "codec_guided_temporal_frames"
        ), "frame_counts": frame_counts,
        "examples": len(rows), "planned_this_invocation": len(planned),
        "completed_this_invocation": len(results),
        "errors_this_invocation": sum(row["error"] is not None for row in results),
        "correct_this_invocation": sum(bool(row["correct"]) for row in results),
        "wall_time_s": wall_s, "throughput_qps": len(results) / wall_s if wall_s else 0.0,
        "ports": ports, "concurrency": args.concurrency,
        "codec_window_s": args.window_s, "decode_max_side": args.decode_max_side,
        "index_timeout_s": args.index_timeout_s, "decode_timeout_s": args.decode_timeout_s,
        "request_timeout_s": args.request_timeout_s,
        "refine_with_clip": args.refine_with_clip,
        "clip_device": args.clip_device if args.refine_with_clip else None,
        "refine_candidates": args.refine_candidates if args.refine_with_clip else 0,
        "refine_regions": args.refine_regions if args.refine_with_clip else 0,
        "refine_radius_s": args.refine_radius_s if args.refine_with_clip else 0.0,
    }
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
