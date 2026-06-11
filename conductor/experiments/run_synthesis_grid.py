import argparse
import base64
import io
import json
import time
import re
from pathlib import Path

import decord
import numpy as np
from PIL import Image
from openai import OpenAI

import torch
import clip

from openai import OpenAI

import os

CLIP_DEVICE = os.environ.get("CLIP_DEVICE", "cuda:1")

VLM_BASE_URL = "http://localhost:9000/v1"
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

client = OpenAI(
    base_url=VLM_BASE_URL,
    api_key="EMPTY",
)

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_VR_CACHE = {}
_CLIP_RANKING_CACHE = {}

EXPERIMENT_CONFIGS = [
    {
        "name": "oneshot_32",
        "method": "oneshot",
        "total_frames": 32,
        "num_windows": 1,
        "frames_per_window": 32,
        "query_conditioned": False,
    },
    {
        "name": "global_32",
        "method": "global_summary",
        "total_frames": 32,
        "num_windows": 1,
        "frames_per_window": 32,
        "query_conditioned": False,
    },
    {
        "name": "global_q_32",
        "method": "global_summary",
        "total_frames": 32,
        "num_windows": 1,
        "frames_per_window": 32,
        "query_conditioned": True,
    },
    {
        "name": "map_4x8",
        "method": "map_summary",
        "total_frames": 32,
        "num_windows": 4,
        "frames_per_window": 8,
        "query_conditioned": False,
    },
    {
        "name": "map_8x4",
        "method": "map_summary",
        "total_frames": 32,
        "num_windows": 8,
        "frames_per_window": 4,
        "query_conditioned": False,
    },
    {
        "name": "qmap_4x8",
        "method": "map_summary",
        "total_frames": 32,
        "num_windows": 4,
        "frames_per_window": 8,
        "query_conditioned": True,
    },
    {
        "name": "qmap_8x4",
        "method": "map_summary",
        "total_frames": 32,
        "num_windows": 8,
        "frames_per_window": 4,
        "query_conditioned": True,
    },
    {
        "name": "clip_k8_f2_stride32",
        "method": "clip_oneshot",
        "clip_topk": 8,
        "frames_per_window": 2,
        "window_len_s": 8.0,
        "stride_s": 32.0,
        "total_frames": 16,
        "num_windows": 8,
        "query_conditioned": False,
    },
    {
        "name": "clip_k10_f2_stride16",
        "method": "clip_oneshot",
        "clip_topk": 10,
        "frames_per_window": 2,
        "window_len_s": 8.0,
        "stride_s": 16.0,
        "total_frames": 20,
        "num_windows": 10,
        "query_conditioned": False,
    },
    {
        "name": "clip_k16_f2_stride16",
        "method": "clip_oneshot",
        "clip_topk": 16,
        "frames_per_window": 2,
        "window_len_s": 8.0,
        "stride_s": 16.0,
        "total_frames": 32,
        "num_windows": 16,
        "query_conditioned": False,
    }
]

def get_vr(video_path):
    if video_path not in _VR_CACHE:
        _VR_CACHE[video_path] = decord.VideoReader(
            video_path,
            ctx=decord.cpu(0),
        )
    return _VR_CACHE[video_path]

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(row, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def make_uniform_windows(duration_s, num_windows):
    window_size = duration_s / num_windows
    return [
        (i * window_size, min((i + 1) * window_size, duration_s))
        for i in range(num_windows)
    ]


def pil_to_data_url(img, max_side=768, jpeg_quality=85):
    img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def sample_uniform_frames(video_path, num_frames, start_s, end_s, max_side=768, jpeg_quality=85):
    vr = get_vr(video_path)
    fps = float(vr.get_avg_fps())
    n = len(vr)

    duration_s = n / fps
    start_s = max(0.0, float(start_s))
    end_s = min(float(end_s), duration_s)

    if end_s <= start_s:
        end_s = min(duration_s, start_s + 1.0)

    if num_frames == 1:
        times = [(start_s + end_s) / 2.0]
    else:
        times = np.linspace(start_s, end_s, num_frames).tolist()

    idxs = []
    timestamps = []

    for t in times:
        idx = int(round(t * fps))
        idx = max(0, min(n - 1, idx))
        idxs.append(idx)
        timestamps.append(idx / fps)

    # Remove duplicate frame indices while preserving order
    unique_idxs = []
    unique_timestamps = []
    seen = set()

    for idx, ts in zip(idxs, timestamps):
        if idx not in seen:
            unique_idxs.append(idx)
            unique_timestamps.append(ts)
            seen.add(idx)

    batch = vr.get_batch(unique_idxs).asnumpy()

    images = []
    for arr in batch:
        img = Image.fromarray(arr).convert("RGB")
        images.append(
            pil_to_data_url(
                img,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
            )
        )

    return {
        "video_path": video_path,
        "timestamps": unique_timestamps,
        "frame_indices": unique_idxs,
        "images": images,
    }


def format_choices(choices):
    return "\n".join(
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(choices)
    )


def build_answer_prompt(question, choices, extra_context=None):
    context = ""
    if extra_context:
        context = f"\nEvidence:\n{extra_context}\n"

    labels = [chr(ord("A") + i) for i in range(len(choices))]
    label_text = ", ".join(labels)

    return f"""
You are answering a multiple-choice question about a video.
{context}
Question:
{question}

Choices:
{format_choices(choices)}

Return only one answer label from: {label_text}.
Do not return any other letter.
""".strip()


def build_global_summary_prompt(item, query_conditioned):
    if query_conditioned:
        return f"""
Summarize only the video evidence needed to answer this question.

Question:
{item["question"]}

Choices:
{format_choices(item["choices"])}

Focus on answer-relevant people, objects, actions, timing, scene changes, and visible text.
Do not answer yet. Only provide evidence from the video.
""".strip()

    return """
Summarize the important visual events in this video.
Focus on people, objects, actions, scene changes, visible text, and notable events.
Be concise but include concrete details.
""".strip()


def build_window_summary_prompt(item, start_s, end_s, query_conditioned):
    if query_conditioned:
        return f"""
This video segment is from {start_s:.1f}s to {end_s:.1f}s.

Question:
{item["question"]}

Choices:
{format_choices(item["choices"])}

Summarize only evidence in this segment that could help answer the question.
If this segment does not contain relevant evidence, say: "No relevant evidence."
Do not guess the answer.
""".strip()

    return f"""
Summarize the visual content in this video segment from {start_s:.1f}s to {end_s:.1f}s.
Focus on concrete actions, objects, people, visible text, and scene changes.
Be concise.
""".strip()


def parse_mcq_label(response, num_choices=None):
    if response is None:
        return None

    if num_choices is None:
        valid = ["A", "B", "C", "D", "E"]
    else:
        valid = [chr(ord("A") + i) for i in range(num_choices)]

    text = response.strip().upper()

    if text in valid:
        return text

    pattern = r"\b(" + "|".join(valid) + r")\b"
    m = re.search(pattern, text)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------
# TODO: Replace these three functions with your actual Qwen/VLM calls.
# ---------------------------------------------------------------------

def call_vlm_images(frames, prompt, max_tokens=512):
    print(f"[REAL VLM CALL] {len(frames['images'])} images", flush=True)

    content = []

    for i, img_url in enumerate(frames["images"]):
        ts = frames["timestamps"][i]

        content.append({
            "type": "text",
            "text": f"Frame {i} at {ts:.1f}s:"
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": img_url,
            },
        })

    content.append({
        "type": "text",
        "text": prompt,
    })

    resp = client.chat.completions.create(
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

    return resp.choices[0].message.content


def call_vlm_answer(frames, prompt):
    return call_vlm_images(
        frames=frames,
        prompt=prompt,
        max_tokens=32,
    )


def call_vlm_text(frames, prompt):
    return call_vlm_images(
        frames=frames,
        prompt=prompt,
        max_tokens=512,
    )


def call_llm_answer(prompt):
    print("[REAL LLM CALL]", flush=True)

    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=32,
    )

    return resp.choices[0].message.content


def get_clip_model(device=None):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    if device is None:
        device = CLIP_DEVICE

    if _CLIP_MODEL is None:
        _CLIP_DEVICE = device if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP ViT-B/32 on {_CLIP_DEVICE}", flush=True)
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=_CLIP_DEVICE)
        _CLIP_MODEL.eval()

    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

def duration_fps_nframes(video_path):
    vr = get_vr(video_path)
    fps = float(vr.get_avg_fps())
    return len(vr) / fps, fps, len(vr)


def make_sliding_windows(duration_s, window_len_s=8.0, stride_s=4.0, margin_s=5.0):
    first_start = margin_s
    last_start = max(first_start, duration_s - margin_s - window_len_s)

    windows = []
    s = first_start

    while s <= last_start:
        windows.append([float(s), float(min(duration_s, s + window_len_s))])
        s += stride_s

    if not windows:
        windows = [[0.0, float(min(duration_s, window_len_s))]]

    return windows


def sample_clip_frames_for_scoring(video_path, window, frames_per_candidate=1):
    vr = get_vr(video_path)
    fps = float(vr.get_avg_fps())
    n = len(vr)

    s, e = window

    if frames_per_candidate == 1:
        times = [(s + e) / 2.0]
    else:
        times = np.linspace(s, e, frames_per_candidate).tolist()

    idxs = []
    for t in times:
        idx = int(round(t * fps))
        idx = max(0, min(n - 1, idx))
        idxs.append(idx)

    idxs = list(dict.fromkeys(idxs))
    batch = vr.get_batch(idxs).asnumpy()
    images = [Image.fromarray(arr).convert("RGB") for arr in batch]

    return images, idxs

@torch.no_grad()
def get_window_embeddings(
    video_path,
    window_len_s,
    stride_s,
    frames_per_candidate=1,
    batch_size=64,
):
    print(
        f"[CLIP BUILD] {Path(video_path).name}",
        flush=True,
    )

    duration_s, _, _ = duration_fps_nframes(video_path)

    candidate_windows = make_sliding_windows(
        duration_s=duration_s,
        window_len_s=window_len_s,
        stride_s=stride_s,
    )

    model, preprocess, device = get_clip_model()

    all_window_features = []
    valid_windows = []


    for start in range(0, len(candidate_windows), batch_size):
        batch_windows = candidate_windows[start:start + batch_size]

        all_images = []
        owner = []

        for bi, window in enumerate(batch_windows):
            imgs, _ = sample_clip_frames_for_scoring(
                video_path,
                window,
                frames_per_candidate=frames_per_candidate,
            )

            all_images.extend(imgs)
            owner.extend([bi] * len(imgs))

        image_tensor = torch.stack(
            [preprocess(img) for img in all_images]
        ).to(device)

        image_feat = model.encode_image(image_tensor)

        image_feat = image_feat / image_feat.norm(
            dim=-1,
            keepdim=True,
        )

        per_window = {}

        for feat, bi in zip(image_feat, owner):
            per_window.setdefault(bi, []).append(feat)

        for bi in range(len(batch_windows)):
            if bi not in per_window:
                continue

            feats = torch.stack(per_window[bi])

            mean_feat = feats.mean(0)
            mean_feat = mean_feat / mean_feat.norm(p=2)

            all_window_features.append(mean_feat.cpu())
            valid_windows.append(batch_windows[bi])

    if not all_window_features:
        raise RuntimeError(
            f"No CLIP features generated for {video_path}"
        )

    all_window_features = torch.stack(all_window_features)

    return {
        "windows": valid_windows,
        "features": all_window_features,
    }


@torch.no_grad()
def clip_topk_windows(
    video_path,
    query,
    k,
    window_len_s=8.0,
    stride_s=4.0,
    frames_per_candidate=1,
):

    cached = get_window_embeddings(
        video_path=video_path,
        window_len_s=window_len_s,
        stride_s=stride_s,
        frames_per_candidate=frames_per_candidate,
    )

    candidate_windows = cached["windows"]
    image_features = cached["features"]

    model, _, device = get_clip_model()

    text_tokens = clip.tokenize(
        [query],
        truncate=True,
    ).to(device)

    text_feat = model.encode_text(text_tokens)

    text_feat = text_feat / text_feat.norm(
        dim=-1,
        keepdim=True,
    )

    scores = (
        image_features.to(device)
        @ text_feat.T
    ).squeeze(-1)

    scores = scores.cpu().numpy()

    window_scores = []

    for window, score in zip(candidate_windows, scores):
        window_scores.append(
            {
                "window": window,
                "score": float(score),
            }
        )

    window_scores.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    top = window_scores[:k]

    return sorted(
        top,
        key=lambda x: x["window"][0],
    )

def sample_frames_from_windows(video_path, windows, frames_per_window, max_side=768, jpeg_quality=85):
    vr = get_vr(video_path)
    fps = float(vr.get_avg_fps())
    n = len(vr)

    all_indices = []
    all_timestamps = []

    for w in windows:
        s, e = w

        if frames_per_window == 1:
            times = [(s + e) / 2.0]
        else:
            times = np.linspace(s, e, frames_per_window).tolist()

        for t in times:
            idx = int(round(t * fps))
            idx = max(0, min(n - 1, idx))
            all_indices.append(idx)
            all_timestamps.append(idx / fps)

    # De-duplicate while preserving order.
    unique_idxs = []
    unique_timestamps = []
    seen = set()

    for idx, ts in zip(all_indices, all_timestamps):
        if idx not in seen:
            unique_idxs.append(idx)
            unique_timestamps.append(ts)
            seen.add(idx)

    batch = vr.get_batch(unique_idxs).asnumpy()

    images = []
    for arr in batch:
        img = Image.fromarray(arr).convert("RGB")
        images.append(
            pil_to_data_url(
                img,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
            )
        )

    return {
        "video_path": video_path,
        "timestamps": unique_timestamps,
        "frame_indices": unique_idxs,
        "images": images,
    }

def build_clip_query(item):
    choices_text = format_choices(item["choices"])
    return f"{item['question']}\n{choices_text}"


def run_clip_oneshot(item, config):
    query = build_clip_query(item)

    top_windows = clip_topk_windows(
        video_path=item["video"],
        query=query,
        k=config["clip_topk"],
        window_len_s=config.get("window_len_s", 8.0),
        stride_s=config.get("stride_s", 4.0),
        frames_per_candidate=1,
    )

    selected_windows = [x["window"] for x in top_windows]

    frames = sample_frames_from_windows(
        video_path=item["video"],
        windows=selected_windows,
        frames_per_window=config["frames_per_window"],
    )

    evidence_hint = "\n".join(
        f"Selected window {i}: {w[0]:.1f}-{w[1]:.1f}s, CLIP score={top_windows[i]['score']:.4f}"
        for i, w in enumerate(selected_windows)
    )

    prompt = build_answer_prompt(
        item["question"],
        item["choices"],
        extra_context=f"These frames were selected from the most query-relevant video windows:\n{evidence_hint}",
    )

    response = call_vlm_answer(frames, prompt)

    return {
        "prediction_label": parse_mcq_label(response, num_choices=len(item["choices"])),
        "prediction_text": response,
        "num_vlm_calls": 1,
        "evidence": {
            "clip_top_windows": top_windows,
            "selected_frame_indices": frames["frame_indices"],
            "selected_timestamps": frames["timestamps"],
        },
    }


def run_oneshot(item, config):
    frames = sample_uniform_frames(
        video_path=item["video"],
        num_frames=config["total_frames"],
        start_s=0.0,
        end_s=float(item["duration_s"]),
    )

    prompt = build_answer_prompt(item["question"], item["choices"])
    response = call_vlm_answer(frames, prompt)

    return {
    "prediction_label": parse_mcq_label(response, num_choices=len(item["choices"])),
    "prediction_text": response,
    "num_vlm_calls": 1,
    "evidence": None,
        }


def run_global_summary(item, config):
    frames = sample_uniform_frames(
        video_path=item["video"],
        num_frames=config["total_frames"],
        start_s=0.0,
        end_s=float(item["duration_s"]),
    )

    summary_prompt = build_global_summary_prompt(
        item,
        query_conditioned=config["query_conditioned"],
    )

    summary = call_vlm_text(frames, summary_prompt)

    answer_prompt = build_answer_prompt(
        item["question"],
        item["choices"],
        extra_context=summary,
    )

    response = call_llm_answer(answer_prompt)

    if response is None:
        print("NONE RESPONSE")
        print("QID:", item["qid"])
        print("CONFIG:", config["name"])

    parsed = parse_mcq_label(
        response,
        num_choices=len(item["choices"])
        )

    if parsed is None:
        print("\n=== FAILED PARSE ===")
        print("QID:", item["qid"])
        print("CONFIG:", config["name"])
        print("RAW RESPONSE:", repr(response))

    return {
    "prediction_label": parsed,
    "prediction_text": response,
    "num_vlm_calls": 1,
    "evidence": {"global_summary": summary},
    }


def run_map_summary(item, config):
    duration_s = float(item["duration_s"])
    windows = make_uniform_windows(duration_s, config["num_windows"])

    window_summaries = []

    for widx, (start_s, end_s) in enumerate(windows):
        frames = sample_uniform_frames(
            video_path=item["video"],
            num_frames=config["frames_per_window"],
            start_s=start_s,
            end_s=end_s,
        )

        prompt = build_window_summary_prompt(
            item,
            start_s=start_s,
            end_s=end_s,
            query_conditioned=config["query_conditioned"],
        )

        summary = call_vlm_text(frames, prompt)

        window_summaries.append({
            "window_idx": widx,
            "start_s": start_s,
            "end_s": end_s,
            "summary": summary,
        })

    evidence_text = "\n\n".join(
        f"[Window {w['window_idx']} | {w['start_s']:.1f}-{w['end_s']:.1f}s]\n{w['summary']}"
        for w in window_summaries
    )

    answer_prompt = build_answer_prompt(
        item["question"],
        item["choices"],
        extra_context=evidence_text,
    )

    response = call_llm_answer(answer_prompt)


    return {
    "prediction_label": parse_mcq_label(response, num_choices=len(item["choices"])),
    "prediction_text": response,
    "num_vlm_calls": config["num_windows"],
        "evidence": {"window_summaries": window_summaries},
    }

def run_single_config(item, config):
    if config["method"] == "oneshot":
        return run_oneshot(item, config)

    if config["method"] == "clip_oneshot":
        return run_clip_oneshot(item, config)

    if config["method"] == "global_summary":
        return run_global_summary(item, config)

    if config["method"] == "map_summary":
        return run_map_summary(item, config)

    raise ValueError(f"Unknown method: {config['method']}")


def run_experiment(dataset, output, max_examples=None, resume=True):
    examples = load_jsonl(dataset)

    if max_examples is not None:
        examples = examples[:max_examples]

    done = set()
    if resume and Path(output).exists():
        for r in load_jsonl(output):
            done.add((r["qid"], r["config_name"]))

    total = len(examples) * len(EXPERIMENT_CONFIGS)
    run_idx = 0

    for i, item in enumerate(examples):
        print(f"\nExample {i+1}/{len(examples)} qid={item['qid']}")

        for config in EXPERIMENT_CONFIGS:
            run_idx += 1

            key = (item["qid"], config["name"])
            if key in done:
                print(f"  [{run_idx}/{total}] skip {config['name']}")
                continue

            print(f"  [{run_idx}/{total}] run {config['name']}")

            start = time.time()

            try:
                pred = run_single_config(item, config)
                error = None
            except Exception as e:
                pred = {
                    "prediction_label": None,
                    "prediction_text": None,
                    "num_vlm_calls": None,
                    "evidence": None,
                }
                error = str(e)

            latency_s = time.time() - start
            prediction_label = pred["prediction_label"]

            row = {
                "qid": item["qid"],
                "video_id": item.get("video_id"),
                "video": item.get("video"),
                "question": item["question"],
                "choices": item["choices"],
                "answer_idx": item["answer_idx"],
                "answer_label": item["answer_label"],
                "answer": item["answer"],
                "question_category": item.get("question_category"),
                "topic_category": item.get("topic_category"),
                "vimio_profile": item.get("vimio_profile"),
                "duration_s": item.get("duration_s"),
                "duration_bucket": item.get("duration_bucket"),
                "lvb_duration_bucket": item.get("lvb_duration_bucket"),

                "config_name": config["name"],
                "method": config["method"],
                "total_frames": config["total_frames"],
                "num_windows": config["num_windows"],
                "frames_per_window": config["frames_per_window"],
                "query_conditioned": config["query_conditioned"],

                "prediction_label": prediction_label,
                "prediction_text": pred["prediction_text"],
                "correct": prediction_label == item["answer_label"],
                "latency_s": latency_s,
                "num_vlm_calls": pred["num_vlm_calls"],
                "evidence": pred["evidence"],
                "error": error,
            }

            append_jsonl(row, output)

            print(
                f"    pred={row['prediction_label']} "
                f"gold={row['answer_label']} "
                f"correct={row['correct']} "
                f"latency={row['latency_s']:.2f}s"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        output=args.output,
        max_examples=args.max_examples,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()