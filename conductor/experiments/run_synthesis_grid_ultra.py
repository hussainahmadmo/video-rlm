import argparse
import base64
import io
import json
import time
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import torch.nn.functional as F
import torch
import clip

from openai import OpenAI

import os

import decord
import sys

print("DECORD FILE:", decord.__file__)
print("PYTHON:", sys.executable)
print("CWD:", __file__)

CLIP_DEVICE = os.environ.get("CLIP_DEVICE", "cuda:0")

VLM_BASE_URL = os.environ.get(
    "VLM_BASE_URL",
    "http://localhost:9000/v1"
)
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LATENCY_STATS = {}
client = OpenAI(
    base_url=VLM_BASE_URL,
    api_key="EMPTY",
)

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_VR_CACHE = {}



from decord import VideoReader, cpu
from decord import gpu
from decord import bridge
bridge.set_bridge("torch")

class DecordVideo:

    def __init__(self, path):
        self.path = path

        self.vr = VideoReader(
            path,
            ctx=cpu(0)
            )
        
        print("CPU decode enabled")

        self.nframes = len(self.vr)

        self.fps = float(
            self.vr.get_avg_fps()
        )

    def get_avg_fps(self):
        return self.fps

    def __len__(self):
        return self.nframes

    def get_batch(self, indices):
        return self.vr.get_batch(indices)

def get_vr(video_path):
    if video_path not in _VR_CACHE:

        print(f"[OPEN VIDEO] {video_path}")

        _VR_CACHE[video_path] = DecordVideo(video_path)

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

    batch = vr.get_batch(unique_idxs)
    print(type(batch))
    print(batch.device)

    images = []
    for arr in batch:
        img = Image.fromarray(arr.cpu().numpy()).convert("RGB")
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
    t0 = time.time()
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

    t_vlm = time.time()

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

    LATENCY_STATS["vlm_call_s"] = time.time() - t0

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



def make_sliding_windows(duration_s, window_len_s=8.0, stride_s=4.0):
    #sliding windows 
    windows = []

    s = 0.0

    while s + window_len_s <= duration_s:
        windows.append([
            float(s),
            float(s + window_len_s),
        ])
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
    batch = vr.get_batch(idxs)
    images = [
        Image.fromarray(arr.cpu().numpy()).convert("RGB")
        for arr in batch
    ]

    return images, idxs


@torch.no_grad()
def clip_topk_windows(
    video_path,
    query,
    k,
    window_len_s=8.0,
    scan_fps=1.0,
):
    t0 = time.time()
    scan = build_frame_scan_embeddings(
        video_path,
        scan_fps=scan_fps,
        window_len_s=window_len_s,
    )

    model, _, device = get_clip_model()

    text_tokens = clip.tokenize(
        [query],
        truncate=True,
    ).to(device)

    torch.cuda.synchronize()

    t_text = time.time()

    text_feat = model.encode_text(text_tokens)

    torch.cuda.synchronize()

    LATENCY_STATS["clip_text_s"] = (
        LATENCY_STATS.get("clip_text_s", 0)
        + (time.time() - t_text)
    )

    text_feat = text_feat / text_feat.norm(
        dim=-1,
        keepdim=True,
    )

    scores = (
        scan["features"]
        @ text_feat.T
    ).squeeze(-1)

    scores = scores.cpu().numpy()
    raw_top = np.argsort(scores)[::-1][:k]
    print("\n[BASELINE TOPK]")

    for idx in raw_top:
        w = scan["windows"][idx]
        print(
            f"window={w[0]:.1f}-{w[1]:.1f}s "
            f"score={scores[idx]:.4f}"
        )

    top_idx = np.argsort(scores)[::-1]

    selected = []
    selected_windows = []

    for idx in top_idx:
        w = scan["windows"][idx]

        overlap = False

        for sw in selected_windows:
            inter = max(0, min(w[1], sw[1]) - max(w[0], sw[0]))
            union = max(w[1], sw[1]) - min(w[0], sw[0])

            if inter / union > 0.5:
                overlap = True
                break

        if not overlap:
            selected.append(idx)
            selected_windows.append(w)

        if len(selected) == k:
            break

    top_idx = selected

    window_scores = []

    for idx in top_idx:

        window_scores.append({
            "window": scan["windows"][idx],
            "score": float(scores[idx]),
        })

    LATENCY_STATS["clip_total_s"] = time.time() - t0

    candidate_windows_examined = len(scan["windows"])

    return {
        "top_windows": sorted(
            window_scores,
            key=lambda x: x["score"],
            reverse=True,
        ),
        "candidate_windows_examined":
            candidate_windows_examined,
    }

def sample_frames_from_windows(
    video_path,
    windows,
    frame_allocations,
    max_side=768,
    jpeg_quality=85,
):
    t0 = time.time()

    vr = get_vr(video_path)
    fps = float(vr.get_avg_fps())
    n = len(vr)

    all_indices = []
    all_timestamps = []

    for (s, e), frames_per_window in zip(
        windows,
        frame_allocations,
    ):

        if frames_per_window <= 0:
            continue

        if frames_per_window == 1:
            times = [(s + e) / 2.0]
        else:
            times = np.linspace(
                s,
                e,
                frames_per_window,
            ).tolist()

        for t in times:
            idx = int(round(t * fps))
            idx = max(0, min(n - 1, idx))

            all_indices.append(idx)
            all_timestamps.append(idx / fps)

    # deduplicate while preserving order
    unique_idxs = []
    unique_timestamps = []
    seen = set()

    for idx, ts in zip(
        all_indices,
        all_timestamps,
    ):
        if idx not in seen:
            unique_idxs.append(idx)
            unique_timestamps.append(ts)
            seen.add(idx)

    torch.cuda.synchronize()

    t_decode = time.time()

    batch = vr.get_batch(unique_idxs)

    torch.cuda.synchronize()

    LATENCY_STATS["answer_decode_s"] = (
        LATENCY_STATS.get("answer_decode_s", 0)
        + (time.time() - t_decode)
    )

    torch.cuda.synchronize()

    t_copy = time.time()

    cpu_batch = batch.cpu().numpy()

    torch.cuda.synchronize()

    LATENCY_STATS["gpu_cpu_copy_s"] = (
        LATENCY_STATS.get("gpu_cpu_copy_s", 0)
        + (time.time() - t_copy)
    )

    images = []

    for arr in cpu_batch:

        t_pil = time.time()

        img = Image.fromarray(arr).convert("RGB")

        LATENCY_STATS["pil_s"] = (
            LATENCY_STATS.get("pil_s", 0)
            + (time.time() - t_pil)
        )

        t_jpeg = time.time()

        images.append(
            pil_to_data_url(
                img,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
            )
        )

        LATENCY_STATS["jpeg_base64_s"] = (
            LATENCY_STATS.get("jpeg_base64_s", 0)
            + (time.time() - t_jpeg)
        )

    LATENCY_STATS["frame_extract_s"] = (
        time.time() - t0
    )

    return {
        "video_path": video_path,
        "timestamps": unique_timestamps,
        "frame_indices": unique_idxs,
        "images": images,
    }


def build_clip_query(item, use_choices=True):

    if use_choices:
        choices_text = format_choices(item["choices"])
        return f"{item['question']}\n{choices_text}"

    return item["question"]

def run_clip_oneshot(item, config):
    LATENCY_STATS.clear()
    query = build_clip_query(
        item,
        use_choices=config.get(
            "use_choices_in_query",
            True,
        ),
    )

    retrieval = clip_topk_windows(
        video_path=item["video"],
        query=query,
        k=config["clip_topk"],
        window_len_s=config["window_len_s"],
        scan_fps=config["scan_fps"],
    )

    top_windows = retrieval["top_windows"]

    selected_windows = [x["window"] for x in top_windows]

    print(
        f"[RETRIEVAL] "
        f"Candidate windows={retrieval['candidate_windows_examined']} "
        f"Selected={len(top_windows)}"
    )

    if len(selected_windows) == 0:
        raise RuntimeError("No windows selected")
    
    TOTAL_FRAMES = config["vlm_budget"]
    base = TOTAL_FRAMES // len(selected_windows)
    extra = TOTAL_FRAMES % len(selected_windows)
    frame_allocations = [
        base + (1 if i < extra else 0)
        for i in range(len(selected_windows))
    ]
    actual_budget = sum(frame_allocations)

    print(
        f"[BUDGET] "
        f"windows={len(selected_windows)} "
        f"actual_budget={actual_budget}"
    )

    frames = sample_frames_from_windows(
        video_path=item["video"],
        windows=selected_windows,
        frame_allocations=frame_allocations,
    )

    if len(frames["images"]) > 64:
        raise RuntimeError(
            f"Too many images: {len(frames['images'])}"
        )

    print(
        f"[VLM INPUT] "
        f"actual_frames={len(frames['frame_indices'])}"
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

    print(
        f"[LATENCY] "
        f"clip={LATENCY_STATS.get('clip_total_s',0):.2f}s "
        f"frames={LATENCY_STATS.get('frame_extract_s',0):.2f}s "
        f"vlm={LATENCY_STATS.get('vlm_call_s',0):.2f}s",
        flush=True,
    )

    print("\n===== LATENCY BREAKDOWN =====")

    for k, v in sorted(LATENCY_STATS.items()):
        print(f"{k:25s}: {v:.3f}s")

    print("============================\n")

    return {
        "prediction_label": parse_mcq_label(
            response,
            num_choices=len(item["choices"])
        ),
        "prediction_text": response,
        "num_vlm_calls": 1,

        "retrieval_effort": {
            "candidate_windows":
                retrieval["candidate_windows_examined"],

            "selected_windows":
                len(top_windows),

            "selected_frames":
                len(frames["frame_indices"]),
        },

        "evidence": {
            "clip_top_windows": top_windows,
            "selected_frame_indices":
                frames["frame_indices"],
            "selected_timestamps":
                frames["timestamps"],
        },
    }

@torch.no_grad()
def build_frame_scan_embeddings(
    video_path,
    scan_fps=2.0,
    window_len_s=8.0,
    ):

    model, _, device = get_clip_model()


    mean = torch.tensor(
        [0.48145466,0.4578275,0.40821073],
        device=device
    ).view(1,3,1,1)

    std = torch.tensor(
        [0.26862954,0.26130258,0.27577711],
        device=device
    ).view(1,3,1,1)

    vr = get_vr(video_path)

    fps = float(vr.get_avg_fps())
    nframes = len(vr)


    #step is computed from the video fps, so it automatically adapts such
    #so the fps is the property of the video. 
    #scan_fps is our desired sampling rate.
    #thats what makes our scan rate time based, regardless of the fps. e.g sample every 1 second
    step = max(1, int(round(fps / scan_fps)))

    scan_idxs = list(range(0, nframes, step))

    print(
        f"[SCAN] {len(scan_idxs)} frames "
        f"({scan_fps} fps)"
    )

    decode_time = 0.0
    encode_time = 0.0

    all_feats = []
    all_times = []
    resize_time = 0.0

    for start in range(0, len(scan_idxs), 256):
        chunk = scan_idxs[start:start+256]

        torch.cuda.synchronize()

        t = time.time()

        batch = vr.get_batch(chunk)
        t0 = time.time()
        torch.cuda.synchronize()



        print(
            "[CHUNK]",
            chunk[0],
            chunk[-1],
            len(chunk)
        )
        print(
            "[BATCH]",
            batch.shape,
            batch.dtype,
            batch.device,
        )
        torch.cuda.synchronize()
        decode_time += time.time() - t

        t_resize = time.time()
        batch = batch.to(device, non_blocking=True)

        #change shape from [N, H, W, C] to [N, C, H, W] - pytorch models pref [batch, channel, height, width]
        batch = batch.permute(0,3,1,2)
        batch = batch.float() / 255.0


        #shrink every image from 1080 * 1920 to 240*240 so CLIP can process.
        batch = F.interpolate(
            batch,
            size=(224,224),
            mode="bicubic",
            align_corners=False,
        )
        batch = (batch - mean) / std
        torch.cuda.synchronize()
        resize_time += time.time() - t_resize
        t = time.time()

        print(
            "[ENCODE BATCH]",
            batch.shape
        )
        feats = model.encode_image(batch)

        torch.cuda.synchronize()

        encode_time += time.time() - t

        feats = feats / feats.norm(
            dim=-1,
            keepdim=True,
        )

        all_feats.append(feats)


        all_times.extend(
            idx / fps
            for idx in chunk
        )

    feats = torch.cat(all_feats, dim=0)

    LATENCY_STATS["clip_decode_s"] = decode_time
    LATENCY_STATS["clip_encode_s"] = encode_time

    duration_s = nframes / fps

    windows = []

    for ts in all_times:

        start = max(
            0.0,
            ts - window_len_s / 2
        )

        end = min(
            duration_s,
            ts + window_len_s / 2
        )

        windows.append([start, end])

    print(
        f"[PROFILE] "
        f"decode={decode_time:.2f}s "
        f"resize={resize_time:.2f}s "
        f"encode={encode_time:.2f}s"
    )
    return {
        "timestamps": all_times,
        "features": feats,
        "windows": windows,
    }



def run_oneshot(item, config):

    duration_s = item.get("duration_s")

    if duration_s is None:
        duration_s, _, _ = duration_fps_nframes(item["video"])

    frames = sample_uniform_frames(
        video_path=item["video"],
        num_frames=config["total_frames"],
        start_s=0.0,
        end_s=float(duration_s),
    )


    prompt = build_answer_prompt(item["question"], item["choices"])
    response = call_vlm_answer(frames, prompt)

    return {
        "prediction_label": parse_mcq_label(response, num_choices=len(item["choices"])),
        "prediction_text": response,
        "num_vlm_calls": 1,
        "evidence": None,
        "retrieval_effort": {
            "candidate_windows": 1,
            "selected_frames": len(frames["frame_indices"]),
                },
        }


def run_global_summary(item, config):
    duration_s = item.get("duration_s")

    if duration_s is None:
        duration_s, _, _ = duration_fps_nframes(item["video"])

    end_s=float(duration_s)
    frames = sample_uniform_frames(
        video_path=item["video"],
        num_frames=config["total_frames"],
        start_s=0.0,
        end_s=float(duration_s),
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
        "retrieval_effort": {
            "candidate_windows": 1,
            "selected_windows": 1,
            "selected_frames": len(frames["frame_indices"]),
        },
    }


def run_map_summary(item, config):
    duration_s = item.get("duration_s")

    if duration_s is None:
        duration_s, _, _ = duration_fps_nframes(item["video"])

    duration_s = float(duration_s)
    windows = make_uniform_windows(duration_s, config["num_windows"])

    window_summaries = []
    actual_frames = 0

    for widx, (start_s, end_s) in enumerate(windows):
        frames = sample_uniform_frames(
            video_path=item["video"],
            num_frames=config["frames_per_window"],
            start_s=start_s,
            end_s=end_s,
        )

        actual_frames += len(frames["frame_indices"])

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
        "retrieval_effort": {
            "candidate_windows": config["num_windows"],
            "selected_windows": config["num_windows"],
            "selected_frames": actual_frames,
            }
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
                import traceback
                traceback.print_exc()

                pred = {
                    "prediction_label": None,
                    "prediction_text": None,
                    "num_vlm_calls": None,
                    "evidence": None,
                    "retrieval_effort": None,
                }

                error = repr(e)

                print("\nERROR =", error)

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
                "stride_s": config.get("stride_s"),
                "window_len_s": config.get("window_len_s"),
                "clip_topk": config.get("clip_topk"),
                "latency_breakdown": dict(LATENCY_STATS),
                "scan_fps": config.get("scan_fps"),
                "config_name": config["name"],
                "method": config["method"],
                "num_windows": config.get("num_windows"),
                "query_conditioned": config.get("query_conditioned"),
                "vlm_budget": config.get("vlm_budget"),
                "prediction_label": prediction_label,
                "prediction_text": pred["prediction_text"],
                "correct": prediction_label == item["answer_label"],
                "latency_s": latency_s,
                "num_vlm_calls": pred["num_vlm_calls"],
                "evidence": pred["evidence"],
                "error": error,
                "retrieval_effort":
                    pred.get("retrieval_effort"),
            }

            if error is not None:
                print("ERROR:", error)

            append_jsonl(row, output)

            print(
                f"    pred={row['prediction_label']} "
                f"gold={row['answer_label']} "
                f"correct={row['correct']} "
                f"latency={row['latency_s']:.2f}s"
            )

def load_config_file(path):

    with open(path) as f:
        return json.load(f)
    

def main():

    global EXPERIMENT_CONFIGS

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--config_file", default=None)

    args = parser.parse_args()

    if args.config_file:
        EXPERIMENT_CONFIGS = load_config_file(
            args.config_file
        )
    else:
        raise ValueError(
            "Must provide --config_file"
        )

    print(
        "Loaded",
        len(EXPERIMENT_CONFIGS),
        "configs"
    )

    run_experiment(
        dataset=args.dataset,
        output=args.output,
        max_examples=args.max_examples,
        resume=not args.no_resume,
    )



if __name__ == "__main__":
    main()