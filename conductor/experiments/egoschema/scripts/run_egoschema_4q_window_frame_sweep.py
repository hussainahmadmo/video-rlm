import argparse
import base64
import io
import json
import time
from pathlib import Path

import torch
import clip
import decord
import numpy as np
from PIL import Image
from openai import OpenAI

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


WINDOWS = [1, 3, 5, 10, 20]
FRAMES_PER_WINDOW = [1, 2, 4, 8, 16]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_video_duration(video_path):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps())
    n = len(vr)
    return n / fps, fps, n


def make_uniform_windows(duration_s, k, window_len_s=8.0, margin_s=5.0):
    """
    Match your old behavior approximately:
    k=1: one early window
    k=3: early/middle/late
    k=5/10/20: spread through video
    """
    if k == 1:
        centers = [margin_s + window_len_s / 2]
    else:
        start_center = margin_s + window_len_s / 2
        end_center = max(start_center, duration_s - margin_s - window_len_s / 2)
        centers = np.linspace(start_center, end_center, k).tolist()

    windows = []
    for c in centers:
        s = max(0.0, c - window_len_s / 2)
        e = min(duration_s, c + window_len_s / 2)
        windows.append([float(s), float(e)])
    return windows

def get_clip_model(device="cuda"):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    if _CLIP_MODEL is None:
        _CLIP_DEVICE = device if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP ViT-B/32 on {_CLIP_DEVICE}", flush=True)
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=_CLIP_DEVICE)
        _CLIP_MODEL.eval()

    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def make_sliding_windows(duration_s, window_len_s=8.0, stride_s=8.0, margin_s=5.0):
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


def sample_clip_frames(video_path, window, frames_per_candidate=1):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
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
def clip_topk_windows(
    video_path,
    query,
    k,
    window_len_s=8.0,
    stride_s=8.0,
    frames_per_candidate=1,
    batch_size=64,
):
    duration_s, _, _ = get_video_duration(video_path)
    candidate_windows = make_sliding_windows(
        duration_s,
        window_len_s=window_len_s,
        stride_s=stride_s,
    )

    model, preprocess, device = get_clip_model()

    text_tokens = clip.tokenize([query], truncate=True).to(device)
    text_feat = model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    window_scores = []

    for start in range(0, len(candidate_windows), batch_size):
        batch_windows = candidate_windows[start:start + batch_size]

        all_images = []
        owner = []

        for bi, window in enumerate(batch_windows):
            imgs, _ = sample_clip_frames(
                video_path,
                window,
                frames_per_candidate=frames_per_candidate,
            )
            all_images.extend(imgs)
            owner.extend([bi] * len(imgs))

        image_tensor = torch.stack([preprocess(img) for img in all_images]).to(device)
        image_feat = model.encode_image(image_tensor)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        sims = (image_feat @ text_feat.T).squeeze(-1).detach().cpu().numpy()

        per_window_scores = [[] for _ in batch_windows]
        for score, bi in zip(sims, owner):
            per_window_scores[bi].append(float(score))

        for scores in per_window_scores:
            window_scores.append(max(scores) if scores else -1e9)

    top_indices = np.argsort(window_scores)[::-1][:k]

    selected = [
        (candidate_windows[i], float(window_scores[i]))
        for i in top_indices
    ]

    selected = sorted(selected, key=lambda x: x[0][0])

    top_windows = [w for w, _ in selected]
    top_scores = [s for _, s in selected]

    return top_windows, top_scores

def sample_frames(video_path, windows, frames_per_window, jpeg_quality=85, max_side=768):
    t0 = time.perf_counter()

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps())
    n = len(vr)

    frame_indices = []
    for s, e in windows:
        if frames_per_window == 1:
            times = [(s + e) / 2.0]
        else:
            times = np.linspace(s, e, frames_per_window).tolist()

        for ts in times:
            idx = int(round(ts * fps))
            idx = max(0, min(n - 1, idx))
            frame_indices.append(idx)

    # Deduplicate but preserve order
    seen = set()
    uniq_indices = []
    for idx in frame_indices:
        if idx not in seen:
            uniq_indices.append(idx)
            seen.add(idx)

    batch = vr.get_batch(uniq_indices).asnumpy()

    images_b64 = []
    for arr in batch:
        img = Image.fromarray(arr)

        # Resize to reduce token/latency variance
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        images_b64.append(b64)

    decode_latency = time.perf_counter() - t0
    return images_b64, uniq_indices, decode_latency


def call_vlm(client, model, question, images_b64, max_tokens=384):
    content = [
        {
            "type": "text",
            "text": (
                "Answer the video question using the sampled frames. "
                "Be specific about the evidence and actions in the video.\n\n"
                f"Question: {question}"
            ),
        }
    ]

    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    latency = time.perf_counter() - t0

    text = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason

    return text, latency, finish_reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_output_token_budget", type=int, default=None)
    ap.add_argument("--summary_budget_fraction", type=float, default=0.75)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--window_len_s", type=float, default=8.0)
    ap.add_argument("--jpeg_quality", type=int, default=85)
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--proposal_strategy", choices=["uniform", "clip"], default="uniform")
    ap.add_argument("--clip_stride_s", type=float, default=8.0)
    ap.add_argument("--clip_frames_per_candidate", type=int, default=1)
    ap.add_argument("--clip_batch_size", type=int, default=64)
    args = ap.parse_args()

    examples = load_jsonl(args.input)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as out:
        total_jobs = len(examples) * len(WINDOWS) * len(FRAMES_PER_WINDOW)
        job_id = 0

        for ex in examples:
            video_path = ex["video"]
            duration_s, fps, nframes = get_video_duration(video_path)

            for k in WINDOWS:
                if args.total_output_token_budget is not None:
                    answer_max_tokens_effective = args.total_output_token_budget
                else:
                    answer_max_tokens_effective = args.max_tokens

                proposal_latency_s = 0.0
                proposal_scores = None
                windows = []

                try:
                    if args.proposal_strategy == "uniform":
                        windows = make_uniform_windows(duration_s, k, args.window_len_s)

                    elif args.proposal_strategy == "clip":
                        pt0 = time.perf_counter()
                        windows, proposal_scores = clip_topk_windows(
                            video_path=video_path,
                            query=ex["question"],
                            k=k,
                            window_len_s=args.window_len_s,
                            stride_s=args.clip_stride_s,
                            frames_per_candidate=args.clip_frames_per_candidate,
                            batch_size=args.clip_batch_size,
                        )
                        proposal_latency_s = time.perf_counter() - pt0

                    else:
                        raise ValueError(args.proposal_strategy)

                except Exception as e:
                    # If proposal itself fails, write failed rows for all frame settings.
                    for f in FRAMES_PER_WINDOW:
                        job_id += 1
                        row = {
                            "video_id": ex.get("video_id"),
                            "qid": ex.get("qid"),
                            "type": "egoschema",
                            "video": video_path,
                            "question": ex.get("question", ""),
                            "answer": ex.get("answer", ""),
                            "action_topk": k,
                            "frames_per_window": f,
                            "num_images": 0,
                            "duration_s": duration_s,
                            "fps": fps,
                            "video_num_frames": nframes,
                            "windows": windows,
                            "decode_latency_s": 0.0,
                            "vlm_latency_s": 0.0,
                            "total_latency_s": 0.0,
                            "pred_text": "",
                            "ok": False,
                            "error": repr(e),
                            "technique": f"{args.proposal_strategy}_one_shot_total_budget" if args.total_output_token_budget is not None else f"{args.proposal_strategy}_one_shot",
                            "total_output_token_budget": args.total_output_token_budget,
                            "summary_budget_fraction": args.summary_budget_fraction,
                            "intermediate_token_budget_total": 0,
                            "per_intermediate_max_tokens_effective": 0,
                            "answer_max_tokens_effective": answer_max_tokens_effective,
                            "num_intermediate_units": 0,
                            "actual_total_output_token_budget_cap": answer_max_tokens_effective,
                            "proposal_strategy": args.proposal_strategy,
                            "proposal_scores": proposal_scores,
                            "clip_stride_s": args.clip_stride_s,
                            "clip_frames_per_candidate": args.clip_frames_per_candidate,
                            "clip_batch_size": args.clip_batch_size,
                            "proposal_latency_s": proposal_latency_s,
                            "total_latency_with_proposal_s": proposal_latency_s,
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                    continue

                for f in FRAMES_PER_WINDOW:
                    job_id += 1
                    print(f"[{job_id}/{total_jobs}] qid={ex['qid']} k={k} f={f}", flush=True)

                    if args.total_output_token_budget is not None:
                        answer_max_tokens_effective = args.total_output_token_budget
                    else:
                        answer_max_tokens_effective = args.max_tokens

                    try:
                        images_b64, frame_indices, decode_latency = sample_frames(
                            video_path,
                            windows,
                            f,
                            jpeg_quality=args.jpeg_quality,
                            max_side=args.max_side,
                        )

                        pred_text, vlm_latency, finish_reason = call_vlm(
                            client,
                            args.model,
                            ex["question"],
                            images_b64,
                            max_tokens=answer_max_tokens_effective,
                        )

                        total_latency = decode_latency + vlm_latency

                        row = {
                            "video_id": ex["video_id"],
                            "qid": ex["qid"],
                            "type": "egoschema",
                            "video": video_path,
                            "question": ex["question"],
                            "answer": ex.get("answer", ""),
                            "action_topk": k,
                            "frames_per_window": f,
                            "num_images": len(images_b64),
                            "duration_s": duration_s,
                            "fps": fps,
                            "video_num_frames": nframes,
                            "windows": windows,
                            "frame_indices": frame_indices,
                            "decode_latency_s": decode_latency,
                            "vlm_latency_s": vlm_latency,
                            "total_latency_s": total_latency,
                            "pred_text": pred_text,
                            "final_finish_reason": finish_reason,
                            "ok": True,
                            "error": None,
                            "technique": f"{args.proposal_strategy}_one_shot_total_budget" if args.total_output_token_budget is not None else f"{args.proposal_strategy}_one_shot",
                            "total_output_token_budget": args.total_output_token_budget,
                            "summary_budget_fraction": args.summary_budget_fraction,
                            "intermediate_token_budget_total": 0,
                            "per_intermediate_max_tokens_effective": 0,
                            "answer_max_tokens_effective": answer_max_tokens_effective,
                            "num_intermediate_units": 0,
                            "actual_total_output_token_budget_cap": answer_max_tokens_effective,
                            "proposal_strategy": args.proposal_strategy,
                            "proposal_scores": proposal_scores,
                            "clip_stride_s": args.clip_stride_s,
                            "clip_frames_per_candidate": args.clip_frames_per_candidate,
                            "clip_batch_size": args.clip_batch_size,
                            "proposal_latency_s": proposal_latency_s,
                            "total_latency_with_proposal_s": proposal_latency_s + total_latency,
                        }

                    except Exception as e:
                        row = {
                            "video_id": ex.get("video_id"),
                            "qid": ex.get("qid"),
                            "type": "egoschema",
                            "video": video_path,
                            "question": ex.get("question", ""),
                            "answer": ex.get("answer", ""),
                            "action_topk": k,
                            "frames_per_window": f,
                            "num_images": 0,
                            "duration_s": duration_s,
                            "fps": fps,
                            "video_num_frames": nframes,
                            "windows": windows,
                            "decode_latency_s": 0.0,
                            "vlm_latency_s": 0.0,
                            "total_latency_s": 0.0,
                            "pred_text": "",
                            "ok": False,
                            "error": repr(e),
                            "technique": f"{args.proposal_strategy}_one_shot_total_budget" if args.total_output_token_budget is not None else f"{args.proposal_strategy}_one_shot",
                            "total_output_token_budget": args.total_output_token_budget,
                            "summary_budget_fraction": args.summary_budget_fraction,
                            "intermediate_token_budget_total": 0,
                            "per_intermediate_max_tokens_effective": 0,
                            "answer_max_tokens_effective": answer_max_tokens_effective,
                            "num_intermediate_units": 0,
                            "actual_total_output_token_budget_cap": answer_max_tokens_effective,
                            "proposal_strategy": args.proposal_strategy,
                            "proposal_scores": proposal_scores,
                            "clip_stride_s": args.clip_stride_s,
                            "clip_frames_per_candidate": args.clip_frames_per_candidate,
                            "clip_batch_size": args.clip_batch_size,
                            "proposal_latency_s": proposal_latency_s,
                            "total_latency_with_proposal_s": proposal_latency_s,
                        }

                    out.write(json.dumps(row) + "\n")
                    out.flush()
            
    
    
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
