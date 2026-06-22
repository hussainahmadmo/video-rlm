import torch
import clip
import argparse
import base64
import io
import json
import time
from pathlib import Path

import decord
import numpy as np
from PIL import Image
from openai import OpenAI

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None




ACTION_TOPKS = [10]
FRAMES_PER_WINDOW = [2]

def format_choices(choices):
    if not choices:
        return ""

    labels = ["A", "B", "C", "D", "E"]
    return "\n".join(
        f"{labels[i]}. {choice}"
        for i, choice in enumerate(choices)
    )


def get_clip_model(device="cuda"):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    if _CLIP_MODEL is None:
        _CLIP_DEVICE = device if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP ViT-B/32 on {_CLIP_DEVICE}", flush=True)
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=_CLIP_DEVICE)
        _CLIP_MODEL.eval()

    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def make_sliding_windows(duration_s, window_len_s=8.0, stride_s=4.0, margin_s=5.0):
    first_start = margin_s
    last_start = max(first_start, duration_s - margin_s - window_len_s)

    windows = []
    s = first_start
    while s <= last_start:
        windows.append([float(s), float(s + window_len_s)])
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
    stride_s=4.0,
    frames_per_candidate=1,
    batch_size=64,
):
    duration_s, _, _ = duration_fps_nframes(video_path)
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
            # max = a window is good if any sampled frame matches the query
            window_scores.append(max(scores) if scores else -1e9)

    top_indices = np.argsort(window_scores)[::-1][:k]

    selected = [
        (candidate_windows[i], float(window_scores[i]))
        for i in top_indices
    ]

    # Sort selected windows by time before sending to VLM.
    selected = sorted(selected, key=lambda x: x[0][0])

    top_windows = [w for w, _ in selected]
    top_scores = [s for _, s in selected]

    return top_windows, top_scores


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def duration_fps_nframes(video_path):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps())
    return len(vr) / fps, fps, len(vr)


def old_uniform_windows(duration_s, k, window_len_s=8.0, margin_s=5.0):
    first_start = margin_s
    last_start = max(first_start, duration_s - margin_s - window_len_s)

    if k == 1:
        starts = [first_start]
    else:
        starts = np.linspace(first_start, last_start, k).tolist()

    return [
        [float(s), float(min(duration_s, s + window_len_s))]
        for s in starts
    ]


def sample_window_frames(video_path, window, frames_per_window, max_side=768, jpeg_quality=85):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps())
    n = len(vr)

    s, e = window
    if frames_per_window == 1:
        times = [(s + e) / 2.0]
    else:
        times = np.linspace(s, e, frames_per_window).tolist()

    idxs = []
    for t in times:
        idx = int(round(t * fps))
        idx = max(0, min(n - 1, idx))
        idxs.append(idx)

    # dedupe preserving order
    seen = set()
    uniq = []
    for idx in idxs:
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)

    batch = vr.get_batch(uniq).asnumpy()

    data_urls = []
    for arr in batch:
        img = Image.fromarray(arr)
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_urls.append(f"data:image/jpeg;base64,{b64}")

    return data_urls, uniq


def call_vlm(client, model, content, max_tokens):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason
    return text, latency, finish_reason


def summarize_window(client, model, question, window_id, window, image_urls, max_tokens):
    content = [
        {
            "type": "text",
            "text": (
                "You are given frames from one short video window. "
                "Describe only the visible actions, objects, and evidence that may help answer the question. "
                "Be concrete and mention object names. Do not over-explain.\n\n"
                f"Question: {question}\n"
                f"Window {window_id}: {window[0]:.1f}s to {window[1]:.1f}s"
            ),
        }
    ]

    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    return call_vlm(client, model, content, max_tokens)


# def final_answer(client, model, question, window_summaries, max_tokens):
#     evidence = "\n".join(
#         [f"Window {i+1}: {s}" for i, s in enumerate(window_summaries)]
#     )

#     content = [
#         {
#             "type": "text",
#             "text": (
#                 "Answer the question using the window evidence below. "
#                 "Mention the concrete evidence from the windows. "
#                 "Be specific about actions, objects, and the overall goal.\n\n"
#                 f"Question: {question}\n\n"
#                 f"Window evidence:\n{evidence}\n\n"
#                 "Final answer:"
#             ),
#         }
#     ]

#     return call_vlm(client, model, content, max_tokens)

def final_answer(client, model, question, window_summaries, choices=None, max_tokens=384):
    evidence = "\n".join(
        [f"Window {i+1}: {s}" for i, s in enumerate(window_summaries)]
    )
    choice_text = format_choices(choices)

    content = [
        {
            "type": "text",
            "text": (
                "Answer the video multiple-choice question using the window evidence below. "
                "Choose exactly one option from A, B, C, D, or E. "
                "Start your response with exactly this format:\n"
                "Answer: <letter>\n"
                "Explanation: <one short sentence>\n\n"
                f"Question: {question}\n\n"
                f"Choices:\n{choice_text}\n\n"
                f"Window evidence:\n{evidence}\n\n"
                "Answer:"
            ),
        }
    ]

    return call_vlm(client, model, content, max_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--window_len_s", type=float, default=8.0)
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--jpeg_quality", type=int, default=85)
    ap.add_argument("--summary_max_tokens", type=int, default=96)
    ap.add_argument("--answer_max_tokens", type=int, default=384)
    ap.add_argument("--proposal_strategy", choices=["uniform", "clip"], default="uniform")
    ap.add_argument("--clip_stride_s", type=float, default=8.0)
    ap.add_argument("--clip_frames_per_candidate", type=int, default=1)
    ap.add_argument("--clip_batch_size", type=int, default=64)

    # Equal total-output-token-budget mode.
    # If set, map-summary splits this total budget between:
    #   1) all per-window summaries together
    #   2) final answer synthesis
    ap.add_argument("--total_output_token_budget", type=int, default=None)
    ap.add_argument("--summary_budget_fraction", type=float, default=0.75)

    args = ap.parse_args()

    if not (0.0 <= args.summary_budget_fraction <= 1.0):
        raise ValueError("--summary_budget_fraction must be between 0 and 1")

    examples = load_jsonl(args.input)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_jobs = len(examples) * len(ACTION_TOPKS) * len(FRAMES_PER_WINDOW)
    job_id = 0

    with open(args.output, "w") as out:
        for ex in examples:
            qid = ex["qid"]
            video_path = ex["video"]
            question = ex["question"]

            duration_s, fps, nframes = duration_fps_nframes(video_path)

            for k in ACTION_TOPKS:
                proposal_latency_s = 0.0
                proposal_scores = None

                if args.proposal_strategy == "uniform":
                    windows = old_uniform_windows(duration_s, k, args.window_len_s)

                elif args.proposal_strategy == "clip":
                    pt0 = time.perf_counter()
                    windows, proposal_scores = clip_topk_windows(
                        video_path=video_path,
                        query=question,
                        k=k,
                        window_len_s=args.window_len_s,
                        stride_s=args.clip_stride_s,
                        frames_per_candidate=args.clip_frames_per_candidate,
                        batch_size=args.clip_batch_size,
                    )
                    proposal_latency_s = time.perf_counter() - pt0

                else:
                    raise ValueError(args.proposal_strategy)

                for fpw in FRAMES_PER_WINDOW:
                    job_id += 1
                    print(f"[{job_id}/{total_jobs}] qid={qid} k={k} f={fpw}", flush=True)

                    decode_latency_s = 0.0
                    vlm_latency_s = 0.0
                    all_frame_indices = []
                    window_summaries = []
                    summary_finish_reasons = []
                    final_lat = 0.0

                    # Equal total-token-budget mode:
                    # map-summary has k intermediate units, one summary per selected window.
                    # We split the same total generated-token budget between:
                    #   summary_budget_fraction for all window summaries combined
                    #   remaining budget for the final answer
                    if args.total_output_token_budget is not None:
                        intermediate_token_budget_total = int(
                            args.total_output_token_budget * args.summary_budget_fraction
                        )
                        answer_max_tokens_effective = (
                            args.total_output_token_budget - intermediate_token_budget_total
                        )
                        per_window_summary_max_tokens_effective = max(
                            1, intermediate_token_budget_total // k
                        )
                        actual_total_output_token_budget_cap = (
                            k * per_window_summary_max_tokens_effective
                            + answer_max_tokens_effective
                        )
                        technique_name = f"{args.proposal_strategy}_map_summary_total_budget"
                    else:
                        intermediate_token_budget_total = k * args.summary_max_tokens
                        answer_max_tokens_effective = args.answer_max_tokens
                        per_window_summary_max_tokens_effective = args.summary_max_tokens
                        actual_total_output_token_budget_cap = (
                            k * per_window_summary_max_tokens_effective
                            + answer_max_tokens_effective
                        )
                        technique_name = f"{args.proposal_strategy}_map_summary"

                    try:

                        for wi, window in enumerate(windows):
                            dt0 = time.perf_counter()
                            image_urls, frame_indices = sample_window_frames(
                                video_path,
                                window,
                                fpw,
                                max_side=args.max_side,
                                jpeg_quality=args.jpeg_quality,
                            )
                            decode_latency_s += time.perf_counter() - dt0
                            all_frame_indices.extend(frame_indices)

                            summary, lat, fr = summarize_window(
                                client,
                                args.model,
                                question,
                                wi + 1,
                                window,
                                image_urls,
                                per_window_summary_max_tokens_effective,
                            )
                            vlm_latency_s += lat
                            window_summaries.append(summary)
                            summary_finish_reasons.append(fr)

                        pred_text, final_lat, final_fr = final_answer(
                            client,
                            args.model,
                            question,
                            window_summaries,
                            choices=ex.get("choices"),
                            max_tokens=answer_max_tokens_effective,
                        )
                        vlm_latency_s += final_lat

                        row = {
                            "video_id": qid,
                            "qid": qid,
                            "type": "egoschema",
                            "video": video_path,
                            "question": question,
                            "answer": ex.get("answer", ""),
                            "action_topk": k,
                            "frames_per_window": fpw,
                            "intended_num_images": k * fpw,
                            "num_images": len(all_frame_indices),
                            "duration_s": duration_s,
                            "fps": fps,
                            "video_num_frames": nframes,
                            "windows": windows,
                            "frame_indices": all_frame_indices,
                            "decode_latency_s": decode_latency_s,
                            "vlm_latency_s": vlm_latency_s,
                            "total_latency_s": decode_latency_s + vlm_latency_s,
                            "pred_text": pred_text,
                            "ok": True,
                            "error": None,
                            "window_summaries": window_summaries,
                            "summary_finish_reasons": summary_finish_reasons,
                            "final_finish_reason": final_fr,
                            "technique": technique_name,
                            "total_output_token_budget": args.total_output_token_budget,
                            "summary_budget_fraction": args.summary_budget_fraction,
                            "intermediate_token_budget_total": intermediate_token_budget_total,
                            "per_intermediate_max_tokens_effective": per_window_summary_max_tokens_effective,
                            "answer_max_tokens_effective": answer_max_tokens_effective,
                            "num_intermediate_units": k,
                            "actual_total_output_token_budget_cap": actual_total_output_token_budget_cap,
                            "final_latency_s": final_lat,
                            "summary_latency_s": vlm_latency_s - final_lat,
                            "proposal_strategy": args.proposal_strategy,
                            "proposal_scores": proposal_scores,
                            "clip_stride_s": args.clip_stride_s,
                            "clip_frames_per_candidate": args.clip_frames_per_candidate,
                            "clip_batch_size": args.clip_batch_size,
                            "proposal_latency_s": proposal_latency_s,
                            "total_latency_with_proposal_s": proposal_latency_s + decode_latency_s + vlm_latency_s,
                        }

                    except Exception as e:
                        row = {
                            "video_id": qid,
                            "qid": qid,
                            "type": "egoschema",
                            "video": video_path,
                            "question": question,
                            "choices": ex.get("choices"),
                            "answer_idx": ex.get("answer_idx"),
                            "answer_label": ex.get("answer_label"),
                            "answer": ex.get("answer", ""),
                            "action_topk": k,
                            "frames_per_window": fpw,
                            "num_images": k * fpw,
                            "duration_s": duration_s,
                            "windows": windows,
                            "decode_latency_s": decode_latency_s,
                            "vlm_latency_s": vlm_latency_s,
                            "total_latency_s": decode_latency_s + vlm_latency_s,
                            "pred_text": "",
                            "ok": False,
                            "error": repr(e),
                            "window_summaries": window_summaries,
                            "summary_finish_reasons": summary_finish_reasons,
                            "final_finish_reason": None,
                            "technique": technique_name,
                            "total_output_token_budget": args.total_output_token_budget,
                            "summary_budget_fraction": args.summary_budget_fraction,
                            "intermediate_token_budget_total": intermediate_token_budget_total,
                            "per_intermediate_max_tokens_effective": per_window_summary_max_tokens_effective,
                            "answer_max_tokens_effective": answer_max_tokens_effective,
                            "num_intermediate_units": k,
                            "actual_total_output_token_budget_cap": actual_total_output_token_budget_cap,
                            "fps": fps,
                            "video_num_frames": nframes,
                            "frame_indices": all_frame_indices,
                            "final_latency_s": final_lat,
                            "intended_num_images": k * fpw,
                            "num_images": len(all_frame_indices),
                            "summary_latency_s": vlm_latency_s - final_lat,
                            "proposal_strategy": args.proposal_strategy,
                            "proposal_scores": proposal_scores,
                            "clip_stride_s": args.clip_stride_s,
                            "clip_frames_per_candidate": args.clip_frames_per_candidate,
                            "clip_batch_size": args.clip_batch_size,
                            "proposal_latency_s": proposal_latency_s,
                            "total_latency_with_proposal_s": proposal_latency_s + decode_latency_s + vlm_latency_s,
                        }

                    out.write(json.dumps(row) + "\n")
                    out.flush()

    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
