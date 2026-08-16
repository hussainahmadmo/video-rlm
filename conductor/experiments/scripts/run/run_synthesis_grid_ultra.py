import argparse
import base64
import io
import json
import time
import re
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import torch.nn.functional as F
import torch
from transformers import AutoModel, AutoProcessor

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

_RETRIEVAL_MODEL = None
_RETRIEVAL_PROCESSOR = None
_CLIP_DEVICE = None
_VR_CACHE = {}


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


from decord import VideoReader, cpu
from decord import gpu
from decord import bridge
bridge.set_bridge("torch")

class DecordVideo:

    def __init__(self, path):
        self.path = path

        self.vr = VideoReader(
            path,
            ctx=cpu(0),
            width=224,
            height=224,
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
    
def init_models():
    global _RETRIEVAL_MODEL
    global _RETRIEVAL_PROCESSOR

    print("Loading SigLIP...")

    _RETRIEVAL_PROCESSOR = AutoProcessor.from_pretrained(
        "google/siglip-base-patch16-224"
    )

    _RETRIEVAL_MODEL = (
        AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        .to(CLIP_DEVICE)
    )

    _RETRIEVAL_MODEL.eval()

    print("SigLIP loaded")

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


def get_retrieval_model(device=None):
    global _RETRIEVAL_MODEL
    global _RETRIEVAL_PROCESSOR

    if device is None:
        device = CLIP_DEVICE

    if _RETRIEVAL_MODEL is None:

        _RETRIEVAL_PROCESSOR = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

        _RETRIEVAL_MODEL = (
            AutoModel.from_pretrained(
                "google/siglip-base-patch16-224"
            )
            .to(device)
        )

        _RETRIEVAL_MODEL.eval()

    return (
        _RETRIEVAL_MODEL,
        _RETRIEVAL_PROCESSOR,
        device,
    )

def extract_feature_tensor(output):
    if isinstance(output, torch.Tensor):
        return output

    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value

    last_hidden = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden[:, 0]

    raise TypeError(
        f"Could not extract feature tensor from {type(output).__name__}"
    )

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


def build_choice_evidence_comparison_prompt(
    *,
    question,
    choices,
    evidence_text,
    task_instruction,
    require_confidence=False,
):
    label_text = ", ".join(
        chr(ord("A") + i)
        for i in range(len(choices))
    )
    confidence_line = (
        "Confidence: <number between 0.0 and 1.0>"
        if require_confidence
        else ""
    )
    confidence_instruction = (
        "\nUse high confidence only if the evidence directly supports "
        "the chosen answer and contradicts the main alternatives."
        if require_confidence
        else ""
    )

    return textwrap.dedent(
        f"""
        You are answering a multiple-choice question about a video.

        Question:
        {question}

        Choices:
        {format_choices(choices)}

        Evidence:
        {evidence_text}

        Task:
        {task_instruction}

        First compare every answer choice against the evidence. For each choice,
        list only concrete facts:
        - supported evidence
        - contradicted evidence
        - missing evidence

        Do not choose by general plausibility. Choose the option whose required
        facts are most directly supported and least contradicted by the evidence.

        Return your final answer at the end in exactly this format:
        Answer: <one label from {label_text}>
        {confidence_line}
        {confidence_instruction}
        """
    ).strip()


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


def build_structured_evidence_summary_prompt(item, start_s, end_s):
    return f"""
This video segment is from {start_s:.1f}s to {end_s:.1f}s.

Question:
{item["question"]}

Choices:
{format_choices(item["choices"])}

Extract concrete visual evidence from this segment. Use the answer choices as a
checklist, but do not choose the final answer.

Return exactly:
Main scene: <place and visible activity>
People: <who is visible and what they are doing>
Tools/objects: <specific visible tools, objects, materials, containers, surfaces>
Actions: <specific physical actions, verbs only if directly visible>
State changes: <what changes from start to end, if anything>
Answer-choice evidence: <which choice facts are supported, contradicted, or missing>
Uncertainty: <important ambiguities or things not visible>

Do not infer hidden intent. Do not rename objects unless they are clearly visible.
If evidence for a choice is missing, say it is missing.
""".strip()


def build_timeline_window_summary_prompt(item, start_s, end_s, evidence_type):
    if evidence_type == "sequence_ordering":
        return f"""
This video segment is from {start_s:.1f}s to {end_s:.1f}s.

Question:
{item["question"]}

Choices:
{format_choices(item["choices"])}

Extract only sequence/order evidence from the sampled frames.
Preserve what appears first, middle, and last inside this segment.
Use the answer choices as a checklist, but do not choose the final answer.

Return exactly:
Scene/state: <main visible scene or state>
Start event: <what is visible first, or "unclear">
Middle event: <what is visible in the middle, or "unclear">
End event: <what is visible last, or "unclear">
Visible transition: <scene/event transition, or "none/unclear">
Matches choice events: <which choice event phrases this segment supports, or "none">
Relevant: <yes/no>
""".strip()

    detail_instruction = ""

    if evidence_type in {
        "screen_state_change",
        "localized_object_attribute",
        "localized_temporal_detail",
    }:
        detail_instruction = (
            "\nPay special attention to small objects, clothing, text, "
            "screen contents, and visible state changes."
        )

    return f"""
This video segment is from {start_s:.1f}s to {end_s:.1f}s.

Question:
{item["question"]}

Choices:
{format_choices(item["choices"])}

Describe the answer-relevant visual evidence in this segment.
Preserve temporal order inside the segment.
If the segment does not contain relevant evidence, say: "No relevant evidence."
Do not choose the final answer yet.{detail_instruction}
""".strip()


def build_timeline_answer_prompt(
    item,
    window_summaries,
    evidence_type,
    *,
    require_confidence=False,
):
    evidence_text = "\n\n".join(
        f"[Window {w['window_idx']} | {w['start_s']:.1f}-{w['end_s']:.1f}s]\n{w['summary']}"
        for w in window_summaries
    )

    if evidence_type == "sequence_ordering":
        instruction = (
            "Use the ordered window summaries as the primary evidence. "
            "First write a concise chronological event list from the evidence. "
            "Then check every answer choice against that event list and mark "
            "which events match, are missing, or are in the wrong order. "
            "Choose the option whose full sequence best matches the evidence. "
            "Do not choose a choice just because it mentions one correct event."
        )
    else:
        instruction = (
            "Use the ordered window summaries as the primary evidence. "
            "For screen-state questions, focus on what changed on screen "
            "and when."
        )

    if require_confidence:
        return build_choice_evidence_comparison_prompt(
            question=item["question"],
            choices=item["choices"],
            evidence_text=(
                f"Evidence type: {evidence_type}\n"
                f"Ordered window evidence:\n{evidence_text}"
            ),
            task_instruction=instruction,
            require_confidence=True,
        )

    return build_choice_evidence_comparison_prompt(
        question=item["question"],
        choices=item["choices"],
        evidence_text=(
            f"Evidence type: {evidence_type}\n"
            f"Ordered window evidence:\n{evidence_text}"
        ),
        task_instruction=instruction,
        require_confidence=False,
    )


def build_timestamped_sequence_event_prompt(item, start_s, end_s):
    midpoint_s = (start_s + end_s) / 2.0
    return f"""
You are extracting one timestamped event record from a video segment.

Segment:
{start_s:.1f}s to {end_s:.1f}s

Representative timestamp:
{midpoint_s:.1f}s

Question:
{item["question"]}

Choices:
{format_choices(item["choices"])}

Use the sampled frames only. Focus on visible events, scene changes, objects,
people, and actions that could help decide the correct temporal sequence.
Do not choose the final answer.

Return exactly:
timestamp_s: {midpoint_s:.1f}
scene: <main visible scene/place/state>
actors: <visible people or "none/unclear">
objects: <important visible objects/text/screens/materials>
event: <one concise visible event at this timestamp>
transition: <what changed from earlier frames in this segment to later frames, or "none/unclear">
choice_event_matches: <answer-choice event phrases supported here, or "none">
confidence: <0.0 to 1.0>

Do not infer events that are not visible. If the frames are ambiguous, say so.
""".strip()


def extract_sequence_event_phrases(text):
    cleaned = str(text)
    cleaned = re.sub(
        r"\b(first|second|third|fourth|fifth|next|then|finally|lastly|afterwards|afterward)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    parts = re.split(
        r"\s*(?:;|,?\s+and\s+finally\b|,?\s+and\s+then\b|,|\.\s*)\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    phrases = []
    seen = set()

    for part in parts:
        phrase = " ".join(part.strip().split())
        if len(phrase.split()) < 3:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        phrases.append(phrase)
        seen.add(key)

    return phrases


def build_sequence_event_phrase_queries(item):
    queries = []
    for idx, choice in enumerate(item.get("choices") or []):
        label = chr(ord("A") + idx)
        phrases = extract_sequence_event_phrases(choice)
        if not phrases:
            phrases = [str(choice)]

        for phrase in phrases:
            queries.append(
                f"{item['question']}\nChoice {label} event: {phrase}"
            )

    return queries or build_clip_query(
        item,
        config={"evidence_type": "sequence_ordering"},
    )


def build_targeted_sequence_event_prompt(item, start_s, end_s):
    event_phrases = []
    for idx, choice in enumerate(item.get("choices") or []):
        label = chr(ord("A") + idx)
        for phrase in extract_sequence_event_phrases(choice):
            event_phrases.append(f"{label}: {phrase}")

    phrase_text = "\n".join(event_phrases[:40])
    midpoint_s = (start_s + end_s) / 2.0

    return f"""
You are verifying whether this video segment contains any answer-choice sequence events.

Segment:
{start_s:.1f}s to {end_s:.1f}s

Representative timestamp:
{midpoint_s:.1f}s

Question:
{item["question"]}

Answer-choice event checklist:
{phrase_text}

Use only the sampled frames. Do not answer the question yet.

Return exactly:
timestamp_s: {midpoint_s:.1f}
scene: <main visible scene/place/state>
event: <specific visible event in this segment>
matched_choice_events: <choice labels and phrases directly supported here, or "none">
missing_or_unclear: <choice events that might be relevant but are not clearly visible>
order_note: <whether this appears before/after another relevant event if visible, or "local only">
confidence: <0.0 to 1.0>

Only mark an event as matched if the visible scene directly supports it.
""".strip()


def build_timestamped_sequence_answer_prompt(item, event_records):
    timeline_text = "\n\n".join(
        f"[t={r['timestamp_s']:.1f}s | window {r['start_s']:.1f}-{r['end_s']:.1f}s]\n{r['event_text']}"
        for r in event_records
    )

    label_text = ", ".join(
        chr(ord("A") + i)
        for i in range(len(item["choices"]))
    )

    return textwrap.dedent(
        f"""
        You are answering a sequence/order question about a video.

        Question:
        {item["question"]}

        Choices:
        {format_choices(item["choices"])}

        Timestamped event records, already sorted by time:
        {timeline_text}

        Task:
        First build a concise ordered timeline using only the timestamped event
        records. Then compare every answer choice against that timeline.
        For each choice, mark:
        - events supported in the same order
        - events missing from the records
        - events contradicted or in the wrong order

        Choose the answer whose complete ordered sequence best matches the
        timestamped timeline. Do not choose a choice because it matches only one
        isolated event.

        Return your final answer at the end in exactly this format:
        Answer: <one label from {label_text}>
        Confidence: <number between 0.0 and 1.0>
        """
    ).strip()


def build_choice_sequence_verifier_prompt(
    item,
    *,
    label,
    choice,
    events,
    event_windows,
):
    event_lines = "\n".join(
        f"{idx + 1}. {event}"
        for idx, event in enumerate(events)
    )
    window_lines = "\n".join(
        f"- {row['window'][0]:.1f}-{row['window'][1]:.1f}s "
        f"score={row.get('score')}"
        for row in event_windows
    )

    return textwrap.dedent(
        f"""
        You are verifying one candidate answer for a video sequence question.

        Question:
        {item["question"]}

        Candidate choice {label}:
        {choice}

        Claimed ordered events in choice {label}:
        {event_lines}

        Candidate evidence windows retrieved for this choice:
        {window_lines}

        The attached frames are sampled from the candidate windows in
        chronological order. Use only visible evidence from those frames.

        Decide whether choice {label}'s events are visible and whether they
        occur in the claimed order.

        Return exactly:
        Choice: {label}
        matched_events: <number matched>/<number claimed>
        order_supported: <yes/no/unclear>
        contradicted_order: <yes/no/unclear>
        missing_events: <short list or none>
        score: <number from -5 to 10>
        rationale: <one concise sentence grounded in the frames>

        Score high only when multiple claimed events are directly visible in
        the correct order. Penalize missing events and wrong order. Do not give
        credit for general plausibility.
        """
    ).strip()


def parse_confidence(response):
    """
    Parse confidence from model response.
    Expected format can include: Confidence: 0.82
    """
    if response is None:
        return None

    text = response.strip()

    m = re.search(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        try:
            conf = float(m.group(1))
            if conf > 1.0:
                conf = conf / 100.0
            return max(0.0, min(1.0, conf))
        except Exception:
            return None

    return None


def parse_sequence_verifier_score(response, total_events):
    text = str(response or "")
    score = None

    m = re.search(
        r"\bscore\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            score = float(m.group(1))
        except Exception:
            score = None

    matched = None
    m = re.search(
        r"\bmatched_events\s*[:=]\s*([0-9]+)\s*/\s*([0-9]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            matched = int(m.group(1))
        except Exception:
            matched = None

    if score is None:
        score = float(matched or 0)
        if re.search(
            r"\border_supported\s*[:=]\s*yes\b",
            text,
            re.IGNORECASE,
        ):
            score += 1.0
        if re.search(
            r"\bcontradicted_order\s*[:=]\s*yes\b",
            text,
            re.IGNORECASE,
        ):
            score -= 2.0

    if matched is None:
        matched = 0

    return {
        "score": score,
        "matched_events": matched,
        "total_events": total_events,
    }


def build_answer_prompt_with_confidence(question, choices, extra_context=None):
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

Return your answer in exactly this format:

Answer: <one label from {label_text}>
Confidence: <number from 0.0 to 1.0>

Use high confidence only if the sampled frames contain enough visual evidence.
Use low confidence if the evidence is incomplete, ambiguous, or missing.
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

    final_pattern = (
        r"(?:FINAL\s+ANSWER|ANSWER)\s*[:=]\s*("
        + "|".join(valid)
        + r")\b"
    )
    m = re.search(final_pattern, text)
    if m:
        return m.group(1)

    if "CHOICE CHECKS" in text or "EVIDENCE TIMELINE" in text:
        return None

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


def call_llm_answer(prompt, max_tokens=32):
    print("[REAL LLM CALL]", flush=True)

    for attempt in range(2):
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        content = resp.choices[0].message.content

        if content:
            return content

        print(
            f"[EMPTY LLM RESPONSE] retry={attempt + 1}",
            flush=True,
        )

    return None


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

    model, processor, device = get_retrieval_model()

    if isinstance(query, (list, tuple)):
        queries = [str(q) for q in query]
    else:
        queries = [str(query)]

    text_inputs = processor(
        text=queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    ).to(device)

    text_inputs = {
        k: v.to(device)
        for k, v in text_inputs.items()
    }

    print("QUERY:")
    for idx, text_query in enumerate(queries):
        print(f"[{idx}] {text_query}")

    print("INPUT_IDS SHAPE:")
    print(text_inputs["input_ids"].shape)

    print("TOKENS:")
    print(text_inputs["input_ids"].shape[1])

    sync_device()

    t_text = time.time()

    text_feat = extract_feature_tensor(
        model.get_text_features(
            **text_inputs
        )
    )

    sync_device()

    LATENCY_STATS["clip_text_s"] = (
        LATENCY_STATS.get("clip_text_s", 0)
        + (time.time() - t_text)
    )

    text_feat = text_feat / text_feat.norm(
        dim=-1,
        keepdim=True,
    )

    scores_by_query = scan["features"] @ text_feat.T
    scores = scores_by_query.max(dim=1).values

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


def get_video_duration_s(item):
    duration_s = item.get("duration_s")

    if duration_s is not None:
        return float(duration_s)

    duration_s, _, _ = duration_fps_nframes(
        item["video"]
    )

    return float(duration_s)


def make_neighbor_windows(
    window,
    *,
    duration_s,
    window_len_s,
):
    start, end = window
    span = max(
        float(window_len_s),
        float(end) - float(start),
    )

    return [
        [
            max(0.0, float(start) - span),
            max(0.0, float(start)),
        ],
        [
            max(0.0, float(start)),
            min(float(duration_s), float(end)),
        ],
        [
            min(float(duration_s), float(end)),
            min(float(duration_s), float(end) + span),
        ],
    ]


def make_uniform_anchor_windows(
    *,
    duration_s,
    window_len_s,
    count=4,
):
    if duration_s <= 0:
        return []

    if count <= 1:
        centers = [duration_s / 2.0]
    else:
        centers = np.linspace(
            0.0,
            duration_s,
            count,
        ).tolist()

    half = float(window_len_s) / 2.0
    windows = []

    for center in centers:
        start = max(0.0, center - half)
        end = min(float(duration_s), center + half)
        if end > start:
            windows.append([start, end])

    return windows


def add_window_if_new(
    selected,
    window,
    *,
    min_gap_s=0.25,
):
    start, end = window
    if end <= start:
        return

    key = (
        round(float(start) / min_gap_s),
        round(float(end) / min_gap_s),
    )

    if key in selected["seen"]:
        return

    selected["seen"].add(key)
    selected["windows"].append(
        [float(start), float(end)]
    )


def apply_profiler_window_hints(
    *,
    item,
    top_windows,
    config,
):
    duration_s = get_video_duration_s(item)
    window_len_s = float(config["window_len_s"])
    selected = {
        "seen": set(),
        "windows": [],
    }

    for row in top_windows:
        if config.get("expand_neighbors"):
            for window in make_neighbor_windows(
                row["window"],
                duration_s=duration_s,
                window_len_s=window_len_s,
            ):
                add_window_if_new(
                    selected,
                    window,
                )
        else:
            add_window_if_new(
                selected,
                row["window"],
            )

    if config.get("include_uniform_anchors"):
        for window in make_uniform_anchor_windows(
            duration_s=duration_s,
            window_len_s=window_len_s,
            count=4,
        ):
            add_window_if_new(
                selected,
                window,
            )

    selected_windows = selected["windows"]

    if config.get("preserve_order"):
        selected_windows = sorted(
            selected_windows,
            key=lambda w: (w[0], w[1]),
        )

    max_windows = max(
        1,
        min(
            int(config.get("max_selected_windows", 24)),
            int(config.get("vlm_budget", 24)),
        ),
    )

    return selected_windows[:max_windows]


def build_profiler_evidence_context(
    *,
    selected_windows,
    top_windows,
    config,
):
    evidence_type = config.get(
        "evidence_type",
        "generic",
    )

    lines = [
        "These frames were selected using the profiler policy.",
        f"Evidence type: {evidence_type}.",
    ]

    if config.get("preserve_order"):
        lines.append(
            "Windows are presented in chronological order; use timing/order when choosing the answer."
        )

    if config.get("expand_neighbors"):
        lines.append(
            "Neighbor windows around retrieved evidence are included to catch first/next/later context."
        )

    if config.get("include_uniform_anchors"):
        lines.append(
            "Uniform timeline anchors are included so broad video context is not lost."
        )

    if evidence_type in {
        "localized_object_attribute",
        "localized_temporal_detail",
        "screen_state_change",
    }:
        lines.append(
            "Pay close attention to small visible details, objects, clothing, text, and screen state changes."
        )

    score_by_window = {
        (
            round(float(row["window"][0]), 3),
            round(float(row["window"][1]), 3),
        ): row.get("score")
        for row in top_windows
    }

    for i, window in enumerate(selected_windows):
        key = (
            round(float(window[0]), 3),
            round(float(window[1]), 3),
        )
        score = score_by_window.get(key)
        if score is None:
            lines.append(
                f"Selected window {i}: {window[0]:.1f}-{window[1]:.1f}s"
            )
        else:
            lines.append(
                f"Selected window {i}: {window[0]:.1f}-{window[1]:.1f}s, CLIP score={score:.4f}"
            )

    return "\n".join(lines)

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

    sync_device()

    t_decode = time.time()

    batch = vr.get_batch(unique_idxs)

    sync_device()

    LATENCY_STATS["answer_decode_s"] = (
        LATENCY_STATS.get("answer_decode_s", 0)
        + (time.time() - t_decode)
    )

    sync_device()

    t_copy = time.time()

    cpu_batch = batch.cpu().numpy()

    sync_device()

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


def build_clip_query(item, config=None):
    choices = item.get("choices") or []
    evidence_type = None

    if config and config.get("sequence_query_mode") == "question_only":
        return item["question"].strip()

    if config:
        evidence_type = config.get("evidence_type")

    if evidence_type is None:
        evidence_type = item.get("evidence_type")

    option_aware_types = {
        "sequence_ordering",
        "screen_state_change",
        "localized_temporal_detail",
        "localized_object_attribute",
        "first_or_next_event",
    }

    if evidence_type in option_aware_types and choices:
        return [
            (
                item["question"]
                + "\nCandidate answer: "
                + str(choice)
            )
            for choice in choices
        ]

    question = item["question"].strip()

    if len(question.split()) <= 12 and choices:
        return [
            question
            + "\nCandidate answer: "
            + str(choice)
            for choice in choices
        ]

    return question


def run_clip_oneshot(item, config):
    LATENCY_STATS.clear()
    query = build_clip_query(
        item,
        config=config,
    )

    retrieval = clip_topk_windows(
        video_path=item["video"],
        query=query,
        k=config["clip_topk"],
        window_len_s=config["window_len_s"],
        scan_fps=config["scan_fps"],
    )

    top_windows = retrieval["top_windows"]

    selected_windows = apply_profiler_window_hints(
        item=item,
        top_windows=top_windows,
        config=config,
    )

    print(
        f"[RETRIEVAL] "
        f"Candidate windows={retrieval['candidate_windows_examined']} "
        f"Selected={len(selected_windows)}"
    )

    if len(selected_windows) == 0:
        raise RuntimeError("No windows selected")
    
    TOTAL_FRAMES = config["vlm_budget"]
    base = TOTAL_FRAMES // len(selected_windows)
    extra = TOTAL_FRAMES % len(selected_windows)

    #step 8 - sample frames for selected windows
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

    evidence_hint = build_profiler_evidence_context(
        selected_windows=selected_windows,
        top_windows=top_windows,
        config=config,
    )

    if config.get("answer_with_confidence"):
        prompt = build_answer_prompt_with_confidence(
            item["question"],
            item["choices"],
            extra_context=evidence_hint,
        )
    else:
        prompt = build_answer_prompt(
            item["question"],
            item["choices"],
            extra_context=evidence_hint,
        )

    response = call_vlm_answer(frames, prompt)
    confidence = parse_confidence(response)

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
        "prediction_confidence": confidence,
        "num_vlm_calls": 1,

        "retrieval_effort": {
            "candidate_windows":
                retrieval["candidate_windows_examined"],

            "selected_windows":
                len(selected_windows),

            "selected_frames":
                len(frames["frame_indices"]),
        },

        "evidence": {
            "clip_top_windows": top_windows,
            "selected_windows": selected_windows,
            "selected_frame_indices":
                frames["frame_indices"],
            "selected_timestamps":
                frames["timestamps"],
        },
    }


def select_windows_for_profiler_policy(item, config):
    evidence_type = config.get("evidence_type")
    duration_s = get_video_duration_s(item)

    if evidence_type == "sequence_ordering":
        num_windows = min(
            int(config.get("timeline_windows", 12)),
            int(config.get("vlm_budget", 32)),
        )
        windows = make_uniform_windows(
            duration_s,
            num_windows,
        )
        return {
            "top_windows": [
                {
                    "window": window,
                    "score": None,
                    "source": "uniform_timeline",
                }
                for window in windows
            ],
            "selected_windows": windows,
            "candidate_windows_examined": num_windows,
            "selection_source": "uniform_timeline",
        }

    query = build_clip_query(
        item,
        config=config,
    )
    retrieval = clip_topk_windows(
        video_path=item["video"],
        query=query,
        k=config["clip_topk"],
        window_len_s=config["window_len_s"],
        scan_fps=config["scan_fps"],
    )
    top_windows = retrieval["top_windows"]
    selected_windows = apply_profiler_window_hints(
        item=item,
        top_windows=top_windows,
        config=config,
    )

    return {
        "top_windows": top_windows,
        "selected_windows": selected_windows,
        "candidate_windows_examined": retrieval[
            "candidate_windows_examined"
        ],
        "selection_source": "option_aware_clip",
    }


def choose_timeline_windows(item, config):
    selected = select_windows_for_profiler_policy(
        item,
        config,
    )
    windows = selected["selected_windows"]

    if config.get("preserve_order") or config.get(
        "evidence_type"
    ) in {
        "sequence_ordering",
        "screen_state_change",
    }:
        windows = sorted(
            windows,
            key=lambda w: (w[0], w[1]),
        )

    max_windows = min(
        len(windows),
        int(config.get("map_max_windows", 12)),
    )
    windows = windows[:max_windows]
    selected["selected_windows"] = windows

    return selected


def choose_targeted_sequence_windows(item, config):
    query = build_sequence_event_phrase_queries(item)
    retrieval = clip_topk_windows(
        video_path=item["video"],
        query=query,
        k=int(config.get("sequence_event_topk", config.get("clip_topk", 8))),
        window_len_s=config["window_len_s"],
        scan_fps=config["scan_fps"],
    )
    selected_windows = apply_profiler_window_hints(
        item=item,
        top_windows=retrieval["top_windows"],
        config={
            **config,
            "preserve_order": True,
            "expand_neighbors": config.get("sequence_expand_neighbors", False),
            "include_uniform_anchors": config.get("include_uniform_anchors", True),
        },
    )

    selected_windows = sorted(
        selected_windows,
        key=lambda w: (w[0], w[1]),
    )

    max_windows = min(
        len(selected_windows),
        int(config.get("map_max_windows", 12)),
    )

    return {
        "top_windows": retrieval["top_windows"],
        "selected_windows": selected_windows[:max_windows],
        "candidate_windows_examined": retrieval[
            "candidate_windows_examined"
        ],
        "selection_source": "choice_event_clip",
        "event_queries": query,
    }


def run_clip_map_answer(item, config):
    LATENCY_STATS.clear()
    evidence_type = config.get(
        "evidence_type",
        "generic",
    )
    selection = choose_timeline_windows(
        item,
        config,
    )
    selected_windows = selection["selected_windows"]

    if not selected_windows:
        raise RuntimeError("No windows selected")

    frames_per_window = int(
        config.get(
            "map_frames_per_window",
            3
            if evidence_type == "sequence_ordering"
            else 4,
        )
    )
    window_summaries = []
    actual_frames = 0

    print(
        f"[MAP] source={selection['selection_source']} "
        f"windows={len(selected_windows)} "
        f"frames_per_window={frames_per_window}"
    )

    for widx, (start_s, end_s) in enumerate(selected_windows):
        frames = sample_uniform_frames(
            video_path=item["video"],
            num_frames=frames_per_window,
            start_s=start_s,
            end_s=end_s,
        )
        actual_frames += len(frames["frame_indices"])

        prompt = build_timeline_window_summary_prompt(
            item,
            start_s,
            end_s,
            evidence_type,
        )
        summary = call_vlm_text(
            frames,
            prompt,
        )
        window_summaries.append({
            "window_idx": widx,
            "start_s": start_s,
            "end_s": end_s,
            "summary": summary,
        })

    answer_prompt = build_timeline_answer_prompt(
        item,
        window_summaries,
        evidence_type,
        require_confidence=bool(
            config.get("answer_with_confidence")
        ),
    )
    response = call_llm_answer(
        answer_prompt,
        max_tokens=(
            256
            if config.get("answer_with_confidence")
            else 128
        ),
    )
    confidence = parse_confidence(response)

    return {
        "prediction_label": parse_mcq_label(
            response,
            num_choices=len(item["choices"]),
        ),
        "prediction_text": response,
        "prediction_confidence": confidence,
        "num_vlm_calls": len(selected_windows) + 1,
        "evidence": {
            "clip_top_windows": selection["top_windows"],
            "selected_windows": selected_windows,
            "window_summaries": window_summaries,
            "selection_source": selection["selection_source"],
        },
        "retrieval_effort": {
            "candidate_windows": selection[
                "candidate_windows_examined"
            ],
            "selected_windows": len(selected_windows),
            "selected_frames": actual_frames,
        },
    }


def run_timestamped_sequence_answer(item, config):
    LATENCY_STATS.clear()
    if config.get("target_choice_events", True):
        selection = choose_targeted_sequence_windows(
            item,
            config,
        )
    else:
        selection = choose_timeline_windows(
            item,
            config,
        )
    selected_windows = selection["selected_windows"]

    if not selected_windows:
        raise RuntimeError("No windows selected")

    frames_per_window = int(
        config.get(
            "timestamp_frames_per_window",
            config.get("map_frames_per_window", 3),
        )
    )

    event_records = []
    actual_frames = 0

    print(
        f"[TIMESTAMPED_SEQUENCE] "
        f"source={selection['selection_source']} "
        f"windows={len(selected_windows)} "
        f"frames_per_window={frames_per_window}"
    )

    for widx, (start_s, end_s) in enumerate(selected_windows):
        frames = sample_uniform_frames(
            video_path=item["video"],
            num_frames=frames_per_window,
            start_s=start_s,
            end_s=end_s,
        )
        actual_frames += len(frames["frame_indices"])

        if config.get("target_choice_events", True):
            prompt = build_targeted_sequence_event_prompt(
                item,
                start_s,
                end_s,
            )
        else:
            prompt = build_timestamped_sequence_event_prompt(
                item,
                start_s,
                end_s,
            )
        event_text = call_vlm_text(
            frames,
            prompt,
        )
        event_records.append({
            "window_idx": widx,
            "start_s": start_s,
            "end_s": end_s,
            "timestamp_s": (start_s + end_s) / 2.0,
            "event_text": event_text,
        })

    answer_prompt = build_timestamped_sequence_answer_prompt(
        item,
        event_records,
    )
    response = call_llm_answer(
        answer_prompt,
        max_tokens=384,
    )
    confidence = parse_confidence(response)

    return {
        "prediction_label": parse_mcq_label(
            response,
            num_choices=len(item["choices"]),
        ),
        "prediction_text": response,
        "prediction_confidence": confidence,
        "num_vlm_calls": len(selected_windows) + 1,
        "evidence": {
            "clip_top_windows": selection["top_windows"],
            "selected_windows": selected_windows,
            "timestamped_events": event_records,
            "selection_source": selection["selection_source"],
            "event_queries": selection.get("event_queries"),
        },
        "retrieval_effort": {
            "candidate_windows": selection[
                "candidate_windows_examined"
            ],
            "selected_windows": len(selected_windows),
            "selected_frames": actual_frames,
        },
    }


def run_choice_sequence_verifier(item, config):
    LATENCY_STATS.clear()
    choice_results = []
    total_candidate_windows = 0
    total_selected_frames = 0
    num_vlm_calls = 0

    event_topk = int(
        config.get(
            "sequence_choice_topk",
            min(6, int(config.get("clip_topk", 8))),
        )
    )
    max_events = int(
        config.get("sequence_choice_max_events", 5)
    )
    max_windows = int(
        config.get("sequence_choice_max_windows", 6)
    )
    frames_per_window = int(
        config.get("sequence_choice_frames_per_window", 2)
    )

    print(
        f"[CHOICE_SEQUENCE_VERIFIER] "
        f"choices={len(item['choices'])} "
        f"topk={event_topk} "
        f"max_windows={max_windows} "
        f"frames_per_window={frames_per_window}"
    )

    for idx, choice in enumerate(item["choices"]):
        label = chr(ord("A") + idx)
        events = extract_sequence_event_phrases(choice)
        if not events:
            events = [str(choice)]
        events = events[:max_events]

        queries = [
            f"{item['question']}\nChoice {label} event: {event}"
            for event in events
        ]
        retrieval = clip_topk_windows(
            video_path=item["video"],
            query=queries,
            k=event_topk,
            window_len_s=config["window_len_s"],
            scan_fps=config["scan_fps"],
        )
        total_candidate_windows += retrieval[
            "candidate_windows_examined"
        ]

        selected_windows = apply_profiler_window_hints(
            item=item,
            top_windows=retrieval["top_windows"],
            config={
                **config,
                "preserve_order": True,
                "expand_neighbors": False,
                "include_uniform_anchors": False,
                "max_selected_windows": max_windows,
                "vlm_budget": max_windows,
            },
        )
        selected_windows = sorted(
            selected_windows,
            key=lambda w: (w[0], w[1]),
        )[:max_windows]

        if not selected_windows:
            choice_results.append({
                "label": label,
                "choice": choice,
                "events": events,
                "score": -5.0,
                "matched_events": 0,
                "total_events": len(events),
                "prediction_text": "No windows selected.",
                "top_windows": retrieval["top_windows"],
                "selected_windows": [],
            })
            continue

        frames = sample_frames_from_windows(
            video_path=item["video"],
            windows=selected_windows,
            frame_allocations=[
                frames_per_window
                for _ in selected_windows
            ],
        )
        total_selected_frames += len(frames["frame_indices"])

        prompt = build_choice_sequence_verifier_prompt(
            item,
            label=label,
            choice=choice,
            events=events,
            event_windows=retrieval["top_windows"],
        )
        response = call_vlm_text(
            frames,
            prompt,
        )
        num_vlm_calls += 1
        parsed = parse_sequence_verifier_score(
            response,
            total_events=len(events),
        )

        choice_results.append({
            "label": label,
            "choice": choice,
            "events": events,
            "score": parsed["score"],
            "matched_events": parsed["matched_events"],
            "total_events": parsed["total_events"],
            "prediction_text": response,
            "top_windows": retrieval["top_windows"],
            "selected_windows": selected_windows,
            "selected_frame_indices": frames["frame_indices"],
            "selected_timestamps": frames["timestamps"],
        })

    best = max(
        choice_results,
        key=lambda row: (
            row["score"],
            row["matched_events"],
            -ord(row["label"]),
        ),
    )
    aggregate_text = "\n\n".join(
        f"Choice {row['label']} score={row['score']} "
        f"matched={row['matched_events']}/{row['total_events']}\n"
        f"{row['prediction_text']}"
        for row in choice_results
    )
    aggregate_text += (
        f"\n\nAnswer: {best['label']}\n"
        f"Confidence: {min(1.0, max(0.0, best['score'] / 10.0)):.2f}"
    )

    return {
        "prediction_label": best["label"],
        "prediction_text": aggregate_text,
        "prediction_confidence": min(
            1.0,
            max(0.0, best["score"] / 10.0),
        ),
        "num_vlm_calls": num_vlm_calls,
        "evidence": {
            "choice_sequence_results": choice_results,
            "selection_source": "choice_sequence_verifier",
        },
        "retrieval_effort": {
            "candidate_windows": total_candidate_windows,
            "selected_windows": sum(
                len(row["selected_windows"])
                for row in choice_results
            ),
            "selected_frames": total_selected_frames,
        },
    }


@torch.no_grad()
def build_frame_scan_embeddings(
    video_path,
    scan_fps=2.0,
    window_len_s=8.0,
    ):

    model, processor, device = get_retrieval_model()

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

        sync_device()

        t = time.time()

        batch = vr.get_batch(chunk)
        t0 = time.time()
        sync_device()

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
        sync_device()
        decode_time += time.time() - t

        t_resize = time.time()
        pil_images = [
            Image.fromarray(x.cpu().numpy()).convert("RGB")
            for x in batch
        ]

        inputs = processor(
            images=pil_images,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        feats = extract_feature_tensor(
            model.get_image_features(
                **inputs
            )
        )
        sync_device()
        resize_time += time.time() - t_resize
        t = time.time()

        sync_device()
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

        if config.get("structured_evidence_summary"):
            prompt = build_structured_evidence_summary_prompt(
                item,
                start_s=start_s,
                end_s=end_s,
            )
        else:
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

    if config.get("choice_compare_answer"):
        answer_prompt = build_choice_evidence_comparison_prompt(
            question=item["question"],
            choices=item["choices"],
            evidence_text=evidence_text,
            task_instruction=(
                "Use the full-video timeline summaries to identify the "
                "main activity, important actions, tools, objects, and "
                "scene changes before selecting an answer. Prefer choices "
                "whose required actions and objects are directly supported "
                "by structured evidence, and penalize choices whose key facts "
                "are listed as missing or contradicted."
            ),
            require_confidence=bool(
                config.get("answer_with_confidence")
            ),
        )
    else:
        answer_prompt = build_answer_prompt(
            item["question"],
            item["choices"],
            extra_context=evidence_text,
        )

    response = call_llm_answer(
        answer_prompt,
        max_tokens=(
            256
            if config.get("choice_compare_answer")
            else 32
        ),
    )
    confidence = parse_confidence(response)


    return {
        "prediction_label": parse_mcq_label(response, num_choices=len(item["choices"])),
        "prediction_text": response,
        "prediction_confidence": confidence,
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

    if config["method"] == "clip_map_answer":
        return run_clip_map_answer(item, config)

    if config["method"] == "timestamped_sequence_answer":
        return run_timestamped_sequence_answer(item, config)

    if config["method"] == "choice_sequence_verifier":
        return run_choice_sequence_verifier(item, config)

    if config["method"] == "global_summary":
        return run_global_summary(item, config)

    if config["method"] == "map_summary":
        return run_map_summary(item, config)
    raise ValueError(f"Unknown method: {config['method']}")


def apply_sequence_oracle_clip_policy(config, item):
    duration_s, _, _ = duration_fps_nframes(item["video"])
    question = str(item.get("question") or "").lower()

    config["method"] = "clip_oneshot"
    config["skip_cost_clamp"] = True
    config["sequence_query_mode"] = "question_only"
    config["answer_with_confidence"] = False
    config["scan_fps"] = 0.015625
    config["clip_topk"] = 8
    config["window_len_s"] = 8.0
    config["vlm_budget"] = 8
    config["expand_neighbors"] = False
    config["preserve_order"] = True
    config["include_uniform_anchors"] = False
    config["sequence_policy"] = "oracle_clip_budget8"

    if duration_s >= 1200:
        config["scan_fps"] = 0.00390625
        config["clip_topk"] = 8
        config["vlm_budget"] = 32
        config["sequence_policy"] = "oracle_sparse_long_budget32"

    if (
        "what happens next" in question
        or "trapped" in question
    ):
        config["scan_fps"] = 0.015625
        config["clip_topk"] = 1
        config["vlm_budget"] = 32
        config["sequence_policy"] = "oracle_next_event_k1_budget32"

    elif (
        "from scene to scene" in question
        or "have in common" in question
    ):
        config["scan_fps"] = 0.015625
        config["clip_topk"] = 8
        config["vlm_budget"] = 2
        config["answer_with_confidence"] = True
        config["sequence_policy"] = "oracle_commonality_budget2"

    return config


def apply_oracle_edge_policy(config, item):
    evidence_type = config.get("evidence_type")
    question = str(item.get("question") or "").lower()

    if evidence_type == "counting_completeness":
        config["method"] = "clip_oneshot"
        config["scan_fps"] = 1.0
        config["clip_topk"] = 8
        config["window_len_s"] = 8.0
        config["vlm_budget"] = 32
        config["expand_neighbors"] = False
        config["preserve_order"] = True
        config["include_uniform_anchors"] = False
        config["answer_with_confidence"] = False
        config["skip_cost_clamp"] = True
        config["edge_policy"] = "oracle_counting_dense_scan"
        return config

    if (
        evidence_type == "generic"
        and (
            "refuses to pay back" in question
            or "what happens if" in question
        )
    ):
        config["method"] = "clip_oneshot"
        config["scan_fps"] = 0.015625
        config["clip_topk"] = 8
        config["window_len_s"] = 8.0
        config["vlm_budget"] = 32
        config["expand_neighbors"] = False
        config["preserve_order"] = True
        config["include_uniform_anchors"] = False
        config["answer_with_confidence"] = False
        config["skip_cost_clamp"] = True
        config["edge_policy"] = "oracle_causal_long_budget32"
        return config

    if (
        evidence_type == "generic"
        and "camera" in question
        and (
            "adjust" in question
            or "interact" in question
        )
    ):
        config["method"] = "clip_oneshot"
        config["scan_fps"] = 0.015625
        config["clip_topk"] = 8
        config["window_len_s"] = 8.0
        config["vlm_budget"] = 2
        config["expand_neighbors"] = False
        config["preserve_order"] = True
        config["include_uniform_anchors"] = False
        config["answer_with_confidence"] = False
        config["skip_cost_clamp"] = True
        config["edge_policy"] = "oracle_camera_budget2"
        return config

    if evidence_type == "localized_object_attribute":
        duration_s = item.get("duration_s")
        try:
            duration_s = float(duration_s)
        except Exception:
            duration_s = get_video_duration_s(item)

        if duration_s >= 60:
            config["method"] = "clip_oneshot"
            config["scan_fps"] = 0.015625
            config["clip_topk"] = 8
            config["window_len_s"] = 4.0
            config["vlm_budget"] = 16
            config["expand_neighbors"] = False
            config["preserve_order"] = True
            config["include_uniform_anchors"] = False
            config["answer_with_confidence"] = False
            config["skip_cost_clamp"] = True
            config["edge_policy"] = "oracle_local_object_budget16"

    return config


UNCERTAIN_ANSWER_TERMS = (
    "cannot determine",
    "can't determine",
    "not enough",
    "insufficient",
    "unclear",
    "ambiguous",
    "not visible",
    "no relevant evidence",
)


def prediction_needs_evidence_fallback(pred):
    if pred.get("prediction_label") is None:
        return True

    confidence = pred.get("prediction_confidence")
    if confidence is not None and confidence < 0.55:
        return True

    text = str(
        pred.get("prediction_text")
        or ""
    ).lower()
    return any(
        term in text
        for term in UNCERTAIN_ANSWER_TERMS
    )


def make_evidence_fallback_config(config, item):
    fallback = dict(config)
    fallback["name"] = (
        config.get("name", "vimio")
        + "_fallback"
    )
    fallback["answer_with_confidence"] = True
    fallback["fallback_source"] = config.get(
        "name",
        "vimio",
    )

    evidence_type = config.get("evidence_type")
    duration_s = item.get("duration_s")
    try:
        duration_s = (
            float(duration_s)
            if duration_s is not None
            else None
        )
    except Exception:
        duration_s = None

    if evidence_type in {
        "sequence_ordering",
        "screen_state_change",
    }:
        fallback["method"] = "clip_map_answer"
        fallback["map_max_windows"] = (
            12
            if evidence_type
            == "sequence_ordering"
            else 10
        )
        fallback["map_frames_per_window"] = (
            3
            if evidence_type
            == "sequence_ordering"
            else 4
        )
        fallback["preserve_order"] = True
        fallback["include_uniform_anchors"] = True
        fallback["vlm_budget"] = max(
            int(
                config.get(
                    "vlm_budget",
                    32,
                )
            ),
            32,
        )
        return fallback

    if evidence_type == "counting_completeness":
        fallback["scan_fps"] = 0.03125
        fallback["clip_topk"] = 8
        fallback["window_len_s"] = 8
        fallback["vlm_budget"] = 32
        fallback["include_uniform_anchors"] = True
        return clamp_vimio_config_cost_aware(
            fallback,
            item,
        )

    if evidence_type in {
        "localized_temporal_detail",
        "localized_object_attribute",
        "first_or_next_event",
    }:
        current_scan = float(
            config.get(
                "scan_fps",
                0.015625,
            )
        )
        fallback["scan_fps"] = (
            0.03125
            if current_scan >= 0.125
            else 0.125
        )
        fallback["clip_topk"] = 8
        fallback["window_len_s"] = 4
        fallback["vlm_budget"] = 32
        fallback["expand_neighbors"] = True
        return clamp_vimio_config_cost_aware(
            fallback,
            item,
        )

    if duration_s is not None and duration_s >= 1200:
        fallback["scan_fps"] = 0.00390625
        fallback["clip_topk"] = 8
        fallback["window_len_s"] = 8
        fallback["vlm_budget"] = 32
    else:
        fallback["scan_fps"] = 0.03125
        fallback["clip_topk"] = 12
        fallback["window_len_s"] = 12
        fallback["vlm_budget"] = 32

    return clamp_vimio_config_cost_aware(
        fallback,
        item,
    )


def run_single_config_with_evidence_fallback(item, config):
    primary = run_single_config(item, config)
    primary["fallback_used"] = False

    if not config.get("enable_evidence_fallback"):
        return primary

    if config.get("method") == "choice_sequence_verifier":
        return primary

    if not prediction_needs_evidence_fallback(primary):
        return primary

    fallback_config = make_evidence_fallback_config(
        config,
        item,
    )

    print(
        "[FALLBACK] weak evidence; rerun "
        f"{fallback_config['method']} "
        f"scan={fallback_config.get('scan_fps')} "
        f"k={fallback_config.get('clip_topk')} "
        f"window={fallback_config.get('window_len_s')} "
        f"budget={fallback_config.get('vlm_budget')}",
        flush=True,
    )

    fallback = run_single_config(
        item,
        fallback_config,
    )
    fallback["fallback_used"] = True
    fallback["primary_prediction_label"] = primary.get(
        "prediction_label"
    )
    fallback["primary_prediction_text"] = primary.get(
        "prediction_text"
    )
    fallback["primary_prediction_confidence"] = primary.get(
        "prediction_confidence"
    )
    fallback["fallback_config"] = {
        key: fallback_config.get(key)
        for key in (
            "method",
            "scan_fps",
            "clip_topk",
            "window_len_s",
            "vlm_budget",
            "evidence_type",
            "map_max_windows",
            "map_frames_per_window",
        )
    }

    if fallback.get("prediction_label") is not None:
        return fallback

    return primary


def clamp_vimio_config_for_long_video(config, item):
    """
    Duration-aware safety clamp for VIMIO.

    On VRBench long videos, high scan_fps caused huge CLIP decode cost.
    The fixed sparse frontier suggests scan0.0156_k8_budget32 works well.
    """
    duration_s = item.get("duration_s")
    bucket = item.get("lvb_duration_bucket") or item.get("duration_bucket")

    if duration_s is None:
        return config

    is_long = duration_s >= 1200 or bucket == "long"

    if is_long:
        config["scan_fps"] = 0.015625
        config["clip_topk"] = 8
        config["window_len_s"] = 8
        config["vlm_budget"] = 32

    return config


def run_experiment(dataset, output, max_examples=None, resume=True, profiler=None):
    examples = load_jsonl(dataset)

    if max_examples is not None:
        examples = examples[:max_examples]

    done = set()
    if resume and Path(output).exists():
        for r in load_jsonl(output):
            done.add((r["qid"], r["config_name"]))


    if profiler is None:
        total = len(examples) * len(EXPERIMENT_CONFIGS)
    else:
        total = len(examples)
    run_idx = 0

    for i, item in enumerate(examples):
        print(f"\nExample {i+1}/{len(examples)} qid={item['qid']}")

        if profiler is None:
            configs = EXPERIMENT_CONFIGS
        else:
            p = profiler.get(item["qid"])

            if p is None:
                print(f"Missing profiler entry for {item['qid']}")
                continue


            config = {
                "name": p.get(
                    "chosen_config",
                    p.get("selected_config", "vimio"),
                ),
                "method": "clip_oneshot",
                "scan_fps": p["probe_fps"],
                "clip_topk": p["probe_topk"],
                "window_len_s": p["window_len_s"],
                "vlm_budget": p["vlm_budget"],
                "evidence_type": p.get("evidence_type"),
                "expand_neighbors": bool(
                    p.get("expand_neighbors")
                ),
                "preserve_order": bool(
                    p.get("preserve_order")
                ),
                "include_uniform_anchors": bool(
                    p.get("include_uniform_anchors")
                ),
                "use_choice_sequence_verifier": bool(
                    p.get("use_choice_sequence_verifier")
                ),
                "scheduler_reason": p.get("scheduler_reason"),
                "scheduler_query_class": p.get("scheduler_query_class"),
                "scheduler_gpu_state": p.get("scheduler_gpu_state"),
                "answer_with_confidence": True,
                "enable_evidence_fallback": bool(
                    p.get("enable_evidence_fallback", True)
                ),
            }

            if config["evidence_type"] == "sequence_ordering":
                if config.get("use_choice_sequence_verifier"):
                    config["method"] = "choice_sequence_verifier"
                else:
                    config = apply_sequence_oracle_clip_policy(
                        config,
                        item,
                    )
                config["map_max_windows"] = (
                    12
                )
                config["map_frames_per_window"] = (
                    3
                )
                config["target_choice_events"] = True
                config["sequence_event_topk"] = config["clip_topk"]
                config["timestamp_frames_per_window"] = 3
                config["sequence_choice_topk"] = min(
                    config["clip_topk"],
                    6,
                )
                config["sequence_choice_max_events"] = 5
                config["sequence_choice_max_windows"] = 6
                config["sequence_choice_frames_per_window"] = 2

            elif config["evidence_type"] == "screen_state_change":
                config["method"] = "clip_map_answer"
                config["map_max_windows"] = 10
                config["map_frames_per_window"] = 4

            elif config["evidence_type"] == "global_process":
                config["method"] = "map_summary"
                config["num_windows"] = 8
                config["frames_per_window"] = 3
                config["query_conditioned"] = False
                config["choice_compare_answer"] = True
                config["structured_evidence_summary"] = True

            config = apply_oracle_edge_policy(
                config,
                item,
            )

            config = clamp_vimio_config_cost_aware(config, item)

            configs = [config]

        for config in configs:
            run_idx += 1

            key = (item["qid"], config["name"])
            if key in done:
                print(f"  [{run_idx}/{total}] skip {config['name']}")
                continue

            print(f"  [{run_idx}/{total}] run {config['name']}")

            start = time.time()

            try:
                pred = run_single_config_with_evidence_fallback(
                    item,
                    config,
                )
                error = None

            except Exception as e:
                import traceback
                traceback.print_exc()

                pred = {
                    "prediction_label": None,
                    "prediction_text": None,
                    "prediction_confidence": None,
                    "num_vlm_calls": None,
                    "evidence": None,
                    "retrieval_effort": None,
                    "fallback_used": False,
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
                "map_max_windows": config.get("map_max_windows"),
                "map_frames_per_window": config.get("map_frames_per_window"),
                "evidence_type": config.get("evidence_type"),
                "expand_neighbors": config.get("expand_neighbors"),
                "preserve_order": config.get("preserve_order"),
                "include_uniform_anchors": config.get("include_uniform_anchors"),
                "prediction_label": prediction_label,
                "prediction_text": pred["prediction_text"],
                "prediction_confidence": pred.get(
                    "prediction_confidence"
                ),
                "correct": prediction_label == item["answer_label"],
                "latency_s": latency_s,
                "num_vlm_calls": pred["num_vlm_calls"],
                "evidence": pred["evidence"],
                "fallback_used": pred.get(
                    "fallback_used",
                    False,
                ),
                "primary_prediction_label": pred.get(
                    "primary_prediction_label"
                ),
                "primary_prediction_text": pred.get(
                    "primary_prediction_text"
                ),
                "primary_prediction_confidence": pred.get(
                    "primary_prediction_confidence"
                ),
                "fallback_config": pred.get(
                    "fallback_config"
                ),
                "scheduler_reason": config.get(
                    "scheduler_reason"
                ),
                "scheduler_query_class": config.get(
                    "scheduler_query_class"
                ),
                "scheduler_gpu_state": config.get(
                    "scheduler_gpu_state"
                ),
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
    

def clamp_vimio_config_cost_aware(config, item):
    """
    Dataset-agnostic VIMIO cost guard.

    It prevents the profiler from selecting very dense retrieval policies
    when the scan cost is too high. This is based only on video duration and
    selected knobs, not dataset name.
    """
    config = dict(config)

    if config.get("skip_cost_clamp"):
        return config

    duration_s = item.get("duration_s")
    if duration_s is None:
        return config

    try:
        duration_s = float(duration_s)
    except Exception:
        return config

    scan_fps = float(config.get("scan_fps", 0.125))
    clip_topk = int(config.get("clip_topk", 8))
    window_len_s = float(config.get("window_len_s", 8))
    vlm_budget = int(config.get("vlm_budget", 32))

    # Estimated retrieval scan frames.
    scan_frames = duration_s * scan_fps

    # Dataset-agnostic scan-frame budget.
    # This is the main knob. It says: do not scan more than this many frames
    # during retrieval unless the video is very short.
    max_scan_frames = 64

    if scan_frames > max_scan_frames:
        config["scan_fps"] = max_scan_frames / duration_s

    # Keep within reasonable supported values / avoid extreme tiny floats.
    allowed_scan_fps = [2.0, 1.0, 0.5, 0.125, 0.03125, 0.015625, 0.00390625]

    # Pick the largest allowed scan_fps that does not exceed the capped value.
    capped = float(config["scan_fps"])
    feasible = [x for x in allowed_scan_fps if x <= capped]

    if feasible:
        config["scan_fps"] = max(feasible)
    else:
        config["scan_fps"] = min(allowed_scan_fps)

    # Also avoid very high VLM budget unless the retrieval is already cheap.
    # This is dataset-agnostic: if retrieval is expensive, don't also increase VLM cost.
    new_scan_frames = duration_s * float(config["scan_fps"])
    if new_scan_frames > 32:
        config["vlm_budget"] = min(vlm_budget, 32)

    return config

def main():
    init_models()
    global EXPERIMENT_CONFIGS

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--profiler_json", default=None,)

    args = parser.parse_args()

    if args.config_file:
        EXPERIMENT_CONFIGS = load_config_file(
            args.config_file
        )
    else:
        raise ValueError(
            "Must provide --config_file"
        )
    
    profiler = None
    if args.profiler_json:
        profiler_rows = load_jsonl(args.profiler_json)
        profiler = {
            r["qid"]: r
            for r in profiler_rows
        }

    run_experiment(
        dataset=args.dataset,
        output=args.output,
        max_examples=args.max_examples,
        resume=not args.no_resume,
        profiler=profiler
    )



if __name__ == "__main__":
    main()
