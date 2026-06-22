# tools.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple
import math
import time

import base64
import io
import json
import os
import time
import tempfile
import requests

from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import clip
_CAPTION_DB = None  # lazy global


def _pil_to_data_url(img: Image.Image, jpeg_quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# ---------------------------
# Caption Index (VideoAgent-style memory)
# ---------------------------

def ocr_window(
    video: str,
    t0: float,
    t1: float,
    stride: float,
    resolution: str,
    *,
    max_frames: int = 8,
    model=None,
    base_url=None,
) -> WindowResult:
    """
    Extract OCR text from sampled frames using an OpenAI-compatible VLM endpoint.
    """
    start = time.time()

    sample_fps = 1.0 / max(1e-6, stride)
    frames: List[Image.Image] = []
    for ts, img in iter_frames(video, fps=sample_fps):
        if ts < t0:
            continue
        if ts > t1:
            break
        frames.append(img)
        if len(frames) >= max_frames:
            break

    if not frames:
        wall = time.time() - start
        return WindowResult(
            t0=t0,
            t1=t1,
            relevance_score=0.0,
            frames_encoded=0,
            dense_seconds=max(0.0, t1 - t0),
            wallclock_s=float(wall),
            evidence={"ocr_text": []},
            source="ocr",
        )

    if not model or not base_url:
        raise ValueError("ocr_window requires both model and base_url")

    content = [
        {
            "type": "text",
            "text": (
                "Extract all visible text from these frames. "
                "Return STRICT JSON ONLY in the format:\n"
                '{"ocr_text": ["...","..."]}'
            ),
        }
    ]

    for img in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _pil_to_data_url(img),
                },
            }
        )

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": content},
        ],
        "max_tokens": 256,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()

    raw = j["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(raw)
        ocr_texts = parsed.get("ocr_text", [])
        if not isinstance(ocr_texts, list):
            ocr_texts = []
    except Exception:
        # fallback: keep raw text if model didn't obey JSON
        ocr_texts = [raw.strip()] if raw.strip() else []

    wall = time.time() - start
    return WindowResult(
        t0=t0,
        t1=t1,
        relevance_score=0.0,
        frames_encoded=len(frames),
        dense_seconds=max(0.0, t1 - t0),
        wallclock_s=float(wall),
        evidence={"ocr_text": ocr_texts, "raw_ocr_response": raw},
        source="ocr",
    )


def _load_caption_db(path: str):
    # Minimal: in-memory list. If big, replace with sqlite/faiss later.
    import json
    db = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            db.append(r)
    return db

def caption_retrieve(
    video: str,
    query: str,
    captions_jsonl: str,
    topk: int = 20,
) -> List[SegmentCandidate]:
    """
    Retrieval over precomputed captions. Cheap compared to dense VLM.
    Returns SegmentCandidates so it plugs into your existing action builder.
    """
    global _CAPTION_DB
    if _CAPTION_DB is None or _CAPTION_DB.get("path") != captions_jsonl:
        _CAPTION_DB = {"path": captions_jsonl, "rows": _load_caption_db(captions_jsonl)}

    rows = _CAPTION_DB["rows"]
    # Filter to current video
    vid_rows = [r for r in rows if r.get("video") == video]

    # Cheap scoring: CLIP text encoder on caption text (or simple token overlap as baseline)
    # We'll use CLIP text encoder since you already have it.
    global _PROBER
    if _PROBER is None:
        _PROBER = CLIPProber()

    # Encode query once
    import torch
    import numpy as np
    import clip

    with torch.no_grad():
        qtok = clip.tokenize([query], truncate=True).to(_PROBER.device)
        qfeat = _PROBER.model.encode_text(qtok)
        qfeat = qfeat / qfeat.norm(dim=-1, keepdim=True)

    scored = []
    # Batch captions to keep it fast
    B = 256
    caps = [r.get("caption", "") for r in vid_rows]
    for i in range(0, len(caps), B):
        batch = caps[i:i+B]
        with torch.no_grad():
            ttok = clip.tokenize(batch, truncate=True).to(_PROBER.device)
            tfeat = _PROBER.model.encode_text(ttok)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            sims = (tfeat @ qfeat.T).squeeze(-1).detach().float().cpu().numpy()

        for j, s in enumerate(sims.tolist()):
            r = vid_rows[i + j]
            scored.append((float(s), float(r["t0"]), float(r["t1"]), r.get("caption", "")))

    scored.sort(key=lambda x: -x[0])
    out = []
    for s, t0, t1, cap in scored[:topk]:
        out.append(SegmentCandidate(t0=t0, t1=t1, score=s))
    return out


@dataclass(frozen=True)
class SegmentCandidate:
    t0: float
    t1: float
    score: float


@dataclass(frozen=True)
class WindowResult:
    t0: float
    t1: float
    relevance_score: float
    frames_encoded: int
    dense_seconds: float
    wallclock_s: float
    evidence: Optional[dict] = None   # NEW
    source: str = "clip"   # "clip"|"ocr"|"asr"|"caption"|"objects"


# ---------------------------
# Video frame sampling backend
# ---------------------------

def _iter_frames_decord(video_path: str, fps: float) -> Iterator[Tuple[float, Image.Image]]:
    """
    Yields (timestamp_seconds, PIL.Image) at approximately `fps`.
    """
    from decord import VideoReader  # type: ignore

    vr = VideoReader(video_path)
    src_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    if fps <= 0:
        raise ValueError("fps must be > 0")

    step = max(1, int(round(src_fps / fps)))
    n = len(vr)

    for idx in range(0, n, step):
        t = idx / src_fps
        frame = vr[idx].asnumpy()  # HWC, RGB
        yield t, Image.fromarray(frame)


def _iter_frames_opencv(video_path: str, fps: float) -> Iterator[Tuple[float, Image.Image]]:
    """
    Yields (timestamp_seconds, PIL.Image) at approximately `fps`.
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        raise ValueError("fps must be > 0")

    step = max(1, int(round(src_fps / fps)))

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / float(src_fps)
            frame_rgb = frame_bgr[:, :, ::-1]  # BGR->RGB
            yield t, Image.fromarray(frame_rgb)
        idx += 1

    cap.release()


def iter_frames(video_path: str, fps: float) -> Iterator[Tuple[float, Image.Image]]:
    """
    Prefer decord but fall back to opencv if decoding fails.
    """

    try:
        import decord

        for item in _iter_frames_decord(video_path, fps):
            yield item
        return

    except Exception as e:
        print(f"[iter_frames] decord failed for {video_path}: {repr(e)} → fallback to opencv", flush=True)

    # fallback
    try:
        for item in _iter_frames_opencv(video_path, fps):
            yield item
    except Exception as e:
        print(f"[iter_frames] opencv also failed for {video_path}: {repr(e)}", flush=True)
        raise

# ---------------------------
# CLIP Prober (OpenAI CLIP)
# ---------------------------

class CLIPProber:
    def __init__(self, device: Optional[str] = None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    def score_frames(self, query: str, frames: List[Image.Image]) -> np.ndarray:
        imgs = torch.stack([self.preprocess(im) for im in frames]).to(self.device)
        text = clip.tokenize([query], truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(imgs)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            sims = (image_features @ text_features.T).squeeze(-1)  # (B,)

        return sims.detach().float().cpu().numpy()

    def probe_video(
        self,
        video_path: str,
        query: str,
        fps: float = 1.0,
        segment_len_s: float = 5.0,
        topk: int = 20,
    ) -> List[SegmentCandidate]:

        try:
            times: List[float] = []
            frames: List[Image.Image] = []

            for t, img in iter_frames(video_path, fps=fps):
                times.append(float(t))
                frames.append(img)

            if not frames:
                return []

            all_scores: List[float] = []
            for i in range(0, len(frames), self.batch_size):
                batch = frames[i:i + self.batch_size]
                sims = self.score_frames(query, batch)
                all_scores.extend([float(x) for x in sims])

            seg_best: dict[int, float] = {}
            for t, s in zip(times, all_scores):
                sid = int(math.floor(t / segment_len_s))
                seg_best[sid] = max(seg_best.get(sid, -1e9), s)

            candidates: List[SegmentCandidate] = []
            for sid, best_score in seg_best.items():
                t0 = sid * segment_len_s
                t1 = t0 + segment_len_s
                candidates.append(
                    SegmentCandidate(t0=t0, t1=t1, score=float(best_score))
                )

            candidates.sort(key=lambda c: (-c.score, c.t0))
            return candidates[:topk]

        except Exception as e:
            print(f"[probe_video] Skipping bad video: {video_path}")
            print(f"  Reason: {repr(e)}")
            return []

# ---------------------------
# Public tool functions
# ---------------------------

_PROBER: Optional[CLIPProber] = None


def probe_index(video: str, query: str, fps: float = 1.0, segment_len_s: float = 5.0, topk: int = 20):
    global _PROBER
    if _PROBER is None:
        _PROBER = CLIPProber()
    return _PROBER.probe_video(video_path=video, query=query, fps=fps, segment_len_s=segment_len_s, topk=topk)


from typing import List, Optional
import time
import math
from PIL import Image

def _frames_in_window_decord(video_path: str, t0: float, t1: float, sample_fps: float) -> List[Image.Image]:
    """
    Random-access sampling with decord: only decode frames in [t0, t1].
    """
    from decord import VideoReader  # type: ignore
    import numpy as np

    vr = VideoReader(video_path)
    src_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    n = len(vr)

    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    # choose frame indices at desired sample_fps
    start_idx = max(0, int(math.floor(t0 * src_fps)))
    end_idx = min(n - 1, int(math.ceil(t1 * src_fps)))

    # step in source-frame units to approximate sample_fps
    step = max(1, int(round(src_fps / sample_fps)))

    idxs = list(range(start_idx, end_idx + 1, step))
    if not idxs:
        return []

    # batch decode
    batch = vr.get_batch(idxs).asnumpy()  # (B, H, W, 3), RGB
    return [Image.fromarray(frame) for frame in batch]


def _frames_in_window_opencv(video_path: str, t0: float, t1: float, sample_fps: float, max_frames: Optional[int] = None) -> List[Image.Image]:
    """
    Seek + decode with OpenCV: only decode frames in [t0, t1].
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    # seek near t0
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t0) * 1000.0)

    step = max(1, int(round(src_fps / sample_fps)))
    frames: List[Image.Image] = []

    # We start reading from the seek point; use time from CAP_PROP_POS_MSEC
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # current timestamp (seconds)
        ts = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if ts > t1:
            break

        # sample every `step` frames by skipping reads
        # (OpenCV doesn't give us the frame index reliably after seeking, so we skip by grabbing extra frames)
        frame_rgb = frame_bgr[:, :, ::-1]
        frames.append(Image.fromarray(frame_rgb))

        if max_frames is not None and len(frames) >= max_frames:
            break

        # skip step-1 frames quickly
        for _ in range(step - 1):
            ok2 = cap.grab()
            if not ok2:
                break

    cap.release()
    return frames


import subprocess

def has_audio_stream(video_path: str) -> bool:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return bool(r.stdout.strip())

def inspect_window(
    video: str,
    t0: float,
    t1: float,
    stride: float,
    resolution: str,
    query: Optional[str] = None,
    source: str = "clip",
):
    global _PROBER
    if _PROBER is None:
        _PROBER = CLIPProber()

    start = time.time()

    duration = max(0.0, t1 - t0)
    sample_fps = 1.0 / max(1e-6, stride)

    # decode only frames in the window (fast)
    try:
        frames = _frames_in_window_decord(video, t0, t1, sample_fps=sample_fps)
    except Exception as e:
        # fallback to OpenCV seek
        # print(f"[inspect_window] decord window decode failed: {repr(e)} -> fallback opencv", flush=True)
        frames = _frames_in_window_opencv(video, t0, t1, sample_fps=sample_fps)

    frames_encoded = len(frames)

    relevance = 0.0
    if frames and query:
        sims: List[float] = []
        for i in range(0, len(frames), _PROBER.batch_size):
            sims.extend(_PROBER.score_frames(query, frames[i:i + _PROBER.batch_size]).tolist())
        relevance = float(max(sims)) if sims else 0.0

    wall = time.time() - start

    return WindowResult(
        t0=t0,
        t1=t1,
        relevance_score=relevance,
        frames_encoded=frames_encoded,
        dense_seconds=duration,
        wallclock_s=float(wall),
        evidence=None,
        source=source,
    )

# keep your existing imports / WindowResult definition


def _extract_audio_clip_ffmpeg(video_path: str, t0: float, t1: float) -> str:
    """
    Extract [t0, t1] audio into a temporary wav file.
    Returns the temp file path.
    """
    import subprocess

    duration = max(0.0, t1 - t0)
    if duration <= 0:
        raise ValueError(f"Invalid audio clip duration: {duration}")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(t0),
        "-i", video_path,
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def asr_window(
    video: str,
    t0: float,
    t1: float,
    stride: float,
    resolution: str,
    *,
    query: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: float = 120.0,
) -> WindowResult:
    """
    Extract ASR text from a video window using an OpenAI-compatible
    /v1/audio/transcriptions endpoint.

    Returns a WindowResult with evidence={"asr_text": "..."}.
        evidence["asr_text"]
        evidence["raw_asr_response"]
        evidence["timing"] = {
            "extract_s" : .....,
            "request_s" : .....,
            "total_s": .....,        
        }
    """
    total_start = time.time()

    if not model or not base_url:
        raise ValueError("asr_window requires both model and base_url")

    audio_path = None
    try:
        #1) Extract audio clip
        extract_start = time.time()
        audio_path = _extract_audio_clip_ffmpeg(video, t0, t1)
        extract_end = time.time()

        print("ASR TEMP AUDIO:", audio_path)
        print("ASR WINDOW:", t0, t1)
        print("ASR AUDIO SIZE:", os.path.getsize(audio_path))

        url = base_url.rstrip("/") + "/audio/transcriptions"

        #2) Send chunk to ASR server
        with open(audio_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_path), f, "audio/wav"),
            }
            data = {
                "model": model,
            }

            request_start = time.time()
            r = requests.post(url, files=files, data=data, timeout=timeout_s)
            request_end = time.time()

            if not r.ok:
                print("ASR STATUS:", r.status_code)
                print("ASR RESPONSE TEXT:", r.text)
                print("ASR URL:", url)
                print("ASR MODEL:", model)
                print("ASR AUDIO PATH:", audio_path)
                print("ASR EXTRACT TIME S:", extract_end - extract_start)
                print("ASR REQUEST TIME S:", request_end - request_start)
                raise RuntimeError(f"ASR request failed: {r.status_code} {r.text}")

            j = r.json()

        #3) Parse transcript
        asr_text = (j.get("text") or "").strip()
        if not isinstance(asr_text, str):
            asr_text = json.dumps(j)

        total_end = time.time()

        extract_s = float(extract_end - extract_start)
        request_s = float(request_end - request_start)
        total_s = float(total_end - total_start)

        print("ASR EXTRACT TIME S:", extract_s)
        print("ASR REQUEST+TRANSCRIBE TIME S:", request_s)
        print("ASR TOTAL TIME S:", total_s)

        return WindowResult(
            t0=t0,
            t1=t1,
            relevance_score=0.0,
            frames_encoded=0,
            dense_seconds=max(0.0, t1 - t0),
            wallclock_s=total_s,
            evidence={
                "asr_text": asr_text,
                "raw_asr_response": j,
                "timing": {
                    "extract_s": extract_s,
                    "request_s": request_s,
                    "total_s": total_s,
                },
            },
            source="asr",
        )

    finally:
        print("KEEPING ASR TEMP AUDIO FOR DEBUG:", audio_path)