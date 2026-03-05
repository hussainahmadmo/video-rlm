# tools.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple
import math
import time

import numpy as np
from PIL import Image

import torch
import clip
_CAPTION_DB = None  # lazy global



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
) -> WindowResult:
    """
    Extract OCR text from sampled frames. Cheap-ish.
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

    # TODO: replace with actual OCR
    ocr_texts = []  # list[str]
    # Example placeholder: empty

    wall = time.time() - start
    return WindowResult(
        t0=t0, t1=t1,
        relevance_score=0.0,
        frames_encoded=len(frames),
        dense_seconds=max(0.0, t1 - t0),
        wallclock_s=float(wall),
        evidence={"ocr_text": ocr_texts},
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
    source: str = "clip"              # NEW



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


def inspect_window(video: str, t0: float, t1: float, stride: float, resolution: str, query: Optional[str] = None):
    global _PROBER
    if _PROBER is None:
        _PROBER = CLIPProber()

    start = time.time()
    duration = max(0.0, t1 - t0)
    frames_encoded = max(1, int(duration / max(1e-6, stride)))

    sample_fps = 1.0 / max(1e-6, stride)

    frames: List[Image.Image] = []
    for ts, img in iter_frames(video, fps=sample_fps):
        if ts < t0:
            continue
        if ts > t1:
            break
        frames.append(img)

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
        evidence = None,  # <-- NEW (text/tags/etc.)
        source = "clip"              # <-- NEW ("clip"|"ocr"|"asr"|"caption"|"objects")

    )