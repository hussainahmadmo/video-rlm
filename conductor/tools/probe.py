from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, List, Tuple
import math

import numpy as np
from PIL import Image

import torch
import clip


@dataclass(frozen=True)
class SegmentCandidate:
    t0: float
    t1: float
    score: float


def _iter_frames_decord(video_path: str, fps: float) -> Iterator[Tuple[float, Image.Image]]:
    from decord import VideoReader

    vr = VideoReader(video_path)
    src_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0

    if fps <= 0:
        raise ValueError("fps must be > 0")

    step = max(1, int(round(src_fps / fps)))
    n = len(vr)

    for idx in range(0, n, step):
        t = idx / src_fps
        frame = vr[idx].asnumpy()
        yield t, Image.fromarray(frame)


def _iter_frames_opencv(video_path: str, fps: float) -> Iterator[Tuple[float, Image.Image]]:
    import cv2

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
            frame_rgb = frame_bgr[:, :, ::-1]
            yield t, Image.fromarray(frame_rgb)

        idx += 1

    cap.release()


def iter_frames(video_path: str, fps: float) -> Iterator[Tuple[float, Image.Image]]:
    try:
        for item in _iter_frames_decord(video_path, fps):
            yield item
        return
    except Exception as e:
        print(f"[iter_frames] decord failed: {repr(e)} -> fallback to opencv", flush=True)

    for item in _iter_frames_opencv(video_path, fps):
        yield item


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

            sims = (image_features @ text_features.T).squeeze(-1)

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


_PROBER: Optional[CLIPProber] = None


def probe_index(
    video: str,
    query: str,
    fps: float = 1.0,
    segment_len_s: float = 5.0,
    topk: int = 20,
) -> List[SegmentCandidate]:
    global _PROBER

    if _PROBER is None:
        _PROBER = CLIPProber()

    return _PROBER.probe_video(
        video_path=video,
        query=query,
        fps=fps,
        segment_len_s=segment_len_s,
        topk=topk,
    )