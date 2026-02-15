# -----------------------------
# Environment (the "video memory")
# -----------------------------
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import time
from dataclasses import dataclass, asdict

import torch
import numpy as np
import cv2

from data_structures import Segment, RefineResult, SearchResult
from video_rlm_minimal import get_video_fps_and_duration, iter_frames_at_fps
from clip_encoder import Clip


@dataclass
class EvidenceItem:
    t: float
    score: float
    kind: str              # "peak" | "top_frame" | "window"
    note: str = ""

@dataclass
class ContextSlice:
    tool: str              # "search_segments" | "refine_in_segment" | "inspect_window"
    query: str
    window: Dict[str, Any] # e.g., {"seg_idx": 3, "t0": 60.0, "t1": 120.0}
    evidence: List[EvidenceItem]
    summary: str
    stats: Dict[str, Any]

class VideoEnv:
    """
    Holds precomputed cheap signals & embeddings.
    Minimal actions:
      - search_segments
      - refine_in_segment
    """

    def __init__(self, video_path: str, clip: Clip, seg_len_s: float = 60.0, base_fps: float = 1.0):
        self.video_path = video_path
        self.clip = clip
        self.seg_len_s = seg_len_s
        self.base_fps = base_fps
        self.native_fps, self.duration = get_video_fps_and_duration(video_path)

        self.base_ts: List[float] = []
        self.base_motion: List[float] = []
        self.base_clip: Optional[torch.Tensor] = None  # (N,D)
        self.segments: List[Segment] = []

        self._text_cache: Dict[str, torch.Tensor] = {}      # query -> (D,) CPU
        self._inspect_cache: Dict[Tuple, Dict[str, Any]] = {}  # (query,t0,t1,fps,top_m) -> result dict
        self._refine_cache: Dict[Tuple, RefineResult] = {}     # (query,seg_idx,dense_fps,window_s) -> result

        self.trace: List[Dict[str, Any]] = []

    def _encode_text_cached(self, query: str) -> torch.Tensor:
        q = self._text_cache.get(query)
        if q is None:
            q = self.clip.encode_text(query)   # ✅ correct
            self._text_cache[query] = q
        return q

    def act(self, call: Dict[str, Any]) -> ContextSlice:
        """
        Unified tool interface for the controller.
        call example:
        {"tool":"search_segments","query":"...", "top_k":5}
        Returns a ContextSlice (compact + structured).
        """
        tool = call["tool"]
        t_start = time.time()

        if tool == "search_segments":
            query = call["query"]
            top_k = int(call.get("top_k", 5))
            out = self.search_segments(query, top_k=top_k)

            if not out:
                cs = ContextSlice(
                    tool="search_segments",
                    query=query,
                    window={"seg_len_s": self.seg_len_s, "duration": self.duration},
                    evidence=[],
                    summary="No segments returned (did you call build_index()? is the video readable?).",
                    stats={"top_k": top_k, "n_segments": len(self.segments)},
                )
            else:
                evidence = [
                    EvidenceItem(t=r.t0, score=r.score, kind="window",
                                note=f"seg={r.seg_idx}, window=[{r.t0:.2f},{r.t1:.2f}]")
                    for r in out
                ]
                summary = f"Top-{len(out)} segments for query='{query}'. Best seg={out[0].seg_idx} score={out[0].score:.3f}."
                cs = ContextSlice(
                    tool="search_segments",
                    query=query,
                    window={"seg_len_s": self.seg_len_s, "duration": self.duration},
                    evidence=evidence,
                    summary=summary,
                    stats={"top_k": top_k, "n_segments": len(self.segments)},
                )

            evidence = [
                EvidenceItem(t=r.t0, score=r.score, kind="window",
                            note=f"seg={r.seg_idx}, window=[{r.t0:.2f},{r.t1:.2f}]")
                for r in out
            ]
            summary = f"Top-{len(out)} segments for query='{query}'. Best seg={out[0].seg_idx} score={out[0].score:.3f}."
            cs = ContextSlice(
                tool="search_segments",
                query=query,
                window={"seg_len_s": self.seg_len_s, "duration": self.duration},
                evidence=evidence,
                summary=summary,
                stats={"top_k": top_k, "n_segments": len(self.segments)},
            )

        elif tool == "refine_in_segment":
            query = call["query"]
            seg_idx = int(call["seg_idx"])
            dense_fps = float(call.get("dense_fps", 8.0))
            window_s = float(call.get("window_s", 2.0))

            key = (query, seg_idx, dense_fps, window_s)
            if key in self._refine_cache:
                rr = self._refine_cache[key]
            else:
                rr = self.refine_in_segment(query, seg_idx=seg_idx, dense_fps=dense_fps, window_s=window_s)
                if rr is not None:
                    self._refine_cache[key] = rr

            if rr is None:
                cs = ContextSlice(
                    tool="refine_in_segment",
                    query=query,
                    window={"seg_idx": seg_idx},
                    evidence=[],
                    summary="No frames sampled / could not refine.",
                    stats={"dense_fps": dense_fps, "window_s": window_s},
                )
            else:
                evidence = [EvidenceItem(t=rr.t0, score=rr.score, kind="window", note=f"[{rr.t0:.2f},{rr.t1:.2f}]")]
                cs = ContextSlice(
                    tool="refine_in_segment",
                    query=query,
                    window={"seg_idx": seg_idx, "t0": rr.t0, "t1": rr.t1},
                    evidence=evidence,
                    summary=f"Refined seg={seg_idx} to [{rr.t0:.2f},{rr.t1:.2f}] score={rr.score:.3f}.",
                    stats={"dense_fps": dense_fps, "window_s": window_s},
                )

        elif tool == "inspect_window":
            query = call["query"]
            t0 = float(call["t0"])
            t1 = float(call["t1"])
            fps = float(call.get("fps", 4.0))
            top_m = int(call.get("top_m", 5))

            key = (query, round(t0, 3), round(t1, 3), fps, top_m)
            if key in self._inspect_cache:
                res = self._inspect_cache[key]
            else:
                res = self.inspect_window(query, t0=t0, t1=t1, fps=fps, top_m=top_m)
                self._inspect_cache[key] = res

            evidence = []
            for p in res.get("peaks", []):
                evidence.append(EvidenceItem(t=float(p["t"]), score=float(p["score"]), kind="peak"))
            for f in res.get("top_frames", []):
                evidence.append(EvidenceItem(t=float(f["t"]), score=float(f["score"]), kind="top_frame"))

            cs = ContextSlice(
                tool="inspect_window",
                query=query,
                window={"t0": res["t0"], "t1": res["t1"]},
                evidence=evidence[:max(top_m, 1)],
                summary=res.get("summary", ""),
                stats={"fps": fps, "top_m": top_m, "n_samples": res.get("n_samples", 0), "score_stats": res.get("score_stats")},
            )

        else:
            raise ValueError(f"Unknown tool: {tool}")

        dt = time.time() - t_start
        self.trace.append({
            "tool": tool,
            "call": call,
            "dt_s": dt,
            "summary": cs.summary,
            "stats": cs.stats,
        })

        
        return cs

    def context_to_dict(self, cs: ContextSlice) -> Dict[str, Any]:
        """
        Convert ContextSlice (dataclasses) into pure JSON-serializable dict.
        This is what you pass to the LLM controller.
        """
        return {
            "tool": cs.tool,
            "query": cs.query,
            "window": cs.window,
            "evidence": [asdict(e) for e in cs.evidence],
            "summary": cs.summary,
            "stats": cs.stats,
        }




    def refine_in_segment(
        self,
        query: str,
        seg_idx: int,
        dense_fps: float = 8.0,
        window_s: float = 2.0
    ) -> Optional[RefineResult]:
        """
        Dense-sample inside a coarse segment, then return the best-scoring time window
        of length ~window_s (by average CLIP similarity).

        Fix: sampling uses a *local* counter (k) instead of CAP_PROP_POS_FRAMES,
        which is global and makes sampling phase-dependent on where the segment starts.
        """
        if seg_idx < 0 or seg_idx >= len(self.segments):
            raise IndexError(f"seg_idx {seg_idx} out of range (n={len(self.segments)})")
        
        q = self._encode_text_cached(query)
        seg = self.segments[seg_idx]

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.video_path}")

        # Seek to segment start
        cap.set(cv2.CAP_PROP_POS_MSEC, seg.t0 * 1000.0)

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Sample every `step` frames to approximate dense_fps
        step = max(1, int(round(native_fps / max(dense_fps, 1e-6))))

        frames: List[np.ndarray] = []
        ts: List[float] = []

        k = 0  # local frame counter within this segment read loop
        while True:
            cur_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if cur_t >= seg.t1:
                break

            ok, fr = cap.read()
            if not ok:
                break

            if (k % step) == 0:
                frames.append(fr)
                ts.append(cur_t)
            k += 1

        cap.release()

        if not frames:
            return None

        emb = self.clip.encode_images_bgr(frames)   # (N,D) on CPU
        sim = (emb @ q).numpy()                     # (N,)

        # Best average similarity in any window of length window_s
        best_score = -1e9
        best_t0 = ts[0]
        best_t1 = ts[0]

        # Two-pointer window for efficiency
        j = 0
        for i in range(len(ts)):
            # advance j while inside window
            while j < len(ts) and ts[j] <= ts[i] + window_s:
                j += 1
            if j > i:
                avg = float(sim[i:j].mean())
                if avg > best_score:
                    best_score = avg
                    best_t0 = ts[i]
                    best_t1 = ts[j - 1]

        return RefineResult(t0=best_t0, t1=best_t1, score=best_score, seg_idx=seg_idx)

    def build_index(self):
        # 1) sample frames at base fps and compute motion + clip
        prev_gray = None
        frames = []
        ts = []
        motion = []

        for t, fr in iter_frames_at_fps(self.video_path, self.base_fps):
            ts.append(t)
            frames.append(fr)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                m = 0.0
            else:
                m = float(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray
            motion.append(m)

        self.base_ts = ts
        self.base_motion = motion
        self.base_clip = self.clip.encode_images_bgr(frames)  # (N,D) CPU

        # 2) pool into segments
        nseg = int(np.ceil(self.duration / self.seg_len_s))
        for i in range(nseg):
            t0 = i * self.seg_len_s
            t1 = min(self.duration, (i + 1) * self.seg_len_s)

            idxs = [j for j, t in enumerate(self.base_ts) if (t0 <= t < t1)]
            if idxs:
                emb = self.base_clip[idxs].mean(dim=0)
                emb = emb / (emb.norm() + 1e-8)
                m = float(np.mean([self.base_motion[j] for j in idxs]))
            else:
                emb = torch.zeros_like(self.clip.encode_text("x"))
                m = 0.0

            self.segments.append(Segment(seg_idx=i, t0=t0, t1=t1, clip_emb=emb, motion_mean=m))

    def search_segments(self, query: str, top_k: int = 5) -> List[SearchResult]:
        q = self._encode_text_cached(query)
        scores = []
        for s in self.segments:
            sc = float((s.clip_emb @ q).item())
            scores.append(SearchResult(seg_idx=s.seg_idx, t0=s.t0, t1=s.t1, score=sc))
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]
    

    def inspect_window(
        self,
        query: str,
        t0: float,
        t1: float,
        fps: float = 4.0,
        top_m: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Sample frames in [t0, t1] at ~fps, compute per-frame CLIP similarity to query,
        and return top evidence timestamps (peaks) + basic stats.
        """
        # Clamp window
        t0 = max(0.0, float(t0))
        t1 = min(float(t1), float(self.duration))
        if t1 <= t0:
            return {"t0": t0, "t1": t1, "fps": float(fps), "n_samples": 0, "peaks": [], "top_frames": [],
                    "score_stats": None, "summary": "Empty/invalid window."}

        q = self._encode_text_cached(query)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.video_path}")

        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / max(float(fps), 1e-6))))

        frames: List[np.ndarray] = []
        ts: List[float] = []
        k = 0
        while True:
            cur_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if cur_t >= t1:
                break
            ok, fr = cap.read()
            if not ok:
                break
            if (k % step) == 0:
                frames.append(fr)
                ts.append(cur_t)
            k += 1
        cap.release()

        if not frames:
            return {"t0": t0, "t1": t1, "fps": float(fps), "n_samples": 0, "peaks": [], "top_frames": [],
                    "score_stats": None, "summary": "No frames sampled (decoder/window issue)."}

        emb = self.clip.encode_images_bgr(frames, batch_size=batch_size)  # (N,D)
        sim = (emb @ q).numpy().astype(np.float32)                        # (N,)

        smin = float(sim.min())
        smean = float(sim.mean())
        smax = float(sim.max())

        # Top frames by score (always exists)
        idx_sorted = np.argsort(-sim)
        top_frames = [{"t": float(ts[int(i)]), "score": float(sim[int(i)])}
                    for i in idx_sorted[:max(top_m, 1)]]

        # Local maxima peaks
        peaks_idx: List[int] = []
        if len(sim) == 1:
            peaks_idx = [0]
        else:
            for i in range(1, len(sim) - 1):
                if sim[i] >= sim[i - 1] and sim[i] >= sim[i + 1]:
                    peaks_idx.append(i)
            if sim[0] > sim[1]:
                peaks_idx.append(0)
            if sim[-1] > sim[-2]:
                peaks_idx.append(len(sim) - 1)

        peaks_idx = sorted(peaks_idx, key=lambda i: float(sim[i]), reverse=True)[:max(top_m, 1)]
        peaks = [{"t": float(ts[i]), "score": float(sim[i])} for i in peaks_idx]

        summary = (
            f"Sampled {len(ts)} frames at ~{fps} FPS in [{t0:.2f},{t1:.2f}]s. "
            f"sim mean={smean:.3f}, max={smax:.3f}. "
            f"Top evidence near t={top_frames[0]['t']:.2f}s."
        )

        return {
            "t0": t0,
            "t1": t1,
            "fps": float(fps),
            "n_samples": int(len(ts)),
            "peaks": peaks,
            "top_frames": top_frames,
            "score_stats": {"min": smin, "mean": smean, "max": smax},
            "summary": summary,
        }