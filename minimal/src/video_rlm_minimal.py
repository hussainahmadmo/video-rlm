from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import time
import json
import numpy as np
import cv2

# -----------------------------
# Video helpers
# -----------------------------

def get_video_fps_and_duration(path: str) -> Tuple[float, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(fps), float(nframes / fps) if fps > 0 else 0.0

def iter_frames_at_fps(path: str, sample_fps: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / sample_fps)))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield (i / native_fps), fr
        i += 1
    cap.release()


