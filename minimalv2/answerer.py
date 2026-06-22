# answerer.py
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import requests
from PIL import Image

from tools import iter_frames

import os
from pathlib import Path
from typing import Literal
import math

Mode = Literal["attribute", "distributed", "ordering", "causal", "microevent"]


def _extract_json(text: str) -> dict:
    text = text.strip()

    if not text:
        raise ValueError("Empty model output")

    try:
        return json.loads(text)
    except Exception:
        pass

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()

    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        candidate = text[i:j+1]
        try:
            return json.loads(candidate)
        except Exception:
            cleaned = (
                candidate
                .replace("True", "true")
                .replace("False", "false")
                .replace("None", "null")
            )
            return json.loads(cleaned)

    raise ValueError(f"Could not parse JSON from model output: {text[:500]}")

def _pil_to_data_url(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _safe(s: str, max_len: int = 80) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
    return s[:max_len]


def _make_grid(images: List[Image.Image], cols: int = 2, tile: int = 336) -> Image.Image:
    if not images:
        return Image.new("RGB", (tile, tile), (0, 0, 0))

    imgs = [im.convert("RGB").resize((tile, tile), Image.BILINEAR) for im in images]
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile, rows * tile), (0, 0, 0))
    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        grid.paste(im, (c * tile, r * tile))
    return grid


def sample_window_frames(
    video_path: str,
    t0: float,
    t1: float,
    *,
    sample_fps: float = 1.0,
    max_frames: int = 4,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    picked_ts: List[float] = []

    for ts, img in iter_frames(video_path, fps=sample_fps):
        if ts < t0:
            continue
        if ts > t1:
            break
        frames.append(img)
        picked_ts.append(ts)
        if len(frames) >= max_frames:
            break

    print(
        f"[sample_window_frames] video={video_path} "
        f"window=({t0:.1f}, {t1:.1f}) fps={sample_fps} "
        f"max_frames={max_frames} picked_ts={picked_ts}"
        f"picked_ts={picked_ts}"

    )
    return frames

def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale <= 1.0:
        return img
    nw, nh = int(w / scale), int(h / scale)
    return img.resize((nw, nh), Image.BILINEAR)

def build_vlm_images_for_window(
    frames: List[Image.Image],
    *,
    mode: Mode,
    tile: int = 336,
    cols: int = 2,
    hi_res: int = 672,
    crop_for_objects: bool = True,
) -> List[Image.Image]:
    """
    Returns a list of PIL images to send for ONE window.
    - microevent: return ONE collage grid (context)
    - attribute/distributed: return ONE high-res (detail), optionally bottom-cropped
    """
    if mode == "microevent":
        grid = _make_grid(frames, cols=cols, tile=tile)
        return [grid]

    # attribute / distributed / ordering / causal:
    img = _center_frame(frames)
    if crop_for_objects and mode in ("attribute", "distributed"):
        # Most objects are on tables / lower frame in LVB-style data
        img = _crop_bottom_half(img)

    # Make it high-res enough to see small objects
    img = _downscale(img, hi_res)
    # Optionally force a square for consistent vision tokens:
    img = img.resize((hi_res, hi_res), Image.BILINEAR)
    return [img]

def _center_frame(frames: List[Image.Image]) -> Image.Image:
    if not frames:
        return Image.new("RGB", (336, 336), (0, 0, 0))
    return frames[len(frames) // 2].convert("RGB")

def _crop_bottom_half(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((0, h // 2, w, h))

@dataclass(frozen=True)
class VLLMAnswererConfig:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 64
    temperature: float = 0.0
    timeout_s: int = 300


class VLLMAnswerer:
    """
    OpenAI-compatible vLLM answerer: POST {base_url}/chat/completions
    """

    def __init__(self, cfg: VLLMAnswererConfig):
        self.cfg = cfg
        root = cfg.base_url.rstrip("/")
        # allow base_url to be passed as ... or .../v1
        if root.endswith("/v1"):
            root = root[:-3]
        self.endpoint = root + "/v1/chat/completions"

    def answer(self,
        *,
        video_path: str,
        windows: List[Tuple[float, float]],
        question: str,
        mode: Mode = "attribute",
        sample_fps: float = 1.0,
        max_frames_per_window: int = 4,
        save_collages_dir: str | None = "debug_collages",
        # hard caps to avoid token/prompt blowups:
        max_windows: int = 2,
        max_images_total: int = 4,
        jpeg_quality: int = 85,) -> str:
        # One collage per window to bound image tokens
        cap = min(max_windows, max_images_total)
        #appy cap to the window list
        windows = windows[:cap]
        collages: List[str] = []

        save_dir = None
        if save_collages_dir is not None:
            save_dir = Path(save_collages_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

    
        for idx, (t0, t1) in enumerate(windows):
            frames = sample_window_frames(
                video_path, t0, t1,
                sample_fps=sample_fps,
                max_frames=max_frames_per_window,
            )
            grid = _make_grid(frames, cols=2, tile=336)
            if save_dir is not None:
                vid = _safe(Path(video_path).stem)
                q = _safe(question)
                fname = f"{vid}__w{idx:02d}__{t0:.1f}-{t1:.1f}__{q}.jpg"
                out_path = save_dir / fname
                grid.save(out_path, format="JPEG", quality=90)
                # optional: print so you see it in logs/debugger
                # print("saved collage:", out_path)

            collages.append(_pil_to_data_url(grid))


        content = [{
                    "type": "text","text": (f"Question: {question}\n"
                    "Return ONLY the answer. No window labels, no explanations."
                    ),
                }]
        
        for (t0, t1), data_url in zip(windows, collages):
            content.append({"type": "text", "text": f"Frames from {t0:.1f}s to {t1:.1f}s"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": "Answer using the provided frames. Output ONLY the final answer text."},
                {"role": "user", "content": content},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

        r = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.cfg.timeout_s,
        )
        if r.status_code >= 400:
            print("vLLM status:", r.status_code)
            print("vLLM response:", r.text[:8000])  # prints the actual reason
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    

    def answer_with_confidence(
        self,
        *,
        video_path: str,
        windows: List[Tuple[float, float]],
        question: str,
        mode: Mode = "attribute",
        sample_fps: float = 1.0,
        max_frames_per_window: int = 4,
        save_collages_dir: str | None = "debug_collages",
        max_windows: int = 2,
        max_images_total: int = 4,
        jpeg_quality: int = 85,) -> tuple[str, float, dict]:
        """
        Returns (answer, confidence, raw_json).
        Confidence is model-reported [0,1] probability that answer is correct given frames.
        """

        cap = min(max_windows, max_images_total)
        windows = windows[:cap]
        collages: List[str] = []

        save_dir = None
        if save_collages_dir is not None:
            save_dir = Path(save_collages_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for idx, (t0, t1) in enumerate(windows):
            frames = sample_window_frames(
                video_path, t0, t1,
                sample_fps=sample_fps,
                max_frames=max_frames_per_window,
            )
            grid = _make_grid(frames, cols=2, tile=336)

            if save_dir is not None:
                vid = _safe(Path(video_path).stem)
                q = _safe(question)
                fname = f"{vid}__w{idx:02d}__{t0:.1f}-{t1:.1f}__{q}.jpg"
                out_path = save_dir / fname
                grid.save(out_path, format="JPEG", quality=90)

            collages.append(_pil_to_data_url(grid, quality=jpeg_quality))

        # IMPORTANT: ask for JSON, not plain text
        content = [{
            "type": "text",
            "text": (
                f"Question: {question}\n"
                "Return STRICT JSON ONLY (no markdown, no extra text) with keys:\n"
                '  {"answer": <string>, "confidence": <number 0..1>}\n'
                "Confidence = how likely the answer is correct GIVEN ONLY the provided frames.\n"
                "If the frames are insufficient/unclear, still guess an answer but set confidence low."
            ),
        }]

        for (t0, t1), data_url in zip(windows, collages):
            content.append({"type": "text", "text": f"Frames from {t0:.1f}s to {t1:.1f}s"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a visual question answering model. Output STRICT JSON only."},
                {"role": "user", "content": content},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

        r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        if r.status_code >= 400:
            print("vLLM status:", r.status_code)
            print("vLLM response:", r.text[:8000])
        r.raise_for_status()

        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()

        try:
            raw = _extract_json(text)
            ans = str(raw.get("answer", "")).strip()
            conf = raw.get("confidence", 0.0)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            return ans, conf, raw
        except Exception as e:
            print("VLM answer parse failed:", e)
            print("RAW VLM TEXT:", repr(text))
            ans, conf = _extract_answer_conf_fallback(text)
            return ans, conf, {
                "error": str(e),
                "raw_text": text[:1000],
                "fallback_parsed_answer": ans,
                "fallback_parsed_confidence": conf,
            }



import re

def _extract_answer_conf_fallback(text: str) -> tuple[str, float]:
    t = text.strip()

    if not t:
        return "", 0.0

    # pattern like: "The answer is square, confidence level: 0.9"
    m = re.search(
        r"answer\s+is\s+(.*?)(?:,|\n|$).*?confidence(?:\s+level)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        ans = m.group(1).strip().strip('"').strip("'")
        conf = float(m.group(2))
        conf = max(0.0, min(1.0, conf))
        return ans, conf

    # pattern like: "answer: square"
    m2 = re.search(r"answer\s*[:=]\s*(.*)", t, flags=re.IGNORECASE)
    if m2:
        ans = m2.group(1).strip().strip('"').strip("'")
        return ans, 0.0

    return t, 0.0