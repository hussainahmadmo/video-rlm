from __future__ import annotations

import time
from typing import Any


class OCRClient:
    def __init__(
        self,
        model_name: str = "ocr-model",
        enable_frame_cache: bool = True,
        enable_result_cache: bool = True,
    ):
        self.model_name = model_name
        self.enable_frame_cache = enable_frame_cache
        self.enable_result_cache = enable_result_cache

        self._frame_cache: dict[tuple, Any] = {}
        self._result_cache: dict[tuple, dict[str, Any]] = {}

        self._frame_cache_hits = 0
        self._frame_cache_misses = 0
        self._result_cache_hits = 0
        self._result_cache_misses = 0
        self._model_calls = 0

        self._timing = {
            "frame_decode_s": 0.0,
            "frame_sample_s": 0.0,
            "result_cache_lookup_s": 0.0,
            "result_cache_hit_return_s": 0.0,
            "model_inference_s": 0.0,
        }

    def _result_cache_key(
        self,
        video_path: str,
        t0: float,
        t1: float,
        fps: float,
        resolution: int | None,
    ) -> tuple:
        return (
            video_path,
            round(t0, 3),
            round(t1, 3),
            fps,
            resolution,
            self.model_name,
        )

    def _frame_cache_key(
        self,
        video_path: str,
        t0: float,
        t1: float,
        fps: float,
        resolution: int | None,
    ) -> tuple:
        return (
            video_path,
            round(t0, 3),
            round(t1, 3),
            fps,
            resolution,
        )

    def reset_timing(self) -> None:
        for k in self._timing:
            self._timing[k] = 0.0

    def get_timing_summary(self) -> dict[str, float]:
        return dict(self._timing)

    def get_cache_summary(self) -> dict[str, int | float]:
        total = self._result_cache_hits + self._result_cache_misses
        return {
            "frame_cache_hits": self._frame_cache_hits,
            "frame_cache_misses": self._frame_cache_misses,
            "result_cache_hits": self._result_cache_hits,
            "result_cache_misses": self._result_cache_misses,
            "result_cache_hit_rate": 0.0 if total == 0 else self._result_cache_hits / total,
            "model_calls": self._model_calls,
            "result_cache_size": len(self._result_cache),
            "frame_cache_size": len(self._frame_cache),
        }

    async def _decode_and_sample_frames(
        self,
        *,
        video_path: str,
        t0: float,
        t1: float,
        fps: float,
        resolution: int | None,
        decoder,
        frame_sampler,
    ):
        frame_key = self._frame_cache_key(video_path, t0, t1, fps, resolution)

        if self.enable_frame_cache and frame_key in self._frame_cache:
            self._frame_cache_hits += 1
            return self._frame_cache[frame_key]

        self._frame_cache_misses += 1

        t_decode0 = time.time()
        frames = decoder.decode_window(
            video_path,
            t0=t0,
            t1=t1,
            fps=fps,
        )
        t_decode1 = time.time()
        self._timing["frame_decode_s"] += (t_decode1 - t_decode0)

        t_sample0 = time.time()
        # If your sampler supports resolution, pass it here.
        try:
            sampled = frame_sampler.sample(frames, resolution=resolution)
        except TypeError:
            sampled = frame_sampler.sample(frames)
        t_sample1 = time.time()
        self._timing["frame_sample_s"] += (t_sample1 - t_sample0)

        if self.enable_frame_cache:
            self._frame_cache[frame_key] = sampled

        return sampled

    async def read_frames(self, frames: list) -> dict[str, Any]:
        """
        Minimal stub OCR model call.
        Replace this later with a real OCR/VLM call.
        """
        self._model_calls += 1

        t0 = time.time()
        texts = []
        for i, _frame in enumerate(frames):
            texts.append(f"dummy OCR text from frame {i}")
        t1 = time.time()

        self._timing["model_inference_s"] += (t1 - t0)

        return {
            "texts": texts,
        }

    async def ocr_window(
        self,
        *,
        video_path: str,
        t0: float,
        t1: float,
        fps: float,
        resolution: int | None,
        decoder,
        frame_sampler,
    ) -> dict[str, Any]:
        # Early result-cache lookup
        t_lookup0 = time.time()
        result_key = self._result_cache_key(video_path, t0, t1, fps, resolution)
        cache_hit = self.enable_result_cache and result_key in self._result_cache
        t_lookup1 = time.time()
        self._timing["result_cache_lookup_s"] += (t_lookup1 - t_lookup0)

        if cache_hit:
            self._result_cache_hits += 1
            t_hit0 = time.time()
            out = dict(self._result_cache[result_key])
            t_hit1 = time.time()
            self._timing["result_cache_hit_return_s"] += (t_hit1 - t_hit0)
            return out

        self._result_cache_misses += 1

        frames = await self._decode_and_sample_frames(
            video_path=video_path,
            t0=t0,
            t1=t1,
            fps=fps,
            resolution=resolution,
            decoder=decoder,
            frame_sampler=frame_sampler,
        )

        ocr = await self.read_frames(frames)

        result = {
            "t0": t0,
            "t1": t1,
            "text": "\n".join(ocr["texts"]),
            "segments": ocr["texts"],
        }

        if self.enable_result_cache:
            self._result_cache[result_key] = dict(result)

        return result