from __future__ import annotations

import time

from conductor.api.schemas import StageResult, StageStatus
from conductor.engine.runners.base import BaseRunner
from conductor.engine.stage_types import StageType


class OCRWindowRunner(BaseRunner):
    def __init__(self, decoder, frame_sampler, ocr_client):
        self.decoder = decoder
        self.frame_sampler = frame_sampler
        self.ocr_client = ocr_client

    async def run(self, *, workflow, spec, parent_results, reuse_cache):
        t0 = float(spec.payload["t0"])
        t1 = float(spec.payload["t1"])
        fps = float(spec.payload.get("fps", 1.0))
        resolution = spec.payload.get("resolution", None)

        key = reuse_cache.make_key(
            "ocr_window",
            video_path=workflow.video_path,
            t0=round(t0, 2),
            t1=round(t1, 2),
            fps=fps,
            resolution=resolution,
        )

        hit = reuse_cache.get(key)
        if hit.hit:
            return StageResult(
                stage_id=spec.stage_id,
                workflow_id=workflow.workflow_id,
                stage_type=StageType.OCR_WINDOW,
                status=StageStatus.DONE,
                artifacts=hit.value,
                metrics={"cache_hit": True},
            )

        if hit.inflight:
            value = await hit.future
            return StageResult(
                stage_id=spec.stage_id,
                workflow_id=workflow.workflow_id,
                stage_type=StageType.OCR_WINDOW,
                status=StageStatus.DONE,
                artifacts=value,
                metrics={"inflight_reuse": True},
            )

        reuse_cache.begin(key)
        start = time.time()

        try:
            ocr = await self.ocr_client.ocr_window(
                video_path=workflow.video_path,
                t0=t0,
                t1=t1,
                fps=fps,
                resolution=resolution,
                decoder=self.decoder,
                frame_sampler=self.frame_sampler,
            )

            value = {
                "t0": t0,
                "t1": t1,
                "texts": ocr["segments"],
                "joined_text": ocr["text"],
            }

            reuse_cache.finish(key, value)

            return StageResult(
                stage_id=spec.stage_id,
                workflow_id=workflow.workflow_id,
                stage_type=StageType.OCR_WINDOW,
                status=StageStatus.DONE,
                artifacts=value,
                metrics={"latency_s": time.time() - start},
            )
        except Exception as e:
            reuse_cache.fail(key, e)
            raise