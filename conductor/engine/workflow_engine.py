from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict

from conductor.api.schemas import StageResult, StageSpec, StageStatus, Workflow
from conductor.engine.cache import ReuseCache
from conductor.engine.stage_types import StageType

import csv
from pathlib import Path


class WorkflowEngine:
    def __init__(
        self,
        runners: Dict[StageType, object],
        reuse_cache: ReuseCache,
        max_visual_concurrency: int = 1,
    ):
        self.runners = runners
        self.reuse_cache = reuse_cache
        self.workflows: Dict[str, Workflow] = {}

        self.max_visual_concurrency = max_visual_concurrency

        # Shared resource gate for heavy visual encoder work
        self.visual_sem = asyncio.Semaphore(max_visual_concurrency)

        # Debug counters
        self._active_visual = 0
        self._max_observed_visual = 0
        print(f"[ENGINE] max_visual_concurrency={max_visual_concurrency}")

    def create_workflow(self, *, question: str, video_path: str, metadata: dict | None = None) -> Workflow:
        workflow_id = str(uuid.uuid4())
        wf = Workflow(
            workflow_id=workflow_id,
            question=question,
            video_path=video_path,
            metadata=metadata or {},
        )
        self.workflows[workflow_id] = wf
        return wf
    
    def add_stage(
        self,
        workflow_id: str,
        stage_type: StageType,
        *,
        payload: dict | None = None,
        depends_on: list[str] | None = None,
    ) -> str:
        #register a new stage in the workflow, does not run it yet.
        wf = self.workflows[workflow_id]
        stage_id = str(uuid.uuid4())

        stage_payload = dict(payload or {})
        stage_payload["_created_t"] = time.time()

        #actual description of stage. 
        #what stage it is, and workflow it belongs to, 
        spec = StageSpec(
            stage_id=stage_id,
            workflow_id=workflow_id,
            stage_type=stage_type,
            payload=stage_payload,
            depends_on=depends_on or [],
        )
        wf.stages[stage_id] = spec
        wf.stage_status[stage_id] = StageStatus.PENDING
        return stage_id

    def ready_stages(self, workflow_id: str) -> list[StageSpec]:
        wf = self.workflows[workflow_id]
        out: list[StageSpec] = []
        for stage_id, spec in wf.stages.items():
            st = wf.stage_status[stage_id]
            if st in {StageStatus.DONE, StageStatus.RUNNING, StageStatus.FAILED}:
                continue

            ready = True
            for dep in spec.depends_on:
                if wf.stage_status.get(dep) != StageStatus.DONE:
                    ready = False
                    break

            if ready:
                wf.stage_status[stage_id] = StageStatus.READY
                out.append(spec)
        return out

    async def run_stage(self, workflow_id: str, stage_id: str) -> StageResult:
        #executes one already added stage.
        #takes a stage that was registered earlier, run its runner, 
        #stores the StageResult, and marks its status
        wf = self.workflows[workflow_id]
        spec = wf.stages[stage_id]
        runner = self.runners[spec.stage_type]

        wf.stage_status[stage_id] = StageStatus.RUNNING

        # parent_results = {
        #     dep: wf.stage_results[dep]
        # for dep in (spec.depends_on or []):
        #     if dep in wf.stage_results
        # }
        parent_results = {
            dep: wf.stage_results[dep]
            for dep in (spec.depends_on or [])
            if dep in wf.stage_results
        }

        created_t = float(spec.payload.get("_created_t", time.time()))
        submit_t = time.time()

        try:
            if spec.stage_type == StageType.VISUAL_INSPECT:
                wait_start_t = time.time()

                async with self.visual_sem:
                    start_t = time.time()

                    self._active_visual += 1
                    self._max_observed_visual = max(
                        self._max_observed_visual,
                        self._active_visual,
                    )

                    print(
                        f"[VISUAL_CONCURRENCY] START "
                        f"stage_id={stage_id} "
                        f"window=({spec.payload.get('t0')},{spec.payload.get('t1')}) "
                        f"active={self._active_visual} "
                        f"max_observed={self._max_observed_visual} "
                        f"limit={self.max_visual_concurrency} "
                        f"t={start_t:.3f}"
                    )

                    try:
                        result: StageResult = await runner.run(
                            workflow=wf,
                            spec=spec,
                            parent_results=parent_results,
                            reuse_cache=self.reuse_cache,
                        )
                    finally:
                        end_t = time.time()

                        print(
                            f"[VISUAL_CONCURRENCY] END "
                            f"stage_id={stage_id} "
                            f"window=({spec.payload.get('t0')},{spec.payload.get('t1')}) "
                            f"active_before_dec={self._active_visual} "
                            f"max_observed={self._max_observed_visual} "
                            f"limit={self.max_visual_concurrency} "
                            f"t={end_t:.3f}"
                        )

                        self._active_visual -= 1

                queue_wait_s = start_t - wait_start_t
            else:
                start_t = time.time()
                result: StageResult = await runner.run(
                    workflow=wf,
                    spec=spec,
                    parent_results=parent_results,
                    reuse_cache=self.reuse_cache,
                )
                end_t = time.time()
                queue_wait_s = 0.0

            result.metrics = dict(result.metrics or {})
            result.metrics.update({
                "created_t": created_t,
                "submit_t": submit_t,
                "start_t": start_t,
                "end_t": end_t,
                "creation_to_submit_s": submit_t - created_t,
                "submit_to_start_s": start_t - submit_t,
                "creation_to_start_s": start_t - created_t,
                "stage_wall_s": end_t - start_t,
                "queue_wait_s": queue_wait_s,
                "creation_to_end_s": end_t - created_t,
            })

            if spec.stage_type == StageType.VISUAL_INSPECT:
                result.metrics["visual_concurrency_limit"] = self.max_visual_concurrency
                result.metrics["max_observed_visual_concurrency"] = self._max_observed_visual

            wf.stage_results[stage_id] = result
            wf.stage_status[stage_id] = result.status
            self.append_stage_trace(wf, spec, result)
            return result

        except Exception as e:
            fail_t = time.time()
            result = StageResult(
                stage_id=stage_id,
                workflow_id=workflow_id,
                stage_type=spec.stage_type,
                status=StageStatus.FAILED,
                artifacts={},
                metrics={
                    "created_t": created_t,
                    "submit_t": submit_t,
                    "start_t": fail_t,
                    "end_t": fail_t,
                    "creation_to_submit_s": submit_t - created_t,
                    "submit_to_start_s": fail_t - submit_t,
                    "creation_to_start_s": fail_t - created_t,
                    "stage_wall_s": 0.0,
                    "queue_wait_s": 0.0,
                    "creation_to_end_s": fail_t - created_t,
                },
                error=str(e),
            )
            wf.stage_results[stage_id] = result
            wf.stage_status[stage_id] = StageStatus.FAILED
            self.append_stage_trace(wf, spec, result)
            return result
        

    async def run_ready_batch(self, workflow_id: str, stage_ids: list[str]) -> list[StageResult]:
        return await asyncio.gather(*(self.run_stage(workflow_id, sid) for sid in stage_ids))

    def get_result(self, workflow_id: str, stage_id: str) -> StageResult:
        return self.workflows[workflow_id].stage_results[stage_id]
    
    def append_stage_trace(self, wf: Workflow, spec: StageSpec, result: StageResult):

        out_path = Path("outputs/stage_timing/stage_trace.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        m = result.metrics or {}

        row = {
            "workflow_id": wf.workflow_id,
            "video_id": wf.metadata.get("video_id", wf.video_path),
            "video_size": wf.metadata.get("video_size", ""),
            "policy_mode": wf.metadata.get("policy_mode", ""),
            "question": wf.question,

            "stage_id": spec.stage_id,
            "stage_type": str(spec.stage_type).replace("StageType.", ""),
            "status": str(result.status).replace("StageStatus.", ""),
            "error": result.error or "",

            "created_t": m.get("created_t", ""),
            "submit_t": m.get("submit_t", ""),
            "start_t": m.get("start_t", ""),
            "end_t": m.get("end_t", ""),
            "stage_wall_s": m.get("stage_wall_s", ""),
            "queue_wait_s": m.get("queue_wait_s", ""),
            "creation_to_start_s": m.get("creation_to_start_s", ""),
            "creation_to_end_s": m.get("creation_to_end_s", ""),

            "t0": spec.payload.get("t0", ""),
            "t1": spec.payload.get("t1", ""),
        }

        fieldnames = list(row.keys())
        file_exists = out_path.exists()

        with out_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
