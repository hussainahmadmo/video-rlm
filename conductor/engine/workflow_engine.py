from __future__ import annotations

import asyncio
import uuid
from typing import Dict

from conductor.api.schemas import StageResult, StageSpec, StageStatus, Workflow
from conductor.engine.cache import ReuseCache
from conductor.engine.stage_types import StageType


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

        # Shared resource gate for heavy visual encoder work
        self.visual_sem = asyncio.Semaphore(max_visual_concurrency)

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
        wf = self.workflows[workflow_id]
        stage_id = str(uuid.uuid4())
        spec = StageSpec(
            stage_id=stage_id,
            workflow_id=workflow_id,
            stage_type=stage_type,
            payload=payload or {},
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
        wf = self.workflows[workflow_id]
        spec = wf.stages[stage_id]
        runner = self.runners[spec.stage_type]

        wf.stage_status[stage_id] = StageStatus.RUNNING

        parent_results = {
            dep: wf.stage_results[dep]
            for dep in spec.depends_on
            if dep in wf.stage_results
        }

        try:
            if spec.stage_type == StageType.VISUAL_INSPECT:
                async with self.visual_sem:
                    result: StageResult = await runner.run(
                        workflow=wf,
                        spec=spec,
                        parent_results=parent_results,
                        reuse_cache=self.reuse_cache,
                    )
            else:
                result: StageResult = await runner.run(
                    workflow=wf,
                    spec=spec,
                    parent_results=parent_results,
                    reuse_cache=self.reuse_cache,
                )

            wf.stage_results[stage_id] = result
            wf.stage_status[stage_id] = result.status
            return result

        except Exception as e:
            result = StageResult(
                stage_id=stage_id,
                workflow_id=workflow_id,
                stage_type=spec.stage_type,
                status=StageStatus.FAILED,
                artifacts={},
                metrics={},
                error=str(e),
            )
            wf.stage_results[stage_id] = result
            wf.stage_status[stage_id] = StageStatus.FAILED
            return result

    async def run_ready_batch(self, workflow_id: str, stage_ids: list[str]) -> list[StageResult]:
        return await asyncio.gather(*(self.run_stage(workflow_id, sid) for sid in stage_ids))

    def get_result(self, workflow_id: str, stage_id: str) -> StageResult:
        return self.workflows[workflow_id].stage_results[stage_id]