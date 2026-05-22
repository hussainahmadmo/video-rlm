from __future__ import annotations

from conductor.api.schemas import StageResult, StageStatus
from conductor.engine.runners.base import BaseRunner
from conductor.engine.stage_types import StageType
from conductor.retrieval.window_proposal import CandidateWindow, build_action_space
from conductor.retrieval.clip_window_ranker import CLIPWindowRanker

class InitialProposalRunner(BaseRunner):
    async def run(self, *, workflow, spec, parent_results, reuse_cache):
        duration_s = float(spec.payload["duration_s"])

        # Coarse scan settings
        window_len_s = 5.0
        stride_s = 5.0

        candidates = []
        t0 = 0.0

        # Generate coarse windows over the full video
        while t0 < duration_s:
            t1 = min(duration_s, t0 + window_len_s)
            candidates.append({
                "t0": round(t0, 3),
                "t1": round(t1, 3),
                "score": 0.0,
                "source": "coarse",
            })
            t0 += stride_s

        # Build initial inspect actions from coarse windows
        actions = []
        for c in candidates:
            mid = 0.5 * (c["t0"] + c["t1"])
            a_t0 = max(0.0, mid - 2.0)
            a_t1 = min(duration_s, mid + 2.0)

            actions.append({
                "kind": "inspect_window",
                "t0": round(a_t0, 3),
                "t1": round(a_t1, 3),
                "stride": 0.5,
                "resolution": "medium",
                "score": c["score"],
                "source": c["source"],
            })

        print("InitialProposalRunner generated", len(candidates), "coarse windows")

        return StageResult(
            stage_id=spec.stage_id,
            workflow_id=workflow.workflow_id,
            stage_type=StageType.INITIAL_PROPOSAL,
            status=StageStatus.DONE,
            artifacts={
                "candidates": candidates,
                "actions": actions,
            },
        )