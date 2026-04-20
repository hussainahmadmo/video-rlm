# conductor/controller/orchestrator.py
from __future__ import annotations

from conductor.engine.stage_types import StageType

from conductor.engine.runners.asr_batch_runner import ASRBatchLocalizeRunner
class Orchestrator:
    def __init__(self, engine):
        self.engine = engine

    async def run(
        self,
        *,
        question: str,
        video_path: str,
        duration_s: float,
        policy_mode: str = "auto",
        baseline_visual_max_frames: int = 60,
        encoder_visual_max_frames: int = 4,
    ) -> dict:
        wf = self.engine.create_workflow(
            question=question,
            video_path=video_path,
            metadata={"duration_s": duration_s},
        )

        profile_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.PROFILE_QUERY,
            payload={},
        )

        proposal_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.INITIAL_PROPOSAL,
            payload={"duration_s": duration_s},
            depends_on=[profile_id],
        )

        await self.engine.run_stage(wf.workflow_id, profile_id)
        await self.engine.run_stage(wf.workflow_id, proposal_id)

        prof = self.engine.get_result(wf.workflow_id, profile_id)
        proposal = self.engine.get_result(wf.workflow_id, proposal_id)

        policy = prof.artifacts["execution_policy"]
        query_profile = prof.artifacts.get("query_profile", {})

        candidates = proposal.artifacts["candidates"]
        actions = proposal.artifacts["actions"]

        preferred_tools = tuple(policy.get("preferred_tools", ("clip",)))
        inspection_pattern = str(query_profile.get("inspection_pattern", "static_localized"))
        reference_type = str(query_profile.get("reference_type", "none"))
        answer_type = str(query_profile.get("answer_type", "visual_object"))
        temporal_requirement = str(query_profile.get("temporal_requirement", "none"))

        probe_topk = int(policy.get("probe_topk", 8))
        action_topk = int(policy.get("action_topk", 8))
        probe_fps = float(policy.get("probe_fps", 1.0))

        if inspection_pattern == "speech_first":
            probe_candidates = candidates
        else:
            probe_candidates = candidates[: min(len(candidates), probe_topk)]

        inspect_actions = actions[: min(len(actions), action_topk)]

        if policy_mode == "baseline_full":
            return await self._run_default_one_pass(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=probe_candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
                inspection_pattern="forced_baseline_full",
            )
        
        
        if policy_mode == "baseline_joint_heavy_60f":
            return await self._run_baseline_joint_heavy_60f(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                duration_s=duration_s,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
                visual_max_frames=baseline_visual_max_frames,
            )

        if policy_mode == "speech_first_targeted":
            return await self._run_speech_first(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                duration_s=duration_s,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=candidates,  # important
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
            )

        if policy_mode == "baseline_asr_then_full_visual":
            return await self._run_asr_then_full_visual(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                duration_s=duration_s,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=candidates,   # broad ASR coverage
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
            )
        
        if policy_mode == "joint_coarse_then_refine":
            return await self._run_joint_coarse_then_refine(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                duration_s=duration_s,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
            )

        if policy_mode == "encoder_aware_60f":
            return await self._run_encoder_aware_60f(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                duration_s=duration_s,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
                visual_max_frames=encoder_visual_max_frames,
            )

        if inspection_pattern == "speech_first":
            return await self._run_speech_first(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                duration_s=duration_s,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=probe_candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
            )

        if inspection_pattern == "ocr_first":
            return await self._run_ocr_first(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                duration_s=duration_s,
                prof=prof,
                proposal=proposal,
                preferred_tools=preferred_tools,
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=probe_candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
            )

        return await self._run_default_one_pass(
            wf=wf,
            profile_id=profile_id,
            proposal_id=proposal_id,
            question=question,
            prof=prof,
            proposal=proposal,
            preferred_tools=preferred_tools,
            reference_type=reference_type,
            answer_type=answer_type,
            temporal_requirement=temporal_requirement,
            probe_candidates=probe_candidates,
            inspect_actions=inspect_actions,
            probe_fps=probe_fps,
            inspection_pattern=inspection_pattern,
        )
    
    async def _run_speech_first(
        self,
        *,
        wf,
        profile_id: str,
        proposal_id: str,
        question: str,
        duration_s: float,
        prof,
        proposal,
        preferred_tools: tuple[str, ...],
        reference_type: str,
        answer_type: str,
        temporal_requirement: str,
        probe_candidates: list[dict],
        inspect_actions: list[dict],
        probe_fps: float,
    ) -> dict:
        """
        True speech-first:
          1. run ASR coarse pass first
          2. derive refined visual windows from ASR
          3. run visual second
          4. optional OCR only if useful
        """
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        # --------------------------------------------------------
        # First pass: ASR only
        # --------------------------------------------------------
        if "asr" in preferred_tools:
            sid = self.engine.add_stage(
                wf.workflow_id,
                StageType.ASR_LOCALIZE_BATCH,
                payload={
                    "windows": [
                    {"t0": w["t0"], "t1": w["t1"]}
                    for w in probe_candidates
                    ]
                },
                depends_on=[proposal_id],
            )
            asr_stage_ids.append(sid)

        if asr_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)
            asr_results = self._collect_asr_batch_results(
                workflow_id=wf.workflow_id,
                asr_stage_ids=asr_stage_ids,
            )
        else:
            asr_results = []

        # --------------------------------------------------------
        # Optional OCR in speech-first only when text is referenced
        # --------------------------------------------------------
        if "ocr" in preferred_tools and reference_type == "text_reference":
            for w in probe_candidates[:4]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "fps": 1.0,
                    },
                    depends_on=[proposal_id],
                )
                ocr_stage_ids.append(sid)

            if ocr_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, ocr_stage_ids)
                ocr_results = self._collect_ocr_results(
                        workflow_id=wf.workflow_id,
                        ocr_stage_ids=ocr_stage_ids,)
            else:
                ocr_results = []

        # --------------------------------------------------------
        # Second pass: refine from ASR results, then visual
        # --------------------------------------------------------
        refined_windows = self._refine_from_asr_results(
            asr_results = asr_results,
            duration_s=duration_s,
            question=question,
            max_windows=4,
        )

        if "clip" in preferred_tools:
            for w in refined_windows:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": 4,
                    },
                    depends_on=asr_stage_ids[:] if asr_stage_ids else [proposal_id],
                )
                visual_stage_ids.append(sid)

            if visual_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, visual_stage_ids)

        map_stage_ids = asr_stage_ids + ocr_stage_ids + visual_stage_ids

        fuse_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.FUSE,
            depends_on=map_stage_ids,
        )
        await self.engine.run_stage(wf.workflow_id, fuse_id)

        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=[fuse_id],
        )
        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_res.artifacts["answer"],
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": "speech_first",
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,
                "num_asr_stages": len(asr_stage_ids),
                "num_ocr_stages": len(ocr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
                "refined_windows": refined_windows,
                "asr_results": asr_results,
                "num_asr_stage_jobs": len(asr_stage_ids),
                "num_asr_results": len(asr_results),
            },
        }

    async def _run_ocr_first(
        self,
        *,
        wf,
        profile_id: str,
        proposal_id: str,
        question: str,
        duration_s: float,
        prof,
        proposal,
        preferred_tools: tuple[str, ...],
        reference_type: str,
        answer_type: str,
        temporal_requirement: str,
        probe_candidates: list[dict],
        inspect_actions: list[dict],
        probe_fps: float,
    ) -> dict:
        """
        First practical OCR-first version:
          1. run OCR first
          2. refine windows from OCR
          3. run visual second
          4. optional ASR only if useful
        """
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        if "ocr" in preferred_tools:
            for w in probe_candidates[:8]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "fps": 1.0,
                    },
                    depends_on=[proposal_id],
                )
                ocr_stage_ids.append(sid)

        if ocr_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, ocr_stage_ids)
            ocr_results = self._collect_ocr_results(
                workflow_id=wf.workflow_id,
                ocr_stage_ids=ocr_stage_ids,
                )
        else:
            ocr_results = []


        if "asr" in preferred_tools and reference_type == "speech_reference":
            for w in probe_candidates[:4]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.ASR_LOCALIZE,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                    },
                    depends_on=[proposal_id],
                )
                asr_stage_ids.append(sid)

            if asr_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)

        refined_windows = self._refine_from_ocr_results(
            ocr_results = ocr_results,
            duration_s=duration_s,
            max_windows=4,
        )

        if "clip" in preferred_tools:
            for w in refined_windows:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": 4,
                    },
                    depends_on=ocr_stage_ids[:] if ocr_stage_ids else [proposal_id],
                )
                visual_stage_ids.append(sid)

            if visual_stage_ids:
                for sid in visual_stage_ids:
                    await self.engine.run_stage(wf.workflow_id, sid)

        map_stage_ids = asr_stage_ids + ocr_stage_ids + visual_stage_ids

        fuse_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.FUSE,
            depends_on=map_stage_ids,
        )
        await self.engine.run_stage(wf.workflow_id, fuse_id)

        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=[fuse_id],
        )
        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_res.artifacts["answer"],
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": "ocr_first",
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,
                "num_asr_stages": len(asr_stage_ids),
                "num_ocr_stages": len(ocr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
                "refined_windows": refined_windows,
                "ocr_results": ocr_results,
            },
        }

    async def _run_default_one_pass(
        self,
        *,
        wf,
        profile_id: str,
        proposal_id: str,
        question: str,
        prof,
        proposal,
        preferred_tools: tuple[str, ...],
        reference_type: str,
        answer_type: str,
        temporal_requirement: str,
        probe_candidates: list[dict],
        inspect_actions: list[dict],
        probe_fps: float,
        inspection_pattern: str,
    ) -> dict:
        """
        Default one-pass workflow-aware fanout.
        """
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []
        map_stage_ids: list[str] = []

        if "asr" in preferred_tools:
            for w in probe_candidates:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.ASR_LOCALIZE,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                    },
                    depends_on=[proposal_id],
                )
                asr_stage_ids.append(sid)
                map_stage_ids.append(sid)

        if "ocr" in preferred_tools:
            for w in probe_candidates[:8]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "fps": 1.0,
                    },
                    depends_on=[proposal_id],
                )
                ocr_stage_ids.append(sid)
                map_stage_ids.append(sid)

        if "clip" in preferred_tools:
            for a in inspect_actions:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": a["t0"],
                        "t1": a["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": 4,
                    },
                    depends_on=[proposal_id],
                )
                visual_stage_ids.append(sid)
                map_stage_ids.append(sid)

        if map_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, map_stage_ids)

        fuse_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.FUSE,
            depends_on=map_stage_ids,
        )
        await self.engine.run_stage(wf.workflow_id, fuse_id)

        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=[fuse_id],
        )
        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_res.artifacts["answer"],
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": inspection_pattern,
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,
                "num_asr_stages": len(asr_stage_ids),
                "num_ocr_stages": len(ocr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
            },
        }
    
    def _refine_from_asr_results(
        self,
        *,
        asr_results: list[dict],
        duration_s: float,
        question: str,
        max_windows: int = 4,
    ) -> list[dict]:
        """
        self: current orchestrator object
        asr_results: first pass ASR outputs, one dict per ASR window
        durations_s: full video duration, used to clamp refined windows
        question: user question, used for relevance scoring and temporal cues
        max_windows: max number of refined windows to return.s
        """
        #first score each ASR result against the question.
        #This lets us rank windows 

        #ASR results after attaching a relevance score.
        scored = []

        #res = one ASR window result
        for res in asr_results:
            #transcript text for this window
            transcript = str(res.get("transcript", "")).strip()
            if not transcript:
                continue #skip empty ASR outputs.

            score = self._score_text_against_question(
                question=question,
                text=transcript,
            )

            item = dict(res)
            item["score"] = score
            scored.append(item)

        scored.sort(key=lambda x: (-x["score"], x["t0"], x["t1"]))

        refined = []
        seen = set()

        for res in scored:
            t0 = float(res["t0"])
            t1 = float(res["t1"])

            mid = 0.5 * (t0 + t1)
            win_t0 = max(0.0, mid - 2.0)
            win_t1 = min(duration_s, mid + 2.0)

            key = (round(win_t0, 2), round(win_t1, 2))
            if key in seen:
                continue

            refined.append({
                "t0": round(win_t0, 3),
                "t1": round(win_t1, 3),
                "score": res["score"],
                "source_t0": t0,
                "source_t1": t1,
                "source_text": res["transcript"],
            })
            seen.add(key)

            if len(refined) >= max_windows:
                break

        q = question.lower()
        if refined and any(x in q for x in ["after", "next", "then", "later"]):
            base = refined[0]
            t0 = min(duration_s, base["t1"])
            t1 = min(duration_s, t0 + 4.0)
            key = (round(t0, 2), round(t1, 2))
            if key not in seen:
                refined.append({
                    "t0": round(t0, 3),
                    "t1": round(t1, 3),
                    "score": base["score"],
                    "source_t0": base["source_t0"],
                    "source_t1": base["source_t1"],
                    "source_text": base["source_text"],
                })

        return refined[:max_windows]


    def _refine_from_ocr_results(
        self,
        *,
        ocr_results: list[dict],
        duration_s: float,
        max_windows: int = 4,
    ) -> list[dict]:
        refined = []

        for res in ocr_results:
            joined_text = str(res.get("joined_text", "")).strip()
            if not joined_text:
                continue

            t0 = float(res["t0"])
            t1 = float(res["t1"])

            mid = 0.5 * (t0 + t1)
            win_t0 = max(0.0, mid - 2.0)
            win_t1 = min(duration_s, mid + 2.0)

            refined.append({
                "t0": round(win_t0, 3),
                "t1": round(win_t1, 3),
            })

        seen = set()
        uniq = []
        for w in refined:
            key = (round(w["t0"], 2), round(w["t1"], 2))
            if key not in seen:
                uniq.append(w)
                seen.add(key)

        return uniq[:max_windows]
    
    def _collect_asr_results(self, *, workflow_id: str, asr_stage_ids: list[str]) -> list[dict]:
        out = []
        for sid in asr_stage_ids:
            res = self.engine.get_result(workflow_id, sid)
            out.append({
                "stage_id": sid,
                "t0": float(res.artifacts["t0"]),
                "t1": float(res.artifacts["t1"]),
                "transcript": str(res.artifacts.get("transcript", "")).strip(),
                "segments": res.artifacts.get("segments", []),
            })
        return out
    
    def _collect_ocr_results(self, *, workflow_id: str, ocr_stage_ids: list[str]) -> list[dict]:
        out = []
        for sid in ocr_stage_ids:
            res = self.engine.get_result(workflow_id, sid)
            out.append({
                "stage_id": sid,
                "t0": float(res.artifacts["t0"]),
                "t1": float(res.artifacts["t1"]),
                "texts": res.artifacts.get("texts", []),
                "joined_text": str(res.artifacts.get("joined_text", "")).strip(),
            })
        return out
    
    def _collect_asr_batch_results(self, *, workflow_id: str, asr_stage_ids: list[str]) -> list[dict]:
        out = []
        for sid in asr_stage_ids:
            res = self.engine.get_result(workflow_id, sid)
            out.extend(res.artifacts.get("results", []))
        return out
    def _score_text_against_question(self, *, question: str, text: str) -> int:
        q_words = set(question.lower().replace("?", "").split())
        t_words = set(text.lower().replace("?", "").split())
        return len(q_words & t_words)

    async def _run_asr_then_full_visual(
    self,
    *,
    wf,
    profile_id: str,
    proposal_id: str,
    question: str,
    duration_s: float,
    prof,
    proposal,
    preferred_tools: tuple[str, ...],
    reference_type: str,
    answer_type: str,
    temporal_requirement: str,
    probe_candidates: list[dict],
    inspect_actions: list[dict],
    probe_fps: float,
) -> dict:
        """
        Strong baseline:
        1. run ASR first
        2. do NOT refine visual windows from ASR
        3. run the broad/full visual fanout
        4. fuse
        5. answer
        """
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        asr_results = []

        # 1) ASR first
        if "asr" in preferred_tools:
            sid = self.engine.add_stage(
                wf.workflow_id,
                StageType.ASR_LOCALIZE_BATCH,
                payload={
                    "windows": [
                        {"t0": w["t0"], "t1": w["t1"]}
                        for w in probe_candidates
                    ]
                },
                depends_on=[proposal_id],
            )
            asr_stage_ids.append(sid)

        if asr_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)
            asr_results = self._collect_asr_batch_results(
                workflow_id=wf.workflow_id,
                asr_stage_ids=asr_stage_ids,
            )

        # 2) Optional OCR if needed
        if "ocr" in preferred_tools:
            for w in probe_candidates[:8]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "fps": 1.0,
                    },
                    depends_on=asr_stage_ids[:] if asr_stage_ids else [proposal_id],
                )
                ocr_stage_ids.append(sid)

            if ocr_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, ocr_stage_ids)

        # 3) Full visual fanout, no ASR-guided pruning
        if "clip" in preferred_tools:
            for a in inspect_actions:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": a["t0"],
                        "t1": a["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": 4,
                    },
                    depends_on=asr_stage_ids[:] if asr_stage_ids else [proposal_id],
                )
                visual_stage_ids.append(sid)

            if visual_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, visual_stage_ids)

        map_stage_ids = asr_stage_ids + ocr_stage_ids + visual_stage_ids

        fuse_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.FUSE,
            depends_on=map_stage_ids,
        )
        await self.engine.run_stage(wf.workflow_id, fuse_id)

        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=[fuse_id],
        )
        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_res.artifacts["answer"],
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": "baseline_asr_then_full_visual",
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,
                "num_asr_stages": len(asr_stage_ids),
                "num_ocr_stages": len(ocr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
                "num_asr_stage_jobs": len(asr_stage_ids),
                "num_asr_results": len(asr_results),
                "num_refined_windows": 0,
                "refined_total_span_s": 0.0,
                "refined_windows": [],
            },
        }
    
    def _build_coarse_chunks(
        self,
        *,
        duration_s: float,
        chunk_len_s: float = 60.0,
    ) -> list[dict]:
        out = []
        t = 0.0
        while t < duration_s:
            t1 = min(duration_s, t + chunk_len_s)
            out.append({"t0": round(t, 3), "t1": round(t1, 3)})
            t = t1
        return out
    

    async def _run_baseline_joint_heavy_60f(
        self,
        *,
        wf,
        profile_id: str,
        proposal_id: str,
        question: str,
        duration_s: float,
        prof,
        proposal,
        preferred_tools: tuple[str, ...],
        reference_type: str,
        answer_type: str,
        temporal_requirement: str,
        probe_candidates: list[dict],
        inspect_actions: list[dict],
        probe_fps: float,
        visual_max_frames: int,) -> dict:
        """
        Heavy baseline for experiment:
        - split video into 60s chunks
        - run ASR on each chunk
        - run heavy visual encoding on each chunk (1 fps, up to 60 frames)
        - no encoder-aware pruning
        """
        asr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        coarse_chunks = self._build_coarse_chunks(
            duration_s=duration_s,
            chunk_len_s=60.0,
        )

        # ASR over every 60s chunk
        if "asr" in preferred_tools:
            for w in coarse_chunks:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.ASR_LOCALIZE,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                    },
                    depends_on=[proposal_id],
                )
                asr_stage_ids.append(sid)

        # Heavy visual: send 60 frames per minute
        if "clip" in preferred_tools:
            for w in coarse_chunks:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "query": question,
                        "fps": 1.0,
                        "max_frames": visual_max_frames,
                    },
                    depends_on=[proposal_id],
                )
                visual_stage_ids.append(sid)


        if asr_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)
        
        if visual_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, visual_stage_ids)

        map_stage_ids = asr_stage_ids + visual_stage_ids

        asr_results = self._collect_asr_results(
            workflow_id=wf.workflow_id,
            asr_stage_ids=asr_stage_ids,
        ) if asr_stage_ids else []

        fuse_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.FUSE,
            depends_on=map_stage_ids,
        )
        await self.engine.run_stage(wf.workflow_id, fuse_id)

        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=[fuse_id],
        )
        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_res.artifacts["answer"],
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": "baseline_joint_heavy_60f",
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,
                "num_asr_stages": len(asr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
                "num_asr_results": len(asr_results),
                "num_refined_windows": 0,
                "refined_total_span_s": 0.0,
                "coarse_chunk_len_s": 60.0,
                "visual_fps": 1.0,
                "visual_max_frames": visual_max_frames,
                "total_visual_frames_sent": len(visual_stage_ids) * visual_max_frames,
            },
        }
    
    async def _run_encoder_aware_60f(
        self,
        *,
        wf,
        profile_id: str,
        proposal_id: str,
        question: str,
        duration_s: float,
        prof,
        proposal,
        preferred_tools: tuple[str, ...],
        reference_type: str,
        answer_type: str,
        temporal_requirement: str,
        probe_candidates: list[dict],
        inspect_actions: list[dict],
        probe_fps: float,
        visual_max_frames: int,) -> dict:

        asr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        coarse_chunks = self._build_coarse_chunks(duration_s=duration_s, chunk_len_s=60.0)

        # 1) ASR on all 60s chunks
        if "asr" in preferred_tools:
            for w in coarse_chunks:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.ASR_LOCALIZE,
                    payload={"t0": w["t0"], "t1": w["t1"]},
                    depends_on=[proposal_id],
                )
                asr_stage_ids.append(sid)

        if asr_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)

        asr_results = self._collect_asr_results(
            workflow_id=wf.workflow_id,
            asr_stage_ids=asr_stage_ids,
        ) if asr_stage_ids else []

        # 2) rank chunks using ASR
        ranked = sorted(
            [
                {
                    "t0": float(r["t0"]),
                    "t1": float(r["t1"]),
                    "score": self._score_text_against_question(
                        question=question,
                        text=str(r.get("transcript", "")),
                    ),
                }
                for r in asr_results
            ],
            key=lambda x: (-x["score"], x["t0"])
        )

        top_chunks = ranked[:4]

        # 3) heavy visual only on selected chunks
        if "clip" in preferred_tools:
            for w in top_chunks:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "query": question,
                        "fps": 1.0,
                        "max_frames": visual_max_frames,
                    },
                    depends_on=asr_stage_ids[:] if asr_stage_ids else [proposal_id],
                )
                visual_stage_ids.append(sid)

            if visual_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, visual_stage_ids)

        map_stage_ids = asr_stage_ids + visual_stage_ids

        fuse_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.FUSE,
            depends_on=map_stage_ids,
        )
        await self.engine.run_stage(wf.workflow_id, fuse_id)

        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=[fuse_id],
        )
        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_res.artifacts["answer"],
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": "encoder_aware_60f",
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,
                "num_asr_stages": len(asr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
                "num_asr_results": len(asr_results),
                "num_refined_windows": len(top_chunks),
                "refined_windows": top_chunks,
                "coarse_chunk_len_s": 60.0,
                "visual_fps": 1.0,
                "visual_max_frames": visual_max_frames,
                "total_visual_frames_sent": len(visual_stage_ids) * visual_max_frames,
            },
        }