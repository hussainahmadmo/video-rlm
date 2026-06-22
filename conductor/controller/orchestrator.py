from __future__ import annotations

import csv
import json
import requests
from pathlib import Path
import time
from conductor.engine.stage_types import StageType
from conductor.engine.runners.asr_batch_runner import ASRBatchLocalizeRunner
from conductor.retrieval.clip_window_ranker import CLIPWindowRanker
from conductor.retrieval.window_proposal import CandidateWindow, build_action_space

class Orchestrator:
    def __init__(self, engine, decoder=None):
        self.engine = engine
        self.clip_ranker = CLIPWindowRanker(
            model_name="ViT-B/32",
            device="cuda",
            use_decord_gpu=True,
        )
        self.decoder = decoder

    async def run(
        self,
        *,
        question: str,
        video_path: str,
        duration_s: float,
        policy_mode: str = "auto",

        # METIS-style runtime knobs
        proposal_strategy: str | None = None,
        run_clip_probe: bool | None = None,
        run_asr: bool | None = None,
        run_ocr: bool | None = None,
        run_vlm: bool | None = None,
        run_cheap_text_stage: bool | None = None,

        probe_fps: float | None = None,
        probe_seg_len_s: float | None = None,
        probe_topk: int | None = None,

        window_len_s: float | None = None,
        action_topk: int | None = None,
        max_vlm_windows: int | None = None,

        answer_max_frames_per_window: int | None = None,
        answer_max_images_total: int | None = None,

        baseline_visual_max_frames: int = 60,
        encoder_visual_max_frames: int = 60,
    ) -> dict:
        
        #create a workflow object - 
        # tell the engine - the user question, the path create a new workflow record for this run - without workflow
        # object the engine does not know which stages belong together.
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

        #make later ASR/OCR/Visual stage depend on it

        proposal_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.INITIAL_PROPOSAL,
            payload={
                "duration_s": duration_s,
                "run_clip_probe": bool(run_clip_probe) if run_clip_probe is not None else True,
                "proposal_strategy": proposal_strategy or "clip_first",
                "probe_fps": probe_fps,
                "probe_seg_len_s": probe_seg_len_s,
                "probe_topk": probe_topk,
                "window_len_s": window_len_s,
            },
            depends_on=[profile_id],
        )

        #run the profile query workflow
        await self.engine.run_stage(wf.workflow_id, profile_id)
        #run the initial proposal stage for the workflow
        await self.engine.run_stage(wf.workflow_id, proposal_id)

        prof = self.engine.get_result(wf.workflow_id, profile_id)
        proposal = self.engine.get_result(wf.workflow_id, proposal_id)


        # ------------------------------------------------------------
        # METIS-style explicit runtime knob overrides.
        # These make experiment configs actually control execution,
        # instead of only being logged as labels.
        # ------------------------------------------------------------
        artifacts = prof.artifacts or {}

        print("[PROFILE_DEBUG] prof.artifacts keys:", list(artifacts.keys()))
        print("[PROFILE_DEBUG] prof.artifacts:", artifacts)

        policy = (
            artifacts.get("execution_policy")
            or artifacts.get("policy")
            or artifacts.get("raw_json", {}).get("execution_policy")
            or artifacts.get("raw_json", {}).get("policy")
            or {}
        )

        query_profile = (
            artifacts.get("query_profile")
            or artifacts.get("profile")
            or artifacts.get("raw_json", {}).get("query_profile")
            or artifacts.get("raw_json", {}).get("profile")
            or {}
        )

        policy = dict(policy)
        query_profile = dict(query_profile)

        # Safe defaults so METIS knob experiments can run even if profiler output is missing.
        policy.setdefault("preferred_tools", ("vlm",))
        policy.setdefault("probe_topk", 8)
        policy.setdefault("action_topk", 2)
        policy.setdefault("probe_fps", 1.0)
        policy.setdefault("probe_seg_len_s", 5.0)
        policy.setdefault("window_len_s", 5.0)
        policy.setdefault("max_vlm_windows", 2)
        policy.setdefault("answer_max_frames_per_window", 2)
        policy.setdefault("answer_max_images_total", 4)

        query_profile.setdefault("inspection_pattern", "vlm_only")
        query_profile.setdefault("reference_type", "none")
        query_profile.setdefault("answer_type", "visual_answer")

        if proposal_strategy is not None:
            policy["proposal_strategy"] = proposal_strategy

        if probe_fps is not None:
            policy["probe_fps"] = float(probe_fps)

        if probe_seg_len_s is not None:
            policy["probe_seg_len_s"] = float(probe_seg_len_s)

        if probe_topk is not None:
            policy["probe_topk"] = int(probe_topk)

        if window_len_s is not None:
            policy["window_len_s"] = float(window_len_s)

        if action_topk is not None:
            policy["action_topk"] = int(action_topk)

        if max_vlm_windows is not None:
            policy["max_vlm_windows"] = int(max_vlm_windows)

        if answer_max_frames_per_window is not None:
            policy["answer_max_frames_per_window"] = int(answer_max_frames_per_window)

        if answer_max_images_total is not None:
            policy["answer_max_images_total"] = int(answer_max_images_total)

        # Convert boolean tool knobs into actual preferred_tools.
        preferred = []

        if run_asr is True:
            preferred.append("asr")
        if run_ocr is True:
            preferred.append("ocr")
        if run_vlm is True:
            preferred.append("vlm")

        # If user explicitly turned tools off/on, use that tool set.
        # Otherwise keep profiler's original preferred_tools.
        if any(x is not None for x in [run_asr, run_ocr, run_vlm]):
            policy["preferred_tools"] = preferred

        # Force inspection pattern from knob combination.
        # This controls which branch below executes.
        if run_asr is True and run_vlm is False and run_ocr is False:
            query_profile["inspection_pattern"] = "asr_only"
        elif run_ocr is True and run_asr is False and run_vlm is False:
            query_profile["inspection_pattern"] = "ocr_only"
        elif run_vlm is True and run_asr is False and run_ocr is False:
            query_profile["inspection_pattern"] = "vlm_only"
        elif run_asr is True and run_vlm is True and run_ocr is False:
            query_profile["inspection_pattern"] = "asr_anchor_then_vlm"
        elif run_asr is True and run_ocr is True and run_vlm is False:
            query_profile["inspection_pattern"] = "speech_first"
        elif run_ocr is True and run_vlm is True and run_asr is False:
            query_profile["inspection_pattern"] = "ocr_first"
        elif run_asr is True and run_ocr is True and run_vlm is True:
            query_profile["inspection_pattern"] = "speech_first"

        # Save explicit knob settings for debugging/results.
        policy["_explicit_knobs"] = {
            "proposal_strategy": proposal_strategy,
            "run_clip_probe": run_clip_probe,
            "run_asr": run_asr,
            "run_ocr": run_ocr,
            "run_vlm": run_vlm,
            "run_cheap_text_stage": run_cheap_text_stage,
            "probe_fps": probe_fps,
            "probe_seg_len_s": probe_seg_len_s,
            "probe_topk": probe_topk,
            "window_len_s": window_len_s,
            "action_topk": action_topk,
            "max_vlm_windows": max_vlm_windows,
            "answer_max_frames_per_window": answer_max_frames_per_window,
            "answer_max_images_total": answer_max_images_total,
        }


        proposal_artifacts = proposal.artifacts or {}

        print("[PROPOSAL_DEBUG] proposal.artifacts keys:", list(proposal_artifacts.keys()))
        print("[PROPOSAL_DEBUG] proposal.artifacts:", proposal_artifacts)

        candidates = (
            proposal_artifacts.get("candidates")
            or proposal_artifacts.get("windows")
            or proposal_artifacts.get("proposals")
            or []
        )

        actions = (
            proposal_artifacts.get("actions")
            or proposal_artifacts.get("inspect_actions")
            or candidates
            or []
        )

        # Final fallback: make uniform 5-second windows over the video.
        # This lets METIS knob experiments run even if InitialProposalRunner is not producing candidates.
        if not candidates or run_clip_probe is False:
            candidates = self._build_coarse_chunks(
                duration_s=duration_s,
                chunk_len_s=float(policy.get("window_len_s", 5.0)),
                )

            actions = candidates

        if not actions:
            actions = candidates
            
        # ------------------------------------------------------------
        # Optional CLIP ranking before selecting top-k visual windows.
        # This is the point where coarse proposal windows become ranked
        # candidate/actions for VLM.
        # ------------------------------------------------------------

        should_clip_rank = (
            run_clip_probe is not False
            and str(policy.get("proposal_strategy", proposal_strategy or "clip_first")) == "clip_first"
            and bool(candidates)
        )
        print(
            f"[CLIP_RANK_ENABLED] should_clip_rank={should_clip_rank} "
            f"num_candidates={len(candidates)} "
            f"proposal_strategy={policy.get('proposal_strategy')} "
            f"run_clip_probe={run_clip_probe}"
        )

        if should_clip_rank:
            try:
                print(
                    f"[CLIP_RANK_ENABLED] "
                    f"num_candidates={len(candidates)} "
                    f"proposal_strategy={policy.get('proposal_strategy')} "
                    f"run_clip_probe={run_clip_probe}"
                )

                candidates, actions = self._clip_rank_candidates_and_actions(
                    question=question,
                    video_path=video_path,
                    candidates=candidates,
                    policy=policy,
                    duration_s=duration_s,
                )

                print(
                    f"[CLIP_RANK_DONE] "
                    f"num_ranked_candidates={len(candidates)} "
                    f"num_actions={len(actions)}"
                )

            except Exception as e:
                import traceback
                print(f"[CLIP_RANK_ERROR] {type(e).__name__}: {e}")
                traceback.print_exc()

                # Fallback: do not kill the whole run.
                # Use unranked coarse candidates/actions.
                if not actions:
                    actions = candidates

        #which tools the pipeline should prefer for this query
        preferred_tools = tuple(
            "vlm" if x == "clip" else x
            for x in policy.get("preferred_tools", ("vlm",))
        )
        
        #highest level workflow the profiler thinks should be used
        inspection_pattern = str(query_profile.get("inspection_pattern", "vlm_only"))
        reference_type = str(query_profile.get("reference_type", "none"))
        answer_type = str(query_profile.get("answer_type", "visual_answer"))

        # Keep this only for old function signatures.
        temporal_requirement = "none"

        probe_topk = int(policy.get("probe_topk", 8))
        action_topk = int(policy.get("action_topk", 8))
        max_vlm_windows_policy = int(policy.get("max_vlm_windows", action_topk))
        probe_fps = float(policy.get("probe_fps", 1.0))

        answer_max_frames_per_window_policy = int(
            policy.get("answer_max_frames_per_window", 2)
        )

        answer_max_images_total_policy = int(
            policy.get("answer_max_images_total", answer_max_frames_per_window_policy * max_vlm_windows_policy)
        )

        window_len_s_policy = float(policy.get("window_len_s", 5.0))

        # Effective number of visual windows allowed.
        effective_max_vlm_windows = min(action_topk, max_vlm_windows_policy)

        # Effective total image budget.
        # This prevents "top4" with 2 frames/window from accidentally exceeding the image budget.
        effective_frames_per_window = max(
            1,
            min(
                answer_max_frames_per_window_policy,
                max(1, answer_max_images_total_policy // max(1, effective_max_vlm_windows)),
            ),
        )

        if inspection_pattern == "speech_first":
            probe_candidates = candidates
        else:
            probe_candidates = candidates[: min(len(candidates), probe_topk)]

        # inspect_actions = actions[: min(len(actions), action_topk, max_vlm_windows_policy)]
        inspect_actions = actions[: min(len(actions), effective_max_vlm_windows)]
        print("\n[VLM_SELECTED_ACTIONS]")
        for a in inspect_actions:
            print(
                f"window=({float(a['t0']):.2f},{float(a['t1']):.2f}) "
                f"score={float(a.get('score', 0.0)):.4f} "
                f"source={a.get('source')}"
            )

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
                answer_max_frames_per_window=effective_frames_per_window,
            )
        
        if policy_mode == "manual_one_pass":
            manual_tools = []

            if run_asr:
                manual_tools.append("asr")
            if run_ocr:
                manual_tools.append("ocr")
            if run_vlm:
                manual_tools.append("vlm")

            return await self._run_default_one_pass(
                wf=wf,
                profile_id=profile_id,
                proposal_id=proposal_id,
                question=question,
                prof=prof,
                proposal=proposal,
                preferred_tools=tuple(manual_tools),
                reference_type=reference_type,
                answer_type=answer_type,
                temporal_requirement=temporal_requirement,
                probe_candidates=probe_candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
                inspection_pattern=f"manual_one_pass_{'+'.join(manual_tools) if manual_tools else 'none'}",
                answer_max_frames_per_window=effective_frames_per_window,
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
                probe_candidates=candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
                visual_max_frames=effective_frames_per_window,
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
                probe_candidates=candidates,
                inspect_actions=inspect_actions,
                probe_fps=probe_fps,
                visual_max_frames=baseline_visual_max_frames,
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


        if policy_mode == "baseline_joint_batched_asr_60f":
            return await self._run_baseline_joint_batched_asr_60f(
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
                visual_max_frames=effective_frames_per_window,
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
                visual_max_frames=effective_frames_per_window,
            )
        
        if inspection_pattern == "asr_anchor_then_vlm":
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

        if inspection_pattern == "asr_only":
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
                visual_max_frames=effective_frames_per_window,
            )

        if inspection_pattern == "vlm_only":
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
                answer_max_frames_per_window=effective_frames_per_window,
            )

        if inspection_pattern == "ocr_only":
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
                visual_max_frames=effective_frames_per_window,
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
            answer_max_frames_per_window=effective_frames_per_window,
        )

    # def _get_anchor_text(self, *, prof, question: str) -> str:
    #     artifacts = prof.artifacts or {}

    #     print("[ANCHOR_DEBUG] prof.artifacts keys:", list(artifacts.keys()))

    #     # Case 1: analysis stored directly.
    #     analysis = artifacts.get("analysis", {})
    #     if isinstance(analysis, dict):
    #         clause = str(analysis.get("primary_question_clause", "")).strip()
    #         if clause:
    #             return clause

    #     # Case 2: raw_json.analysis.
    #     raw_json = artifacts.get("raw_json", {})
    #     if isinstance(raw_json, dict):
    #         raw_analysis = raw_json.get("analysis", {})
    #         if isinstance(raw_analysis, dict):
    #             clause = str(raw_analysis.get("primary_question_clause", "")).strip()
    #             if clause:
    #                 return clause

    #     # Case 3: maybe profile runner stored profiler_result dict nested.
    #     profiler_result = artifacts.get("profiler_result", {})
    #     if isinstance(profiler_result, dict):
    #         analysis = profiler_result.get("analysis", {})
    #         if isinstance(analysis, dict):
    #             clause = str(analysis.get("primary_question_clause", "")).strip()
    #             if clause:
    #                 return clause

    #         raw_json = profiler_result.get("raw_json", {})
    #         if isinstance(raw_json, dict):
    #             raw_analysis = raw_json.get("analysis", {})
    #             if isinstance(raw_analysis, dict):
    #                 clause = str(raw_analysis.get("primary_question_clause", "")).strip()
    #                 if clause:
    #                     return clause

    #     # Case 4: query_profile may not have clause, but print to debug.
    #     print("[ANCHOR_DEBUG] could not find primary_question_clause.")
    #     print("[ANCHOR_DEBUG] prof.artifacts:", artifacts)

    #     return question

    def _fallback_extract_anchor_from_question(self, question: str) -> str:
        q = question.strip().rstrip("?")
        low = q.lower()

        markers = [
            "when the speaker says",
            "while the speaker says",
            "speaker says",
            "speaker said",
            "narrator says",
            "narrator said",
            "person says",
            "person said",
            "he says",
            "she says",
            "they say",
        ]

        for marker in markers:
            if marker in low:
                idx = low.index(marker) + len(marker)
                anchor = q[idx:].strip(" :\"'")
                if anchor:
                    return anchor

        return q


    def _get_anchor_text(self, *, prof, question: str) -> str:
        artifacts = prof.artifacts or {}

        print("[ANCHOR_DEBUG] prof.artifacts keys:", list(artifacts.keys()))

        analysis = artifacts.get("analysis", {})
        if isinstance(analysis, dict):
            print("[ANCHOR_DEBUG] analysis:", analysis)

            for key in [
                "speech_anchor",
                "spoken_anchor",
                "anchor_text",
                "primary_question_clause",
            ]:
                clause = str(analysis.get(key, "")).strip()

                # Only trust the profiler field if it is not just the full question.
                if clause and clause.lower() != question.lower().strip():
                    return clause

        # Fallback: extract phrase after "speaker says ..."
        return self._fallback_extract_anchor_from_question(question)

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
        visual_max_frames: int = 4,
    ) -> dict:
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        if True:
            sid = self.engine.add_stage(
                wf.workflow_id,
                StageType.ASR_LOCALIZE_BATCH,
                payload={
                    "windows": [{"t0": w["t0"], "t1": w["t1"]} for w in probe_candidates]
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

        if "ocr" in preferred_tools and reference_type == "text_reference":
            for w in probe_candidates[:4]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={"t0": w["t0"], "t1": w["t1"], "fps": 1.0},
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
        else:
            ocr_results = []

        refined_windows = self._refine_from_asr_results(
            asr_results=asr_results,
            duration_s=duration_s,
            question=question,
            max_windows=4,
        )

        if "vlm" in preferred_tools:
            for w in refined_windows:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": int(visual_max_frames),
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

        answer_text = answer_res.artifacts.get("answer")

        if answer_text is None:

            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = [profile_id, proposal_id] + map_stage_ids + [fuse_id, answer_id]
        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )
        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            policy_mode="speech_first_targeted",
            video_id=wf.video_path,
            question=question,
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
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
                "ocr_results": ocr_results,
                "num_asr_stage_jobs": len(asr_stage_ids),
                "num_asr_results": len(asr_results),
                "stage_timing": stage_timing,
                "stage_timing_csv": stage_timing_csv,

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
        visual_max_frames: int,
    ) -> dict:
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        if "ocr" in preferred_tools:
            for w in probe_candidates[:8]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={"t0": w["t0"], "t1": w["t1"], "fps": 1.0},
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
                    payload={"t0": w["t0"], "t1": w["t1"]},
                    depends_on=[proposal_id],
                )
                asr_stage_ids.append(sid)

            if asr_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)

        refined_windows = self._refine_from_ocr_results(
            ocr_results=ocr_results,
            duration_s=duration_s,
            max_windows=4,
        )

        if "vlm" in preferred_tools:
            for w in refined_windows:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": int(visual_max_frames),
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

        answer_text = answer_res.artifacts.get("answer")

        if answer_text is None:

            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = [profile_id, proposal_id] + map_stage_ids + [fuse_id, answer_id]
        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )
        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            policy_mode="ocr_first",
            video_id=wf.video_path,
            question=question,
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
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
                "stage_timing": stage_timing,
                "stage_timing_csv": stage_timing_csv,
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
        answer_max_frames_per_window: int = 4,
    ) -> dict:
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []
        map_stage_ids: list[str] = []

        if "asr" in preferred_tools:
            for w in probe_candidates:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.ASR_LOCALIZE,
                    payload={"t0": w["t0"], "t1": w["t1"]},
                    depends_on=[proposal_id],
                )
                asr_stage_ids.append(sid)
                map_stage_ids.append(sid)

        if "ocr" in preferred_tools:
            for w in probe_candidates[:8]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={"t0": w["t0"], "t1": w["t1"], "fps": 1.0},
                    depends_on=[proposal_id],
                )
                ocr_stage_ids.append(sid)
                map_stage_ids.append(sid)

        if "vlm" in preferred_tools:
            for a in inspect_actions:
                print(
                    f"[SEND_TO_VLM] window=({float(a['t0']):.2f},{float(a['t1']):.2f}) "
                    f"score={float(a.get('score', 0.0)):.4f} "
                    f"source={a.get('source')}"
                )
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": a["t0"],
                        "t1": a["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": answer_max_frames_per_window,
                    },
                    depends_on=[proposal_id],
                )
                visual_stage_ids.append(sid)
                map_stage_ids.append(sid)

        if map_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, map_stage_ids)

        visual_results = self._collect_visual_results(
                workflow_id=wf.workflow_id,
                visual_stage_ids=visual_stage_ids,
                    ) if visual_stage_ids else []
        total_decoded_frames = sum(v["num_decoded_frames"] for v in visual_results)
        total_sampled_frames = sum(v["num_sampled_frames"] for v in visual_results)
        total_decode_s = sum(v["decode_s"] for v in visual_results)
        total_sample_s = sum(v["sample_s"] for v in visual_results)
        total_request_s = sum(v["request_s"] for v in visual_results)
        total_visual_stage_wall_s = sum(v["visual_stage_wall_s"] for v in visual_results)

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

        answer_text = answer_res.artifacts.get("answer")

        if answer_text is None:

            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = [profile_id, proposal_id] + map_stage_ids + [fuse_id, answer_id]
        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )
        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            policy_mode=inspection_pattern,
            video_id=wf.video_path,
            question=question,
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
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
                "stage_timing": stage_timing,
                "stage_timing_csv": stage_timing_csv,
                "visual_results": visual_results,
                "total_visual_frames_sent": total_sampled_frames,
                "total_decoded_frames": total_decoded_frames,
                "total_sampled_frames": total_sampled_frames,
                "total_decode_s": total_decode_s,
                "total_sample_s": total_sample_s,
                "total_request_s": total_request_s,
                "total_visual_stage_wall_s": total_visual_stage_wall_s,
                "answer_max_frames_per_window": int(answer_max_frames_per_window),
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
        scored = []

        for res in asr_results:
            transcript = str(res.get("transcript", "")).strip()
            if not transcript:
                continue

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

            refined.append(
                {
                    "t0": round(win_t0, 3),
                    "t1": round(win_t1, 3),
                    "score": res["score"],
                    "source_t0": t0,
                    "source_t1": t1,
                    "source_text": res["transcript"],
                }
            )
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
                refined.append(
                    {
                        "t0": round(t0, 3),
                        "t1": round(t1, 3),
                        "score": base["score"],
                        "source_t0": base["source_t0"],
                        "source_t1": base["source_t1"],
                        "source_text": base["source_text"],
                    }
                )

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

            refined.append({"t0": round(win_t0, 3), "t1": round(win_t1, 3)})

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
            out.append(
                {
                    "stage_id": sid,
                    "t0": float(res.artifacts["t0"]),
                    "t1": float(res.artifacts["t1"]),
                    "transcript": str(res.artifacts.get("transcript", "")).strip(),
                    "segments": res.artifacts.get("segments", []),
                }
            )
        return out

    def _collect_ocr_results(self, *, workflow_id: str, ocr_stage_ids: list[str]) -> list[dict]:
        out = []
        for sid in ocr_stage_ids:
            res = self.engine.get_result(workflow_id, sid)
            out.append(
                {
                    "stage_id": sid,
                    "t0": float(res.artifacts["t0"]),
                    "t1": float(res.artifacts["t1"]),
                    "texts": res.artifacts.get("texts", []),
                    "joined_text": str(res.artifacts.get("joined_text", "")).strip(),
                }
            )
        return out
    
    def _collect_visual_results(self, *, workflow_id: str, visual_stage_ids: list[str]) -> list[dict]:
        out = []
        for sid in visual_stage_ids:
            res = self.engine.get_result(workflow_id, sid)
            a = res.artifacts
            out.append({
                "stage_id": sid,
                "t0": float(a.get("t0", 0.0)),
                "t1": float(a.get("t1", 0.0)),
                "fps": a.get("fps"),
                "max_frames": a.get("max_frames"),
                "window_span_s": a.get("window_span_s", 0.0),
                "num_decoded_frames": a.get("num_decoded_frames", 0),
                "num_sampled_frames": a.get("num_sampled_frames", 0),
                "decode_s": a.get("decode_s", 0.0),
                "sample_s": a.get("sample_s", 0.0),
                "request_s": a.get("request_s", 0.0),
                "visual_stage_wall_s": a.get("visual_stage_wall_s", 0.0),
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
        visual_max_frames: int = 4,
    ) -> dict:
        asr_stage_ids: list[str] = []
        ocr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        asr_results = []

        if "asr" in preferred_tools:
            sid = self.engine.add_stage(
                wf.workflow_id,
                StageType.ASR_LOCALIZE_BATCH,
                payload={"windows": [{"t0": w["t0"], "t1": w["t1"]} for w in probe_candidates]},
                depends_on=[proposal_id],
            )
            asr_stage_ids.append(sid)

        if asr_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)
            asr_results = self._collect_asr_batch_results(
                workflow_id=wf.workflow_id,
                asr_stage_ids=asr_stage_ids,
            )

        if "ocr" in preferred_tools:
            for w in probe_candidates[:8]:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.OCR_WINDOW,
                    payload={"t0": w["t0"], "t1": w["t1"], "fps": 1.0},
                    depends_on=asr_stage_ids[:] if asr_stage_ids else [proposal_id],
                )
                ocr_stage_ids.append(sid)

            if ocr_stage_ids:
                await self.engine.run_ready_batch(wf.workflow_id, ocr_stage_ids)

        if "vlm" in preferred_tools:
            for a in inspect_actions:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": a["t0"],
                        "t1": a["t1"],
                        "query": question,
                        "fps": probe_fps,
                        "max_frames": int(visual_max_frames),
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

        answer_text = answer_res.artifacts.get("answer")

        if answer_text is None:

            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = [profile_id, proposal_id] + map_stage_ids + [fuse_id, answer_id]
        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )
        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            policy_mode="baseline_asr_then_full_visual",
            video_id=wf.video_path,
            question=question,
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
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
                "stage_timing": stage_timing,
                "stage_timing_csv": stage_timing_csv,
            },
        }

    def _build_coarse_chunks(
        self,
        *,
        duration_s: float,
        chunk_len_s: float = 5,
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
        visual_max_frames: int,
    ) -> dict:
        asr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        coarse_chunks = self._build_coarse_chunks(
            duration_s=duration_s,
            chunk_len_s=5.0,
        )

        if True:
            for w in coarse_chunks:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.ASR_LOCALIZE,
                    payload={"t0": w["t0"], "t1": w["t1"]},
                    depends_on=[proposal_id],
                )
                asr_stage_ids.append(sid)

        if True:
            for w in coarse_chunks:
                sid = self.engine.add_stage(
                    wf.workflow_id,
                    StageType.VISUAL_INSPECT,
                    payload={
                        "t0": w["t0"],
                        "t1": w["t1"],
                        "query": question,
                        "fps": 1.0,
                        "max_frames": int(visual_max_frames),
                    },
                    depends_on=[proposal_id],
                )
                visual_stage_ids.append(sid)

        if asr_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, asr_stage_ids)

        if visual_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, visual_stage_ids)

        collect_t0 = time.time()

        visual_results = self._collect_visual_results(
                workflow_id=wf.workflow_id,
                visual_stage_ids=visual_stage_ids,
            ) if visual_stage_ids else []
        
        collect_visual_s = time.time() - collect_t0

        total_decoded_frames = sum(v["num_decoded_frames"] for v in visual_results)
        total_sampled_frames = sum(v["num_sampled_frames"] for v in visual_results)
        total_decode_s = sum(v["decode_s"] for v in visual_results)
        total_sample_s = sum(v["sample_s"] for v in visual_results)
        total_request_s = sum(v["request_s"] for v in visual_results)
        total_visual_stage_wall_s = sum(v["visual_stage_wall_s"] for v in visual_results)
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
        answer_text = answer_res.artifacts.get("answer")
        if answer_text is None:
            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = [profile_id, proposal_id] + map_stage_ids + [fuse_id, answer_id]
        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )
        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            policy_mode="baseline_joint_heavy_60f",
            video_id=wf.video_path,
            question=question,
        )

        workflow_summary_csv = self._summarize_workflow_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            visual_results=visual_results,
            policy_mode="baseline_joint_heavy_60f",  # or encoder_aware_60f
            video_id=wf.video_path,
            question=question,
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
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
                "coarse_chunk_len_s": 5.0,
                "visual_fps": 1.0,
                "visual_max_frames": int(visual_max_frames),
                "total_visual_frames_sent": total_sampled_frames,
                "stage_timing": stage_timing,
                "stage_timing_csv": stage_timing_csv,
                "visual_results": visual_results,
                "total_decoded_frames": total_decoded_frames,
                "total_sampled_frames": total_sampled_frames,
                "total_decode_s": total_decode_s,
                "total_sample_s": total_sample_s,
                "total_request_s": total_request_s,
                "total_visual_stage_wall_s": total_visual_stage_wall_s,
                "workflow_summary_csv": workflow_summary_csv,
            },
        }
    
    async def _run_baseline_joint_batched_asr_60f(
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
        visual_max_frames: int,
    ) -> dict:
        asr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        coarse_chunks = self._build_coarse_chunks(
            duration_s=duration_s,
            chunk_len_s=5.0,
        )

        # Full-video ASR, but submitted as ONE batched stage.
        asr_sid = self.engine.add_stage(
            wf.workflow_id,
            StageType.ASR_LOCALIZE_BATCH,
            payload={
                "windows": [
                    {"t0": w["t0"], "t1": w["t1"]}
                    for w in coarse_chunks
                ]
            },
            depends_on=[proposal_id],
        )
        asr_stage_ids.append(asr_sid)

        await self.engine.run_stage(wf.workflow_id, asr_sid)

        asr_results = self._collect_asr_batch_results(
            workflow_id=wf.workflow_id,
            asr_stage_ids=asr_stage_ids,
        )

        # Full-video visual sweep, same as baseline.
        for w in coarse_chunks:
            sid = self.engine.add_stage(
                wf.workflow_id,
                StageType.VISUAL_INSPECT,
                payload={
                    "t0": w["t0"],
                    "t1": w["t1"],
                    "query": question,
                    "fps": 1.0,
                    "max_frames": int(visual_max_frames),
                },
                depends_on=[proposal_id],
            )
            visual_stage_ids.append(sid)

        if visual_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, visual_stage_ids)

        visual_results = self._collect_visual_results(
            workflow_id=wf.workflow_id,
            visual_stage_ids=visual_stage_ids,
        ) if visual_stage_ids else []

        total_decoded_frames = sum(v["num_decoded_frames"] for v in visual_results)
        total_sampled_frames = sum(v["num_sampled_frames"] for v in visual_results)
        total_decode_s = sum(v["decode_s"] for v in visual_results)
        total_sample_s = sum(v["sample_s"] for v in visual_results)
        total_request_s = sum(v["request_s"] for v in visual_results)
        total_visual_stage_wall_s = sum(v["visual_stage_wall_s"] for v in visual_results)

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

        answer_text = answer_res.artifacts.get("answer")
        if answer_text is None:
            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = [profile_id, proposal_id] + map_stage_ids + [fuse_id, answer_id]

        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )

        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            policy_mode="baseline_joint_batched_asr_60f",
            video_id=wf.video_path,
            question=question,
        )

        workflow_summary_csv = self._summarize_workflow_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            visual_results=visual_results,
            policy_mode="baseline_joint_batched_asr_60f",
            video_id=wf.video_path,
            question=question,
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": "baseline_joint_batched_asr_60f",
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,
                "num_asr_stages": len(asr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
                "num_asr_results": len(asr_results),
                "coarse_chunk_len_s": 5.0,
                "visual_fps": 1.0,
                "visual_max_frames": int(visual_max_frames),
                "total_visual_frames_sent": total_sampled_frames,
                "stage_timing": stage_timing,
                "stage_timing_csv": stage_timing_csv,
                "workflow_summary_csv": workflow_summary_csv,
                "visual_results": visual_results,
                "total_decoded_frames": total_decoded_frames,
                "total_sampled_frames": total_sampled_frames,
                "total_decode_s": total_decode_s,
                "total_sample_s": total_sample_s,
                "total_request_s": total_request_s,
                "total_visual_stage_wall_s": total_visual_stage_wall_s,
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
        visual_max_frames: int,
    ) -> dict:
        heavy_asr_stage_ids: list[str] = []
        visual_stage_ids: list[str] = []

        heavy_asr_results = []
        visual_results = []

        cheap_rerank_s = 0.0
        heavy_rerank_s = 0.0
        collect_cheap_asr_s = 0.0
        collect_heavy_asr_s = 0.0
        collect_visual_s = 0.0
        anchor_extract_s = 0.0

        workflow_t0 = time.time()

        def mark(name: str):
            print(f"[ENCODER_AWARE_TIMER] {name}: {time.time() - workflow_t0:.3f}s")

        mark("start encoder aware")

        coarse_chunks = self._build_coarse_chunks(
            duration_s=duration_s,
            chunk_len_s=5.0,
        )

        visual_top_k = max(1, len(inspect_actions))

        # ------------------------------------------------------------
        # 1) Heavy ASR batch over all coarse chunks
        # ------------------------------------------------------------
        mark("before submit HEAVY ASR_LOCALIZE_BATCH")

        heavy_sid = self.engine.add_stage(
            wf.workflow_id,
            StageType.ASR_LOCALIZE_BATCH,
            payload={
                "windows": [
                    {"t0": w["t0"], "t1": w["t1"]}
                    for w in coarse_chunks
                ]
            },
            depends_on=[proposal_id],
        )

        heavy_asr_stage_ids.append(heavy_sid)

        await self.engine.run_stage(wf.workflow_id, heavy_sid)

        mark("after HEAVY ASR_LOCALIZE_BATCH done")

        collect_t0 = time.time()
        heavy_asr_results = self._collect_asr_batch_results(
            workflow_id=wf.workflow_id,
            asr_stage_ids=heavy_asr_stage_ids,
        )
        collect_heavy_asr_s = time.time() - collect_t0

        # ------------------------------------------------------------
        # 2) Extract anchor and rank ASR chunks
        # ------------------------------------------------------------
        anchor_t0 = time.time()
        anchor_text = self._get_anchor_text(prof=prof, question=question)
        anchor_extract_s = time.time() - anchor_t0

        print(f"[ENCODER_AWARE] anchor_text={repr(anchor_text)}")

        mark("before heavy ASR fallback rerank")

        heavy_rerank_t0 = time.time()
        ranked = self._rank_asr_chunks_fallback(
            anchor_text=anchor_text,
            asr_results=heavy_asr_results,
            top_k=visual_top_k,
        )
        heavy_rerank_s = time.time() - heavy_rerank_t0

        mark("after heavy ASR fallback rerank")

        print("\n[ENCODER_AWARE] heavy ASR chunks ranked:")
        for w in ranked:
            print(
                f"  window=({float(w['t0']):.2f},{float(w['t1']):.2f}) "
                f"score={w.get('score')} "
                f"transcript={repr(str(w.get('transcript', ''))[:160])} "
                f"reason={repr(w.get('rerank_reason', ''))}"
            )

        if not ranked:
            ranked = [
                {
                    "t0": float(w["t0"]),
                    "t1": float(w["t1"]),
                    "score": 0.0,
                    "transcript": "",
                }
                for w in coarse_chunks[:visual_top_k]
            ]

        top_chunks = ranked[:visual_top_k]

        # ------------------------------------------------------------
        # 3) Visual inspect only top chunks
        # ------------------------------------------------------------
        print("\n[ENCODER_AWARE] top_chunks selected for VLM:")
        for w in top_chunks:
            print(
                f"  window=({float(w['t0']):.2f},{float(w['t1']):.2f}) "
                f"score={w.get('score')} "
                f"transcript={repr(str(w.get('transcript', ''))[:160])}"
            )

        mark("before submit VISUAL_INSPECT")

        for w in top_chunks:
            sid = self.engine.add_stage(
                wf.workflow_id,
                StageType.VISUAL_INSPECT,
                payload={
                    "t0": max(0.0, float(w["t0"]) - 2.0),
                    "t1": min(duration_s, float(w["t1"]) + 2.0),
                    "query": question,
                    "fps": 1.0,
                    "max_frames": int(visual_max_frames),
                },
                depends_on=heavy_asr_stage_ids[:] if heavy_asr_stage_ids else [proposal_id],
            )
            visual_stage_ids.append(sid)

        if visual_stage_ids:
            await self.engine.run_ready_batch(wf.workflow_id, visual_stage_ids)

            mark("after VISUAL_INSPECT done")

            collect_t0 = time.time()
            visual_results = self._collect_visual_results(
                workflow_id=wf.workflow_id,
                visual_stage_ids=visual_stage_ids,
            )
            collect_visual_s = time.time() - collect_t0

        total_decoded_frames = sum(v["num_decoded_frames"] for v in visual_results)
        total_sampled_frames = sum(v["num_sampled_frames"] for v in visual_results)
        total_decode_s = sum(v["decode_s"] for v in visual_results)
        total_sample_s = sum(v["sample_s"] for v in visual_results)
        total_request_s = sum(v["request_s"] for v in visual_results)
        total_visual_stage_wall_s = sum(v["visual_stage_wall_s"] for v in visual_results)

        map_stage_ids = heavy_asr_stage_ids + visual_stage_ids

        # ------------------------------------------------------------
        # 4) Fuse and answer
        # ------------------------------------------------------------
        mark("before FUSE")

        fuse_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.FUSE,
            depends_on=map_stage_ids,
        )

        await self.engine.run_stage(wf.workflow_id, fuse_id)

        mark("after FUSE")

        mark("before ANSWER")

        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=[fuse_id],
        )

        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)

        mark("after ANSWER")

        answer_text = answer_res.artifacts.get("answer")
        if answer_text is None:
            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = [profile_id, proposal_id] + map_stage_ids + [fuse_id, answer_id]

        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )

        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            policy_mode="encoder_aware_60f",
            video_id=wf.video_path,
            question=question,
        )

        workflow_summary_csv = self._summarize_workflow_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            visual_results=visual_results,
            policy_mode="encoder_aware_60f",
            video_id=wf.video_path,
            question=question,
        )

        measured_orchestration_s = (
            cheap_rerank_s
            + heavy_rerank_s
            + collect_cheap_asr_s
            + collect_heavy_asr_s
            + collect_visual_s
            + anchor_extract_s
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts,
            "proposal": proposal.artifacts,
            "debug": {
                "inspection_pattern": "encoder_aware_60f",
                "reference_type": reference_type,
                "answer_type": answer_type,
                "temporal_requirement": temporal_requirement,
                "preferred_tools": preferred_tools,

                "num_cheap_asr_stages": 0,
                "num_heavy_asr_stages": len(heavy_asr_stage_ids),
                "num_visual_stages": len(visual_stage_ids),
                "num_asr_results": len(heavy_asr_results),

                "num_refined_windows": len(top_chunks),
                "refined_windows": top_chunks,

                "coarse_chunk_len_s": 5.0,
                "visual_top_k": visual_top_k,
                "visual_fps": 1.0,
                "visual_max_frames": int(visual_max_frames),

                "total_visual_frames_sent": total_sampled_frames,
                "stage_timing": stage_timing,
                "stage_timing_csv": stage_timing_csv,
                "workflow_summary_csv": workflow_summary_csv,

                "cheap_asr_results": [],
                "heavy_asr_results": heavy_asr_results,
                "visual_results": visual_results,

                "total_decoded_frames": total_decoded_frames,
                "total_sampled_frames": total_sampled_frames,
                "total_decode_s": total_decode_s,
                "total_sample_s": total_sample_s,
                "total_request_s": total_request_s,
                "total_visual_stage_wall_s": total_visual_stage_wall_s,

                "cheap_rerank_s": cheap_rerank_s,
                "heavy_rerank_s": heavy_rerank_s,
                "rerank_s": cheap_rerank_s + heavy_rerank_s,
                "collect_cheap_asr_s": collect_cheap_asr_s,
                "collect_heavy_asr_s": collect_heavy_asr_s,
                "collect_visual_s": collect_visual_s,
                "anchor_extract_s": anchor_extract_s,
                "measured_orchestration_s": measured_orchestration_s,
            },
        }
        

    def _collect_stage_timing(self, *, workflow_id: str, stage_ids: list[str]) -> list[dict]:
        out = []
        for sid in stage_ids:
            res = self.engine.get_result(workflow_id, sid)
            m = res.metrics or {}
            out.append(
                {
                    "stage_id": sid,
                    "stage_type": str(res.stage_type),
                    "creation_to_submit_s": m.get("creation_to_submit_s", 0.0),
                    "submit_to_start_s": m.get("submit_to_start_s", 0.0),
                    "queue_wait_s": m.get("queue_wait_s", 0.0),
                    "stage_wall_s": m.get("stage_wall_s", 0.0),
                    "creation_to_end_s": m.get("creation_to_end_s", 0.0),
                }
            )
        return out

    def _append_stage_timing_csv(
        self,
        *,
        workflow_id: str,
        stage_ids: list[str],
        policy_mode: str,
        video_id: str,
        question: str,
        out_path: str = "outputs/stage_timing/all_stage_timing.csv",
    ) -> str:
        rows = self._collect_stage_timing(
            workflow_id=workflow_id,
            stage_ids=stage_ids,
        )

        for r in rows:
            r["workflow_id"] = workflow_id
            r["policy_mode"] = policy_mode
            r["video_id"] = video_id
            r["question"] = question

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "workflow_id",
            "policy_mode",
            "video_id",
            "question",
            "stage_id",
            "stage_type",
            "creation_to_submit_s",
            "submit_to_start_s",
            "queue_wait_s",
            "stage_wall_s",
            "creation_to_end_s",
        ]

        file_exists = path.exists()

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

        return str(path)
    
    def _collect_cheap_visual_results(self, *, workflow_id: str, cheap_stage_ids: list[str]) -> list[dict]:
        out = []
        for sid in cheap_stage_ids:
            res = self.engine.get_result(workflow_id, sid)
            a = res.artifacts
            out.append(
                {
                    "stage_id": sid,
                    "t0": float(a.get("t0", 0.0)),
                    "t1": float(a.get("t1", 0.0)),
                    "cheap_score": float(a.get("cheap_score", 0.0)),
                    "cheap_caption": str(a.get("cheap_caption", "")),
                    "num_decoded_frames": a.get("num_decoded_frames", 0),
                    "num_sampled_frames": a.get("num_sampled_frames", 0),
                }
            )
        return out
    
    def _summarize_workflow_timing(
    self,
    *,
    workflow_id: str,
    stage_ids: list[str],
    visual_results: list[dict],
    policy_mode: str,
    video_id: str,
    question: str,
    out_path: str = "outputs/stage_timing/workflow_summary.csv",
) -> str:
        rows = self._collect_stage_timing(
            workflow_id=workflow_id,
            stage_ids=stage_ids,
        )

        if not rows:
            return ""

        # End-to-end latency as observed by the engine timeline.
        # This assumes creation_to_end_s is measured relative to stage creation.
        # For cleaner measurement, also measure around orchestrator.run externally.
        e2e_s = max(float(r.get("creation_to_end_s", 0.0)) for r in rows)

        total_stage_wall_s = sum(float(r.get("stage_wall_s", 0.0)) for r in rows)
        total_queue_wait_s = sum(float(r.get("queue_wait_s", 0.0)) for r in rows)

        by_type = {}
        for r in rows:
            st = str(r.get("stage_type", "unknown"))
            by_type[st] = by_type.get(st, 0.0) + float(r.get("stage_wall_s", 0.0))

        total_visual_frames = sum(int(v.get("num_sampled_frames", 0)) for v in visual_results)
        total_decoded_frames = sum(int(v.get("num_decoded_frames", 0)) for v in visual_results)
        total_visual_request_s = sum(float(v.get("request_s", 0.0)) for v in visual_results)
        total_visual_wall_s = sum(float(v.get("visual_stage_wall_s", 0.0)) for v in visual_results)
        total_decode_s = sum(float(v.get("decode_s", 0.0)) for v in visual_results)
        total_sample_s = sum(float(v.get("sample_s", 0.0)) for v in visual_results)

        summary = {
            "workflow_id": workflow_id,
            "policy_mode": policy_mode,
            "video_id": video_id,
            "question": question,
            "num_stages": len(rows),
            "num_visual_jobs": len(visual_results),
            "total_sampled_frames": total_visual_frames,
            "total_decoded_frames": total_decoded_frames,
            "total_visual_request_s": total_visual_request_s,
            "total_visual_stage_wall_s": total_visual_wall_s,
            "total_decode_s": total_decode_s,
            "total_sample_s": total_sample_s,
            "workflow_e2e_s": e2e_s,
            "total_stage_wall_s": total_stage_wall_s,
            "total_queue_wait_s": total_queue_wait_s,
            "profile_wall_s": by_type.get(str(StageType.PROFILE_QUERY), 0.0),
            "proposal_wall_s": by_type.get(str(StageType.INITIAL_PROPOSAL), 0.0),
            "asr_wall_s": (
                by_type.get(str(StageType.ASR_LOCALIZE), 0.0)
                + by_type.get(str(StageType.ASR_LOCALIZE_BATCH), 0.0)
                + by_type.get(str(StageType.ASR_LOCALIZE_CHEAP), 0.0)
            ),
            "visual_wall_s": by_type.get(str(StageType.VISUAL_INSPECT), 0.0),
            "fuse_wall_s": by_type.get(str(StageType.FUSE), 0.0),
            "answer_wall_s": by_type.get(str(StageType.ANSWER), 0.0),
        }

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(summary.keys())
        file_exists = path.exists()

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

        return str(path)


    def _rank_asr_chunks_llm(
            self,
            *,
            anchor_text: str,
            asr_results: list[dict],
            profiler_base_url: str = "http://localhost:8003/v1",
            profiler_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
            top_k: int = 4,
            timeout_s: float = 120.0,
        ) -> list[dict]:
        """
        LLM-based semantic reranker for ASR chunks.

        Returns ASR chunks sorted by semantic match to anchor_text.
        No hardcoded keywords.
        """
        if not asr_results:
            return []

        candidates = []
        for i, r in enumerate(asr_results):
            transcript = str(r.get("transcript", "")).strip()
            candidates.append({
                "idx": i,
                "t0": float(r.get("t0", 0.0)),
                "t1": float(r.get("t1", 0.0)),
                "transcript": transcript,
            })

        prompt = {
            "anchor_text": anchor_text,
            "task": (
                "Rank the transcript chunks by how likely they contain the spoken anchor. "
                "Use semantic meaning, paraphrases, and ASR errors. Do not rely only on exact keywords."
            ),
            "output_schema": {
                "ranked": [
                    {
                        "idx": "integer candidate idx",
                        "score": "float 0 to 1",
                        "reason": "short reason"
                    }
                ]
            },
            "candidates": candidates,
        }

        url = profiler_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": profiler_model,
            "temperature": 0,
            "max_tokens": 768,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an ASR chunk reranker for video QA. "
                        "Return STRICT JSON only. No markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt),
                },
            ],
        }

        r = requests.post(
            url,
            json=payload,
            timeout=timeout_s,
            headers={"Authorization": "Bearer EMPTY"},
        )
        r.raise_for_status()

        text = r.json()["choices"][0]["message"]["content"].strip()
        parsed = self._parse_json_object_lenient(text)

        ranked = parsed.get("ranked", [])
        by_idx = {i: dict(r) for i, r in enumerate(asr_results)}

        out = []
        for item in ranked:
            idx = int(item.get("idx", -1))
            if idx not in by_idx:
                continue

            rr = by_idx[idx]
            rr["score"] = float(item.get("score", 0.0))
            rr["rerank_reason"] = str(item.get("reason", ""))
            out.append(rr)

        # Fallback if LLM returns bad JSON or empty ranking.
        if not out:
            out = []
            for i, r0 in enumerate(asr_results):
                rr = dict(r0)
                rr["score"] = 0.0
                rr["rerank_reason"] = "fallback_unranked"
                out.append(rr)

        out.sort(key=lambda x: (-float(x.get("score", 0.0)), float(x.get("t0", 0.0))))
        return out[:top_k]

    
    def _parse_json_object_lenient(self, text: str) -> dict:
        text = text.strip()

        print("[ASR_RERANKER_RAW]", repr(text[:1000]))

        if text.startswith("```"):
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                return {"ranked": obj}
        except Exception:
            pass

        decoder = json.JSONDecoder()

        for i, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                obj, _end = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, list):
                    return {"ranked": obj}
            except Exception:
                continue

        raise RuntimeError(f"Could not parse JSON from reranker output: {text[:1000]}")
    
    def _rank_asr_chunks_fallback(
        self,
        *,
        anchor_text: str,
        asr_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        def toks(s: str) -> set[str]:
            stop = {
                "what", "which", "when", "where", "who", "why", "how",
                "the", "a", "an", "is", "are", "was", "were", "to",
                "of", "in", "on", "at", "and", "or", "this", "that",
                "speaker", "says", "said", "shown", "visual", "object", "scene",
            }

            out = set()
            for w in s.lower().replace("?", " ").split():
                w = w.strip(".,:;!\"'()[]{}")
                if len(w) > 2 and w not in stop:
                    out.add(w)
            return out

        anchor_tokens = toks(anchor_text)
        ranked = []

        for r in asr_results:
            transcript_tokens = toks(str(r.get("transcript", "")))
            score = len(anchor_tokens & transcript_tokens) / max(1, len(anchor_tokens))

            rr = dict(r)
            rr["score"] = float(score)
            rr["rerank_reason"] = "fallback_token_overlap"
            ranked.append(rr)

        ranked.sort(key=lambda x: (-float(x.get("score", 0.0)), float(x.get("t0", 0.0))))
        return ranked[:top_k]

    def _clip_rank_candidates_and_actions(
        self,
        *,
        question: str,
        video_path: str,
        candidates: list[dict],
        policy: dict,
        duration_s: float,
    ) -> tuple[list[dict], list[dict]]:
        """
        Rank coarse candidate windows with CLIP, then rebuild inspect actions
        from the ranked windows.

        Input candidates are dicts:
        {"t0": ..., "t1": ..., "score": ..., "source": ...}

        Output candidates/actions are also dicts so the rest of orchestrator
        can keep using a["t0"], a["t1"], etc.
        """
        candidate_objs = [
            CandidateWindow(
                t0=float(c["t0"]),
                t1=float(c["t1"]),
                score=float(c.get("score", 0.0)),
                source=str(c.get("source", "coarse")),
            )
            for c in candidates
        ]

        if self.decoder is None:
            print("[CLIP_RANK_ERROR] self.decoder is None; falling back to unranked actions")

            actions = build_action_space(
                candidate_objs,
                window_len_s=float(policy.get("window_len_s", 5.0)),
                topk=int(policy.get("action_topk", 8)),
            )

            action_dicts = [a.__dict__ for a in actions]

            for a in action_dicts:
                a["t0"] = max(0.0, float(a["t0"]))
                a["t1"] = min(float(duration_s), float(a["t1"]))

            return candidates, action_dicts

        ranked = self.clip_ranker.rank_windows(
            question=question,
            video_path=video_path,
            windows=candidate_objs,
            decoder=self.decoder,
            frames_per_window=int(policy.get("clip_frames_per_window", 2)),
            fps=float(policy.get("probe_fps", 1.0)),
        )

        action_topk = int(policy.get("action_topk", 8))
        window_len_s = float(policy.get("window_len_s", 5.0))

        actions = build_action_space(
            ranked,
            window_len_s=window_len_s,
            topk=action_topk,
        )

        ranked_dicts = [c.__dict__ for c in ranked]
        action_dicts = [a.__dict__ for a in actions]
        for a in action_dicts:
            a["t0"] = max(0.0, float(a["t0"]))
            a["t1"] = min(float(duration_s), float(a["t1"]))

        print("\n[CLIP_RANKED_CANDIDATES]")
        for c in ranked_dicts[:10]:
            print(
                f"window=({float(c['t0']):.2f},{float(c['t1']):.2f}) "
                f"score={float(c.get('score', 0.0)):.4f} "
                f"source={c.get('source')}"
            )

        return ranked_dicts, action_dicts