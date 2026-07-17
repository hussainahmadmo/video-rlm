from __future__ import annotations

import csv
import json
import requests
from pathlib import Path
import time
from conductor.engine.stage_types import StageType
from conductor.retrieval.clip_window_ranker import CLIPWindowRanker
from conductor.retrieval.window_proposal import CandidateWindow, build_action_space
from conductor.profiler.llm_profiler import (
    profile_query_llm,
    ResourceState,
)

from conductor.engine.services.hierarchical_retriever import (
    HierarchicalRetriever,
    RetrievalWindow,
)
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

        probe_fps: float | None = None,
        probe_topk: int | None = None,
        action_topk: int | None = None,
        window_len_s: float | None = None,

        answer_max_frames_per_window: int | None = None,
        answer_max_images_total: int | None = None,

        baseline_visual_max_frames: int = 60,
        encoder_visual_max_frames: int = 60,
    ) -> dict:


        if policy_mode == "baseline_full":
            raise ValueError("baseline_full is not implemented in this VIMIO-only orchestrator")

        if policy_mode not in {"auto", "override"}:
            raise ValueError(f"Unknown policy_mode={policy_mode!r}")
        
        wf = self.engine.create_workflow(
            question=question,
            video_path=video_path,
            metadata={"duration_s": duration_s},
        )

        profile_id = None
        prof = None
        profile_artifacts: dict = {}

        if policy_mode == "auto":
            # Normal VIMIO: run the profiler and use its compiled policy.
            profile_id = self.engine.add_stage(
                wf.workflow_id,
                StageType.PROFILE_QUERY,
                payload={
                    "duration_s": duration_s,
                    "duration_min": duration_s / 60.0,
                },
            )

            await self.engine.run_stage(
                wf.workflow_id,
                profile_id,
            )

            prof = self.engine.get_result(
                wf.workflow_id,
                profile_id,
            )

            profile_artifacts = prof.artifacts or {}

            print(
                "[PROFILE_DEBUG] prof.artifacts keys:",
                list(profile_artifacts.keys()),
            )
            print(
                "[PROFILE_DEBUG] prof.artifacts:",
                profile_artifacts,
            )

            policy = (
                profile_artifacts.get("execution_policy")
                or profile_artifacts.get("policy")
                or profile_artifacts.get("raw_json", {}).get(
                    "execution_policy"
                )
                or profile_artifacts.get("raw_json", {}).get("policy")
                or {}
            )

            policy = dict(policy)

            policy.setdefault("probe_topk", 8)
            policy.setdefault("action_topk", 8)
            policy.setdefault("probe_fps", 1.0)
            policy.setdefault("window_len_s", 8.0)

            vlm_budget = int(
                policy.get("vlm_budget", 32)
            )

            policy.setdefault(
                "answer_max_images_total",
                vlm_budget,
            )
            policy.setdefault(
                "answer_max_frames_per_window",
                4,
            )

        else:
            # Controlled experiment: skip the profiler completely.
            required = {
                "probe_fps": probe_fps,
                "probe_topk": probe_topk,
                "action_topk": action_topk,
                "window_len_s": window_len_s,
                "answer_max_images_total": (
                    answer_max_images_total
                ),
            }

            missing = [
                key
                for key, value in required.items()
                if value is None
            ]

            if missing:
                raise ValueError(
                    "override mode requires explicit values for: "
                    + ", ".join(missing)
                )

            policy = {
                "probe_fps": float(probe_fps),
                "probe_topk": int(probe_topk),
                "action_topk": int(action_topk),
                "window_len_s": float(window_len_s),
                "answer_max_images_total": int(
                    answer_max_images_total
                ),
                "answer_max_frames_per_window": int(
                    answer_max_frames_per_window
                    if answer_max_frames_per_window is not None
                    else 4
                ),
            }

            print(
                "[POLICY_OVERRIDE] Skipped PROFILE_QUERY"
            )

        policy["probe_fps"] = self._coerce_policy_number(policy, "probe_fps", 1.0, float)
        policy["probe_topk"] = self._coerce_policy_number(policy, "probe_topk", 8, int)
        policy["action_topk"] = self._coerce_policy_number(policy, "action_topk", 8, int)
        policy["window_len_s"] = self._coerce_policy_number(policy, "window_len_s", 8.0, float)
        policy["answer_max_frames_per_window"] = self._coerce_policy_number(
            policy, "answer_max_frames_per_window", 4, int
        )
        policy["answer_max_images_total"] = self._coerce_policy_number(
            policy, "answer_max_images_total", 32, int
        )
        policy.setdefault("retrieval_mode", "hierarchical")
        policy.setdefault("retrieval_budget", 32)
        policy.setdefault("retrieval_coarse_count", 16)
        policy.setdefault("retrieval_parents_to_refine", 4)
        policy.setdefault("clip_frames_per_window", 1)



        # ADD THIS BLOCK HERE
        print("\n[FINAL_POLICY]")
        for k in [
            "retrieval_mode",
            "retrieval_budget",
            "retrieval_coarse_count",
            "retrieval_parents_to_refine",
            "clip_frames_per_window",
            "probe_fps",
            "probe_topk",
            "action_topk",
            "window_len_s",
            "answer_max_frames_per_window",
            "answer_max_images_total",
        ]:
            
            print(f"{k}: {policy.get(k)}")

        policy["_explicit_knobs"] = {
            "probe_fps": probe_fps,
            "probe_topk": probe_topk,
            "action_topk": action_topk,
            "window_len_s": window_len_s,
            "answer_max_frames_per_window": answer_max_frames_per_window,
            "answer_max_images_total": answer_max_images_total,
        }

        # Now create proposal using profiler-selected policy


        proposal_dependencies = (
            [profile_id]
            if profile_id is not None
            else []
        )

        proposal_analysis = (
            profile_artifacts.get("analysis")
            or {}
        )

        proposal_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.INITIAL_PROPOSAL,
            payload={
                "duration_s": duration_s,
                "execution_policy": dict(policy),
                "analysis": dict(proposal_analysis),
            },
            depends_on=proposal_dependencies,
        )


        await self.engine.run_stage(wf.workflow_id, proposal_id)
        proposal = self.engine.get_result(wf.workflow_id, proposal_id)
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

        # Flat-retrieval fallback: create uniform chunks using
        # the policy-selected window length.
        if not candidates:
            candidates = self._build_coarse_chunks(
                duration_s=duration_s,
                chunk_len_s=float(
                    policy.get("window_len_s", 5.0)
                ),
            )
            actions = candidates

        if not actions:
            actions = candidates

        # Preserve the original proposal candidates for flat fallback.
        proposal_candidates = list(candidates)
            
        # ------------------------------------------------------------
        # Optional CLIP ranking before selecting top-k visual windows.
        # This is the point where coarse proposal windows become ranked
        # candidate/actions for VLM.
        # ------------------------------------------------------------


        retrieval_mode = str(
            policy.get("retrieval_mode", "flat")
        )

        try:
            print(
                f"[RETRIEVAL_ENABLED] "
                f"mode={retrieval_mode} "
                f"proposal_candidates={len(candidates)}"
            )

            if retrieval_mode == "hierarchical":
                candidates, actions = (
                    self._hierarchical_rank_candidates_and_actions(
                        question=question,
                        video_path=video_path,
                        policy=policy,
                        duration_s=duration_s,
                    )
                )

            elif retrieval_mode == "flat":
                if not proposal_candidates:
                    raise ValueError(
                        "Flat retrieval requires candidate windows"
                    )

                candidates, actions = (
                    self._clip_rank_candidates_and_actions(
                        question=question,
                        video_path=video_path,
                        candidates=proposal_candidates,
                        policy=policy,
                        duration_s=duration_s,
                    )
                )

            else:
                raise ValueError(
                    f"Unknown retrieval_mode={retrieval_mode!r}"
                )

            print(
                f"[RETRIEVAL_DONE] "
                f"mode={retrieval_mode} "
                f"candidates={len(candidates)} "
                f"actions={len(actions)}"
            )

        except Exception as e:
            import traceback

            print(
                f"[RETRIEVAL_ERROR] "
                f"mode={retrieval_mode} "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

            if retrieval_mode == "hierarchical" and proposal_candidates:
                print(
                    "[RETRIEVAL_FALLBACK] "
                    "Falling back to flat retrieval"
                )

                candidates, actions = (
                    self._clip_rank_candidates_and_actions(
                        question=question,
                        video_path=video_path,
                        candidates=proposal_candidates,
                        policy=policy,
                        duration_s=duration_s,
                    )
                )
            else:
                raise

        action_topk = int(policy.get("action_topk", 8))
        max_vlm_windows_policy = int(policy.get("max_vlm_windows", action_topk))
        probe_fps = float(policy.get("probe_fps", 1.0))

        answer_max_frames_per_window_policy = int(
            policy.get("answer_max_frames_per_window", 2)
        )

        answer_max_images_total_policy = int(
            policy.get("answer_max_images_total", answer_max_frames_per_window_policy * max_vlm_windows_policy)
        )

        # Effective number of visual windows allowed.

        effective_max_vlm_windows = min(
            action_topk,
            max_vlm_windows_policy,
            max(1, answer_max_images_total_policy),
        )

        

        # Effective total image budget.
        # This prevents "top4" with 2 frames/window from accidentally exceeding the image budget.
        effective_frames_per_window = max(
            1,
            min(
                answer_max_frames_per_window_policy,
                max(1, answer_max_images_total_policy // max(1, effective_max_vlm_windows)),
            ),
        )

        inspect_actions = actions[: min(len(actions), effective_max_vlm_windows)]

        print(
            f"\n[SELECTED]"
            f" candidates={len(candidates)}"
            f" actions={len(actions)}"
            f" selected={len(inspect_actions)}"
            f" probe_topk={policy.get('probe_topk')}"
            f" action_topk={policy.get('action_topk')}"
        )

        print(
            f"\n[VLM_BUDGET]"
            f" windows={len(inspect_actions)}"
            f" frames_per_window={effective_frames_per_window}"
            f" total_images_budget={answer_max_images_total_policy}"
            f" effective_max_vlm_windows={effective_max_vlm_windows}"
        )

        print("\n[VLM_SELECTED_ACTIONS]")
        for a in inspect_actions:
            print(
                f"window=({float(a['t0']):.2f},{float(a['t1']):.2f}) "
                f"score={float(a.get('score', 0.0)):.4f} "
                f"source={a.get('source')}"
            )

        if not inspect_actions:
            raise ValueError("VIMIO selected zero visual actions; cannot run visual one-pass.")


        return await self._run_default_one_pass(
            wf=wf,
            profile_id=profile_id,
            proposal_id=proposal_id,
            question=question,
            prof=prof,
            proposal=proposal,
            inspect_actions=inspect_actions,
            probe_fps=probe_fps,
            answer_max_frames_per_window=effective_frames_per_window,
            policy=policy,
            policy_mode=policy_mode,
        )

        
    async def _run_default_one_pass(
        self,
        *,
        wf,
        profile_id,
        proposal_id,
        question,
        prof,
        proposal,
        inspect_actions: list[dict],
        probe_fps: float,
        answer_max_frames_per_window: int = 4,
        policy: dict | None = None,
        policy_mode: str = "auto",
    ):
        
        visual_stage_ids: list[str] = []
        map_stage_ids: list[str] = []

        for a in inspect_actions:
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

        await self.engine.run_ready_batch(
            wf.workflow_id,
            map_stage_ids,
        )

        print("===================================")
        print("VISUAL BATCH FINISHED")
        print("===================================")

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

        print("CREATING ANSWER")
        answer_id = self.engine.add_stage(
            wf.workflow_id,
            StageType.ANSWER,
            depends_on=map_stage_ids,
        )
        print("ANSWER ID", answer_id)
        print("RUNNING ANSWER")
        answer_res = await self.engine.run_stage(wf.workflow_id, answer_id)
        print("ANSWER DONE")
        answer_text = answer_res.artifacts.get("answer")

        if answer_text is None:

            answer_text = f"ERROR: {answer_res.artifacts.get('error', 'answer stage failed')}"

        all_stage_ids = []

        if profile_id is not None:
            all_stage_ids.append(profile_id)

        all_stage_ids.append(proposal_id)
        all_stage_ids.extend(map_stage_ids)
        all_stage_ids.append(answer_id)
        stage_timing = self._collect_stage_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
        )
        stage_timing_csv = self._append_stage_timing_csv(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            video_id=wf.video_path,
            question=question,
            policy_mode=policy_mode,
        )

        workflow_summary_csv = self._summarize_workflow_timing(
            workflow_id=wf.workflow_id,
            stage_ids=all_stage_ids,
            visual_results=visual_results,
            policy_mode=policy_mode,
            video_id=wf.video_path,
            question=question,
        )

        return {
            "workflow_id": wf.workflow_id,
            "answer": answer_text,
            "confidence": answer_res.artifacts.get("confidence"),
            "profile": prof.artifacts if prof is not None else {},
            "proposal": proposal.artifacts,
            "debug": {
                "num_visual_stages": len(visual_stage_ids),
                "inspect_actions": inspect_actions,
                "effective_policy": policy or {},
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
                "workflow_summary_csv": workflow_summary_csv,
            },
        }
    
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
        video_id: str,
        question: str,
        out_path: str = "outputs/stage_timing/all_stage_timing.csv",
        policy_mode: str = "vimio",
    ) -> str:
        rows = self._collect_stage_timing(
            workflow_id=workflow_id,
            stage_ids=stage_ids,
        )

        for r in rows:
            r["workflow_id"] = workflow_id
            r["video_id"] = video_id
            r["question"] = question
            r["policy_mode"] = policy_mode

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
            "visual_wall_s": by_type.get(str(StageType.VISUAL_INSPECT), 0.0),
            "fuse_wall_s": 0.0,
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


    def _score_hierarchical_windows_with_clip(
        self,
        *,
        question: str,
        video_path: str,
        windows: list[RetrievalWindow],
        policy: dict,
    ) -> list[float]:
        if self.decoder is None:
            raise RuntimeError(
                "Hierarchical CLIP scoring requires a decoder"
            )

        candidate_objs = [
            CandidateWindow(
                t0=float(window.t0),
                t1=float(window.t1),
                score=0.0,
                source="hierarchical_candidate",
            )
            for window in windows
        ]

        ranked = self.clip_ranker.rank_windows(
            question=question,
            video_path=video_path,
            windows=candidate_objs,
            decoder=self.decoder,
            # Keep this at 1 for the fixed retrieval-frame budget.
            frames_per_window=1,
            fps=float(policy.get("probe_fps", 1.0)),
        )

        score_by_window = {
            (
                round(float(candidate.t0), 6),
                round(float(candidate.t1), 6),
            ): float(candidate.score)
            for candidate in ranked
        }

        missing_windows = [
            window
            for window in windows
            if (
                round(float(window.t0), 6),
                round(float(window.t1), 6),
            )
            not in score_by_window
        ]

        if missing_windows:
            missing_preview = [
                (float(window.t0), float(window.t1))
                for window in missing_windows[:5]
            ]

            raise RuntimeError(
                f"CLIP returned scores for "
                f"{len(score_by_window)} of {len(windows)} windows. "
                f"Missing examples: {missing_preview}"
            )

        return [
            score_by_window[
                (
                    round(float(window.t0), 6),
                    round(float(window.t1), 6),
                )
            ]
            for window in windows
        ]


    def _hierarchical_rank_candidates_and_actions(
        self,
        *,
        question: str,
        video_path: str,
        policy: dict,
        duration_s: float,
    ) -> tuple[list[dict], list[dict]]:
        if self.decoder is None:
            raise RuntimeError(
                "Hierarchical retrieval requires a decoder"
            )

        retrieval_budget = int(
            policy.get("retrieval_budget", 32)
        )

        retriever = HierarchicalRetriever(
            retrieval_budget=retrieval_budget
        )

        print(
            "\n[HIERARCHICAL_RETRIEVAL]"
            f" duration={duration_s:.2f}s"
            f" budget={retrieval_budget}"
            f" coarse={policy.get('retrieval_coarse_count', 16)}"
            f" parents={policy.get('retrieval_parents_to_refine', 4)}"
            f" final_topk={policy.get('action_topk', 4)}"
        )

        ranked_windows = retriever.retrieve(
            question=question,
            duration_s=duration_s,
            score_windows=lambda query, windows: (
                self._score_hierarchical_windows_with_clip(
                    question=query,
                    video_path=video_path,
                    windows=windows,
                    policy=policy,
                )
            ),
            coarse_count=int(
                policy.get("retrieval_coarse_count", 16)
            ),
            parents_to_refine=int(
                policy.get("retrieval_parents_to_refine", 4)
            ),
            final_topk=int(
                policy.get("action_topk", 4)
            ),
        )


        candidates = [
            {
                "t0": float(window.t0),
                "t1": float(window.t1),
                "score": float(window.score),
                "source": "hierarchical_clip",
                "level": int(window.level),
            }
            for window in ranked_windows
        ]

        actions = build_action_space(
            [
                CandidateWindow(
                    t0=c["t0"],
                    t1=c["t1"],
                    score=c["score"],
                    source=c["source"],
                )
                for c in candidates
            ],
            window_len_s=float(
                policy.get("window_len_s", 4.0)
            ),
            topk=int(
                policy.get("action_topk", 4)
            ),
        )

        action_dicts = [action.__dict__ for action in actions]

        for action in action_dicts:
            action["t0"] = max(0.0, float(action["t0"]))
            action["t1"] = min(
                float(duration_s),
                float(action["t1"]),
            )
            action["source"] = "hierarchical_clip"
            action["retrieval_level"] = 1

        return candidates, action_dicts

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

            probe_topk = int(policy.get("probe_topk", len(candidate_objs)))
            candidate_objs = candidate_objs[: min(len(candidate_objs), probe_topk)]
            actions = build_action_space(
                candidate_objs,
                window_len_s=float(policy.get("window_len_s", 5.0)),
                topk=int(policy.get("action_topk", 8)),
            )

            ranked_dicts = [c.__dict__ for c in candidate_objs]
            action_dicts = [a.__dict__ for a in actions]

            for a in action_dicts:
                a["t0"] = max(0.0, float(a["t0"]))
                a["t1"] = min(float(duration_s), float(a["t1"]))


            return ranked_dicts, action_dicts

        ranked = self.clip_ranker.rank_windows(
            question=question,
            video_path=video_path,
            windows=candidate_objs,
            decoder=self.decoder,
            frames_per_window=int(policy.get("clip_frames_per_window", 2)),
            fps=float(policy.get("probe_fps", 1.0)),
        )

        probe_topk = int(policy.get("probe_topk", len(ranked)))
        ranked = ranked[: min(len(ranked), probe_topk)]

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
    
    def _coerce_policy_number(self, policy: dict, key: str, default, typ):
        try:
            value = policy.get(key, default)
            if value is None:
                return default
            return typ(value)
        except Exception:
            print(f"[POLICY_WARN] Invalid {key}={policy.get(key)!r}; using default={default}")
            return default