# orchestrator.py
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from query_profile import classify_query
from state import BeliefState
from actions import build_action_space
from scheduler import pick_next_action, propose_followup_windows
from stopping import stopping_rule
from budget import Budget
from tools import probe_index, inspect_window

from answerer import VLLMAnswerer, VLLMAnswererConfig


def run(
    query: str,
    video: str,
    *,
    # probe config
    probe_fps: float = 1.0,
    probe_seg_len_s: float = 5.0,
    probe_topk: int = 50,
    # dense windows config
    window_len_s: float = 4.0,
    action_topk: int = 50,
    # budget
    max_dense_seconds: float = 20.0,
    max_frames: int = 2000,
    max_wallclock_s: float = 60.0,
    # final answer config (vLLM OpenAI-compatible)
    answer_model: Optional[str] = None,
    answer_base_url: str = "http://localhost:8000/v1",
    answer_max_tokens: int = 64,
):
    profile = classify_query(query)
    state = BeliefState()
    direction = detect_direction(query)
    budget = Budget(
        max_dense_seconds=max_dense_seconds,
        max_frames=max_frames,
        max_wallclock_s=max_wallclock_s,
    )

    # 1) CLIP probe (cheap)
    candidates = probe_index(
        video=video,
        query=query,
        fps=probe_fps,
        segment_len_s=probe_seg_len_s,
        topk=probe_topk,
    )

    # 2) Build action space
    actions = build_action_space(
        candidates,
        window_len=window_len_s,
        topk=action_topk,
    )

    trace = []

    # 3) Deterministic schedule loop
    while not stopping_rule(state, profile, budget):
        action = pick_next_action(profile, state, actions)
        if action is None:
            break

        res = inspect_window(
            video=video,
            t0=action.t0,
            t1=action.t1,
            stride=action.stride,
            resolution=action.resolution,
            query=query,
        )

        trace.append(
            {
                "action": {
                    "t0": action.t0,
                    "t1": action.t1,
                    "stride": action.stride,
                    "resolution": action.resolution,
                },
                "result": {
                    "relevance_score": res.relevance_score,
                    "frames_encoded": res.frames_encoded,
                    "dense_seconds": res.dense_seconds,
                    "wallclock_s": res.wallclock_s,
                },
            }
        )

        state.update_from_window(res)
        
        # -- microevent-- follows-up: inspect a temporally adjacent window ---
        #only run this for microevent queries.

        if profile.mode == "microevent" and state.steps == 1:
            direction = detect_direction(query)
            width = (action.t1 - action.t0)

            followups = propose_followup_windows(
                (action.t0, action.t1),
                direction=direction,
                width_s=width,
                gaps_s=[0.0, width, 3*width, 8*width],)   # <- tune this)

            best_before = state.best_relevance_score

            for (ft0, ft1) in followups:
                # avoid duplicates / invalid windows
                if ft1 <= ft0 or (ft0, ft1) in state.windows:
                    continue

                follow_res = inspect_window(
                    video=video,
                    t0=ft0,
                    t1=ft1,
                    stride=action.stride,
                    resolution=action.resolution,
                    query=query,
                )


                trace.append(
                    {
                        "action": {
                            "t0": ft0,
                            "t1": ft1,
                            "stride": action.stride,
                            "resolution": action.resolution,
                            "followup_of": [action.t0, action.t1],
                            "direction": direction,
                            "gap_s": (ft0 - action.t1) if direction == "after" else (action.t0 - ft1),
                        },
                        "result": {
                            "relevance_score": follow_res.relevance_score,
                            "frames_encoded": follow_res.frames_encoded,
                            "dense_seconds": follow_res.dense_seconds,
                            "wallclock_s": follow_res.wallclock_s,
                        },
                    }
                )
                state.update_from_window(follow_res)

                if (state.best_relevance_score - best_before) >= profile.eps_marginal_gain:
                    break


    # 4) Final answer step (optional but you asked to add it)
    pred = None
    if answer_model is not None:
        ans = VLLMAnswerer(
            VLLMAnswererConfig(
                model=answer_model,
                base_url=answer_base_url,
                max_tokens=answer_max_tokens,
                temperature=0.0,
            )
        )
        pred = ans.answer(
            video_path=video,
            windows=list(state.windows),
            question=query,
            sample_fps=1.0,
            max_frames_per_window=4,
        )

    return {
        "pred": pred,
        "profile": asdict(profile),
        "trace": trace,
        "reasoning_metrics": {
            "distinct_windows": state.distinct_windows,
            "best_relevance_score": state.best_relevance_score,
            "score_improvement": state.score_improvement,
            "dense_seconds_encoded": state.dense_seconds_encoded,
            "approx_frames_encoded": state.approx_frames_encoded,
            "inspect_wallclock_s": state.inspect_wallclock_s,
            "steps": state.steps,
            "windows": list(state.windows),
        },
    }


def detect_direction(q: str) -> str:
    q = q.lower()
    after = any(w in q for w in ["after", "afterwards", "then", "next", "following", "later"])
    before = any(w in q for w in ["before", "previously", "earlier"])
    if after and not before:
        return "after"
    if before and not after:
        return "before"
    return "after"  # default: most questions mean forward in time