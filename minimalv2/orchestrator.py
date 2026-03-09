# orchestrator.py
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Dict, Any
import time

from state import BeliefState
from actions import build_action_space
from scheduler import pick_next_action, propose_followup_windows
from stopping import stopping_rule
from budget import Budget
from tools import probe_index, inspect_window, ocr_window
from cheap_answerer import TextAnswerer, TextAnswererConfig
from model_registry import default_registry, pick_answer_model, pick_text_model
from answerer import VLLMAnswerer, VLLMAnswererConfig

# NEW:
from llm_profiler import profile_query_llm

REG = default_registry()


def run(
    query: str,
    video: str,
    *,
    # budget
    max_dense_seconds: float = 20.0,
    max_frames: int = 2000,
    max_wallclock_s: float = 60.0,

    # profiler
    profiler_model: str = "YOUR_TEXT_PROFILER_MODEL",
    profiler_base_url: str = "http://localhost:8001/v1",

    # OCR stage
    ocr_model: Optional[str] = None,
    ocr_base_url: Optional[str] = None,

    # final answer model
    answer_base_url: str = "http://localhost:8000/v1",
    answer_max_tokens: int = 64,

    # overrides
    force_answer_model: Optional[str] = None,
    disable_answer: bool = False,
):
    # 0) LLM profiler decides policy + routing tiers + knobs
    policy = profile_query_llm(
        query,
        base_url=profiler_base_url,
        model=profiler_model,
        temperature=0.0,
        timeout_s=30.0,
    )

    state = BeliefState()
    direction = detect_direction(query)

    budget = Budget(
        max_dense_seconds=max_dense_seconds,
        max_frames=max_frames,
        max_wallclock_s=max_wallclock_s,
    )

    # Pull knobs from policy (policy owns config now)
    probe_fps = policy.probe_fps
    probe_seg_len_s = policy.probe_seg_len_s
    probe_topk = policy.probe_topk
    window_len_s = policy.window_len_s
    action_topk = policy.action_topk

    # 1) CLIP probe (cheap; scans full video at probe_fps)
    probe_start = time.time()
    clip_candidates = probe_index(
        video=video,
        query=query,
        fps=probe_fps,
        segment_len_s=probe_seg_len_s,
        topk=probe_topk,
    )
    probe_wall = time.time() - probe_start

    # If you implemented update_from_probe, keep it
    if hasattr(state, "update_from_probe"):
        state.update_from_probe(
            wallclock_s=probe_wall,
            segments_scanned=len(clip_candidates),
            fps=probe_fps,
        )
    else:
        # safe fallback so you don't crash if state doesn't have it yet
        state.probe_wallclock_s = getattr(state, "probe_wallclock_s", 0.0) + float(probe_wall)

    # 2) Build action space
    # Keep your microevent tighter windows logic, but driven by policy.mode/require_temporal_pair
    if policy.require_temporal_pair or policy.mode == "microevent":
        actions = build_action_space(
            clip_candidates,
            window_len=2.0,                 # tight
            topk=action_topk,
            strides=list(policy.strides),   # usually (0.25, 0.5)
            resolutions=["high"],
        )
    else:
        actions = build_action_space(
            clip_candidates,
            window_len=window_len_s,        # policy default
            topk=action_topk,
            strides=list(policy.strides),   # usually (0.5,)
            resolutions=["high"],
        )

    trace: list[Dict[str, Any]] = []

    # 3) Deterministic schedule loop
    while not stopping_rule(state, policy, budget):
        action = pick_next_action(policy, state, actions, direction=direction)
        if action is None:
            break

        res = inspect_window(
            video=video,
            t0=action.t0,
            t1=action.t1,
            stride=action.stride,
            resolution=action.resolution,
            query=query,
            source="clip",
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
                    "source": getattr(res, "source", "clip"),
                },
            }
        )

        state.update_from_window(res)

        # 3b) microevent followups (your existing behavior)
        if policy.mode == "microevent" and state.steps == 1:
            direction = detect_direction(query)
            width = (action.t1 - action.t0)

            followups = propose_followup_windows(
                (action.t0, action.t1),
                direction=direction,
                width_s=width,
                gaps_s=[k * width for k in range(12)],
            )

            best_before = state.best_relevance_score

            for (ft0, ft1) in followups:
                if ft1 <= ft0 or (ft0, ft1) in state.windows:
                    continue

                follow_res = inspect_window(
                    video=video,
                    t0=ft0,
                    t1=ft1,
                    stride=action.stride,
                    resolution=action.resolution,
                    query=query,
                    source="clip",
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
                            "source": getattr(follow_res, "source", "clip"),
                        },
                    }
                )
                state.update_from_window(follow_res)

                if (state.best_relevance_score - best_before) >= policy.eps_marginal_gain:
                    break

    # 4) Final answer step (model routing)
    pred = None
    chosen_answer_model = None
    fallback_answer_model = None
    fallback_used = False
    answer_conf = None  # float | None
    answer_raw = None
    fallback_raw = None

    if not disable_answer:
        if force_answer_model is not None:
            chosen_answer_model = force_answer_model
        else:
            chosen_answer_model = pick_answer_model(REG, policy.answer_tier)

        vlm_windows = select_vlm_windows(policy, trace, direction=direction, k_followups=1)[:2]

        if not vlm_windows:
            return {
                "pred": None,
                "policy": asdict(policy),
                "trace": trace,
                "routing": {
                    "profiler_model": profiler_model,
                    "answer_model": None,
                    "fallback_answer_model": None,
                    "fallback_used": False,
                    "answer_conf": None,
                },
                "reasoning_metrics": {
                    "distinct_windows": state.distinct_windows,
                    "best_relevance_score": state.best_relevance_score,
                    "score_improvement": state.score_improvement,
                    "dense_seconds_encoded": state.dense_seconds_encoded,
                    "approx_frames_encoded": state.approx_frames_encoded,
                    "inspect_wallclock_s": state.inspect_wallclock_s,
                    "steps": state.steps,
                    "windows": list(state.windows),
                    "probe_wallclock_s": getattr(state, "probe_wallclock_s", None),
                    "probe_segments_scanned": getattr(state, "probe_segments_scanned", None),
                },
            }

        if getattr(policy, "enable_cheap_stage", False):
            evidence_parts = []

            for (t0, t1) in vlm_windows:
                if "ocr" in policy.preferred_tools:
                    o = ocr_window(
                        video,
                        t0,
                        t1,
                        stride=0.5,
                        resolution="high",
                        max_frames=8,
                        model=ocr_model,
                        base_url = ocr_base_url
                    txts = (o.evidence or {}).get("ocr_text", [])
                    if txts:
                        evidence_parts.append(
                            f"[OCR {t0:.1f}-{t1:.1f}] " + " ".join(txts)
                        )

                # later:
                # if "caption" in policy.preferred_tools:
                #     ...
                # if "asr" in policy.preferred_tools:
                #     ...

            evidence = "\n".join(evidence_parts).strip()

            if evidence:
                text_model = pick_text_model(REG, policy.cheap_answer_tier)

                if text_model is not None:
                    ta = TextAnswerer(
                        TextAnswererConfig(
                            model=text_model,
                            base_url=profiler_base_url,
                        )
                    )

                    cheap_pred, cheap_conf, cheap_raw = ta.answer_with_confidence(
                        question=query,
                        evidence=evidence,
                    )

                    if cheap_conf >= policy.text_answer_min_conf:
                        return {
                            "pred": cheap_pred,
                            "policy": asdict(policy),
                            "trace": trace,
                            "routing": {
                                "profiler_model": profiler_model,
                                "answer_model": None,
                                "text_answer_model": text_model,
                                "text_answer_conf": cheap_conf,
                                "fallback_answer_model": None,
                                "fallback_used": False,
                                "answer_conf": cheap_conf,
                                "stage": "cheap_text",
                                "answer_raw": cheap_raw,

                            },
                            "reasoning_metrics": {
                                "distinct_windows": state.distinct_windows,
                                "best_relevance_score": state.best_relevance_score,
                                "score_improvement": state.score_improvement,
                                "dense_seconds_encoded": state.dense_seconds_encoded,
                                "approx_frames_encoded": state.approx_frames_encoded,
                                "inspect_wallclock_s": state.inspect_wallclock_s,
                                "steps": state.steps,
                                "windows": list(state.windows),
                                "probe_wallclock_s": getattr(state, "probe_wallclock_s", None),
                                "probe_segments_scanned": getattr(state, "probe_segments_scanned", None),
                            },
                        }

        
        if chosen_answer_model is not None:
            ans = VLLMAnswerer(
                VLLMAnswererConfig(
                    model=chosen_answer_model,
                    base_url=answer_base_url,
                    max_tokens=answer_max_tokens,
                    temperature=0.0,
                )
            )

            #Primary (must produce answer confidence)
            pred, answer_conf, answer_raw = ans.answer_with_confidence(
                video_path=video,
                windows=vlm_windows,
                question=query,
                sample_fps=1.0,
                max_frames_per_window=2,
                mode=policy.mode,
                max_windows=2,
                max_images_total=2,
                jpeg_quality=85,
            )
            

            # ---- fallback attempt if low confidence ----
            if (policy.fallback_answer_tier != "none"
                and answer_conf is not None
                and float(answer_conf) < float(policy.min_answer_conf)
            ):
                fallback_answer_model = pick_answer_model(REG, policy.fallback_answer_tier)

                if fallback_answer_model is not None and fallback_answer_model != chosen_answer_model:
                    ans2 = VLLMAnswerer(
                        VLLMAnswererConfig(
                            model=fallback_answer_model,
                            base_url=answer_base_url,
                            max_tokens=answer_max_tokens,
                            temperature=0.0,
                        )
                    )
                    pred2, conf2, fallback_raw = ans2.answer_with_confidence(
                        video_path=video,
                        windows=vlm_windows,
                        question=query,
                        sample_fps=1.0,
                        max_frames_per_window=2,
                        mode=policy.mode,
                        max_windows=2,
                        max_images_total=2,
                        jpeg_quality=85,
                    )

                    # keep the more confident one
                    if float(conf2) >= float(answer_conf):
                        pred = pred2
                        answer_conf = conf2
                        answer_raw = fallback_raw
                        fallback_used = True
    return {
        "pred": pred,
        "policy": asdict(policy),
        "trace": trace,
        "routing": {
            "profiler_model": profiler_model,
            "answer_model": chosen_answer_model,
            "fallback_answer_model": fallback_answer_model,
            "fallback_used": fallback_used,
            "answer_conf": answer_conf,
            "answer_raw" : answer_raw
        },
        "reasoning_metrics": {
            "distinct_windows": state.distinct_windows,
            "best_relevance_score": state.best_relevance_score,
            "score_improvement": state.score_improvement,
            "dense_seconds_encoded": state.dense_seconds_encoded,
            "approx_frames_encoded": state.approx_frames_encoded,
            "inspect_wallclock_s": state.inspect_wallclock_s,
            "steps": state.steps,
            "windows": list(state.windows),
            "probe_wallclock_s": getattr(state, "probe_wallclock_s", None),
            "probe_segments_scanned": getattr(state, "probe_segments_scanned", None),
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
    return "after"


def select_vlm_windows(policy, trace, *, direction: str, k_followups: int = 1):
    if not trace:
        return []

    best = max(trace, key=lambda e: float(e["result"].get("relevance_score", 0.0)))
    anchor = best["action"]
    a0, a1 = float(anchor["t0"]), float(anchor["t1"])

    scored = []
    for e in trace:
        act = e["action"]
        res = e["result"]
        t0, t1 = float(act["t0"]), float(act["t1"])
        s = float(res.get("relevance_score", 0.0))
        scored.append(((t0, t1), s))

    if policy.mode != "microevent":
        scored.sort(key=lambda x: -x[1])
        return [w for (w, _) in scored[: min(3, len(scored))]]

    if direction == "after":
        follow = [x for x in scored if x[0][0] >= a1]
        follow.sort(key=lambda x: (-x[1], x[0][0] - a1))
    else:
        follow = [x for x in scored if x[0][1] <= a0]
        follow.sort(key=lambda x: (-x[1], a0 - x[0][1]))

    out = [(a0, a1)]
    out.extend([w for (w, _) in follow[:k_followups]])
    out = list(dict.fromkeys(out))
    out.sort(key=lambda w: w[0])
    return out



