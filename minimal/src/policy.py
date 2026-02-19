from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Dict, Any

Tax = Literal["S1", "S2", "S3", "S4", "S5"]


@dataclass
class TaxonomyPlan:
    tax: Tax
    intent: str

    # ---- coarse routing ----
    coarse_top_k: int = 3

    # ---- escalation ladders ----
    inspect_fps_ladder: List[float] = field(default_factory=lambda: [8.0, 16.0, 30.0, 60.0])
    refine_window_ladder_s: List[float] = field(default_factory=lambda: [5.0, 3.0, 1.5])
    refine_dense_fps_ladder: List[float] = field(default_factory=lambda: [8.0, 16.0, 30.0, 60.0])

    # ---- coverage requirements ----
    require_before_after: bool = False
    require_lookback: bool = False
    require_multiple_windows: bool = False
    min_distinct_windows: int = 2

    # ---- stagnation & repetition control ----
    max_repeat_same_window: int = 1
    min_score_improve: float = 0.01  # (not used yet in gate_call)

    # ---- budget / fallback ----
    allow_full_pass_fallback: bool = True
    max_inspect_calls: int = 12
    max_refine_calls: int = 6
    max_total_calls: int = 20


def classify_query_to_taxonomy(q: str) -> TaxonomyPlan:
    ql = (q or "").lower()

    if any(x in ql for x in ["before", "after", "change", "difference", "compared", "when did", "first", "then"]):
        return TaxonomyPlan(tax="S5", intent="temporal ordering/contrast",
                            require_before_after=True, require_multiple_windows=True, coarse_top_k=6)

    if any(x in ql for x in ["why", "how did", "how was", "what caused", "explain", "enable"]):
        return TaxonomyPlan(tax="S3", intent="delayed causal evidence", require_lookback=True, coarse_top_k=6)

    if any(x in ql for x in ["press", "tap", "hit", "strike", "flip", "turn", "flick", "exact moment"]):
        # For S2 you can *bias* by giving higher fps ladders if you want, but yours already includes 30/60.
        return TaxonomyPlan(tax="S2", intent="microevent/transient")

    if any(x in ql for x in ["how many", "every time", "repeated", "pattern", "throughout", "all the times"]):
        return TaxonomyPlan(tax="S4", intent="distributed evidence across time",
                            require_multiple_windows=True, coarse_top_k=6)

    return TaxonomyPlan(tax="S1", intent="fine-grained attribute/binding")


def _call_sig(call: Dict[str, Any]) -> Tuple:
    tool = call.get("tool")
    if tool == "inspect_window":
        return (tool, call.get("t0"), call.get("t1"), call.get("fps"), call.get("query"))
    if tool == "refine_in_segment":
        return (tool, call.get("seg_idx"), call.get("dense_fps"), call.get("window_s"), call.get("query"))
    if tool == "search_segments":
        return (tool, call.get("top_k"), call.get("query"))
    if tool == "inspect_window_heavy":
        return (tool, call.get("t0"), call.get("t1"), call.get("fps"), call.get("query"))
    return (tool, tuple(sorted(call.items())))


# ------------------------------------------------------------
# ESCALATION POLICY: cheap → heavy model
#
# Motivation:
#   If we repeatedly inspect the same window with the cheap
#   captioner (e.g., BLIP) and the judge is still unconvinced,
#   we escalate to a heavier VLM.
#
# Why?
#   - Cheap captions are often object-centric ("a bowl of onions")
#   - S2 microevents require stronger temporal/action grounding
#   - Judge may stall due to weak signal
#
# Research goal:
#   Measure how often heavy escalation is required.
#   Ideally this number stays small (<10–15% of cases).
#
# This is a *controlled fallback*, not default behavior.
# ------------------------------------------------------------


def gate_call(
    call: Optional[Dict[str, Any]],
    plan: TaxonomyPlan,
    allowlist: Dict[str, Any],
    gs: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    # plan guard
    if plan is None:
        return None, "no_plan"

    # null call guard
    if call is None:
        return None, "no_call"

    tool = call.get("tool")
    if not tool:
        return None, "missing_tool"

    # init gs fields defensively (helps when you forget to add keys)
    gs.setdefault("tried_calls", set())
    gs.setdefault("inspect_counts", {})
    gs.setdefault("n_calls", 0)
    gs.setdefault("n_inspect", 0)
    gs.setdefault("n_refine", 0)
    gs.setdefault("n_heavy", 0)

    # global budgets
    if gs["n_calls"] >= plan.max_total_calls:
        return None, "budget_total"

    if tool == "inspect_window" and gs["n_inspect"] >= plan.max_inspect_calls:
        return None, "budget_inspect"

    if tool == "refine_in_segment" and gs["n_refine"] >= plan.max_refine_calls:
        return None, "budget_refine"

    if tool == "inspect_window_heavy" and gs["n_heavy"] >= plan.max_heavy_calls:
        return None, "budget_heavy"

    # ---- clamp ladders BEFORE computing signature ----
    if tool == "inspect_window":
        fps = float(call["fps"])
        allowed = sorted(plan.inspect_fps_ladder)
        if allowed and fps not in allowed:
            fps2 = next((x for x in allowed if x >= fps), allowed[-1])
            call = dict(call)
            call["fps"] = float(fps2)

    if tool == "refine_in_segment":
        dfps = float(call["dense_fps"])
        ws = float(call["window_s"])
        dfps_allowed = sorted(plan.refine_dense_fps_ladder)
        ws_allowed = sorted(plan.refine_window_ladder_s, reverse=True)  # bigger->smaller

        if dfps_allowed and dfps not in dfps_allowed:
            dfps2 = next((x for x in dfps_allowed if x >= dfps), dfps_allowed[-1])
            call = dict(call)
            call["dense_fps"] = float(dfps2)

        if ws_allowed and ws not in ws_allowed:
            ws2 = next((x for x in ws_allowed if x <= ws), ws_allowed[-1])
            call = dict(call)
            call["window_s"] = float(ws2)

    # prevent exact repeats (AFTER clamping)
    sig = _call_sig(call)
    if sig in gs["tried_calls"]:
        return None, "repeat_exact"

    # prevent excessive repeats of same window+fps (ignore query string)
    if tool == "inspect_window":
        k = (float(call["t0"]), float(call["t1"]), float(call["fps"]))
        gs["inspect_counts"][k] = gs["inspect_counts"].get(k, 0) + 1

        if gs["inspect_counts"][k] > plan.max_repeat_same_window:
            # allow one heavy retry instead of returning None
            if gs["n_heavy"] < plan.max_heavy_calls:
                heavy = dict(call)
                heavy["tool"] = "inspect_window_heavy"

                # record attempted normal call so we don't loop back into it
                gs["tried_calls"].add(sig)
                gs["n_calls"] += 1
                gs["n_inspect"] += 1

                heavy_sig = _call_sig(heavy)
                if heavy_sig in gs["tried_calls"]:
                    return None, "repeat_exact_heavy"

                gs["tried_calls"].add(heavy_sig)
                gs["n_calls"] += 1
                gs["n_heavy"] += 1
                return heavy, "escalate_to_heavy"

            return None, "repeat_same_window"

    # accept
    gs["tried_calls"].add(sig)
    gs["n_calls"] += 1
    if tool == "inspect_window":
        gs["n_inspect"] += 1
    if tool == "refine_in_segment":
        gs["n_refine"] += 1
    if tool == "inspect_window_heavy":
        gs["n_heavy"] += 1

    return call, "ok"