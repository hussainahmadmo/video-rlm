from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Dict, Any
from controller_utils import _enforce_allowlist
Tax = Literal["S1", "S2", "S3", "S4", "S5"]
S5Mode = Literal["ordering", "contrast"]

@dataclass
class TaxonomyPlan:
    tax: Tax
    intent: str
    s5_mode: S5Mode | None = None  # NEW

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
    max_heavy_calls: int = 2


import re

def classify_query_to_taxonomy(q: str) -> TaxonomyPlan:
    ql = (q or "").lower().strip()

    # --- S5B: contrast / change (true before-vs-after reasoning) ---
    # These require paired states and explicit comparison.
    contrast_patterns = [
        r"\bwhat changed\b",
        r"\bwhat is different\b",
        r"\bdifference\b",
        r"\bcompare(d)?\b",
        r"\bversus\b|\bvs\b",
        r"\bbefore and after\b",
        r"\bhow did .* change\b",
        r"\bchange(d)? (in|between)\b",
    ]
    if any(re.search(p, ql) for p in contrast_patterns):
        return TaxonomyPlan(
            tax="S5",
            intent="temporal contrast/change",
            s5_mode= "contrast",
            require_before_after=True,
            require_multiple_windows=True,
            coarse_top_k=6,
        )

    # --- S5A: ordering / earliest / temporal localization ---
    # These need the earliest relevant occurrence, not necessarily contrast.
    ordering_patterns = [
        r"\bfirst\b",
        r"\bearliest\b",
        r"\bwhen did\b",
        r"\bwhat happened (first|before)\b",
        r"\bafter\b",
        r"\bbefore\b",
        r"\bthen\b",
        r"\bnext\b",
        r"\bin what order\b",
        r"\bsequence\b",
    ]
    if any(re.search(p, ql) for p in ordering_patterns):
        # NOTE: we do NOT force require_before_after=True here,
        # because "first" usually wants min-time among relevant items.
        return TaxonomyPlan(
            tax="S5",
            intent="temporal ordering/earliest",
            s5_mode="ordering",
            require_before_after=False,           # <-- key change
            require_multiple_windows=True,        # you can keep True to encourage coverage
            coarse_top_k=6,
        )

    # --- S3: delayed causal evidence ---
    if any(x in ql for x in ["why", "how did", "how was", "what caused", "explain", "enable"]):
        return TaxonomyPlan(tax="S3", intent="delayed causal evidence", require_lookback=True, coarse_top_k=6)

    # --- S2: microevent / transient ---
    if any(x in ql for x in ["press", "tap", "hit", "strike", "flip", "turn", "flick", "exact moment"]):
        return TaxonomyPlan(tax="S2", intent="microevent/transient")

    # --- S4: distributed evidence ---
    if any(x in ql for x in ["how many", "every time", "repeated", "pattern", "throughout", "all the times"]):
        return TaxonomyPlan(tax="S4", intent="distributed evidence across time",
                            require_multiple_windows=True, coarse_top_k=6)

    # --- S1: attribute/binding ---
    return TaxonomyPlan(tax="S1", intent="fine-grained attribute/binding")

def _call_signature(call: dict):
    tool = call.get("tool")
    if tool == "search_segments":
        return (tool, call.get("query"), int(call.get("top_k", 0)))
    if tool == "refine_in_segment":
        return (tool, call.get("query"), int(call.get("seg_idx")), float(call.get("dense_fps")), float(call.get("window_s")))
    if tool in ("inspect_window", "inspect_window_heavy"):
        return (tool, call.get("query"), float(call.get("t0")), float(call.get("t1")), float(call.get("fps")), int(call.get("top_m")))
    if tool == "summarize_answer":
        return (tool,)
    # fallback
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


from typing import Dict, Any, Tuple
from controller_utils import _enforce_allowlist

def _round2(x: float) -> float:
    return round(float(x), 2)

def gate_call(call: Dict[str, Any],
              plan,
              allowlist: Dict[str, Any],
              gs: Dict[str, Any],
              heavy_enabled: bool):
    if plan is None:
        return None, "no_plan"
    if call is None:
        return None, "no_call"

    tool = call.get("tool")
    if not tool:
        return None, "missing_tool"

    # ---- block heavy tool if heavy isn't enabled ----
    if tool == "inspect_window_heavy" and not heavy_enabled:
        return None, "heavy_disabled"

    # ---- init gs ----
    gs.setdefault("tried_calls", set())
    gs.setdefault("inspect_counts", {})   # (t0,t1) -> count of CHEAP inspections
    gs.setdefault("n_calls", 0)
    gs.setdefault("n_inspect", 0)
    gs.setdefault("n_refine", 0)
    gs.setdefault("n_heavy", 0)

    # ---- budget checks ----
    if gs["n_calls"] >= plan.max_total_calls:
        return None, "budget_total"
    if tool == "inspect_window" and gs["n_inspect"] >= plan.max_inspect_calls:
        return None, "budget_inspect"
    if tool == "refine_in_segment" and gs["n_refine"] >= plan.max_refine_calls:
        return None, "budget_refine"
    if tool == "inspect_window_heavy" and gs["n_heavy"] >= plan.max_heavy_calls:
        return None, "budget_heavy"

    # ---- clamp ladders (normalize call) ----
    if tool in ("inspect_window", "inspect_window_heavy"):
        fps = float(call["fps"])
        allowed = sorted(plan.inspect_fps_ladder or [])
        if allowed and fps not in allowed:
            fps2 = next((x for x in allowed if x >= fps), allowed[-1])
            call = dict(call)
            call["fps"] = float(fps2)

    if tool == "refine_in_segment":
        dfps = float(call["dense_fps"])
        ws = float(call["window_s"])
        dfps_allowed = sorted(plan.refine_dense_fps_ladder or [])
        ws_allowed = sorted(plan.refine_window_ladder_s or [], reverse=True)

        if dfps_allowed and dfps not in dfps_allowed:
            dfps2 = next((x for x in dfps_allowed if x >= dfps), dfps_allowed[-1])
            call = dict(call)
            call["dense_fps"] = float(dfps2)

        if ws_allowed and ws not in ws_allowed:
            ws2 = next((x for x in ws_allowed if x <= ws), ws_allowed[-1])
            call = dict(call)
            call["window_s"] = float(ws2)

    # ---- enforce allowlist AFTER clamping ----
    tool = call.get("tool")  # refresh in case call was copied/modified
    if tool in ("inspect_window", "inspect_window_heavy"):
        try:
            _enforce_allowlist(call, allowlist)
        except ValueError as e:
            return None, f"allowlist_violation:{e}"

    # ---- repeat-window / escalation logic ----
    # IMPORTANT: only track repeats for CHEAP inspect_window, not heavy.
    if tool == "inspect_window":
        k = (_round2(call["t0"]), _round2(call["t1"]))
        gs["inspect_counts"][k] = gs["inspect_counts"].get(k, 0) + 1

        if gs["inspect_counts"][k] > plan.max_repeat_same_window:
            if not heavy_enabled:
                return None, "heavy_disabled" #or repeat same window
            if gs["n_heavy"] >= plan.max_heavy_calls:
                    return None, "budget_heavy"   # or repeat_same_window
            heavy = dict(call)
            heavy["tool"] = "inspect_window_heavy"
            call = heavy
            tool = "inspect_window_heavy"

    # ---- final repeat_exact check on FINAL normalized call ----
    sig = _call_signature(call)
    if sig in gs["tried_calls"]:
        return None, "repeat_exact"

    gs["tried_calls"].add(sig)
    return call, "ok"