# controller.py
from __future__ import annotations

import json
from typing import Any, Dict
from clip_encoder import Clip
from env import VideoEnv
from typing import Any, Dict, List
from openai import OpenAI
from tracing import trace_judge, trace_gate
from policy import TaxonomyPlan, classify_query_to_taxonomy, gate_call  # (or _gate_call if you keep underscore)
#(Hussain) Change captioner such that we can import multiple captioners
from captioner import BlipCaptioner
from controller_utils import AllowlistState, finalize_allowlist, available_tools_for
from controller_utils import finalize_allowlist, available_tools_for, _extract_numbers_from_notes, _enforce_allowlist
from reasoning.order_ops import (ordering_mode, evidence_for_summary, needs_heavy_detail)
from tracing import trace_controller_action
from reasoning.evidence_adapters import add_tool_result_to_evidence
from reasoning.evidence_state import EvidenceState
import copy



SUMMARY_SYSTEM_PROMPT = (
    """
        You are an evidence-grounded video QA system.

        You MUST return a valid tool call using the tool:
        summarize_answer

        The output MUST be a JSON object with the following fields:

        {
        "tool": "summarize_answer",
        "answer": string,
        "evidence": list of 1–3 short quotes/snippets (include timestamps if available),
        "abstain": boolean,
        "reason": string
        }

        Rules:
        - You must use ONLY the provided memory evidence.
        - If evidence clearly supports an answer, provide it.
        - If evidence does not support a specific answer, set:
            "abstain": true
        - If the best you can do is restate the question, you MUST abstain.
        - If evidence contradicts itself and cannot be resolved, you MUST abstain.
        - Never omit the "tool" field.
        - Never output text outside the JSON object.
    """
)

SYSTEM_PROMPT = (
  "You are a video controller. Output ONLY a single JSON object.\n"
  "The JSON MUST have key 'tool'. Do NOT use 'tool_to_call' or any other key.\n"
  "Schema:\n"
  " - search_segments: {tool:'search_segments', query:str, top_k:int}\n"
  " - refine_in_segment: {tool:'refine_in_segment', query:str, seg_idx:int, dense_fps:float, window_s:float}\n"
  " - inspect_window: {tool:'inspect_window', query:str, t0:float, t1:float, fps:float, top_m:int}\n"
  " - summarize_answer: {tool:'summarize_answer', answer:str}\n"
  " - inspect_window_heavy: {tool:'inspect_window_heavy', query:str, t0:float, t1:float, fps:float, top_m:int}\n"
  "Return EXACTLY one of these objects and nothing else."
)

TOOL_SCHEMA = {
  "search_segments": ["tool", "query", "top_k"],
  "refine_in_segment": ["tool", "query", "seg_idx", "dense_fps", "window_s"],
  "inspect_window": ["tool", "query", "t0", "t1", "fps", "top_m"],
  "summarize_answer": ["tool", "answer"],
  "inspect_window_heavy": ["tool", "query", "t0", "t1", "fps", "top_m"],   # NEWs
}

MAX_DEPTH = 6
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class Node:
    depth: int
    memory: List[Dict[str, Any]]
    tried: set = field(default_factory=set)
    plan: Optional["TaxonomyPlan"] = None  # forward reference avoids ordering issues
    evidence: Optional["EvidenceState"] = None

    branch_score: float = float("-inf")
    chosen_seg: Optional[int] = None
    chosen_window: Optional[Tuple[float, float]] = None

    hypothesis: str = ""
    critique: str = ""
    last_rewritten_query: Optional[str] = None

def _has_caption(step: Dict[str, Any]) -> bool:
    for tf in (step.get("top_frames") or []):
        if tf.get("caption") and str(tf["caption"]).strip():
            return True
    for ev in (step.get("evidence") or []):
        if isinstance(ev, str) and ev.strip():
            return True
        if isinstance(ev, dict) and (ev.get("note") or "").strip():
            return True
    return False

def _evidence_slices(memory, k=6):
    out = []
    for m in memory:
        tool = m.get("tool") or m.get("name")
        if tool not in ("inspect_window", "inspect_window_heavy"):
            continue

        payload = m.get("result") or m.get("obs") or m

        # harvest evidence
        for tf in payload.get("top_frames", []) or []:
            ev = tf.get("vlm_answer") or tf.get("caption")
            if ev:
                out.append({"timestamp": tf.get("t"), "evidence": ev})

        if not payload.get("top_frames") and payload.get("summary"):
            out.append({"evidence": payload["summary"]})

    return out[-k:]

def _best_score_in_memory(memory: List[Dict[str, Any]]) -> float:
    best = float("-inf")
    for step in memory:
        for ev in step.get("evidence", []) or []:
            try:
                best = max(best, float(ev.get("score", float("-inf"))))
            except:
                pass
    return best

def _top_segments_from_memory(memory: List[Dict[str, Any]]) -> List[int]:
    segs = []
    for step in memory:
        if step.get("tool") == "search_segments":
            for ev in step.get("evidence", []) or []:
                note = ev.get("note", "")
                if "seg=" in note:
                    try:
                        seg = int(note.split("seg=")[1].split(",")[0])
                        segs.append(seg)
                    except:
                        pass
    # preserve order, unique
    out = []
    for s in segs:
        if s not in out:
            out.append(s)
    return out

def _validate_tool_call(
    call: Dict[str, Any],
    allowed_tools: List[str],
    tool_schema: Dict[str, List[str]],) -> None:
    """
    Validate a single tool call object, e.g.
      {"tool":"inspect_window","query":"...","t0":0.0,"t1":1.0,"fps":4.0,"top_m":5}
    """
    if not isinstance(call, dict):
        raise ValueError(f"Tool call must be a dict. Got: {type(call)} {call}")

    tool = call.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"Tool call missing valid 'tool' string. Got: {call}")

    if allowed_tools and tool not in set(allowed_tools):
        raise ValueError(f"Tool '{tool}' not in allowed_tools={allowed_tools}. Call={call}")

    if tool not in tool_schema:
        raise ValueError(f"Unknown tool '{tool}'. Known tools: {list(tool_schema.keys())}")

    required = tool_schema[tool]
    missing = [k for k in required if k not in call]
    if missing:
        raise ValueError(f"Tool '{tool}' missing required keys {missing}. Call={call}")

    # Optional: reject unexpected keys (keeps model honest)
    allowed_keys = set(required)
    extra = [k for k in call.keys() if k not in allowed_keys]
    if extra:
        raise ValueError(f"Tool '{tool}' has unexpected keys {extra}. Allowed keys={sorted(allowed_keys)}. Call={call}")

    # Light type checks (optional but helps catch nonsense)
    if tool == "search_segments":
        if not isinstance(call["query"], str):
            raise ValueError(f"search_segments.query must be str. Call={call}")
        if not isinstance(call["top_k"], (int, float)):
            raise ValueError(f"search_segments.top_k must be number. Call={call}")
        
    if tool == "refine_in_segment":
        if not isinstance(call["window_s"], (int, float)):
            raise ValueError(f"refine_in_segment.window_s must be number. Got: {call}")
        if not isinstance(call["dense_fps"], (int, float)):
            raise ValueError(f"refine_in_segment.dense_fps must be number. Got: {call}")
        
    if tool in ("inspect_window","inspect_window_heavy"):
        for k in ("t0", "t1", "fps", "top_m"):
            if not isinstance(call[k], (int, float)):
                raise ValueError(f"inspect_window.{k} must be number. Call={call}")
        if float(call["t1"]) <= float(call["t0"]):
            raise ValueError(f"inspect_window requires t1 > t0. Call={call}")

    if tool == "summarize_answer":
        if not isinstance(call["answer"], str):
            raise ValueError(f"summarize_answer.answer must be str. Call={call}")

from reasoning.order_ops import evidence_for_summary
def _summarize_with_llm(llm: VLLMJsonToolLLM, query: str, memory: List[Dict[str, Any]]) -> str:
    evidence_mem = evidence_for_summary(memory, k = 6, query=query)
    #Optional deterministic heavy refinement for ordering queries
    mode = ordering_mode(query)
    prompt = {
        "system": SUMMARY_SYSTEM_PROMPT,
        "goal": query,
        "available_tools": ["summarize_answer"],  # force summarize
        "tool_schema": {"summarize_answer": ["tool", "answer","evidence","abstain", "reason"]},   # ✅ restrict
        "memory": evidence_mem,
    }

    print("EVIDENCE_MEM_LEN", len(evidence_mem))
    print("EVIDENCE_MEM_SAMPLE", evidence_mem[:1])
    call = llm.decide(prompt)

    if call.get("abstain"):
        return f"I’m not sure from the available evidence. ({call.get('reason','insufficient evidence')})"

    return call.get("answer", "")

def _step(env, node: Node, call: Dict[str, Any], gs: Dict[str, Any]) -> Optional[Node]:
    allow = finalize_allowlist(
        node.memory,
        seg_len_s=float(getattr(env, "seg_len_s", 60.0)),
        duration_s=float(getattr(env, "duration_s", 0.0) or 0.0),
    )
    heavy_enabled = getattr(env, "heavy_vlm", None) is not None

    call2, why = gate_call(call, node.plan, allow.to_dict(), gs, heavy_enabled)
    trace_gate(env.trace, node.depth, proposed=call, final=call2, why=why, allowlist=allow.to_dict())

    if call2 is None:
        return None

    cs = exec_call(env, call2, gs)
    cs_raw = cs if isinstance(cs, dict) else env.context_to_dict(cs)
    payload = cs_raw.get("result", cs_raw)

    evt = compact_slice({
        "tool": call2["tool"],
        "call": call2,
        **payload,
    })

    child = Node(
        depth=node.depth + 1,
        memory=node.memory + [evt],
        plan=node.plan,
        tried=set(node.tried),
        evidence=copy.deepcopy(node.evidence) if node.evidence else EvidenceState(),
    )
    add_tool_result_to_evidence(child.evidence, call2, payload)

    # if Node.evidence exists (recommended), ensure it’s there
    if child.evidence is None:
        from reasoning.evidence_state import EvidenceState
        child.evidence = EvidenceState()

    add_tool_result_to_evidence(child.evidence, call2, payload)

    gs["best_memory"] = child.memory
    # optional debug snapshot:
    # gs["best_evidence"] = child.evidence

    return child

def _judge_node(env, llm, query, node):
    allow = finalize_allowlist(node.memory, seg_len_s=env.seg_len_s, duration_s=env.duration_s)

    allowed_tools = ["search_segments", "refine_in_segment"]
    if allow.allowed_windows:
        allowed_tools.append("inspect_window")
        if getattr(env, "heavy_vlm", None) is not None:
            allowed_tools.append("inspect_window_heavy")

    # compact judge history
    judge_history = []
    for t in env.trace:
        if t.get("tool") == "judge":
            d = t.get("decision", {}) or {}
            judge_history.append({
                "depth": t.get("depth"),
                "when": t.get("when"),
                "phase": t.get("phase"),
                "signal_type": d.get("signal_type"),
                "convinced": d.get("convinced"),
                "hypothesis": d.get("hypothesis"),
                "critique": d.get("critique"),
                "missing_signal": d.get("missing_signal"),
                "next_call": d.get("next_call"),
                "backtrack": d.get("backtrack"),
            })

    gate_history = []
    for t in env.trace:
        if t.get("tool") == "gate":
            gate_history.append({
                "depth": t.get("depth"),
                "why": t.get("why"),
                "proposed": t.get("proposed"),
                "final": t.get("final"),
                "allowlist_summary": t.get("allowlist_summary"),
            })

    controller_state = {
        "tried": sorted(list(node.tried)) if hasattr(node, "tried") else None,
        "branch_score": getattr(node, "branch_score", None),
        "last_rewritten_query": getattr(node, "last_rewritten_query", None),
        "plan": None if node.plan is None else node.plan.__dict__,
    }

    judge_input = {
        "goal": query,
        "depth": node.depth,
        "allowed_tools": allowed_tools,
        "tool_schema": TOOL_SCHEMA,
        "memory": node.memory,
        "allowlist": allow.to_dict(),
        "judge_history": judge_history,
        "gate_history": gate_history[-20:],
        "controller_state": controller_state,
    }

    j = llm.judge(
        goal=query,
        memory=node.memory,
        allowlist=allow.to_dict(),
        allowed_tools=allowed_tools,
        tool_schema=TOOL_SCHEMA,
        taxonomy_plan=node.plan,
        judge_history=judge_history,
        gate_history=gate_history[-20:],
        controller_state=controller_state,
        temperature=0.0,
    )

    return j, allow, allowed_tools, judge_input

def _do(env, llm, query, node, gs, chosen_call, why, decision=None):
    # 1) controller chose something
    trace_controller_action(env.trace, node.depth, chosen_call, why, decision=decision)

    # 2) gate + execute (gate/tool logs already happen inside _step)
    child = _step(env, node, chosen_call, gs)

    # 3) recurse
    return None if child is None else dfs_backtracking_controller(env, llm, query, child, gs)

def dfs_backtracking_controller(env, llm, query, node, gs):
    if node.depth >= MAX_DEPTH:
        return None

    node.branch_score = max(node.branch_score, _best_score_in_memory(node.memory))

    # ---- Pre-judge when we have any memory ----
    if node.memory:
        j, allow, _allowed_tools, judge_input = _judge_node(env, llm, query, node)
        j["_judge_input"] = {
            "goal": query,
            "allowed_tools": _allowed_tools,
            "tool_schema": TOOL_SCHEMA,
            "memory": node.memory,
            "allowlist": allow.to_dict(),
            "taxonomy_plan": None if node.plan is None else node.plan.__dict__,
        }

        trace_judge(
            env.trace, node.depth,
            when="pre", phase="judge_pre",
            query=query, allowlist=allow.to_dict(),
            decision=j, judge_input=judge_input,
        )

        if j.get("convinced", False):
            return _summarize_with_llm(llm, query, node.memory)

        bt = j.get("backtrack")
        if bt:
            t = bt.get("type")

            if t == "try_next_segment":
                return None

            if t == "expand_search":
                chosen = {"tool": "search_segments", "query": query, "top_k": 6}
                return _do(env, llm, query, node, gs, chosen, "judge.backtrack=expand_search", decision=j)

            if t == "rewrite_query":
                rq = bt.get("query", query)
                node.last_rewritten_query = rq
                chosen = {"tool": "search_segments", "query": rq, "top_k": 6}
                return _do(env, llm, query, node, gs, chosen, "judge.backtrack=rewrite_query", decision=j)

        nc = j.get("next_call")
        if nc:
            ans = _do(env, llm, query, node, gs, nc, "judge.next_call", decision=j)
            if ans is not None:
                return ans
            #if blocked/fails, continue to deterministic DFS fallback below


            return _do(env, llm, query, node, gs, nc, "judge.next_call", decision=j)

    # ---- If no segments yet, do initial search ----
    seg_candidates = _top_segments_from_memory(node.memory)
    if not seg_candidates:
        chosen = {"tool": "search_segments", "query": query, "top_k": 3}
        return _do(env, llm, query, node, gs, chosen, "bootstrap.no_segments")

    # ---- Manual DFS over candidate segments ----
    for seg in seg_candidates:
        if ("seg", seg) in node.tried:
            continue
        node.tried.add(("seg", seg))

        chosen_refine = {
            "tool": "refine_in_segment",
            "query": query,
            "seg_idx": seg,
            "dense_fps": 8.0,
            "window_s": 2.0,
        }
        # log+step but DON'T immediately recurse: we need child1 for windows
        trace_controller_action(env.trace, node.depth, chosen_refine, f"dfs.refine.seg={seg}")
        child1 = _step(env, node, chosen_refine, gs)
        if child1 is None:
            continue

        wins = _top_windows_from_memory(child1.memory)
        if not wins:
            continue
        t0, t1 = wins[-1]

        chosen_inspect = {
            "tool": "inspect_window",
            "query": query,
            "t0": t0,
            "t1": t1,
            "fps": 4.0,
            "top_m": 5,
        }
        trace_controller_action(env.trace, child1.depth, chosen_inspect, "dfs.inspect")
        child2 = _step(env, child1, chosen_inspect, gs)
        if child2 is None:
            continue

        # post-judge after inspect
        j2, allow2, _allowed_tools2, judge_input2 = _judge_node(env, llm, query, child2)
        trace_judge(
            env.trace, child2.depth,
            when="post", phase="judge_post",
            query=query, allowlist=allow2.to_dict(),
            decision=j2, judge_input=judge_input2,
        )

        if j2.get("convinced", False):
            return _summarize_with_llm(llm, query, child2.memory)

        bt2 = j2.get("backtrack")
        if bt2 and bt2.get("type") == "try_next_segment":
            continue

        ans = dfs_backtracking_controller(env, llm, query, child2, gs)
        if ans is not None:
            return ans

    # ---- fallback expand search once ----
    if ("expand_search", None) not in node.tried:
        node.tried.add(("expand_search", None))
        chosen = {"tool": "search_segments", "query": query, "top_k": 6}
        return _do(env, llm, query, node, gs, chosen, "fallback.expand_search_once")

    return None

from dataclasses import dataclass
from typing import Literal, Optional

Tax = Literal["S1","S2","S3","S4","S5"]

from dataclasses import dataclass, field
from typing import List, Optional

def exec_call(env, call, gs):
    if call is None:
        return None

    tool = call.get("tool")
    if not tool:
        raise ValueError(f"exec_call: missing tool in call={call}")

    # run tool first; only charge if it worked
    cs = env.act(call)

    # ---- increment budgets AFTER success ----
    gs.setdefault("n_calls", 0)
    gs.setdefault("n_inspect", 0)
    gs.setdefault("n_refine", 0)
    gs.setdefault("n_heavy", 0)

    gs["n_calls"] += 1
    if tool == "inspect_window":
        gs["n_inspect"] += 1
    elif tool == "refine_in_segment":
        gs["n_refine"] += 1
    elif tool == "inspect_window_heavy":
        gs["n_heavy"] += 1

    return cs

def run_recursive_controller(env: VideoEnv, llm: VLLMJsonToolLLM, query: str) -> str:
    plan = classify_query_to_taxonomy(query)
    root = Node(depth=0, memory=[], plan=plan, evidence=EvidenceState())


    env.trace = []
    global_state = {
        "tried_calls": set(),          # hashable signatures of tool calls
        "inspect_counts": {},          # (t0,t1,fps)->count
        "n_calls": 0,
        "n_inspect": 0,
        "n_refine": 0,
        "best_score": float("-inf"),
        "best_memory": [],
        "n_heavy": 0,
        "evidence_state": EvidenceState()

    }

    ans = dfs_backtracking_controller(env, llm, query, root, global_state)

    if ans is None:
        mem = global_state.get("best_memory") or root.memory
        return _summarize_with_llm(llm, query, mem) or "No confident answer found."
    return ans

def _safe_load_json(text: str) -> Dict[str, Any]:
    """
    Robust JSON loader for model outputs.
    - Accepts raw JSON
    - Also tolerates ```json ... ``` fences
    - If extra text exists, tries to extract the first {...} block
    """
    s = (text or "").strip()

    # Strip common fenced code blocks
    if s.startswith("```"):
        lines = s.splitlines()
        # drop first line ``` or ```json
        if lines:
            lines = lines[1:]
        # drop last line ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # First attempt: direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Second attempt: find first JSON object substring
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        candidate = _repair_numeric_exprs(candidate)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from model output.\nRaw:\n{text}\nCandidate:\n{candidate}\nError: {e}")

    raise ValueError(f"Could not find a JSON object in model output.\nRaw:\n{text}")

import re

def _repair_numeric_exprs(s: str) -> str:
    """
    Repairs simple numeric expressions like:
        "t1": 53.68 + 5.0
    into:
        "t1": 58.68
    """
    pat = re.compile(r'(:\s*)(-?\d+(?:\.\d+)?)(\s*([+\-])\s*)(\d+(?:\.\d+)?)')

    def repl(m):
        prefix = m.group(1)
        a = float(m.group(2))
        op = m.group(4)
        b = float(m.group(5))
        val = a + b if op == "+" else a - b
        return f"{prefix}{val:.6f}".rstrip("0").rstrip(".")

    return pat.sub(repl, s)

def _last_window_from_memory(memory):
    wins = _top_windows_from_memory(memory)
    return list(wins[-1]) if wins else None


class VLLMJsonToolLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model


    def judge(
        self,
        goal: str,
        memory: List[Dict[str, Any]],
        allowlist: Dict[str, Any],
        allowed_tools: List[str],
        tool_schema: Dict[str, List[str]],
        temperature: float = 0.0,
        taxonomy_plan: Optional["TaxonomyPlan"] = None,
        judge_history: Optional[List[Dict[str, Any]]] = None,
        gate_history: Optional[List[Dict[str, Any]]] = None,
        controller_state: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
        # allow passing S1–S5 policy into the judge prompt
        """
        Text-RLM-style judge:
          - forms/updates hypothesis
          - critiques evidence sufficiency
          - either proposes a next tool call OR requests backtracking
        Must output JSON.
        """
        judge_system = (
                "You are a recursive judge/controller for video reasoning.\n"
                "You maintain a hypothesis and request evidence to confirm/refute it.\n"
                "Your reasoning MUST be grounded in the Signal Taxonomy S1–S5.\n"
                "\n"
                "Signal Taxonomy (use these labels explicitly):\n"
                "S1 Perceptual availability: signal is sampled but visually subtle/ambiguous -> need higher spatial detail / better view.\n"
                "S2 Microevent/transient: event is brief and may be missed by low FPS -> need temporal zoom / higher FPS around change.\n"
                "S3 Delayed relevance: early event matters only after later outcome -> reason from outcome then backtrack earlier.\n"
                "S4 Distributed evidence: no single moment suffices; answer requires aggregating multiple weak signals across time.\n"
                "S5 Ordering/contrast: answer depends on before/after comparison or relative order between states.\n"
                "\n"
                "You may:\n"
                "  (A) propose next_call (one tool call) using ONLY allowed seg/window ids from allowlist\n"
                "  (B) request backtrack if evidence is insufficient or contradictory\n"
                "Stop only when convinced.\n"
                "\n"
                "Output ONLY valid JSON with keys:\n"
                "hypothesis, critique, convinced, signal_type, missing_signal, next_call, backtrack\n"
                "Where:\n"
                "- signal_type is one of: 'S1','S2','S3','S4','S5'\n"
                "- missing_signal is a short string describing exactly what evidence is missing.\n"
                "\n"
                "IMPORTANT TOOL RULES (do not violate):\n"
                "- refine_in_segment parameters:\n"
                "    * seg_idx: int\n"
                "    * dense_fps: float\n"
                "    * window_s: float DURATION IN SECONDS (e.g., 1.0, 2.0, 5.0). NOT a [t0,t1] list.\n"
                "- inspect_window parameters (ALL REQUIRED):\n"
                "    * query: str\n"
                "    * t0,t1: floats ABSOLUTE TIMES in seconds from video start\n"
                "    * fps: float\n"
                "    * top_m: int\n"
                "- Never output window_s as a list. If you have a time range [t0,t1], you must call inspect_window instead.\n"
                "\n"
                "JSON rule: if you propose next_call, it MUST exactly match the tool schema (no missing keys).\n"
                "Selection policy by taxonomy:\n"
                "- If S1: prefer inspect_window on a short interval but with higher visual certainty.\n"
                "- If S2: prefer inspect_window with higher fps (temporal zoom) near a transition window.\n"
                "- If S3: backtrack to earlier windows/segments than the current best evidence.\n"
                "- If S4: request multiple windows across time until evidence stabilizes.\n"
                "- If S5: explicitly retrieve two windows (BEFORE and AFTER) and compare ordering.\n"
            )

        user_payload = {
            "goal": goal,
            "allowed_tools": allowed_tools,
            "tool_schema": tool_schema,
            "memory": memory,    
            "allowlist": allowlist,    # allowed_seg_idx, allowed_windows
            "taxonomy_plan": None if taxonomy_plan is None else taxonomy_plan.__dict__,  # expose plan knobs to judge
            "judge_history": judge_history,
            "gate_history" : gate_history,
            "controller_state" : controller_state,
            "output_format": {
                "hypothesis": "string (your current best guess)",
                "critique": "string (what's missing / what conflicts)",
                "convinced": "boolean",
                "next_call": "object or null (one of the allowed tools except summarize_answer unless convinced)",
                "backtrack": "object or null, one of: "
                             "{type:'try_next_segment'} | {type:'expand_search'} | {type:'rewrite_query', query:'...'}"
            },
            "rules": [
                "If convinced==true: do not propose next_call; controller will summarize separately.",
                "If not convinced: propose either next_call OR backtrack (exactly one).",
                "If proposing inspect/refine: use only seg_idx/t0/t1 from allowlist.",
                "Be conservative: if you didn't inspect frames, you're usually not convinced."
            ]
        }


        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )

        raw = (resp.choices[0].message.content or "").strip()
        j = _safe_load_json(raw)
        

        # Basic shape checks
        if "convinced" not in j:
            raise ValueError(f"Judge JSON missing 'convinced'. Raw:\n{raw}")
        if "hypothesis" not in j:
            j["hypothesis"] = ""
        if "critique" not in j:
            j["critique"] = ""

        # Enforce: if not convinced -> exactly one of next_call/backtrack
        next_call = j.get("next_call", None)
        backtrack = j.get("backtrack", None)

        if bool(j["convinced"]):
            j["next_call"] = None
            j["backtrack"] = None
            return j

        #problem - if model return both then pick a deterministic policy
        if next_call is not None and backtrack is not None:
            j["backtrack"] = None
            backtrack = None

        #enforce: must have exactly one
        next_call = j.get("next_call", None)
        backtrack = j.get("backtrack", None)

        if (next_call is None) == (backtrack is None):
            # both None or both set => not acceptable
            raise ValueError(
                f"Judge must set exactly one of next_call/backtrack when not convinced.\nRaw:\n{raw}"
            )

        # ... same up to (next_call is None) == (backtrack is None) check ...

        if next_call is not None:
            try:
                _validate_tool_call(next_call, allowed_tools, tool_schema)
                _enforce_allowlist(next_call, allowlist)
            except ValueError as e:
                j["next_call"] = None
                j["backtrack"] = {"type": "expand_search"}
                j["critique"] = (j.get("critique", "") + f" | allowlist_violation: {e}").strip()
                j["convinced"] = False

        # refresh locals after any mutation
        next_call = j.get("next_call", None)
        backtrack = j.get("backtrack", None)

        if backtrack is not None:
            if not isinstance(backtrack, dict) or "type" not in backtrack:
                raise ValueError(f"Bad backtrack object: {backtrack}")
            if backtrack["type"] not in ("try_next_segment", "expand_search", "rewrite_query"):
                raise ValueError(f"Unknown backtrack type: {backtrack['type']}")

        return j

    def decide(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = prompt.get("system", SYSTEM_PROMPT)
        tool_schema = prompt.get("tool_schema", TOOL_SCHEMA)
        memory = (prompt.get("memory") or [])[-4:]
        available = prompt.get("available_tools", []) or []
        if available == ["summarize_answer"]:
            allow = {}   # not needed for summarization
        else:
            allow = _extract_numbers_from_notes(memory)

        user_payload = {
            "goal": prompt["goal"],
            "available_tools": prompt["available_tools"],
            "tool_schema": tool_schema,
            "memory": memory,
            "allowlist": allow,
        }

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )

        text = (resp.choices[0].message.content or "").strip()

        try:
            call = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON.\nRaw:\n{text}\nError: {e}")

        if not isinstance(call, dict) or "tool" not in call:
            raise ValueError(f"Tool call must be a JSON object with key 'tool'. Got: {call}")

        tool = call["tool"]

        # Enforce allowed tools for this iteration
        allowed = set(prompt.get("available_tools", []) or [])
        if allowed and tool not in allowed:
            raise ValueError(
                f"LLM picked tool '{tool}' but allowed_tools={sorted(allowed)}.\n"
                f"Raw response:\n{text}")

        if tool not in TOOL_SCHEMA:
            raise ValueError(f"Unknown tool '{tool}'. Must be one of {list(TOOL_SCHEMA.keys())}")

        required = TOOL_SCHEMA[tool]
        missing = [k for k in required if k not in call]
        if missing:
            raise ValueError(f"Tool '{tool}' missing required keys: {missing}. Got: {call}")

        return call

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

    

def compact_slice(cs_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = {"tool": cs_dict.get("tool")}

    if "call" in cs_dict:
        out["call"] = cs_dict["call"]
    if "summary" in cs_dict:
        out["summary"] = cs_dict["summary"]
    if "window" in cs_dict:
        out["window"] = cs_dict["window"]
    if "stats" in cs_dict:
        out["stats"] = cs_dict["stats"]

    # keep top_frames (small)
    tfs = cs_dict.get("top_frames") or []
    if tfs:
        tf_keep = []
        for tf in tfs[:5]:
            t = tf.get("t")
            if t is None:
                continue
            tf_keep.append({
                "t": float(t),
                "score": tf.get("score"),
                "caption": tf.get("caption"),
                "vlm_answer": tf.get("vlm_answer"),
            })
        if tf_keep:
            out["top_frames"] = tf_keep

    # keep evidence notes too (ensure dict)
    ev = cs_dict.get("evidence", []) or []
    ev_out = []
    for e in ev[:10]:
        if isinstance(e, dict):
            ev_out.append(e)
        else:
            try:
                from dataclasses import asdict
                ev_out.append(asdict(e))
            except:
                ev_out.append({"note": str(e)})
    out["evidence"] = ev_out

    return out

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

def _round_win(t0: float, t1: float, ndigits: int = 2) -> Tuple[float, float]:
    # rounding makes “unique windows” robust to tiny float diffs
    return (round(float(t0), ndigits), round(float(t1), ndigits))

def compute_trace_metrics(
    trace: List[Dict[str, Any]],
    video_duration_s: float,) -> Dict[str, Any]:
    tool_counts: Dict[str, int] = {}
    inspect_windows: List[Tuple[float, float]] = []
    unique_inspect_windows = set()

    n_inspect_calls = 0
    n_refine_calls = 0
    n_search_calls = 0

    inspect_seconds_total = 0.0
    refine_seconds_total = 0.0

    # If you want a proxy for “how much expensive encoding happened”:
    # - refine_in_segment: typically *leads to* a window, and you later inspect it
    # - inspect_window: is the expensive high-fps sampling + captioning
    # We’ll treat inspect_window windows as “dense seconds”.
    dense_seconds_encoded = 0.0

    # Optional: count unique inspected frames (rough)
    approx_inspected_frames = 0

    for step in trace:
        tool = step.get("tool", "UNKNOWN")
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

        call = step.get("call", {}) or {}
        stats = step.get("stats", {}) or {}

        if tool == "search_segments":
            n_search_calls += 1

        elif tool == "refine_in_segment":
            n_refine_calls += 1
            # wallclock time spent inside refine (from your trace)
            refine_seconds_total += float(step.get("dt_s", 0.0) or 0.0)

        elif tool in ("inspect_window", "inspect_window_heavy"):
            n_inspect_calls += 1
            inspect_seconds_total += float(step.get("dt_s", 0.0) or 0.0)

            t0 = float(call["t0"])
            t1 = float(call["t1"])
            w = _round_win(t0, t1, ndigits=2)

            inspect_windows.append(w)
            unique_inspect_windows.add(w)

            dense_seconds_encoded += max(0.0, (t1 - t0))

            # if you logged n_samples, use it; else approximate via fps*(t1-t0)
            if "n_samples" in stats:
                approx_inspected_frames += int(stats["n_samples"])
            else:
                fps = float(call.get("fps", 0.0) or 0.0)
                approx_inspected_frames += int(max(0.0, fps * (t1 - t0)))

    pct_video_touched = 0.0
    if video_duration_s > 0:
        pct_video_touched = 100.0 * dense_seconds_encoded / video_duration_s

    return {
        "video_duration_s": video_duration_s,

        "tool_counts": tool_counts,
        "n_search_calls": n_search_calls,
        "n_refine_calls": n_refine_calls,
        "n_inspect_calls": n_inspect_calls,

        "n_unique_inspect_windows": len(unique_inspect_windows),
        "inspect_windows": inspect_windows[-10:],  # last few for debugging

        "dense_seconds_encoded": dense_seconds_encoded,
        "pct_video_touched": pct_video_touched,

        "inspect_wallclock_s": inspect_seconds_total,
        "refine_wallclock_s": refine_seconds_total,

        "approx_inspected_frames": approx_inspected_frames,
    }

def minimal_video_rlm(video_path: str, query: str):
    clip = Clip()

    cap = BlipCaptioner()
    env = VideoEnv(video_path, clip, seg_len_s=60.0, base_fps=1., captioner= cap)

    print("Building index (1 FPS CLIP + motion)...")
    env.build_index()

    llm = VLLMJsonToolLLM(model="Qwen/Qwen2.5-7B-Instruct")

    answer = run_recursive_controller(env, llm, query)
    print("\nFINAL ANSWER:")
    print(answer)

    print("\n=== TOOL TRACE (env.trace) ===")
    print(json.dumps(env.trace, indent=2))

    duration_s = getattr(env, "duration_s", 0.0)
    m = compute_trace_metrics(env.trace, float(duration_s))
    print("\n=== METRICS ===")
    print(json.dumps(m, indent=2))



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    minimal_video_rlm(args.video, args.query)