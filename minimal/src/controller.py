# controller.py
from __future__ import annotations

import json
from typing import Any, Dict
from clip_encoder import Clip
from env import VideoEnv
from typing import Any, Dict, List
from openai import OpenAI


#(Hussain) Change captioner such that we can important multiple captioners
from captioner import BlipCaptioner

SYSTEM_PROMPT = (
  "You are a video controller. Output ONLY a single JSON object.\n"
  "The JSON MUST have key 'tool'. Do NOT use 'tool_to_call' or any other key.\n"
  "Schema:\n"
  " - search_segments: {tool:'search_segments', query:str, top_k:int}\n"
  " - refine_in_segment: {tool:'refine_in_segment', query:str, seg_idx:int, dense_fps:float, window_s:float}\n"
  " - inspect_window: {tool:'inspect_window', query:str, t0:float, t1:float, fps:float, top_m:int}\n"
  " - summarize_answer: {tool:'summarize_answer', answer:str}\n"
  "Return EXACTLY one of these objects and nothing else."
)

TOOL_SCHEMA = {
  "search_segments": ["tool", "query", "top_k"],
  "refine_in_segment": ["tool", "query", "seg_idx", "dense_fps", "window_s"],
  "inspect_window": ["tool", "query", "t0", "t1", "fps", "top_m"],
  "summarize_answer": ["tool", "answer"],
}

MAX_DEPTH = 6
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class Node:
    depth: int
    memory: List[Dict[str, Any]]
    tried: set = field(default_factory=set)

    branch_score: float = float("-inf")
    chosen_seg: Optional[int] = None
    chosen_window: Optional[Tuple[float, float]] = None

    hypothesis: str = ""
    critique: str = ""
    last_rewritten_query: Optional[str] = None


def _best_score_in_memory(memory: List[Dict[str, Any]]) -> float:
    best = float("-inf")
    for step in memory:
        for ev in step.get("evidence", []) or []:
            try:
                best = max(best, float(ev.get("score", float("-inf"))))
            except:
                pass
    return best

def _extract_numbers_from_notes(memory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Optional helper: build an allowlist of seg_idx and windows the model can use."""
    segs = set()
    windows = []  # list of (t0,t1)
    for step in memory:
        for ev in step.get("evidence", []) or []:
            note = ev.get("note", "") or ""
            # seg=3, window=[180.00,240.00]
            if "seg=" in note:
                try:
                    seg = int(note.split("seg=")[1].split(",")[0])
                    segs.add(seg)
                except:
                    pass
            if "window=[" in note:
                try:
                    w = note.split("window=[", 1)[1].split("]")[0]
                    t0s, t1s = w.split(",")
                    windows.append((float(t0s), float(t1s)))
                except:
                    pass
            # [205.76,207.68]
            if note.startswith("[") and "," in note and note.endswith("]"):
                try:
                    w = note[1:-1]
                    t0s, t1s = w.split(",")
                    windows.append((float(t0s), float(t1s)))
                except:
                    pass
    return {"allowed_seg_idx": sorted(segs), "allowed_windows": windows[-10:]}

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

def _top_windows_from_memory(memory: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    wins = []
    for step in memory:
        for ev in step.get("evidence", []) or []:
            note = ev.get("note", "")
            if "window=[" in note:
                try:
                    w = note.split("window=[", 1)[1].split("]")[0]
                    t0s, t1s = w.split(",")
                    wins.append((float(t0s), float(t1s)))
                except:
                    pass
            if note.startswith("[") and note.endswith("]") and "," in note:
                try:
                    w = note[1:-1]
                    t0s, t1s = w.split(",")
                    wins.append((float(t0s), float(t1s)))
                except:
                    pass
    # unique, keep order
    out = []
    for w in wins:
        if w not in out:
            out.append(w)
    return out

def _validate_tool_call(
    call: Dict[str, Any],
    allowed_tools: List[str],
    tool_schema: Dict[str, List[str]],
) -> None:
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

    if tool == "inspect_window":
        for k in ("t0", "t1", "fps", "top_m"):
            if not isinstance(call[k], (int, float)):
                raise ValueError(f"inspect_window.{k} must be number. Call={call}")
        if float(call["t1"]) <= float(call["t0"]):
            raise ValueError(f"inspect_window requires t1 > t0. Call={call}")

    if tool == "summarize_answer":
        if not isinstance(call["answer"], str):
            raise ValueError(f"summarize_answer.answer must be str. Call={call}")

def _summarize_with_llm(llm: VLLMJsonToolLLM, query: str, memory: List[Dict[str, Any]]) -> str:
    prompt = {
        "system": SYSTEM_PROMPT,
        "goal": query,
        "available_tools": ["summarize_answer"],  # force summarize
        "memory": memory[-4:],
    }
    call = llm.decide(prompt)
    return call.get("answer", "")

def dfs_backtracking_controller(
    env: VideoEnv,
    llm: VLLMJsonToolLLM,
    query: str,
    node: Node,
) -> Optional[str]:
    if node.depth >= MAX_DEPTH:
        return None

    node.branch_score = max(node.branch_score, _best_score_in_memory(node.memory))

    # ------------------------------------------------------------------
    # ### JUDGE INSERT #1: early "should we stop / backtrack / act?" gate
    # ------------------------------------------------------------------
    # Only bother judging once we have *some* evidence (or after an inspect)
    if node.memory:
        allow = _extract_numbers_from_notes(node.memory)

        j = llm.judge(
            goal=query,
            memory=node.memory,
            allowlist=allow,
            allowed_tools=["search_segments", "refine_in_segment", "inspect_window"],
            tool_schema=TOOL_SCHEMA,
            temperature=0.0,
        )

        node.hypothesis = j.get("hypothesis", "")
        node.critique = j.get("critique", "")

        if j.get("convinced", False):
            return _summarize_with_llm(llm, query, node.memory)

        bt = j.get("backtrack")
        if bt:
            t = bt.get("type")
            if t == "try_next_segment":
                return None  # <-- THIS is DFS backtracking: parent tries next seg
            if t == "expand_search":
                # do a broader search from *this* node and continue DFS
                call = {"tool": "search_segments", "query": query, "top_k": 6}
                cs = env.act(call)
                cs_dict = env.context_to_dict(cs)
                child = Node(depth=node.depth + 1, memory=node.memory + [compact_slice(cs_dict)])
                return dfs_backtracking_controller(env, llm, query, child)
            if t == "rewrite_query":
                rq = bt.get("query", query)
                node.last_rewritten_query = rq
                call = {"tool": "search_segments", "query": rq, "top_k": 6}
                cs = env.act(call)
                cs_dict = env.context_to_dict(cs)
                child = Node(depth=node.depth + 1, memory=node.memory + [compact_slice(cs_dict)])
                return dfs_backtracking_controller(env, llm, rq, child)

        nc = j.get("next_call")
        if nc:
            cs = env.act(nc)
            cs_dict = env.context_to_dict(cs)
            child = Node(depth=node.depth + 1, memory=node.memory + [compact_slice(cs_dict)])
            return dfs_backtracking_controller(env, llm, query, child)


    seg_candidates = _top_segments_from_memory(node.memory)
    if not seg_candidates:
        call = {"tool": "search_segments", "query": query, "top_k": 3}
        cs = env.act(call)
        cs_dict = env.context_to_dict(cs)
        new_mem = node.memory + [compact_slice(cs_dict)]
        child = Node(depth=node.depth + 1, memory=new_mem)
        return dfs_backtracking_controller(env, llm, query, child)

    for seg in seg_candidates:
        if ("seg", seg) in node.tried:
            continue
        node.tried.add(("seg", seg))

        refine_call = {"tool": "refine_in_segment", "query": query, "seg_idx": seg, "dense_fps": 8.0, "window_s": 2.0}
        cs = env.act(refine_call)
        cs_dict = env.context_to_dict(cs)
        mem1 = node.memory + [compact_slice(cs_dict)]

        wins = _top_windows_from_memory(mem1)
        if not wins:
            continue
        (t0, t1) = wins[-1]

        insp_call = {"tool": "inspect_window", "query": query, "t0": t0, "t1": t1, "fps": 4.0, "top_m": 5}
        cs2 = env.act(insp_call)
        cs2_dict = env.context_to_dict(cs2)
        mem2 = mem1 + [compact_slice(cs2_dict)]

        child = Node(
            depth=node.depth + 1,
            memory=mem2,
            chosen_seg=seg,
            chosen_window=(t0, t1),
        )

        # --------------------------------------------------------------
        # ### JUDGE INSERT #2: judge *after* you inspected (most useful)
        # --------------------------------------------------------------
        allow2 = _extract_numbers_from_notes(child.memory)
        j2 = llm.judge(
            goal=query,
            memory=child.memory,
            allowlist=allow2,
            allowed_tools=["search_segments", "refine_in_segment", "inspect_window"],
            tool_schema=TOOL_SCHEMA,
            temperature=0.0,
        )
        child.hypothesis = j2.get("hypothesis", "")
        child.critique = j2.get("critique", "")

        if j2.get("convinced", False):
            return _summarize_with_llm(llm, query, child.memory)

        bt2 = j2.get("backtrack")
        if bt2 and bt2.get("type") == "try_next_segment":
            continue  # <-- try next seg (this is local backtracking)

        # Otherwise, continue DFS down this branch (it may inspect more windows, etc.)
        ans = dfs_backtracking_controller(env, llm, query, child)
        if ans is not None:
            return ans

    if ("expand_search", None) not in node.tried:
        node.tried.add(("expand_search", None))
        call = {"tool": "search_segments", "query": query, "top_k": 6}
        cs = env.act(call)
        cs_dict = env.context_to_dict(cs)
        child = Node(depth=node.depth + 1, memory=node.memory + [compact_slice(cs_dict)])
        return dfs_backtracking_controller(env, llm, query, child)

    return None


def run_recursive_controller(env: VideoEnv, llm: VLLMJsonToolLLM, query: str) -> str:
    root = Node(depth=0, memory=[])
    ans = dfs_backtracking_controller(env, llm, query, root)
    if ans is None:
        # last resort: summarize what we have (or return unsure)
        return _summarize_with_llm(llm, query, root.memory) or "No confident answer found."
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
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from model output.\nRaw:\n{text}\nCandidate:\n{candidate}\nError: {e}")

    raise ValueError(f"Could not find a JSON object in model output.\nRaw:\n{text}")

def _enforce_allowlist(call: Dict[str, Any], allow: Dict[str, Any]) -> None:
    wins = allow.get("allowed_windows", []) or []
    tool = call.get("tool")

    if tool == "inspect_window" and wins:
        t0, t1 = float(call["t0"]), float(call["t1"])
        eps = 1e-3

        ok = False
        for (a, b) in wins:
            a, b = float(a), float(b)
            if (t0 + eps) >= a and (t1 - eps) <= b:
                ok = True
                break

        if not ok:
            raise ValueError(
                f"inspect_window [{t0},{t1}] not contained in any allowed window. "
                f"Allowed: {wins[-10:]}"
            )

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
    ) -> Dict[str, Any]:
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
            "You may:\n"
            "  (A) propose next_call (one tool call) using ONLY allowed seg/window ids from allowlist\n"
            "  (B) request backtrack if evidence is insufficient or contradictory\n"
            "Stop only when convinced.\n"
            "Output ONLY valid JSON.\n"
        )

        user_payload = {
            "goal": goal,
            "allowed_tools": allowed_tools,
            "tool_schema": tool_schema,
            "memory": memory[-6:],     # keep it small
            "allowlist": allowlist,    # allowed_seg_idx, allowed_windows
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

        if (next_call is None) == (backtrack is None):
            # both None or both set => not acceptable
            raise ValueError(
                f"Judge must set exactly one of next_call/backtrack when not convinced.\nRaw:\n{raw}"
            )

        if next_call is not None:
            _validate_tool_call(next_call, allowed_tools, tool_schema)
            _enforce_allowlist(next_call, allowlist)


        if backtrack is not None:
            if not isinstance(backtrack, dict) or "type" not in backtrack:
                raise ValueError(f"Bad backtrack object: {backtrack}")
            if backtrack["type"] not in ("try_next_segment", "expand_search", "rewrite_query"):
                raise ValueError(f"Unknown backtrack type: {backtrack}")

        return j

    def decide(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = prompt.get("system", SYSTEM_PROMPT)

        memory = (prompt.get("memory") or [])[-4:]
        allow = _extract_numbers_from_notes(memory)

        user_payload = {
            "goal": prompt["goal"],
            "available_tools": prompt["available_tools"],
            "tool_schema": TOOL_SCHEMA,
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
                f"Raw response:\n{text}"
            )

        if tool not in TOOL_SCHEMA:
            raise ValueError(f"Unknown tool '{tool}'. Must be one of {list(TOOL_SCHEMA.keys())}")

        required = TOOL_SCHEMA[tool]
        missing = [k for k in required if k not in call]
        if missing:
            raise ValueError(f"Tool '{tool}' missing required keys: {missing}. Got: {call}")

        return call
    

def compact_slice(cs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Keep evidence compact so the LLM doesn't drown."""
    out = {"tool": cs_dict.get("tool")}
    if "window" in cs_dict: out["window"] = cs_dict["window"]
    if "stats" in cs_dict: out["stats"] = cs_dict["stats"]

    ev = cs_dict.get("evidence", []) or []
    # keep only top few items
    out["evidence"] = ev[:5]
    return out

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



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    minimal_video_rlm(args.video, args.query)