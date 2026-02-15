# controller.py
from __future__ import annotations

import json
from typing import Any, Dict
from clip_encoder import Clip
from env import VideoEnv


class JsonToolLLM:
    """
    Stub interface: swap implementation with vLLM/OpenAI client.
    Must return a single dict tool-call, e.g.:
      {"tool":"search_segments","query":"...","top_k":3}
    """
    def __init__(self, client=None, model: str = ""):
        self.client = client
        self.model = model

    def decide(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        # ---- Replace this stub with real model call ----
        # For now, raise to make sure you don't think it's "LLM driven" when it isn't.
        raise NotImplementedError("Wire this to vLLM/OpenAI. Must return JSON tool call.")

def compact_slice(cs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Keep evidence compact so the LLM doesn't drown."""
    out = {"tool": cs_dict.get("tool")}
    if "window" in cs_dict: out["window"] = cs_dict["window"]
    if "stats" in cs_dict: out["stats"] = cs_dict["stats"]

    ev = cs_dict.get("evidence", []) or []
    # keep only top few items
    out["evidence"] = ev[:5]
    return out


def run_llm_controller(env, llm: JsonToolLLM, query: str, max_iters: int = 6) -> List[Dict[str, Any]]:
    memory: List[Dict[str, Any]] = []

    TOOL_SCHEMA = {
        "search_segments": ["tool", "query", "top_k"],
        "refine_in_segment": ["tool", "query", "seg_idx", "dense_fps", "window_s"],
        "inspect_window": ["tool", "query", "t0", "t1", "fps", "top_m"],
        "summarize_answer": ["tool", "answer"],
    }

    for it in range(max_iters):
        prompt = {
            "system": (
                "You are a video controller. Output ONLY a single JSON object.\n"
                "Pick exactly ONE tool call.\n"
                "Use only seg_idx/timestamps that appear in memory evidence.\n"
                "If enough evidence exists, call summarize_answer."
            ),
            "goal": query,
            "available_tools": list(TOOL_SCHEMA.keys()),
            "memory": memory[-4:],  # keep last few steps
        }

        print("\n" + "=" * 90)
        print(f"ITER {it} - LLM INPUT")
        print(json.dumps(prompt, indent=2))

        tool_call = llm.decide(prompt)

        print(f"\nITER {it} - LLM OUTPUT (tool call)")
        print(json.dumps(tool_call, indent=2))

        if not isinstance(tool_call, dict) or "tool" not in tool_call:
            raise ValueError(f"LLM output must be dict with 'tool'. Got: {tool_call}")

        # Execute
        cs = env.act(tool_call)
        cs_dict = env.context_to_dict(cs)

        print(f"\nITER {it} - ENV OUTPUT (evidence)")
        print(json.dumps(cs_dict, indent=2))

        memory.append(compact_slice(cs_dict))

        if tool_call["tool"] == "summarize_answer":
            break

    print("\n=== TOOL TRACE (env.trace) ===")
    print(json.dumps(env.trace, indent=2))

    return memory

def minimal_video_rlm(video_path: str, query: str):
    clip = Clip()
    env = VideoEnv(video_path, clip, seg_len_s=60.0, base_fps=1.0)

    print("Building index (1 FPS CLIP + motion)...")
    env.build_index()

    llm = JsonToolLLM(client=None, model="")  # <-- wire real vLLM client here
    memory = run_llm_controller(env, llm, query=query, max_iters=6)
    return memory



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    minimal_video_rlm(args.video, args.query)