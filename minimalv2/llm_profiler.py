# llm_profiler.py
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Optional
import requests

from policy_schema import ExecutionPolicy



PROFILER_SYSTEM_PROMPT = """You are a profiling and routing controller for a video question answering system.
Your job: produce an ExecutionPolicy JSON that decides:
- reasoning mode (attribute/ordering/microevent/distributed/causal)
- tool plan (clip/ocr/caption/asr/objects)
- model tiers:
  - probe_tier
  - caption_tier
  - ocr_tier
  - asr_tier
  - cheap_answer_tier
  - answer_tier
  - fallback_answer_tier
  each in {none,cheap,medium,strong}
- whether to enable a cheap text-only answer stage before the VLM:
  - enable_cheap_stage: bool
  - text_answer_min_conf: float in [0,1]
- windowing/budget knobs:
  - probe_fps
  - probe_seg_len_s
  - probe_topk
  - window_len_s
  - strides
  - action_topk
- stopping knobs:
  - max_steps
  - eps_marginal_gain
- escalation thresholds:
  - min_retrieval_conf
  - min_answer_conf

IMPORTANT:
- Output must be STRICT JSON only (no markdown, no explanations, no extra text).
- Use conservative costs: prefer cheap tools/models if likely sufficient.
- If question asks about reading words on screen/signs/subtitles, include OCR in preferred_tools.
- If question asks about counting "how many" or "throughout the video", use distributed + captions/asr if helpful.
- If question asks for ordering of events or "sequence", use ordering (or microevent if it asks about after/before a specific event).
- If question asks "what happens after/before/then/next", use microevent and require_temporal_pair=true, smaller windows, finer stride.
- Enable the cheap text-only answer stage when OCR/caption/ASR evidence is likely enough to answer before running a VLM.
- If cheap text-only evidence is unlikely to be sufficient, set enable_cheap_stage=false.
- If unsure, choose attribute with clip and moderate steps.
- If the question mentions written words, labels, marked areas, logos, titles, signs, or text, include "ocr" in preferred_tools and set ocr_tier to at least "cheap".
- If the question is likely answered by spoken content, narration, explanation, or dialogue, include "asr" in preferred_tools and set asr_tier to at least "cheap".
- Questions asking what someone says, explains, mentions, or which option is correct based on speech should usually use ASR before relying on CLIP alone.
- If the answer is unlikely to be recoverable from frames alone, prefer ASR-enabled routing.

Example JSON output format:

{
  "mode": "attribute",
  "anchor_policy": "best_score",
  "coverage_target": 1,
  "require_temporal_pair": false,
  "preferred_tools": ["clip"],
  "probe_tier": "cheap",
  "caption_tier": "none",
  "ocr_tier": "none",
  "asr_tier": "none",
  "enable_cheap_stage": false,
  "cheap_answer_tier": "none",
  "text_answer_min_conf": 0.75,
  "answer_tier": "cheap",
  "fallback_answer_tier": "none",
  "probe_fps": 1.0,
  "probe_seg_len_s": 5.0,
  "probe_topk": 50,
  "window_len_s": 4.0,
  "strides": [0.5],
  "action_topk": 50,
  "max_steps": 20,
  "eps_marginal_gain": 0.01,
  "escalate_if_low_confidence": true,
  "min_retrieval_conf": 0.15,
  "min_answer_conf": 0.35,
  "confidence": 0.8,
  "rationale": "Simple visual attribute question."
}

Output STRICT JSON only.
"""

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Tries to parse JSON even if model included some stray text.
    """
    text = text.strip()
    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # heuristic: find first '{' and last '}'
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i : j + 1])

    raise ValueError("Could not parse JSON from profiler output")


def _coerce_policy(d: Dict[str, Any]) -> ExecutionPolicy:
    def get(name: str, default=None):
        return d.get(name, default)

    preferred_tools = tuple(get("preferred_tools", ("clip",)))
    strides = tuple(get("strides", (0.5,)))

    policy = ExecutionPolicy(
        mode=get("mode"),
        anchor_policy=get("anchor_policy", "best_score"),
        coverage_target=int(get("coverage_target", 1)),
        require_temporal_pair=bool(get("require_temporal_pair", False)),

        preferred_tools=preferred_tools,

        probe_tier=get("probe_tier", "cheap"),
        caption_tier=get("caption_tier", "none"),
        ocr_tier=get("ocr_tier", "none"),
        asr_tier=get("asr_tier", "none"),

        # cheap text stage
        enable_cheap_stage=bool(get("enable_cheap_stage", False)),
        cheap_answer_tier=get("cheap_answer_tier", "none"),
        text_answer_min_conf=float(get("text_answer_min_conf", 0.75)),

        # VLM answer stage
        answer_tier=get("answer_tier", "cheap"),

        # safest single-model default
        fallback_answer_tier=get("fallback_answer_tier", "none"),

        probe_fps=float(get("probe_fps", 1.0)),
        probe_seg_len_s=float(get("probe_seg_len_s", 5.0)),
        probe_topk=int(get("probe_topk", 50)),

        window_len_s=float(get("window_len_s", 4.0)),
        strides=strides,
        action_topk=int(get("action_topk", 50)),

        max_steps=int(get("max_steps", 20)),
        eps_marginal_gain=float(get("eps_marginal_gain", 0.01)),

        escalate_if_low_confidence=bool(get("escalate_if_low_confidence", True)),
        min_retrieval_conf=float(get("min_retrieval_conf", 0.15)),
        min_answer_conf=float(get("min_answer_conf", 0.35)),

        confidence=float(get("confidence", 0.6)),
        rationale=str(get("rationale", "")),
    )

    if policy.mode not in ("attribute", "ordering", "microevent", "distributed", "causal"):
        raise ValueError(f"Bad mode: {policy.mode}")
    if policy.coverage_target < 1 or policy.coverage_target > 10:
        raise ValueError(f"Bad coverage_target: {policy.coverage_target}")
    if policy.window_len_s <= 0:
        raise ValueError("window_len_s must be > 0")
    if any(s <= 0 for s in policy.strides):
        raise ValueError("strides must be > 0")
    if policy.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if not (0.0 <= policy.text_answer_min_conf <= 1.0):
        raise ValueError("text_answer_min_conf must be in [0,1]")
    if not (0.0 <= policy.min_answer_conf <= 1.0):
        raise ValueError("min_answer_conf must be in [0,1]")

    return policy


def profile_query_llm(
    query: str,
    *,
    base_url: str = "http://localhost:8000/v1",
    model: str = "gpt-4o-mini",   # you should set to a *text* model served by vLLM, or any OpenAI-compatible endpoint
    temperature: float = 0.0,
    timeout_s: float = 30.0,
) -> ExecutionPolicy:
    """
    Calls OpenAI-compatible /chat/completions on `base_url`.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": PROFILER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": 512,
    }

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()

    text = j["choices"][0]["message"]["content"]
    print("\n===== RAW PROFILER OUTPUT =====")
    print(text)
    print("===== END RAW PROFILER OUTPUT =====\n")
    d = _extract_json(text)
    d = _patch_policy_with_guardrails(query, d)
    return _coerce_policy(d)

def _looks_speechish_question(query: str) -> bool:
    q = query.lower()

    speech_markers = [
        "say", "says", "said",
        "speak", "speaks", "speaking",
        "talk", "talking", "talks",
        "explain", "explains", "explaining", "explained",
        "mention", "mentions", "mentioned",
        "according to",
        "what does he say",
        "what does she say",
        "what is he saying",
        "what is she saying",
        "what is being explained",
        "what is the speaker saying",
        "what does the narrator say",
        "which of the following",
    ]

    return any(m in q for m in speech_markers)

def _patch_policy_with_guardrails(query: str, d: Dict[str, Any]) -> Dict[str, Any]:
    q = query.lower()

    # ---------- OCR/textish guardrail ----------
    text_markers = [
        "text", "word", "words", "written",
        "caption", "subtitles",
        "sign", "label", "logo", "title",
        "headline", "name", "number", "date",
        "displayed", "shown", "marked",
        "complete text",
    ]

    if any(m in q for m in text_markers):
        preferred = list(d.get("preferred_tools", ["clip"]))

        # OCR should be first tool
        if "ocr" not in preferred:
            preferred = ["ocr"] + preferred
        else:
            preferred.remove("ocr")
            preferred = ["ocr"] + preferred

        d["preferred_tools"] = preferred

        if d.get("ocr_tier", "none") == "none":
            d["ocr_tier"] = "cheap"

        d["enable_cheap_stage"] = True
        if d.get("cheap_answer_tier", "none") == "none":
            d["cheap_answer_tier"] = "cheap"

    # ---------- Speech/ASR guardrail ----------
    if _looks_speechish_question(query):
        preferred = list(d.get("preferred_tools", ["clip"]))

        # ASR should be first tool
        if "asr" not in preferred:
            preferred = ["asr"] + preferred
        else:
            preferred.remove("asr")
            preferred = ["asr"] + preferred

        d["preferred_tools"] = preferred

        if d.get("asr_tier", "none") == "none":
            d["asr_tier"] = "cheap"

        d["enable_cheap_stage"] = True
        if d.get("cheap_answer_tier", "none") == "none":
            d["cheap_answer_tier"] = "cheap"

        # speech questions usually not microevent
        temporal_markers = ["after", "before", "next", "then", "later", "earlier"]
        if d.get("mode") == "microevent" and not any(t in q for t in temporal_markers):
            d["mode"] = "attribute"
            d["require_temporal_pair"] = False

    # ---------- consistency ----------
    if d.get("enable_cheap_stage", False) and d.get("cheap_answer_tier", "none") == "none":
        d["cheap_answer_tier"] = "cheap"

    return d