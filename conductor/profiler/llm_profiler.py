from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple
import requests

from .policy_schema import ExecutionPolicy
from .query_profile_schema import QueryProfile

PROFILER_SYSTEM_PROMPT = """
You are a query profiler for a video question answering system.

Given a user query, output STRICT JSON ONLY with the following schema:

{
  "analysis": {
    "primary_question_clause": "<string>",
    "primary_question_target": "visual" | "speech" | "text" | "speech_with_visual_anchor" | "text_with_visual_anchor",
    "needs_vlm": true | false,
    "needs_asr": true | false,
    "needs_ocr": true | false,
    "notes": "<short string>",
    "visual_detail_level": "low" | "medium" | "high",
    "object_scale": "large" | "medium" | "small" | "unknown",
    "needs_precise_local_view": true | false
  },
  "profile": {
    "evidence_sources": ["vlm" | "ocr" | "asr", ...],
    "answer_type": "visual_answer" | "spoken_span" | "text_span",
    "reference_type": "none" | "direct_answer" | "text_reference" | "speech_reference" | "visual_reference",
    "inspection_pattern": "vlm_only" | "asr_only" | "ocr_only" | "vlm_anchor_then_asr" | "asr_anchor_then_vlm" | "vlm_anchor_then_ocr" | "ocr_anchor_then_vlm",
    "confidence": <float 0..1>,
    "rationale": "<short string>"
  }
}

Decision rules:

- Use "asr_only" when the question asks only what was said or spoken.
- Use "vlm_only" when the question asks only about visible objects, actions, scenes, colors, shape, clothing, or visual attributes.
- Use "ocr_only" when the question asks only for visible written text.
- Use "vlm_anchor_then_asr" when the question asks what was said while/when a visible object, action, person, scene, or event is shown.
- Use "asr_anchor_then_vlm" when the question asks what visual thing appears when/after/before a spoken phrase is said.
- Use "vlm_anchor_then_ocr" when the question asks what text is visible on a visually specified object or region.
- Use "ocr_anchor_then_vlm" when visible text identifies the window, but the answer is visual.
- Do not use temporal ordering categories for now.
- Do not output "clip". Use "vlm" for all visual inspection.
- If both speech and visual anchoring are needed, evidence_sources must include both "vlm" and "asr".
- If both text and visual anchoring are needed, evidence_sources must include both "vlm" and "ocr".
- enable_cheap_stage should be true only for asr_only or ocr_only.
- Output JSON only.

"""


@dataclass
class ProfilerResult:
    analysis: Dict[str, Any]
    query_profile: QueryProfile
    execution_policy: ExecutionPolicy
    raw_profile: Dict[str, Any]
    raw_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis,
            "query_profile": asdict(self.query_profile),
            "execution_policy": asdict(self.execution_policy),
            "raw_profile": self.raw_profile,
            "raw_json": self.raw_json,
        }


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i : j + 1])

    raise ValueError("Could not parse JSON from profiler output")


def _coerce_query_profile(d: Dict[str, Any]) -> QueryProfile:
    evidence_sources = tuple(d.get("evidence_sources", ["vlm"]))
    if not evidence_sources:
        evidence_sources = ("vlm",)

    evidence_sources = tuple(
        "vlm" if x == "clip" else x
        for x in evidence_sources
    )

    return QueryProfile(
        evidence_sources=evidence_sources,
        answer_type=str(d.get("answer_type", "visual_answer")),
        reference_type=str(d.get("reference_type", "none")),
        inspection_pattern=str(d.get("inspection_pattern", "vlm_only")),
        enable_cheap_stage=bool(d.get("enable_cheap_stage", False)),
        confidence=float(d.get("confidence", 0.6)),
        rationale=str(d.get("rationale", "")),
    )


def _query_profile_to_execution_policy(
    qp: QueryProfile,
    analysis: Dict[str, Any] | None = None,
) -> ExecutionPolicy:
    analysis = analysis or {}

    preferred_tools = tuple(
        "vlm" if x == "clip" else x
        for x in qp.evidence_sources
    )

    ocr_tier = "cheap" if "ocr" in preferred_tools else "none"
    asr_tier = "cheap" if "asr" in preferred_tools else "none"

    # Simple benchmark-only defaults.
    mode = "attribute"
    require_temporal_pair = False
    window_len_s = 5.0
    strides = (0.5,)
    max_steps = 8

    probe_fps = 0.5
    probe_topk = 8
    action_topk = 4

    pattern = qp.inspection_pattern

    if pattern == "vlm_anchor_then_asr":
        # VLM first, then ASR on selected visual windows.
        probe_topk = 8
        action_topk = 4
        enable_cheap_stage = False
        cheap_answer_tier = "none"
        answer_tier = "heavy"

    elif pattern == "asr_anchor_then_vlm":
        # ASR first, then VLM on selected speech windows.
        probe_topk = 16
        action_topk = 4
        enable_cheap_stage = False
        cheap_answer_tier = "none"
        answer_tier = "heavy"

    elif pattern == "asr_only":
        probe_topk = 16
        action_topk = 0
        enable_cheap_stage = True
        cheap_answer_tier = "cheap"
        answer_tier = "cheap"

    elif pattern == "ocr_only":
        probe_topk = 8
        action_topk = 0
        enable_cheap_stage = True
        cheap_answer_tier = "cheap"
        answer_tier = "cheap"

    elif pattern == "vlm_only":
        probe_topk = 8
        action_topk = 8
        enable_cheap_stage = False
        cheap_answer_tier = "none"
        answer_tier = "heavy"

    elif pattern == "vlm_anchor_then_ocr":
        probe_topk = 8
        action_topk = 4
        enable_cheap_stage = False
        cheap_answer_tier = "none"
        answer_tier = "heavy"

    elif pattern == "ocr_anchor_then_vlm":
        probe_topk = 16
        action_topk = 4
        enable_cheap_stage = False
        cheap_answer_tier = "none"
        answer_tier = "heavy"

    else:
        enable_cheap_stage = bool(qp.enable_cheap_stage)
        cheap_answer_tier = "cheap" if enable_cheap_stage else "none"
        answer_tier = "cheap" if enable_cheap_stage else "heavy"

    return ExecutionPolicy(
        mode=mode,
        anchor_policy="best_score",
        coverage_target=1,
        require_temporal_pair=require_temporal_pair,
        preferred_tools=preferred_tools,
        probe_tier="cheap",
        caption_tier="none",
        ocr_tier=ocr_tier,
        asr_tier=asr_tier,
        enable_cheap_stage=enable_cheap_stage,
        cheap_answer_tier=cheap_answer_tier,
        text_answer_min_conf=0.75,
        answer_tier=answer_tier,
        fallback_answer_tier="none",
        probe_fps=probe_fps,
        probe_seg_len_s=60,
        probe_topk=probe_topk,
        window_len_s=window_len_s,
        strides=strides,
        action_topk=action_topk,
        max_steps=max_steps,
        eps_marginal_gain=0.01,
        escalate_if_low_confidence=True,
        min_retrieval_conf=0.15,
        min_answer_conf=0.35,
        confidence=qp.confidence,
        rationale=qp.rationale,
    )

def _patch_query_profile_with_guardrails(query: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Benchmark-only profiler patch.

    Goal:
    - Do not use temporal categories.
    - Do not output clip.
    - Use VLM for expensive visual inspection.
    - Use ASR/OCR as cheap anchors when they can reduce VLM work.
    """

    analysis = d.get("analysis", {}) if isinstance(d.get("analysis"), dict) else {}
    profile = d.get("profile", d) if isinstance(d.get("profile", d), dict) else dict(d)
    profile = dict(profile)

    # LLM-decided fields from the new prompt.
    needs_vlm = bool(analysis.get("needs_vlm", False))
    needs_asr = bool(analysis.get("needs_asr", False))
    needs_ocr = bool(analysis.get("needs_ocr", False))
    target = str(analysis.get("primary_question_target", "")).strip()

    # Normalize any old output.
    evidence = profile.get("evidence_sources", [])
    evidence = ["vlm" if x == "clip" else x for x in evidence]
    profile["evidence_sources"] = evidence

    pattern = str(profile.get("inspection_pattern", "")).strip()

    valid_patterns = {
        "vlm_only",
        "asr_only",
        "ocr_only",
        "vlm_anchor_then_asr",
        "asr_anchor_then_vlm",
        "vlm_anchor_then_ocr",
        "ocr_anchor_then_vlm",
    }

    # Trust the LLM pattern if it is valid.
    # Otherwise infer from the LLM analysis booleans, not word matching.
    if pattern not in valid_patterns:
        if needs_asr and needs_vlm:
            if target == "speech_with_visual_anchor":
                pattern = "vlm_anchor_then_asr"
            else:
                pattern = "asr_anchor_then_vlm"

        elif needs_ocr and needs_vlm:
            if target == "text_with_visual_anchor":
                pattern = "vlm_anchor_then_ocr"
            else:
                pattern = "ocr_anchor_then_vlm"

        elif needs_asr:
            pattern = "asr_only"

        elif needs_ocr:
            pattern = "ocr_only"

        else:
            pattern = "vlm_only"

    profile["inspection_pattern"] = pattern

    if pattern == "asr_only":
        profile["evidence_sources"] = ["asr"]
        profile["answer_type"] = "spoken_span"
        profile["reference_type"] = "direct_answer"
        profile["enable_cheap_stage"] = True

    elif pattern == "ocr_only":
        profile["evidence_sources"] = ["ocr"]
        profile["answer_type"] = "text_span"
        profile["reference_type"] = "direct_answer"
        profile["enable_cheap_stage"] = True

    elif pattern == "vlm_only":
        profile["evidence_sources"] = ["vlm"]
        profile["answer_type"] = "visual_answer"
        profile["reference_type"] = "none"
        profile["enable_cheap_stage"] = False

    elif pattern == "vlm_anchor_then_asr":
        # Expensive VLM first finds visually relevant windows;
        # ASR only runs on those selected windows.
        profile["evidence_sources"] = ["vlm", "asr"]
        profile["answer_type"] = "spoken_span"
        profile["reference_type"] = "visual_reference"
        profile["enable_cheap_stage"] = False

    elif pattern == "asr_anchor_then_vlm":
        # Cheap ASR scans broadly;
        # expensive VLM only runs on ASR-selected windows.
        profile["evidence_sources"] = ["asr", "vlm"]
        profile["answer_type"] = "visual_answer"
        profile["reference_type"] = "speech_reference"
        profile["enable_cheap_stage"] = False

    elif pattern == "vlm_anchor_then_ocr":
        profile["evidence_sources"] = ["vlm", "ocr"]
        profile["answer_type"] = "text_span"
        profile["reference_type"] = "visual_reference"
        profile["enable_cheap_stage"] = False

    elif pattern == "ocr_anchor_then_vlm":
        profile["evidence_sources"] = ["ocr", "vlm"]
        profile["answer_type"] = "visual_answer"
        profile["reference_type"] = "text_reference"
        profile["enable_cheap_stage"] = False

    # Remove old field if model emitted it.
    profile.pop("temporal_requirement", None)

    return profile


def profile_query_llm(
    query: str,
    *,
    base_url: str = "http://localhost:8000/v1",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_s: float = 30.0,
    api_key: str | None = None,
) -> ProfilerResult:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": PROFILER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": 768,
    }

    r = requests.post(
    url,
    json=payload,
    timeout=timeout_s,
    headers={"Authorization": "Bearer EMPTY"},
    )
    r.raise_for_status()
    j = r.json()

    text = j["choices"][0]["message"]["content"]
    print("\n===== RAW PROFILER OUTPUT =====")
    print(text)
    print("===== END RAW PROFILER OUTPUT =====\n")

    raw_json = _extract_json(text)
    analysis = raw_json.get("analysis", {})
    raw_profile = raw_json.get("profile", raw_json)

    print("\n===== RAW PROFILER ANALYSIS =====")
    print(json.dumps(analysis, indent=2))
    print("\n===== RAW PARSED PROFILE =====")
    print(json.dumps(raw_profile, indent=2))

    patched_profile = _patch_query_profile_with_guardrails(query, raw_json)

    print("\n===== PATCHED PROFILE =====")
    print(json.dumps(patched_profile, indent=2))

    qp = _coerce_query_profile(patched_profile)
    policy = _query_profile_to_execution_policy(qp, analysis=analysis)

    print("\n===== COMPILED EXECUTION POLICY =====")
    print(asdict(policy))

    return ProfilerResult(
        analysis=analysis,
        query_profile=qp,
        execution_policy=policy,
        raw_profile=patched_profile,
        raw_json=raw_json,
    )


def profile_query_llm_legacy(
    query: str,
    *,
    base_url: str = "http://localhost:8000/v1",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_s: float = 30.0,
) -> Tuple[ExecutionPolicy, Dict[str, Any]]:
    """
    Backward-compatible wrapper for older code that expects:
        policy, analysis = profile_query_llm(...)
    """
    result = profile_query_llm(
        query=query,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
    )
    return result.execution_policy, result.analysis