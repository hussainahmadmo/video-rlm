# llm_profiler.py
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Optional
import requests

from policy_schema import ExecutionPolicy

#analysis = semantic understanding of the question.
#profile = 
PROFILER_SYSTEM_PROMPT = """
You are a query profiler for a video question answering system.

Given a user query, output STRICT JSON ONLY with the following schema:

{
  "analysis": {
    "scene_localization_cues": ["<string>", ...],
    "primary_question_clause": "<string>",
    "primary_question_target": "visual_attribute" | "visual_object" | "text_span" | "spoken_span" | "event" | "count" | "temporal_relation",
    "required_evidence": ["clip" | "ocr" | "asr", ...],
    "is_scene_description_only": true | false,
    "needs_visible_text": true | false,
    "needs_spoken_content": true | false,
    "needs_temporal_reasoning": true | false,
    "notes": "<short string>",
    "visual_detail_level": "low" | "medium" | "high",
    "object_scale": "large" | "medium" | "small" | "unknown",
    "needs_precise_local_view": true | false,
    "notes": "<short string>"
  },
  "profile": {
    "evidence_sources": ["clip" | "ocr" | "asr", ...],
    "answer_type": "visual_attribute" | "visual_object" | "text_span" | "spoken_span" | "event" | "count" | "temporal_relation",
    "reference_type": "none" | "direct_answer" | "text_reference" | "speech_reference",
    "temporal_requirement": "none" | "ordering" | "before_after" | "distributed" | "causal",
    "inspection_pattern": "static_localized" | "temporal_followup" | "distributed_scan" | "speech_first" | "ocr_first",
    "enable_cheap_stage": true | false,
    "confidence": <float 0..1>,
    "rationale": "<short string>"
  }
}

Guidelines:
- First identify the actual thing being asked in the final question clause.
- Earlier descriptive text is often only for scene localization.
- Put scene-only descriptive cues into analysis.scene_localization_cues.
- Put the real ask into analysis.primary_question_clause.
- Use "ocr" only when written text on screen is important.
- Use "asr" only when spoken language is important.
- Use "clip" when the answer is mainly visual.
- Use multiple evidence_sources only if more than one is clearly needed.
- If the question asks for the text itself, use answer_type="text_span" and reference_type="direct_answer".
- If the question mentions text only to identify a region/object, use reference_type="text_reference" and keep the true answer_type visual.
- If the question asks for a visual property like shape, color, material, size, clothing, handheld item, or visible object, use clip and a visual answer_type.
- If the question asks what happens first / before / after, use answer_type="temporal_relation".
- Use temporal_requirement="none" for static questions.
- Use inspection_pattern="static_localized" for static localized visual questions.
- Use inspection_pattern="speech_first" for ASR-first questions.
- Use inspection_pattern="ocr_first" for OCR-first questions.
- enable_cheap_stage should be true only when OCR/ASR text alone may answer the question.
- Do not let long scene descriptions dominate tool choice.
- Output STRICT JSON only.
- Also assess how visually demanding the question is.
- Set analysis.visual_detail_level="high" for small handheld objects, accessories, fine-grained object distinctions, shape/material/border questions, or subtle visible differences.
- Set analysis.object_scale="small" when the target object is likely small relative to the frame.
- Set analysis.needs_precise_local_view=true when answering likely benefits from tighter or higher-detail visual inspection.
- Use low or medium when the target is large, obvious, or scene-level.

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
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_s: float = 30.0,
):
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

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()

    text = j["choices"][0]["message"]["content"]
    print("\n===== RAW PROFILER OUTPUT =====")
    print(text)
    print("===== END RAW PROFILER OUTPUT =====\n")

    d = _extract_json(text)

    print("\n===== RAW PARSED PROFILER JSON =====")
    print(json.dumps(d, indent=2))

    analysis = d.get("analysis", {})
    profile = d.get("profile", d)

    print("\n===== RAW PROFILER ANALYSIS =====")
    print(json.dumps(analysis, indent=2))

    print("\n===== RAW PARSED PROFILE =====")
    print(json.dumps(profile, indent=2))

    profile = _patch_query_profile_with_guardrails(query, profile)

    print("\n===== PATCHED PROFILE =====")
    print(json.dumps(profile, indent=2))

    qp = _coerce_query_profile(profile)
    policy = _query_profile_to_execution_policy(qp)

    print("\n===== COMPILED EXECUTION POLICY =====")
    print(asdict(policy))

    return policy, analysis

from query_profile_schema import QueryProfile


def _query_profile_to_execution_policy(qp: QueryProfile) -> ExecutionPolicy:
    preferred_tools = tuple(qp.evidence_sources)

    ocr_tier = "cheap" if "ocr" in qp.evidence_sources else "none"
    asr_tier = "cheap" if "asr" in qp.evidence_sources else "none"

    if qp.temporal_requirement in ("ordering", "before_after"):
        mode = "ordering"
        require_temporal_pair = True
        window_len_s = 2.0
        max_steps = 10
        strides = (0.5,)
    elif qp.temporal_requirement == "distributed":
        mode = "distributed"
        require_temporal_pair = False
        window_len_s = 4.0
        max_steps = 20
        strides = (0.5,)
    elif qp.temporal_requirement == "causal":
        mode = "causal"
        require_temporal_pair = False
        window_len_s = 4.0
        max_steps = 20
        strides = (0.5,)
    else:
        mode = "attribute"
        require_temporal_pair = False
        window_len_s = 4.0
        max_steps = 20
        strides = (0.5,)

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
        enable_cheap_stage=qp.enable_cheap_stage,
        cheap_answer_tier="cheap" if qp.enable_cheap_stage else "none",
        text_answer_min_conf=0.75,
        answer_tier="cheap",
        fallback_answer_tier="none",
        probe_fps=1.0,
        probe_seg_len_s=5.0,
        probe_topk=50,
        window_len_s=window_len_s,
        strides=strides,
        action_topk=50,
        max_steps=max_steps,
        eps_marginal_gain=0.01,
        escalate_if_low_confidence=True,
        min_retrieval_conf=0.15,
        min_answer_conf=0.35,
        confidence=qp.confidence,
        rationale=qp.rationale,
    )

def _coerce_query_profile(d: Dict[str, Any]) -> QueryProfile:
    evidence_sources = tuple(d.get("evidence_sources", ["clip"]))

    if not evidence_sources:
        evidence_sources = ("clip",)

    return QueryProfile(
        evidence_sources=evidence_sources,
        answer_type=str(d.get("answer_type", "visual_object")),
        reference_type=str(d.get("reference_type", "none")),
        temporal_requirement=str(d.get("temporal_requirement", "none")),
        inspection_pattern=str(d.get("inspection_pattern", "static_localized")),
        enable_cheap_stage=bool(d.get("enable_cheap_stage", False)),
        confidence=float(d.get("confidence", 0.6)),
        rationale=str(d.get("rationale", "")),
    )

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


def _looks_textish_question(query: str) -> bool:
    q = query.lower()

    text_markers = [
        "text",
        "word",
        "words",
        "written",
        "subtitle",
        "subtitles",
        "caption",
        "captions",
        "sign",
        "label",
        "logo",
        "title",
        "headline",
        "name",
        "number",
        "date",
        "displayed",
        "shown",
        "marked",
        "complete text",
        "what is written",
        "what does the text say",
    ]
    return any(m in q for m in text_markers)

def _patch_query_profile_with_guardrails(query: str, d: Dict[str, Any]) -> Dict[str, Any]:
    q = query.lower()

    # Support both nested {"analysis": ..., "profile": ...} and flat dicts
    analysis = d.get("analysis", {}) if isinstance(d.get("analysis"), dict) else {}
    profile = d.get("profile", d) if isinstance(d.get("profile", d), dict) else dict(d)
    profile = dict(profile)  # copy

    primary_target = analysis.get("primary_question_target")
    needs_spoken = bool(analysis.get("needs_spoken_content", False))
    needs_visible_text = bool(analysis.get("needs_visible_text", False))
    needs_temporal = bool(analysis.get("needs_temporal_reasoning", False))
    required_evidence = list(analysis.get("required_evidence", []))


    # Fallback when patch receives only flat profile
    if primary_target is None:
        primary_target = profile.get("answer_type")


    text_markers = [
        "text", "word", "words", "written", "subtitle", "subtitles",
        "caption", "captions", "sign", "label", "logo", "title",
        "headline", "name", "number", "date", "displayed", "shown",
        "marked", "complete text", "what is written", "what does the text say",
    ]

    speech_markers = [
        "say", "says", "said", "speak", "speaks", "speaking",
        "talk", "talking", "talks", "explain", "explains", "explaining",
        "explained", "mention", "mentions", "mentioned", "according to",
        "what does he say", "what does she say", "what is he saying",
        "what is she saying", "what is being explained",
        "what is the speaker saying", "what does the narrator say",
    ]

    attribute_markers = [
        "shape", "color", "material", "size", "border", "area", "region"
    ]

    ordering_markers = [
        "first", "before", "after", "next", "then", "later", "earlier"
    ]

    has_text = any(m in q for m in text_markers)
    has_speech = any(m in q for m in speech_markers)
    has_attr = any(m in q for m in attribute_markers)
    has_order = any(m in q for m in ordering_markers)

    # ------------------------------------------------------------------
    # 1) Respect structured analysis when available
    # ------------------------------------------------------------------



    # Spoken answer
    if primary_target == "spoken_span":
        profile["evidence_sources"] = ["asr"]
        profile["answer_type"] = "spoken_span"
        profile["reference_type"] = "direct_answer"
        profile["temporal_requirement"] = "ordering" if needs_temporal else "none"
        profile["inspection_pattern"] = "speech_first"
        profile["enable_cheap_stage"] = True
        return profile

    # Visual target, no speech/text needed -> pure clip
    if primary_target in {"visual_object", "visual_attribute"} and not needs_spoken and not needs_visible_text:
        profile["evidence_sources"] = ["clip"]
        profile["answer_type"] = primary_target
        profile["reference_type"] = "none"
        profile["temporal_requirement"] = "ordering" if needs_temporal else "none"
        profile["inspection_pattern"] = "temporal_followup" if needs_temporal else "static_localized"
        profile["enable_cheap_stage"] = False
        return profile

    # Visual target, but speech needed to localize the moment -> ASR + clip
    if primary_target in {"visual_object", "visual_attribute"} and needs_spoken:
        profile["evidence_sources"] = ["asr", "clip"]
        profile["answer_type"] = primary_target
        profile["reference_type"] = "speech_reference"
        profile["temporal_requirement"] = "ordering" if needs_temporal else "none"
        profile["inspection_pattern"] = "speech_first"
        profile["enable_cheap_stage"] = False
        return profile

    # Visible text answer
    if primary_target == "text_span" or needs_visible_text:
        if primary_target in {"visual_object", "visual_attribute"}:
            profile["evidence_sources"] = ["ocr", "clip"]
            profile["answer_type"] = primary_target
            profile["reference_type"] = "text_reference"
            profile["temporal_requirement"] = "none"
            profile["inspection_pattern"] = "ocr_first"
            profile["enable_cheap_stage"] = False
        else:
            profile["evidence_sources"] = ["ocr"]
            profile["answer_type"] = "text_span"
            profile["reference_type"] = "direct_answer"
            profile["temporal_requirement"] = "none"
            profile["inspection_pattern"] = "ocr_first"
            profile["enable_cheap_stage"] = True
        return profile

    # Temporal relation
    if primary_target == "temporal_relation":
        # If speech is involved, support both
        if needs_spoken:
            profile["evidence_sources"] = ["asr", "clip"]
            profile["reference_type"] = "speech_reference"
            profile["inspection_pattern"] = "speech_first"
        else:
            profile["evidence_sources"] = ["clip"]
            profile["reference_type"] = "none"
            profile["inspection_pattern"] = "temporal_followup"
        profile["answer_type"] = "temporal_relation"
        profile["temporal_requirement"] = "ordering"
        profile["enable_cheap_stage"] = False
        return profile
    
    if analysis:
        return profile

    # ------------------------------------------------------------------
    # 2) Fallback heuristics only if analysis is missing/unhelpful
    # ------------------------------------------------------------------

    # text is mentioned, but answer requested is a visual property
    if has_text and has_attr:
        profile["evidence_sources"] = ["ocr", "clip"]
        profile["answer_type"] = "visual_attribute"
        profile["reference_type"] = "text_reference"
        profile["temporal_requirement"] = "none"
        profile["inspection_pattern"] = "static_localized"
        profile["enable_cheap_stage"] = False
        return profile

    # direct text answer
    if has_text and not has_attr:
        profile["evidence_sources"] = ["ocr"]
        profile["answer_type"] = "text_span"
        profile["reference_type"] = "direct_answer"
        profile["inspection_pattern"] = "ocr_first"
        profile["enable_cheap_stage"] = True
        if not has_order:
            profile["temporal_requirement"] = "none"
        return profile

    # speech as reference only should not force spoken answer for visual targets
    if has_speech and any(x in q for x in [
        "on the screen", "visible", "present", "not present",
        "which object", "what object", "wearing", "holding", "in his hand", "in her hand"
    ]):
        profile["evidence_sources"] = ["asr", "clip"]
        profile["answer_type"] = "visual_object"
        profile["reference_type"] = "speech_reference"
        profile["temporal_requirement"] = "none"
        profile["inspection_pattern"] = "speech_first"
        profile["enable_cheap_stage"] = False
        return profile

    # pure speech answer
    if has_speech:
        profile["evidence_sources"] = ["asr"]
        if has_order:
            profile["answer_type"] = "temporal_relation"
            profile["temporal_requirement"] = "ordering"
        else:
            profile["answer_type"] = "spoken_span"
            profile["temporal_requirement"] = "none"
        profile["reference_type"] = "direct_answer"
        profile["inspection_pattern"] = "speech_first"
        profile["enable_cheap_stage"] = True
        return profile

    # visual ordering
    if has_order:
        profile["evidence_sources"] = ["clip"]
        profile["answer_type"] = "temporal_relation"
        profile["reference_type"] = "none"
        profile["temporal_requirement"] = "ordering"
        profile["inspection_pattern"] = "temporal_followup"
        profile["enable_cheap_stage"] = False
        return profile

    # ordinary visual attribute/object
    if has_attr:
        profile["evidence_sources"] = ["clip"]
        profile["answer_type"] = "visual_attribute"
        profile["reference_type"] = "none"
        profile["temporal_requirement"] = "none"
        profile["inspection_pattern"] = "static_localized"
        profile["enable_cheap_stage"] = False
        return profile

    return profile