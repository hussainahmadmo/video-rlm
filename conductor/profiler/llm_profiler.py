from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import requests

from .policy_schema import ExecutionPolicy
from .query_profile_schema import QueryProfile


# ---------------------------------------------------------------------
# 1. LLM profiler prompt
# ---------------------------------------------------------------------

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
    "inspection_pattern": "vlm_only" | "asr_only" | "ocr_only" | "vlm_anchor_then_asr" | "asr_anchor_then_vlm" | "vlm_anchor_then_ocr" | "ocr_anchor_then_vlm" | "asr_anchor_then_ocr",
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
- Output JSON only.
"""


# ---------------------------------------------------------------------
# 2. Data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceState:
    """
    Runtime serving state used by the resource-aware selector.

    You can later replace this with real measurements from your scheduler,
    GPU monitor, or VLM/encoder queue.
    """
    free_gpu_mem_gb: float = 999.0
    encoder_queue_len: int = 0
    vlm_queue_len: int = 0
    load_level: str = "low"  # "low" | "medium" | "high"


@dataclass(frozen=True)
class WorkflowConfig:
    """
    A candidate multimodal workflow configuration.

    This is the VIMIO equivalent of a METIS candidate RAG configuration.
    """
    name: str

    # High-level workflow structure
    evidence_sources: Tuple[str, ...]
    inspection_pattern: str

    # Execution budget
    probe_fps: float
    probe_topk: int
    action_topk: int
    window_len_s: float
    max_steps: int

    # Used for ranking candidates inside the pruned space
    quality_tier: str  # "risky" | "medium" | "safe"
    rationale: str = ""


@dataclass
class ProfilerResult:
    analysis: Dict[str, Any]
    query_profile: QueryProfile

    # METIS-style outputs
    candidate_configs: List[WorkflowConfig]
    chosen_config: WorkflowConfig

    execution_policy: ExecutionPolicy

    # Debugging / audit fields
    raw_profile: Dict[str, Any]
    raw_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis,
            "query_profile": asdict(self.query_profile),
            "candidate_configs": [asdict(c) for c in self.candidate_configs],
            "chosen_config": asdict(self.chosen_config),
            "execution_policy": asdict(self.execution_policy),
            "raw_profile": self.raw_profile,
            "raw_json": self.raw_json,
        }


# ---------------------------------------------------------------------
# 3. JSON parsing and profile coercion
# ---------------------------------------------------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON from the profiler output.
    Allows for small mistakes where the model wraps JSON in extra text.
    """
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
    """
    Convert a raw profiler dict into your QueryProfile schema.
    """
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


def _patch_query_profile_with_guardrails(
    query: str,
    d: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Guardrails for profiler output.

    Goal:
    - Do not use temporal categories.
    - Do not output clip.
    - Use VLM for expensive visual inspection.
    - Use ASR/OCR as cheap anchors when they can reduce VLM work.
    """
    analysis = d.get("analysis", {}) if isinstance(d.get("analysis"), dict) else {}
    profile = d.get("profile", d) if isinstance(d.get("profile", d), dict) else dict(d)
    profile = dict(profile)

    needs_vlm = bool(analysis.get("needs_vlm", False))
    needs_asr = bool(analysis.get("needs_asr", False))
    needs_ocr = bool(analysis.get("needs_ocr", False))
    target = str(analysis.get("primary_question_target", "")).strip()

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
        profile["evidence_sources"] = ["vlm", "asr"]
        profile["answer_type"] = "spoken_span"
        profile["reference_type"] = "visual_reference"
        profile["enable_cheap_stage"] = False

    elif pattern == "asr_anchor_then_vlm":
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

    profile.pop("temporal_requirement", None)

    return profile


# ---------------------------------------------------------------------
# 4. LLM query profiling
# ---------------------------------------------------------------------

def _call_profiler_llm(
    query: str,
    *,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: float,
    api_key: str | None,
) -> Dict[str, Any]:
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

    headers = {
        "Authorization": f"Bearer {api_key or 'EMPTY'}",
    }

    r = requests.post(
        url,
        json=payload,
        timeout=timeout_s,
        headers=headers,
    )
    r.raise_for_status()

    j = r.json()
    text = j["choices"][0]["message"]["content"]

    print("\n===== RAW PROFILER OUTPUT =====")
    print(text)
    print("===== END RAW PROFILER OUTPUT =====\n")

    return _extract_json(text)


# ---------------------------------------------------------------------
# 5. METIS-style candidate configuration pruning
# ---------------------------------------------------------------------

def profile_to_candidate_configs(
    qp: QueryProfile,
    analysis: Dict[str, Any] | None = None,
) -> List[WorkflowConfig]:
    """
    Rule-based mapping from query profile to a pruned set of candidate workflow
    configurations.

    This is the VIMIO equivalent of METIS mapping:
        query profile -> pruned RAG configuration space

    The profiler does not choose exact budgets. It narrows the space.
    """
    analysis = analysis or {}
    pattern = qp.inspection_pattern

    visual_detail = str(analysis.get("visual_detail_level", "medium"))
    needs_precise = bool(analysis.get("needs_precise_local_view", False))

    # ---------------------------------------------------------------
    # ASR-only: no visual work should be created.
    # ---------------------------------------------------------------
    if pattern == "asr_only":
        return [
            WorkflowConfig(
                name="asr_only_cheap",
                evidence_sources=("asr",),
                inspection_pattern="asr_only",
                probe_fps=0.0,
                probe_topk=16,
                action_topk=0,
                window_len_s=60.0,
                max_steps=2,
                answer_tier="cheap",
                cheap_answer_tier="cheap",
                quality_tier="safe",
                rationale="Speech-only query; avoid visual encoder/VLM work.",
            )
        ]

    # ---------------------------------------------------------------
    # OCR-only: cheap text extraction; no VLM action by default.
    # ---------------------------------------------------------------
    if pattern == "ocr_only":
        return [
            WorkflowConfig(
                name="ocr_only_small",
                evidence_sources=("ocr",),
                inspection_pattern="ocr_only",
                probe_fps=0.5,
                probe_topk=8,
                action_topk=0,
                window_len_s=5.0,
                max_steps=4,
                answer_tier="cheap",
                cheap_answer_tier="cheap",
                quality_tier="medium",
                rationale="OCR-only query with small visual sampling budget.",
            ),
            WorkflowConfig(
                name="ocr_only_large",
                evidence_sources=("ocr",),
                inspection_pattern="ocr_only",
                probe_fps=1.0,
                probe_topk=16,
                action_topk=0,
                window_len_s=5.0,
                max_steps=6,
                answer_tier="cheap",
                cheap_answer_tier="cheap",
                quality_tier="safe",
                rationale="OCR-only query with larger sampling budget.",
            ),
        ]

    # ---------------------------------------------------------------
    # VLM-only: visual evidence is directly needed.
    # ---------------------------------------------------------------
    if pattern == "vlm_only":
        configs = [
            WorkflowConfig(
                name="vlm_small",
                evidence_sources=("vlm",),
                inspection_pattern="vlm_only",
                probe_fps=0.5,
                probe_topk=8,
                action_topk=4,
                window_len_s=5.0,
                max_steps=6,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="medium",
                rationale="Small VLM budget for visual query.",
            ),
            WorkflowConfig(
                name="vlm_medium",
                evidence_sources=("vlm",),
                inspection_pattern="vlm_only",
                probe_fps=0.5,
                probe_topk=12,
                action_topk=6,
                window_len_s=5.0,
                max_steps=8,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="safe",
                rationale="Medium VLM budget for visual query.",
            ),
        ]

        if visual_detail == "high" or needs_precise:
            configs.append(
                WorkflowConfig(
                    name="vlm_large_precise",
                    evidence_sources=("vlm",),
                    inspection_pattern="vlm_only",
                    probe_fps=1.0,
                    probe_topk=16,
                    action_topk=8,
                    window_len_s=8.0,
                    max_steps=10,
                    answer_tier="heavy",
                    cheap_answer_tier="none",
                    quality_tier="safe",
                    rationale="Larger VLM budget for fine-grained visual detail.",
                )
            )

        return configs

    # ---------------------------------------------------------------
    # ASR anchors VLM: cheap ASR scans broadly, VLM only selected windows.
    # ---------------------------------------------------------------
    if pattern == "asr_anchor_then_vlm":
        return [
            WorkflowConfig(
                name="asr_to_vlm_small",
                evidence_sources=("asr", "vlm"),
                inspection_pattern="asr_anchor_then_vlm",
                probe_fps=0.5,
                probe_topk=8,
                action_topk=2,
                window_len_s=5.0,
                max_steps=5,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="medium",
                rationale="ASR localizes evidence; run VLM on a small number of windows.",
            ),
            WorkflowConfig(
                name="asr_to_vlm_medium",
                evidence_sources=("asr", "vlm"),
                inspection_pattern="asr_anchor_then_vlm",
                probe_fps=0.5,
                probe_topk=16,
                action_topk=4,
                window_len_s=5.0,
                max_steps=8,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="safe",
                rationale="ASR localizes evidence; run VLM on a moderate number of windows.",
            ),
            WorkflowConfig(
                name="asr_to_vlm_large",
                evidence_sources=("asr", "vlm"),
                inspection_pattern="asr_anchor_then_vlm",
                probe_fps=1.0,
                probe_topk=24,
                action_topk=8,
                window_len_s=8.0,
                max_steps=10,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="safe",
                rationale="Larger ASR-anchored VLM workflow for harder visual evidence.",
            ),
        ]

    # ---------------------------------------------------------------
    # VLM anchors ASR: visual event/window first, then speech in that window.
    # ---------------------------------------------------------------
    if pattern == "vlm_anchor_then_asr":
        return [
            WorkflowConfig(
                name="vlm_to_asr_small",
                evidence_sources=("vlm", "asr"),
                inspection_pattern="vlm_anchor_then_asr",
                probe_fps=0.5,
                probe_topk=8,
                action_topk=2,
                window_len_s=5.0,
                max_steps=5,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="medium",
                rationale="VLM finds visual anchor; ASR runs on selected windows.",
            ),
            WorkflowConfig(
                name="vlm_to_asr_medium",
                evidence_sources=("vlm", "asr"),
                inspection_pattern="vlm_anchor_then_asr",
                probe_fps=0.5,
                probe_topk=12,
                action_topk=4,
                window_len_s=5.0,
                max_steps=8,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="safe",
                rationale="VLM finds visual anchor with moderate budget.",
            ),
        ]

    # ---------------------------------------------------------------
    # OCR anchors VLM: text identifies window/region, final answer visual.
    # ---------------------------------------------------------------
    if pattern == "ocr_anchor_then_vlm":
        return [
            WorkflowConfig(
                name="ocr_to_vlm_small",
                evidence_sources=("ocr", "vlm"),
                inspection_pattern="ocr_anchor_then_vlm",
                probe_fps=0.5,
                probe_topk=8,
                action_topk=2,
                window_len_s=5.0,
                max_steps=5,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="medium",
                rationale="OCR anchors the visual window; small VLM budget.",
            ),
            WorkflowConfig(
                name="ocr_to_vlm_medium",
                evidence_sources=("ocr", "vlm"),
                inspection_pattern="ocr_anchor_then_vlm",
                probe_fps=0.5,
                probe_topk=16,
                action_topk=4,
                window_len_s=5.0,
                max_steps=8,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="safe",
                rationale="OCR anchors the visual window; medium VLM budget.",
            ),
        ]

    # ---------------------------------------------------------------
    # VLM anchors OCR: visual region/window first, then read text.
    # ---------------------------------------------------------------
    if pattern == "vlm_anchor_then_ocr":
        return [
            WorkflowConfig(
                name="vlm_to_ocr_small",
                evidence_sources=("vlm", "ocr"),
                inspection_pattern="vlm_anchor_then_ocr",
                probe_fps=0.5,
                probe_topk=8,
                action_topk=2,
                window_len_s=5.0,
                max_steps=5,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="medium",
                rationale="VLM finds visual region; OCR reads text.",
            ),
            WorkflowConfig(
                name="vlm_to_ocr_medium",
                evidence_sources=("vlm", "ocr"),
                inspection_pattern="vlm_anchor_then_ocr",
                probe_fps=0.5,
                probe_topk=12,
                action_topk=4,
                window_len_s=5.0,
                max_steps=8,
                answer_tier="heavy",
                cheap_answer_tier="none",
                quality_tier="safe",
                rationale="VLM finds visual region; OCR reads text with larger budget.",
            ),
        ]

    # Fallback
    return [
        WorkflowConfig(
            name="default_vlm_medium",
            evidence_sources=("vlm",),
            inspection_pattern="vlm_only",
            probe_fps=0.5,
            probe_topk=8,
            action_topk=4,
            window_len_s=5.0,
            max_steps=8,
            answer_tier="heavy",
            cheap_answer_tier="none",
            quality_tier="medium",
            rationale="Fallback visual workflow.",
        )
    ]


# ---------------------------------------------------------------------
# 6. Resource-aware config selection
# ---------------------------------------------------------------------

def estimate_config_cost(config: WorkflowConfig) -> float:
    """
    Simple cost proxy.

    Replace this later with real measurements:
    - estimated visual encoder time
    - estimated VLM latency
    - GPU memory usage
    - current queueing delay
    """
    # ASR/OCR-only configs are cheap.
    if "vlm" not in config.evidence_sources:
        return 1.0 + 0.05 * config.probe_topk

    # Visual cost proxy.
    visual_cost = (
        config.action_topk
        * max(config.window_len_s, 1.0)
        * max(config.probe_fps, 0.1)
    )

    if config.answer_tier == "heavy":
        visual_cost *= 2.0

    # More max steps means more possible iterative inspection.
    visual_cost *= 1.0 + 0.05 * config.max_steps

    return visual_cost


def _cost_budget_for_resource_state(resource_state: ResourceState) -> float:
    """
    Convert current system load into a rough cost budget.
    """
    if resource_state.load_level == "high":
        return 20.0
    if resource_state.load_level == "medium":
        return 50.0
    return 100.0


def choose_config_for_current_resources(
    candidates: List[WorkflowConfig],
    resource_state: ResourceState,
) -> WorkflowConfig:
    """
    METIS-style best-fit selector.

    METIS chooses the richest quality-preserving RAG config that fits GPU memory.
    VIMIO chooses the richest quality-preserving workflow config that fits the
    current encoder/VLM resource state.
    """
    if not candidates:
        raise ValueError("No candidate workflow configurations provided.")

    budget = _cost_budget_for_resource_state(resource_state)

    fitting: List[Tuple[float, WorkflowConfig]] = []
    for cfg in candidates:
        cost = estimate_config_cost(cfg)
        if cost <= budget:
            fitting.append((cost, cfg))

    if fitting:
        # Richest config that still fits.
        return max(fitting, key=lambda x: x[0])[1]

    # If nothing fits, fallback to cheapest candidate.
    return min(candidates, key=estimate_config_cost)


# ---------------------------------------------------------------------
# 7. Compile chosen WorkflowConfig into your existing ExecutionPolicy
# ---------------------------------------------------------------------

def workflow_config_to_execution_policy(
    config: WorkflowConfig,
    qp: QueryProfile,
) -> ExecutionPolicy:
    preferred_tools = tuple(
        "vlm" if x == "clip" else x
        for x in config.evidence_sources
    )

    return ExecutionPolicy(
        mode="attribute",
        anchor_policy="best_score",
        coverage_target=1,
        require_temporal_pair=False,

        preferred_tools=preferred_tools,

        probe_tier="cheap",
        caption_tier="none",
        ocr_tier="cheap" if "ocr" in preferred_tools else "none",
        asr_tier="cheap" if "asr" in preferred_tools else "none",

        enable_cheap_stage=config.inspection_pattern in ("asr_only", "ocr_only"),
        cheap_answer_tier=config.cheap_answer_tier,
        text_answer_min_conf=0.75,

        answer_tier=config.answer_tier,
        fallback_answer_tier="none",

        probe_fps=config.probe_fps,
        probe_seg_len_s=60,
        probe_topk=config.probe_topk,

        window_len_s=config.window_len_s,
        strides=(0.5,),
        action_topk=config.action_topk,

        max_steps=config.max_steps,
        eps_marginal_gain=0.01,
        escalate_if_low_confidence=True,
        min_retrieval_conf=0.15,
        min_answer_conf=0.35,

        confidence=qp.confidence,
        rationale=(
            f"{qp.rationale} | chosen_config={config.name}; "
            f"config_rationale={config.rationale}"
        ),
    )


# ---------------------------------------------------------------------
# 8. Main entry point: METIS-style VIMIO profiler + selector
# ---------------------------------------------------------------------

def profile_query_llm(
    query: str,
    *,
    base_url: str = "http://localhost:8000/v1",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_s: float = 30.0,
    api_key: str | None = None,
    resource_state: ResourceState | None = None,
) -> ProfilerResult:
    """
    Main VIMIO profiling API.

    This replaces the old direct path:

        query -> QueryProfile -> ExecutionPolicy

    with the METIS-style path:

        query -> QueryProfile -> candidate WorkflowConfigs
              -> resource-aware selection -> ExecutionPolicy
    """
    if resource_state is None:
        resource_state = ResourceState()

    raw_json = _call_profiler_llm(
        query=query,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        api_key=api_key,
    )

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

    candidate_configs = profile_to_candidate_configs(qp, analysis=analysis)

    print("\n===== CANDIDATE WORKFLOW CONFIGS =====")
    print(json.dumps([asdict(c) for c in candidate_configs], indent=2))

    chosen_config = choose_config_for_current_resources(
        candidates=candidate_configs,
        resource_state=resource_state,
    )

    print("\n===== CHOSEN WORKFLOW CONFIG =====")
    print(json.dumps(asdict(chosen_config), indent=2))

    policy = workflow_config_to_execution_policy(chosen_config, qp)

    print("\n===== COMPILED EXECUTION POLICY =====")
    print(json.dumps(asdict(policy), indent=2))

    return ProfilerResult(
        analysis=analysis,
        query_profile=qp,
        candidate_configs=candidate_configs,
        chosen_config=chosen_config,
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