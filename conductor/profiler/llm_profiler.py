from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import requests

# ---------------------------------------------------------------------
# 1. LLM profiler prompt
# ---------------------------------------------------------------------

PROFILER_SYSTEM_PROMPT = """
You are a query profiler for a video question answering system.

Your job is to estimate how difficult a visual question is and what retrieval budget it may require.

Output STRICT JSON ONLY:

{
  "analysis": {
    "visual_detail_level": "low" | "medium" | "high",
    "object_scale": "large" | "medium" | "small",
    "needs_precise_local_view": true | false,
    "rationale": "<short explanation>"
  }
}

Guidelines:

visual_detail_level:
- low:
  large objects, simple actions, scene-level questions.
  Example: "Is the person cooking?"
  Example: "What sport is being played?"

- medium:
  requires identifying objects, interactions, or moderate detail.
  Example: "What tool is the person using?"
  Example: "What is the person carrying?"

- high:
  requires fine-grained visual evidence.
  Example: "What number is on the jersey?"
  Example: "What color is the label on the bottle?"
  Example: "Which button is pressed?"

object_scale:
- large: object occupies a substantial portion of the frame.
- medium: object is visible but not dominant.
- small: object is small, distant, or hard to see.

needs_precise_local_view:
- true when answering depends on a small region, fine detail,
  text-like visual evidence, or subtle object attributes.
- false otherwise.

Output JSON only.
"""


@dataclass(frozen=True)
class BudgetConfig:

    # Human-readable configuration name, e.g., "asr_to_vlm_medium".
    name: str
    # Higher values provide denser coverage but increase preprocessing cost.
    probe_fps: float
    # Number of coarse candidate windows retained before final inspection.
    # This controls the size of the candidate pool.
    probe_topk: int
    # Main one-shot budget: number of candidate windows inspected by the selected
    # expensive stage, such as VLM or OCR. This replaces iterative max_steps.
    action_topk: int
    # Length, in seconds, of each candidate window inspected.
    # Longer windows provide more context but increase processing cost.
    window_len_s: float
    # Coarse expected quality/robustness label. Currently useful for logging,
    # analysis, and future scheduler policies.
    quality_tier: str  # "risky" | "medium" | "safe"
    # Human-readable explanation for why this candidate exists.
    answer_tier: str = "heavy"
    cheap_answer_tier: str = "none"
    max_steps: int = 1
    rationale: str = ""



# ---------------------------------------------------------------------
# Default workflow budgets
# ---------------------------------------------------------------------

LOW_BUDGET = BudgetConfig(
    name="low_budget",
    probe_fps=0.5,
    probe_topk=4,
    action_topk=4,
    window_len_s=8.0,
    quality_tier="risky",
)

MEDIUM_BUDGET = BudgetConfig(
    name="medium_budget",
    probe_fps=1.0,
    probe_topk=8,
    action_topk=8,
    window_len_s=8.0,
    quality_tier="medium",
)

HIGH_BUDGET = BudgetConfig(
    name="high_budget",
    probe_fps=2.0,
    probe_topk=16,
    action_topk=16,
    window_len_s=8.0,
    quality_tier="safe",
)



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


@dataclass
class ProfilerResult:
    analysis: Dict[str, Any]
    candidate_configs: List[BudgetConfig]
    requested_config: BudgetConfig
    chosen_config: BudgetConfig
    execution_policy: Dict[str, Any]
    raw_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis,
            "candidate_configs": [asdict(c) for c in self.candidate_configs],
            "chosen_config": asdict(self.chosen_config),
            "execution_policy": self.execution_policy,
            "raw_json": self.raw_json,
            "requested_config": asdict(self.requested_config),
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


    print(url)
    print(json.dumps(payload, indent=2))
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
# 6. Resource-aware config selection
# ---------------------------------------------------------------------

def estimate_config_cost(config: BudgetConfig) -> float:
    return (
        config.probe_fps
        * config.probe_topk
        * config.action_topk
    )


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
    candidates: List[BudgetConfig],
    resource_state: ResourceState,
) -> BudgetConfig:
    """
    METIS-style best-fit selector.

    METIS chooses the richest quality-preserving RAG config that fits GPU memory.
    VIMIO chooses the richest quality-preserving workflow config that fits the
    current encoder/VLM resource state.
    """
    if not candidates:
        raise ValueError("No candidate workflow configurations provided.")

    budget = _cost_budget_for_resource_state(resource_state)

    fitting: List[Tuple[float, BudgetConfig]] = []
    for cfg in candidates:
        cost = estimate_config_cost(cfg)
        if cost <= budget:
            fitting.append((cost, cfg))

    if fitting:
        # Richest config that still fits.
        return max(fitting, key=lambda x: x[0])[1]

    # If nothing fits, fallback to cheapest candidate.
    return min(candidates, key=estimate_config_cost)


def budget_to_policy(config: BudgetConfig):

    return {
        "probe_fps": config.probe_fps,
        "probe_topk": config.probe_topk,
        "action_topk": config.action_topk,
        "window_len_s": config.window_len_s,
    }
# ---------------------------------------------------------------------
# 8. Main entry point: METIS-style VIMIO profiler + selector
# ---------------------------------------------------------------------

def profile_query_llm(
    query: str,
    *,
    duration_s,
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

        query -> QueryProfile -> candidate BudgetConfig
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
    print("\n===== RAW PROFILER ANALYSIS =====")
    print(json.dumps(analysis, indent=2))
    # Temporary: current orchestrator is visual-only.

    candidate_configs = analysis_to_candidate_configs(analysis)

    requested_config = choose_budget_from_profile(
        analysis,
        duration_s,
    )

    chosen_config = adapt_budget_for_resources(
        requested_config,
        resource_state,
    )

    print("\n===== CANDIDATE WORKFLOW CONFIGS =====")
    print(json.dumps([asdict(c) for c in candidate_configs], indent=2))

    print("\n===== REQUESTED WORKFLOW CONFIG =====")
    print(json.dumps(asdict(requested_config), indent=2))

    print("\n===== RESOURCE STATE =====")
    print(json.dumps(asdict(resource_state), indent=2))

    print("\n===== CHOSEN WORKFLOW CONFIG =====")
    print(json.dumps(asdict(chosen_config), indent=2))

    policy = budget_to_policy(chosen_config)

    print("\n===== COMPILED EXECUTION POLICY =====")
    print(json.dumps(policy, indent=2))

    print("\n===== COMPILED EXECUTION POLICY =====")
    print(json.dumps(policy, indent=2))

    return ProfilerResult(
        analysis=analysis,
        candidate_configs=candidate_configs,
        requested_config=requested_config,
        chosen_config=chosen_config,
        execution_policy=policy,
        raw_json=raw_json,
    )



def choose_scan_rate(requirement, duration_s):

    if duration_s < 60:

        table = {
            "local": 2.0,
            "medium": 1.0,
            "global": 0.5,
        }

    elif duration_s < 300:

        table = {
            "local": 1.0,
            "medium": 0.5,
            "global": 0.125,
        }

    else:

        table = {
            "local": 0.5,
            "medium": 0.125,
            "global": 0.03125,
        }

    return table[requirement]

def choose_budget_from_profile(
    analysis,
    duration_s,
):

    probe_fps = choose_scan_rate(
        analysis["temporal_requirement"],
        duration_s,
    )

    topk = choose_topk(
        analysis["candidate_requirement"]
    )

    window = choose_window(
        analysis["context_requirement"]
    )

    budget = choose_vlm_budget(
        analysis["precision_requirement"]
    )

    return BudgetConfig(
        name="dynamic",
        probe_fps=probe_fps,
        probe_topk=topk,
        action_topk=topk,
        window_len_s=window,
        vlm_budget=budget,
        quality_tier="dynamic",
    )


def profile_query_llm_legacy(
    query: str,
    *,
    duration_s,
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



def analysis_to_candidate_configs(
    analysis: Dict[str, Any],
) -> List[BudgetConfig]:

    analysis = analysis or {}

    detail = analysis.get("visual_detail_level", "medium")
    precise = analysis.get("needs_precise_local_view", False)
    scale = analysis.get("object_scale", "medium")

    if detail == "low" and not precise:
        return [
            BudgetConfig(
                name="low_budget",
                probe_fps=0.5,
                probe_topk=4,
                action_topk=4,
                window_len_s=8.0,
                quality_tier="risky",
            ),
            BudgetConfig(
                name="medium_budget",
                probe_fps=1.0,
                probe_topk=8,
                action_topk=8,
                window_len_s=8.0,
                quality_tier="medium",
            ),
        ]

    if detail == "medium":
        return [
            BudgetConfig(
                name="medium_budget",
                probe_fps=1.0,
                probe_topk=8,
                action_topk=8,
                window_len_s=8.0,
                quality_tier="medium",
            ),
            BudgetConfig(
                name="high_budget",
                probe_fps=2.0,
                probe_topk=12,
                action_topk=12,
                window_len_s=8.0,
                quality_tier="safe",
            ),
        ]

    return [
        BudgetConfig(
            name="high_budget",
            probe_fps=2.0,
            probe_topk=16,
            action_topk=16,
            window_len_s=8.0,
            quality_tier="safe",
        )
    ]





def adapt_budget_for_resources(
    requested: BudgetConfig,
    resource_state: ResourceState,
) -> BudgetConfig:

    if resource_state.load_level == "low":
        return requested

    if resource_state.load_level == "medium":

        if requested.name == "high_budget":
            return MEDIUM_BUDGET

        return requested

    if resource_state.load_level == "high":

        if requested.name == "high_budget":
            return MEDIUM_BUDGET

        if requested.name == "medium_budget":
            return LOW_BUDGET

        return requested