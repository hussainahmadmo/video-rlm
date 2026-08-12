from __future__ import annotations

import json
import math
import random
import re

from dataclasses import (
    asdict,
    dataclass,
    replace,
)
from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Tuple,
)
FIRST_OCCURRENCE_QUERY_TERMS = (
    "first person",
    "first object",
    "first item",
    "first time",
    "first event",
    "first action",
    "first thing",
)

LAST_OCCURRENCE_QUERY_TERMS = (
    "last person",
    "last object",
    "last item",
    "last time",
    "last event",
    "last action",
    "last thing",
)

import requests

@dataclass
class AdaptiveProfilerResult:
    source: str
    router_decision: RouterDecision
    chosen_config: BudgetConfig
    execution_policy: Dict[str, Any]
    llm_result: ProfilerResult | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "router_decision": asdict(
                self.router_decision
            ),
            "chosen_config": asdict(
                self.chosen_config
            ),
            "execution_policy": (
                self.execution_policy
            ),
            "llm_result": (
                self.llm_result.to_dict()
                if self.llm_result is not None
                else None
            ),
        }
    
# ============================================================================
# 1. Profiler prompt
# ============================================================================

PROFILER_SYSTEM_PROMPT = """
You are a semantic execution profiler for a long-video question-answering
system.

Your job has TWO levels:

LEVEL 1:
Determine whether the query should use a fixed one-pass execution plan
or a bounded agentic evidence-acquisition plan.

LEVEL 2:
Describe the structure of video evidence needed to answer it.

You are NOT choosing exact frame rates, top-k values, or GPU budgets.
Those are compiled later.

Return STRICT JSON ONLY.
Do not use markdown.
Do not output explanatory text outside JSON.

Output schema:

{
  "analysis": {
    "execution_mode":
      "oneshot" |
      "agentic",

    "evidence_requirement":
      "local" |
      "global" |
      "count" |
      "temporal",

    "temporal_relation":
      "none" |
      "before" |
      "after" |
      "beginning" |
      "end" |
      "ordering",

    "reasoning_type":
      "local_event" |
      "fine_detail" |
      "counting" |
      "repetition" |
      "temporal_order" |
      "causal" |
      "global_summary" |
      "state_change" |
      "multi_hop" |
      "ambiguous",

    "answer_type":
      "multiple_choice" |
      "short_text" |
      "count" |
      "timestamp" |
      "yes_no" |
      "summary",

    "coverage_requirement":
      "targeted" |
      "multi_region" |
      "full_timeline",

    "selection_mode":
      "top_k" |
      "all_positive" |
      "all_positive_and_uncertain" |
      "uniform" |
      "beginning_end" |
      "multi_event",

    "temporal_requirement":
      "local" |
      "medium" |
      "global",

    "temporal_operation":
      "none" |
      "before_after" |
      "ordering" |
      "duration" |
      "frequency" |
      "state_transition" |
      "cross_segment_comparison",

    "candidate_requirement":
      "few" |
      "medium" |
      "many",

    "context_requirement":
      "short" |
      "medium" |
      "long",

    "precision_requirement":
      "low" |
      "medium" |
      "high",

    "aggregation_type":
      "none" |
      "occurrences" |
      "distinct_entities" |
      "simultaneous_max" |
      "segments" |
      "duration",

    "identity_requirement":
      "none" |
      "object_tracking" |
      "person_tracking" |
      "cross_window_reidentification",

    "spatial_strategy":
      "full_frame" |
      "object_crop" |
      "multi_crop" |

    "required_modalities": [
      "visual"
    ],

    "event_density":
      "sparse" |
      "medium" |
      "dense" |
      "unknown",

    "ambiguity":
      "low" |
      "medium" |
      "high",

    "profile_confidence": 0.0,

    "miss_risk":
      "low" |
      "medium" |
      "high",

    "answer_sensitivity":
      "approximate" |
      "exact",

    "fallback_requirement":
      "none" |
      "expand_coverage" |
      "full_high",

    "rationale":
      "<one concise explanation>"
  }
}

EXECUTION-MODE RULES:

ONESHOT means the query can be answered using one fixed evidence-acquisition
program without an open-ended search loop.

Examples:

- locate one object/action/person once and inspect that scene
- inspect the beginning directly
- inspect the end directly
- use a fixed uniform sample when no adaptive search is needed

AGENTIC means the query inherently requires evidence acquisition that depends
on previously observed evidence.

Typical examples:

- count distinct events across the whole video
- find an anchor and inspect what happened before or after it
- reconstruct a workflow across separated parts of a video
- verify multiple temporal stages
- resolve ambiguity by refining an already localized interval

IMPORTANT:

A LOCAL query should usually NOT become an open-ended search agent.

If one semantic retrieval pass can locate the scene, classify it as:

execution_mode = "oneshot"
evidence_requirement = "local"

GLOBAL workflow/summary/sequence questions should use:

execution_mode = "agentic"
evidence_requirement = "global"

but their agentic behavior should remain inside chronological global
coverage. They should not repeatedly perform unrelated semantic retrieval.

COUNTING questions should use:

execution_mode = "agentic"
evidence_requirement = "count"

TEMPORAL before/after questions should use:

execution_mode = "agentic"
evidence_requirement = "temporal"

Beginning/end questions can usually use fixed direct boundary inspection:

execution_mode = "oneshot"
evidence_requirement = "temporal"

TEMPORAL RELATION:

Use:
- "before" for what happened before an anchor
- "after" for what happened after an anchor
- "beginning" for first/start/opening questions
- "end" for last/final/ending questions
- "ordering" when multiple separated events must be ordered
- "none" otherwise

OTHER RULES:

1. Counting and repetition require full-timeline coverage.

2. Counting must not use top-k-only selection.

3. Exact occurrence counting:
   aggregation_type = "occurrences"
   temporal_operation = "frequency"

4. Counting distinct people or objects:
   aggregation_type = "distinct_entities"
   identity_requirement = "cross_window_reidentification"

5. Maximum simultaneously visible entities:
   aggregation_type = "simultaneous_max"

6. Workflow, summary, sequence, and major-stage questions require chronological
   global evidence. Do NOT convert these into repeated local searches for
   incidental objects or tools.

7. Temporal-order and multi-hop questions require multiple separated regions.

8. Fine-detail questions involving pointing, text, labels, clothing,
   identity, small objects, gaze, or precise interactions require high
   precision.

9. State-change questions generally require beginning/end or separated states.

10. This experiment is VISUAL ONLY.
    required_modalities must always be ["visual"].
    Do not request audio, OCR, subtitles, or metadata.

11. profile_confidence must be between 0.0 and 1.0.

12. Answer choices may help determine what evidence distinguishes the answers,
    but answer-choice text is not itself visual evidence.

Return JSON only.
"""
    
# ============================================================================
# 2. Allowed values
# ============================================================================

ALLOWED_VALUES: Dict[str, set[str]] = {
    "execution_mode": {
        "oneshot",
        "agentic",
    },

    "evidence_requirement": {
        "local",
        "global",
        "count",
        "temporal",
    },

    "temporal_relation": {
        "none",
        "before",
        "after",
        "beginning",
        "end",
        "ordering",
    },

    "reasoning_type": {
        "local_event",
        "fine_detail",
        "counting",
        "repetition",
        "temporal_order",
        "causal",
        "global_summary",
        "state_change",
        "multi_hop",
        "ambiguous",
    },

    "answer_type": {
        "multiple_choice",
        "short_text",
        "count",
        "timestamp",
        "yes_no",
        "summary",
    },

    "coverage_requirement": {
        "targeted",
        "multi_region",
        "full_timeline",
    },

    "selection_mode": {
        "top_k",
        "all_positive",
        "all_positive_and_uncertain",
        "uniform",
        "beginning_end",
        "multi_event",
    },

    "temporal_requirement": {
        "local",
        "medium",
        "global",
    },

    "temporal_operation": {
        "none",
        "before_after",
        "ordering",
        "duration",
        "frequency",
        "state_transition",
        "cross_segment_comparison",
    },

    "candidate_requirement": {
        "few",
        "medium",
        "many",
    },

    "context_requirement": {
        "short",
        "medium",
        "long",
    },

    "precision_requirement": {
        "low",
        "medium",
        "high",
    },

    "aggregation_type": {
        "none",
        "occurrences",
        "distinct_entities",
        "simultaneous_max",
        "segments",
        "duration",
    },

    "identity_requirement": {
        "none",
        "object_tracking",
        "person_tracking",
        "cross_window_reidentification",
    },

    "spatial_strategy": {
        "full_frame",
        "object_crop",
        "multi_crop",
    },

    "event_density": {
        "sparse",
        "medium",
        "dense",
        "unknown",
    },

    "ambiguity": {
        "low",
        "medium",
        "high",
    },

    "miss_risk": {
        "low",
        "medium",
        "high",
    },

    "answer_sensitivity": {
        "approximate",
        "exact",
    },

    "fallback_requirement": {
        "none",
        "expand_coverage",
        "full_high",
    },
}


ALLOWED_MODALITIES = {
    "visual",
}


# ============================================================================
# 3. Data structures
# ============================================================================

@dataclass(frozen=True)
class ResourceState:
    free_gpu_mem_gb: float = 999.0
    encoder_queue_len: int = 0
    vlm_queue_len: int = 0
    load_level: str = "low"


@dataclass(frozen=True)
class RouterDecision:
    policy_name: str

    execution_mode: str
    evidence_requirement: str
    temporal_relation: str

    confidence: float
    reason: str
    out_of_distribution: bool = False


@dataclass(frozen=True)
class CatalogPolicy:
    name: str

    probe_fps: float
    probe_topk: int
    action_topk: int

    window_len_s: float
    vlm_budget: int

    expand_neighbors: bool = False
    preserve_order: bool = False
    include_uniform_anchors: bool = False


@dataclass(frozen=True)
class BudgetConfig:
    name: str

    # ------------------------------------------------------------
    # Macro execution decision.
    # ------------------------------------------------------------

    execution_mode: str
    evidence_requirement: str
    temporal_relation: str

    # ------------------------------------------------------------
    # Semantic execution strategy.
    # ------------------------------------------------------------

    reasoning_type: str
    answer_type: str

    coverage_mode: str
    selection_mode: str

    temporal_operation: str
    aggregation_type: str

    identity_requirement: str
    spatial_strategy: str

    required_modalities: Tuple[str, ...]

    # ------------------------------------------------------------
    # Low-fidelity probe.
    # ------------------------------------------------------------

    probe_fps: float
    chunk_len_s: float
    frames_per_chunk: int

    # ------------------------------------------------------------
    # Candidate selection.
    # ------------------------------------------------------------

    probe_topk: int | None
    action_topk: int | None

    candidate_threshold: float
    uncertainty_threshold: float

    # ------------------------------------------------------------
    # High-fidelity refinement.
    # ------------------------------------------------------------

    window_len_s: float
    high_frames_per_window: int
    high_spatial_tier: str

    # ------------------------------------------------------------
    # Event consolidation.
    # ------------------------------------------------------------

    merge_gap_s: float

    # ------------------------------------------------------------
    # VLM answering budget.
    # ------------------------------------------------------------

    vlm_budget: int
    quality_tier: str

    # ------------------------------------------------------------
    # Correctness / fallback.
    # ------------------------------------------------------------

    fallback_mode: str
    min_temporal_coverage: float

    profile_confidence: float
    miss_risk: str
    answer_sensitivity: str

    # ------------------------------------------------------------
    # Execution hints.
    # ------------------------------------------------------------

    evidence_type: str = "generic"

    expand_neighbors: bool = False
    preserve_order: bool = False
    include_uniform_anchors: bool = False

    # ------------------------------------------------------------
    # Agent bounds.
    # ------------------------------------------------------------

    max_steps: int = 1
    max_local_searches: int = 1
    max_global_scans: int = 0
    max_count_scans: int = 0
    max_temporal_searches: int = 0
    max_density_refinements: int = 0
    max_contrastive_checks: int = 0

    # ------------------------------------------------------------
    # Compatibility.
    # ------------------------------------------------------------

    answer_tier: str = "heavy"
    cheap_answer_tier: str = "none"

    rationale: str = ""


@dataclass
class ProfilerResult:
    analysis: Dict[str, Any]

    candidate_configs: List[
        BudgetConfig
    ]

    requested_config: BudgetConfig
    chosen_config: BudgetConfig

    execution_policy: Dict[str, Any]

    raw_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis":
                self.analysis,

            "candidate_configs": [
                asdict(config)
                for config
                in self.candidate_configs
            ],

            "requested_config":
                asdict(
                    self.requested_config
                ),

            "chosen_config":
                asdict(
                    self.chosen_config
                ),

            "execution_policy":
                self.execution_policy,

            "raw_json":
                self.raw_json,
        }




# ============================================================================
# 4. Policy catalog
# ============================================================================

POLICY_CATALOG: Dict[
    str,
    CatalogPolicy,
] = {

    # ------------------------------------------------------------
    # Cheap/local fixed path.
    # ------------------------------------------------------------

    "cheap": CatalogPolicy(
        name="cheap",

        probe_fps=0.015625,

        probe_topk=4,
        action_topk=4,

        window_len_s=8.0,
        vlm_budget=8,
    ),

    # ------------------------------------------------------------
    # Slightly stronger local path.
    # ------------------------------------------------------------

    "detail": CatalogPolicy(
        name="detail",

        probe_fps=0.05,

        probe_topk=4,
        action_topk=4,

        window_len_s=8.0,
        vlm_budget=16,

        expand_neighbors=True,
    ),

    # ------------------------------------------------------------
    # Chronological/global path.
    # ------------------------------------------------------------

    "global": CatalogPolicy(
        name="global",

        probe_fps=0.015625,

        probe_topk=8,
        action_topk=8,

        window_len_s=16.0,
        vlm_budget=16,

        preserve_order=True,
        include_uniform_anchors=True,
    ),

    # ------------------------------------------------------------
    # Sequence path.
    # ------------------------------------------------------------

    "sequence": CatalogPolicy(
        name="sequence",

        probe_fps=0.015625,

        probe_topk=8,
        action_topk=8,

        window_len_s=16.0,
        vlm_budget=16,

        preserve_order=True,
        include_uniform_anchors=True,
    ),

    # ------------------------------------------------------------
    # Temporal anchor relation.
    # ------------------------------------------------------------

    "temporal": CatalogPolicy(
        name="temporal",

        probe_fps=0.05,

        probe_topk=4,
        action_topk=4,

        window_len_s=16.0,
        vlm_budget=16,

        preserve_order=True,
    ),

    # ------------------------------------------------------------
    # Counting.
    # ------------------------------------------------------------

    "counting": CatalogPolicy(
        name="counting",

        probe_fps=0.03125,

        probe_topk=8,
        action_topk=8,

        window_len_s=8.0,
        vlm_budget=32,

        preserve_order=True,
        include_uniform_anchors=True,
    ),

    # ------------------------------------------------------------
    # Very long video.
    # ------------------------------------------------------------

    "long_sparse": CatalogPolicy(
        name="long_sparse",

        probe_fps=0.00390625,

        probe_topk=8,
        action_topk=8,

        window_len_s=8.0,
        vlm_budget=32,

        preserve_order=True,
        include_uniform_anchors=True,
    ),
}


# ============================================================================
# 5. Defaults
# ============================================================================
DEFAULT_ANALYSIS: Dict[str, Any] = {
    # ------------------------------------------------------------
    # Macro execution decision
    # ------------------------------------------------------------
    "execution_mode": "oneshot",
    "evidence_requirement": "local",
    "temporal_relation": "none",

    # ------------------------------------------------------------
    # Semantic question type
    # ------------------------------------------------------------
    "reasoning_type": "ambiguous",
    "answer_type": "multiple_choice",

    # ------------------------------------------------------------
    # Evidence geometry
    # ------------------------------------------------------------
    "coverage_requirement": "targeted",
    "selection_mode": "top_k",
    "temporal_requirement": "local",
    "temporal_operation": "none",

    # ------------------------------------------------------------
    # Candidate/context requirements
    # ------------------------------------------------------------
    "candidate_requirement": "few",
    "context_requirement": "medium",

    # ------------------------------------------------------------
    # Visual precision
    # ------------------------------------------------------------
    "precision_requirement": "medium",
    "spatial_strategy": "full_frame",

    # ------------------------------------------------------------
    # Aggregation / tracking
    # ------------------------------------------------------------
    "aggregation_type": "none",
    "identity_requirement": "none",

    # ------------------------------------------------------------
    # Visual-only system
    # ------------------------------------------------------------
    "required_modalities": ["visual"],

    # ------------------------------------------------------------
    # Difficulty / uncertainty
    # ------------------------------------------------------------
    "event_density": "unknown",
    "ambiguity": "medium",
    "profile_confidence": 0.5,
    "miss_risk": "medium",

    # ------------------------------------------------------------
    # Correctness / fallback
    # ------------------------------------------------------------
    "answer_sensitivity": "exact",
    "fallback_requirement": "none",

    # ------------------------------------------------------------
    # Runtime semantic hint
    # ------------------------------------------------------------
    "evidence_type": "generic",

    # ------------------------------------------------------------
    # Human-readable profiler explanation
    # ------------------------------------------------------------
    "rationale": "",
}

# ============================================================================
# 6. Query text cues
# ============================================================================

COUNTING_QUERY_TERMS = (
    "how many",
    "number of times",
    "how often",
    "count",
    "counting",
    "total number",
)


GLOBAL_SUMMARY_QUERY_TERMS = (
    "summarize",
    "summary",
    "overall",
    "overall process",
    "overall workflow",
    "workflow",
    "key sequence",
    "sequence of events",
    "major stages",
    "main stages",
    "throughout the video",
    "throughout video",
    "from beginning to end",
    "from start to finish",
    "primary goal",
    "main goal",
    "core process",
)


SEQUENCE_QUERY_TERMS = (
    "sequence of scenes",
    "sequence of events",
    "what order",
    "which order",
    "order of events",
    "sequential order",
    "stages",
    "what happens next",
    "scene to scene",
)


BEGINNING_QUERY_TERMS = (
    "at the beginning",
    "at the start",
    "beginning of the video",
    "start of the video",
    "initially",
    "starts with",
    "starts by",
    "opening scene",
    "opening of the video",
)


END_QUERY_TERMS = (
    "at the end",
    "end of the video",
    "ending of the video",
    "final scene",
    "finally in the video",
    "ends with",
    "ends by",
)


BEFORE_QUERY_TERMS = (
    "before",
    "prior to",
    "preceding",
)


AFTER_QUERY_TERMS = (
    "what happens after",
    "what happened after",
    "what does he do after",
    "what does she do after",
    "what occurs after",
    "after the",
    "after he",
    "after she",
    "after they",
    "what happens next",
)


FINE_DETAIL_QUERY_TERMS = (
    "wearing",
    "color",
    "colour",
    "holding",
    "written",
    "text",
    "sign",
    "logo",
    "glasses",
    "pointing",
    "looking at",
    "wristwatch",
    "small object",
)


SCREEN_STATE_QUERY_TERMS = (
    "screen",
    "website",
    "cursor",
    "display screen",
    "on the screen",
    "page",
    "subtitles",
)


# ============================================================================
# 7. Helpers
# ============================================================================

def _query_contains_any(
    query: str,
    terms: Sequence[str],
) -> bool:

    normalized = (
        query
        .strip()
        .lower()
    )

    for term in terms:

        if term in normalized:
            return True

    return False


def _coerce_enum(
    analysis: Dict[str, Any],
    field: str,
) -> None:

    if analysis.get(
        field
    ) not in ALLOWED_VALUES[field]:

        analysis[field] = (
            DEFAULT_ANALYSIS[field]
        )


def _extract_json(
    text: str,
) -> Dict[str, Any]:

    text = text.strip()

    try:
        parsed = json.loads(
            text
        )

    except json.JSONDecodeError:

        start = text.find(
            "{"
        )

        end = text.rfind(
            "}"
        )

        if (
            start < 0
            or end <= start
        ):
            raise ValueError(
                "Could not locate JSON object "
                "in profiler output."
            )

        parsed = json.loads(
            text[
                start:
                end + 1
            ]
        )

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "Profiler output must "
            "be a JSON object."
        )

    return parsed


# ============================================================================
# 8. LLM call
# ============================================================================

def _call_profiler_llm(
    query: str,
    *,
    choices: Sequence[str] | None,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: float,
    api_key: str | None,
    verbose: bool,
) -> Dict[str, Any]:

    url = (
        base_url.rstrip("/")
        + "/chat/completions"
    )

    question_payload = {
        "video_question":
            query,

        "answer_choices":
            (
                list(choices)
                if choices is not None
                else None
            ),
    }

    user_prompt = (
        "Profile the following VideoQA query.\n"
        + json.dumps(
            question_payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\nReturn only the JSON object "
        "specified by the system prompt."
    )

    payload = {
        "model":
            model,

        "temperature":
            temperature,

        "messages": [
            {
                "role":
                    "system",

                "content":
                    PROFILER_SYSTEM_PROMPT,
            },
            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ],

        "max_tokens":
            1600,
    }

    headers = {
        "Authorization":
            f"Bearer {api_key or 'EMPTY'}",

        "Content-Type":
            "application/json",
    }

    if verbose:
        print(
            "\n===== PROFILER REQUEST =====",
            flush=True,
        )

        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

    try:

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout_s,
        )

        response.raise_for_status()

    except requests.RequestException as error:

        response_text = ""

        if (
            getattr(
                error,
                "response",
                None,
            )
            is not None
        ):
            response_text = (
                error
                .response
                .text[:2000]
            )

        raise RuntimeError(
            "Profiler request failed. "
            f"URL={url}. "
            f"Response={response_text}"
        ) from error

    try:

        response_json = (
            response.json()
        )

        text = (
            response_json[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
        )

    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:

        raise RuntimeError(
            "Unexpected profiler response: "
            + response.text[:2000]
        ) from error

    if verbose:

        print(
            "\n===== RAW PROFILER OUTPUT =====",
            flush=True,
        )

        print(
            text,
            flush=True,
        )

    return _extract_json(
        text
    )


# ============================================================================
# 9. Infer temporal relation from surface form
# ============================================================================

def infer_temporal_relation(
    query: str,
) -> str:

    if _query_contains_any(
        query,
        BEGINNING_QUERY_TERMS,
    ):
        return "beginning"

    if _query_contains_any(
        query,
        END_QUERY_TERMS,
    ):
        return "end"

    if _query_contains_any(
        query,
        BEFORE_QUERY_TERMS,
    ):
        return "before"

    if _query_contains_any(
        query,
        AFTER_QUERY_TERMS,
    ):
        return "after"

    if _query_contains_any(
        query,
        SEQUENCE_QUERY_TERMS,
    ):
        return "ordering"

    return "none"


# ============================================================================
# 10. Deterministic execution routing
# ============================================================================

def apply_execution_overrides(
    analysis: Dict[str, Any],
    *,
    query: str,
) -> Dict[str, Any]:

    is_first_or_last_occurrence = (
        _query_contains_any(
            query,
            FIRST_OCCURRENCE_QUERY_TERMS,
        )
        or _query_contains_any(
            query,
            LAST_OCCURRENCE_QUERY_TERMS,
        )
    )

        # ------------------------------------------------------------
    # FIRST / LAST OCCURRENCE
    #
    # "first object/person/action" is NOT the same thing as
    # "at the beginning of the video".
    #
    # We must search chronologically for the occurrence.
    # ------------------------------------------------------------

    if is_first_or_last_occurrence:

        analysis["execution_mode"] = "agentic"
        analysis["evidence_requirement"] = "temporal"
        analysis["temporal_relation"] = "ordering"

        analysis["reasoning_type"] = "temporal_order"

        analysis["coverage_requirement"] = "multi_region"
        analysis["selection_mode"] = "multi_event"

        analysis["temporal_requirement"] = "global"
        analysis["temporal_operation"] = "ordering"

        analysis["candidate_requirement"] = "medium"

        analysis["miss_risk"] = "high"
        analysis["fallback_requirement"] = "expand_coverage"

        analysis["evidence_type"] = "first_last_occurrence"

        return analysis
    
    is_counting = (
        analysis[
            "reasoning_type"
        ] in {
            "counting",
            "repetition",
        }
        or _query_contains_any(
            query,
            COUNTING_QUERY_TERMS,
        )
    )

    is_global = (
        analysis[
            "reasoning_type"
        ] == "global_summary"
        or _query_contains_any(
            query,
            GLOBAL_SUMMARY_QUERY_TERMS,
        )
    )

    is_sequence = (
        analysis[
            "reasoning_type"
        ] in {
            "temporal_order",
            "multi_hop",
        }
        or _query_contains_any(
            query,
            SEQUENCE_QUERY_TERMS,
        )
    )

    relation = (
        infer_temporal_relation(
            query
        )
    )

    # ------------------------------------------------------------
    # COUNT
    # ------------------------------------------------------------

    if is_counting:

        analysis[
            "execution_mode"
        ] = "agentic"

        analysis[
            "evidence_requirement"
        ] = "count"

        analysis[
            "temporal_relation"
        ] = "none"

        analysis[
            "reasoning_type"
        ] = "counting"

        analysis[
            "coverage_requirement"
        ] = "full_timeline"

        analysis[
            "selection_mode"
        ] = "all_positive_and_uncertain"

        analysis[
            "temporal_requirement"
        ] = "global"

        analysis[
            "temporal_operation"
        ] = "frequency"

        analysis[
            "candidate_requirement"
        ] = "many"

        if (
            analysis[
                "aggregation_type"
            ]
            == "none"
        ):
            analysis[
                "aggregation_type"
            ] = "occurrences"

        analysis[
            "answer_sensitivity"
        ] = "exact"

        analysis[
            "miss_risk"
        ] = "high"

        analysis[
            "fallback_requirement"
        ] = "expand_coverage"

        analysis[
            "evidence_type"
        ] = "counting_completeness"

        return analysis

    # ------------------------------------------------------------
    # Explicit BEFORE / AFTER
    # ------------------------------------------------------------

    if relation in {
        "before",
        "after",
    }:

        analysis[
            "execution_mode"
        ] = "agentic"

        analysis[
            "evidence_requirement"
        ] = "temporal"

        analysis[
            "temporal_relation"
        ] = relation

        analysis[
            "reasoning_type"
        ] = "temporal_order"

        analysis[
            "coverage_requirement"
        ] = "multi_region"

        analysis[
            "selection_mode"
        ] = "multi_event"

        analysis[
            "temporal_requirement"
        ] = "medium"

        analysis[
            "temporal_operation"
        ] = "before_after"

        analysis[
            "miss_risk"
        ] = "high"

        analysis[
            "evidence_type"
        ] = "anchor_temporal_relation"

        return analysis

    # ------------------------------------------------------------
    # Beginning/end are fixed direct boundary programs.
    # ------------------------------------------------------------

    if relation in {
        "beginning",
        "end",
    }:

        analysis[
            "execution_mode"
        ] = "oneshot"

        analysis[
            "evidence_requirement"
        ] = "temporal"

        analysis[
            "temporal_relation"
        ] = relation

        analysis[
            "coverage_requirement"
        ] = "targeted"

        analysis[
            "selection_mode"
        ] = "beginning_end"

        analysis[
            "temporal_requirement"
        ] = "local"

        analysis[
            "temporal_operation"
        ] = "ordering"

        analysis[
            "candidate_requirement"
        ] = "few"

        analysis[
            "evidence_type"
        ] = (
            "boundary_beginning"
            if relation
            == "beginning"
            else "boundary_end"
        )

        return analysis

    # ------------------------------------------------------------
    # GLOBAL / workflow / sequence
    # ------------------------------------------------------------

    if (
        is_global
        or is_sequence
    ):

        analysis[
            "execution_mode"
        ] = "agentic"

        analysis[
            "evidence_requirement"
        ] = "global"

        analysis[
            "temporal_relation"
        ] = (
            "ordering"
            if is_sequence
            else "none"
        )

        if is_sequence:
            analysis[
                "reasoning_type"
            ] = "temporal_order"

            analysis[
                "temporal_operation"
            ] = "ordering"

        else:
            analysis[
                "reasoning_type"
            ] = "global_summary"

        analysis[
            "coverage_requirement"
        ] = "full_timeline"

        analysis[
            "selection_mode"
        ] = "uniform"

        analysis[
            "temporal_requirement"
        ] = "global"

        analysis[
            "context_requirement"
        ] = "long"

        analysis[
            "candidate_requirement"
        ] = "many"

        analysis[
            "evidence_type"
        ] = (
            "sequence_ordering"
            if is_sequence
            else "global_process"
        )

        return analysis

    # ------------------------------------------------------------
    # LOCAL
    #
    # Important architectural rule:
    #
    # LOCAL does not mean open-ended agent search.
    #
    # Perform at most one semantic localization pass.
    # ------------------------------------------------------------

    analysis[
        "execution_mode"
    ] = "oneshot"

    analysis[
        "evidence_requirement"
    ] = "local"

    analysis[
        "temporal_relation"
    ] = "none"

    analysis[
        "coverage_requirement"
    ] = "targeted"

    analysis[
        "selection_mode"
    ] = "top_k"

    analysis[
        "temporal_requirement"
    ] = "local"

    if (
        analysis[
            "candidate_requirement"
        ]
        == "many"
    ):
        analysis[
            "candidate_requirement"
        ] = "medium"

    if _query_contains_any(
        query,
        FINE_DETAIL_QUERY_TERMS,
    ):

        analysis[
            "reasoning_type"
        ] = "fine_detail"

        analysis[
            "precision_requirement"
        ] = "high"

        if (
            analysis[
                "spatial_strategy"
            ]
            == "full_frame"
        ):
            analysis[
                "spatial_strategy"
            ] = "object_crop"

        analysis[
            "evidence_type"
        ] = "localized_visual_detail"

    else:

        if (
            analysis[
                "reasoning_type"
            ]
            == "ambiguous"
        ):
            analysis[
                "reasoning_type"
            ] = "local_event"

        analysis[
            "evidence_type"
        ] = "localized_event"

    return analysis


# ============================================================================
# 11. Validate/coerce LLM analysis
# ============================================================================

def coerce_and_validate_analysis(
    raw_analysis: Dict[str, Any],
    *,
    query: str = "",
) -> Dict[str, Any]:

    if not isinstance(
        raw_analysis,
        dict,
    ):
        raw_analysis = {}

    analysis = {
        **DEFAULT_ANALYSIS,
        **raw_analysis,
    }

    analysis["required_modalities"] = ["visual"]

    for field in ALLOWED_VALUES:
        _coerce_enum(
            analysis,
            field,
        )

    # Visual-only experiment.
    # Never allow profiler output to introduce another modality.

    try:
        confidence = float(
            analysis.get(
                "profile_confidence",
                0.5,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        confidence = 0.5

    analysis[
        "profile_confidence"
    ] = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    rationale = analysis.get(
        "rationale",
        "",
    )

    if not isinstance(
        rationale,
        str,
    ):
        rationale = str(
            rationale
        )

    analysis[
        "rationale"
    ] = rationale[:1000]

    # ------------------------------------------------------------
    # Basic semantic invariants.
    # ------------------------------------------------------------

    reasoning_type = (
        analysis[
            "reasoning_type"
        ]
    )

    if reasoning_type == "fine_detail":
        analysis["precision_requirement"] = "high"

        if analysis["spatial_strategy"] == "full_frame":
            analysis["spatial_strategy"] = "object_crop"

    if (
        analysis[
            "aggregation_type"
        ]
        == "distinct_entities"
        and analysis[
            "identity_requirement"
        ]
        == "none"
    ):
        analysis[
            "identity_requirement"
        ] = (
            "cross_window_reidentification"
        )

    # ------------------------------------------------------------
    # Macro routing is applied LAST.
    #
    # This prevents semantic labels from overriding the execution
    # geometry.
    # ------------------------------------------------------------

    analysis = (
        apply_execution_overrides(
            analysis,
            query=query,
        )
    )

    return analysis


# ============================================================================
# 12. Cheap router
# ============================================================================

def route_query_cheap(
    query: str,
    *,
    duration_s: float,
    choices: Sequence[str] | None = None,
) -> RouterDecision:

    is_first_or_last_occurrence = (
        _query_contains_any(
            query,
            FIRST_OCCURRENCE_QUERY_TERMS,
        )
        or _query_contains_any(
            query,
            LAST_OCCURRENCE_QUERY_TERMS,
        )
    )
    # ------------------------------------------------------------
    # First / last occurrence.
    # ------------------------------------------------------------
    if is_first_or_last_occurrence:
        return RouterDecision(
            policy_name="temporal",
            execution_mode="agentic",
            evidence_requirement="temporal",
            temporal_relation="ordering",
            confidence=0.94,
            reason="first_last_occurrence",
        )

    relation = (
        infer_temporal_relation(
            query
        )
    )

    is_counting = (
        _query_contains_any(
            query,
            COUNTING_QUERY_TERMS,
        )
    )

    is_global = (
        _query_contains_any(
            query,
            GLOBAL_SUMMARY_QUERY_TERMS,
        )
    )

    is_sequence = (
        _query_contains_any(
            query,
            SEQUENCE_QUERY_TERMS,
        )
    )

    is_detail = (
        _query_contains_any(
            query,
            FINE_DETAIL_QUERY_TERMS,
        )
        or _query_contains_any(
            query,
            SCREEN_STATE_QUERY_TERMS,
        )
    )

    # ------------------------------------------------------------
    # Count.
    # ------------------------------------------------------------

    if is_counting:

        return RouterDecision(
            policy_name=
                "counting",

            execution_mode=
                "agentic",

            evidence_requirement=
                "count",

            temporal_relation=
                "none",

            confidence=
                0.98,

            reason=
                "counting",
        )

    # ------------------------------------------------------------
    # Before/after.
    # ------------------------------------------------------------

    if relation in {
        "before",
        "after",
    }:

        return RouterDecision(
            policy_name=
                "temporal",

            execution_mode=
                "agentic",

            evidence_requirement=
                "temporal",

            temporal_relation=
                relation,

            confidence=
                0.94,

            reason=
                f"temporal_{relation}",
        )

    # ------------------------------------------------------------
    # Beginning/end.
    # ------------------------------------------------------------

    if relation in {
        "beginning",
        "end",
    }:

        return RouterDecision(
            policy_name=
                "detail",

            execution_mode=
                "oneshot",

            evidence_requirement=
                "temporal",

            temporal_relation=
                relation,

            confidence=
                0.95,

            reason=
                f"boundary_{relation}",
        )

    # ------------------------------------------------------------
    # Global.
    # ------------------------------------------------------------

    if (
        is_global
        or is_sequence
    ):

        return RouterDecision(
            policy_name=(
                "sequence"
                if is_sequence
                else "global"
            ),

            execution_mode=
                "agentic",

            evidence_requirement=
                "global",

            temporal_relation=(
                "ordering"
                if is_sequence
                else "none"
            ),

            confidence=
                0.94,

            reason=(
                "sequence"
                if is_sequence
                else "global"
            ),
        )

    # ------------------------------------------------------------
    # Very long but localized questions stay local.
    #
    # Length alone should NOT turn them into global agent search.
    # ------------------------------------------------------------

    if is_detail:

        return RouterDecision(
            policy_name=(
                "long_sparse"
                if duration_s >= 1200
                else "detail"
            ),

            execution_mode=
                "oneshot",

            evidence_requirement=
                "local",

            temporal_relation=
                "none",

            confidence=
                0.90,

            reason=
                "local_detail",
        )

    # ------------------------------------------------------------
    # Generic MCQ.
    # ------------------------------------------------------------

    if choices:

        return RouterDecision(
            policy_name=(
                "long_sparse"
                if duration_s >= 1200
                else "cheap"
            ),

            execution_mode=
                "oneshot",

            evidence_requirement=
                "local",

            temporal_relation=
                "none",

            confidence=
                0.80,

            reason=
                "generic_local_mcq",
        )

    # ------------------------------------------------------------
    # Unknown -> LLM.
    # ------------------------------------------------------------

    return RouterDecision(
        policy_name=
            "cheap",

        execution_mode=
            "oneshot",

        evidence_requirement=
            "local",

        temporal_relation=
            "none",

        confidence=
            0.40,

        reason=
            "uncertain",

        out_of_distribution=
            True,
    )


# ============================================================================
# 13. Numeric helpers
# ============================================================================

def choose_topk(
    requirement: str,
) -> int:

    return {
        "few": 4,
        "medium": 6,
        "many": 8,
    }[
        requirement
    ]


def choose_window_length(
    *,
    context_requirement: str,
    evidence_requirement: str,
) -> float:

    base = {
        "short": 4.0,
        "medium": 8.0,
        "long": 16.0,
    }[
        context_requirement
    ]

    if evidence_requirement in {
        "temporal",
        "global",
    }:
        return max(
            base,
            16.0,
        )

    if evidence_requirement == "count":
        return max(
            base,
            8.0,
        )

    return base


def choose_vlm_budget(
    precision_requirement: str,
) -> int:

    return {
        "low": 8,
        "medium": 16,
        "high": 32,
    }[
        precision_requirement
    ]


def choose_high_frames_per_window(
    *,
    evidence_requirement: str,
    precision_requirement: str,
) -> int:

    if (
        precision_requirement
        == "high"
    ):
        return 16

    if evidence_requirement in {
        "count",
        "temporal",
    }:
        return 12

    return 8


def choose_thresholds(
    *,
    evidence_requirement: str,
    miss_risk: str,
) -> Tuple[
    float,
    float,
]:

    if evidence_requirement == "count":

        candidate = 0.35
        uncertainty = 0.15

    elif evidence_requirement in {
        "global",
        "temporal",
    }:

        candidate = 0.40
        uncertainty = 0.20

    else:

        candidate = 0.45
        uncertainty = 0.25

    if miss_risk == "high":

        candidate -= 0.05
        uncertainty -= 0.05

    return (
        max(
            0.0,
            candidate,
        ),

        max(
            0.0,
            uncertainty,
        ),
    )


def choose_spatial_tier(
    *,
    precision_requirement: str,
    spatial_strategy: str,
) -> str:

    return {
        "low": "low",
        "medium": "medium",
        "high": "high",
    }[
        precision_requirement
    ]


def choose_merge_gap(
    evidence_requirement: str,
) -> float:

    if evidence_requirement == "count":
        return 2.0

    if evidence_requirement in {
        "temporal",
        "global",
    }:
        return 1.0

    return 0.5


# ============================================================================
# 14. Agent bounds
# ============================================================================

def choose_agent_bounds(
    *,
    execution_mode: str,
    evidence_requirement: str,
    temporal_relation: str,
) -> Dict[str, int]:

    # ------------------------------------------------------------
    # Fixed / bounded one-shot programs.
    # ------------------------------------------------------------

    if execution_mode == "oneshot":

        if evidence_requirement == "local":
            return {
                "max_steps": 2,
                "max_local_searches": 1,
                "max_global_scans": 0,
                "max_count_scans": 0,
                "max_temporal_searches": 0,
                "max_density_refinements": 1,
                "max_contrastive_checks": 1,
            }

        if (
            evidence_requirement == "temporal"
            and temporal_relation in {
                "beginning",
                "end",
            }
        ):
            return {
                "max_steps": 2,
                "max_local_searches": 0,
                "max_global_scans": 0,
                "max_count_scans": 0,
                "max_temporal_searches": 0,
                "max_density_refinements": 1,
                "max_contrastive_checks": 1,
            }

        return {
            "max_steps": 1,
            "max_local_searches": 0,
            "max_global_scans": 0,
            "max_count_scans": 0,
            "max_temporal_searches": 0,
            "max_density_refinements": 0,
            "max_contrastive_checks": 0,
        }

    # ------------------------------------------------------------
    # Global chronological execution.
    # ------------------------------------------------------------

    if evidence_requirement == "global":
        return {
            "max_steps": 2,
            "max_local_searches": 0,
            "max_global_scans": 2,
            "max_count_scans": 0,
            "max_temporal_searches": 0,
            "max_density_refinements": 1,
            "max_contrastive_checks": 0,
        }

    # ------------------------------------------------------------
    # Counting.
    # ------------------------------------------------------------

    if evidence_requirement == "count":
        return {
            "max_steps": 4,
            "max_local_searches": 0,
            "max_global_scans": 0,
            "max_count_scans": 1,
            "max_temporal_searches": 2,
            "max_density_refinements": 2,
            "max_contrastive_checks": 0,
        }

    # ------------------------------------------------------------
    # Temporal relationship / first-last occurrence.
    # ------------------------------------------------------------

    if evidence_requirement == "temporal":
        return {
            "max_steps": 3,
            "max_local_searches": 1,
            "max_global_scans": 0,
            "max_count_scans": 0,
            "max_temporal_searches": 1,
            "max_density_refinements": 1,
            "max_contrastive_checks": 1,
        }

    return {
        "max_steps": 2,
        "max_local_searches": 1,
        "max_global_scans": 0,
        "max_count_scans": 0,
        "max_temporal_searches": 0,
        "max_density_refinements": 1,
        "max_contrastive_checks": 1,
    }


# ============================================================================
# 15. Compile profile -> requested config
# ============================================================================

def compile_execution_policy(
    *,
    analysis: Dict[str, Any],
    duration_s: float,
) -> BudgetConfig:

    if duration_s <= 0:
        raise ValueError(
            "duration_s must be positive"
        )

    execution_mode = (
        analysis[
            "execution_mode"
        ]
    )

    requirement = (
        analysis[
            "evidence_requirement"
        ]
    )

    relation = (
        analysis[
            "temporal_relation"
        ]
    )

    # ------------------------------------------------------------
    # Scan defaults.
    # ------------------------------------------------------------

    if requirement == "count":

        probe_fps = 0.03125
        chunk_len_s = (
            1.0
            / probe_fps
        )
        frames_per_chunk = 1

    elif requirement == "global":

        probe_fps = (
            0.015625
            if duration_s < 1200
            else 0.00390625
        )

        chunk_len_s = (
            1.0
            / probe_fps
        )

        frames_per_chunk = 1

    elif requirement == "temporal":

        probe_fps = 0.05
        chunk_len_s = 20.0
        frames_per_chunk = 1

    else:

        probe_fps = (
            0.05
            if duration_s < 1200
            else 0.00390625
        )

        chunk_len_s = (
            1.0
            / probe_fps
        )

        frames_per_chunk = 1

    # ------------------------------------------------------------
    # Top-k.
    # ------------------------------------------------------------

    if (
        analysis[
            "selection_mode"
        ]
        == "top_k"
    ):

        topk: int | None = (
            choose_topk(
                analysis[
                    "candidate_requirement"
                ]
            )
        )

    else:

        topk = None

    # ------------------------------------------------------------
    # Window.
    # ------------------------------------------------------------

    window_len_s = (
        choose_window_length(
            context_requirement=
                analysis[
                    "context_requirement"
                ],

            evidence_requirement=
                requirement,
        )
    )

    # ------------------------------------------------------------
    # Budgets.
    # ------------------------------------------------------------

    vlm_budget = (
        choose_vlm_budget(
            analysis[
                "precision_requirement"
            ]
        )
    )

    if requirement == "count":
        vlm_budget = max(
            32,
            vlm_budget,
        )

    if requirement == "global":
        vlm_budget = max(
            16,
            vlm_budget,
        )

    high_frames = (
        choose_high_frames_per_window(
            evidence_requirement=
                requirement,

            precision_requirement=
                analysis[
                    "precision_requirement"
                ],
        )
    )

    candidate_threshold, uncertainty_threshold = (
        choose_thresholds(
            evidence_requirement=
                requirement,

            miss_risk=
                analysis[
                    "miss_risk"
                ],
        )
    )

    bounds = (
        choose_agent_bounds(
            execution_mode=
                execution_mode,

            evidence_requirement=
                requirement,

            temporal_relation=
                relation,
        )
    )

    return BudgetConfig(
        name=(
            f"dynamic_"
            f"{execution_mode}_"
            f"{requirement}"
        ),

        execution_mode=
            execution_mode,

        evidence_requirement=
            requirement,

        temporal_relation=
            relation,

        reasoning_type=
            analysis[
                "reasoning_type"
            ],

        answer_type=
            analysis[
                "answer_type"
            ],

        coverage_mode=
            analysis[
                "coverage_requirement"
            ],

        selection_mode=
            analysis[
                "selection_mode"
            ],

        temporal_operation=
            analysis[
                "temporal_operation"
            ],

        aggregation_type=
            analysis[
                "aggregation_type"
            ],

        identity_requirement=
            analysis[
                "identity_requirement"
            ],

        spatial_strategy=
            analysis[
                "spatial_strategy"
            ],

        required_modalities=
            tuple(
                analysis[
                    "required_modalities"
                ]
            ),

        probe_fps=
            probe_fps,

        chunk_len_s=
            chunk_len_s,

        frames_per_chunk=
            frames_per_chunk,

        probe_topk=
            topk,

        action_topk=
            topk,

        candidate_threshold=
            candidate_threshold,

        uncertainty_threshold=
            uncertainty_threshold,

        window_len_s=
            window_len_s,

        high_frames_per_window=
            high_frames,

        high_spatial_tier=
            choose_spatial_tier(
                precision_requirement=
                    analysis[
                        "precision_requirement"
                    ],

                spatial_strategy=
                    analysis[
                        "spatial_strategy"
                    ],
            ),

        merge_gap_s=
            choose_merge_gap(
                requirement
            ),

        vlm_budget=
            vlm_budget,

        quality_tier=
            "requested",

        fallback_mode=
            analysis[
                "fallback_requirement"
            ],

        min_temporal_coverage=(
            1.0
            if requirement
            in {
                "global",
                "count",
            }
            else 0.0
        ),

        profile_confidence=
            analysis[
                "profile_confidence"
            ],

        miss_risk=
            analysis[
                "miss_risk"
            ],

        answer_sensitivity=
            analysis[
                "answer_sensitivity"
            ],

        evidence_type=
            analysis.get(
                "evidence_type",
                "generic",
            ),

        expand_neighbors=(
            requirement
            == "local"
            and analysis[
                "precision_requirement"
            ]
            == "high"
        ),

        preserve_order=(
            requirement
            in {
                "global",
                "count",
                "temporal",
            }
        ),

        include_uniform_anchors=(
            requirement
            in {
                "global",
                "count",
            }
        ),

        **bounds,

        rationale=
            analysis.get(
                "rationale",
                "",
            ),
    )


# ============================================================================
# 16. Cheap-router config
# ============================================================================

def catalog_policy_to_budget_config(
    *,
    decision: RouterDecision,
) -> BudgetConfig:

    catalog = (
        POLICY_CATALOG[
            decision.policy_name
        ]
    )

    bounds = (
        choose_agent_bounds(
            execution_mode=
                decision.execution_mode,

            evidence_requirement=
                decision.evidence_requirement,

            temporal_relation=
                decision.temporal_relation,
        )
    )

    requirement = (
        decision.evidence_requirement
    )

    if requirement == "count":

        reasoning_type = "counting"
        coverage_mode = "full_timeline"
        selection_mode = "all_positive_and_uncertain"
        temporal_operation = "frequency"
        aggregation_type = "occurrences"

    elif requirement == "global":

        reasoning_type = "global_summary"
        coverage_mode = "full_timeline"
        selection_mode = "uniform"
        temporal_operation = (
            "ordering"
            if decision.temporal_relation
            == "ordering"
            else "none"
        )
        aggregation_type = "segments"

    elif requirement == "temporal":

        reasoning_type = "temporal_order"
        coverage_mode = (
            "targeted"
            if decision.temporal_relation
            in {
                "beginning",
                "end",
            }
            else "multi_region"
        )
        selection_mode = (
            "beginning_end"
            if decision.temporal_relation
            in {
                "beginning",
                "end",
            }
            else "multi_event"
        )
        temporal_operation = (
            "before_after"
            if decision.temporal_relation
            in {
                "before",
                "after",
            }
            else "ordering"
        )
        aggregation_type = "segments"

    else:

        reasoning_type = "local_event"
        coverage_mode = "targeted"
        selection_mode = "top_k"
        temporal_operation = "none"
        aggregation_type = "none"

    return BudgetConfig(
        name=(
            f"router_"
            f"{catalog.name}"
        ),

        execution_mode=
            decision.execution_mode,

        evidence_requirement=
            requirement,

        temporal_relation=
            decision.temporal_relation,

        reasoning_type=
            reasoning_type,

        answer_type=
            "multiple_choice",

        coverage_mode=
            coverage_mode,

        selection_mode=
            selection_mode,

        temporal_operation=
            temporal_operation,

        aggregation_type=
            aggregation_type,

        identity_requirement=
            "none",

        spatial_strategy=
            "full_frame",

        required_modalities=
            ("visual",),

        probe_fps=
            catalog.probe_fps,

        chunk_len_s=(
            1.0
            / catalog.probe_fps
        ),

        frames_per_chunk=
            1,

        probe_topk=(
            catalog.probe_topk
            if selection_mode
            == "top_k"
            else None
        ),

        action_topk=(
            catalog.action_topk
            if selection_mode
            == "top_k"
            else None
        ),

        candidate_threshold=
            0.4,

        uncertainty_threshold=
            0.2,

        window_len_s=
            catalog.window_len_s,

        high_frames_per_window=
            min(
                16,
                max(
                    8,
                    catalog.vlm_budget,
                ),
            ),

        high_spatial_tier=
            "medium",

        merge_gap_s=
            choose_merge_gap(
                requirement
            ),

        vlm_budget=
            catalog.vlm_budget,

        quality_tier=
            "router",

        fallback_mode=(
            "expand_coverage"
            if decision.execution_mode
            == "agentic"
            else "none"
        ),

        min_temporal_coverage=(
            1.0
            if requirement
            in {
                "global",
                "count",
            }
            else 0.0
        ),

        profile_confidence=
            decision.confidence,

        miss_risk=
            "medium",

        answer_sensitivity=
            "exact",

        evidence_type=
            decision.reason,

        expand_neighbors=
            catalog.expand_neighbors,

        preserve_order=
            catalog.preserve_order,

        include_uniform_anchors=
            catalog.include_uniform_anchors,

        **bounds,

        rationale=
            decision.reason,
    )


def budget_config_from_router(
    decision: RouterDecision,
) -> BudgetConfig:

    return (
        catalog_policy_to_budget_config(
            decision=decision,
        )
    )


# ============================================================================
# 17. Resource adaptation
# ============================================================================

def adapt_budget_for_resources(
    requested: BudgetConfig,
    resource_state: ResourceState,
) -> BudgetConfig:

    load = (
        resource_state.load_level
    )

    if load not in {
        "low",
        "medium",
        "high",
    }:
        raise ValueError(
            "load_level must be "
            "low, medium, or high"
        )

    if load == "low":

        return replace(
            requested,
            quality_tier=
                "resource_low",
        )

    # ------------------------------------------------------------
    # Preserve control flow.
    #
    # Resource pressure must NOT turn GLOBAL into LOCAL, COUNT into
    # TOP-K, etc.
    # ------------------------------------------------------------

    if load == "medium":

        return replace(
            requested,

            high_frames_per_window=
                max(
                    6,
                    requested
                    .high_frames_per_window
                    // 2,
                ),

            vlm_budget=
                max(
                    8,
                    requested.vlm_budget
                    // 2,
                ),

            quality_tier=
                "resource_medium",
        )

    return replace(
        requested,

        high_frames_per_window=
            max(
                4,
                requested
                .high_frames_per_window
                // 3,
            ),

        vlm_budget=
            max(
                8,
                requested.vlm_budget
                // 4,
            ),

        quality_tier=
            "resource_high",
    )


# ============================================================================
# 18. Allowed actions
# ============================================================================

def allowed_actions_for_config(
    config: BudgetConfig,
) -> List[str]:

    if config.execution_mode == "oneshot":

        if (
            config.evidence_requirement
            == "local"
        ):

            return [
                "SEARCH_LOCAL",
                "INCREASE_DENSITY",
                "COMPARE_CHOICES",
                "ANSWER",
            ]

        if (
            config.evidence_requirement
            == "temporal"
        ):

            return [
                "INCREASE_DENSITY",
                "COMPARE_CHOICES",
                "ANSWER",
            ]

        return [
            "ANSWER",
        ]

    if (
        config.evidence_requirement
        == "global"
    ):

        return [
            "GLOBAL_SCAN",
            "INCREASE_DENSITY",
            "ANSWER",
        ]

    if (
        config.evidence_requirement
        == "count"
    ):

        return [
            "COUNT_EVENTS",
            "INCREASE_DENSITY",
            "SEARCH_BEFORE",
            "SEARCH_AFTER",
            "ANSWER",
        ]

    if (
        config.evidence_requirement
        == "temporal"
    ):

        return [
            "SEARCH_LOCAL",
            "SEARCH_BEFORE",
            "SEARCH_AFTER",
            "INCREASE_DENSITY",
            "COMPARE_CHOICES",
            "ANSWER",
        ]

    return [
        "SEARCH_LOCAL",
        "INCREASE_DENSITY",
        "ANSWER",
    ]


# ============================================================================
# 19. Config -> runtime policy
# ============================================================================

def budget_to_policy(
    config: BudgetConfig,
) -> Dict[str, Any]:

    return {
        # ============================================================
        # MACRO CONTROL PLANE
        # ============================================================

        "execution_mode":
            config.execution_mode,

        "evidence_requirement":
            config.evidence_requirement,

        "temporal_relation":
            config.temporal_relation,

        "allowed_actions":
            allowed_actions_for_config(
                config
            ),

        # ============================================================
        # HARD BOUNDS
        # ============================================================

        "max_steps":
            config.max_steps,

        "max_local_searches":
            config.max_local_searches,

        "max_global_scans":
            config.max_global_scans,

        "max_count_scans":
            config.max_count_scans,

        "max_temporal_searches":
            config.max_temporal_searches,

        "max_density_refinements":
            config.max_density_refinements,

        "max_contrastive_checks":
            config.max_contrastive_checks,

        # ============================================================
        # SEMANTIC PROFILE
        # ============================================================

        "reasoning_type":
            config.reasoning_type,

        "answer_type":
            config.answer_type,

        "coverage_mode":
            config.coverage_mode,

        "selection_mode":
            config.selection_mode,

        "temporal_operation":
            config.temporal_operation,

        "aggregation_type":
            config.aggregation_type,

        "identity_requirement":
            config.identity_requirement,

        "spatial_strategy":
            config.spatial_strategy,

        "required_modalities":
            list(
                config.required_modalities
            ),

        # ============================================================
        # RETRIEVAL
        # ============================================================

        "probe_fps":
            config.probe_fps,

        "chunk_len_s":
            config.chunk_len_s,

        "frames_per_chunk":
            config.frames_per_chunk,

        "probe_topk":
            config.probe_topk,

        "action_topk":
            config.action_topk,

        "candidate_threshold":
            config.candidate_threshold,

        "uncertainty_threshold":
            config.uncertainty_threshold,

        # ============================================================
        # REFINEMENT
        # ============================================================

        "window_len_s":
            config.window_len_s,

        "high_frames_per_window":
            config.high_frames_per_window,

        "high_spatial_tier":
            config.high_spatial_tier,

        "merge_gap_s":
            config.merge_gap_s,

        # ============================================================
        # ANSWER
        # ============================================================

        "answer_max_images_total":
            config.vlm_budget,

        "answer_max_frames_per_window":
            config.high_frames_per_window,

        "answer_tier":
            config.answer_tier,

        "cheap_answer_tier":
            config.cheap_answer_tier,

        # ============================================================
        # ROBUSTNESS
        # ============================================================

        "fallback_mode":
            config.fallback_mode,

        "min_temporal_coverage":
            config.min_temporal_coverage,

        "profile_confidence":
            config.profile_confidence,

        "miss_risk":
            config.miss_risk,

        "answer_sensitivity":
            config.answer_sensitivity,

        "evidence_type":
            config.evidence_type,

        "expand_neighbors":
            config.expand_neighbors,

        "preserve_order":
            config.preserve_order,

        "include_uniform_anchors":
            config.include_uniform_anchors,

        "quality_tier":
            config.quality_tier,

        "rationale":
            config.rationale,
    }


# ============================================================================
# 20. Cost estimate
# ============================================================================

def estimate_config_cost(
    config: BudgetConfig,
    *,
    duration_s: float,
) -> float:

    if duration_s <= 0:
        raise ValueError(
            "duration_s must be positive"
        )

    probe_frames = max(
        1.0,
        duration_s
        * config.probe_fps,
    )

    if config.execution_mode == "oneshot":

        high_windows = (
            config.action_topk
            or 1
        )

    elif (
        config.evidence_requirement
        == "global"
    ):

        high_windows = (
            config.max_global_scans
        )

    elif (
        config.evidence_requirement
        == "count"
    ):

        high_windows = max(
            1,
            config.max_density_refinements,
        )

    else:

        high_windows = max(
            1,
            config.max_temporal_searches,
        )

    high_frames = (
        high_windows
        * config.high_frames_per_window
    )

    return float(
        probe_frames
        + 4.0 * high_frames
        + config.vlm_budget
    )


# ============================================================================
# 21. LLM profiling
# ============================================================================

def profile_query_llm(
    query: str,
    *,
    duration_s: float,
    choices: Sequence[str] | None = None,
    base_url: str = "http://localhost:8000/v1",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_s: float = 30.0,
    api_key: str | None = None,
    resource_state: ResourceState | None = None,
    verbose: bool = True,
) -> ProfilerResult:

    if not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    if duration_s <= 0:
        raise ValueError(
            "duration_s must be positive"
        )

    if resource_state is None:
        resource_state = (
            ResourceState()
        )

    raw_json = (
        _call_profiler_llm(
            query=query,
            choices=choices,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            api_key=api_key,
            verbose=verbose,
        )
    )

    analysis = (
        coerce_and_validate_analysis(
            raw_json.get(
                "analysis",
                {},
            ),
            query=query,
        )
    )

    requested = (
        compile_execution_policy(
            analysis=analysis,
            duration_s=duration_s,
        )
    )

    low = (
        adapt_budget_for_resources(
            requested,
            ResourceState(
                load_level="low",
            ),
        )
    )

    medium = (
        adapt_budget_for_resources(
            requested,
            ResourceState(
                free_gpu_mem_gb=
                    resource_state
                    .free_gpu_mem_gb,

                encoder_queue_len=
                    resource_state
                    .encoder_queue_len,

                vlm_queue_len=
                    resource_state
                    .vlm_queue_len,

                load_level=
                    "medium",
            ),
        )
    )

    high = (
        adapt_budget_for_resources(
            requested,
            ResourceState(
                free_gpu_mem_gb=
                    resource_state
                    .free_gpu_mem_gb,

                encoder_queue_len=
                    resource_state
                    .encoder_queue_len,

                vlm_queue_len=
                    resource_state
                    .vlm_queue_len,

                load_level=
                    "high",
            ),
        )
    )

    candidates = [
        low,
        medium,
        high,
    ]

    chosen = (
        adapt_budget_for_resources(
            requested,
            resource_state,
        )
    )

    policy = (
        budget_to_policy(
            chosen
        )
    )

    if verbose:

        print(
            "\n===== VALIDATED ANALYSIS =====",
            flush=True,
        )

        print(
            json.dumps(
                analysis,
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        print(
            "\n===== EXECUTION POLICY =====",
            flush=True,
        )

        print(
            json.dumps(
                policy,
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

    return ProfilerResult(
        analysis=
            analysis,

        candidate_configs=
            candidates,

        requested_config=
            requested,

        chosen_config=
            chosen,

        execution_policy=
            policy,

        raw_json=
            raw_json,
    )


# ============================================================================
# 22. Adaptive cheap-router + LLM path
# ============================================================================

def profile_query_adaptive(
    query: str,
    *,
    duration_s: float,
    choices: Sequence[str] | None = None,

    confidence_threshold: float = 0.75,
    exploration_rate: float = 0.02,
    random_seed: int | None = None,

    base_url: str = "http://localhost:8000/v1",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_s: float = 30.0,
    api_key: str | None = None,

    resource_state: ResourceState | None = None,
    verbose: bool = True,
) -> AdaptiveProfilerResult:

    if resource_state is None:
        resource_state = (
            ResourceState()
        )

    router = (
        route_query_cheap(
            query,
            duration_s=duration_s,
            choices=choices,
        )
    )

    rng = random.Random(
        random_seed
    )

    low_confidence = (
        router.confidence
        < confidence_threshold
    )

    exploration = (
        rng.random()
        < exploration_rate
    )

    use_llm = (
        router.out_of_distribution
        or low_confidence
        or exploration
    )

    # ------------------------------------------------------------
    # FAST PATH
    # ------------------------------------------------------------

    if not use_llm:

        requested = (
            budget_config_from_router(
                router
            )
        )

        chosen = (
            adapt_budget_for_resources(
                requested,
                resource_state,
            )
        )

        policy = (
            budget_to_policy(
                chosen
            )
        )

        if verbose:

            print(
                "\n===== ROUTER FAST PATH =====",
                flush=True,
            )

            print(
                json.dumps(
                    {
                        "router":
                            asdict(
                                router
                            ),

                        "execution_policy":
                            policy,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )

        return AdaptiveProfilerResult(
            source=
                "router",

            router_decision=
                router,

            chosen_config=
                chosen,

            execution_policy=
                policy,
        )

    # ------------------------------------------------------------
    # LLM EXPERT PATH
    # ------------------------------------------------------------

    if verbose:

        print(
            "\n===== LLM PROFILER ESCALATION =====",
            flush=True,
        )

        print(
            json.dumps(
                asdict(
                    router
                ),
                indent=2,
            ),
            flush=True,
        )

    llm_result = (
        profile_query_llm(
            query=query,
            duration_s=duration_s,
            choices=choices,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            api_key=api_key,
            resource_state=resource_state,
            verbose=verbose,
        )
    )

    return AdaptiveProfilerResult(
        source=
            "llm",

        router_decision=
            router,

        chosen_config=
            llm_result
            .chosen_config,

        execution_policy=
            llm_result
            .execution_policy,

        llm_result=
            llm_result,
    )


# ============================================================================
# 23. Backward-compatible wrapper
# ============================================================================

def profile_query_llm_legacy(
    query: str,
    *,
    duration_s: float,
    choices: Sequence[str] | None = None,
    base_url: str = "http://localhost:8000/v1",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_s: float = 30.0,
    api_key: str | None = None,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:

    result = (
        profile_query_llm(
            query=query,
            duration_s=duration_s,
            choices=choices,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            api_key=api_key,
        )
    )

    return (
        result.execution_policy,
        result.analysis,
    )


# ============================================================================
# 24. Standalone test
# ============================================================================

if __name__ == "__main__":

    examples = [

        (
            "What is the man doing with the box?",
            900.0,
        ),

        (
            "How many times does the man enter the room?",
            900.0,
        ),

        (
            "What happens after the man opens the door?",
            900.0,
        ),

        (
            "What happens at the beginning of the video?",
            900.0,
        ),

        (
            "Describe the key sequence of events in the video "
            "and summarize the overall workflow.",
            180.0,
        ),
    ]

    for question, duration in examples:

        print(
            "\n"
            + "=" * 80
        )

        print(
            "QUESTION:",
            question,
        )

        decision = (
            route_query_cheap(
                question,
                duration_s=duration,
                choices=[
                    "A",
                    "B",
                    "C",
                    "D",
                ],
            )
        )

        print(
            json.dumps(
                asdict(
                    decision
                ),
                indent=2,
            )
        )

        config = (
            budget_config_from_router(
                decision
            )
        )

        print(
            json.dumps(
                budget_to_policy(
                    config
                ),
                indent=2,
            )
        )