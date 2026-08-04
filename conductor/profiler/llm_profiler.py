from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import requests


# ============================================================================
# 1. Profiler prompt
# ============================================================================

PROFILER_SYSTEM_PROMPT = """
You are a semantic query profiler for a long-video question-answering system.

Your task is NOT to choose exact frame rates, top-k values, or GPU budgets.
Your task is to infer the structure of evidence needed to answer the question.

Return STRICT JSON ONLY. Do not use markdown or explanatory text outside JSON.

Output schema:

{
  "analysis": {
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
      "ocr_crop" |
      "face_crop",

    "required_modalities": [
      "visual" |
      "audio" |
      "ocr" |
      "subtitle" |
      "metadata"
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

    "rationale": "<one concise explanation>"
  }
}

Interpretation rules:

1. Counting and repetition questions generally require full-timeline coverage.
2. Counting must not use top-k-only selection.
3. Exact occurrence counting should use:
   - aggregation_type = "occurrences"
   - temporal_operation = "frequency"
4. Counting distinct people or objects should use:
   - aggregation_type = "distinct_entities"
   - cross-window identity tracking
5. Questions asking how many are visible at once should use:
   - aggregation_type = "simultaneous_max"
6. Temporal-order and multi-hop questions require multiple separated regions.
7. Fine-detail, text-reading, label, number, color, and small-object questions
   require high precision and may require a crop.
8. Global-summary questions require broad uniform coverage.
9. State-change questions generally need beginning/end or multiple-state evidence.
10. Ambiguous or low-confidence questions should use broader coverage and fallback.
11. Include audio, OCR, subtitles, or metadata only when genuinely required.
12. profile_confidence must be a number between 0.0 and 1.0.
13. If answer choices are provided, use them to disambiguate the intended task.

Return JSON only.
"""


# ============================================================================
# 2. Allowed values
# ============================================================================

ALLOWED_VALUES: Dict[str, set[str]] = {
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
        "ocr_crop",
        "face_crop",
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
    "audio",
    "ocr",
    "subtitle",
    "metadata",
}


# ============================================================================
# 3. Data structures
# ============================================================================

@dataclass(frozen=True)
class ResourceState:
    """
    Runtime serving state.

    Replace these placeholders later with real scheduler and GPU measurements.
    """

    free_gpu_mem_gb: float = 999.0
    encoder_queue_len: int = 0
    vlm_queue_len: int = 0
    load_level: str = "low"  # low | medium | high


@dataclass(frozen=True)
class BudgetConfig:
    name: str

    # Semantic execution strategy.
    reasoning_type: str
    answer_type: str
    coverage_mode: str
    selection_mode: str
    temporal_operation: str
    aggregation_type: str
    identity_requirement: str
    spatial_strategy: str
    required_modalities: Tuple[str, ...]

    # Low-fidelity global probe.
    probe_fps: float
    chunk_len_s: float
    frames_per_chunk: int

    # Candidate selection.
    probe_topk: int | None
    action_topk: int | None
    candidate_threshold: float
    uncertainty_threshold: float

    # High-fidelity refinement.
    window_len_s: float
    high_frames_per_window: int
    high_spatial_tier: str

    # Event consolidation.
    merge_gap_s: float

    # VLM answering budget.
    vlm_budget: int
    quality_tier: str

    # Correctness and fallback behavior.
    fallback_mode: str
    min_temporal_coverage: float
    profile_confidence: float
    miss_risk: str
    answer_sensitivity: str

    # Existing compatibility fields.
    answer_tier: str = "heavy"
    cheap_answer_tier: str = "none"
    max_steps: int = 1
    rationale: str = ""


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
            "candidate_configs": [
                asdict(config)
                for config in self.candidate_configs
            ],
            "requested_config": asdict(
                self.requested_config
            ),
            "chosen_config": asdict(
                self.chosen_config
            ),
            "execution_policy": self.execution_policy,
            "raw_json": self.raw_json,
        }


# ============================================================================
# 4. JSON parsing
# ============================================================================

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse strict JSON, while tolerating accidental surrounding text.
    """

    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "Could not locate a JSON object in profiler output"
            )

        try:
            parsed = json.loads(
                text[start:end + 1]
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Profiler returned malformed JSON"
            ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "Profiler output must be a JSON object"
        )

    return parsed


# ============================================================================
# 5. LLM call
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
        "video_question": query,
        "answer_choices": (
            list(choices)
            if choices is not None
            else None
        ),
    }

    user_prompt = (
        "Profile the following video question.\n"
        + json.dumps(
            question_payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\nReturn only the JSON object specified "
        "by the system prompt."
    )

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": PROFILER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_tokens": 1400,
    }

    headers = {
        "Authorization": (
            f"Bearer {api_key or 'EMPTY'}"
        ),
        "Content-Type": "application/json",
    }

    if verbose:
        print(
            "\n===== PROFILER REQUEST =====",
            flush=True,
        )
        print(
            url,
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
            timeout=timeout_s,
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        response_text = ""

        if (
            getattr(error, "response", None)
            is not None
        ):
            response_text = (
                error.response.text[:2000]
            )

        raise RuntimeError(
            "Profiler request failed. "
            f"URL={url}. "
            f"Response={response_text}"
        ) from error

    try:
        response_json = response.json()
        text = response_json[
            "choices"
        ][0]["message"]["content"]
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        raise RuntimeError(
            "Profiler server returned an unexpected "
            f"response: {response.text[:2000]}"
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
        print(
            "===== END RAW PROFILER OUTPUT =====\n",
            flush=True,
        )

    return _extract_json(text)


# ============================================================================
# 6. Profile validation and coercion
# ============================================================================

DEFAULT_ANALYSIS: Dict[str, Any] = {
    "reasoning_type": "ambiguous",
    "answer_type": "short_text",
    "coverage_requirement": "multi_region",
    "selection_mode": "all_positive_and_uncertain",
    "temporal_requirement": "medium",
    "temporal_operation": "none",
    "candidate_requirement": "medium",
    "context_requirement": "medium",
    "precision_requirement": "medium",
    "aggregation_type": "none",
    "identity_requirement": "none",
    "spatial_strategy": "full_frame",
    "required_modalities": ["visual"],
    "event_density": "unknown",
    "ambiguity": "medium",
    "profile_confidence": 0.5,
    "miss_risk": "medium",
    "answer_sensitivity": "exact",
    "fallback_requirement": "expand_coverage",
    "rationale": "",
}


def _coerce_enum(
    analysis: Dict[str, Any],
    field: str,
) -> None:
    allowed = ALLOWED_VALUES[field]
    value = analysis.get(field)

    if value not in allowed:
        analysis[field] = (
            DEFAULT_ANALYSIS[field]
        )


def _normalize_modalities(
    value: Any,
) -> List[str]:
    if not isinstance(value, list):
        return ["visual"]

    result: List[str] = []

    for modality in value:
        if (
            isinstance(modality, str)
            and modality in ALLOWED_MODALITIES
            and modality not in result
        ):
            result.append(modality)

    if not result:
        result.append("visual")

    return result


def coerce_and_validate_analysis(
    raw_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize the LLM output and enforce hard policy invariants.

    The LLM estimates evidence structure. This function prevents invalid
    combinations from reaching the execution-policy compiler.
    """

    if not isinstance(raw_analysis, dict):
        raw_analysis = {}

    analysis = {
        **DEFAULT_ANALYSIS,
        **raw_analysis,
    }

    for field in ALLOWED_VALUES:
        _coerce_enum(
            analysis,
            field,
        )

    analysis["required_modalities"] = (
        _normalize_modalities(
            analysis.get(
                "required_modalities"
            )
        )
    )

    try:
        confidence = float(
            analysis.get(
                "profile_confidence",
                0.5,
            )
        )
    except (TypeError, ValueError):
        confidence = 0.5

    analysis["profile_confidence"] = max(
        0.0,
        min(1.0, confidence),
    )

    rationale = analysis.get(
        "rationale",
        "",
    )

    if not isinstance(rationale, str):
        rationale = str(rationale)

    analysis["rationale"] = rationale[
        :1000
    ]

    reasoning_type = analysis[
        "reasoning_type"
    ]

    # ------------------------------------------------------------------
    # Counting invariants
    # ------------------------------------------------------------------

    if reasoning_type == "counting":
        analysis["answer_type"] = "count"
        analysis[
            "coverage_requirement"
        ] = "full_timeline"

        analysis[
            "temporal_requirement"
        ] = "global"

        analysis[
            "temporal_operation"
        ] = "frequency"

        analysis[
            "candidate_requirement"
        ] = "many"

        if analysis["selection_mode"] == "top_k":
            analysis[
                "selection_mode"
            ] = "all_positive_and_uncertain"

        if (
            analysis["aggregation_type"]
            == "none"
        ):
            analysis[
                "aggregation_type"
            ] = "occurrences"

        analysis[
            "answer_sensitivity"
        ] = "exact"

        if (
            analysis["fallback_requirement"]
            == "none"
        ):
            analysis[
                "fallback_requirement"
            ] = "expand_coverage"

        analysis["miss_risk"] = (
            "high"
            if analysis[
                "profile_confidence"
            ] < 0.8
            else analysis["miss_risk"]
        )

    # ------------------------------------------------------------------
    # Repetition invariants
    # ------------------------------------------------------------------

    if reasoning_type == "repetition":
        analysis[
            "coverage_requirement"
        ] = "full_timeline"

        analysis[
            "temporal_requirement"
        ] = "global"

        analysis[
            "temporal_operation"
        ] = "frequency"

        if analysis["selection_mode"] == "top_k":
            analysis[
                "selection_mode"
            ] = "all_positive_and_uncertain"

        if (
            analysis["fallback_requirement"]
            == "none"
        ):
            analysis[
                "fallback_requirement"
            ] = "expand_coverage"

    # ------------------------------------------------------------------
    # Fine-detail invariants
    # ------------------------------------------------------------------

    if reasoning_type == "fine_detail":
        analysis[
            "precision_requirement"
        ] = "high"

        if (
            analysis["spatial_strategy"]
            == "full_frame"
        ):
            modalities = set(
                analysis["required_modalities"]
            )

            if "ocr" in modalities:
                analysis[
                    "spatial_strategy"
                ] = "ocr_crop"
            else:
                analysis[
                    "spatial_strategy"
                ] = "object_crop"

        if (
            analysis["fallback_requirement"]
            == "none"
        ):
            analysis[
                "fallback_requirement"
            ] = "expand_coverage"

    # ------------------------------------------------------------------
    # Global-summary invariants
    # ------------------------------------------------------------------

    if reasoning_type == "global_summary":
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
            "answer_type"
        ] = "summary"

    # ------------------------------------------------------------------
    # Temporal and multi-hop invariants
    # ------------------------------------------------------------------

    if reasoning_type == "temporal_order":
        analysis[
            "coverage_requirement"
        ] = "multi_region"

        analysis[
            "selection_mode"
        ] = "multi_event"

        analysis[
            "temporal_operation"
        ] = "ordering"

        analysis[
            "candidate_requirement"
        ] = (
            "medium"
            if analysis[
                "candidate_requirement"
            ] == "few"
            else analysis[
                "candidate_requirement"
            ]
        )

    if reasoning_type == "multi_hop":
        analysis[
            "coverage_requirement"
        ] = "multi_region"

        analysis[
            "selection_mode"
        ] = "multi_event"

        if (
            analysis["temporal_operation"]
            == "none"
        ):
            analysis[
                "temporal_operation"
            ] = "cross_segment_comparison"

    # ------------------------------------------------------------------
    # State-change invariants
    # ------------------------------------------------------------------

    if reasoning_type == "state_change":
        analysis[
            "coverage_requirement"
        ] = "multi_region"

        analysis[
            "selection_mode"
        ] = "beginning_end"

        analysis[
            "temporal_operation"
        ] = "state_transition"

    # ------------------------------------------------------------------
    # Ambiguity and low-confidence invariants
    # ------------------------------------------------------------------

    if (
        reasoning_type == "ambiguous"
        or analysis["ambiguity"] == "high"
        or analysis["profile_confidence"] < 0.55
    ):
        if (
            analysis["coverage_requirement"]
            == "targeted"
        ):
            analysis[
                "coverage_requirement"
            ] = "multi_region"

        if analysis["selection_mode"] == "top_k":
            analysis[
                "selection_mode"
            ] = "all_positive_and_uncertain"

        if (
            analysis["fallback_requirement"]
            == "none"
        ):
            analysis[
                "fallback_requirement"
            ] = "expand_coverage"

        analysis["miss_risk"] = "high"

    # Distinct-identity counting requires tracking.
    if (
        analysis["aggregation_type"]
        == "distinct_entities"
        and analysis[
            "identity_requirement"
        ] == "none"
    ):
        analysis[
            "identity_requirement"
        ] = "cross_window_reidentification"

    # OCR questions require OCR modality.
    if (
        analysis["spatial_strategy"]
        == "ocr_crop"
        and "ocr"
        not in analysis[
            "required_modalities"
        ]
    ):
        analysis[
            "required_modalities"
        ].append("ocr")

    return analysis


# ============================================================================
# 7. Numeric policy helpers
# ============================================================================

def choose_topk(
    requirement: str,
) -> int:
    return {
        "few": 4,
        "medium": 8,
        "many": 16,
    }[requirement]


def choose_window_length(
    context_requirement: str,
    reasoning_type: str,
) -> float:
    base = {
        "short": 4.0,
        "medium": 8.0,
        "long": 16.0,
    }[context_requirement]

    if reasoning_type == "causal":
        return max(base, 16.0)

    if reasoning_type in {
        "temporal_order",
        "multi_hop",
    }:
        return max(base, 12.0)

    if reasoning_type == "counting":
        return max(base, 8.0)

    return base


def choose_vlm_budget(
    precision_requirement: str,
) -> int:
    return {
        "low": 16,
        "medium": 32,
        "high": 64,
    }[precision_requirement]


def choose_chunk_scan(
    *,
    reasoning_type: str,
    duration_s: float,
    event_density: str,
    miss_risk: str,
) -> Tuple[float, int]:
    """
    Return:
        chunk_len_s,
        frames_per_chunk

    Full-timeline tasks preserve coverage by assigning at least one frame
    to every chunk. The resource adapter may lower frames per chunk, but
    it should not silently discard chunks.
    """

    if reasoning_type in {
        "counting",
        "repetition",
    }:
        if event_density == "dense":
            chunk_len_s = 4.0
            frames_per_chunk = 4
        elif event_density == "sparse":
            chunk_len_s = 8.0
            frames_per_chunk = 3
        else:
            chunk_len_s = 8.0
            frames_per_chunk = 3

    elif reasoning_type in {
        "temporal_order",
        "multi_hop",
        "state_change",
    }:
        chunk_len_s = 12.0
        frames_per_chunk = 2

    elif reasoning_type == "global_summary":
        chunk_len_s = 16.0
        frames_per_chunk = 2

    elif reasoning_type == "fine_detail":
        chunk_len_s = 12.0
        frames_per_chunk = 1

    else:
        chunk_len_s = 16.0
        frames_per_chunk = 1

    # Short videos can use denser global probes.
    if duration_s <= 60:
        chunk_len_s = min(
            chunk_len_s,
            4.0,
        )
        frames_per_chunk = max(
            frames_per_chunk,
            2,
        )

    elif duration_s <= 300:
        chunk_len_s = min(
            chunk_len_s,
            8.0,
        )

    # High miss risk should receive denser temporal evidence.
    if miss_risk == "high":
        frames_per_chunk += 1

    return (
        chunk_len_s,
        frames_per_chunk,
    )


def choose_high_frames_per_window(
    *,
    reasoning_type: str,
    precision_requirement: str,
) -> int:
    if reasoning_type == "fine_detail":
        return 24

    if precision_requirement == "high":
        return 16

    if reasoning_type in {
        "counting",
        "repetition",
        "temporal_order",
        "multi_hop",
    }:
        return 12

    return 8


def choose_spatial_tier(
    *,
    precision_requirement: str,
    spatial_strategy: str,
) -> str:
    if spatial_strategy in {
        "ocr_crop",
        "face_crop",
        "object_crop",
        "multi_crop",
    }:
        return "high_crop"

    return {
        "low": "low",
        "medium": "medium",
        "high": "high",
    }[precision_requirement]


def choose_thresholds(
    *,
    reasoning_type: str,
    miss_risk: str,
) -> Tuple[float, float]:
    """
    Return:
        candidate_threshold,
        uncertainty_threshold

    Lower thresholds retain more candidates and improve recall.
    """

    if reasoning_type in {
        "counting",
        "repetition",
    }:
        candidate = 0.35
        uncertainty = 0.15

    elif reasoning_type in {
        "temporal_order",
        "multi_hop",
        "causal",
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
        max(0.0, candidate),
        max(0.0, uncertainty),
    )


def choose_merge_gap(
    reasoning_type: str,
) -> float:
    if reasoning_type in {
        "counting",
        "repetition",
    }:
        return 2.0

    if reasoning_type in {
        "temporal_order",
        "multi_hop",
    }:
        return 1.0

    return 0.5


# ============================================================================
# 8. Compile semantic profile into requested numeric configuration
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

    reasoning_type = analysis[
        "reasoning_type"
    ]

    coverage_mode = analysis[
        "coverage_requirement"
    ]

    selection_mode = analysis[
        "selection_mode"
    ]

    chunk_len_s, frames_per_chunk = (
        choose_chunk_scan(
            reasoning_type=reasoning_type,
            duration_s=duration_s,
            event_density=analysis[
                "event_density"
            ],
            miss_risk=analysis[
                "miss_risk"
            ],
        )
    )

    probe_fps = (
        frames_per_chunk
        / chunk_len_s
    )

    if selection_mode == "top_k":
        topk: int | None = choose_topk(
            analysis[
                "candidate_requirement"
            ]
        )
    else:
        topk = None

    window_len_s = choose_window_length(
        analysis[
            "context_requirement"
        ],
        reasoning_type,
    )

    if reasoning_type in {
        "counting",
        "repetition",
    }:
        window_len_s = max(
            window_len_s,
            chunk_len_s,
        )

    vlm_budget = choose_vlm_budget(
        analysis[
            "precision_requirement"
        ]
    )

    if reasoning_type == "fine_detail":
        vlm_budget = max(
            vlm_budget,
            64,
        )

    if reasoning_type in {
        "counting",
        "repetition",
        "global_summary",
    }:
        vlm_budget = max(
            vlm_budget,
            32,
        )

    candidate_threshold, uncertainty_threshold = (
        choose_thresholds(
            reasoning_type=reasoning_type,
            miss_risk=analysis[
                "miss_risk"
            ],
        )
    )

    high_frames_per_window = (
        choose_high_frames_per_window(
            reasoning_type=reasoning_type,
            precision_requirement=analysis[
                "precision_requirement"
            ],
        )
    )

    high_spatial_tier = choose_spatial_tier(
        precision_requirement=analysis[
            "precision_requirement"
        ],
        spatial_strategy=analysis[
            "spatial_strategy"
        ],
    )

    min_temporal_coverage = (
        1.0
        if coverage_mode == "full_timeline"
        else 0.0
    )

    requested = BudgetConfig(
        name=f"dynamic_{reasoning_type}",
        reasoning_type=reasoning_type,
        answer_type=analysis[
            "answer_type"
        ],
        coverage_mode=coverage_mode,
        selection_mode=selection_mode,
        temporal_operation=analysis[
            "temporal_operation"
        ],
        aggregation_type=analysis[
            "aggregation_type"
        ],
        identity_requirement=analysis[
            "identity_requirement"
        ],
        spatial_strategy=analysis[
            "spatial_strategy"
        ],
        required_modalities=tuple(
            analysis[
                "required_modalities"
            ]
        ),
        probe_fps=probe_fps,
        chunk_len_s=chunk_len_s,
        frames_per_chunk=(
            frames_per_chunk
        ),
        probe_topk=topk,
        action_topk=topk,
        candidate_threshold=(
            candidate_threshold
        ),
        uncertainty_threshold=(
            uncertainty_threshold
        ),
        window_len_s=window_len_s,
        high_frames_per_window=(
            high_frames_per_window
        ),
        high_spatial_tier=(
            high_spatial_tier
        ),
        merge_gap_s=choose_merge_gap(
            reasoning_type
        ),
        vlm_budget=vlm_budget,
        quality_tier="requested",
        fallback_mode=analysis[
            "fallback_requirement"
        ],
        min_temporal_coverage=(
            min_temporal_coverage
        ),
        profile_confidence=analysis[
            "profile_confidence"
        ],
        miss_risk=analysis[
            "miss_risk"
        ],
        answer_sensitivity=analysis[
            "answer_sensitivity"
        ],
        max_steps=(
            2
            if analysis[
                "fallback_requirement"
            ] != "none"
            else 1
        ),
        rationale=analysis.get(
            "rationale",
            "",
        ),
    )

    return calibrate_policy_to_oracle_frontier(
        requested,
        analysis=analysis,
        duration_s=duration_s,
    )


def calibrate_policy_to_oracle_frontier(
    requested: BudgetConfig,
    *,
    analysis: Dict[str, Any],
    duration_s: float,
) -> BudgetConfig:
    """
    Snap dynamic profiler knobs toward the measured fixed-config frontier.

    The consistent oracle audit selects cheap budget2/budget8 policies for
    most short and medium questions, and a sparse scan0.0039_k8_budget32
    policy for very long VRBench-style videos. This keeps semantic profiling
    intact while making numeric knobs match that empirical frontier.
    """

    if duration_s >= 1200:
        scan_fps = 0.00390625
        chunk_len_s = 1.0 / scan_fps
        vlm_budget = 32
        name_suffix = "oracle_sparse_long"

    else:
        scan_fps = 0.015625
        chunk_len_s = 1.0 / scan_fps

        needs_detail_budget = (
            analysis[
                "precision_requirement"
            ] == "high"
            or requested.reasoning_type
            in {
                "fine_detail",
                "counting",
                "repetition",
            }
            or requested.spatial_strategy
            in {
                "ocr_crop",
                "face_crop",
                "object_crop",
                "multi_crop",
            }
        )

        if needs_detail_budget:
            vlm_budget = 16
            name_suffix = "oracle_budget16"
        elif (
            requested.coverage_mode
            == "targeted"
            or requested.miss_risk == "low"
        ):
            vlm_budget = 2
            name_suffix = "oracle_budget2"
        else:
            vlm_budget = 8
            name_suffix = "oracle_budget8"

    topk = (
        8
        if requested.selection_mode == "top_k"
        else requested.probe_topk
    )

    return replace(
        requested,
        name=(
            requested.name
            + "_"
            + name_suffix
        ),
        probe_fps=scan_fps,
        chunk_len_s=chunk_len_s,
        frames_per_chunk=1,
        probe_topk=topk,
        action_topk=topk,
        window_len_s=8.0,
        vlm_budget=vlm_budget,
        high_frames_per_window=min(
            requested.high_frames_per_window,
            max(
                2,
                vlm_budget,
            ),
        ),
        quality_tier="oracle_frontier",
        rationale=(
            requested.rationale
            + " Calibrated to the measured fixed-config oracle frontier."
        ).strip(),
    )


# ============================================================================
# 9. Resource adaptation
# ============================================================================

def _halve_optional_topk(
    value: int | None,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None

    return max(
        minimum,
        value // 2,
    )


def _reduce_budget_without_increase(
    value: int,
    *,
    divisor: int,
    minimum: int,
) -> int:
    return min(
        value,
        max(
            minimum,
            value // divisor,
        ),
    )


def adapt_budget_for_resources(
    requested: BudgetConfig,
    resource_state: ResourceState,
) -> BudgetConfig:
    """
    Reduce cost without violating evidence-coverage invariants.

    Full-timeline tasks degrade spatial/per-chunk fidelity before temporal
    coverage. This is important for counting and repetition.
    """

    load_level = resource_state.load_level

    if load_level not in {
        "low",
        "medium",
        "high",
    }:
        raise ValueError(
            "resource_state.load_level must be "
            "'low', 'medium', or 'high'"
        )

    if load_level == "low":
        return replace(
            requested,
            quality_tier="resource_low",
        )

    full_timeline_task = (
        requested.coverage_mode
        == "full_timeline"
    )

    if full_timeline_task:
        if load_level == "medium":
            new_frames_per_chunk = max(
                2,
                requested.frames_per_chunk - 1,
            )

            return replace(
                requested,
                name=(
                    requested.name
                    + "_resource_medium"
                ),
                frames_per_chunk=(
                    new_frames_per_chunk
                ),
                probe_fps=(
                    new_frames_per_chunk
                    / requested.chunk_len_s
                ),
                high_frames_per_window=max(
                    6,
                    requested.high_frames_per_window
                    // 2,
                ),
                vlm_budget=(
                    _reduce_budget_without_increase(
                        requested.vlm_budget,
                        divisor=2,
                        minimum=16,
                    )
                ),
                quality_tier=(
                    "resource_medium"
                ),
            )

        # High-load full-timeline policy:
        # retain every chunk, but inspect fewer frames per chunk.
        return replace(
            requested,
            name=(
                requested.name
                + "_resource_high"
            ),
            frames_per_chunk=1,
            probe_fps=(
                1.0
                / requested.chunk_len_s
            ),
            high_frames_per_window=max(
                4,
                requested.high_frames_per_window
                // 3,
            ),
            vlm_budget=(
                _reduce_budget_without_increase(
                    requested.vlm_budget,
                    divisor=4,
                    minimum=8,
                )
            ),
            fallback_mode=(
                "expand_coverage"
                if requested.fallback_mode
                == "none"
                else requested.fallback_mode
            ),
            quality_tier="resource_high",
        )

    # Targeted and multi-region requests can reduce top-k under load.
    if load_level == "medium":
        return replace(
            requested,
            name=(
                requested.name
                + "_resource_medium"
            ),
            probe_topk=(
                _halve_optional_topk(
                    requested.probe_topk,
                    minimum=4,
                )
            ),
            action_topk=(
                _halve_optional_topk(
                    requested.action_topk,
                    minimum=4,
                )
            ),
            high_frames_per_window=max(
                6,
                requested.high_frames_per_window
                // 2,
            ),
            vlm_budget=(
                _reduce_budget_without_increase(
                    requested.vlm_budget,
                    divisor=2,
                    minimum=16,
                )
            ),
            quality_tier=(
                "resource_medium"
            ),
        )

    new_frames_per_chunk = max(
        1,
        requested.frames_per_chunk - 1,
    )

    return replace(
        requested,
        name=(
            requested.name
            + "_resource_high"
        ),
        frames_per_chunk=(
            new_frames_per_chunk
        ),
        probe_fps=(
            new_frames_per_chunk
            / requested.chunk_len_s
        ),
        probe_topk=(
            _halve_optional_topk(
                requested.probe_topk,
                minimum=2,
            )
        ),
        action_topk=(
            _halve_optional_topk(
                requested.action_topk,
                minimum=2,
            )
        ),
        high_frames_per_window=max(
            4,
            requested.high_frames_per_window
            // 3,
        ),
        vlm_budget=(
            _reduce_budget_without_increase(
                requested.vlm_budget,
                divisor=4,
                minimum=8,
            )
        ),
        fallback_mode=(
            "expand_coverage"
            if requested.fallback_mode
            == "none"
            else requested.fallback_mode
        ),
        quality_tier="resource_high",
    )


# ============================================================================
# 10. Cost estimation and candidate selection
# ============================================================================

def estimate_config_cost(
    config: BudgetConfig,
    *,
    duration_s: float,
) -> float:
    """
    Approximate execution cost in normalized work units.

    This is intentionally simple. Replace its constants with fitted latency
    or token-cost models once you have profiler traces.
    """

    if duration_s <= 0:
        raise ValueError(
            "duration_s must be positive"
        )

    chunk_count = math.ceil(
        duration_s
        / config.chunk_len_s
    )

    if config.coverage_mode == "full_timeline":
        low_frames = (
            chunk_count
            * config.frames_per_chunk
        )
    else:
        low_frames = min(
            duration_s
            * config.probe_fps,
            chunk_count
            * config.frames_per_chunk,
        )

    if config.action_topk is not None:
        estimated_high_windows = (
            config.action_topk
        )

    elif config.selection_mode == "uniform":
        estimated_high_windows = (
            chunk_count
        )

    elif config.selection_mode in {
        "all_positive",
        "all_positive_and_uncertain",
    }:
        # Placeholder candidate-rate estimate.
        estimated_high_windows = max(
            1,
            math.ceil(0.25 * chunk_count),
        )

    elif config.selection_mode == "beginning_end":
        estimated_high_windows = 2

    elif config.selection_mode == "multi_event":
        estimated_high_windows = min(
            8,
            chunk_count,
        )

    else:
        estimated_high_windows = min(
            4,
            chunk_count,
        )

    high_frames = (
        estimated_high_windows
        * config.high_frames_per_window
    )

    modality_multiplier = 1.0

    if "audio" in config.required_modalities:
        modality_multiplier += 0.20

    if "ocr" in config.required_modalities:
        modality_multiplier += 0.25

    if config.spatial_strategy != "full_frame":
        modality_multiplier += 0.20

    answer_multiplier = max(
        0.25,
        config.vlm_budget / 32.0,
    )

    return float(
        modality_multiplier
        * (
            low_frames
            + 4.0 * high_frames
            + 16.0 * answer_multiplier
        )
    )


def _cost_budget_for_resource_state(
    resource_state: ResourceState,
) -> float:
    """
    Temporary normalized cost ceiling.

    Tune these values from actual measurements.
    """

    base = {
        "low": 100_000.0,
        "medium": 50_000.0,
        "high": 20_000.0,
    }.get(resource_state.load_level)

    if base is None:
        raise ValueError(
            "Invalid resource load level"
        )

    memory_factor = max(
        0.25,
        min(
            2.0,
            resource_state.free_gpu_mem_gb
            / 24.0,
        ),
    )

    queue_penalty = (
        1.0
        + 0.05
        * resource_state.encoder_queue_len
        + 0.05
        * resource_state.vlm_queue_len
    )

    return (
        base
        * memory_factor
        / queue_penalty
    )


def choose_config_for_current_resources(
    candidates: List[BudgetConfig],
    *,
    resource_state: ResourceState,
    duration_s: float,
) -> BudgetConfig:
    """
    Choose the richest candidate that fits the current normalized budget.
    """

    if not candidates:
        raise ValueError(
            "No candidate workflow configurations provided"
        )

    budget = _cost_budget_for_resource_state(
        resource_state
    )

    scored = [
        (
            estimate_config_cost(
                config,
                duration_s=duration_s,
            ),
            config,
        )
        for config in candidates
    ]

    fitting = [
        item
        for item in scored
        if item[0] <= budget
    ]

    if fitting:
        # Highest-cost candidate that still fits.
        return max(
            fitting,
            key=lambda item: item[0],
        )[1]

    # Nothing fits: choose the least costly safe candidate.
    return min(
        scored,
        key=lambda item: item[0],
    )[1]


# ============================================================================
# 11. Convert configuration into an orchestrator policy
# ============================================================================

def budget_to_policy(
    config: BudgetConfig,
) -> Dict[str, Any]:
    return {
        # Semantic execution strategy.
        "reasoning_type": (
            config.reasoning_type
        ),
        "answer_type": (
            config.answer_type
        ),
        "coverage_mode": (
            config.coverage_mode
        ),
        "selection_mode": (
            config.selection_mode
        ),
        "temporal_operation": (
            config.temporal_operation
        ),
        "aggregation_type": (
            config.aggregation_type
        ),
        "identity_requirement": (
            config.identity_requirement
        ),
        "spatial_strategy": (
            config.spatial_strategy
        ),
        "required_modalities": list(
            config.required_modalities
        ),

        # Low-fidelity scan.
        "probe_fps": config.probe_fps,
        "chunk_len_s": config.chunk_len_s,
        "frames_per_chunk": (
            config.frames_per_chunk
        ),

        # Candidate selection.
        "probe_topk": config.probe_topk,
        "action_topk": (
            config.action_topk
        ),
        "candidate_threshold": (
            config.candidate_threshold
        ),
        "uncertainty_threshold": (
            config.uncertainty_threshold
        ),

        # High-fidelity refinement.
        "window_len_s": (
            config.window_len_s
        ),
        "high_frames_per_window": (
            config.high_frames_per_window
        ),
        "high_spatial_tier": (
            config.high_spatial_tier
        ),

        # Event aggregation.
        "merge_gap_s": config.merge_gap_s,

        # Answer-stage compatibility fields.
        "answer_max_images_total": (
            config.vlm_budget
        ),
        "answer_max_frames_per_window": (
            config.high_frames_per_window
        ),
        "answer_tier": config.answer_tier,
        "cheap_answer_tier": (
            config.cheap_answer_tier
        ),
        "max_steps": config.max_steps,

        # Correctness and fallback.
        "fallback_mode": (
            config.fallback_mode
        ),
        "min_temporal_coverage": (
            config.min_temporal_coverage
        ),
        "profile_confidence": (
            config.profile_confidence
        ),
        "miss_risk": config.miss_risk,
        "answer_sensitivity": (
            config.answer_sensitivity
        ),
        "quality_tier": (
            config.quality_tier
        ),
        "rationale": config.rationale,
    }


# ============================================================================
# 12. Main entry point
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
    """
    Profile a VideoQA query and compile a resource-aware execution plan.

    Flow:

        question
        -> semantic evidence profile
        -> validated/coerced profile
        -> requested BudgetConfig
        -> safe resource-adapted candidates
        -> selected BudgetConfig
        -> executable orchestrator policy
    """

    if not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    if duration_s <= 0:
        raise ValueError(
            "duration_s must be positive"
        )

    if resource_state is None:
        resource_state = ResourceState()

    raw_json = _call_profiler_llm(
        query=query,
        choices=choices,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        api_key=api_key,
        verbose=verbose,
    )

    raw_analysis = raw_json.get(
        "analysis",
        {},
    )

    analysis = coerce_and_validate_analysis(
        raw_analysis
    )

    requested_config = compile_execution_policy(
        analysis=analysis,
        duration_s=duration_s,
    )

    low_load_config = adapt_budget_for_resources(
        requested_config,
        ResourceState(
            free_gpu_mem_gb=max(
                resource_state.free_gpu_mem_gb,
                24.0,
            ),
            load_level="low",
        ),
    )

    medium_load_config = adapt_budget_for_resources(
        requested_config,
        ResourceState(
            free_gpu_mem_gb=(
                resource_state.free_gpu_mem_gb
            ),
            encoder_queue_len=(
                resource_state.encoder_queue_len
            ),
            vlm_queue_len=(
                resource_state.vlm_queue_len
            ),
            load_level="medium",
        ),
    )

    high_load_config = adapt_budget_for_resources(
        requested_config,
        ResourceState(
            free_gpu_mem_gb=(
                resource_state.free_gpu_mem_gb
            ),
            encoder_queue_len=(
                resource_state.encoder_queue_len
            ),
            vlm_queue_len=(
                resource_state.vlm_queue_len
            ),
            load_level="high",
        ),
    )

    candidate_configs = [
        low_load_config,
        medium_load_config,
        high_load_config,
    ]

    # Remove exact duplicate configurations.
    unique_candidates: List[
        BudgetConfig
    ] = []

    seen: set[str] = set()

    for config in candidate_configs:
        serialized = json.dumps(
            asdict(config),
            sort_keys=True,
        )

        if serialized not in seen:
            unique_candidates.append(
                config
            )
            seen.add(serialized)

    candidate_configs = unique_candidates

    # Since candidates already correspond to load tiers, directly adapting
    # to the current load gives predictable behavior. The best-fit selector
    # remains available for future scheduler-driven selection.
    chosen_config = adapt_budget_for_resources(
        requested_config,
        resource_state,
    )

    policy = budget_to_policy(
        chosen_config
    )

    if verbose:
        print(
            "\n===== VALIDATED PROFILER ANALYSIS =====",
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
            "\n===== REQUESTED CONFIG =====",
            flush=True,
        )
        print(
            json.dumps(
                asdict(requested_config),
                indent=2,
            ),
            flush=True,
        )

        print(
            "\n===== CANDIDATE CONFIGS =====",
            flush=True,
        )
        print(
            json.dumps(
                [
                    {
                        **asdict(config),
                        "estimated_cost": (
                            estimate_config_cost(
                                config,
                                duration_s=duration_s,
                            )
                        ),
                    }
                    for config in candidate_configs
                ],
                indent=2,
            ),
            flush=True,
        )

        print(
            "\n===== RESOURCE STATE =====",
            flush=True,
        )
        print(
            json.dumps(
                asdict(resource_state),
                indent=2,
            ),
            flush=True,
        )

        print(
            "\n===== CHOSEN CONFIG =====",
            flush=True,
        )
        print(
            json.dumps(
                asdict(chosen_config),
                indent=2,
            ),
            flush=True,
        )

        print(
            "\n===== COMPILED EXECUTION POLICY =====",
            flush=True,
        )
        print(
            json.dumps(
                policy,
                indent=2,
            ),
            flush=True,
        )

    return ProfilerResult(
        analysis=analysis,
        candidate_configs=(
            candidate_configs
        ),
        requested_config=(
            requested_config
        ),
        chosen_config=chosen_config,
        execution_policy=policy,
        raw_json=raw_json,
    )


# ============================================================================
# 13. Backward-compatible wrapper
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compatibility wrapper for callers expecting:

        policy, analysis = profile_query_llm_legacy(...)
    """

    result = profile_query_llm(
        query=query,
        duration_s=duration_s,
        choices=choices,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        api_key=api_key,
    )

    return (
        result.execution_policy,
        result.analysis,
    )


# ============================================================================
# 14. Optional standalone test
# ============================================================================

if __name__ == "__main__":
    example_question = (
        "How many times does the man enter the room?"
    )

    example_choices = [
        "One time",
        "Two times",
        "Three times",
        "Four times",
    ]

    result = profile_query_llm(
        query=example_question,
        choices=example_choices,
        duration_s=3600.0,
        base_url="http://localhost:8000/v1",
        model="gpt-4o-mini",
        resource_state=ResourceState(
            free_gpu_mem_gb=24.0,
            encoder_queue_len=0,
            vlm_queue_len=0,
            load_level="low",
        ),
        verbose=True,
    )

    print(
        "\n===== FINAL RESULT =====",
        flush=True,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
