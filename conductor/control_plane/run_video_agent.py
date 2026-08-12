from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import time

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from PIL import Image
from decord.video_reader import VideoReader
from openai import OpenAI
from transformers import AutoModel, AutoProcessor

from conductor.profiler.llm_profiler import (
    ResourceState,
    profile_query_adaptive,
)


# ============================================================
# Runtime leakage protection
# ============================================================

FORBIDDEN_RUNTIME_KEYS = {
    "gold",
    "correct",
    "answer_label",
    "answer_idx",
    "oracle_correct",
    "oracle_selected_config",
    "execution_tier",
    "tier",
    "num_correct_configs",
    "num_configs_evaluated",
}


# ============================================================
# Runtime actions
# ============================================================

VALID_ACTIONS = {
    "SEARCH_LOCAL",
    "SEARCH_BEFORE",
    "SEARCH_AFTER",
    "GLOBAL_SCAN",
    "COUNT_EVENTS",
    "INCREASE_DENSITY",
    "VERIFY_DETAIL",
    "IMAGE_CAPTION",
    "ZOOM_CAPTION",
    "OBJECT_DETECTION",
    "OBJECT_TRACKING",
    "ANSWER",
}


PLAN_ACTIONS = {
    "SEARCH_LOCAL",
    "SEARCH_BEFORE",
    "SEARCH_AFTER",
    "GLOBAL_SCAN",
    "COUNT_EVENTS",
    "INCREASE_DENSITY",
    "VERIFY_DETAIL",
    "IMAGE_CAPTION",
    "ZOOM_CAPTION",
    "OBJECT_DETECTION",
    "OBJECT_TRACKING",
}


# ============================================================
# State
# ============================================================

@dataclass
class Evidence:
    action: str
    query: str
    start_s: float
    end_s: float
    timestamps: list[float]
    observation: str
    confidence: str
    latency_s: float
    confidence_score: float | None = None
    uncertainty: float | None = None
    tool_name: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class AgentState:
    question: str
    duration_s: float
    policy: dict[str, Any]

    evidence: list[Evidence] = field(
        default_factory=list
    )

    memory_bank: dict[str, Any] = field(
        default_factory=dict
    )

    retrieval_plan: dict[str, Any] | None = None
    retrieval_plans: list[dict[str, Any]] = field(
        default_factory=list
    )

    answer_confidence: float = 0.0
    answer_assessments: list[dict[str, Any]] = field(
        default_factory=list
    )

    tool_uncertainties: list[dict[str, Any]] = field(
        default_factory=list
    )

    final_answer: str | None = None
    supported_answer: str | None = None
    supported_answer_reason: str | None = None
    total_latency_s: float = 0.0

    local_search_count: int = 0
    global_scan_count: int = 0
    count_scan_count: int = 0
    temporal_search_count: int = 0
    density_count: int = 0
    verify_count: int = 0
    image_caption_count: int = 0
    zoom_caption_count: int = 0
    detection_count: int = 0
    tracking_count: int = 0

    active_evidence_id: int | None = None


# ============================================================
# Utilities
# ============================================================

def load_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize_answer(value):
    if value is None:
        return ""

    value = str(value).strip()

    match = re.search(
        r"\b([A-H])\b",
        value.upper(),
    )

    if match:
        return match.group(1)

    return value


def valid_choice_labels(
    choices: list[str],
):
    return {
        chr(ord("A") + index)
        for index in range(
            len(choices)
        )
    }


def format_choices(
    choices: list[str],
):
    if not choices:
        return "No choices provided."

    return "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice
        in enumerate(choices)
    )


FINE_DETAIL_TERMS = (
    "wearing",
    "holding",
    "hand",
    "gesture",
    "screen",
    "subtitle",
    "appeared",
    "changed",
    "change occurs",
    "color",
    "colour",
    "object",
    "doing with",
    "what is he doing",
    "what is she doing",
    "what happened",
    "what was done",
)


SEQUENCE_TERMS = (
    "sequence",
    "order",
    "ordered",
    "before",
    "after",
    "then",
    "following events",
    "following scenes",
    "workflow",
    "steps",
    "stages",
)


RECURRENCE_TERMS = (
    "first appeared",
    "first appear",
    "appeared first",
    "appeared afterward",
    "appears afterward",
    "appears after",
    "has appeared",
    "have appeared",
    "appeared in",
    "appeared",
    "first plant",
    "first time",
    "in which scene",
    "in which of the following scenes",
    "which character appears afterward",
    "where has",
    "where did",
)


def is_fine_detail_question(
    question: str,
):
    text = str(question or "").lower()
    return any(
        term in text
        for term in FINE_DETAIL_TERMS
    )


def is_sequence_question(
    question: str,
):
    text = str(question or "").lower()
    return any(
        term in text
        for term in SEQUENCE_TERMS
    )


def is_recurrence_question(
    question: str,
):
    text = str(question or "").lower()
    return any(
        term in text
        for term in RECURRENCE_TERMS
    )


def neutral_profiler_result(
    *,
    execution_mode="agentic",
):
    policy = {
        "execution_mode": execution_mode,
        "answer_type": "multiple_choice",
        "selection_mode": "uniform",
        "required_modalities": ["visual"],
        "probe_fps": 0.05,
        "chunk_len_s": 8.0,
        "frames_per_chunk": 4,
        "probe_topk": 8,
        "action_topk": 8,
        "candidate_threshold": 0.0,
        "uncertainty_threshold": 0.5,
        "window_len_s": 16.0,
        "high_frames_per_window": 12,
        "high_spatial_tier": "medium",
        "merge_gap_s": 4.0,
        "vlm_budget": 999,
        "quality_tier": "adaptive",
        "fallback_mode": "agent_plan",
        "min_temporal_coverage": 0.0,
        "expand_neighbors": True,
        "preserve_order": True,
        "include_uniform_anchors": True,
        "max_steps": 999,
        "max_local_searches": 999,
        "max_global_scans": 999,
        "max_count_scans": 999,
        "max_temporal_searches": 999,
        "max_density_refinements": 999,
        "max_contrastive_checks": 999,
        "answer_tier": "heavy",
        "cheap_answer_tier": "none",
        "rationale": "neutral no-profiler policy",
    }

    router_decision = SimpleNamespace(
        policy_name="no_profiler",
        execution_mode=execution_mode,
        evidence_requirement="none",
        temporal_relation="none",
        confidence=1.0,
        reason="Profiler disabled; using neutral policy.",
        out_of_distribution=False,
    )

    return SimpleNamespace(
        source="disabled",
        router_decision=router_decision,
        chosen_config=SimpleNamespace(
            name="no_profiler",
        ),
        execution_policy=policy,
        llm_result=None,
    )


def choice_discriminator_query(
    *,
    question: str,
    choices: list[str],
):
    return (
        "Resolve the fine-grained visual discriminator for this "
        "multiple-choice question. Inspect which option-specific "
        "visual action, gesture, object interaction, person identity, "
        "scene identity, or ordered event is explicitly visible. "
        "Question: "
        f"{question} Choices: {format_choices(choices)}"
    )


def labels_mentioned_in_text(
    text,
):
    labels = set()

    for match in re.finditer(
        r"\b(?:option|choice)\s+([A-Z])\b",
        str(text),
        flags=re.IGNORECASE,
    ):
        labels.add(
            match.group(1).upper()
        )

    return labels


def supported_labels_mentioned_in_text(
    text,
):
    labels = set()

    pattern = (
        r"\b(?:aligns with|matches|corresponds to|supports|"
        r"best supports|is consistent with)\s+"
        r"(?:(?:the\s+)?(?:answer|option|choice)\s+"
        r"['\"]?([A-Z])(?:\b|\.)|['\"]([A-Z])\.)"
    )

    for match in re.finditer(
        pattern,
        str(text),
        flags=re.IGNORECASE,
    ):
        labels.add(
            (
                match.group(1)
                or match.group(2)
            ).upper()
        )

    return labels


UNCERTAIN_SUPPORT_PHRASES = (
    "no direct evidence",
    "no explicit evidence",
    "not directly supported",
    "does not directly support",
    "does not provide direct evidence",
    "does not provide clear evidence",
    "insufficient evidence",
    "insufficient visual support",
    "unclear",
    "cannot determine",
    "could be",
    "might be",
    "may be",
    "plausible",
    "suggests",
    "inferred",
    "not enough information",
)


CHOICE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "he",
    "her",
    "him",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "one",
    "she",
    "that",
    "the",
    "their",
    "them",
    "then",
    "they",
    "to",
    "was",
    "with",
}


def normalized_word_tokens(
    text,
):
    tokens = []

    for token in re.findall(
        r"[a-z0-9]+",
        str(text).lower(),
    ):
        if token in CHOICE_STOPWORDS:
            continue

        for suffix in (
            "ing",
            "ed",
            "es",
            "s",
        ):
            if (
                len(token) > len(suffix) + 3
                and token.endswith(suffix)
            ):
                token = token[
                    : -len(suffix)
                ]
                break

        tokens.append(
            token
        )

    return tokens


def token_is_present(
    token,
    text_tokens,
):
    if token in text_tokens:
        return True

    prefix = token[:5]

    return any(
        other.startswith(prefix)
        or token.startswith(other[:5])
        for other in text_tokens
        if len(other) >= 4
    )


def choice_text_supported_by_text(
    choice,
    text,
):
    choice_tokens = normalized_word_tokens(
        choice
    )

    if not choice_tokens:
        return False

    text_tokens = normalized_word_tokens(
        text
    )

    hits = sum(
        1
        for token in set(
            choice_tokens
        )
        if token_is_present(
            token,
            text_tokens,
        )
    )

    required = 1
    if len(
        set(choice_tokens)
    ) >= 3:
        required = 2

    return hits >= required


def text_has_uncertain_support(
    text,
):
    lowered = str(text).lower()
    return any(
        phrase in lowered
        for phrase in UNCERTAIN_SUPPORT_PHRASES
    )


def is_why_question(
    question,
):
    return str(
        question
    ).strip().lower().startswith(
        "why "
    )


def is_after_question(
    question,
):
    lowered = str(
        question
    ).lower()

    return (
        " after " in lowered
        or lowered.startswith(
            "after "
        )
    )


def infer_question_skills(
    question,
):
    lowered = str(
        question
    ).lower()

    skills = set()

    if is_after_question(
        question
    ):
        skills.add(
            "TEMPORAL_AFTER"
        )

    if (
        " before " in lowered
        or lowered.startswith(
            "before "
        )
        or " prior to " in lowered
    ):
        skills.add(
            "TEMPORAL_BEFORE"
        )

    explicit_sequence = (
        "sequence" in lowered
        or "order" in lowered
        or "first" in lowered
        and (
            "then" in lowered
            or "finally" in lowered
        )
    )

    if (
        explicit_sequence
        or (
            is_sequence_question(
                question
            )
            and not (
                skills
                & {
                    "TEMPORAL_AFTER",
                    "TEMPORAL_BEFORE",
                }
            )
        )
    ):
        skills.add(
            "SEQUENCE_ORDER"
        )

    if is_recurrence_question(
        question
    ):
        skills.add(
            "RECURRENCE"
        )

    if is_why_question(
        question
    ):
        skills.add(
            "CAUSAL_WHY"
        )

    if (
        "how many" in lowered
        or "count" in lowered
        or "number of" in lowered
    ):
        skills.add(
            "COUNTING"
        )

    if is_fine_detail_question(
        question
    ):
        skills.add(
            "LOCAL_DETAIL"
        )

    if not skills:
        skills.add(
            "GLOBAL_SUMMARY"
        )

    return skills


def truncate_text(
    value,
    *,
    max_chars=500,
):
    text = str(
        value
        if value is not None
        else ""
    )

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + " ...[truncated]"
    )


def compact_evidence_payload(
    evidence: list[Evidence],
    *,
    max_items=48,
    max_observation_chars=800,
):
    items = []

    start = max(
        0,
        len(evidence)
        - max_items,
    )

    for index, item in enumerate(
        evidence[start:],
        start=start,
    ):
        items.append({
            "evidence_id":
                index,

            "action":
                item.action,

            "query":
                truncate_text(
                    item.query,
                    max_chars=300,
                ),

            "range": [
                item.start_s,
                item.end_s,
            ],

            "observation":
                truncate_text(
                    item.observation,
                    max_chars=max_observation_chars,
                ),

            "confidence":
                item.confidence,

            "confidence_score":
                item.confidence_score,

            "uncertainty":
                item.uncertainty,
        })

    return items


def compact_memory_for_prompt(
    state: AgentState,
    *,
    max_segments=None,
    max_tool_results=48,
    max_caption_chars=500,
):
    memory = state.memory_bank

    segments = (
        memory.get(
            "segment_captions",
            [],
        )
        or []
    )

    if max_segments is None:
        max_segments = int(
            state.policy.get(
                "max_prompt_memory_segments",
                64,
            )
            or 64
        )

    max_segments = max(
        1,
        int(
            max_segments
        ),
    )

    selected_ids = set()

    for value in memory.get(
        "summary_relevant_segments",
        [],
    ) or []:
        try:
            selected_ids.add(
                int(value)
            )
        except (TypeError, ValueError):
            pass

    if len(segments) > max_segments:
        uniform = np.linspace(
            0,
            len(segments) - 1,
            max_segments,
            dtype=int,
        )

        selected_ids.update(
            int(index)
            for index in uniform
        )

        selected_ids = set(
            sorted(
                selected_ids
            )[:max_segments]
        )

    else:
        selected_ids.update(
            range(
                len(segments)
            )
        )

    compact_segments = []

    for index, segment in enumerate(
        segments
    ):
        if index not in selected_ids:
            continue

        compact_segments.append({
            "segment_id":
                segment.get(
                    "segment_id"
                ),

            "range": [
                segment.get(
                    "start_s"
                ),
                segment.get(
                    "end_s"
                ),
            ],

            "caption":
                truncate_text(
                    segment.get(
                        "caption",
                        "",
                    ),
                    max_chars=max_caption_chars,
                ),

            "confidence_score":
                segment.get(
                    "confidence_score"
                ),
        })

    tool_results = (
        memory.get(
            "tool_results",
            [],
        )
        or []
    )

    compact_tools = []

    for item in tool_results[
        -max_tool_results:
    ]:
        compact_tools.append({
            "evidence_id":
                item.get(
                    "evidence_id"
                ),

            "tool":
                item.get(
                    "tool"
                ),

            "action":
                item.get(
                    "action"
                ),

            "range":
                item.get(
                    "range"
                ),

            "observation":
                truncate_text(
                    item.get(
                        "observation",
                        "",
                    ),
                    max_chars=800,
                ),

            "confidence_score":
                item.get(
                    "confidence_score"
                ),
        })

    return {
        "caption_model":
            memory.get(
                "caption_model"
            ),

        "caption_cache_hit":
            memory.get(
                "caption_cache_hit"
            ),

        "context_coverage":
            memory.get(
                "context_coverage"
            ),

        "num_total_segments":
            memory.get(
                "num_total_segments"
            ),

        "num_context_segments":
            memory.get(
                "num_context_segments",
                len(segments),
            ),

        "num_segments_in_prompt":
            len(
                compact_segments
            ),

        "video_summary":
            truncate_text(
                memory.get(
                    "video_summary",
                    "",
                ),
                max_chars=3000,
            ),

        "summary_confidence_score":
            memory.get(
                "summary_confidence_score"
            ),

        "summary_relevant_segments":
            memory.get(
                "summary_relevant_segments",
                [],
            ),

        "segment_captions":
            compact_segments,

        "tool_results":
            compact_tools,
    }


def extract_json(text):
    if text is None:
        return None

    text = str(text).strip()

    if text.startswith("```"):
        text = text.strip("`")

        if text.lower().startswith(
            "json"
        ):
            text = text[4:].strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        try:
            return json.loads(
                text[
                    start:
                    end + 1
                ]
            )

        except Exception:
            pass

    return None


def image_to_data_uri(
    image: Image.Image,
    *,
    max_side: int = 512,
):
    image = image.convert("RGB")

    width, height = image.size

    scale = min(
        1.0,
        max_side / max(width, height),
    )

    if scale < 1.0:
        image = image.resize(
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def normalize_query(value):
    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def same_range(
    a_start,
    a_end,
    b_start,
    b_end,
    tolerance_s=1.0,
):
    return (
        abs(a_start - b_start)
        <= tolerance_s
        and
        abs(a_end - b_end)
        <= tolerance_s
    )


def confidence_to_score(
    value,
):
    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    text = str(
        value
    ).strip().lower()

    if text in {
        "high",
        "certain",
    }:
        return 0.85

    if text in {
        "medium",
        "moderate",
    }:
        return 0.60

    if text in {
        "low",
        "uncertain",
    }:
        return 0.30

    try:
        numeric = float(
            text
        )
    except ValueError:
        return None

    if numeric > 1.0:
        numeric = numeric / 5.0

    return max(
        0.0,
        min(
            1.0,
            numeric,
        ),
    )


def score_to_uncertainty(
    value,
):
    score = confidence_to_score(
        value
    )

    if score is None:
        return None

    return max(
        0.0,
        min(
            1.0,
            1.0 - score,
        ),
    )


def assert_no_forbidden_keys(
    obj,
    path="runtime_input",
):
    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            if key in FORBIDDEN_RUNTIME_KEYS:
                raise RuntimeError(
                    "Oracle leakage detected: "
                    f"{path}.{key}"
                )

            assert_no_forbidden_keys(
                value,
                f"{path}.{key}",
            )

    elif isinstance(
        obj,
        list,
    ):

        for index, value in enumerate(
            obj
        ):
            assert_no_forbidden_keys(
                value,
                f"{path}[{index}]",
            )


def sanitize_policy(
    policy: dict[str, Any],
):
    """
    Normalize the profiler/runtime action contract.

    The current profiler uses COMPARE_CHOICES in a few policies.
    Evidence acquisition should not inspect choices, so convert
    that to VERIFY_DETAIL.
    """

    policy = dict(
        policy
        or {}
    )

    allowed = []

    for action in policy.get(
        "allowed_actions",
        [],
    ):
        action = str(
            action
        ).upper()

        if action == "COMPARE_CHOICES":
            action = "VERIFY_DETAIL"

        if (
            action in VALID_ACTIONS
            and action not in allowed
        ):
            allowed.append(
                action
            )

    if "ANSWER" not in allowed:
        allowed.append(
            "ANSWER"
        )

    policy[
        "allowed_actions"
    ] = allowed

    return policy


def action_counts(
    state: AgentState,
):
    return {
        "SEARCH_LOCAL":
            state.local_search_count,

        "GLOBAL_SCAN":
            state.global_scan_count,

        "COUNT_EVENTS":
            state.count_scan_count,

        "TEMPORAL_SEARCH":
            state.temporal_search_count,

        "INCREASE_DENSITY":
            state.density_count,

        "VERIFY_DETAIL":
            state.verify_count,

        "IMAGE_CAPTION":
            state.image_caption_count,

        "ZOOM_CAPTION":
            state.zoom_caption_count,

        "OBJECT_DETECTION":
            state.detection_count,

        "OBJECT_TRACKING":
            state.tracking_count,
    }


# ============================================================
# Temporal anchor extraction
# ============================================================

def derive_temporal_anchor_query(
    question: str,
    relation: str,
):
    """
    SigLIP should retrieve the anchor EVENT, not the complete
    relational question.

    Example:

        What happens after the man opens the door?

    should retrieve:

        the man opens the door

    and only THEN move forward in time.
    """

    text = str(
        question
    ).strip()

    relation = str(
        relation
    ).lower()

    if relation == "after":

        patterns = (
            r"\bafter\b\s+(.+?)[?.!]*$",
            r"\bfollowing\b\s+(.+?)[?.!]*$",
        )

    elif relation == "before":

        patterns = (
            r"\bbefore\b\s+(.+?)[?.!]*$",
            r"\bprior to\b\s+(.+?)[?.!]*$",
        )

    else:
        return text

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            anchor = (
                match.group(1)
                .strip()
            )

            if anchor:
                return anchor

    return text


# ============================================================
# Video backend
# ============================================================

class VideoBackend:

    def __init__(
        self,
        path: str,
    ):
        self.path = path

        self.vr = VideoReader(
            path
        )

        self.fps = float(
            self.vr.get_avg_fps()
        )

        self.num_frames = len(
            self.vr
        )

        self.duration_s = (
            self.num_frames
            / self.fps
        )


    def frame_at(
        self,
        timestamp_s: float,
    ):
        timestamp_s = max(
            0.0,
            min(
                float(
                    timestamp_s
                ),
                self.duration_s
                - 1.0 / self.fps,
            ),
        )

        index = int(
            round(
                timestamp_s
                * self.fps
            )
        )

        index = max(
            0,
            min(
                index,
                self.num_frames - 1,
            ),
        )

        array = (
            self.vr[index]
            .asnumpy()
        )

        return Image.fromarray(
            array
        )


    def sample_range(
        self,
        start_s: float,
        end_s: float,
        num_frames: int,
    ):
        start_s = max(
            0.0,
            float(start_s),
        )

        end_s = min(
            self.duration_s,
            float(end_s),
        )

        if end_s <= start_s:

            end_s = min(
                self.duration_s,
                start_s + 1.0,
            )

        num_frames = max(
            1,
            int(num_frames),
        )

        timestamps = np.linspace(
            start_s,
            end_s,
            num_frames,
        )

        frames = [
            self.frame_at(
                float(timestamp)
            )
            for timestamp
            in timestamps
        ]

        return (
            [
                float(timestamp)
                for timestamp
                in timestamps
            ],
            frames,
        )


class UltralyticsObjectTools:

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        conf: float = 0.25,
        max_detections: int = 20,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Object tools require ultralytics. Install with: "
                "pip install ultralytics"
            ) from exc

        self.model_name = model_name
        self.device = device
        self.conf = float(conf)
        self.max_detections = int(max_detections)

        print(
            "Loading object detector:",
            model_name,
            "on",
            device,
        )

        self.model = YOLO(
            model_name
        )


    def _target_terms(
        self,
        target,
    ):
        text = str(
            target
            or ""
        ).lower()

        terms = set(
            re.findall(
                r"[a-z0-9]+",
                text,
            )
        )

        stop = {
            "the",
            "a",
            "an",
            "this",
            "that",
            "object",
            "person",
            "track",
            "detect",
            "find",
            "where",
            "what",
            "which",
            "with",
            "and",
            "for",
            "in",
            "on",
            "of",
            "to",
            "is",
            "are",
        }

        return {
            term
            for term in terms
            if term not in stop
        }


    def _result_items(
        self,
        *,
        result,
        timestamp,
        target_terms,
        include_track_id=False,
    ):
        names = (
            result.names
            if hasattr(
                result,
                "names",
            )
            else {}
        )

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:
            return []

        items = []

        for index, box in enumerate(
            boxes
        ):
            if (
                len(items)
                >= self.max_detections
            ):
                break

            try:
                cls_id = int(
                    box.cls.item()
                )
            except Exception:
                cls_id = -1

            name = str(
                names.get(
                    cls_id,
                    cls_id,
                )
            )

            if target_terms:
                name_terms = set(
                    re.findall(
                        r"[a-z0-9]+",
                        name.lower(),
                    )
                )

                if not (
                    target_terms
                    & name_terms
                ):
                    continue

            try:
                conf = float(
                    box.conf.item()
                )
            except Exception:
                conf = 0.0

            try:
                xyxy = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )
            except Exception:
                xyxy = []

            item = {
                "timestamp_s":
                    float(timestamp),

                "label":
                    name,

                "bbox_xyxy":
                    [
                        round(
                            float(value),
                            2,
                        )
                        for value in xyxy
                    ],

                "confidence":
                    conf,
            }

            if include_track_id:
                track_id = getattr(
                    box,
                    "id",
                    None,
                )

                if track_id is not None:
                    try:
                        item[
                            "track_id"
                        ] = int(
                            track_id.item()
                        )
                    except Exception:
                        pass

            items.append(
                item
            )

        return items


    def detect(
        self,
        *,
        frames,
        timestamps,
        target=None,
    ):
        target_terms = self._target_terms(
            target
        )

        arrays = [
            np.asarray(
                frame
            )
            for frame in frames
        ]

        t0 = time.time()

        results = self.model.predict(
            source=
                arrays,

            device=
                self.device,

            conf=
                self.conf,

            verbose=
                False,
        )

        latency = (
            time.time()
            - t0
        )

        detections = []

        for timestamp, result in zip(
            timestamps,
            results,
        ):
            detections.extend(
                self._result_items(
                    result=
                        result,

                    timestamp=
                        timestamp,

                    target_terms=
                        target_terms,
                )
            )

        best_conf = max(
            [
                item[
                    "confidence"
                ]
                for item in detections
            ]
            or [
                0.0,
            ]
        )

        return {
            "detections":
                detections,

            "confidence_score":
                best_conf,

            "latency_s":
                latency,
        }


    def track(
        self,
        *,
        frames,
        timestamps,
        target=None,
    ):
        target_terms = self._target_terms(
            target
        )

        arrays = [
            np.asarray(
                frame
            )
            for frame in frames
        ]

        t0 = time.time()

        try:
            results = self.model.track(
                source=
                    arrays,

                device=
                    self.device,

                conf=
                    self.conf,

                persist=
                    True,

                verbose=
                    False,
            )
        except Exception:
            fallback = self.detect(
                frames=
                    frames,

                timestamps=
                    timestamps,

                target=
                    target,
            )

            fallback[
                "tracking_fallback"
            ] = True

            return fallback

        latency = (
            time.time()
            - t0
        )

        tracks = []

        for timestamp, result in zip(
            timestamps,
            results,
        ):
            tracks.extend(
                self._result_items(
                    result=
                        result,

                    timestamp=
                        timestamp,

                    target_terms=
                        target_terms,

                    include_track_id=
                        True,
                )
            )

        best_conf = max(
            [
                item[
                    "confidence"
                ]
                for item in tracks
            ]
            or [
                0.0,
            ]
        )

        return {
            "tracks":
                tracks,

            "confidence_score":
                best_conf,

            "latency_s":
                latency,
        }


# ============================================================
# SigLIP retriever
# ============================================================

class SigLIPRetriever:

    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        self.device = device

        print(
            "Loading SigLIP:",
            model_name,
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModel
            .from_pretrained(
                model_name
            )
            .eval()
            .to(
                device
            )
        )


    @torch.no_grad()
    def rank_frames(
        self,
        *,
        query: str,
        timestamps: list[float],
        frames: list[Image.Image],
        topk: int,
        batch_size: int = 4,
    ):
        if not frames:
            return []

        all_scores = []

        for start in range(
            0,
            len(frames),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(frames),
            )

            batch_frames = (
                frames[
                    start:
                    end
                ]
            )

            text_config = getattr(
                self.model.config,
                "text_config",
                None,
            )

            max_length = getattr(
                text_config,
                "max_position_embeddings",
                64,
            )

            inputs = self.processor(
                text=[
                    query
                ] * len(batch_frames),

                images=
                    batch_frames,

                padding=
                    "max_length",

                truncation=
                    True,

                max_length=
                    max_length,

                return_tensors=
                    "pt",
            )

            inputs = {
                key:
                    value.to(
                        self.device
                    )

                for key, value
                in inputs.items()
            }

            outputs = self.model(
                **inputs
            )

            if hasattr(
                outputs,
                "logits_per_image",
            ):
                matrix = (
                    outputs
                    .logits_per_image
                )

            elif hasattr(
                outputs,
                "logits_per_text",
            ):
                matrix = (
                    outputs
                    .logits_per_text
                )

            else:
                raise RuntimeError(
                    "SigLIP output has no "
                    "logits_per_image/text"
                )

            scores = (
                matrix.diagonal()
            )

            all_scores.extend(
                scores
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            del inputs
            del outputs
            del scores
            del matrix

            if str(
                self.device
            ).startswith(
                "cuda"
            ):
                torch.cuda.empty_cache()

        scores = np.asarray(
            all_scores,
            dtype=np.float32,
        )

        order = np.argsort(
            -scores
        )

        return [
            {
                "timestamp_s":
                    float(
                        timestamps[index]
                    ),

                "score":
                    float(
                        scores[index]
                    ),
            }

            for index
            in order[
                : min(
                    int(topk),
                    len(order),
                )
            ]
        ]


# ============================================================
# VLM evidence inspector
# ============================================================

class EvidenceInspector:

    def __init__(
        self,
        client: OpenAI,
        model: str,
        caption_model: str | None = None,
        caption_client: OpenAI | None = None,
        caption_prompt_style: str = "videoagent2",
    ):
        self.client = client
        self.model = model
        self.caption_model = (
            caption_model
            or model
        )
        self.caption_client = (
            caption_client
            or client
        )
        self.caption_prompt_style = str(
            caption_prompt_style
            or "videoagent2"
        ).lower()


    def caption_frames(
        self,
        *,
        question: str,
        timestamps: list[float],
        frames: list[Image.Image],
        purpose: str,
    ):
        if self.caption_prompt_style == "generic":
            caption_instruction = f"""
You caption chronological video frames for a long-video QA
agent memory bank.

Question:
{question}

Purpose:
{purpose}

Describe visible people, objects, actions, temporal order, and
state changes. Keep it concise but specific.

For uncertainty-aware reasoning, return a numeric confidence from
0.0 to 1.0 for the caption as a whole.

Do not use answer choices.
"""
        else:
            caption_instruction = f"""
You are a dense segment-level video captioner for a long-video
QA memory bank, similar to the general-context captioner used
in VideoAgent-style systems.

Question:
{question}

Purpose:
{purpose}

Caption this short chronological segment using only visible
evidence. Preserve fine details that multiple-choice questions
often depend on.

Write one compact chronological caption that includes:
- main person identity labels if visible, such as C or another
  person
- exact objects/tools/materials being touched, held, used, moved,
  opened, placed, cleaned, wiped, poured, cut, painted, molded, or
  inspected
- hand actions and object interactions, not just broad activity
- scene/text/screen/laptop/canvas/cloth/mold/tool changes if visible
- temporal order across the supplied frames
- state changes in objects or the environment
- uncertainty only when a detail is unclear

Avoid vague captions such as "C interacts with an object" when the
object/action can be named. Do not infer motives, outcomes, or answer
choices that are not visually supported.

Return confidence based on how visually grounded the caption is.
"""

        content = [{
            "type":
                "text",

            "text":
                f"""
{caption_instruction}

Return ONLY JSON:

{{
  "caption":
      "chronological caption",

  "confidence_score":
      0.75
}}
"""
        }]

        for timestamp, frame in zip(
            timestamps,
            frames,
        ):
            content.append({
                "type":
                    "text",

                "text":
                    f"Frame at {timestamp:.2f}s",
            })

            content.append({
                "type":
                    "image_url",

                "image_url": {
                    "url":
                        image_to_data_uri(
                            frame
                        )
                },
            })

        t0 = time.time()

        response = (
            self.caption_client
            .chat.completions
            .create(
                model=
                    self.caption_model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        content,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        raw = (
            response
            .choices[0]
            .message.content
        )

        parsed = extract_json(
            raw
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return {
                "caption":
                    str(raw),

                "confidence_score":
                    0.3,

                "latency_s":
                    latency,
            }

        score = confidence_to_score(
            parsed.get(
                "confidence_score",
                parsed.get(
                    "confidence",
                    0.5,
                ),
            )
        )

        if score is None:
            score = 0.5

        return {
            "caption":
                str(
                    parsed.get(
                        "caption",
                        parsed.get(
                            "observation",
                            "",
                        ),
                    )
                ),

            "confidence_score":
                score,

            "latency_s":
                latency,
        }


    def summarize_context(
        self,
        *,
        question: str,
        segment_captions: list[dict],
        max_segments_per_batch=40,
    ):
        if not segment_captions:
            return {
                "summary": "",
                "relevant_segments": [],
                "confidence_score": 0.0,
                "latency_s": 0.0,
            }

        partial_summaries = []
        relevant_segments = []
        total_latency = 0.0

        for start in range(
            0,
            len(segment_captions),
            max_segments_per_batch,
        ):
            batch = segment_captions[
                start:
                start + max_segments_per_batch
            ]

            payload = []

            for item in batch:
                payload.append({
                    "segment_id":
                        item.get(
                            "segment_id"
                        ),

                    "range": [
                        item.get(
                            "start_s"
                        ),
                        item.get(
                            "end_s"
                        ),
                    ],

                    "caption":
                        truncate_text(
                            item.get(
                                "caption",
                                "",
                            ),
                            max_chars=500,
                        ),

                    "confidence_score":
                        item.get(
                            "confidence_score"
                        ),
                })

            prompt = f"""
Summarize this chronological portion of a long video for a
VideoQA memory bank.

Question:
{question}

Segment captions:
{json.dumps(payload, ensure_ascii=False)}

Preserve major events, people, objects, actions, temporal order,
and details potentially relevant to the question. Do not answer
the multiple-choice question.

Return ONLY JSON:

{{
  "summary": "chronological partial summary",
  "relevant_segments": [0, 3],
  "confidence_score": 0.75
}}
"""

            t0 = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=512,
            )

            total_latency += time.time() - t0

            parsed = extract_json(
                response.choices[0].message.content
            )

            if isinstance(parsed, dict):
                partial_summaries.append(
                    str(
                        parsed.get(
                            "summary",
                            "",
                        )
                    )
                )

                values = parsed.get(
                    "relevant_segments",
                    [],
                )

                if isinstance(
                    values,
                    list,
                ):
                    for value in values:
                        try:
                            relevant_segments.append(
                                int(value)
                            )
                        except (TypeError, ValueError):
                            pass

            else:
                partial_summaries.append(
                    str(
                        response.choices[0].message.content
                    )
                )

        final_prompt = f"""
Combine these chronological partial video summaries for a
VideoQA memory bank.

Question:
{question}

Partial summaries:
{json.dumps(partial_summaries, ensure_ascii=False)}

Preserve chronological order and question-relevant details.
Do not answer the multiple-choice question.

Return ONLY JSON:

{{
  "summary": "chronological whole-video summary",
  "relevant_segments": [0, 3],
  "confidence_score": 0.75
}}
"""

        t0 = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": final_prompt,
                }
            ],
            temperature=0,
            max_tokens=768,
        )

        total_latency += time.time() - t0

        parsed = extract_json(
            response.choices[0].message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return {
                "summary":
                    str(
                        response.choices[0].message.content
                    ),

                "relevant_segments":
                    sorted(
                        set(
                            relevant_segments
                        )
                    ),

                "confidence_score":
                    0.5,

                "latency_s":
                    total_latency,
            }

        final_relevant = parsed.get(
            "relevant_segments",
            relevant_segments,
        )

        if not isinstance(
            final_relevant,
            list,
        ):
            final_relevant = relevant_segments

        normalized_relevant = []

        for value in final_relevant:
            try:
                normalized_relevant.append(
                    int(value)
                )
            except (TypeError, ValueError):
                pass

        score = confidence_to_score(
            parsed.get(
                "confidence_score",
                parsed.get(
                    "confidence",
                    0.5,
                ),
            )
        )

        if score is None:
            score = 0.5

        return {
            "summary":
                str(
                    parsed.get(
                        "summary",
                        "",
                    )
                ),

            "relevant_segments":
                sorted(
                    set(
                        normalized_relevant
                        or relevant_segments
                    )
                ),

            "confidence_score":
                score,

            "latency_s":
                total_latency,
        }
    
    def inspect(
        self,
        *,
        question: str,
        timestamps: list[float],
        frames: list[Image.Image],
        action: str,
        query: str,
    ):
        if action in {
            "GLOBAL_SCAN",
            "GLOBAL_SCAN_CHUNK",
        }:

            instruction = """
            These frames were sampled chronologically from one portion of
            the video.

            Recover the visible progression of events, not merely the
            single dominant activity.

            For each distinct visible stage, describe:

            - what the main person is doing
            - what object, tool, or material is involved
            - any visible state change
            - how the activity differs from the preceding visible stage

            Preserve chronological order.

            For workflow or sequence questions, explicitly identify
            multiple distinct stages when the frames support them.

            For example, prefer:

            "First C prepares the material, then works on it, and later
            inspects or modifies the result."

            over:

            "C is creating art."

            Do NOT collapse distinct chronological stages into one generic
            description.

            Do not invent intermediate events.

            Do not claim that something is the first or last occurrence
            unless the supplied evidence establishes that.
            """

        elif action == "SEARCH_BEFORE":

            instruction = """
These frames occur immediately BEFORE an already localized
anchor interval.

Determine what visibly occurs before the anchor.

Do not describe the anchor itself as the answer unless the
frames establish that it also occurs in this interval.
"""

        elif action == "SEARCH_AFTER":

            instruction = """
                These frames occur immediately AFTER an already localized
                anchor interval.

                Determine what visibly occurs after the anchor.

                Do not describe the anchor itself as the answer unless the
                frames establish that it also occurs in this interval.
                """
        elif action in {"INCREASE_DENSITY",}:
            instruction = f"""
                This is a higher-density inspection of an already selected
                temporal interval.

                The unresolved visual fact is:

                {query}

                Focus specifically on resolving that fact.

                Inspect fine temporal transitions, gestures, hand shape,
                hand motion, identity, object interaction, state changes,
                tools, pointing direction, and other visually discriminating
                details.

                Do not simply repeat the coarse observation if the denser
                frames establish additional information.
                """
        elif action == "VERIFY_DETAIL":
                instruction = f"""
            This is a fine-grained verification step.

            The coarse event has ALREADY been localized.
            Do NOT merely repeat the coarse event.

            Unresolved visual fact:
            {query}

            Inspect the supplied chronological frames carefully.

            Determine the SPECIFIC fine-grained visual attribute needed
            to resolve the unresolved fact.

            For hand/gesture questions, explicitly inspect:

            - which hand
            - finger configuration
            - number of visibly extended fingers when discernible
            - palm orientation
            - pointing direction
            - waving vs pointing vs counting vs grasping vs resting
            - motion across consecutive frames
            - interaction with nearby objects
            - identity of the person

            Distinguish "raising the hand" from what the raised hand
            is actually doing.

            Do not return "raising the hand" when the unresolved fact asks
            for the gesture performed with that hand.

            If the detail cannot be established from these frames, explicitly
            say that it is unresolved.
            """
        else:

            instruction = """
                These frames were selected by semantic retrieval.

                Determine whether the requested anchor event, object, person,
                or action is actually visible.

                Semantic similarity is only candidate localization. It does
                not establish before/after/first/last semantics by itself.
                """

        content = [{
            "type":
                "text",

            "text":
                f"""
You inspect VIDEO evidence for a VideoQA system.

Question:
{question}

Evidence query:
{query}

Action:
{action}

{instruction}

RULES:

1. Describe only what the supplied images visually establish.

2. Images are shown in chronological order.

3. Distinguish:
   - visible
   - looking at
   - touching
   - holding
   - interacting with
   - acting on

4. Preserve person identity when possible.

5. Do not infer temporal relations that the supplied frames do
   not establish.

6. If answer choices appear in the evidence query, use them only
   as visual hypotheses to inspect. Do not treat a choice as evidence;
   report only what the supplied frames visibly establish.

Return ONLY JSON:

{{
  "observation":
      "precise factual evidence",
  "confidence":
      "medium"
}}
"""
        }]

        for timestamp, frame in zip(
            timestamps,
            frames,
        ):

            content.append({
                "type":
                    "text",

                "text":
                    f"Frame at "
                    f"{timestamp:.2f}s",
            })

            content.append({
                "type":
                    "image_url",

                "image_url": {
                    "url":
                        image_to_data_uri(
                            frame
                        )
                },
            })

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        content,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        text = (
            response
            .choices[0]
            .message.content
            .strip()
        )

        parsed = extract_json(
            text
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return {
                "observation":
                    text,

                "confidence":
                    "low",

                "confidence_score":
                    0.30,

                "latency_s":
                    latency,
            }

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "observation":
                str(
                    parsed.get(
                        "observation",
                        "",
                    )
                ),

            "confidence":
                confidence,

            "confidence_score":
                confidence_to_score(
                    parsed.get(
                        "confidence_score",
                        confidence,
                    )
                ),

            "latency_s":
                latency,
        }


    def aggregate_evidence(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        purpose: str,
    ):
        payload = []

        for index, item in enumerate(
            evidence
        ):

            payload.append({
                "evidence_id":
                    index,

                "action":
                    item.action,

                "range": [
                    item.start_s,
                    item.end_s,
                ],

                "observation":
                    item.observation,

                "confidence":
                    item.confidence,
            })

        prompt = f"""
Combine chronological VIDEO observations.

Question:
{question}

Purpose:
{purpose}

Evidence:
{json.dumps(
    payload,
    ensure_ascii=False,
)}

RULES:

1. Preserve chronology.

2. Do not invent events.

3. Merge redundant observations.

4. Keep temporal relations explicit.

5. Do not use answer choices.

Return ONLY JSON:

{{
  "observation":
      "combined factual evidence",
  "confidence":
      "medium"
}}
"""

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return {
                "observation":
                    str(
                        response
                        .choices[0]
                        .message.content
                    ),

                "confidence":
                    "low",

                "latency_s":
                    latency,
            }

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "observation":
                str(
                    parsed.get(
                        "observation",
                        "",
                    )
                ),

            "confidence":
                confidence,

            "latency_s":
                latency,
        }


    def inspect_count_chunk(
        self,
        *,
        question: str,
        timestamps: list[float],
        frames: list[Image.Image],
    ):
        content = [{
            "type":
                "text",

            "text":
                f"""
You are scanning ONE chronological chunk of a video for a
COUNTING question.

Question:
{question}

Determine whether this chunk contains one or more
count-relevant occurrences.

Do not count each sampled frame separately.

Adjacent frames showing one continuous event are one
occurrence.

Return ONLY JSON:

{{
  "candidate":
      true,

  "description":
      "description of count-relevant occurrence",

  "confidence":
      "medium"
}}

If no occurrence is established:

{{
  "candidate":
      false,

  "description":
      "",

  "confidence":
      "low"
}}
"""
        }]

        for timestamp, frame in zip(
            timestamps,
            frames,
        ):

            content.append({
                "type":
                    "text",

                "text":
                    f"Frame at "
                    f"{timestamp:.2f}s",
            })

            content.append({
                "type":
                    "image_url",

                "image_url": {
                    "url":
                        image_to_data_uri(
                            frame
                        )
                },
            })

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        content,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return {
                "candidate":
                    False,

                "description":
                    "",

                "confidence":
                    "low",

                "latency_s":
                    latency,
            }

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "candidate":
                parsed.get(
                    "candidate"
                ) is True,

            "description":
                str(
                    parsed.get(
                        "description",
                        "",
                    )
                ),

            "confidence":
                confidence,

            "latency_s":
                latency,
        }


    def aggregate_count(
        self,
        *,
        question: str,
        candidates: list[dict],
    ):
        prompt = f"""
Deduplicate chronological candidate events for a VIDEO
counting question.

Question:
{question}

Candidates:
{json.dumps(
    candidates,
    ensure_ascii=False,
)}

RULES:

1. Merge adjacent candidates belonging to the same event.

2. Do not double-count one event appearing in neighboring
   windows.

3. Count only visually supported occurrences.

4. Do not use answer choices.

Return ONLY JSON:

{{
  "observation":
      "deduplicated counting evidence",

  "estimated_count":
      0,

  "confidence":
      "medium"
}}
"""

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return {
                "observation":
                    str(
                        response
                        .choices[0]
                        .message.content
                    ),

                "estimated_count":
                    None,

                "confidence":
                    "low",

                "latency_s":
                    latency,
            }

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "observation":
                str(
                    parsed.get(
                        "observation",
                        "",
                    )
                ),

            "estimated_count":
                parsed.get(
                    "estimated_count"
                ),

            "confidence":
                confidence,

            "latency_s":
                latency,
        }


# ============================================================
# Controller
# ============================================================

class VideoAgentController:

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ):
        self.client = client
        self.model = model


    def retrieval_coverage(
        self,
        state: AgentState,
    ):
        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        evidence_type = str(
            state.policy.get(
                "evidence_type",
                "",
            )
        )

        needs_fine_detail = (
            evidence_type in {
                "local_detail",
                "generic_local_mcq",
            }
            or is_fine_detail_question(
                state.question
            )
        )

        needs_sequence_detail = (
            evidence_type == "sequence"
            or (
                requirement == "global"
                and relation == "ordering"
            )
            or is_sequence_question(
                state.question
            )
        )

        min_sequence_segments = int(
            state.policy.get(
                "min_sequence_segments",
                3,
            )
            or 3
        )

        tool_budget_profile = str(
            state.policy.get(
                "tool_budget_profile",
                "adaptive",
            )
        ).lower()

        if tool_budget_profile == "summary_light":
            min_sequence_segments = min(
                min_sequence_segments,
                1,
            )

        elif tool_budget_profile == "strict":
            min_sequence_segments = max(
                min_sequence_segments,
                3,
            )

        has_retrieval_evidence = any(
            evidence.action != "GENERAL_CONTEXT"
            for evidence in state.evidence
        )

        has_global_scan = any(
            evidence.action == "GLOBAL_SCAN"
            for evidence in state.evidence
        )

        has_count_scan = any(
            evidence.action == "COUNT_EVENTS"
            for evidence in state.evidence
        )

        local_ids = [
            index
            for index, evidence
            in enumerate(
                state.evidence
            )
            if evidence.action == "SEARCH_LOCAL"
        ]

        has_fine_detail = any(
            evidence.action in {
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
                "IMAGE_CAPTION",
                "OBJECT_DETECTION",
                "OBJECT_TRACKING",
            }
            for evidence in state.evidence
        )

        sequence_detail_count = sum(
            1
            for evidence in state.evidence
            if evidence.action in {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }
        )

        return {
            "requirement":
                requirement,

            "relation":
                relation,

            "evidence_type":
                evidence_type,

            "needs_fine_detail":
                needs_fine_detail,

            "needs_sequence_detail":
                needs_sequence_detail,

            "min_sequence_segments":
                min_sequence_segments,

            "tool_budget_profile":
                tool_budget_profile,

            "has_retrieval_evidence":
                has_retrieval_evidence,

            "has_global_scan":
                has_global_scan,

            "has_count_scan":
                has_count_scan,

            "local_ids":
                local_ids,

            "has_localization":
                bool(
                    local_ids
                ),

            "has_fine_detail":
                has_fine_detail,

            "sequence_detail_count":
                sequence_detail_count,
        }


    def coverage_gap(
        self,
        state: AgentState,
    ):
        coverage = self.retrieval_coverage(
            state
        )

        policy_mode = str(
            state.policy.get(
                "profiler_policy_mode",
                "hint",
            )
        ).lower()

        if policy_mode != "hard":
            return None

        if not coverage[
            "has_retrieval_evidence"
        ]:
            if (
                coverage["requirement"] == "count"
            ):
                return {
                    "tool": "COUNT_EVENTS",
                    "query": state.question,
                    "reason": "summary is only initial memory; count evidence is required",
                }

            if (
                coverage["requirement"] == "global"
                or coverage["needs_sequence_detail"]
            ):
                return {
                    "tool": "GLOBAL_SCAN",
                    "query": state.question,
                    "reason": "summary is only initial memory; global evidence is required",
                }

            return {
                "tool": "SEARCH_LOCAL",
                "query": state.question,
                "reason": "summary is only initial memory; localized evidence is required",
            }

        if (
            coverage["requirement"] == "count"
            and not coverage["has_count_scan"]
        ):
            return {
                "tool": "COUNT_EVENTS",
                "query": state.question,
                "reason": "count question requires explicit count scan",
            }

        if (
            (
                coverage["requirement"] == "global"
                or coverage["needs_sequence_detail"]
            )
            and not coverage["has_global_scan"]
        ):
            return {
                "tool": "GLOBAL_SCAN",
                "query": state.question,
                "reason": "global/sequence question requires chronological scan",
            }

        if (
            coverage["needs_sequence_detail"]
            and coverage["sequence_detail_count"]
            < coverage["min_sequence_segments"]
        ):
            return {
                "tool": "IMAGE_CAPTION",
                "query": (
                    "caption the next distinct chronological segment "
                    "to verify ordered workflow details"
                ),
                "reason": (
                    "sequence question requires multiple distinct "
                    "post-summary tool captions"
                ),
            }

        if (
            coverage["needs_fine_detail"]
            and not coverage["has_localization"]
        ):
            return {
                "tool": "SEARCH_LOCAL",
                "query": state.question,
                "reason": "local-detail question needs localization before verification",
            }

        if (
            coverage["needs_fine_detail"]
            and coverage["has_localization"]
            and not coverage["has_fine_detail"]
        ):
            return {
                "tool": "VERIFY_DETAIL",
                "query": state.question,
                "evidence_id": coverage["local_ids"][0],
                "reason": "localized coarse setup needs fine visual verification",
            }

        return None


    def assess_answer(
        self,
        *,
        state: AgentState,
        choices: list[str],
        confidence_threshold: float,
    ):
        evidence_payload = compact_evidence_payload(
            state.evidence,
        )

        memory = compact_memory_for_prompt(
            state,
        )

        prompt = f"""
Assess whether current long-video memory is sufficient to answer
the multiple-choice question.

Question:
{state.question}

Choices:
{format_choices(choices)}

Memory bank:
{json.dumps(memory, ensure_ascii=False)}

Evidence:
{json.dumps(evidence_payload, ensure_ascii=False)}

Confidence threshold:
{confidence_threshold:.2f}

RULES:

1. Use only memory/evidence collected from video tools.

2. Return an answer candidate and numeric confidence on a 0 to 5
   scale, where 5 means fully sufficient visual support.

3. Treat high tool uncertainty as a reason to gather more evidence.

4. If confidence is below the threshold, explain the missing visual
   information needed for the next retrieval plan.

5. Do not mark sufficient unless the selected choice is directly
   supported by visual evidence or the video memory.

6. For local_detail or generic_local_mcq questions, coarse
   localization is not enough. If multiple choices share the same
   visible setup, confidence must stay below threshold until evidence
   resolves the discriminating detail such as gesture direction,
   exact hand action, object interaction, clothing, person identity,
   or scene identity.

7. Full-video segment memory may be sufficient when it explicitly
   supports the selected choice's distinguishing visual claims. Ask for
   more evidence only when memory/evidence does not resolve the visual
   difference between choices.

8. Do not treat absence of evidence for other choices as positive
   evidence for one choice. The selected choice's unique visual claim
   must itself be explicitly supported.

9. If sufficient is true, include support_evidence_ids containing the
   evidence_id values that directly support the selected choice's
   unique visual claims. If support comes from segment memory, include
   support_segment_ids. Do not cite evidence that does not mention the
   claimed object/action/order.

Return ONLY JSON:

{{
  "answer": "",
  "confidence": 0.0,
  "sufficient": false,
  "support_evidence_ids": [],
  "support_segment_ids": [],
  "missing_information": "",
  "reason": ""
}}
"""

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return (
                {
                    "answer": "",
                    "confidence": 0.0,
                    "sufficient": False,
                    "missing_information":
                        "answer assessment parse failure",
                    "reason":
                        str(
                            response
                            .choices[0]
                            .message.content
                        ),
                },
                latency,
            )

        valid = valid_choice_labels(
            choices
        )

        answer = normalize_answer(
            parsed.get(
                "answer",
                "",
            )
        )

        if answer not in valid:
            answer = ""

        try:
            confidence = float(
                parsed.get(
                    "confidence",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                5.0,
                confidence,
            ),
        )

        parsed[
            "answer"
        ] = answer

        parsed[
            "confidence"
        ] = confidence

        evidence_type = str(
            state.policy.get(
                "evidence_type",
                "",
            )
        )

        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        needs_fine_detail = (
            evidence_type in {
                "local_detail",
                "generic_local_mcq",
            }
            or is_fine_detail_question(
                state.question
            )
        )

        needs_sequence_detail = (
            evidence_type == "sequence"
            or (
                requirement == "global"
                and relation == "ordering"
            )
            or is_sequence_question(
                state.question
            )
        )

        discriminator_query = choice_discriminator_query(
            question=
                state.question,

            choices=
                choices,
        )

        coverage_gap = self.coverage_gap(
            state
        )

        has_localization = any(
            evidence.action == "SEARCH_LOCAL"
            for evidence in state.evidence
        )

        has_global_scan = any(
            evidence.action == "GLOBAL_SCAN"
            for evidence in state.evidence
        )

        has_retrieval_evidence = any(
            evidence.action != "GENERAL_CONTEXT"
            for evidence in state.evidence
        )

        has_fine_detail = any(
            evidence.action in {
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
                "IMAGE_CAPTION",
                "OBJECT_DETECTION",
                "OBJECT_TRACKING",
            }
            for evidence in state.evidence
        )

        has_sequence_detail = any(
            evidence.action in {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }
            for evidence in state.evidence
        )

        if str(
            parsed.get(
                "missing_information",
                "",
            )
        ).strip().lower() in {
            "specific unresolved visual fact",
            "remaining fact",
        }:
            parsed[
                "missing_information"
            ] = (
                "the next distinct chronological stage or "
                "question-relevant visual fact not yet covered"
            )

        missing_text = str(
            parsed.get(
                "missing_information",
                "",
            )
        ).strip().lower()

        if (
            needs_fine_detail
            and (
                "fine-grained visual discriminator" in missing_text
                or "fine grained visual discriminator" in missing_text
                or "among the answer choices" in missing_text
                or "among choices" in missing_text
            )
        ):
            parsed[
                "missing_information"
            ] = discriminator_query

        if str(
            parsed.get(
                "reason",
                "",
            )
        ).strip().lower() == "brief evidence-grounded assessment":
            parsed[
                "reason"
            ] = ""

        if coverage_gap is not None:
            parsed[
                "sufficient"
            ] = False

            parsed[
                "confidence"
            ] = min(
                confidence,
                max(
                    0.0,
                    confidence_threshold - 1.0,
                ),
            )

            confidence = float(
                parsed[
                    "confidence"
                ]
            )

            parsed[
                "missing_information"
            ] = str(
                coverage_gap.get(
                    "query",
                    coverage_gap.get(
                        "reason",
                        "",
                    ),
                )
            )

        reason_text = str(
            parsed.get(
                "reason",
                "",
            )
        )

        mentioned_labels = (
            supported_labels_mentioned_in_text(
                reason_text
            )
            & valid
        )

        if (
            answer
            and mentioned_labels
            and answer not in mentioned_labels
        ):
            parsed[
                "sufficient"
            ] = False

            parsed[
                "confidence"
            ] = min(
                float(
                    parsed.get(
                        "confidence",
                        0.0,
                    )
                ),
                max(
                    0.0,
                    confidence_threshold - 1.0,
                ),
            )

            confidence = float(
                parsed[
                    "confidence"
                ]
            )

            parsed[
                "missing_information"
            ] = (
                "the assessment answer label conflicts with its "
                "own evidence-grounded reason; verify the choice "
                "whose text is explicitly supported"
            )

        unsupported_reason = (
            "no direct evidence" in reason_text.lower()
            or "does not provide direct evidence" in reason_text.lower()
            or "there is no evidence" in reason_text.lower()
        )

        if unsupported_reason:
            parsed[
                "sufficient"
            ] = False

            parsed[
                "confidence"
            ] = min(
                float(
                    parsed.get(
                        "confidence",
                        0.0,
                    )
                ),
                max(
                    0.0,
                    confidence_threshold - 1.0,
                ),
            )

            confidence = float(
                parsed[
                    "confidence"
                ]
            )

            if not parsed.get(
                "missing_information"
            ):
                parsed[
                    "missing_information"
                ] = discriminator_query

        if (
            needs_fine_detail
            and has_localization
            and not has_fine_detail
        ):
            parsed[
                "sufficient"
            ] = False

            parsed[
                "confidence"
            ] = min(
                confidence,
                max(
                    0.0,
                    confidence_threshold - 1.0,
                ),
            )

            confidence = float(
                parsed[
                    "confidence"
                ]
            )

            parsed[
                "missing_information"
            ] = discriminator_query

        if (
            needs_fine_detail
            and confidence < 5.0
        ):
            parsed[
                "sufficient"
            ] = False

            parsed[
                "confidence"
            ] = min(
                float(
                    parsed.get(
                        "confidence",
                        0.0,
                    )
                ),
                4.0,
            )

            confidence = float(
                parsed[
                    "confidence"
                ]
            )

            if not parsed.get(
                "missing_information"
            ):
                parsed[
                    "missing_information"
                ] = discriminator_query

        if (
            needs_sequence_detail
            and has_global_scan
            and not has_sequence_detail
        ):
            parsed[
                "sufficient"
            ] = False

            parsed[
                "confidence"
            ] = min(
                float(
                    parsed.get(
                        "confidence",
                        0.0,
                    )
                ),
                max(
                    0.0,
                    confidence_threshold - 1.0,
                ),
            )

            confidence = float(
                parsed[
                    "confidence"
                ]
            )

            parsed[
                "missing_information"
            ] = (
                "the exact ordered workflow details that distinguish "
                "the answer choices"
            )

        support_evidence_ids = []

        for value in parsed.get(
            "support_evidence_ids",
            [],
        ) or []:
            try:
                index = int(
                    value
                )
            except (TypeError, ValueError):
                continue

            if (
                0 <= index < len(
                    state.evidence
                )
            ):
                support_evidence_ids.append(
                    index
                )

        support_segment_ids = []

        segments = (
            state.memory_bank.get(
                "segment_captions",
                [],
            )
            or []
        )

        valid_segment_ids = {
            int(
                segment.get(
                    "segment_id",
                    -1,
                )
            )
            for segment in segments
        }

        for value in parsed.get(
            "support_segment_ids",
            [],
        ) or []:
            try:
                segment_id = int(
                    value
                )
            except (TypeError, ValueError):
                continue

            if segment_id in valid_segment_ids:
                support_segment_ids.append(
                    segment_id
                )

        parsed[
            "support_evidence_ids"
        ] = sorted(
            set(
                support_evidence_ids
            )
        )

        parsed[
            "support_segment_ids"
        ] = sorted(
            set(
                support_segment_ids
            )
        )

        if (
            parsed.get(
                "sufficient"
            ) is True
            and has_retrieval_evidence
            and not parsed[
                "support_evidence_ids"
            ]
            and not parsed[
                "support_segment_ids"
            ]
        ):
            parsed[
                "sufficient"
            ] = False

            parsed[
                "confidence"
            ] = min(
                float(
                    parsed.get(
                        "confidence",
                        0.0,
                    )
                ),
                max(
                    0.0,
                    confidence_threshold - 1.0,
                ),
            )

            confidence = float(
                parsed[
                    "confidence"
                ]
            )

            parsed[
                "missing_information"
            ] = (
                "explicit support evidence ids or segment ids for "
                "the selected choice's visual claims"
            )

        parsed[
            "sufficient"
        ] = (
            parsed.get(
                "sufficient"
            ) is True
            and answer in valid
            and confidence >= confidence_threshold
        )

        return (
            parsed,
            latency,
        )


    def create_plan(
        self,
        *,
        state: AgentState,
        assessment: dict[str, Any],
    ):
        return self._plan_from_memory(
            state=
                state,

            assessment=
                assessment,

            previous_plan=
                None,

            mode=
                "create",
        )


    def adjust_plan(
        self,
        *,
        state: AgentState,
        assessment: dict[str, Any],
        previous_plan: dict[str, Any] | None,
    ):
        return self._plan_from_memory(
            state=
                state,

            assessment=
                assessment,

            previous_plan=
                previous_plan,

            mode=
                "adjust",
        )


    def _plan_from_memory(
        self,
        *,
        state: AgentState,
        assessment: dict[str, Any],
        previous_plan: dict[str, Any] | None,
        mode: str,
    ):
        evidence_payload = compact_evidence_payload(
            state.evidence,
        )

        memory = compact_memory_for_prompt(
            state,
        )

        coverage_gap = self.coverage_gap(
            state
        )

        if coverage_gap is not None:
            plan = dict(
                coverage_gap
            )

            if (
                plan.get(
                    "tool"
                )
                == "VERIFY_DETAIL"
            ):
                plan[
                    "query"
                ] = (
                    assessment.get(
                        "missing_information"
                    )
                    or plan.get(
                        "query"
                    )
                    or state.question
                )

            return (
                plan,
                0.0,
            )

        prompt = f"""
Create or adjust one next information retrieval plan for a
long-video QA agent.

Mode:
{mode}

Question:
{state.question}

Answer assessment:
{json.dumps(assessment, ensure_ascii=False)}

Memory bank:
{json.dumps(memory, ensure_ascii=False)}

Evidence:
{json.dumps(evidence_payload, ensure_ascii=False)}

Previous plan:
{json.dumps(previous_plan, ensure_ascii=False)}

Available tools:
- SEARCH_LOCAL: semantic retrieval to localize event/person/object
- SEARCH_BEFORE: inspect before an evidence_id anchor
- SEARCH_AFTER: inspect after an evidence_id anchor
- GLOBAL_SCAN: chronological whole-video coverage
- COUNT_EVENTS: count-relevant whole-video enumeration
- INCREASE_DENSITY: denser inspection of an evidence_id interval
- VERIFY_DETAIL: fine detail check of an evidence_id interval
- IMAGE_CAPTION: caption a chosen time range
- ZOOM_CAPTION: crop/zoom a region in a chosen time range and caption
- OBJECT_DETECTION: detect objects in a chosen time range
- OBJECT_TRACKING: track an object over a chosen time range

RULES:

1. Output exactly one next executable tool call.

2. Prefer segment captions and video summary to choose time ranges
   when possible.

3. Prefer VERIFY_DETAIL or ZOOM_CAPTION when a coarse scene is
   localized but a fine visual detail is missing.

4. Prefer SEARCH_BEFORE/SEARCH_AFTER only when temporal relation
   is exactly "before" or "after" and an evidence_id anchor exists.

5. Use uncertainty: low-confidence tool results should be verified
   with a different or finer tool.

6. Do not use answer choices as evidence.

7. Choose GLOBAL_SCAN when missing information requires broad
   chronological coverage of the video.

8. Choose IMAGE_CAPTION when missing information is localized to a
   time range or segment but needs clearer visual description.

9. Choose SEARCH_LOCAL to localize an event/person/object not yet
   grounded in memory.

10. Choose VERIFY_DETAIL, ZOOM_CAPTION, OBJECT_DETECTION, or
    OBJECT_TRACKING for fine-grained visual details.

Return ONLY JSON:

{{
  "tool": "",
  "query": "",
  "start_s": null,
  "end_s": null,
  "evidence_id": null,
  "target": "",
  "bbox": null,
  "reason": ""
}}
"""

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            parsed = {
                "tool": "SEARCH_LOCAL",
                "query":
                    assessment.get(
                        "missing_information",
                        state.question,
                    ),
                "reason":
                    "plan parse failure; fall back to semantic search",
            }

        tool = str(
            parsed.get(
                "tool",
                parsed.get(
                    "action",
                    "SEARCH_LOCAL",
                ),
            )
        ).upper()

        if tool == "COMPARE_CHOICES":
            tool = "VERIFY_DETAIL"

        if tool not in PLAN_ACTIONS:
            tool = "SEARCH_LOCAL"

        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "",
            )
        )

        evidence_type = str(
            state.policy.get(
                "evidence_type",
                "",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        has_global = any(
            evidence.action == "GLOBAL_SCAN"
            for evidence in state.evidence
        )

        needs_fine_detail = (
            evidence_type in {
                "local_detail",
                "generic_local_mcq",
            }
            or is_fine_detail_question(
                state.question
            )
        )

        needs_sequence_detail = (
            evidence_type == "sequence"
            or (
                requirement == "global"
                and relation == "ordering"
            )
            or is_sequence_question(
                state.question
            )
        )

        local_ids = [
            index
            for index, evidence
            in enumerate(
                state.evidence
            )
            if evidence.action == "SEARCH_LOCAL"
        ]

        has_fine_detail = any(
            evidence.action in {
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
                "IMAGE_CAPTION",
                "OBJECT_DETECTION",
                "OBJECT_TRACKING",
            }
            for evidence in state.evidence
        )

        needs_sequence_detail = (
            evidence_type == "sequence"
            or (
                requirement == "global"
                and relation == "ordering"
            )
        )

        has_sequence_detail = any(
            evidence.action in {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }
            for evidence in state.evidence
        )

        if (
            (
                requirement == "global"
                or evidence_type == "sequence"
            )
            and not has_global
        ):
            tool = "GLOBAL_SCAN"

        elif (
            relation == "ordering"
            and tool in {
                "SEARCH_BEFORE",
                "SEARCH_AFTER",
            }
        ):
            tool = (
                "GLOBAL_SCAN"
                if not has_global
                else "IMAGE_CAPTION"
            )

        elif (
            (
                requirement == "global"
                or evidence_type == "sequence"
            )
            and has_global
            and tool == "SEARCH_LOCAL"
        ):
            tool = "IMAGE_CAPTION"

        elif (
            needs_fine_detail
            and local_ids
            and not has_fine_detail
        ):
            tool = "VERIFY_DETAIL"
            parsed[
                "evidence_id"
            ] = local_ids[0]
            parsed[
                "query"
            ] = (
                assessment.get(
                    "missing_information"
                )
                or state.question
            )

        elif (
            needs_sequence_detail
            and has_global
            and not has_sequence_detail
        ):
            tool = "IMAGE_CAPTION"
            parsed[
                "evidence_id"
            ] = None
            parsed[
                "start_s"
            ] = None
            parsed[
                "end_s"
            ] = None
            parsed[
                "query"
            ] = (
                assessment.get(
                    "missing_information"
                )
                or "the exact ordered workflow details"
            )

        parsed[
            "tool"
        ] = tool

        parsed[
            "action"
        ] = tool

        if not parsed.get(
            "query"
        ):
            parsed[
                "query"
            ] = (
                assessment.get(
                    "missing_information"
                )
                or state.question
            )

        return (
            parsed,
            latency,
        )


    def choose_action(
        self,
        state: AgentState,
    ):
        history = []

        for index, evidence in enumerate(
            state.evidence
        ):

            history.append({
                "evidence_id":
                    index,

                "action":
                    evidence.action,

                "query":
                    evidence.query,

                "range": [
                    evidence.start_s,
                    evidence.end_s,
                ],

                "observation":
                    evidence.observation,

                "confidence":
                    evidence.confidence,
            })

        allowed = (
            state.policy.get(
                "allowed_actions",
                ["ANSWER"],
            )
        )

        prompt = f"""
        You control an adaptive VideoQA evidence-acquisition agent.

        Your goal is to acquire enough visual evidence to answer the
        question accurately.

        Use the current memory and evidence to decide whether to
        answer or acquire one more piece of visual evidence. Continue
        searching and refining evidence until the question is
        answerable or the runtime round limit is reached.

Question:
{state.question}

Video duration:
{state.duration_s:.2f}s

Allowed actions:
{json.dumps(
    allowed
)}

Action counts:
{json.dumps(
    action_counts(state)
)}

Evidence:
{json.dumps(
    history,
    ensure_ascii=False,
)}

ACTION MEANINGS:

SEARCH_LOCAL
    Semantically localize an event/person/object/action.

SEARCH_BEFORE
    Inspect immediately before an existing evidence interval.

SEARCH_AFTER
    Inspect immediately after an existing evidence interval.

GLOBAL_SCAN
    Chronological whole-video sampling.

COUNT_EVENTS
    Whole-video enumeration for counting.

INCREASE_DENSITY
    Sample more frames from an already known interval.

VERIFY_DETAIL
    Reinspect a localized interval for fine visual detail.

ANSWER
    Stop acquiring evidence.


IMPORTANT:

1. Continue acquiring useful evidence when current evidence is
   genuinely insufficient.

2. Choose ONLY from Allowed actions.

3. Do not use answer choices.

4. SEARCH_BEFORE, SEARCH_AFTER, and VERIFY_DETAIL require an
   evidence_id.

5. Do not repeat an identical semantic search.

6. If SEARCH_LOCAL has already localized the correct person,
   object, action, or event but a fine visual detail remains
   unresolved, prefer VERIFY_DETAIL on that evidence_id.

7. Do not rerun whole-video semantic retrieval just to get more
   detail about an already-localized event.

8. Use another SEARCH_LOCAL only when:
   - the current localization appears wrong, or
   - a genuinely different temporal region must be found.

9. Never use ground truth.

Return ONLY JSON:

{{
  "action":
      "SEARCH_LOCAL",

  "query":
      "visual evidence to find",

  "evidence_id":
      null,

  "reason":
      "why this action helps"
}}
"""

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(parsed, dict):
            return (
                {
                    "action": "SEARCH_LOCAL",
                    "query": state.question,
                    "evidence_id": None,
                    "reason":
                        "Controller parse failure; continue evidence acquisition.",
                },
                latency,
            )

        action = str(
            parsed.get(
                "action",
                "ANSWER",
            )
        ).upper()

        if action not in allowed:
            if "SEARCH_LOCAL" in allowed:
                action = "SEARCH_LOCAL"
            elif "GLOBAL_SCAN" in allowed:
                action = "GLOBAL_SCAN"
            else:
                action = "ANSWER"

        if action not in VALID_ACTIONS:
            action = "ANSWER"

        parsed[
            "action"
        ] = action

        return (
            parsed,
            latency,
        )


    def assess_sufficiency(
        self,
        state: AgentState,
    ):
        history = compact_evidence_payload(
            state.evidence,
        )

        memory = compact_memory_for_prompt(
            state,
            max_segments=int(
                state.policy.get(
                    "max_final_memory_segments",
                    96,
                )
                or 96
            ),
        )

        allowed = (
            state.policy.get(
                "allowed_actions",
                ["ANSWER"],
            )
        )

        prompt = f"""
Determine whether the accumulated VIDEO evidence is sufficient
to answer the question.

Question:
{state.question}

Profiler evidence requirement:
{state.policy.get("evidence_requirement")}

Temporal relation:
{state.policy.get("temporal_relation")}

Evidence type:
{state.policy.get("evidence_type")}

Allowed next actions:
{json.dumps(
    allowed
)}

Evidence:
{json.dumps(
    history,
    ensure_ascii=False,
)}

RULES:

1. Use only acquired VIDEO evidence.

2. Do not use answer choices.

3. Determine exactly which visual fact or facts are required
   to answer the question.

4. Compare those required facts against EVERY evidence item.

5. If an evidence item explicitly establishes a requested
   fact, treat that fact as observed. Do NOT ask to reconfirm it.

6. If the accumulated evidence directly establishes the
   requested visual fact, return sufficient=true.

7. Before/after questions require:
   - a localized anchor
   - evidence from the correct side of the anchor

8. Counting questions require adequate whole-video count evidence.

9. Global questions require adequate chronological coverage.

10. Do not claim firstness or lastness from semantic retrieval alone.

11. Return insufficient only when a SPECIFIC required visual fact
    is absent from ALL accumulated evidence.

12. If sufficient=false, recommended_action MUST NOT be ANSWER.

13. If a localized interval contains the correct person/object/event
    but finer detail is missing, recommend VERIFY_DETAIL and provide
    its evidence_id.

14. Return the strongest relevant evidence_id.

15. Do not restate an already observed coarse event as the
    missing_visual_fact.

    Example:

    Question:
    "What is he doing with his raised right hand?"

    Evidence:
    "The man raises his right hand."

    This evidence establishes only the coarse localization event.
    It does NOT answer the question.

    The missing visual fact is:
    "the specific gesture, finger configuration, direction, or
    motion performed with the raised right hand."

16. For questions asking "what is he/she doing with X",
    distinguish the existence/state of X from the specific action
    performed with X.

17. recommended_query must target ONLY the unresolved attribute,
    not repeat an already established coarse event.
Return ONLY JSON:

{{
  "sufficient":
      false,

  "missing_visual_fact":
      "remaining fact",

  "recommended_action":
      "VERIFY_DETAIL",

  "recommended_query":
      "what to inspect",

  "evidence_id":
      0
}}
"""

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return (
                None,
                latency,
            )

        parsed[
            "sufficient"
        ] = (
            parsed.get(
                "sufficient"
            )
            is True
        )

        recommended = str(
            parsed.get(
                "recommended_action",
                "ANSWER",
            )
        ).upper()

        if recommended not in allowed:
            recommended = "ANSWER"

        parsed[
            "recommended_action"
        ] = recommended

        return (
            parsed,
            latency,
        )


    def answer_from_evidence(
        self,
        *,
        state: AgentState,
        choices: list[str],
    ):
        valid = valid_choice_labels(
            choices
        )

        history = compact_evidence_payload(
            state.evidence,
        )

        memory = compact_memory_for_prompt(
            state,
            max_segments=int(
                state.policy.get(
                    "max_final_memory_segments",
                    96,
                )
                or 96
            ),
        )

        prompt = f"""
            Answer the multiple-choice VideoQA question using ONLY the
            accumulated VIDEO memory and evidence.

            Question:
            {state.question}

            Choices:
            {format_choices(
                choices
            )}

            Evidence:
            {json.dumps(
                history,
                ensure_ascii=False,
            )}

            VideoAgent2 memory bank:
            {json.dumps(
                memory,
                ensure_ascii=False,
            )}

            RULES:

            1. The evidence is the factual source.

            2. Treat answer choices as hypotheses, not evidence.

            3. Do not invent visual facts because an option is plausible.

            4. Respect before/after/ordering/counting semantics.

            5. Prefer explicit visual evidence.

            6. You MUST select exactly one answer choice.

            7. Ignore prior answer-assessment labels. They are controller
            diagnostics and may be inconsistent. Choose from the evidence
            and memory shown above only.

            8. If the evidence is incomplete, select the choice best supported
            by the available evidence. Do not return an empty prediction.

            9. For local-detail questions, do not choose a choice merely
            because it repeats a shared setup visible in all choices.
            Use the evidence that resolves the distinguishing detail:
            gesture direction, exact hand action, object interaction,
            clothing/person identity, or scene identity.

            10. If evidence only establishes the shared coarse setup, prefer
            the choice whose unique detail is explicitly supported by the
            finest-grained evidence. Do not infer unsupported details.

            11. Do not treat absence of evidence for the other choices as
            positive evidence for a selected choice. A choice such as
            "reaching into", "cutting", "throwing away", "stepping on",
            "counting", "pointing", or a named scene must be selected only
            when that unique action or scene is explicitly visible in the
            finest evidence.

            12. Do not equate nearby but different actions. "Holding",
            "opening", or "examining" a box is not evidence of "reaching
            into" the box unless a hand visibly goes into the box. A raised
            hand is not evidence of a specific gesture unless the fingers,
            palm, direction, or motion are visible.

            13. If your reason says the evidence aligns with option B,
            prediction must be "B". The prediction label and reason must
            refer to the same option.

            14. For sequence/workflow questions, compare the ordered
            discriminators in the choices directly. Pay special attention
            to the first referenced object or scene, tool order, cloth use,
            and whether the evidence says laptop, art board, sky, flowers,
            or another target. Prefer the choice whose ordered details are
            explicitly present in the video memory/evidence.

            15. Before predicting, create option_support for every choice.
            For each label, list:
            - unique_claims: the claims that distinguish this option from
              the others
            - supported_claims: unique claims directly supported by memory
              or evidence
            - contradicted_claims: unique claims contradicted by memory or
              evidence
            - unknown_claims: unique claims not established either way

            16. Do not choose an option whose first-step discriminator is
            unknown or contradicted when another option's first-step
            discriminator is supported. For the art workflow example, if
            memory says the person looks at a laptop first, choose the
            laptop option over an art-board option even if both mention
            painting and cloth use.

            Return ONLY JSON:

            {{
            "prediction":
                "",

            "option_support":
                {{
                  "A": {{
                    "unique_claims": [],
                    "supported_claims": [],
                    "contradicted_claims": [],
                    "unknown_claims": []
                  }}
                }},

            "reason":
                "brief evidence-grounded reason",

            "confidence":
                "low"
            }}

            A non-empty prediction must be one of:
            {sorted(valid)}
            """

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            parsed = {
                "prediction": "",
                "reason": str(
                    response
                    .choices[0]
                    .message.content
                ),
                "confidence": "low",
            }

        prediction = normalize_answer(
            parsed.get(
                "prediction",
                "",
            )
        )

        if prediction not in valid:
            retry_prompt = f"""
            You MUST choose exactly one answer.

            Question:
            {state.question}

            Choices:
            {format_choices(choices)}

            Evidence:
            {json.dumps(history, ensure_ascii=False)}

            Even if the evidence is incomplete, choose the answer that is
            best supported by the visual evidence.

            Return ONLY JSON:

            {{
            "prediction": "A"
            }}

            Prediction must be exactly one of:
            {sorted(valid)}
            """

            retry_start = time.time()

            retry = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": retry_prompt,
                    }
                ],
                temperature=0,
            )

            latency += time.time() - retry_start

            retry_parsed = extract_json(
                retry.choices[0].message.content
            )

            if isinstance(retry_parsed, dict):
                prediction = normalize_answer(
                    retry_parsed.get(
                        "prediction",
                        "",
                    )
                )

        # Absolute fallback: evaluation should never receive blank.
        if prediction not in valid:
            print(
                "WARNING: final answer model "
                "failed to return a valid choice"
            )
            prediction = ""

        reason = str(
            parsed.get(
                "reason",
                "",
            )
        )

        supported_labels = (
            supported_labels_mentioned_in_text(
                reason
            )
            & valid
        )

        if (
            len(supported_labels) == 1
            and prediction not in supported_labels
        ):
            prediction = next(
                iter(
                    supported_labels
                )
            )

        option_support = parsed.get(
            "option_support",
            {},
        )

        if isinstance(
            option_support,
            dict,
        ):
            viable = []

            for label in sorted(
                valid
            ):
                item = option_support.get(
                    label,
                    {},
                )

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                supported = item.get(
                    "supported_claims",
                    [],
                )

                contradicted = item.get(
                    "contradicted_claims",
                    [],
                )

                unknown = item.get(
                    "unknown_claims",
                    [],
                )

                if (
                    supported
                    and not contradicted
                ):
                    viable.append(
                        (
                            label,
                            len(
                                supported
                            ),
                            len(
                                unknown
                                or []
                            ),
                        )
                    )

            if viable:
                viable.sort(
                    key=lambda item: (
                        -item[1],
                        item[2],
                        item[0],
                    )
                )

                best_label = viable[0][0]

                if (
                    prediction not in valid
                    or best_label != prediction
                ):
                    prediction = best_label

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "prediction":
                prediction,

            "reason":
                reason,

            "confidence":
                confidence,

            "latency_s":
                latency,
        }


    def remap_answer_to_choice(
        self,
        *,
        state: AgentState,
        choices: list[str],
        prediction: str,
        reason: str,
    ):
        valid = valid_choice_labels(
            choices
        )

        prediction = normalize_answer(
            prediction
        )

        history = compact_evidence_payload(
            state.evidence,
        )

        memory = compact_memory_for_prompt(
            state,
            max_segments=int(
                state.policy.get(
                    "max_final_memory_segments",
                    96,
                )
                or 96
            ),
        )

        prompt = f"""
        Verify the final multiple-choice label. The evidence/reason may
        describe the right visual answer while using the wrong option letter.

        Question:
        {state.question}

        Choices:
        {format_choices(choices)}

        Current prediction:
        {prediction}

        Current reason:
        {reason}

        Evidence:
        {json.dumps(history, ensure_ascii=False)}

        Video memory:
        {json.dumps(memory, ensure_ascii=False)}

        RULES:
        1. Use the evidence and video memory as the factual source.
        2. Compare the visual answer described by the evidence/reason
           against the exact text of every option.
        3. Return the option letter whose text best matches the supported
           visual facts.
        4. If the current prediction already matches the option text, keep it.
        5. Do not choose an option only because the question setup repeats.

        Return ONLY JSON:

        {{
          "prediction": "A",
          "reason": "brief explanation of the option-text match"
        }}

        Prediction must be exactly one of:
        {sorted(valid)}
        """

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            return {
                "prediction":
                    prediction
                    if prediction in valid
                    else "",

                "reason":
                    reason,

                "latency_s":
                    latency,
            }

        remapped = normalize_answer(
            parsed.get(
                "prediction",
                "",
            )
        )

        if remapped not in valid:
            remapped = (
                prediction
                if prediction in valid
                else ""
            )

        remap_reason = str(
            parsed.get(
                "reason",
                "",
            )
            or reason
        )

        return {
            "prediction":
                remapped,

            "reason":
                remap_reason,

            "latency_s":
                latency,
        }


    def answer_sequence_from_timeline(
        self,
        *,
        state: AgentState,
        choices: list[str],
    ):
        valid = valid_choice_labels(
            choices
        )

        timeline = []

        for evidence in state.evidence:
            if evidence.action not in {
                "GLOBAL_SCAN",
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }:
                continue

            timeline.append({
                "action":
                    evidence.action,

                "start_s":
                    evidence.start_s,

                "end_s":
                    evidence.end_s,

                "observation":
                    evidence.observation,

                "confidence":
                    evidence.confidence,
            })

        timeline.sort(
            key=lambda item: (
                float(
                    item.get(
                        "start_s",
                        0.0,
                    )
                    or 0.0
                ),
                float(
                    item.get(
                        "end_s",
                        0.0,
                    )
                    or 0.0
                ),
            )
        )

        memory = compact_memory_for_prompt(
            state,
            max_segments=int(
                state.policy.get(
                    "max_final_memory_segments",
                    96,
                )
                or 96
            ),
        )

        prompt = f"""
        Answer this sequence/order VideoQA question by comparing each
        option against timestamped timeline evidence.

        Question:
        {state.question}

        Choices:
        {format_choices(choices)}

        Timestamped tool evidence:
        {json.dumps(timeline, ensure_ascii=False)}

        Video memory:
        {json.dumps(memory, ensure_ascii=False)}

        RULES:
        1. First extract an ordered list of visible events/scenes from
           the timestamped evidence and memory.
        2. For every option, compare event 1, event 2, event 3, etc.
           against the extracted timeline in chronological order.
        3. Reject an option if its first event is missing or contradicted
           when another option's first event is supported.
        4. Do not choose an option only because it contains familiar
           objects; the order must match.
        5. Prefer the option with the strongest ordered match.
        6. Return exactly one option label.

        Return ONLY JSON:

        {{
          "prediction": "A",
          "ordered_events": [],
          "option_order_support": {{
            "A": {{
              "matched_events": [],
              "contradictions": [],
              "unknown_events": []
            }}
          }},
          "reason": "brief timestamp-grounded reason",
          "confidence": "low"
        }}

        Prediction must be exactly one of:
        {sorted(valid)}
        """

        t0 = time.time()

        response = (
            self.client
            .chat.completions
            .create(
                model=
                    self.model,

                messages=[{
                    "role":
                        "user",

                    "content":
                        prompt,
                }],

                temperature=
                    0,
            )
        )

        latency = (
            time.time()
            - t0
        )

        parsed = extract_json(
            response
            .choices[0]
            .message.content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            parsed = {
                "prediction": "",
                "reason": str(
                    response
                    .choices[0]
                    .message.content
                ),
                "confidence": "low",
            }

        prediction = normalize_answer(
            parsed.get(
                "prediction",
                "",
            )
        )

        if prediction not in valid:
            prediction = ""

        confidence = str(
            parsed.get(
                "confidence",
                "low",
            )
        ).lower()

        if confidence not in {
            "low",
            "medium",
            "high",
        }:
            confidence = "low"

        return {
            "prediction":
                prediction,

            "reason":
                str(
                    parsed.get(
                        "reason",
                        "",
                    )
                ),

            "confidence":
                confidence,

            "latency_s":
                latency,
        }


# ============================================================
# Shared video execution helpers
# ============================================================

class VideoExecutionBase:

    def __init__(
        self,
        *,
        inspector,
        retriever,
        object_tools=None,
        global_chunk_size: int = 4,
    ):
        self.inspector = inspector
        self.retriever = retriever
        self.object_tools = object_tools

        self.global_chunk_size = max(
            1,
            int(global_chunk_size),
        )


    def inspect_window(
        self,
        *,
        video,
        question,
        action,
        query,
        start_s,
        end_s,
        num_frames,
    ):
        start_s = max(
            0.0,
            float(start_s),
        )

        end_s = min(
            video.duration_s,
            float(end_s),
        )

        if end_s <= start_s:
            return None

        timestamps, frames = (
            video.sample_range(
                start_s,
                end_s,
                num_frames,
            )
        )

        result = (
            self.inspector
            .inspect(
                question=
                    question,

                timestamps=
                    timestamps,

                frames=
                    frames,

                action=
                    action,

                query=
                    query,
            )
        )

        return Evidence(
            action=
                action,

            query=
                query,

            start_s=
                start_s,

            end_s=
                end_s,

            timestamps=
                timestamps,

            observation=
                result[
                    "observation"
                ],

            confidence=
                result[
                    "confidence"
                ],

            latency_s=
                float(
                    result[
                        "latency_s"
                    ]
                ),

            confidence_score=
                confidence_to_score(
                    result.get(
                        "confidence_score",
                        result.get(
                            "confidence"
                        ),
                    )
                ),

            uncertainty=
                score_to_uncertainty(
                    result.get(
                        "confidence_score",
                        result.get(
                            "confidence"
                        ),
                    )
                ),

            tool_name=
                action,
        )


    def semantic_candidates(
        self,
        *,
        video,
        query,
        probe_fps,
        topk,
    ):
        probe_fps = max(
            float(probe_fps),
            1.0 / max(
                video.duration_s,
                1.0,
            ),
        )

        count = int(
            math.ceil(
                video.duration_s
                * probe_fps
            )
        )

        count = max(
            2,
            min(
                256,
                count,
            ),
        )

        timestamps, frames = (
            video.sample_range(
                0.0,
                video.duration_s,
                count,
            )
        )

        print(
            "    semantic probe frames:",
            count,
        )

        return (
            self.retriever
            .rank_frames(
                query=
                    query,

                timestamps=
                    timestamps,

                frames=
                    frames,

                topk=
                    max(
                        1,
                        int(topk),
                    ),
            )
        )


# ============================================================
# Fixed / one-shot executor
# ============================================================

class FixedVIMIOExecutor(
    VideoExecutionBase
):

    def __init__(
        self,
        *,
        controller,
        inspector,
        retriever,
        global_chunk_size,
    ):
        super().__init__(
            inspector=
                inspector,

            retriever=
                retriever,

            object_tools=
                None,

            global_chunk_size=
                global_chunk_size,
        )

        self.controller = controller


    def run(
        self,
        *,
        video,
        question,
        choices,
        policy,
    ):
        """
        Fixed execution.

        No adaptive controller loop.

        Profiler chooses the geometry once and we execute it.
        """

        policy = sanitize_policy(
            policy
        )

        state = AgentState(
            question=
                question,

            duration_s=
                video.duration_s,

            policy=
                policy,
        )

        requirement = str(
            policy.get(
                "evidence_requirement",
                "local",
            )
        )

        relation = str(
            policy.get(
                "temporal_relation",
                "none",
            )
        )

        probe_fps = float(
            policy.get(
                "probe_fps",
                0.05,
            )
        )

        topk = (
            policy.get(
                "action_topk"
            )
            or
            policy.get(
                "probe_topk"
            )
            or
            4
        )

        window_len_s = float(
            policy.get(
                "window_len_s",
                8.0,
            )
        )

        frames_per_window = int(
            policy.get(
                "high_frames_per_window",
                8,
            )
        )

        total_budget = int(
            policy.get(
                "answer_max_images_total",
                frames_per_window,
            )
        )

        trajectory = []

        # ----------------------------------------------------
        # Direct beginning/end fixed execution
        # ----------------------------------------------------

        if (
            requirement == "temporal"
            and relation in {
                "beginning",
                "end",
            }
        ):

            if relation == "beginning":

                start_s = 0.0

                end_s = min(
                    video.duration_s,
                    window_len_s,
                )

            else:

                end_s = (
                    video.duration_s
                )

                start_s = max(
                    0.0,
                    end_s
                    - window_len_s,
                )

            evidence = self.inspect_window(
                video=
                    video,

                question=
                    question,

                action=
                    "INCREASE_DENSITY",

                query=
                    question,

                start_s=
                    start_s,

                end_s=
                    end_s,

                num_frames=
                    min(
                        frames_per_window,
                        total_budget,
                    ),
            )

            if evidence is not None:
                state.evidence.append(
                    evidence
                )

                state.total_latency_s += (
                    evidence.latency_s
                )

            trajectory.append({
                "round":
                    1,

                "decision": {
                    "action":
                        "FIXED_BOUNDARY",

                    "relation":
                        relation,

                    "start_s":
                        start_s,

                    "end_s":
                        end_s,
                },
            })

        # ----------------------------------------------------
        # Fixed uniform program
        # ----------------------------------------------------

        elif policy.get(
            "selection_mode"
        ) == "uniform":

            num_frames = max(
                1,
                min(
                    total_budget,
                    int(
                        math.ceil(
                            video.duration_s
                            * probe_fps
                        )
                    ),
                ),
            )

            evidence = self.inspect_window(
                video=
                    video,

                question=
                    question,

                action=
                    "GLOBAL_SCAN",

                query=
                    question,

                start_s=
                    0.0,

                end_s=
                    video.duration_s,

                num_frames=
                    num_frames,
            )

            if evidence is not None:
                state.evidence.append(
                    evidence
                )

                state.total_latency_s += (
                    evidence.latency_s
                )

            trajectory.append({
                "round":
                    1,

                "decision": {
                    "action":
                        "FIXED_UNIFORM",

                    "num_frames":
                        num_frames,
                },
            })

        # ----------------------------------------------------
        # Fixed local semantic retrieval
        # ----------------------------------------------------

        else:

            ranked = (
                self.semantic_candidates(
                    video=
                        video,

                    query=
                        question,

                    probe_fps=
                        probe_fps,

                    topk=
                        int(topk),
                )
            )

            if ranked:

                max_windows = max(
                    1,
                    total_budget
                    // max(
                        1,
                        frames_per_window,
                    ),
                )

                max_windows = min(
                    max_windows,
                    len(ranked),
                    int(topk),
                )

                remaining_budget = (
                    total_budget
                )

                for candidate in ranked[
                    :max_windows
                ]:

                    anchor = float(
                        candidate[
                            "timestamp_s"
                        ]
                    )

                    half = (
                        window_len_s
                        / 2.0
                    )

                    start_s = max(
                        0.0,
                        anchor - half,
                    )

                    end_s = min(
                        video.duration_s,
                        anchor + half,
                    )

                    num_frames = min(
                        frames_per_window,
                        remaining_budget,
                    )

                    if num_frames <= 0:
                        break

                    evidence = (
                        self.inspect_window(
                            video=
                                video,

                            question=
                                question,

                            action=
                                "SEARCH_LOCAL",

                            query=
                                question,

                            start_s=
                                start_s,

                            end_s=
                                end_s,

                            num_frames=
                                num_frames,
                        )
                    )

                    if evidence is not None:

                        state.evidence.append(
                            evidence
                        )

                        state.total_latency_s += (
                            evidence.latency_s
                        )

                    remaining_budget -= (
                        num_frames
                    )

            trajectory.append({
                "round":
                    1,

                "decision": {
                    "action":
                        "FIXED_LOCAL",

                    "probe_fps":
                        probe_fps,

                    "topk":
                        topk,

                    "window_len_s":
                        window_len_s,

                    "frame_budget":
                        total_budget,
                },
            })

        # ----------------------------------------------------
        # One final answer call
        # ----------------------------------------------------

        answer = (
            self.controller
            .answer_from_evidence(
                state=
                    state,

                choices=
                    choices,
            )
        )

        state.total_latency_s += float(
            answer[
                "latency_s"
            ]
        )

        state.final_answer = (
            answer[
                "prediction"
            ]
        )

        trajectory.append({
            "round":
                "final_answer",

            "decision": {
                "action":
                    "ANSWER",

                "answer":
                    state.final_answer,

                "reason":
                    answer[
                        "reason"
                    ],

                "confidence":
                    answer[
                        "confidence"
                    ],
            },
        })

        return (
            state,
            trajectory,
        )


# ============================================================
# Agentic executor
# ============================================================

class VideoAgent(
    VideoExecutionBase
):

    def __init__(
        self,
        *,
        controller,
        inspector,
        retriever,
        max_rounds,
        global_chunk_size,
        object_tools=None,
        videoagent2_context: bool = True,
        context_fps: float = 1.0,
        context_segment_s: float = 4.0,
        context_frames_per_segment: int = 4,
        max_context_segments: int = 32,
        context_coverage: str = "adaptive",
        caption_cache_dir: str | None = None,
        disable_caption_cache: bool = False,
        caption_workers: int = 1,
        answer_confidence_threshold: float = 5.0,
    ):
        super().__init__(
            inspector=
                inspector,

            retriever=
                retriever,

            object_tools=
                object_tools,

            global_chunk_size=
                global_chunk_size,
        )

        self.controller = controller

        self.max_rounds = int(
            max_rounds
        )

        self.videoagent2_context = bool(
            videoagent2_context
        )

        self.context_fps = float(
            context_fps
        )

        self.context_segment_s = float(
            context_segment_s
        )

        self.context_frames_per_segment = int(
            context_frames_per_segment
        )

        self.max_context_segments = int(
            max_context_segments
        )

        self.context_coverage = str(
            context_coverage
        ).lower()

        self.caption_cache_dir = (
            Path(
                caption_cache_dir
            )
            if caption_cache_dir
            else None
        )

        self.disable_caption_cache = bool(
            disable_caption_cache
        )

        self.caption_workers = max(
            1,
            int(
                caption_workers
            ),
        )

        self.answer_confidence_threshold = float(
            answer_confidence_threshold
        )


    # ========================================================
    # VideoAgent2 memory / plan helpers
    # ========================================================

    def caption_cache_path(
        self,
        *,
        video,
        num_segments,
        segment_ids,
    ):
        if (
            self.disable_caption_cache
            or self.caption_cache_dir is None
        ):
            return None

        video_path = str(
            getattr(
                video,
                "path",
                "",
            )
            or getattr(
                video,
                "video_path",
                "",
            )
            or getattr(
                video,
                "_path",
                "",
            )
        )

        try:
            stat = Path(
                video_path
            ).stat()
            video_mtime = stat.st_mtime
            video_size = stat.st_size
        except OSError:
            video_mtime = None
            video_size = None

        payload = {
            "video_path":
                video_path,

            "video_duration_s":
                round(
                    float(
                        video.duration_s
                    ),
                    3,
                ),

            "video_mtime":
                video_mtime,

            "video_size":
                video_size,

            "caption_model":
                self.inspector.caption_model,

            "caption_prompt_style":
                self.inspector.caption_prompt_style,

            "context_fps":
                self.context_fps,

            "context_segment_s":
                self.context_segment_s,

            "context_frames_per_segment":
                self.context_frames_per_segment,

            "context_coverage":
                self.context_coverage,

            "num_segments":
                num_segments,

            "segment_ids":
                segment_ids,
        }

        digest = hashlib.sha1(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            self.caption_cache_dir
            / f"{digest}.json"
        )


    def load_caption_cache(
        self,
        path,
    ):
        if path is None:
            return None

        try:
            with Path(path).open() as handle:
                payload = json.load(handle)
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

        if not isinstance(
            payload.get(
                "segment_captions"
            ),
            list,
        ):
            return None

        return payload


    def save_caption_cache(
        self,
        *,
        path,
        payload,
    ):
        if path is None:
            return

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = path.with_suffix(
            ".tmp"
        )

        with tmp.open("w") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
            )

        tmp.replace(
            path
        )


    def context_segment_ids(
        self,
        *,
        state,
        num_segments,
    ):
        coverage = self.context_coverage

        if coverage not in {
            "all",
            "adaptive",
            "sparse",
        }:
            coverage = "adaptive"

        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        evidence_type = str(
            state.policy.get(
                "evidence_type",
                "",
            )
        )

        if coverage == "all":
            return list(
                range(
                    num_segments
                )
            )

        if coverage == "adaptive":
            needs_broad_memory = (
                requirement in {
                    "global",
                    "count",
                }
                or relation == "ordering"
                or evidence_type in {
                    "sequence",
                    "global",
                }
            )

            if needs_broad_memory:
                return list(
                    range(
                        num_segments
                    )
                )

        max_segments = int(
            self.max_context_segments
        )

        if max_segments <= 0:
            return list(
                range(
                    num_segments
                )
            )

        max_segments = max(
            1,
            min(
                max_segments,
                num_segments,
            ),
        )

        if num_segments <= max_segments:
            return list(
                range(
                    num_segments
                )
            )

        selected = np.linspace(
            0,
            num_segments - 1,
            max_segments,
            dtype=int,
        )

        return list(
            dict.fromkeys(
                int(value)
                for value in selected
            )
        )


    def acquire_general_context(
        self,
        *,
        state,
        video,
    ):
        if not self.videoagent2_context:
            state.memory_bank = {
                "segment_captions": [],
                "video_summary": "",
                "tool_results": [],
            }
            return

        segment_s = max(
            1.0,
            self.context_segment_s,
        )

        num_segments = max(
            1,
            int(
                math.ceil(
                    video.duration_s
                    / segment_s
                )
            ),
        )

        segment_ids = self.context_segment_ids(
            state=
                state,

            num_segments=
                num_segments,
        )

        cache_path = self.caption_cache_path(
            video=
                video,

            num_segments=
                num_segments,

            segment_ids=
                segment_ids,
        )

        cached = self.load_caption_cache(
            cache_path
        )

        segment_captions = (
            cached.get(
                "segment_captions",
                [],
            )
            if cached
            else []
        )

        total_latency = 0.0

        if cached:
            print(
                "    caption cache hit:",
                cache_path,
            )

        else:
            print(
                "    caption cache miss:",
                cache_path,
            )

            def make_caption_job(
                segment_id,
            ):
                start_s = float(
                    segment_id
                    * segment_s
                )

                end_s = min(
                    video.duration_s,
                    start_s
                    + segment_s,
                )

                if end_s <= start_s:
                    return None

                frame_count = max(
                    1,
                    min(
                        self.context_frames_per_segment,
                        int(
                            math.ceil(
                                (end_s - start_s)
                                * max(
                                    self.context_fps,
                                    0.01,
                                )
                            )
                        ),
                    ),
                )

                timestamps, frames = (
                    video.sample_range(
                        start_s,
                        end_s,
                        frame_count,
                    )
                )

                return {
                    "segment_id":
                        segment_id,

                    "start_s":
                        start_s,

                    "end_s":
                        end_s,

                    "timestamps":
                        timestamps,

                    "frames":
                        frames,
                }


            def run_caption_job(
                job,
            ):
                caption = (
                    self.inspector
                    .caption_frames(
                        question=
                            "",

                        timestamps=
                            job[
                                "timestamps"
                            ],

                        frames=
                            job[
                                "frames"
                            ],

                        purpose=
                            "generic reusable segment caption for long-video memory",
                    )
                )

                score = confidence_to_score(
                    caption.get(
                        "confidence_score"
                    )
                )

                return {
                    "segment_id":
                        job[
                            "segment_id"
                        ],

                    "start_s":
                        job[
                            "start_s"
                        ],

                    "end_s":
                        job[
                            "end_s"
                        ],

                    "timestamps":
                        job[
                            "timestamps"
                        ],

                    "caption":
                        caption[
                            "caption"
                        ],

                    "confidence_score":
                        score,

                    "uncertainty":
                        score_to_uncertainty(
                            score
                        ),

                    "latency_s":
                        float(
                            caption[
                                "latency_s"
                            ]
                        ),
                }


            pending = []
            segment_iter = iter(
                segment_ids
            )

            with ThreadPoolExecutor(
                max_workers=
                    self.caption_workers
            ) as executor:

                while True:

                    while (
                        len(pending)
                        < self.caption_workers
                    ):
                        try:
                            segment_id = next(
                                segment_iter
                            )
                        except StopIteration:
                            break

                        job = make_caption_job(
                            segment_id
                        )

                        if job is None:
                            continue

                        pending.append(
                            executor.submit(
                                run_caption_job,
                                job,
                            )
                        )

                    if not pending:
                        break

                    done, not_done = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )

                    pending = list(
                        not_done
                    )

                    for future in done:
                        item = future.result()
                        total_latency += float(
                            item.pop(
                                "latency_s"
                            )
                        )
                        segment_captions.append(
                            item
                        )

            segment_captions.sort(
                key=lambda item:
                    int(
                        item.get(
                            "segment_id",
                            0,
                        )
                    )
            )

            self.save_caption_cache(
                path=
                    cache_path,

                payload={
                    "caption_model":
                        self.inspector.caption_model,

                    "caption_prompt_style":
                        self.inspector.caption_prompt_style,

                    "context_coverage":
                        self.context_coverage,

                    "context_fps":
                        self.context_fps,

                    "context_segment_s":
                        self.context_segment_s,

                    "context_frames_per_segment":
                        self.context_frames_per_segment,

                    "num_total_segments":
                        num_segments,

                    "context_segment_ids":
                        segment_ids,

                    "segment_captions":
                        segment_captions,
                },
            )

        summary = (
            self.inspector
            .summarize_context(
                question=
                    state.question,

                segment_captions=
                    segment_captions,
            )
        )

        total_latency += float(
            summary[
                "latency_s"
            ]
        )

        summary_score = confidence_to_score(
            summary.get(
                "confidence_score"
            )
        )

        state.memory_bank = {
            "caption_model":
                self.inspector.caption_model,

            "caption_prompt_style":
                self.inspector.caption_prompt_style,

            "caption_cache_path":
                str(
                    cache_path
                )
                if cache_path is not None
                else "",

            "caption_cache_hit":
                bool(
                    cached
                ),

            "context_coverage":
                self.context_coverage,

            "num_total_segments":
                num_segments,

            "num_context_segments":
                len(
                    segment_ids
                ),

            "context_segment_ids":
                segment_ids,

            "segment_captions":
                segment_captions,

            "video_summary":
                summary[
                    "summary"
                ],

            "summary_relevant_segments":
                summary.get(
                    "relevant_segments",
                    [],
                ),

            "summary_confidence_score":
                summary_score,

            "summary_uncertainty":
                score_to_uncertainty(
                    summary_score
                ),

            "tool_results":
                [],
        }

        state.total_latency_s += (
            total_latency
        )


    def record_new_evidence(
        self,
        *,
        state,
        start_index,
        plan=None,
    ):
        tool_results = state.memory_bank.setdefault(
            "tool_results",
            [],
        )

        for evidence_index in range(
            start_index,
            len(state.evidence),
        ):
            evidence = state.evidence[
                evidence_index
            ]

            score = confidence_to_score(
                evidence.confidence_score
                if evidence.confidence_score is not None
                else evidence.confidence
            )

            uncertainty = score_to_uncertainty(
                score
            )

            evidence.confidence_score = score
            evidence.uncertainty = uncertainty

            item = {
                "evidence_id":
                    evidence_index,

                "tool":
                    evidence.tool_name
                    or evidence.action,

                "action":
                    evidence.action,

                "query":
                    evidence.query,

                "range": [
                    evidence.start_s,
                    evidence.end_s,
                ],

                "observation":
                    evidence.observation,

                "confidence_score":
                    score,

                "uncertainty":
                    uncertainty,

                "plan":
                    plan,
            }

            tool_results.append(
                item
            )

            state.tool_uncertainties.append({
                "evidence_id":
                    evidence_index,

                "tool":
                    evidence.tool_name
                    or evidence.action,

                "confidence_score":
                    score,

                "uncertainty":
                    uncertainty,
            })


    def plan_to_decision(
        self,
        *,
        plan,
        state,
    ):
        decision = dict(
            plan
            or {}
        )

        action = str(
            decision.get(
                "tool",
                decision.get(
                    "action",
                    "SEARCH_LOCAL",
                ),
            )
        ).upper()

        if action == "COMPARE_CHOICES":
            action = "VERIFY_DETAIL"

        if action not in PLAN_ACTIONS:
            action = "SEARCH_LOCAL"

        coverage_gap = (
            self.controller
            .coverage_gap(
                state
            )
        )

        if coverage_gap is not None:
            decision.update(
                coverage_gap
            )
            action = str(
                coverage_gap.get(
                    "tool",
                    coverage_gap.get(
                        "action",
                        action,
                    ),
                )
            ).upper()

        profiler_policy_mode = str(
            state.policy.get(
                "profiler_policy_mode",
                "hint",
            )
        ).lower()

        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "",
            )
        )

        evidence_type = str(
            state.policy.get(
                "evidence_type",
                "",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        has_global = any(
            evidence.action == "GLOBAL_SCAN"
            for evidence in state.evidence
        )

        needs_fine_detail = (
            evidence_type in {
                "local_detail",
                "generic_local_mcq",
            }
            or is_fine_detail_question(
                state.question
            )
        )

        needs_recurrence = is_recurrence_question(
            state.question
        )

        if needs_recurrence:
            needs_fine_detail = False

        local_ids = [
            index
            for index, evidence
            in enumerate(
                state.evidence
            )
            if evidence.action == "SEARCH_LOCAL"
        ]

        has_fine_detail = any(
            evidence.action in {
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
                "IMAGE_CAPTION",
                "OBJECT_DETECTION",
                "OBJECT_TRACKING",
            }
            for evidence in state.evidence
        )

        needs_sequence_detail = (
            evidence_type == "sequence"
            or (
                requirement == "global"
                and relation == "ordering"
            )
            or is_sequence_question(
                state.question
            )
        )

        has_sequence_detail = any(
            evidence.action in {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }
            for evidence in state.evidence
        )

        if (
            needs_fine_detail
            and local_ids
            and not has_fine_detail
        ):
            best = self.best_local_evidence(
                state
            )
            action = "VERIFY_DETAIL"
            if best is not None:
                decision[
                    "evidence_id"
                ] = best[0]

        elif profiler_policy_mode == "hard":
            if (
                (
                    requirement == "global"
                    or evidence_type == "sequence"
                )
                and not has_global
            ):
                action = "GLOBAL_SCAN"

            elif (
                relation == "ordering"
                and action in {
                    "SEARCH_BEFORE",
                    "SEARCH_AFTER",
                }
            ):
                action = (
                    "GLOBAL_SCAN"
                    if not has_global
                    else "IMAGE_CAPTION"
                )

            elif (
                (
                    requirement == "global"
                    or evidence_type == "sequence"
                )
                and has_global
                and action == "SEARCH_LOCAL"
            ):
                action = "IMAGE_CAPTION"

            elif (
                needs_sequence_detail
                and has_global
                and not has_sequence_detail
            ):
                action = "IMAGE_CAPTION"
                decision[
                    "evidence_id"
                ] = None
                decision[
                    "start_s"
                ] = None
                decision[
                    "end_s"
                ] = None

        decision[
            "action"
        ] = action

        decision[
            "tool"
        ] = action

        if not decision.get(
            "query"
        ):
            decision[
                "query"
            ] = state.question

        return decision


    # ========================================================
    # Evidence selection
    # ========================================================

    def evidence_relevance_score(
        self,
        evidence: Evidence,
    ):
        score = 0

        if evidence.confidence == "high":
            score += 2

        elif evidence.confidence == "medium":
            score += 1

        observation = (
            evidence
            .observation
            .lower()
        )

        negative_phrases = (
            "not visible",
            "cannot determine",
            "cannot identify",
            "unclear",
            "not present",
            "no evidence",
        )

        if any(
            phrase in observation
            for phrase
            in negative_phrases
        ):
            score -= 2

        return score


    def best_evidence(
        self,
        state,
        *,
        actions=None,
    ):
        if actions is None:
            actions = VALID_ACTIONS

        candidates = [
            (
                index,
                evidence,
            )

            for index, evidence
            in enumerate(
                state.evidence
            )

            if evidence.action
            in actions
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item:
                self.evidence_relevance_score(
                    item[1]
                ),
        )

    def best_uncaptioned_local_evidence(
        self,
        state,
    ):
        captioned_ranges = [
            (
                evidence.start_s,
                evidence.end_s,
            )
            for evidence in state.evidence
            if evidence.action in {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }
        ]

        candidates = []

        for index, evidence in enumerate(
            state.evidence
        ):
            if evidence.action != "SEARCH_LOCAL":
                continue

            already_captioned = any(
                abs(evidence.start_s - start_s) <= 2.0
                and abs(evidence.end_s - end_s) <= 2.0
                for start_s, end_s
                in captioned_ranges
            )

            if not already_captioned:
                candidates.append(
                    (
                        index,
                        evidence,
                    )
                )

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item:
            self.evidence_relevance_score(
                item[1]
            ),
        )

    def best_local_evidence(
        self,
        state,
    ):
        candidates = [
            (
                index,
                evidence,
            )
            for index, evidence
            in enumerate(
                state.evidence
            )
            if evidence.action == "SEARCH_LOCAL"
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item:
                self.evidence_relevance_score(
                    item[1]
                ),
        )

    def assessment_support_text(
        self,
        assessment,
        state,
    ):
        if state is None:
            return ""

        chunks = []

        for value in assessment.get(
            "support_evidence_ids",
            [],
        ) or []:
            try:
                index = int(
                    value
                )
            except (TypeError, ValueError):
                continue

            if not (
                0 <= index < len(
                    state.evidence
                )
            ):
                continue

            evidence = state.evidence[
                index
            ]
            chunks.append(
                str(
                    evidence.observation
                )
            )

        segment_ids = set()
        for value in assessment.get(
            "support_segment_ids",
            [],
        ) or []:
            try:
                segment_ids.add(
                    int(
                        value
                    )
                )
            except (TypeError, ValueError):
                continue

        for segment in (
            state.memory_bank.get(
                "segment_captions",
                [],
            )
            or []
        ):
            try:
                segment_id = int(
                    segment.get(
                        "segment_id",
                        -1,
                    )
                )
            except (TypeError, ValueError):
                continue

            if segment_id in segment_ids:
                chunks.append(
                    json.dumps(
                        segment,
                        ensure_ascii=False,
                    )
                )

        return "\n".join(
            chunks
        )

    def assessment_has_option_alignment(
        self,
        *,
        assessment,
        choices,
        answer,
        state,
        confidence,
    ):
        answer_index = ord(
            answer
        ) - ord("A")

        if not (
            0 <= answer_index < len(
                choices
            )
        ):
            return False

        selected_choice = choices[
            answer_index
        ]

        reason_text = str(
            assessment.get(
                "reason",
                "",
            )
            or ""
        )

        missing_text = str(
            assessment.get(
                "missing_information",
                "",
            )
            or ""
        )

        combined_text = (
            reason_text
            + "\n"
            + missing_text
        )

        mentioned_labels = (
            supported_labels_mentioned_in_text(
                combined_text
            )
            & valid_choice_labels(
                choices
            )
        )

        if (
            mentioned_labels
            and answer not in mentioned_labels
        ):
            return False

        support_text = self.assessment_support_text(
            assessment,
            state,
        )

        skills = (
            infer_question_skills(
                state.question
            )
            if state is not None
            else {
                "GLOBAL_SUMMARY"
            }
        )

        option_supported = choice_text_supported_by_text(
            selected_choice,
            reason_text + "\n" + support_text,
        )

        support_option_supported = (
            choice_text_supported_by_text(
                selected_choice,
                support_text,
            )
            if support_text
            else False
        )

        is_global_summary_only = (
            skills == {
                "GLOBAL_SUMMARY"
            }
        )

        if (
            confidence < 5.0
            and text_has_uncertain_support(
                combined_text
            )
        ):
            return False

        if is_global_summary_only:
            return True

        needs_option_support = bool(
            skills
            & {
                "LOCAL_DETAIL",
                "SEQUENCE_ORDER",
                "RECURRENCE",
                "COUNTING",
                "TEMPORAL_BEFORE",
            }
        )

        if (
            needs_option_support
            and confidence < 5.0
            and not option_supported
        ):
            return False

        if (
            "CAUSAL_WHY" in skills
            and not (
                support_option_supported
                or (
                    confidence >= 5.0
                    and option_supported
                )
            )
        ):
            return False

        if (
            "TEMPORAL_AFTER" in skills
            or "TEMPORAL_BEFORE" in skills
        ):
            temporal_actions = {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }

            if "TEMPORAL_AFTER" in skills:
                temporal_actions.add(
                    "SEARCH_AFTER"
                )

            has_temporal_evidence = False

            for value in assessment.get(
                "support_evidence_ids",
                [],
            ) or []:
                try:
                    index = int(
                        value
                    )
                except (TypeError, ValueError):
                    continue

                if not (
                    0 <= index < len(
                        state.evidence
                    )
                ):
                    continue

                if state.evidence[
                    index
                ].action in temporal_actions:
                    has_temporal_evidence = True
                    break

            if not has_temporal_evidence:
                return False

            if text_has_uncertain_support(
                combined_text
            ):
                return False

            if not (
                support_option_supported
                or (
                    confidence >= 5.0
                    and option_supported
                )
            ):
                return False

        return True

    def assessment_has_supported_answer(
        self,
        assessment,
        choices,
        state=None,
    ):
        answer = normalize_answer(
            assessment.get(
                "answer"
            )
        )

        if answer not in valid_choice_labels(
            choices
        ):
            return False

        confidence = float(
            assessment.get(
                "confidence",
                0.0,
            )
            or 0.0
        )

        support_evidence_ids = (
            assessment.get(
                "support_evidence_ids"
            )
            or []
        )

        support_segment_ids = (
            assessment.get(
                "support_segment_ids"
            )
            or []
        )

        missing_information = str(
            assessment.get(
                "missing_information",
                "",
            )
            or ""
        ).strip()

        if confidence < 4.0:
            return False

        if missing_information:
            return False

        if not self.assessment_has_option_alignment(
            assessment=
                assessment,

            choices=
                choices,

            answer=
                answer,

            state=
                state,

            confidence=
                confidence,
        ):
            return False

        skills = (
            infer_question_skills(
                state.question
            )
            if state is not None
            else set()
        )

        needs_recurrence = (
            state is not None
            and "RECURRENCE" in skills
        )

        needs_sequence = (
            state is not None
            and "SEQUENCE_ORDER" in skills
        )

        needs_fine_detail = (
            state is not None
            and "LOCAL_DETAIL" in skills
            and not needs_recurrence
        )

        if needs_fine_detail:
            if confidence < 5.0:
                return False

            fine_actions = {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }

            for value in support_evidence_ids:
                try:
                    index = int(
                        value
                    )
                except (TypeError, ValueError):
                    continue

                if not (
                    0 <= index < len(
                        state.evidence
                    )
                ):
                    continue

                if (
                    state.evidence[
                        index
                    ].action
                    in fine_actions
                ):
                    return True

            return False

        if needs_recurrence:
            distinct_windows = set()

            for value in support_evidence_ids:
                try:
                    index = int(
                        value
                    )
                except (TypeError, ValueError):
                    continue

                if not (
                    0 <= index < len(
                        state.evidence
                    )
                ):
                    continue

                evidence = state.evidence[
                    index
                ]

                if evidence.action not in {
                    "SEARCH_LOCAL",
                    "IMAGE_CAPTION",
                    "VERIFY_DETAIL",
                    "ZOOM_CAPTION",
                }:
                    continue

                distinct_windows.add(
                    (
                        round(
                            float(
                                evidence.start_s
                            ),
                            1,
                        ),
                        round(
                            float(
                                evidence.end_s
                            ),
                            1,
                        ),
                    )
                )

            if len(
                distinct_windows
            ) < 2:
                return False

        if needs_sequence:
            sequence_evidence = [
                evidence
                for evidence in state.evidence
                if evidence.action in {
                    "GLOBAL_SCAN",
                    "IMAGE_CAPTION",
                    "VERIFY_DETAIL",
                    "ZOOM_CAPTION",
                }
            ]

            if len(
                sequence_evidence
            ) < 3:
                return False

        return bool(
            support_evidence_ids
            or support_segment_ids
        )


    def resolve_anchor(
        self,
        state,
        decision,
    ):
        evidence_id = (
            decision.get(
                "evidence_id"
            )
        )

        try:
            evidence_id = int(
                evidence_id
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not (
            0 <= evidence_id
            < len(state.evidence)
        ):
            return None

        return (
            evidence_id,
            state.evidence[
                evidence_id
            ],
        )


    # ========================================================
    # Budget helpers
    # ========================================================

    def allowed(
        self,
        state,
        action,
    ):
        return (
            action
            in state.policy.get(
                "allowed_actions",
                [],
            )
        )


    def action_budget_exhausted(
        self,
        state,
        action,
    ):
        policy = state.policy

        if action == "SEARCH_LOCAL":

            return (
                state.local_search_count
                >= int(
                    policy.get(
                        "max_local_searches",
                        0,
                    )
                )
            )

        if action == "GLOBAL_SCAN":

            return (
                state.global_scan_count
                >= int(
                    policy.get(
                        "max_global_scans",
                        0,
                    )
                )
            )

        if action == "COUNT_EVENTS":

            return (
                state.count_scan_count
                >= int(
                    policy.get(
                        "max_count_scans",
                        0,
                    )
                )
            )

        if action in {
            "SEARCH_BEFORE",
            "SEARCH_AFTER",
        }:

            return (
                state.temporal_search_count
                >= int(
                    policy.get(
                        "max_temporal_searches",
                        0,
                    )
                )
            )

        if action == "INCREASE_DENSITY":

            return (
                state.density_count
                >= int(
                    policy.get(
                        "max_density_refinements",
                        0,
                    )
                )
            )

        if action == "VERIFY_DETAIL":

            return (
                state.verify_count
                >= int(
                    policy.get(
                        "max_contrastive_checks",
                        0,
                    )
                )
            )

        return False


    # ========================================================
    # Initial action from PROFILER
    # ========================================================

    def choose_initial_decision(
        self,
        *,
        state,
    ):
        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "local",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        evidence_type = str(
            state.policy.get(
                "evidence_type",
                "",
            )
        )

        # ----------------------------------------------------
        # Count
        # ----------------------------------------------------

        if requirement == "count":

            return {
                "action":
                    "COUNT_EVENTS",

                "query":
                    state.question,

                "reason":
                    "Profiler requires count-complete "
                    "whole-video evidence.",
            }

        # ----------------------------------------------------
        # Global
        # ----------------------------------------------------

        if is_recurrence_question(
            state.question
        ):

            return {
                "action":
                    "SEARCH_LOCAL",

                "query":
                    state.question,

                "reason":
                    "Recurrence question requires multiple "
                    "candidate occurrence windows.",
            }

        if is_sequence_question(
            state.question
        ):

            return {
                "action":
                    "IMAGE_CAPTION",

                "query":
                    "caption chronological segment for ordered timeline",

                "reason":
                    "Lexical sequence question requires ordered "
                    "timeline evidence.",
            }

        if requirement == "global":

            return {
                "action":
                    "GLOBAL_SCAN",

                "query":
                    state.question,

                "reason":
                    "Profiler requires chronological "
                    "global evidence.",
            }

        # ----------------------------------------------------
        # Before / after
        # ----------------------------------------------------

        if (
            requirement == "temporal"
            and relation in {
                "before",
                "after",
            }
        ):

            anchor_query = (
                derive_temporal_anchor_query(
                    state.question,
                    relation,
                )
            )

            return {
                "action":
                    "SEARCH_LOCAL",

                "query":
                    anchor_query,

                "reason":
                    "First localize the anchor; "
                    "temporal relation is applied afterward.",
            }

        # ----------------------------------------------------
        # Ordering / first-last occurrence
        # ----------------------------------------------------

        if (
            requirement == "temporal"
            and relation == "ordering"
        ):

            if (
                evidence_type
                == "first_last_occurrence"
                and self.allowed(
                    state,
                    "GLOBAL_SCAN",
                )
            ):

                return {
                    "action":
                        "GLOBAL_SCAN",

                    "query":
                        state.question,

                    "reason":
                        "First/last occurrence requires "
                        "chronological coverage.",
                }

            return {
                "action":
                    "SEARCH_LOCAL",

                "query":
                    state.question,

                "reason":
                    "Locate candidate temporal events.",
            }

        # ----------------------------------------------------
        # Agentic fallback
        # ----------------------------------------------------

        return {
            "action":
                "SEARCH_LOCAL",

            "query":
                state.question,

            "reason":
                "Profiler selected agentic localized evidence.",
        }


    # ========================================================
    # Deterministic temporal follow-up
    # ========================================================

    def required_temporal_followup(
        self,
        state,
        *,
        evidence_start_index,
    ):
        """
        This is the key difference from ordinary retrieval.

        Retrieval locates the anchor.

        The runtime explicitly applies BEFORE/AFTER semantics.
        """

        if (
            state.policy.get(
                "evidence_requirement"
            )
            != "temporal"
        ):
            return None

        relation = state.policy.get(
            "temporal_relation"
        )

        if relation not in {
            "before",
            "after",
        }:
            return None

        new_candidates = []

        for index in range(
            evidence_start_index,
            len(state.evidence),
        ):

            evidence = (
                state.evidence[
                    index
                ]
            )

            if (
                evidence.action
                == "SEARCH_LOCAL"
            ):
                new_candidates.append(
                    (
                        index,
                        evidence,
                    )
                )

        if not new_candidates:
            return None

        evidence_id, _ = max(
            new_candidates,

            key=lambda item:
                self.evidence_relevance_score(
                    item[1]
                ),
        )

        action = (
            "SEARCH_BEFORE"
            if relation == "before"
            else "SEARCH_AFTER"
        )

        if not self.allowed(
            state,
            action,
        ):
            return None

        #CABALITY EXP: Do not bound temporal followups.
        # if self.action_budget_exhausted(
        #     state,
        #     action,
        # ):
        #     return None

        return {
            "action":
                action,

            "evidence_id":
                evidence_id,

            "query":
                state.question,

            "reason":
                f"Profiler requires explicit "
                f"{relation} evidence relative "
                f"to localized anchor.",
        }


    # ========================================================
    # Normalize controller output
    # ========================================================

    def normalize_decision(
        self,
        *,
        decision,
        state,
    ):
        decision = dict(
            decision
            or {}
        )

        action = str(
            decision.get(
                "action",
                "ANSWER",
            )
        ).upper()

        if action == "COMPARE_CHOICES":
            action = "VERIFY_DETAIL"

        if action not in VALID_ACTIONS:
            action = "ANSWER"

        coverage_gap = (
            self.controller
            .coverage_gap(
                state
            )
        )

        if (
            coverage_gap is not None
        ):
            decision.update(
                coverage_gap
            )
            action = str(
                coverage_gap.get(
                    "tool",
                    coverage_gap.get(
                        "action",
                        action,
                    ),
                )
            ).upper()

        profiler_policy_mode = str(
            state.policy.get(
                "profiler_policy_mode",
                "hint",
            )
        ).lower()

        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "",
            )
        )

        evidence_type = str(
            state.policy.get(
                "evidence_type",
                "",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        has_global = any(
            evidence.action == "GLOBAL_SCAN"
            for evidence in state.evidence
        )

        needs_fine_detail = (
            evidence_type in {
                "local_detail",
                "generic_local_mcq",
            }
            or is_fine_detail_question(
                state.question
            )
        )

        needs_recurrence = is_recurrence_question(
            state.question
        )

        if needs_recurrence:
            needs_fine_detail = False

        local_ids = [
            index
            for index, evidence
            in enumerate(
                state.evidence
            )
            if evidence.action == "SEARCH_LOCAL"
        ]

        has_fine_detail = any(
            evidence.action in {
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
                "IMAGE_CAPTION",
                "OBJECT_DETECTION",
                "OBJECT_TRACKING",
            }
            for evidence in state.evidence
        )

        needs_sequence_detail = (
            evidence_type == "sequence"
            or (
                requirement == "global"
                and relation == "ordering"
            )
            or is_sequence_question(
                state.question
            )
        )

        has_sequence_detail = any(
            evidence.action in {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }
            for evidence in state.evidence
        )

        recurrence_caption_count = sum(
            1
            for evidence in state.evidence
            if evidence.action in {
                "IMAGE_CAPTION",
                "VERIFY_DETAIL",
                "ZOOM_CAPTION",
            }
        )

        if (
            action == "ANSWER"
            and needs_recurrence
            and (
                not local_ids
                or recurrence_caption_count < min(
                    3,
                    len(
                        local_ids
                    ),
                )
            )
        ):
            action = (
                "SEARCH_LOCAL"
                if not local_ids
                else "IMAGE_CAPTION"
            )
            decision[
                "query"
            ] = state.question
            decision[
                "reason"
            ] = (
                "recurrence question cannot answer until multiple "
                "candidate occurrence windows are inspected"
            )

        elif (
            action == "ANSWER"
            and needs_sequence_detail
            and len(
                [
                    evidence
                    for evidence in state.evidence
                    if evidence.action in {
                        "IMAGE_CAPTION",
                        "VERIFY_DETAIL",
                        "ZOOM_CAPTION",
                        "GLOBAL_SCAN",
                    }
                ]
            ) < 3
        ):
            action = "IMAGE_CAPTION"
            decision[
                "query"
            ] = (
                "caption the next chronological segment and preserve "
                "event order for sequence comparison"
            )
            decision[
                "reason"
            ] = (
                "sequence question needs at least three ordered "
                "timeline evidence items before answering"
            )

        if (
            action != "ANSWER"
            and needs_recurrence
            and not local_ids
        ):
            action = "SEARCH_LOCAL"
            decision[
                "query"
            ] = state.question
            decision[
                "reason"
            ] = (
                "recurrence question needs multiple candidate "
                "occurrence windows"
            )

        elif (
            action != "ANSWER"
            and needs_recurrence
            and local_ids
            and recurrence_caption_count < min(
                3,
                len(
                    local_ids
                ),
            )
        ):
            best = self.best_uncaptioned_local_evidence(
                state
            )

            if best is not None:
                evidence_id, _ = best
                action = "IMAGE_CAPTION"
                decision[
                    "evidence_id"
                ] = evidence_id
                decision.pop(
                    "start_s",
                    None,
                )
                decision.pop(
                    "end_s",
                    None,
                )
                decision[
                    "query"
                ] = (
                    "caption this candidate occurrence and keep "
                    "its timestamp for recurrence comparison"
                )
                decision[
                    "reason"
                ] = (
                    "recurrence question requires multiple "
                    "timestamped occurrences before answering"
                )

        elif (
            action != "ANSWER"
            and needs_sequence_detail
            and not has_sequence_detail
        ):
            action = "IMAGE_CAPTION"
            decision.pop(
                "evidence_id",
                None,
            )
            decision.pop(
                "start_s",
                None,
            )
            decision.pop(
                "end_s",
                None,
            )
            decision[
                "query"
            ] = (
                "caption the next chronological segment and preserve "
                "the order of visible events for sequence comparison"
            )
            decision[
                "reason"
            ] = (
                "sequence question needs ordered timeline evidence "
                "before local semantic search"
            )

        elif (
            action == "SEARCH_LOCAL"
            and needs_sequence_detail
            and has_sequence_detail
        ):
            action = "IMAGE_CAPTION"
            decision.pop(
                "evidence_id",
                None,
            )
            decision.pop(
                "start_s",
                None,
            )
            decision.pop(
                "end_s",
                None,
            )
            decision[
                "query"
            ] = (
                "caption another distinct chronological segment to "
                "verify event order against the choices"
            )
            decision[
                "reason"
            ] = (
                "continue building chronological timeline instead "
                "of repeating semantic local search"
            )

        elif (
            action != "ANSWER"
            and needs_fine_detail
            and local_ids
            and not has_fine_detail
        ):
            best = self.best_local_evidence(
                state
            )
            action = "VERIFY_DETAIL"
            if best is not None:
                decision[
                    "evidence_id"
                ] = best[0]

        elif (
            action == "SEARCH_LOCAL"
            and local_ids
        ):
            best = self.best_uncaptioned_local_evidence(
                state
            )

            if best is not None:
                evidence_id, _ = best
                action = "IMAGE_CAPTION"
                decision[
                    "evidence_id"
                ] = evidence_id
                decision.pop(
                    "start_s",
                    None,
                )
                decision.pop(
                    "end_s",
                    None,
                )
                decision[
                    "reason"
                ] = (
                    "local evidence already exists; caption the best "
                    "uncaptioned retrieved window instead of repeating "
                    "semantic search"
                )

        elif (
            action != "ANSWER"
            and profiler_policy_mode == "hard"
        ):
            if (
                (
                    requirement == "global"
                    or evidence_type == "sequence"
                )
                and not has_global
            ):
                action = "GLOBAL_SCAN"

            elif (
                relation == "ordering"
                and action in {
                    "SEARCH_BEFORE",
                    "SEARCH_AFTER",
                }
            ):
                action = (
                    "GLOBAL_SCAN"
                    if not has_global
                    else "IMAGE_CAPTION"
                )

            elif (
                (
                    requirement == "global"
                    or evidence_type == "sequence"
                )
                and has_global
                and action == "SEARCH_LOCAL"
            ):
                action = "IMAGE_CAPTION"

            elif (
                needs_sequence_detail
                and has_global
                and not has_sequence_detail
            ):
                action = "IMAGE_CAPTION"
                decision[
                    "evidence_id"
                ] = None
                decision[
                    "start_s"
                ] = None
                decision[
                    "end_s"
                ] = None

        # CAPABILITY EXPERIMENT:
        # DO NOT apply profiler per-action budgets.
        # if (
        #     action != "ANSWER"
        #     and self.action_budget_exhausted(
        #         state,
        #         action,
        #     )
        # ):
        #     action = "ANSWER"

        decision[
            "action"
        ] = action

        decision[
            "tool"
        ] = action

        if action in {
            "SEARCH_BEFORE",
            "SEARCH_AFTER",
            "VERIFY_DETAIL",
        }:

            # VERIFY_DETAIL can use the existing active
            # refinement anchor even if the controller
            # does not return an evidence_id.
            has_active_anchor = (
                action == "VERIFY_DETAIL"
                and state.active_evidence_id is not None
                and 0 <= state.active_evidence_id < len(state.evidence)
            )

            if (
                not has_active_anchor
                and self.resolve_anchor(
                    state,
                    decision,
                ) is None
            ):
                return {
                    "action": "SEARCH_LOCAL",
                    "query": decision.get(
                        "query",
                        state.question,
                    ),
                    "reason":
                        "Requested operation needs a localized "
                        "anchor; localize it first.",
                }

        return decision


    # ========================================================
    # Local semantic search
    # ========================================================

    def execute_local_search(
        self,
        *,
        state,
        video,
        query,
    ):
        state.local_search_count += 1
        state.active_evidence_id = None

        probe_fps = float(
            state.policy.get(
                "probe_fps",
                0.05,
            )
        )

        topk = (
            state.policy.get(
                "action_topk"
            )
            or
            state.policy.get(
                "probe_topk"
            )
            or
            4
        )

        window_len_s = float(
            state.policy.get(
                "window_len_s",
                16.0,
            )
        )

        frames_per_window = int(
            state.policy.get(
                "high_frames_per_window",
                8,
            )
        )

        ranked = (
            self.semantic_candidates(
                video=
                    video,

                query=
                    query,

                probe_fps=
                    probe_fps,

                topk=
                    int(topk),
            )
        )

        if not ranked:
            return

        # Agentic localization should inspect a small number of
        # strongest anchor hypotheses, not blindly all top-k.
        num_candidates = min(
            6,
            len(ranked),
        )

        print(
            "    local candidates:",
            [
                (
                    round(
                        item[
                            "timestamp_s"
                        ],
                        2,
                    ),
                    round(
                        item[
                            "score"
                        ],
                        4,
                    ),
                )
                for item
                in ranked[
                    :num_candidates
                ]
            ],
        )

        for candidate in ranked[
            :num_candidates
        ]:

            anchor = float(
                candidate[
                    "timestamp_s"
                ]
            )

            half = (
                window_len_s
                / 2.0
            )

            start_s = max(
                0.0,
                anchor - half,
            )

            end_s = min(
                video.duration_s,
                anchor + half,
            )

            already_seen = any(
                same_range(
                    evidence.start_s,
                    evidence.end_s,
                    start_s,
                    end_s,
                    tolerance_s=
                        2.0,
                )

                for evidence
                in state.evidence
            )

            if already_seen:
                continue

            evidence = self.inspect_window(
                video=
                    video,

                question=
                    state.question,

                action=
                    "SEARCH_LOCAL",

                query=
                    query,

                start_s=
                    start_s,

                end_s=
                    end_s,

                num_frames=
                    frames_per_window,
            )

            if evidence is not None:

                state.evidence.append(
                    evidence
                )

                state.total_latency_s += (
                    evidence.latency_s
                )


    # ========================================================
    # Global scan
    # ========================================================

    def execute_global_scan(
        self,
        *,
        state,
        video,
        query,
    ):
        state.global_scan_count += 1
        state.active_evidence_id = None
        probe_fps = float(
            state.policy.get(
                "probe_fps",
                0.015625,
            )
        )

        base_frames = int(
            math.ceil(
                video.duration_s
                * max(
                    probe_fps,
                    0.10,
                )
            )
        )

        base_frames = max(
            24,
            min(
                128,
                base_frames,
            ),
        )

        num_frames = min(
            192,
            base_frames
            * state.global_scan_count,
        )

        timestamps, frames = (
            video.sample_range(
                0.0,
                video.duration_s,
                num_frames,
            )
        )

        chunk_results = []

        total_latency = 0.0

        for start in range(
            0,
            len(frames),
            self.global_chunk_size,
        ):

            end = min(
                start
                + self.global_chunk_size,
                len(frames),
            )

            chunk_ts = (
                timestamps[
                    start:
                    end
                ]
            )

            chunk_frames = (
                frames[
                    start:
                    end
                ]
            )

            result = (
                self.inspector
                .inspect(
                    question=
                        state.question,

                    timestamps=
                        chunk_ts,

                    frames=
                        chunk_frames,

                    action=
                        "GLOBAL_SCAN_CHUNK",

                    query=
                        query,
                )
            )

            total_latency += float(
                result[
                    "latency_s"
                ]
            )

            chunk_results.append(
                Evidence(
                    action=
                        "GLOBAL_SCAN",

                    query=
                        query,

                    start_s=
                        float(
                            chunk_ts[0]
                        ),

                    end_s=
                        float(
                            chunk_ts[-1]
                        ),

                    timestamps=
                        chunk_ts,

                    observation=
                        result[
                            "observation"
                        ],

                    confidence=
                        result[
                            "confidence"
                        ],

                    latency_s=
                        result[
                            "latency_s"
                        ],
                )
            )

        combined = (
            self.inspector
            .aggregate_evidence(
                question=
                    state.question,

                evidence=
                    chunk_results,

                purpose=
                    "chronological whole-video coverage",
            )
        )

        total_latency += float(
            combined[
                "latency_s"
            ]
        )

        evidence = Evidence(
            action=
                "GLOBAL_SCAN",

            query=
                query,

            start_s=
                0.0,

            end_s=
                video.duration_s,

            timestamps=
                timestamps,

            observation=
                combined[
                    "observation"
                ],

            confidence=
                combined[
                    "confidence"
                ],

            latency_s=
                total_latency,
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            total_latency
        )


    # ========================================================
    # Count scan
    # ========================================================

    def execute_count_scan(
        self,
        *,
        state,
        video,
        query,
    ):
        state.count_scan_count += 1

        probe_fps = float(
            state.policy.get(
                "probe_fps",
                0.03125,
            )
        )

        num_frames = int(
            math.ceil(
                video.duration_s
                * probe_fps
            )
        )

        num_frames = max(
            24,
            min(
                160,
                num_frames,
            ),
        )

        timestamps, frames = (
            video.sample_range(
                0.0,
                video.duration_s,
                num_frames,
            )
        )

        candidates = []

        total_latency = 0.0

        chunk_size = max(
            4,
            self.global_chunk_size,
        )

        for start in range(
            0,
            len(frames),
            chunk_size,
        ):

            end = min(
                start
                + chunk_size,
                len(frames),
            )

            ts = (
                timestamps[
                    start:
                    end
                ]
            )

            fs = (
                frames[
                    start:
                    end
                ]
            )

            result = (
                self.inspector
                .inspect_count_chunk(
                    question=
                        state.question,

                    timestamps=
                        ts,

                    frames=
                        fs,
                )
            )

            total_latency += float(
                result[
                    "latency_s"
                ]
            )

            if result[
                "candidate"
            ]:

                candidates.append({
                    "start_s":
                        float(
                            ts[0]
                        ),

                    "end_s":
                        float(
                            ts[-1]
                        ),

                    "description":
                        result[
                            "description"
                        ],

                    "confidence":
                        result[
                            "confidence"
                        ],
                })

        aggregate = (
            self.inspector
            .aggregate_count(
                question=
                    state.question,

                candidates=
                    candidates,
            )
        )

        total_latency += float(
            aggregate[
                "latency_s"
            ]
        )

        evidence = Evidence(
            action=
                "COUNT_EVENTS",

            query=
                query,

            start_s=
                0.0,

            end_s=
                video.duration_s,

            timestamps=
                timestamps,

            observation=(
                f"{aggregate['observation']} "
                f"Estimated distinct count: "
                f"{aggregate.get('estimated_count')}."
            ),

            confidence=
                aggregate[
                    "confidence"
                ],

            latency_s=
                total_latency,
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            total_latency
        )


    # ========================================================
    # Temporal search
    # ========================================================

    def execute_temporal_search(
        self,
        *,
        state,
        video,
        decision,
        direction,
    ):
        resolved = (
            self.resolve_anchor(
                state,
                decision,
            )
        )

        if resolved is None:
            return

        _, anchor = resolved

        state.temporal_search_count += 1

        window_len_s = float(
            state.policy.get(
                "window_len_s",
                16.0,
            )
        )

        frames = int(
            state.policy.get(
                "high_frames_per_window",
                12,
            )
        )

        if direction == "before":

            end_s = float(
                anchor.start_s
            )

            start_s = max(
                0.0,
                end_s
                - window_len_s,
            )

            action = "SEARCH_BEFORE"

        else:

            start_s = float(
                anchor.end_s
            )

            end_s = min(
                video.duration_s,
                start_s
                + window_len_s,
            )

            action = "SEARCH_AFTER"

        if end_s <= start_s:
            return

        if self.range_already_seen(
            state=
                state,

            start_s=
                start_s,

            end_s=
                end_s,

            action=
                action,

            tolerance_s=
                2.0,
        ):
            fallback_start, fallback_end = (
                self.next_unseen_segment_range(
                    state=
                        state,

                    video=
                        video,
                )
            )

            fallback = dict(
                decision
            )
            fallback[
                "action"
            ] = "IMAGE_CAPTION"
            fallback[
                "tool"
            ] = "IMAGE_CAPTION"
            fallback[
                "start_s"
            ] = fallback_start
            fallback[
                "end_s"
            ] = fallback_end
            fallback[
                "query"
            ] = (
                "chronological context for unresolved "
                "ordering/sequence question"
            )

            self.execute_image_caption(
                state=
                    state,

                video=
                    video,

                decision=
                    fallback,
            )

            return

        evidence = self.inspect_window(
            video=
                video,

            question=
                state.question,

            action=
                action,

            query=
                decision.get(
                    "query",
                    state.question,
                ),

            start_s=
                start_s,

            end_s=
                end_s,

            num_frames=
                frames,
        )

        if evidence is not None:

            state.evidence.append(
                evidence
            )

            state.total_latency_s += (
                evidence.latency_s
            )


    # ========================================================
    # Density / verify
    # ========================================================

    def execute_refinement(
        self,
        *,
        state,
        video,
        decision,
        verify=False,
    ):

        # Stay on current refinement chain.
        if (
            state.active_evidence_id is not None
            and
            0 <= state.active_evidence_id < len(state.evidence)
        ):
            anchor_id = state.active_evidence_id
            anchor = state.evidence[anchor_id]

        else:
            resolved = self.resolve_anchor(
                state,
                decision,
            )

            if resolved is not None:
                anchor_id, anchor = resolved

            else:
                best = self.best_evidence(
                    state,
                    actions={
                        "SEARCH_LOCAL",
                        "INCREASE_DENSITY",
                        "VERIFY_DETAIL",
                    },
                )

                if best is None:
                    return

                anchor_id, anchor = best

            state.active_evidence_id = anchor_id

        coarse_len = (
            anchor.end_s
            - anchor.start_s
        )

        # Different geometry for coarse refinement
        # versus fine detail verification.
        if verify:
            state.verify_count += 1
            action = "VERIFY_DETAIL"

            candidate_count = max(
                32,
                min(
                    96,
                    int(
                        math.ceil(
                            coarse_len * 4.0
                        )
                    ),
                ),
            )

            refinement_window_s = 2.5
            num_frames = 24

        else:
            state.density_count += 1
            action = "INCREASE_DENSITY"

            candidate_count = max(
                12,
                min(
                    48,
                    int(
                        math.ceil(
                            coarse_len * 2.0
                        )
                    ),
                ),
            )

            refinement_window_s = 4.0
            num_frames = 16

        query = str(
            decision.get(
                "query",
                state.question,
            )
        )

        print(
            "    refinement query:",
            query,
        )

        # Search densely inside current interval.
        timestamps, frames = video.sample_range(
            anchor.start_s,
            anchor.end_s,
            candidate_count,
        )

        ranked = self.retriever.rank_frames(
            query=query,
            timestamps=timestamps,
            frames=frames,
            topk=5,
        )

        if not ranked:
            return

        print(
            "    refinement candidates:",
            [
                (
                    round(
                        item["timestamp_s"],
                        2,
                    ),
                    round(
                        item["score"],
                        4,
                    ),
                )
                for item in ranked
            ],
        )
        # Inspect several DISTINCT refinement hypotheses rather than
        # blindly trusting the single highest SigLIP score.

        max_refinement_windows = (
            3 if verify else 2
        )

        selected_windows = []

        for candidate in ranked:

            center_t = float(
                candidate["timestamp_s"]
            )

            start_s = max(
                anchor.start_s,
                center_t
                - refinement_window_s / 2.0,
            )

            end_s = min(
                anchor.end_s,
                center_t
                + refinement_window_s / 2.0,
            )

            duplicate = any(
                same_range(
                    start_s,
                    end_s,
                    old_start,
                    old_end,
                    tolerance_s=
                        refinement_window_s / 2.0,
                )
                for old_start, old_end
                in selected_windows
            )

            if duplicate:
                continue

            selected_windows.append(
                (
                    start_s,
                    end_s,
                )
            )

            if (
                len(selected_windows)
                >= max_refinement_windows
            ):
                break


        if not selected_windows:
            return


        print(
            "    selected refinement windows:",
            [
                (
                    round(start_s, 2),
                    round(end_s, 2),
                )
                for start_s, end_s
                in selected_windows
            ],
        )


        for start_s, end_s in selected_windows:

            evidence = self.inspect_window(
                video=video,
                question=state.question,
                action=action,
                query=query,
                start_s=start_s,
                end_s=end_s,
                num_frames=num_frames,
            )

            if evidence is None:
                continue

            state.evidence.append(
                evidence
            )

            new_evidence_id = (
                len(state.evidence) - 1
            )

            state.total_latency_s += (
                evidence.latency_s
            )

            print(
                "    refinement chain:",
                anchor_id,
                "->",
                new_evidence_id,
                f"[{evidence.start_s:.2f}, "
                f"{evidence.end_s:.2f}]",
            )


    def range_from_plan(
        self,
        *,
        state,
        video,
        decision,
    ):
        start_s = decision.get(
            "start_s"
        )

        end_s = decision.get(
            "end_s"
        )

        try:
            if start_s is not None and end_s is not None:
                start_s = float(
                    start_s
                )

                end_s = float(
                    end_s
                )

                if end_s > start_s:
                    return (
                        max(
                            0.0,
                            start_s,
                        ),
                        min(
                            video.duration_s,
                            end_s,
                        ),
                    )
        except (TypeError, ValueError):
            pass

        resolved = self.resolve_anchor(
            state,
            decision,
        )

        if resolved is not None:
            _, evidence = resolved
            return (
                evidence.start_s,
                evidence.end_s,
            )

        best = self.best_evidence(
            state,
            actions={
                "SEARCH_LOCAL",
                "GLOBAL_SCAN",
                "INCREASE_DENSITY",
                "VERIFY_DETAIL",
                "IMAGE_CAPTION",
                "ZOOM_CAPTION",
            },
        )

        if best is not None:
            _, evidence = best
            return (
                evidence.start_s,
                evidence.end_s,
            )

        window_len_s = float(
            state.policy.get(
                "window_len_s",
                16.0,
            )
        )

        return (
            0.0,
            min(
                video.duration_s,
                window_len_s,
            ),
        )


    def range_already_seen(
        self,
        *,
        state,
        start_s,
        end_s,
        action=None,
        tolerance_s=1.0,
    ):
        for evidence in state.evidence:
            if (
                action is not None
                and evidence.action != action
            ):
                continue

            if same_range(
                evidence.start_s,
                evidence.end_s,
                start_s,
                end_s,
                tolerance_s=tolerance_s,
            ):
                return True

        return False


    def next_unseen_segment_range(
        self,
        *,
        state,
        video,
    ):
        segments = (
            state.memory_bank.get(
                "segment_captions",
                [],
            )
            or []
        )

        for segment in segments:
            try:
                start_s = float(
                    segment[
                        "start_s"
                    ]
                )
                end_s = float(
                    segment[
                        "end_s"
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue

            if not self.range_already_seen(
                state=
                    state,

                start_s=
                    start_s,

                end_s=
                    end_s,

                tolerance_s=
                    2.0,
            ):
                return (
                    start_s,
                    min(
                        video.duration_s,
                        end_s,
                    ),
                )

        window_len_s = float(
            state.policy.get(
                "window_len_s",
                16.0,
            )
        )

        for start_s in np.linspace(
            0.0,
            max(
                0.0,
                video.duration_s
                - window_len_s,
            ),
            8,
        ):
            end_s = min(
                video.duration_s,
                float(start_s)
                + window_len_s,
            )

            if not self.range_already_seen(
                state=
                    state,

                start_s=
                    float(start_s),

                end_s=
                    end_s,

                tolerance_s=
                    2.0,
            ):
                return (
                    float(start_s),
                    end_s,
                )

        return (
            0.0,
            min(
                video.duration_s,
                window_len_s,
            ),
        )


    def execute_image_caption(
        self,
        *,
        state,
        video,
        decision,
    ):
        state.image_caption_count += 1

        explicit_range = (
            decision.get(
                "start_s"
            ) is not None
            and decision.get(
                "end_s"
            ) is not None
        )

        explicit_anchor = (
            decision.get(
                "evidence_id"
            ) is not None
        )

        if (
            not explicit_range
            and not explicit_anchor
            and (
                state.policy.get(
                    "evidence_requirement"
                ) == "global"
                or state.policy.get(
                    "evidence_type"
                ) == "sequence"
                or is_sequence_question(
                    state.question
                )
            )
        ):
            start_s, end_s = (
                self.next_unseen_segment_range(
                    state=
                        state,

                    video=
                        video,
                )
            )

        else:
            start_s, end_s = self.range_from_plan(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,
            )

        if end_s <= start_s:
            return

        num_frames = int(
            decision.get(
                "num_frames",
                state.policy.get(
                    "high_frames_per_window",
                    8,
                ),
            )
            or 8
        )

        timestamps, frames = video.sample_range(
            start_s,
            end_s,
            num_frames,
        )

        result = (
            self.inspector
            .caption_frames(
                question=
                    state.question,

                timestamps=
                    timestamps,

                frames=
                    frames,

                purpose=
                    str(
                        decision.get(
                            "query",
                            state.question,
                        )
                    ),
            )
        )

        score = confidence_to_score(
            result.get(
                "confidence_score"
            )
        )

        evidence = Evidence(
            action=
                "IMAGE_CAPTION",

            query=
                str(
                    decision.get(
                        "query",
                        state.question,
                    )
                ),

            start_s=
                start_s,

            end_s=
                end_s,

            timestamps=
                timestamps,

            observation=
                result[
                    "caption"
                ],

            confidence=
                "high"
                if score and score >= 0.75
                else "medium"
                if score and score >= 0.45
                else "low",

            latency_s=
                float(
                    result[
                        "latency_s"
                    ]
                ),

            confidence_score=
                score,

            uncertainty=
                score_to_uncertainty(
                    score
                ),

            tool_name=
                "IMAGE_CAPTION",

            metadata={
                "purpose":
                    decision.get(
                        "query",
                        state.question,
                    ),
            },
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            evidence.latency_s
        )


    def execute_zoom_caption(
        self,
        *,
        state,
        video,
        decision,
    ):
        state.zoom_caption_count += 1

        start_s, end_s = self.range_from_plan(
            state=
                state,

            video=
                video,

            decision=
                decision,
        )

        if end_s <= start_s:
            return

        bbox = decision.get(
            "bbox"
        )

        num_frames = int(
            decision.get(
                "num_frames",
                6,
            )
            or 6
        )

        timestamps, frames = video.sample_range(
            start_s,
            end_s,
            num_frames,
        )

        zoomed = []

        for frame in frames:
            if (
                isinstance(
                    bbox,
                    list,
                )
                and len(bbox) == 4
            ):
                width, height = frame.size

                try:
                    x1, y1, x2, y2 = [
                        float(value)
                        for value in bbox
                    ]
                except (TypeError, ValueError):
                    x1 = y1 = 0.0
                    x2 = y2 = 1.0

                if max(
                    x1,
                    y1,
                    x2,
                    y2,
                ) <= 1.5:
                    x1 *= width
                    x2 *= width
                    y1 *= height
                    y2 *= height

                left = max(
                    0,
                    int(
                        min(
                            x1,
                            x2,
                        )
                    ),
                )

                upper = max(
                    0,
                    int(
                        min(
                            y1,
                            y2,
                        )
                    ),
                )

                right = min(
                    width,
                    int(
                        max(
                            x1,
                            x2,
                        )
                    ),
                )

                lower = min(
                    height,
                    int(
                        max(
                            y1,
                            y2,
                        )
                    ),
                )

                if right > left and lower > upper:
                    crop = frame.crop(
                        (
                            left,
                            upper,
                            right,
                            lower,
                        )
                    )
                else:
                    crop = frame
            else:
                crop = frame

            zoomed.append(
                crop.resize(
                    frame.size,
                    Image.Resampling.LANCZOS,
                )
            )

        target = str(
            decision.get(
                "target",
                decision.get(
                    "query",
                    state.question,
                ),
            )
        )

        result = (
            self.inspector
            .caption_frames(
                question=
                    state.question,

                timestamps=
                    timestamps,

                frames=
                    zoomed,

                purpose=
                    "zoomed inspection: "
                    + target,
            )
        )

        score = confidence_to_score(
            result.get(
                "confidence_score"
            )
        )

        evidence = Evidence(
            action=
                "ZOOM_CAPTION",

            query=
                target,

            start_s=
                start_s,

            end_s=
                end_s,

            timestamps=
                timestamps,

            observation=
                result[
                    "caption"
                ],

            confidence=
                "high"
                if score and score >= 0.75
                else "medium"
                if score and score >= 0.45
                else "low",

            latency_s=
                float(
                    result[
                        "latency_s"
                    ]
                ),

            confidence_score=
                score,

            uncertainty=
                score_to_uncertainty(
                    score
                ),

            tool_name=
                "ZOOM_CAPTION",

            metadata={
                "bbox":
                    bbox,

                "target":
                    target,
            },
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            evidence.latency_s
        )


    def execute_object_detection(
        self,
        *,
        state,
        video,
        decision,
    ):
        if self.object_tools is None:
            fallback = dict(
                decision
            )
            fallback[
                "query"
            ] = (
                "OBJECT_DETECTION fallback: "
                + str(
                    decision.get(
                        "query",
                        state.question,
                    )
                )
            )
            state.detection_count += 1
            self.execute_zoom_caption(
                state=
                    state,

                video=
                    video,

                decision=
                    fallback,
            )
            return

        state.detection_count += 1

        start_s, end_s = self.range_from_plan(
            state=
                state,

            video=
                video,

            decision=
                decision,
        )

        if end_s <= start_s:
            return

        num_frames = int(
            decision.get(
                "num_frames",
                8,
            )
            or 8
        )

        timestamps, frames = video.sample_range(
            start_s,
            end_s,
            num_frames,
        )

        target = str(
            decision.get(
                "target",
                decision.get(
                    "query",
                    state.question,
                ),
            )
        )

        result = self.object_tools.detect(
            frames=
                frames,

            timestamps=
                timestamps,

            target=
                target,
        )

        score = confidence_to_score(
            result.get(
                "confidence_score"
            )
        )

        detections = result.get(
            "detections",
            [],
        )

        observation = json.dumps(
            {
                "target":
                    target,

                "detections":
                    detections,
            },
            ensure_ascii=False,
        )

        evidence = Evidence(
            action=
                "OBJECT_DETECTION",

            query=
                target,

            start_s=
                start_s,

            end_s=
                end_s,

            timestamps=
                timestamps,

            observation=
                observation,

            confidence=
                "high"
                if score and score >= 0.75
                else "medium"
                if score and score >= 0.45
                else "low",

            latency_s=
                float(
                    result[
                        "latency_s"
                    ]
                ),

            confidence_score=
                score,

            uncertainty=
                score_to_uncertainty(
                    score
                ),

            tool_name=
                "OBJECT_DETECTION",

            metadata={
                "target":
                    target,

                "num_detections":
                    len(
                        detections
                    ),
            },
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            evidence.latency_s
        )


    def execute_object_tracking(
        self,
        *,
        state,
        video,
        decision,
    ):
        if self.object_tools is None:
            fallback = dict(
                decision
            )
            fallback[
                "query"
            ] = (
                "OBJECT_TRACKING fallback: "
                + str(
                    decision.get(
                        "query",
                        state.question,
                    )
                )
            )
            state.tracking_count += 1
            self.execute_zoom_caption(
                state=
                    state,

                video=
                    video,

                decision=
                    fallback,
            )
            return

        state.tracking_count += 1

        start_s, end_s = self.range_from_plan(
            state=
                state,

            video=
                video,

            decision=
                decision,
        )

        if end_s <= start_s:
            return

        num_frames = int(
            decision.get(
                "num_frames",
                12,
            )
            or 12
        )

        timestamps, frames = video.sample_range(
            start_s,
            end_s,
            num_frames,
        )

        target = str(
            decision.get(
                "target",
                decision.get(
                    "query",
                    state.question,
                ),
            )
        )

        result = self.object_tools.track(
            frames=
                frames,

            timestamps=
                timestamps,

            target=
                target,
        )

        score = confidence_to_score(
            result.get(
                "confidence_score"
            )
        )

        tracks = (
            result.get(
                "tracks"
            )
            or result.get(
                "detections",
                [],
            )
        )

        observation = json.dumps(
            {
                "target":
                    target,

                "tracks":
                    tracks,

                "tracking_fallback":
                    bool(
                        result.get(
                            "tracking_fallback",
                            False,
                        )
                    ),
            },
            ensure_ascii=False,
        )

        evidence = Evidence(
            action=
                "OBJECT_TRACKING",

            query=
                target,

            start_s=
                start_s,

            end_s=
                end_s,

            timestamps=
                timestamps,

            observation=
                observation,

            confidence=
                "high"
                if score and score >= 0.75
                else "medium"
                if score and score >= 0.45
                else "low",

            latency_s=
                float(
                    result[
                        "latency_s"
                    ]
                ),

            confidence_score=
                score,

            uncertainty=
                score_to_uncertainty(
                    score
                ),

            tool_name=
                "OBJECT_TRACKING",

            metadata={
                "target":
                    target,

                "num_tracks":
                    len(
                        tracks
                    ),

                "tracking_fallback":
                    bool(
                        result.get(
                            "tracking_fallback",
                            False,
                        )
                    ),
            },
        )

        state.evidence.append(
            evidence
        )

        state.total_latency_s += (
            evidence.latency_s
        )


    def execute_decision(
        self,
        *,
        state,
        video,
        decision,
    ):
        action = (
            decision[
                "action"
            ]
        )

        query = str(
            decision.get(
                "query",
                state.question,
            )
        )

        if action == "SEARCH_LOCAL":

            self.execute_local_search(
                state=
                    state,

                video=
                    video,

                query=
                    query,
            )

            return

        if action == "GLOBAL_SCAN":

            self.execute_global_scan(
                state=
                    state,

                video=
                    video,

                query=
                    query,
            )

            return

        if action == "COUNT_EVENTS":

            self.execute_count_scan(
                state=
                    state,

                video=
                    video,

                query=
                    query,
            )

            return

        if action == "SEARCH_BEFORE":

            self.execute_temporal_search(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,

                direction=
                    "before",
            )

            return

        if action == "SEARCH_AFTER":

            self.execute_temporal_search(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,

                direction=
                    "after",
            )

            return

        if action == "INCREASE_DENSITY":

            self.execute_refinement(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,

                verify=
                    False,
            )

            return

        if action == "VERIFY_DETAIL":

            self.execute_refinement(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,

                verify=
                    True,
            )

            return

        if action == "IMAGE_CAPTION":

            self.execute_image_caption(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,
            )

            return

        if action == "ZOOM_CAPTION":

            self.execute_zoom_caption(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,
            )

            return

        if action == "OBJECT_DETECTION":

            self.execute_object_detection(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,
            )

            return

        if action == "OBJECT_TRACKING":

            self.execute_object_tracking(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,
            )

            return


    # ========================================================
    # Sufficiency -> decision
    # ========================================================

    def decision_from_sufficiency(
        self,
        *,
        state,
        sufficiency,
    ):
        if not sufficiency:
            return {
                "action": "SEARCH_LOCAL",
                "query": state.question,
                "reason":
                    "Sufficiency parser failed; continue evidence acquisition.",
            }

        if sufficiency.get("sufficient"):
            return {
                "action": "ANSWER",
                "reason": "Evidence sufficient.",
            }

        action = str(
            sufficiency.get(
                "recommended_action",
                "",
            )
        ).upper()

        query = (
            sufficiency.get("recommended_query")
            or sufficiency.get("missing_visual_fact")
            or state.question
        )

        requirement = str(
            state.policy.get(
                "evidence_requirement",
                "local",
            )
        )

        relation = str(
            state.policy.get(
                "temporal_relation",
                "none",
            )
        )

        # -----------------------------------------
        # Enforce evidence geometry.
        # -----------------------------------------

        if requirement == "global":
            action = "GLOBAL_SCAN"
            sufficiency["evidence_id"] = None

        elif requirement == "count":
            action = "COUNT_EVENTS"
            sufficiency["evidence_id"] = None

        elif (
            requirement == "temporal"
            and relation == "before"
        ):
            action = "SEARCH_BEFORE"

        elif (
            requirement == "temporal"
            and relation == "after"
        ):
            action = "SEARCH_AFTER"

        elif action not in VALID_ACTIONS or action == "ANSWER":
            if state.evidence:
                action = "VERIFY_DETAIL"
            else:
                action = "SEARCH_LOCAL"

        decision = {
            "action": action,
            "query": query,
            "reason":
                sufficiency.get(
                    "missing_visual_fact",
                    "",
                ),
        }

        raw_evidence_id = sufficiency.get(
            "evidence_id"
        )

        try:
            evidence_id = int(
                raw_evidence_id
            )
        except (TypeError, ValueError):
            evidence_id = None

        if (
            evidence_id is not None
            and 0 <= evidence_id < len(state.evidence)
        ):
            decision["evidence_id"] = evidence_id

            if (
                action in {
                    "VERIFY_DETAIL",
                    "INCREASE_DENSITY",
                }
                and state.active_evidence_id is None
            ):
                state.active_evidence_id = evidence_id

        return decision

    # ========================================================
    # Main agentic loop
    # ========================================================

    def run(
        self,
        *,
        video,
        question,
        choices,
        policy,
    ):
        policy = sanitize_policy(
            policy
        )

        if (
            policy.get(
                "execution_mode"
            )
            != "agentic"
        ):
            raise ValueError(
                "VideoAgent received non-agentic policy"
            )

        state = AgentState(
            question=
                question,

            duration_s=
                video.duration_s,

            policy=
                policy,
        )

        trajectory = []

        self.acquire_general_context(
            state=
                state,

            video=
                video,
        )

        trajectory.append({
            "round":
                "general_context",

            "decision": {
                "action":
                    "GENERAL_CONTEXT",

                "segments":
                    len(
                        state.memory_bank.get(
                            "segment_captions",
                            [],
                        )
                    ),

                "summary_confidence_score":
                    state.memory_bank.get(
                        "summary_confidence_score"
                    ),
            },
        })

        max_steps = self.max_rounds

        no_progress = 0
        previous_plan = None

        for round_index in range(
            max_steps
        ):

            # --------------------------------------------
            # VideoAgent2 Phase 2: answer assessment.
            # --------------------------------------------

            (
                assessment,
                assessment_latency,
            ) = (
                self.controller
                .assess_answer(
                    state=
                        state,

                    choices=
                        choices,

                    confidence_threshold=
                        self.answer_confidence_threshold,
                )
            )

            state.total_latency_s += (
                assessment_latency
            )

            state.answer_confidence = float(
                assessment.get(
                    "confidence",
                    0.0,
                )
            )

            state.answer_assessments.append(
                assessment
            )

            print(
                "    answer assessment:",
                assessment,
            )

            trajectory.append({
                "round":
                    round_index + 1,

                "phase":
                    "answer_assessment",

                "assessment":
                    assessment,

                "controller_latency_s":
                    assessment_latency,
            })

            if assessment.get(
                "sufficient"
            ) and self.assessment_has_supported_answer(
                assessment,
                choices,
                state=state,
            ):
                print(
                    "    confidence threshold reached:",
                    state.answer_confidence,
                )
                state.supported_answer = normalize_answer(
                    assessment.get(
                        "answer"
                    )
                )
                state.supported_answer_reason = str(
                    assessment.get(
                        "reason",
                        "",
                    )
                )
                break

            if self.assessment_has_supported_answer(
                assessment,
                choices,
                state=state,
            ):
                state.supported_answer = normalize_answer(
                    assessment.get(
                        "answer"
                    )
                )
                state.supported_answer_reason = str(
                    assessment.get(
                        "reason",
                        "",
                    )
                )
                print(
                    "    supported answer reached:",
                    state.supported_answer,
                    "confidence:",
                    state.answer_confidence,
                )
                break

            # --------------------------------------------
            # VideoAgent2 Phase 3: plan create / adjust.
            # --------------------------------------------

            if previous_plan is None:
                (
                    plan,
                    plan_latency,
                ) = (
                    self.controller
                    .create_plan(
                        state=
                            state,

                        assessment=
                            assessment,
                    )
                )

                plan_phase = (
                    "plan_create"
                )

            else:
                (
                    plan,
                    plan_latency,
                ) = (
                    self.controller
                    .adjust_plan(
                        state=
                            state,

                        assessment=
                            assessment,

                        previous_plan=
                            previous_plan,
                    )
                )

                plan_phase = (
                    "plan_adjust"
                )

            state.total_latency_s += (
                plan_latency
            )

            state.retrieval_plan = plan
            state.retrieval_plans.append(
                plan
            )

            decision = self.plan_to_decision(
                plan=
                    plan,

                state=
                    state,
            )

            decision = (
                self.normalize_decision(
                    decision=
                        decision,

                    state=
                        state,
                )
            )

            action = decision[
                "action"
            ]

            trajectory.append({
                "round":
                    round_index + 1,

                "phase":
                    plan_phase,

                "plan":
                    plan,

                "decision":
                    decision,

                "controller_latency_s":
                    plan_latency,
            })

            print(
                f"  round={round_index + 1} "
                f"action={action}"
            )

            if action == "ANSWER":
                break

            # --------------------------------------------
            # VideoAgent2 Phase 4: execute retrieval plan.
            # --------------------------------------------

            evidence_before = len(
                state.evidence
            )

            self.execute_decision(
                state=
                    state,

                video=
                    video,

                decision=
                    decision,
            )

            evidence_after = len(
                state.evidence
            )

            if (
                evidence_after
                == evidence_before
            ):

                no_progress += 1
                previous_plan = plan

                if no_progress >= 1:
                    fallback_start, fallback_end = (
                        self.next_unseen_segment_range(
                            state=
                                state,

                            video=
                                video,
                        )
                    )

                    fallback_decision = {
                        "action":
                            "IMAGE_CAPTION",

                        "tool":
                            "IMAGE_CAPTION",

                        "query":
                            (
                                assessment.get(
                                    "missing_information"
                                )
                                or "inspect a new chronological segment "
                                "that may resolve the unanswered visual "
                                "difference between choices"
                            ),

                        "start_s":
                            fallback_start,

                        "end_s":
                            fallback_end,

                        "reason":
                            (
                                "previous tool produced no new evidence; "
                                "inspect the next unseen segment"
                            ),
                    }

                    self.execute_image_caption(
                        state=
                            state,

                        video=
                            video,

                        decision=
                            fallback_decision,
                    )

                    fallback_after = len(
                        state.evidence
                    )

                    if fallback_after > evidence_before:
                        no_progress = 0

                        self.record_new_evidence(
                            state=
                                state,

                            start_index=
                                evidence_before,

                            plan=
                                fallback_decision,
                        )

                        trajectory.append({
                            "round":
                                round_index + 1,

                            "phase":
                                "no_progress_fallback",

                            "decision":
                                fallback_decision,
                        })

                        for evidence_index in range(
                            evidence_before,
                            fallback_after,
                        ):
                            evidence = state.evidence[
                                evidence_index
                            ]

                            print(
                                "    evidence",
                                evidence_index,
                                f"[{evidence.action}]",
                                f"{evidence.start_s:.2f}",
                                "->",
                                f"{evidence.end_s:.2f}",
                            )

                            print(
                                "      ",
                                evidence.observation,
                            )

                        continue

                if no_progress >= 2:
                    break

                continue

            no_progress = 0
            previous_plan = plan

            self.record_new_evidence(
                state=
                    state,

                start_index=
                    evidence_before,

                plan=
                    plan,
            )

            for evidence_index in range(
                evidence_before,
                evidence_after,
            ):

                evidence = (
                    state.evidence[
                        evidence_index
                    ]
                )

                print(
                    "    evidence",
                    evidence_index,
                    f"[{evidence.action}]",
                    f"{evidence.start_s:.2f}",
                    "->",
                    f"{evidence.end_s:.2f}",
                )

                print(
                    "      ",
                    evidence.observation,
                )

        # ====================================================
        # Final answer from accumulated memory/evidence.
        # ====================================================

        if state.supported_answer:
            remap = (
                self.controller
                .remap_answer_to_choice(
                    state=
                        state,

                    choices=
                        choices,

                    prediction=
                        state.supported_answer,

                    reason=
                        state.supported_answer_reason
                        or "",
                )
            )

            state.total_latency_s += float(
                remap[
                    "latency_s"
                ]
            )

            answer = {
                "prediction":
                    remap[
                        "prediction"
                    ],

                "reason":
                    remap[
                        "reason"
                    ]
                    or state.supported_answer_reason
                    or "High-confidence supported answer from assessment.",

                "confidence":
                    state.answer_confidence,

                "latency_s":
                    remap[
                        "latency_s"
                    ],
            }

            state.final_answer = (
                answer[
                    "prediction"
                ]
            )

        else:
            if is_sequence_question(
                state.question
            ):
                answer = (
                    self.controller
                    .answer_sequence_from_timeline(
                        state=
                            state,

                        choices=
                            choices,
                    )
                )

            else:
                answer = (
                    self.controller
                    .answer_from_evidence(
                        state=
                            state,

                        choices=
                            choices,
                    )
                )

            state.total_latency_s += float(
                answer[
                    "latency_s"
                ]
            )

            state.final_answer = (
                answer[
                    "prediction"
                ]
            )

            remap = (
                self.controller
                .remap_answer_to_choice(
                    state=
                        state,

                    choices=
                        choices,

                    prediction=
                        state.final_answer,

                    reason=
                        answer[
                            "reason"
                        ],
                )
            )

            state.total_latency_s += float(
                remap[
                    "latency_s"
                ]
            )

            state.final_answer = (
                remap[
                    "prediction"
                ]
            )

            answer[
                "prediction"
            ] = state.final_answer

            answer[
                "reason"
            ] = (
                remap[
                    "reason"
                ]
                or answer[
                    "reason"
                ]
            )

            answer[
                "latency_s"
            ] += float(
                remap[
                    "latency_s"
                ]
            )

        trajectory.append({
            "round":
                "final_answer",

            "decision": {
                "action":
                    "ANSWER",

                "answer":
                    state.final_answer,

                "reason":
                    answer[
                        "reason"
                    ],

                "confidence":
                    answer[
                        "confidence"
                    ],
            },

            "controller_latency_s":
                answer[
                    "latency_s"
                ],
        })

        return (
            state,
            trajectory,
        )


    def precompute_caption_cache(
        self,
        *,
        video,
        question,
        policy,
    ):
        policy = sanitize_policy(
            policy
        )

        state = AgentState(
            question=
                question,

            duration_s=
                video.duration_s,

            policy=
                policy,
        )

        self.acquire_general_context(
            state=
                state,

            video=
                video,
        )

        trajectory = [{
            "round":
                "general_context",

            "decision": {
                "action":
                    "PRECOMPUTE_CAPTION_CACHE",

                "segments":
                    len(
                        state.memory_bank.get(
                            "segment_captions",
                            [],
                        )
                    ),

                "caption_cache_hit":
                    state.memory_bank.get(
                        "caption_cache_hit"
                    ),

                "caption_cache_path":
                    state.memory_bank.get(
                        "caption_cache_path"
                    ),
            },
        }]

        state.final_answer = ""

        return (
            state,
            trajectory,
        )


# ============================================================
# Dataset helpers
# ============================================================

def build_video_index(
    sweep_results,
):
    index = {}

    for row in sweep_results:

        dataset = str(
            row.get(
                "dataset",
                row.get(
                    "source_dataset",
                    "unknown",
                ),
            )
        )

        video_id = str(
            row.get(
                "video_id",
                "",
            )
        )

        value = row.get(
            "video"
        )

        if value:

            index[
                (
                    dataset,
                    video_id,
                )
            ] = str(
                value
            )

    return index


def resolve_video_path(
    candidate,
    video_index,
):
    dataset = str(
        candidate.get(
            "dataset",
            "unknown",
        )
    )

    video_id = str(
        candidate.get(
            "video_id",
            "",
        )
    )

    for field_name in (
        "video",
        "video_path",
    ):

        value = candidate.get(
            field_name
        )

        if (
            value
            and Path(
                value
            ).exists()
        ):
            return str(
                value
            )

    candidate_path = (
        video_index.get(
            (
                dataset,
                video_id,
            )
        )
    )

    if (
        candidate_path
        and Path(
            candidate_path
        ).exists()
    ):
        return candidate_path

    raise FileNotFoundError(
        "Could not resolve video "
        f"dataset={dataset} "
        f"video_id={video_id}"
    )


# ============================================================
# Top-level execution
# ============================================================

def execute_profiled_query(
    *,
    video,
    question,
    choices,
    profiler_result,
    fixed_executor,
    agent_executor,
    force_agentic=False,
    force_oneshot=False,
    min_sequence_segments=3,
    tool_budget_profile="adaptive",
    profiler_policy_mode="hint",
    max_prompt_memory_segments=64,
    max_final_memory_segments=96,
    precompute_caption_cache_only=False,
):
    # 1. Get profiler policy
    policy = sanitize_policy(
        profiler_result.execution_policy
    )

    policy = dict(
        policy
    )
    policy[
        "min_sequence_segments"
    ] = max(
        1,
        int(
            min_sequence_segments
        ),
    )

    policy[
        "max_prompt_memory_segments"
    ] = max(
        1,
        int(
            max_prompt_memory_segments
        ),
    )

    policy[
        "max_final_memory_segments"
    ] = max(
        1,
        int(
            max_final_memory_segments
        ),
    )

    requested_tool_profile = str(
        tool_budget_profile
    ).lower()

    if requested_tool_profile not in {
        "adaptive",
        "summary_light",
        "strict",
    }:
        requested_tool_profile = "adaptive"

    requested_profiler_policy_mode = str(
        profiler_policy_mode
    ).lower()

    if requested_profiler_policy_mode not in {
        "hint",
        "hard",
    }:
        requested_profiler_policy_mode = "hint"

    policy[
        "profiler_policy_mode"
    ] = requested_profiler_policy_mode

    requirement = str(
        policy.get(
            "evidence_requirement",
            "",
        )
    )

    relation = str(
        policy.get(
            "temporal_relation",
            "none",
        )
    )

    evidence_type = str(
        policy.get(
            "evidence_type",
            "",
        )
    )

    if requested_tool_profile == "adaptive":
        if (
            evidence_type in {
                "local_detail",
                "generic_local_mcq",
            }
            or relation in {
                "before",
                "after",
                "ordering",
            }
            or evidence_type == "sequence"
        ):
            policy["tool_budget_profile"] = "strict"

        elif requirement == "global":
            policy["tool_budget_profile"] = "summary_light"

        else:
            policy["tool_budget_profile"] = "adaptive"

    else:
        policy["tool_budget_profile"] = requested_tool_profile

    original_mode = policy.get(
        "execution_mode"
    )

    # ========================================================
    # FORCE-AGENTIC CAPABILITY OVERRIDE
    # START HERE
    # ========================================================

    if (
        force_agentic
        and force_oneshot
    ):
        raise ValueError(
            "Use only one of --force-agentic or --force-oneshot."
        )

    if force_agentic:
        # Always execute through the adaptive agent.
        policy["execution_mode"] = "agentic"

        # Capability experiment:
        # profiler describes the question, but DOES NOT restrict
        # which evidence-acquisition operation may be used.
        policy["allowed_actions"] = [
            "SEARCH_LOCAL",
            "SEARCH_BEFORE",
            "SEARCH_AFTER",
            "GLOBAL_SCAN",
            "COUNT_EVENTS",
            "INCREASE_DENSITY",
            "VERIFY_DETAIL",
            "IMAGE_CAPTION",
            "ZOOM_CAPTION",
            "OBJECT_DETECTION",
            "OBJECT_TRACKING",
            "ANSWER",
        ]

        # Generous initial search geometry.
        policy["probe_fps"] = max(
            0.05,
            float(
                policy.get(
                    "probe_fps",
                    0.05,
                )
            ),
        )

        policy["probe_topk"] = max(
            8,
            int(
                policy.get(
                    "probe_topk",
                    8,
                )
                or 8
            ),
        )

        policy["action_topk"] = max(
            8,
            int(
                policy.get(
                    "action_topk",
                    8,
                )
                or 8
            ),
        )

        policy["window_len_s"] = max(
            16.0,
            float(
                policy.get(
                    "window_len_s",
                    16.0,
                )
            ),
        )

        policy["high_frames_per_window"] = max(
            12,
            int(
                policy.get(
                    "high_frames_per_window",
                    12,
                )
                or 12
            ),
        )

        print(
            "    FORCE AGENTIC CAPABILITY MODE:",
            original_mode,
            "-> agentic",
        )
    elif force_oneshot:
        policy = dict(policy)

        policy["execution_mode"] = "oneshot"

        requirement = str(
            policy.get(
                "evidence_requirement",
                "",
            )
        )

        relation = str(
            policy.get(
                "temporal_relation",
                "none",
            )
        )

        evidence_type = str(
            policy.get(
                "evidence_type",
                "",
            )
        )

        if (
            requirement in {
                "global",
                "count",
            }
            or relation == "ordering"
            or evidence_type == "sequence"
        ):
            policy["selection_mode"] = "uniform"

        print(
            "    FORCE FIXED ONESHOT MODE:",
            original_mode,
            "-> oneshot",
        )
    # ========================================================
    # FORCE-AGENTIC CAPABILITY OVERRIDE
    # END HERE
    # ========================================================

    # 2. NOW decide which executor gets called.
    mode = policy.get(
        "execution_mode"
    )

    print(
        "    profiler source:",
        profiler_result.source,
    )

    print(
        "    original execution_mode:",
        original_mode,
    )

    print(
        "    execution_mode:",
        mode,
    )

    print(
        "    evidence_requirement:",
        policy.get(
            "evidence_requirement"
        ),
    )

    print(
        "    temporal_relation:",
        policy.get(
            "temporal_relation"
        ),
    )

    print(
        "    evidence_type:",
        policy.get(
            "evidence_type"
        ),
    )

    if precompute_caption_cache_only:
        state, trajectory = agent_executor.precompute_caption_cache(
            video=
                video,

            question=
                question,

            policy=
                policy,
        )

        return (
            state,
            trajectory,
            policy,
            "caption_cache",
        )

    # 3. Dispatch
    if mode == "oneshot":

        state, trajectory = fixed_executor.run(
            video=video,
            question=question,
            choices=choices,
            policy=policy,
        )

        execution_path = "oneshot"

    elif mode == "agentic":

        state, trajectory = agent_executor.run(
            video=video,
            question=question,
            choices=choices,
            policy=policy,
        )

        execution_path = "agentic"

    else:
        raise RuntimeError(
            "Unknown profiler execution mode: "
            f"{mode}"
        )

    return (
        state,
        trajectory,
        policy,
        execution_path,
    )

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force-agentic",
        action="store_true",
        help="Force all questions through the agentic execution path.",
    )

    parser.add_argument(
        "--force-oneshot",
        action="store_true",
        help="Force all questions through the fixed one-shot execution path.",
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--sweep-results",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--base-url",
        default=
            "http://localhost:9000/v1",
    )

    parser.add_argument(
        "--caption-base-url",
        default=None,
        help="Optional OpenAI-compatible endpoint for captioning; defaults to --base-url.",
    )

    parser.add_argument(
        "--vlm-model",
        default=
            "Qwen/"
            "Qwen2.5-VL-7B-Instruct",
    )

    parser.add_argument(
        "--caption-model",
        default=None,
        help="Optional cheaper VLM for segment/image captioning; defaults to --vlm-model.",
    )

    parser.add_argument(
        "--caption-prompt-style",
        choices=[
            "videoagent2",
            "generic",
        ],
        default="videoagent2",
        help=(
            "Prompt style for segment captions. videoagent2 asks for "
            "dense action/object/state-change captions."
        ),
    )

    parser.add_argument(
        "--caption-workers",
        type=int,
        default=1,
        help="Concurrent segment-caption requests on caption cache misses.",
    )

    parser.add_argument(
        "--enable-object-tools",
        action="store_true",
        help="Enable YOLO-backed OBJECT_DETECTION and OBJECT_TRACKING tools.",
    )

    parser.add_argument(
        "--detector-model",
        default="yolo11n.pt",
        help="Ultralytics YOLO model path/name for object tools.",
    )

    parser.add_argument(
        "--detector-device",
        default="cpu",
        help="Device for object tools, e.g. cpu, cuda:0, cuda:1.",
    )

    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold for object tools.",
    )

    parser.add_argument(
        "--profiler-model",
        default=None,
    )

    parser.add_argument(
        "--disable-profiler",
        action="store_true",
        help=(
            "Skip the LLM profiler and use a neutral execution policy. "
            "This is automatically enabled for --force-agentic."
        ),
    )

    parser.add_argument(
        "--siglip-model",
        default=
            "google/"
            "siglip-so400m-patch14-384",
    )

    parser.add_argument(
        "--clip-device",
        default=
            "cpu",
    )

    parser.add_argument(
        "--profiler-confidence-threshold",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--profiler-exploration-rate",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--disable-videoagent2-context",
        action="store_true",
        help="Disable segment-caption memory acquisition in the agentic path.",
    )

    parser.add_argument(
        "--caption-cache-dir",
        default=
            "conductor/experiments/self_improving/data/video_agent_caption_cache",
        help="Directory for reusable per-video segment-caption caches.",
    )

    parser.add_argument(
        "--disable-caption-cache",
        action="store_true",
        help="Regenerate segment captions instead of loading/writing caption cache.",
    )

    parser.add_argument(
        "--precompute-caption-cache-only",
        action="store_true",
        help="Only build/load caption cache and query summary; skip tools and final answering.",
    )

    parser.add_argument(
        "--context-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--context-segment-s",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--context-frames-per-segment",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-context-segments",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--context-coverage",
        choices=[
            "all",
            "adaptive",
            "sparse",
        ],
        default="adaptive",
        help=(
            "Segment-caption coverage for VideoAgent2 memory: "
            "all captions every segment; adaptive captions every "
            "segment for global/sequence/count and sparse for local; "
            "sparse always caps by --max-context-segments."
        ),
    )

    parser.add_argument(
        "--answer-confidence-threshold",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--min-sequence-segments",
        type=int,
        default=3,
        help="Minimum distinct post-summary caption/tool segments before sequence questions may answer.",
    )

    parser.add_argument(
        "--tool-budget-profile",
        choices=[
            "adaptive",
            "summary_light",
            "strict",
        ],
        default="adaptive",
        help=(
            "Runtime retrieval pressure after summary memory: "
            "summary_light uses fewer follow-up tools, strict keeps "
            "sequence/local-detail gates, adaptive chooses by query type."
        ),
    )

    parser.add_argument(
        "--profiler-policy-mode",
        choices=[
            "hint",
            "hard",
        ],
        default="hint",
        help=(
            "How strongly profiler labels constrain agent control: "
            "hint lets answer assessment and memory choose tools; hard "
            "keeps the older deterministic global/sequence gates."
        ),
    )

    parser.add_argument(
        "--max-prompt-memory-segments",
        type=int,
        default=64,
        help="Maximum segment captions included in controller prompts; full captions remain cached/output.",
    )

    parser.add_argument(
        "--max-final-memory-segments",
        type=int,
        default=96,
        help="Maximum segment captions included in the final answer prompt.",
    )

    parser.add_argument(
        "--global-chunk-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dedupe-input",
        action="store_true",
        help="Skip duplicate dataset/video/qid/question rows before applying --limit.",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split input rows across this many deterministic shards before applying --limit.",
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index to run, in [0, --num-shards).",
    )

    args = parser.parse_args()

    if (
        args.force_agentic
        and args.force_oneshot
    ):
        parser.error(
            "--force-agentic and --force-oneshot are mutually exclusive"
        )

    if args.num_shards < 1:
        parser.error(
            "--num-shards must be >= 1"
        )

    if (
        args.shard_index < 0
        or args.shard_index >= args.num_shards
    ):
        parser.error(
            "--shard-index must be in [0, --num-shards)"
        )

    profiler_model = (
        args.profiler_model
        or args.vlm_model
    )

    candidates = load_jsonl(
        args.input
    )

    if args.dedupe_input:
        deduped = []
        seen_candidates = set()

        for candidate in candidates:
            key = (
                str(
                    candidate.get(
                        "dataset",
                        "unknown",
                    )
                ),
                str(
                    candidate.get(
                        "video_id",
                        "",
                    )
                ),
                str(
                    candidate.get(
                        "qid",
                        "",
                    )
                ),
                str(
                    candidate.get(
                        "question",
                        "",
                    )
                ),
            )

            if key in seen_candidates:
                continue

            seen_candidates.add(
                key
            )
            deduped.append(
                candidate
            )

        candidates = deduped

    if args.num_shards > 1:
        candidates = [
            candidate
            for index, candidate
            in enumerate(
                candidates
            )
            if (
                index % args.num_shards
                == args.shard_index
            )
        ]

    sweep_rows = load_jsonl(
        args.sweep_results
    )

    # ========================================================
    # Offline gold lookup
    # ========================================================

    gold_by_question = {}

    for row in sweep_rows:

        key = (
            str(
                row.get(
                    "dataset",
                    row.get(
                        "source_dataset",
                        "unknown",
                    ),
                )
            ),

            str(
                row.get(
                    "video_id",
                    "",
                )
            ),

            str(
                row.get(
                    "qid",
                    "",
                )
            ),

            str(
                row.get(
                    "question",
                    "",
                )
            ),
        )

        gold = row.get(
            "answer_label"
        )

        if gold is not None:

            gold_by_question[
                key
            ] = normalize_answer(
                gold
            )

    video_index = (
        build_video_index(
            sweep_rows
        )
    )

    if args.limit > 0:

        candidates = (
            candidates[
                :args.limit
            ]
        )

    # ========================================================
    # Models
    # ========================================================

    client = OpenAI(
        base_url=
            args.base_url,

        api_key=
            "EMPTY",
    )

    caption_client = OpenAI(
        base_url=
            (
                args.caption_base_url
                or args.base_url
            ),

        api_key=
            "EMPTY",
    )

    retriever = SigLIPRetriever(
        model_name=
            args.siglip_model,

        device=
            args.clip_device,
    )

    inspector = EvidenceInspector(
        client=
            client,

        model=
            args.vlm_model,

        caption_model=
            (
                args.caption_model
                or args.vlm_model
            ),

        caption_client=
            caption_client,

        caption_prompt_style=
            args.caption_prompt_style,
    )

    controller = VideoAgentController(
        client=
            client,

        model=
            args.vlm_model,
    )

    object_tools = None

    if args.enable_object_tools:
        object_tools = UltralyticsObjectTools(
            model_name=
                args.detector_model,

            device=
                args.detector_device,

            conf=
                args.detector_conf,
        )

    fixed_executor = (
        FixedVIMIOExecutor(
            controller=
                controller,

            inspector=
                inspector,

            retriever=
                retriever,

            global_chunk_size=
                args.global_chunk_size,
        )
    )

    agent_executor = VideoAgent(
        controller=
            controller,

        inspector=
            inspector,

        retriever=
            retriever,

        object_tools=
            object_tools,

        max_rounds=
            args.max_rounds,

        global_chunk_size=
            args.global_chunk_size,

        videoagent2_context=
            not args.disable_videoagent2_context,

        context_fps=
            args.context_fps,

        context_segment_s=
            args.context_segment_s,

        context_frames_per_segment=
            args.context_frames_per_segment,

        max_context_segments=
            args.max_context_segments,

        context_coverage=
            args.context_coverage,

        caption_cache_dir=
            args.caption_cache_dir,

        disable_caption_cache=
            args.disable_caption_cache,

        caption_workers=
            args.caption_workers,

        answer_confidence_threshold=
            args.answer_confidence_threshold,
    )

    # ========================================================
    # Output / resume
    # ========================================================

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    done = set()

    if output.exists():

        for row in load_jsonl(
            output
        ):

            done.add((
                str(
                    row.get(
                        "dataset",
                        "unknown",
                    )
                ),

                str(
                    row.get(
                        "video_id",
                        "",
                    )
                ),

                str(
                    row.get(
                        "qid",
                        "",
                    )
                ),

                str(
                    row.get(
                        "question",
                        "",
                    )
                ),
            ))

    pending = []

    for candidate in candidates:

        key = (
            str(
                candidate.get(
                    "dataset",
                    "unknown",
                )
            ),

            str(
                candidate.get(
                    "video_id",
                    "",
                )
            ),

            str(
                candidate.get(
                    "qid",
                    "",
                )
            ),

            str(
                candidate.get(
                    "question",
                    "",
                )
            ),
        )

        if key not in done:

            pending.append(
                candidate
            )

    print(
        "candidates:",
        len(candidates),
    )

    print(
        "already done:",
        len(done),
    )

    print(
        "remaining:",
        len(pending),
    )

    total_finished = 0
    total_answered = 0
    total_correct = 0
    total_latency = 0.0
    total_agentic = 0
    total_oneshot = 0

    # ========================================================
    # Evaluation loop
    # ========================================================

    with output.open(
        "a"
    ) as output_file:

        for index, candidate in enumerate(
            pending,
            start=1,
        ):

            question = str(
                candidate[
                    "question"
                ]
            )

            choices = (
                candidate.get(
                    "choices"
                )
                or []
            )

            dataset = str(
                candidate.get(
                    "dataset",
                    "unknown",
                )
            )

            video_id = str(
                candidate.get(
                    "video_id",
                    "",
                )
            )

            qid = str(
                candidate.get(
                    "qid",
                    "",
                )
            )

            key = (
                dataset,
                video_id,
                qid,
                question,
            )

            video_path = (
                resolve_video_path(
                    candidate,
                    video_index,
                )
            )

            print()
            print(
                "=" * 80
            )

            print(
                f"[{index}/{len(pending)}]"
            )

            print(
                "dataset:",
                dataset,
            )

            print(
                "video:",
                video_path,
            )

            print(
                "question:",
                question,
            )

            video = VideoBackend(
                video_path
            )

            # =================================================
            # PROFILER
            #
            # Gold/oracle information is NOT visible here.
            # =================================================

            disable_profiler = (
                args.disable_profiler
                or args.force_agentic
                or args.precompute_caption_cache_only
            )

            if disable_profiler:
                profiler_result = neutral_profiler_result(
                    execution_mode=
                        (
                            "oneshot"
                            if args.force_oneshot
                            else "agentic"
                        ),
                )

                profiler_latency = 0.0

            else:
                profiler_start = time.time()

                profiler_result = (
                    profile_query_adaptive(
                        query=
                            question,

                        duration_s=
                            video.duration_s,

                        choices=
                            choices,

                        confidence_threshold=
                            args
                            .profiler_confidence_threshold,

                        exploration_rate=
                            args
                            .profiler_exploration_rate,

                        base_url=
                            args.base_url,

                        model=
                            profiler_model,

                        temperature=
                            0.0,

                        resource_state=
                            ResourceState(
                                load_level=
                                    "low"
                            ),

                        verbose=
                            False,
                    )
                )

                profiler_latency = (
                    time.time()
                    - profiler_start
                )

            policy = sanitize_policy(
                profiler_result
                .execution_policy
            )

            # =================================================
            # Leakage assertion
            # =================================================

            runtime_input = {
                "question":
                    question,

                "policy":
                    policy,
            }

            assert_no_forbidden_keys(
                runtime_input
            )

            # =================================================
            # Execute profiler-selected mode
            # =================================================

            execution_start = (
                time.time()
            )

            (
                state,
                trajectory,
                policy,
                execution_path,
            ) = (
                execute_profiled_query(
                    video=video,
                    question=question,
                    choices=choices,
                    profiler_result=profiler_result,
                    fixed_executor=fixed_executor,
                    agent_executor=agent_executor,
                    force_agentic=args.force_agentic,
                    force_oneshot=args.force_oneshot,
                    min_sequence_segments=args.min_sequence_segments,
                    tool_budget_profile=args.tool_budget_profile,
                    profiler_policy_mode=args.profiler_policy_mode,
                    max_prompt_memory_segments=args.max_prompt_memory_segments,
                    max_final_memory_segments=args.max_final_memory_segments,
                    precompute_caption_cache_only=args.precompute_caption_cache_only,
                )
            )

            execution_latency = (
                time.time()
                - execution_start
            )

            wall_latency = (
                profiler_latency
                + execution_latency
            )

            # =================================================
            # Gold is accessed ONLY after online execution.
            # =================================================

            gold = (
                gold_by_question.get(
                    key
                )
            )

            prediction = normalize_answer(
                state.final_answer
            )

            if (
                prediction
                not in valid_choice_labels(
                    choices
                )
            ):
                prediction = ""

            correct = (
                gold is not None
                and prediction == gold
                and not args.precompute_caption_cache_only
            )

            if execution_path == "agentic":
                total_agentic += 1

            elif execution_path == "caption_cache":
                pass

            else:
                total_oneshot += 1

            result = {
                "dataset":
                    dataset,

                "video_id":
                    video_id,

                "qid":
                    qid,

                "question":
                    question,

                "choices":
                    choices,

                "video_path":
                    video_path,

                "profiler_source":
                    profiler_result.source,

                "profiler_router":
                    (
                        profiler_result
                        .router_decision
                        .__dict__
                    ),

                "profiler_policy":
                    policy,

                "execution_path":
                    execution_path,

                "gold":
                    gold,

                "prediction":
                    prediction,

                "correct":
                    bool(
                        correct
                    ),

                "precompute_caption_cache_only":
                    bool(
                        args.precompute_caption_cache_only
                    ),

                "force_agentic":
                    bool(args.force_agentic),

                "force_oneshot":
                    bool(args.force_oneshot),

                "profiler_original_execution_mode":
                    (
                        profiler_result
                        .execution_policy
                        .get(
                            "execution_mode"
                        )
                    ),

                "evidence": [
                    {
                        "action":
                            evidence.action,

                        "query":
                            evidence.query,

                        "start_s":
                            evidence.start_s,

                        "end_s":
                            evidence.end_s,

                        "timestamps":
                            evidence.timestamps,

                        "observation":
                            evidence.observation,

                        "confidence":
                            evidence.confidence,

                        "confidence_score":
                            evidence.confidence_score,

                        "uncertainty":
                            evidence.uncertainty,

                        "tool_name":
                            evidence.tool_name,

                        "metadata":
                            evidence.metadata,

                        "latency_s":
                            evidence.latency_s,
                    }

                    for evidence
                    in state.evidence
                ],

                "memory_bank":
                    state.memory_bank,

                "retrieval_plans":
                    state.retrieval_plans,

                "answer_assessments":
                    state.answer_assessments,

                "answer_confidence":
                    state.answer_confidence,

                "tool_uncertainties":
                    state.tool_uncertainties,

                "trajectory":
                    trajectory,

                "action_counts":
                    action_counts(
                        state
                    ),

                "profiler_latency_s":
                    profiler_latency,

                "execution_latency_s":
                    execution_latency,

                "agent_internal_latency_s":
                    state
                    .total_latency_s,

                "wall_latency_s":
                    wall_latency,

                "source":
                    "profiled_vimio_agent_videoagent2_v17",
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            total_finished += 1

            if not args.precompute_caption_cache_only:
                total_answered += 1

                total_correct += int(
                    correct
                )

            total_latency += (
                wall_latency
            )

            print(
                "PATH:",
                execution_path,
            )

            print(
                "FINAL:",
                prediction,
                "gold:",
                gold,
                "correct:",
                correct,
            )

            print(
                "profiler latency:",
                profiler_latency,
            )

            print(
                "execution latency:",
                execution_latency,
            )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "===== PROFILED VIDEOQA SUMMARY ====="
    )

    print(
        "finished:",
        total_finished,
    )

    print(
        "oneshot:",
        total_oneshot,
    )

    print(
        "agentic:",
        total_agentic,
    )

    print(
        "answered:",
        total_answered,
    )

    if total_answered:

        print(
            "accuracy:",
            total_correct
            / total_answered,
        )

    elif args.precompute_caption_cache_only:
        print(
            "accuracy:",
            "n/a (caption cache precompute only)",
        )

    if total_finished:

        print(
            "avg_latency_s:",
            total_latency
            / total_finished,
        )


if __name__ == "__main__":
    main()
