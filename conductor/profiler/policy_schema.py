from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: str
    anchor_policy: str
    coverage_target: int
    require_temporal_pair: bool

    preferred_tools: Tuple[str, ...]

    probe_tier: str
    caption_tier: str
    ocr_tier: str
    asr_tier: str

    enable_cheap_stage: bool
    cheap_answer_tier: str
    text_answer_min_conf: float

    answer_tier: str
    fallback_answer_tier: str

    probe_fps: float
    probe_seg_len_s: float
    probe_topk: int

    window_len_s: float
    strides: Tuple[float, ...]
    action_topk: int

    max_steps: int
    eps_marginal_gain: float

    escalate_if_low_confidence: bool
    min_retrieval_conf: float
    min_answer_conf: float

    confidence: float
    rationale: str