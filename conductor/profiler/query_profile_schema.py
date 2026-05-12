from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class QueryProfile:
    evidence_sources: Tuple[str, ...]
    answer_type: str
    reference_type: str
    inspection_pattern: str
    enable_cheap_stage: bool
    confidence: float
    rationale: str