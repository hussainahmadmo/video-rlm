# conductor/controller/execution_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LoadLevel = Literal["low", "medium", "high"]

WorkflowFamily = Literal[
    "visual_only",
    "asr_only",
    "ocr_only",
    "asr_anchored_visual",
    "ocr_anchored_visual",
    "multimodal",
]

ProposalStrategy = Literal[
    "uniform",
    "clip_first",
    "asr_first",
    "ocr_first",
    "hybrid",
]

SynthesisStrategy = Literal[
    "one_shot",
    "global_summary",
    "map_summary",
]

SchedulingPolicy = Literal[
    "fifo",
    "demand_aware",
    "quality_aware",
    "load_aware",
]


@dataclass
class SchedulerState:
    """
    Runtime resource/load state used by the scheduler.
    Later these can come from real vLLM queue measurements.
    """

    load_level: LoadLevel = "low"
    current_active_requests: int = 0
    available_memory_fraction: float = 1.0
    queue_delay_s: float = 0.0


@dataclass
class ExecutionConfig:
    """
    One candidate workflow/budget variant for long-video QA.

    A config specifies:
      1. which evidence workflow family is allowed,
      2. how candidate windows are proposed,
      3. how much visual evidence is processed,
      4. how selected evidence is synthesized,
      5. how costly the config is expected to be under load.
    """

    name: str

    # ------------------------------------------------------------
    # Semantic workflow choice
    # ------------------------------------------------------------
    workflow_family: WorkflowFamily = "visual_only"
    proposal_strategy: ProposalStrategy = "clip_first"

    # ------------------------------------------------------------
    # Visual budget knobs
    # ------------------------------------------------------------
    window_len_s: float = 8.0
    action_topk: int = 10
    frames_per_window: int = 2
    max_images_total: int = 20

    # Optional cheap probe knobs.
    probe_fps: float = 1.0
    probe_seg_len_s: float = 5.0
    probe_topk: int = 20

    # ------------------------------------------------------------
    # Synthesis strategy
    # ------------------------------------------------------------
    synthesis_strategy: SynthesisStrategy = "one_shot"

    # Total generated-token budget across all calls.
    total_output_token_budget: int = 4096

    # Fraction of generated tokens assigned to intermediate summaries.
    # Used only for global_summary and map_summary.
    summary_budget_fraction: float = 0.75

    # ------------------------------------------------------------
    # Serving / scheduling knobs
    # ------------------------------------------------------------
    scheduling_policy: SchedulingPolicy = "fifo"

    # Cost fields can be filled by a cost model.
    estimated_vlm_calls: int = 1
    estimated_num_images: int = 20
    estimated_output_tokens: int = 4096
    estimated_cost: float = 1.0

    # Optional priority score used by load-aware scheduler.
    priority: float = 0.0

    # ------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------
    max_side: int = 768
    jpeg_quality: int = 85

    def validate(self) -> None:
        if self.action_topk <= 0:
            raise ValueError("action_topk must be > 0")
        if self.frames_per_window <= 0:
            raise ValueError("frames_per_window must be > 0")
        if self.max_images_total <= 0:
            raise ValueError("max_images_total must be > 0")
        if self.total_output_token_budget <= 0:
            raise ValueError("total_output_token_budget must be > 0")
        if not (0.0 <= self.summary_budget_fraction <= 1.0):
            raise ValueError("summary_budget_fraction must be between 0 and 1")

        intended_images = self.action_topk * self.frames_per_window
        if self.max_images_total != intended_images:
            # Not necessarily fatal, but it is useful to detect accidental mismatch.
            raise ValueError(
                f"max_images_total={self.max_images_total} does not match "
                f"action_topk * frames_per_window = {intended_images}"
            )

    def infer_cost_fields(self) -> None:
        self.estimated_num_images = self.action_topk * self.frames_per_window
        self.estimated_output_tokens = self.total_output_token_budget

        if self.synthesis_strategy == "one_shot":
            self.estimated_vlm_calls = 1
        elif self.synthesis_strategy == "global_summary":
            self.estimated_vlm_calls = 2
        elif self.synthesis_strategy == "map_summary":
            self.estimated_vlm_calls = self.action_topk + 1
        else:
            raise ValueError(f"Unknown synthesis_strategy={self.synthesis_strategy}")

        # Simple cost proxy. Later replace with measured latency model.
        self.estimated_cost = (
            self.estimated_vlm_calls
            + 0.05 * self.estimated_num_images
            + 0.0001 * self.estimated_output_tokens
        )