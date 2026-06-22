from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Dict, Any, Literal

import time

from state import BeliefState
from actions import build_action_space
from scheduler import pick_next_action, propose_followup_windows
from stopping import stopping_rule
from budget import Budget
from tools import probe_index, inspect_window, ocr_window, asr_window
from cheap_answerer import TextAnswerer, TextAnswererConfig
from model_registry import default_registry, pick_answer_model, pick_text_model
from answerer import VLLMAnswerer, VLLMAnswererConfig
from llm_profiler import profile_query_llm
REG = default_registry()

LoadLevel = Literal["low", "medium", "high"]
ProposalStrategy = Literal["clip_first", "asr_first", "hybrid"]


# ---------------------------------------------------------------------
# Query profile vs execution config
# ---------------------------------------------------------------------

@dataclass
class QueryProfile:
    mode: str
    preferred_tools: tuple[str, ...]
    primary_question_target: str = ""
    reference_type: str = ""
    needs_spoken_content: bool = False
    visual_detail_level: str = "normal"
    object_scale: str = "normal"
    needs_precise_local_view: bool = False
    require_temporal_pair: bool = False

    # profiler/policy-exposed knobs
    probe_fps: float = 1.0
    probe_seg_len_s: float = 5.0
    probe_topk: int = 8
    window_len_s: float = 4.0
    action_topk: int = 8
    strides: tuple[float, ...] = (0.5,)
    eps_marginal_gain: float = 0.05

    # answer / fallback defaults
    answer_tier: str = "medium"
    fallback_answer_tier: str = "none"
    cheap_answer_tier: str = "cheap"
    enable_cheap_stage: bool = False
    text_answer_min_conf: float = 0.75
    min_answer_conf: float = 0.60
@dataclass
class SchedulerState:
    load_level: LoadLevel = "low"
    current_active_requests: int = 0
    available_memory_fraction: float = 1.0
    queue_delay_s: float = 0.0
@dataclass
class ExecutionConfig:
    name: str

    # strategy
    proposal_strategy: ProposalStrategy = "clip_first"
    run_clip_probe: bool = True
    run_search_loop: bool = True
    run_vlm: bool = True
    run_ocr: bool = False
    run_asr: bool = False
    run_cheap_text_stage: bool = False
    run_fallback_vlm: bool = False

    # modality breadth
    preferred_tools: tuple[str, ...] = field(default_factory=tuple)

    # probe / search knobs
    probe_fps: float = 1.0
    probe_seg_len_s: float = 5.0
    probe_topk: int = 8
    window_len_s: float = 4.0
    action_topk: int = 8
    strides: tuple[float, ...] = (0.5,)
    require_temporal_pair: bool = False
    eps_marginal_gain: float = 0.05

    # evidence knobs
    asr_chunk_len_s: float = 30.0
    asr_chunk_topk: int = 5
    ocr_max_frames: int = 8
    max_vlm_windows: int = 2
    expand_first_window_s: float = 2.0

    # answer knobs
    answer_tier: str = "medium"
    fallback_answer_tier: str = "none"
    cheap_answer_tier: str = "cheap"
    answer_max_frames_per_window: int = 2
    answer_max_images_total: int = 2
    answer_jpeg_quality: int = 85
    text_answer_min_conf: float = 0.75
    min_answer_conf: float = 0.60

    # cost estimate for load-aware scheduling
    estimated_cost: float = 1.0
@dataclass
class RunContext:
    query: str
    video: str

    # profile / config
    query_profile: Optional[QueryProfile] = None
    execution_config: Optional[ExecutionConfig] = None
    raw_policy: Any = None
    profiler_analysis: Dict[str, Any] = field(default_factory=dict)
    profiler_model: str = ""
    profiler_base_url: str = ""

    # runtime state
    state: BeliefState | None = None
    budget: Budget | None = None
    direction: str = "after"

    # probe / search
    clip_candidates: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    trace: list[Dict[str, Any]] = field(default_factory=list)

    # final selection / evidence
    vlm_windows: list[tuple[float, float]] = field(default_factory=list)
    ocr_outputs: list = field(default_factory=list)
    asr_outputs: list = field(default_factory=list)
    asr_windows: list[tuple[float, float]] = field(default_factory=list)
    total_asr_time_s: float = 0.0

    # answer outputs
    pred: Optional[str] = None
    chosen_answer_model: Optional[str] = None
    fallback_answer_model: Optional[str] = None
    fallback_used: bool = False
    answer_conf: Optional[float] = None
    answer_raw: Any = None
    fallback_raw: Any = None

    # logging
    decision_log: Dict[str, Any] = field(default_factory=lambda: {
        "query": None,
        "profiler": {},
        "query_profile": {},
        "candidate_configs": [],
        "scheduler": {},
        "execution_config": {},
        "probe": {},
        "scheduler_steps": [],
        "window_selection": {},
        "answer_stage": {},
        "stop_reason": None,
    })

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def reasoning_metrics(state: BeliefState) -> dict:
    return {
        "distinct_windows": state.distinct_windows,
        "best_relevance_score": state.best_relevance_score,
        "score_improvement": state.score_improvement,
        "dense_seconds_encoded": state.dense_seconds_encoded,
        "approx_frames_encoded": state.approx_frames_encoded,
        "inspect_wallclock_s": state.inspect_wallclock_s,
        "steps": state.steps,
        "windows": list(state.windows),
        "probe_wallclock_s": getattr(state, "probe_wallclock_s", None),
        "probe_segments_scanned": getattr(state, "probe_segments_scanned", None),
    }

def detect_direction(q: str) -> str:
    q = q.lower()
    after = any(w in q for w in ["after", "afterwards", "then", "next", "following", "later"])
    before = any(w in q for w in ["before", "previously", "earlier"])
    if after and not before:
        return "after"
    if before and not after:
        return "before"
    return "after"

def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def estimate_config_cost(config: ExecutionConfig) -> float:
    cost = 0.0

    if config.run_clip_probe:
        cost += 0.5 * config.probe_fps * max(1, config.probe_topk / 4)

    if config.run_asr:
        cost += 1.5 * config.asr_chunk_topk

    if config.run_ocr:
        cost += 2.0 * config.max_vlm_windows

    if config.run_vlm:
        cost += (
            4.0
            + 0.7 * config.answer_max_frames_per_window
            + 0.5 * config.answer_max_images_total
        )

    if config.run_fallback_vlm:
        cost += 3.0

    return cost

def budget_threshold_for_load(scheduler_state: SchedulerState) -> float:
    if scheduler_state.load_level == "low":
        return 100.0
    if scheduler_state.load_level == "medium":
        return 16.0
    return 8.0


# ---------------------------------------------------------------------
# Profiling + config adaptation
# ---------------------------------------------------------------------

def infer_query_profile(ctx: RunContext) -> None:
    #Run the LLM based profiler on the query.
    #Returns:
    # - policy: raw execution policy suggested by the profiler
    # - profiler_analysis: semantic analysis of the query
    policy, profiler_analysis = profile_query_llm(
        ctx.query,
        base_url=ctx.profiler_base_url,
        model=ctx.profiler_model,
        temperature=0.0,
        timeout_s=30.0,
    )
    # Save raw profiler outputs in the context for later inspection/debugging.
    ctx.raw_policy = policy
    ctx.profiler_analysis = profiler_analysis
    # Infer temporal direction words from the query, e.g. before/after/next.
    ctx.direction = detect_direction(ctx.query)

    #take messy/raw profiler output and convert it into one standard profile object
    qp = QueryProfile(
        mode=str(_safe_get(policy, "mode", "attribute")),
        preferred_tools=tuple(_safe_get(policy, "preferred_tools", ())),
        primary_question_target=str(profiler_analysis.get("primary_question_target", "") or "").strip(),
        reference_type=str(_safe_get(policy, "reference_type", "") or "").strip(),
        needs_spoken_content=bool(profiler_analysis.get("needs_spoken_content", False)),
        visual_detail_level=str(profiler_analysis.get("visual_detail_level", "normal")),
        object_scale=str(profiler_analysis.get("object_scale", "normal")),
        needs_precise_local_view=bool(profiler_analysis.get("needs_precise_local_view", False)),
        require_temporal_pair=bool(_safe_get(policy, "require_temporal_pair", False)),
        probe_fps=float(_safe_get(policy, "probe_fps", 1.0)),
        probe_seg_len_s=float(_safe_get(policy, "probe_seg_len_s", 5.0)),
        probe_topk=int(_safe_get(policy, "probe_topk", 8)),
        window_len_s=float(_safe_get(policy, "window_len_s", 4.0)),
        action_topk=int(_safe_get(policy, "action_topk", 8)),
        strides=tuple(_safe_get(policy, "strides", (0.5,))),
        eps_marginal_gain=float(_safe_get(policy, "eps_marginal_gain", 0.05)),
        answer_tier=str(_safe_get(policy, "answer_tier", "medium")),
        fallback_answer_tier=str(_safe_get(policy, "fallback_answer_tier", "none")),
        cheap_answer_tier=str(_safe_get(policy, "cheap_answer_tier", "cheap")),
        enable_cheap_stage=bool(_safe_get(policy, "enable_cheap_stage", False)),
        text_answer_min_conf=float(_safe_get(policy, "text_answer_min_conf", 0.75)),
        min_answer_conf=float(_safe_get(policy, "min_answer_conf", 0.60)),
    )

    ctx.query_profile = qp
    ctx.decision_log["query"] = ctx.query
    ctx.decision_log["profiler"] = {
        "profiler_model": ctx.profiler_model,
        "profiler_base_url": ctx.profiler_base_url,
        "analysis": profiler_analysis,
        "compiled_policy": asdict(policy) if hasattr(policy, "__dataclass_fields__") else str(policy),
    }
    ctx.decision_log["query_profile"] = asdict(qp)

    print("PROFILER ANALYSIS:", profiler_analysis)
    print("QUERY PROFILE:", asdict(qp))

#first adaptation step
def candidate_configs_from_profile(qp: QueryProfile) -> list[ExecutionConfig]:
    needs_high_detail = (
        qp.visual_detail_level == "high"
        or qp.object_scale == "small"
        or qp.needs_precise_local_view
    )

    base_answer_frames = 4 if needs_high_detail else 2
    base_answer_images = 4 if needs_high_detail else 2
    base_jpeg = 92 if needs_high_detail else 85

    cfgs: list[ExecutionConfig] = []

    # Speech-heavy / direct spoken answer: cheap ASR-first path
    if qp.primary_question_target == "spoken_span" or qp.preferred_tools == ("asr",):
        cfgs.append(
            ExecutionConfig(
                name="asr_cheap",
                proposal_strategy="asr_first",
                run_clip_probe=False,
                run_search_loop=False,
                run_vlm=False,
                run_asr=True,
                run_cheap_text_stage=True,
                run_fallback_vlm=False,
                preferred_tools=("asr",),
                asr_chunk_len_s=30.0,
                asr_chunk_topk=5,
                cheap_answer_tier=qp.cheap_answer_tier,
                text_answer_min_conf=qp.text_answer_min_conf,
                estimated_cost=0.0,
            )
        )
        cfgs.append(
            ExecutionConfig(
                name="asr_plus_vlm",
                proposal_strategy="asr_first",
                run_clip_probe=False,
                run_search_loop=False,
                run_vlm=True,
                run_asr=True,
                run_cheap_text_stage=True,
                run_fallback_vlm=(qp.fallback_answer_tier != "none"),
                preferred_tools=("asr",),
                answer_tier=qp.answer_tier,
                fallback_answer_tier=qp.fallback_answer_tier,
                answer_max_frames_per_window=2,
                answer_max_images_total=2,
                answer_jpeg_quality=85,
                cheap_answer_tier=qp.cheap_answer_tier,
                text_answer_min_conf=qp.text_answer_min_conf,
                min_answer_conf=qp.min_answer_conf,
                estimated_cost=0.0,
            )
        )

    # OCR-heavy
    if "ocr" in qp.preferred_tools and "asr" not in qp.preferred_tools:
        cfgs.append(
            ExecutionConfig(
                name="ocr_cheap",
                proposal_strategy="clip_first",
                run_clip_probe=True,
                run_search_loop=True,
                run_vlm=False,
                run_ocr=True,
                run_asr=False,
                run_cheap_text_stage=True,
                preferred_tools=("ocr",),
                probe_fps=qp.probe_fps,
                probe_seg_len_s=qp.probe_seg_len_s,
                probe_topk=qp.probe_topk,
                window_len_s=qp.window_len_s,
                action_topk=qp.action_topk,
                strides=qp.strides,
                require_temporal_pair=qp.require_temporal_pair,
                eps_marginal_gain=qp.eps_marginal_gain,
                ocr_max_frames=8,
                max_vlm_windows=2,
                cheap_answer_tier=qp.cheap_answer_tier,
                text_answer_min_conf=qp.text_answer_min_conf,
                estimated_cost=0.0,
            )
        )

    # Visual-only or visual-primary
    if "ocr" not in qp.preferred_tools and "asr" not in qp.preferred_tools:
        cfgs.append(
             ExecutionConfig(
                name="visual_medium",
                proposal_strategy="clip_first",
                run_clip_probe=True,
                run_search_loop=True,
                run_vlm=True,
                run_ocr=False,
                run_asr=False,
                run_cheap_text_stage=False,
                run_fallback_vlm=False,
                preferred_tools=("clip",),
                probe_fps=qp.probe_fps,
                probe_seg_len_s=qp.probe_seg_len_s,
                probe_topk=qp.probe_topk,
                window_len_s=qp.window_len_s,
                action_topk=qp.action_topk,
                strides=qp.strides,
                require_temporal_pair=qp.require_temporal_pair,
                eps_marginal_gain=qp.eps_marginal_gain,
                answer_tier=qp.answer_tier,
                answer_max_frames_per_window=2,
                answer_max_images_total=2,
                answer_jpeg_quality=85,
                min_answer_conf=qp.min_answer_conf,
                estimated_cost=0.0,
            )
        )
        cfgs.append(
            ExecutionConfig(
                name="visual_expensive",
                proposal_strategy="clip_first",
                run_clip_probe=True,
                run_search_loop=True,
                run_vlm=True,
                run_ocr=False,
                run_asr=False,
                run_cheap_text_stage=False,
                run_fallback_vlm=(qp.fallback_answer_tier != "none"),
                preferred_tools=("clip",),
                probe_fps=qp.probe_fps,
                probe_seg_len_s=qp.probe_seg_len_s,
                probe_topk=max(qp.probe_topk, 8),
                window_len_s=qp.window_len_s,
                action_topk=max(qp.action_topk, 8),
                strides=qp.strides,
                require_temporal_pair=qp.require_temporal_pair,
                eps_marginal_gain=qp.eps_marginal_gain,
                answer_tier=qp.answer_tier,
                fallback_answer_tier=qp.fallback_answer_tier,
                answer_max_frames_per_window=base_answer_frames,
                answer_max_images_total=base_answer_images,
                answer_jpeg_quality=base_jpeg,
                min_answer_conf=qp.min_answer_conf,
                estimated_cost=0.0,
            )
        )

    # Mixed modality
    if "asr" in qp.preferred_tools and "ocr" in qp.preferred_tools:
        cfgs.append(
            ExecutionConfig(
                name="mixed_full",
                proposal_strategy="hybrid",
                run_clip_probe=True,
                run_search_loop=True,
                run_vlm=True,
                run_ocr=True,
                run_asr=True,
                run_cheap_text_stage=True,
                run_fallback_vlm=(qp.fallback_answer_tier != "none"),
                preferred_tools=qp.preferred_tools,
                probe_fps=qp.probe_fps,
                probe_seg_len_s=qp.probe_seg_len_s,
                probe_topk=qp.probe_topk,
                window_len_s=qp.window_len_s,
                action_topk=qp.action_topk,
                strides=qp.strides,
                require_temporal_pair=qp.require_temporal_pair,
                eps_marginal_gain=qp.eps_marginal_gain,
                answer_tier=qp.answer_tier,
                fallback_answer_tier=qp.fallback_answer_tier,
                answer_max_frames_per_window=base_answer_frames,
                answer_max_images_total=base_answer_images,
                answer_jpeg_quality=base_jpeg,
                cheap_answer_tier=qp.cheap_answer_tier,
                text_answer_min_conf=qp.text_answer_min_conf,
                min_answer_conf=qp.min_answer_conf,
                estimated_cost=0.0,
            )
        )

    # ASR + visual mixed
    if "asr" in qp.preferred_tools and "ocr" not in qp.preferred_tools:
        cfgs.append(
            ExecutionConfig(
                name="speech_visual",
                proposal_strategy="hybrid",
                run_clip_probe=True,
                run_search_loop=True,
                run_vlm=True,
                run_ocr=False,
                run_asr=True,
                run_cheap_text_stage=True,
                run_fallback_vlm=(qp.fallback_answer_tier != "none"),
                preferred_tools=qp.preferred_tools,
                probe_fps=qp.probe_fps,
                probe_seg_len_s=qp.probe_seg_len_s,
                probe_topk=qp.probe_topk,
                window_len_s=qp.window_len_s,
                action_topk=qp.action_topk,
                strides=qp.strides,
                require_temporal_pair=qp.require_temporal_pair,
                eps_marginal_gain=qp.eps_marginal_gain,
                answer_tier=qp.answer_tier,
                fallback_answer_tier=qp.fallback_answer_tier,
                answer_max_frames_per_window=2,
                answer_max_images_total=2,
                answer_jpeg_quality=85,
                cheap_answer_tier=qp.cheap_answer_tier,
                text_answer_min_conf=qp.text_answer_min_conf,
                min_answer_conf=qp.min_answer_conf,
                estimated_cost=0.0,
            )
        )

    for cfg in cfgs:
        cfg.estimated_cost = estimate_config_cost(cfg)

    if not cfgs:
        cfgs.append(
            ExecutionConfig(
                name="default_visual",
                proposal_strategy="clip_first",
                run_clip_probe=True,
                run_search_loop=True,
                run_vlm=True,
                answer_tier=qp.answer_tier,
                answer_max_frames_per_window=2,
                answer_max_images_total=2,
                answer_jpeg_quality=85,
                estimated_cost=0.0,
            )
        )
        cfgs[-1].estimated_cost = estimate_config_cost(cfgs[-1])

    return cfgs

def config_priority_for_load(cfg: ExecutionConfig, scheduler_state: SchedulerState) -> tuple:
    """
    Higher tuple is better.
    Under low load, prefer richer multimodal / non-CLIP-only configs.
    Under high load, prefer cheaper configs.
    """
    multimodal_score = int(cfg.run_asr) + int(cfg.run_ocr) + int(cfg.run_vlm)
    clip_only_penalty = int(
        cfg.run_clip_probe and not cfg.run_asr and not cfg.run_ocr
    )

    if scheduler_state.load_level == "low":
        return (
            multimodal_score,                  # prefer more modalities
            int(cfg.proposal_strategy == "hybrid"),
            int(cfg.proposal_strategy == "asr_first"),
            -clip_only_penalty,                # avoid pure clip-first if alternatives exist
            cfg.estimated_cost,                # among similar configs, allow richer cost
        )

    if scheduler_state.load_level == "medium":
        return (
            int(cfg.proposal_strategy == "hybrid"),
            multimodal_score,
            -clip_only_penalty,
            -cfg.estimated_cost,               # slightly prefer cheaper
        )

    # high load
    return (
        -cfg.estimated_cost,                  # cheapest first
        int(cfg.run_cheap_text_stage),
        int(not cfg.run_vlm),
    )

def pick_fitting_config(
    candidate_configs: list[ExecutionConfig],
    scheduler_state: SchedulerState,
) -> ExecutionConfig:
    threshold = budget_threshold_for_load(scheduler_state)

    fitting = [cfg for cfg in candidate_configs if cfg.estimated_cost <= threshold]
    pool = fitting if fitting else candidate_configs

    pool = sorted(
        pool,
        key=lambda cfg: config_priority_for_load(cfg, scheduler_state),
        reverse=True,
    )
    return pool[0]

def filter_candidate_configs_for_load(
    candidate_configs: list[ExecutionConfig],
    scheduler_state: SchedulerState,
) -> list[ExecutionConfig]:
    """
    Optional pre-filter before config selection.

    Under low load:
    - prefer richer configs if they exist
    - specifically keep configs that use ASR, OCR, or non-clip-only
      proposal strategies such as hybrid / asr_first

    Under medium/high load:
    - do not filter; let the ranking function decide

    If no richer configs exist, return the original list unchanged.
    """
    if scheduler_state.load_level != "low":
        return candidate_configs

    richer = [
        cfg for cfg in candidate_configs
        if cfg.run_asr or cfg.run_ocr or cfg.proposal_strategy in {"hybrid", "asr_first"}
    ]
    return richer if richer else candidate_configs

def select_config_with_override(
    candidate_configs: list[ExecutionConfig],
    scheduler_state: SchedulerState,
    force_config_name: Optional[str] = None,
) -> ExecutionConfig:
    if force_config_name is not None:
        for cfg in candidate_configs:
            if cfg.name == force_config_name:
                return cfg
        available = [cfg.name for cfg in candidate_configs]
        raise ValueError(
            f"Unknown force_config_name={force_config_name!r}. "
            f"Available configs: {available}"
        )

    return pick_fitting_config(candidate_configs, scheduler_state)

def list_candidate_config_names(
    query: str,
    *,
    profiler_model: str,
    profiler_base_url: str,
) -> list[str]:
    tmp = RunContext(
        query=query,
        video="",
        profiler_model=profiler_model,
        profiler_base_url=profiler_base_url,
    )
    infer_query_profile(tmp)
    return [cfg.name for cfg in candidate_configs_from_profile(tmp.query_profile)]


def setup_run(
    ctx: RunContext,
    *,
    max_dense_seconds: float,
    max_frames: int,
    max_wallclock_s: float,
    scheduler_state: SchedulerState,
    force_config_name: Optional[str] = None,
    force_execution_config: Optional[ExecutionConfig] = None,
) -> None:
    # Step 1:
    # Run the query profiler to produce a normalized QueryProfile.
    # This gives us the query type, preferred tools, and default knobs.
    infer_query_profile(ctx)
    # Step 2:
    # Generate all candidate execution configs that are compatible
    # with the query profile.
    candidate_configs = candidate_configs_from_profile(ctx.query_profile)

    # Step 3:
    # Only apply load-based filtering when we are NOT forcing a config.
    # Why:
    # If force_config_name is provided (e.g. "visual_expensive" for a baseline),
    # we should not remove that config before override selection happens.
    # Otherwise, low-load filtering could remove the forced config an cause an "Unknown force_config_name" error.
    if force_config_name is None:
        candidate_configs = [force_execution_config]
        ctx.execution_config = force_execution_config
    else:
        candidate_configs = candidate_configs_from_profile(ctx.query_profile)
        if force_config_name is None:
            candidate_configs = filter_candidate_configs_for_load(
                candidate_configs,
                scheduler_state
            )
        ctx.execution_config = select_config_with_override(
            candidate_configs,
            scheduler_state,
            force_config_name=force_config_name,
        )
            
    # Step 4:
    # Pick the final execution config.
    # - If force_config_name is set, this bypasses normal load-aware selection
    #   and directly chooses the requested config.
    # - Otherwise, the scheduler selects the best fitting config according
    #   to cost budget and load-aware priority ranking.

    ctx.state = BeliefState()
    ctx.budget = Budget(
        max_dense_seconds=max_dense_seconds,
        max_frames=max_frames,
        max_wallclock_s=max_wallclock_s,
    )

    ctx.decision_log["candidate_configs"] = [asdict(c) for c in candidate_configs]
    ctx.decision_log["scheduler"] = {
        "load_level": scheduler_state.load_level,
        "current_active_requests": scheduler_state.current_active_requests,
        "available_memory_fraction": scheduler_state.available_memory_fraction,
        "queue_delay_s": scheduler_state.queue_delay_s,
        "budget_threshold": budget_threshold_for_load(scheduler_state),
        "force_config_name": force_config_name,
        "override_used": force_config_name is not None,
    }
    ctx.decision_log["execution_config"] = asdict(ctx.execution_config)

    print("SELECTED EXECUTION CONFIG:", asdict(ctx.execution_config))


# ---------------------------------------------------------------------
# Probe / search
# ---------------------------------------------------------------------

def run_probe(ctx: RunContext) -> None:
    cfg = ctx.execution_config
    if not cfg.run_clip_probe:
        return

    probe_start = time.time()
    ctx.clip_candidates = probe_index(
        video=ctx.video,
        query=ctx.query,
        fps=cfg.probe_fps,
        segment_len_s=cfg.probe_seg_len_s,
        topk=cfg.probe_topk,
    )
    probe_wall = time.time() - probe_start

    print("TOP CLIP CANDIDATES:")
    for i, c in enumerate(ctx.clip_candidates[:20]):
        print(i, {
            "t0": getattr(c, "t0", None),
            "t1": getattr(c, "t1", None),
            "score": getattr(c, "score", None),
        })

    ctx.decision_log["probe"] = {
        "proposal_strategy": cfg.proposal_strategy,
        "probe_fps": cfg.probe_fps,
        "probe_seg_len_s": cfg.probe_seg_len_s,
        "probe_topk": cfg.probe_topk,
        "num_candidates": len(ctx.clip_candidates),
        "probe_wallclock_s": probe_wall,
        "top_candidates_preview": [
            {
                "t0": getattr(c, "t0", None),
                "t1": getattr(c, "t1", None),
                "score": getattr(c, "score", None),
            }
            for c in ctx.clip_candidates[:10]
        ],
    }

    if hasattr(ctx.state, "update_from_probe"):
        ctx.state.update_from_probe(
            wallclock_s=probe_wall,
            segments_scanned=len(ctx.clip_candidates),
            fps=cfg.probe_fps,
        )
    else:
        ctx.state.probe_wallclock_s = getattr(ctx.state, "probe_wallclock_s", 0.0) + float(probe_wall)


def build_actions_for_config(ctx: RunContext) -> None:
    cfg = ctx.execution_config
    if not cfg.run_search_loop:
        return

    q = ctx.query.lower()

    is_next_action = any(x in q for x in [
        "what did he do next",
        "what did she do next",
        "what happened next",
        "what happens next",
        "do next",
    ])

    is_fine_microevent = any(x in q for x in [
        "immediately",
        "exact moment",
        "the instant",
    ])

    adaptive_window_len_s = cfg.window_len_s

    if cfg.require_temporal_pair or ctx.query_profile.mode == "microevent":
        if is_fine_microevent:
            adaptive_window_len_s = 2.0
        elif is_next_action:
            adaptive_window_len_s = 4.0
        else:
            adaptive_window_len_s = 3.0

    ctx.actions = build_action_space(
        ctx.clip_candidates,
        window_len=adaptive_window_len_s,
        topk=cfg.action_topk,
        strides=list(cfg.strides),
        resolutions=["high"],
    )


def run_temporal_followups(ctx: RunContext, action) -> None:
    cfg = ctx.execution_config
    direction = detect_direction(ctx.query)
    width = action.t1 - action.t0

    followups = propose_followup_windows(
        (action.t0, action.t1),
        direction=direction,
        width_s=width,
        gaps_s=[k * width for k in range(12)],
    )

    best_before = ctx.state.best_relevance_score

    for (ft0, ft1) in followups:
        if ft1 <= ft0 or (ft0, ft1) in ctx.state.windows:
            continue

        follow_res = inspect_window(
            video=ctx.video,
            t0=ft0,
            t1=ft1,
            stride=action.stride,
            resolution=action.resolution,
            query=ctx.query,
            source="clip",
        )

        ctx.trace.append(
            {
                "action": {
                    "t0": ft0,
                    "t1": ft1,
                    "stride": action.stride,
                    "resolution": action.resolution,
                    "followup_of": [action.t0, action.t1],
                    "direction": direction,
                    "gap_s": (ft0 - action.t1) if direction == "after" else (action.t0 - ft1),
                },
                "result": {
                    "relevance_score": follow_res.relevance_score,
                    "frames_encoded": follow_res.frames_encoded,
                    "dense_seconds": follow_res.dense_seconds,
                    "wallclock_s": follow_res.wallclock_s,
                    "source": getattr(follow_res, "source", "clip"),
                },
            }
        )
        ctx.state.update_from_window(follow_res)

        if (ctx.state.best_relevance_score - best_before) >= cfg.eps_marginal_gain:
            break

#Main search loop:
# - pick next candididate window
# - inspect it(compute relevance)
# - log result and update state
# - optionally expand to temporal follow-ups
# Stops when budget or progress criteria are met
def run_search_loop(ctx: RunContext) -> None:
    cfg = ctx.execution_config
    if not cfg.run_search_loop:
        return

    while not stopping_rule(ctx.state, ctx.raw_policy, ctx.budget):
        action = pick_next_action(ctx.raw_policy, ctx.state, ctx.actions, direction=ctx.direction)
        if action is None:
            break

        res = inspect_window(
            video=ctx.video,
            t0=action.t0,
            t1=action.t1,
            stride=action.stride,
            resolution=action.resolution,
            query=ctx.query,
            source="clip",
        )

        ctx.trace.append(
            {
                "action": {
                    "t0": action.t0,
                    "t1": action.t1,
                    "stride": action.stride,
                    "resolution": action.resolution,
                },
                "result": {
                    "relevance_score": res.relevance_score,
                    "frames_encoded": res.frames_encoded,
                    "dense_seconds": res.dense_seconds,
                    "wallclock_s": res.wallclock_s,
                    "source": getattr(res, "source", "clip"),
                },
            }
        )

        ctx.state.update_from_window(res)

        if (
            ctx.query_profile.mode in {"microevent", "ordering"}
            and cfg.require_temporal_pair
            and ctx.state.steps == 1
        ):
            run_temporal_followups(ctx, action)


# ---------------------------------------------------------------------
# Evidence selection helpers
# ---------------------------------------------------------------------

#Select final windows for VLM answering
# For standard queries, keep the top-scoring inspected windows
# For microevent queries, use the best window as an anchor and add 
# temporally adjacent follow-up windows to preserve before/after context.
def select_vlm_windows(query_mode: str, trace, *, direction: str, k_followups: int = 1):
    if not trace:
        return []

    best = max(trace, key=lambda e: float(e["result"].get("relevance_score", 0.0)))
    anchor = best["action"]
    a0, a1 = float(anchor["t0"]), float(anchor["t1"])

    scored = []
    for e in trace:
        act = e["action"]
        res = e["result"]
        t0, t1 = float(act["t0"]), float(act["t1"])
        s = float(res.get("relevance_score", 0.0))
        scored.append(((t0, t1), s))

    print("WINDOW SCORES:", [{"t0": t0, "t1": t1, "score": s} for ((t0, t1), s) in scored])

    if query_mode != "microevent":
        scored.sort(key=lambda x: -x[1])
        selected = [w for (w, _) in scored[: min(3, len(scored))]]
        print("SELECTION RULE:", "top_score")
        print("SELECTED VLM WINDOWS:", selected)
        return selected

    if direction == "after":
        follow = [x for x in scored if x[0][0] >= a1]
        follow.sort(key=lambda x: (-x[1], x[0][0] - a1))
    else:
        follow = [x for x in scored if x[0][1] <= a0]
        follow.sort(key=lambda x: (-x[1], a0 - x[0][1]))

    out = [(a0, a1)]
    out.extend([w for (w, _) in follow[:k_followups]])
    out = list(dict.fromkeys(out))
    out.sort(key=lambda w: w[0])

    print("SELECTION RULE:", "microevent_followup")
    print("ANCHOR WINDOW:", (a0, a1))
    print("SELECTED VLM WINDOWS:", out)
    return out


import cv2

def get_video_duration_s(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps is None or fps <= 0 or frame_count is None or frame_count <= 0:
        raise RuntimeError(f"Could not determine duration for video: {video_path}")

    return float(frame_count) / float(fps)


#Coarse ASR-first window proposal:
#partition the video into fixed-duration chunks, transcribe each chunk,
#score chunks by transcript overlap with query, and return the
#top-ranked windows as ASR candidates for downstream answering.

from concurrent.futures import ThreadPoolExecutor, as_completed

def propose_asr_windows_from_chunks(
    ctx: RunContext,
    *,
    asr_model: str,
    asr_base_url: str,
    chunk_len_s: float = 30.0,
    topk: int = 5,
    max_workers: int = 8,
) -> list[dict]:
    """
    Returns selected scored ASR chunks, not just windows.
    Each item has:
      {"window": (t0, t1), "score": ..., "asr": ..., "text_preview": ...}

    Logged timing:
      - asr_chunk_scan_wallclock_s: end-to-end elapsed time for scanning this video
      - asr_chunk_request_time_sum_s: sum of per-chunk request_s (excludes extraction)
      - asr_chunk_request_time_avg_s: average per-chunk request_s
      - asr_chunk_request_time_max_s: max per-chunk request_s
    """
    if ctx.clip_candidates:
        max_t1 = max(float(getattr(c, "t1", 0.0)) for c in ctx.clip_candidates)
    elif ctx.vlm_windows:
        max_t1 = max(float(t1) for _, t1 in ctx.vlm_windows)
    else:
        max_t1 = get_video_duration_s(ctx.video)

    if max_t1 <= 0:
        return []

    windows: list[tuple[float, float]] = []
    t = 0.0
    while t < max_t1:
        windows.append((t, min(t + chunk_len_s, max_t1)))
        t += chunk_len_s

    q_words = set(ctx.query.lower().replace("?", "").replace(",", "").split())

    def _run_one(window: tuple[float, float]) -> dict:
        t0, t1 = window
        a = asr_window(
            ctx.video,
            t0,
            t1,
            stride=0.5,
            resolution="high",
            query=ctx.query,
            model=asr_model,
            base_url=asr_base_url,
        )

        txt = str(((a.evidence or {}).get("asr_text", "") or "")).lower()
        txt_words = set(txt.replace("?", "").replace(",", "").split())
        overlap = len(q_words & txt_words)

        timing = (a.evidence or {}).get("timing", {}) or {}
        request_s = timing.get("request_s", None)

        return {
            "window": (t0, t1),
            "score": overlap,
            "asr": a,
            "text_preview": txt[:200],
            "request_s": float(request_s) if request_s is not None else None,
        }

    scan_start = time.time()

    scored: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_one, w) for w in windows]
        for fut in as_completed(futures):
            scored.append(fut.result())

    scan_end = time.time()
    scan_wallclock_s = float(scan_end - scan_start)

    scored.sort(key=lambda x: x["score"], reverse=True)

    request_times = [
        float(s["request_s"])
        for s in scored
        if s.get("request_s") is not None
    ]

    proposal_total_request_s = sum(request_times)
    proposal_avg_request_s = (
        proposal_total_request_s / len(request_times) if request_times else None
    )
    proposal_max_request_s = max(request_times) if request_times else None

    ctx.decision_log["answer_stage"]["asr_chunk_scan_wallclock_s"] = scan_wallclock_s
    ctx.decision_log["answer_stage"]["asr_chunk_request_time_sum_s"] = proposal_total_request_s
    ctx.decision_log["answer_stage"]["asr_chunk_request_time_avg_s"] = proposal_avg_request_s
    ctx.decision_log["answer_stage"]["asr_chunk_request_time_max_s"] = proposal_max_request_s

    print("WHOLE VIDEO ASR SCAN WALLCLOCK S:", scan_wallclock_s)
    print("TOTAL ASR REQUEST TIME S (NO EXTRACTION):", proposal_total_request_s)
    print("AVG ASR CHUNK REQUEST TIME S (NO EXTRACTION):", proposal_avg_request_s)
    print("MAX ASR CHUNK REQUEST TIME S (NO EXTRACTION):", proposal_max_request_s)

    ctx.decision_log["answer_stage"]["asr_chunk_scan"] = [
        {
            "t0": float(s["window"][0]),
            "t1": float(s["window"][1]),
            "score": float(s["score"]),
            "text_preview": s["text_preview"],
            "request_s": s["request_s"],
        }
        for s in scored
    ]

    top = [s for s in scored[:topk] if s["score"] > 0]
    if not top:
        top = scored[:topk]

    return top

def merge_time_windows(
    windows_a: list[tuple[float, float]],
    windows_b: list[tuple[float, float]],
    max_windows: int = 2,
) -> list[tuple[float, float]]:
    xs = sorted(
        [(float(t0), float(t1)) for (t0, t1) in (windows_a + windows_b)],
        key=lambda w: (w[0], w[1]),
    )
    if not xs:
        return []

    merged = [xs[0]]
    for (t0, t1) in xs[1:]:
        last_t0, last_t1 = merged[-1]
        if t0 <= last_t1:
            merged[-1] = (last_t0, max(last_t1, t1))
        else:
            merged.append((t0, t1))

    return merged[:max_windows]


# ---------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------

#Select final visual evidence windows from the search trace, cap their count
#expand the primary window for extra content, store them for VLM/OCR use.
def select_visual_windows(ctx: RunContext) -> None:
    cfg = ctx.execution_config
    if not cfg.run_vlm and not cfg.run_ocr:
        ctx.vlm_windows = []
        return

    ctx.vlm_windows = select_vlm_windows(
        ctx.query_profile.mode,
        ctx.trace,
        direction=ctx.direction,
        k_followups=1,
    )[:cfg.max_vlm_windows]

    if not ctx.vlm_windows:
        return

    first_t0, first_t1 = ctx.vlm_windows[0]
    ctx.vlm_windows[0] = (
        max(0.0, first_t0 - cfg.expand_first_window_s),
        first_t1 + cfg.expand_first_window_s,
    )

    print("EXPANDED FIRST VLM WINDOW:", ctx.vlm_windows[0])

    ctx.decision_log["window_selection"]["vlm_windows"] = [
        {"t0": float(t0), "t1": float(t1)}
        for (t0, t1) in ctx.vlm_windows
    ]


def collect_ocr_evidence(
    ctx: RunContext,
    *,
    ocr_model: Optional[str],
    ocr_base_url: Optional[str],
) -> None:
    cfg = ctx.execution_config
    if not cfg.run_ocr:
        return

    if not ocr_model or not ocr_base_url:
        raise ValueError("OCR requested by config, but ocr_model or ocr_base_url is not set.")

    for (t0, t1) in ctx.vlm_windows:
        o = ocr_window(
            ctx.video,
            t0,
            t1,
            stride=0.5,
            resolution="high",
            max_frames=cfg.ocr_max_frames,
            model=ocr_model,
            base_url=ocr_base_url,
        )
        ctx.ocr_outputs.append(((t0, t1), o))


def collect_asr_evidence(
    ctx: RunContext,
    *,
    asr_model: Optional[str],
    asr_base_url: Optional[str],
) -> None:
    cfg = ctx.execution_config
    if not cfg.run_asr:
        return

    if not asr_model or not asr_base_url:
        raise ValueError("ASR requested by config, but asr_model or asr_base_url is not set.")

    if cfg.proposal_strategy == "asr_first":
        selected = propose_asr_windows_from_chunks(
            ctx,
            asr_model=asr_model,
            asr_base_url=asr_base_url,
            chunk_len_s=cfg.asr_chunk_len_s,
            topk=cfg.asr_chunk_topk,
        )
        ctx.asr_windows = [x["window"] for x in selected]
        ctx.asr_outputs = [(x["window"], x["asr"]) for x in selected]
    elif cfg.proposal_strategy == "hybrid":
        asr_from_chunks = propose_asr_windows_from_chunks(
            ctx,
            asr_model=asr_model,
            asr_base_url=asr_base_url,
            chunk_len_s=cfg.asr_chunk_len_s,
            topk=cfg.asr_chunk_topk,
        )
        asr_chunk_windows = [x["window"] for x in asr_from_chunks]
        ctx.asr_windows = merge_time_windows(
            ctx.vlm_windows,
            asr_chunk_windows,
            max_windows=cfg.max_vlm_windows,
        )
    else:
        ctx.asr_windows = list(ctx.vlm_windows)
        if not ctx.asr_windows:
            extra = []
            for c in ctx.clip_candidates[: min(8, len(ctx.clip_candidates))]:
                t0 = float(c.t0)
                t1 = float(c.t1)
                w = (max(0.0, t0 - 2.0), t1 + 2.0)
                if w not in extra:
                    extra.append(w)
            ctx.asr_windows = extra[:cfg.max_vlm_windows]

    print("ASR WINDOWS:", ctx.asr_windows)
    print("TOTAL ASR TIME FOR WHOLE VIDEO S:", ctx.total_asr_time_s)

    ctx.decision_log["answer_stage"]["asr_windows"] = [
        {"t0": float(t0), "t1": float(t1)}
        for (t0, t1) in ctx.asr_windows
    ]

    if cfg.proposal_strategy != "asr_first":
        for (t0, t1) in ctx.asr_windows:
            a = asr_window(
                ctx.video,
                t0,
                t1,
                stride=0.5,
                resolution="high",
                query=ctx.query,
                model=asr_model,
                base_url=asr_base_url,
            )
            ctx.asr_outputs.append(((t0, t1), a))
            ctx.total_asr_time_s += float(a.wallclock_s)

def collect_evidence(
    ctx: RunContext,
    *,
    ocr_model: Optional[str],
    ocr_base_url: Optional[str],
    asr_model: Optional[str],
    asr_base_url: Optional[str],
) -> None:
    select_visual_windows(ctx)
    collect_ocr_evidence(ctx, ocr_model=ocr_model, ocr_base_url=ocr_base_url)
    collect_asr_evidence(ctx, asr_model=asr_model, asr_base_url=asr_base_url)


# ---------------------------------------------------------------------
# Answer stages
# ---------------------------------------------------------------------

def build_text_evidence(ctx: RunContext) -> str:
    evidence_parts = []

    for (t0, t1), o in ctx.ocr_outputs:
        txts = (o.evidence or {}).get("ocr_text", [])
        if txts:
            evidence_parts.append(f"[OCR {t0:.1f}-{t1:.1f}] " + " ".join(txts))

    for (t0, t1), a in ctx.asr_outputs:
        txt = (a.evidence or {}).get("asr_text", "")
        if txt:
            evidence_parts.append(f"[ASR {t0:.1f}-{t1:.1f}] {txt}")

    return "\n".join(evidence_parts).strip()


def try_cheap_answer(ctx: RunContext) -> Optional[dict]:
    cfg = ctx.execution_config
    if not cfg.run_cheap_text_stage:
        return None

    evidence = build_text_evidence(ctx)
    print("CHEAP EVIDENCE:", evidence)

    if not evidence:
        return None

    text_model = pick_text_model(REG, cfg.cheap_answer_tier)
    if text_model is None:
        return None

    ta = TextAnswerer(
        TextAnswererConfig(
            model=text_model,
            base_url=ctx.profiler_base_url,
        )
    )

    cheap_pred, cheap_conf, cheap_raw = ta.answer_with_confidence(
        question=ctx.query,
        evidence=evidence,
    )

    print("CHEAP TEXT MODEL:", text_model)
    print("CHEAP PRED:", cheap_pred)
    print("CHEAP CONF:", cheap_conf)
    print("CHEAP RAW:", cheap_raw)

    if cheap_conf < cfg.text_answer_min_conf:
        return None

    return {
        "pred": cheap_pred,
        "query_profile": asdict(ctx.query_profile),
        "execution_config": asdict(ctx.execution_config),
        "trace": ctx.trace,
        "decision_log": ctx.decision_log,
        "routing": {
            "profiler_model": ctx.profiler_model,
            "answer_model": None,
            "text_answer_model": text_model,
            "text_answer_conf": cheap_conf,
            "fallback_answer_model": None,
            "fallback_used": False,
            "answer_conf": cheap_conf,
            "stage": "cheap_text",
            "answer_raw": cheap_raw,
        },
        "reasoning_metrics": reasoning_metrics(ctx.state),
    }


def run_vlm_answer(
    ctx: RunContext,
    *,
    answer_base_url: str,
    answer_max_tokens: int,
    force_answer_model: Optional[str],
) -> None:
    cfg = ctx.execution_config
    if not cfg.run_vlm:
        return

    if force_answer_model is not None:
        ctx.chosen_answer_model = force_answer_model
    else:
        ctx.chosen_answer_model = pick_answer_model(REG, cfg.answer_tier)

    if ctx.chosen_answer_model is None or not ctx.vlm_windows:
        return

    print("FALLING THROUGH TO VLM")
    print("ANSWER MODEL:", ctx.chosen_answer_model)
    print("ANSWER BASE URL:", answer_base_url)
    print("ANSWER INPUT WINDOWS:", ctx.vlm_windows)

    ans = VLLMAnswerer(
        VLLMAnswererConfig(
            model=ctx.chosen_answer_model,
            base_url=answer_base_url,
            max_tokens=answer_max_tokens,
            temperature=0.0,
        )
    )

    ctx.pred, ctx.answer_conf, ctx.answer_raw = ans.answer_with_confidence(
        video_path=ctx.video,
        windows=ctx.vlm_windows,
        question=ctx.query,
        sample_fps=1.0,
        max_frames_per_window=cfg.answer_max_frames_per_window,
        mode=ctx.query_profile.mode,
        max_windows=cfg.max_vlm_windows,
        max_images_total=cfg.answer_max_images_total,
        jpeg_quality=cfg.answer_jpeg_quality,
    )

    print("VLM ANSWER CONF:", ctx.answer_conf)
    print("VLM ANSWER RAW:", ctx.answer_raw)

    if (
        cfg.run_fallback_vlm
        and cfg.fallback_answer_tier != "none"
        and ctx.answer_conf is not None
        and float(ctx.answer_conf) < float(cfg.min_answer_conf)
    ):
        ctx.fallback_answer_model = pick_answer_model(REG, cfg.fallback_answer_tier)

        if ctx.fallback_answer_model is not None and ctx.fallback_answer_model != ctx.chosen_answer_model:
            ans2 = VLLMAnswerer(
                VLLMAnswererConfig(
                    model=ctx.fallback_answer_model,
                    base_url=answer_base_url,
                    max_tokens=answer_max_tokens,
                    temperature=0.0,
                )
            )
            pred2, conf2, fallback_raw = ans2.answer_with_confidence(
                video_path=ctx.video,
                windows=ctx.vlm_windows,
                question=ctx.query,
                sample_fps=1.0,
                max_frames_per_window=2,
                mode=ctx.query_profile.mode,
                max_windows=cfg.max_vlm_windows,
                max_images_total=2,
                jpeg_quality=85,
            )

            if float(conf2) >= float(ctx.answer_conf):
                ctx.pred = pred2
                ctx.answer_conf = conf2
                ctx.answer_raw = fallback_raw
                ctx.fallback_used = True


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

def build_result(ctx: RunContext) -> dict:
    return {
        "pred": ctx.pred,
        "query_profile": asdict(ctx.query_profile) if ctx.query_profile else None,
        "execution_config": asdict(ctx.execution_config) if ctx.execution_config else None,
        "trace": ctx.trace,
        "decision_log": ctx.decision_log,
        "routing": {
            "selected_config_name": ctx.execution_config.name if ctx.execution_config else None,
            "profiler_model": ctx.profiler_model,
            "answer_model": ctx.chosen_answer_model,
            "fallback_answer_model": ctx.fallback_answer_model,
            "fallback_used": ctx.fallback_used,
            "answer_conf": ctx.answer_conf,
            "answer_raw": ctx.answer_raw,
        },
        "reasoning_metrics": reasoning_metrics(ctx.state),
    }



# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------



def run(
    query: str,
    video: str,
    *,
    max_dense_seconds: float = 20.0,
    max_frames: int = 2000,
    max_wallclock_s: float = 60.0,
    profiler_model: str = "YOUR_TEXT_PROFILER_MODEL",
    profiler_base_url: str = "http://localhost:8001/v1",
    ocr_model: Optional[str] = None,
    ocr_base_url: Optional[str] = None,
    answer_base_url: str = "http://localhost:8000/v1",
    answer_max_tokens: int = 64,
    force_answer_model: Optional[str] = None,
    disable_answer: bool = False,
    asr_model: Optional[str] = None,
    asr_base_url: Optional[str] = None,
    load_level: LoadLevel = "low",
    current_active_requests: int = 0,
    available_memory_fraction: float = 1.0,
    queue_delay_s: float = 0.0,
    force_config_name: Optional[str] = None,
    force_execution_config: Optional[ExecutionConfig] = None,
):
    """
    force_config_name: 
        Optional ablation override. If None, the system uses normal load aware config
        For example, under high load the scheduler may choose a cheaper config
        such as "asr_cheap"; setting force_config_name bypasses that choice.
    force_execution_config:
        Optional fully fixed execution config. If provided, bypasses profiler-driven
        candidate generation and load-based config selection. Useful for true baselines
        like asr_for_all / ocr_for_all / unified_vlm.
    """


    ctx = RunContext(
        query=query,
        video=video,
        profiler_model=profiler_model,
        profiler_base_url=profiler_base_url,
    )

    scheduler_state = SchedulerState(
        load_level=load_level,
        current_active_requests=current_active_requests,
        available_memory_fraction=available_memory_fraction,
        queue_delay_s=queue_delay_s,
    )

    setup_run(
        ctx,
        max_dense_seconds=max_dense_seconds,
        max_frames=max_frames,
        max_wallclock_s=max_wallclock_s,
        scheduler_state=scheduler_state,
        force_config_name=force_config_name,
        force_execution_config=force_execution_config,
    )

    cfg = ctx.execution_config
    print("\n=== SELECTED CONFIG ===")
    print("config_name:", cfg.name)
    print("proposal_strategy:", cfg.proposal_strategy)
    print("preferred_tools:", cfg.preferred_tools)
    print("run_clip_probe:", cfg.run_clip_probe)
    print("run_search_loop:", cfg.run_search_loop)
    print("run_vlm:", cfg.run_vlm)
    print("run_ocr:", cfg.run_ocr)
    print("run_asr:", cfg.run_asr)
    print("run_cheap_text_stage:", cfg.run_cheap_text_stage)
    print("run_fallback_vlm:", cfg.run_fallback_vlm)
    print("estimated_cost:", cfg.estimated_cost)
    print("=======================\n")

    run_probe(ctx)
    build_actions_for_config(ctx)
    run_search_loop(ctx)

    if disable_answer:
        return build_result(ctx)

    collect_evidence(
        ctx,
        ocr_model=ocr_model,
        ocr_base_url=ocr_base_url,
        asr_model=asr_model,
        asr_base_url=asr_base_url,
    )

    cheap_result = try_cheap_answer(ctx)
    if cheap_result is not None:
        return cheap_result

    run_vlm_answer(
        ctx,
        answer_base_url=answer_base_url,
        answer_max_tokens=answer_max_tokens,
        force_answer_model=force_answer_model,
    )
    return build_result(ctx)



def make_fixed_baseline_config(method: str) -> Optional[ExecutionConfig]:
    """
    Return a true fixed baseline config that does NOT depend on the profiler.
    This is used for benchmarks like asr_for_all / ocr_for_all / unified_vlm.
    """
    if method == "asr_for_all":
        cfg = ExecutionConfig(
            name="asr_for_all_forced",
            proposal_strategy="asr_first",
            run_clip_probe=False,
            run_search_loop=False,
            run_vlm=False,
            run_ocr=False,
            run_asr=True,
            run_cheap_text_stage=True,
            run_fallback_vlm=False,
            preferred_tools=("asr",),
            asr_chunk_len_s=30.0,
            asr_chunk_topk=5,
            cheap_answer_tier="cheap",
            text_answer_min_conf=0.75,
        )
        cfg.estimated_cost = estimate_config_cost(cfg)
        return cfg

    if method == "ocr_for_all":
        cfg = ExecutionConfig(
            name="ocr_for_all_forced",
            proposal_strategy="clip_first",
            run_clip_probe=True,
            run_search_loop=True,
            run_vlm=False,
            run_ocr=True,
            run_asr=False,
            run_cheap_text_stage=True,
            run_fallback_vlm=False,
            preferred_tools=("ocr",),
            probe_fps=1.0,
            probe_seg_len_s=5.0,
            probe_topk=50,
            window_len_s=4.0,
            action_topk=50,
            strides=(0.5,),
            ocr_max_frames=8,
            max_vlm_windows=2,
            cheap_answer_tier="cheap",
            text_answer_min_conf=0.75,
        )
        cfg.estimated_cost = estimate_config_cost(cfg)
        return cfg

    if method == "ocr_asr_vlm_for_all":
        cfg = ExecutionConfig(
            name="ocr_asr_vlm_for_all_forced",
            proposal_strategy="hybrid",
            run_clip_probe=True,
            run_search_loop=True,
            run_vlm=True,
            run_ocr=True,
            run_asr=True,
            run_cheap_text_stage=True,
            run_fallback_vlm=False,
            preferred_tools=("ocr", "asr"),
            probe_fps=1.0,
            probe_seg_len_s=5.0,
            probe_topk=50,
            window_len_s=4.0,
            action_topk=50,
            strides=(0.5,),
            asr_chunk_len_s=30.0,
            asr_chunk_topk=5,
            ocr_max_frames=8,
            max_vlm_windows=2,
            expand_first_window_s=2.0,
            answer_tier="cheap",
            answer_max_frames_per_window=4,
            answer_max_images_total=4,
            answer_jpeg_quality=92,
            cheap_answer_tier="cheap",
            text_answer_min_conf=0.75,
            min_answer_conf=0.35,
        )
        cfg.estimated_cost = estimate_config_cost(cfg)
        return cfg

    if method == "unified_vlm":
        cfg = ExecutionConfig(
            name="unified_vlm_forced",
            proposal_strategy="clip_first",
            run_clip_probe=True,
            run_search_loop=True,
            run_vlm=True,
            run_ocr=False,
            run_asr=False,
            run_cheap_text_stage=False,
            run_fallback_vlm=False,
            preferred_tools=("clip",),
            probe_fps=1.0,
            probe_seg_len_s=5.0,
            probe_topk=50,
            window_len_s=4.0,
            action_topk=50,
            strides=(0.5,),
            max_vlm_windows=2,
            expand_first_window_s=2.0,
            answer_tier="cheap",
            answer_max_frames_per_window=4,
            answer_max_images_total=4,
            answer_jpeg_quality=92,
            min_answer_conf=0.35,
        )
        cfg.estimated_cost = estimate_config_cost(cfg)
        return cfg

    return None