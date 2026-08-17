from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conductor.profiler.resource_aware_selector import (
    Budget2GateSelector,
    ResourceAwareSelector,
    apply_gpu_pressure,
)


PARETO_CONFIGS = {
    "budget2": {
        "name": "budget2",
        "probe_fps": 0.015625,
        "probe_topk": 8,
        "window_len_s": 8,
        "vlm_budget": 2,
    },
    "scan0.0039_k8_budget32": {
        "name": "scan0.0039_k8_budget32",
        "probe_fps": 0.00390625,
        "probe_topk": 8,
        "window_len_s": 8,
        "vlm_budget": 32,
    },
    "w4_k8_budget16": {
        "name": "w4_k8_budget16",
        "probe_fps": 0.015625,
        "probe_topk": 8,
        "window_len_s": 4,
        "vlm_budget": 16,
    },
    "budget32": {
        "name": "budget32",
        "probe_fps": 0.015625,
        "probe_topk": 8,
        "window_len_s": 8,
        "vlm_budget": 32,
    },
    "scan0.03125_k8_budget32": {
        "name": "scan0.03125_k8_budget32",
        "probe_fps": 0.03125,
        "probe_topk": 8,
        "window_len_s": 8,
        "vlm_budget": 32,
    },
}


TIER_ORDER = [
    "budget2",
    "scan0.0039_k8_budget32",
    "w4_k8_budget16",
    "budget32",
    "scan0.03125_k8_budget32",
]


HARD_QUERY_TERMS = (
    "sequence",
    "summarize",
    "summary",
    "overall process",
    "key steps",
    "workflow",
    "turning point",
    "crucial moment",
    "before",
    "after",
    "then",
    "why",
    "compare",
    "relationship",
    "pattern",
    "repeatedly",
    "primary objective",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def query_gpu_state() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
            "avg_gpu_util_pct": None,
            "avg_mem_used_pct": None,
            "gpus": [],
        }

    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        idx, util, mem_used, mem_total = parts
        try:
            mem_used_f = float(mem_used)
            mem_total_f = float(mem_total)
            gpus.append(
                {
                    "index": int(idx),
                    "gpu_util_pct": float(util),
                    "mem_used_mib": mem_used_f,
                    "mem_total_mib": mem_total_f,
                    "mem_used_pct": (
                        100.0 * mem_used_f / mem_total_f
                        if mem_total_f
                        else 0.0
                    ),
                }
            )
        except ValueError:
            continue

    if not gpus:
        return {
            "available": False,
            "error": "nvidia-smi returned no parseable GPUs",
            "avg_gpu_util_pct": None,
            "avg_mem_used_pct": None,
            "gpus": [],
        }

    return {
        "available": True,
        "error": None,
        "avg_gpu_util_pct": mean(gpu["gpu_util_pct"] for gpu in gpus),
        "avg_mem_used_pct": mean(gpu["mem_used_pct"] for gpu in gpus),
        "gpus": gpus,
    }


def base_tier_from_gpu(
    gpu_state: dict[str, Any],
    *,
    low_util_pct: float,
    high_util_pct: float,
    critical_util_pct: float,
    high_mem_pct: float,
) -> str:
    if not gpu_state.get("available"):
        return "scan0.0039_k8_budget32"

    util = float(gpu_state.get("avg_gpu_util_pct") or 0.0)
    mem = float(gpu_state.get("avg_mem_used_pct") or 0.0)

    if util >= critical_util_pct or mem >= high_mem_pct:
        return "budget2"
    if util >= high_util_pct:
        return "scan0.0039_k8_budget32"
    if util <= low_util_pct and mem < 70:
        return "budget32"
    return "w4_k8_budget16"


def is_hard_query(row: dict[str, Any]) -> bool:
    text = str(row.get("question") or "").lower()
    if any(term in text for term in HARD_QUERY_TERMS):
        return True

    category = str(
        row.get("question_category")
        or row.get("vimio_profile")
        or row.get("topic_category")
        or ""
    ).lower()
    return any(
        term in category
        for term in (
            "temporal",
            "reasoning",
            "sequence",
            "process",
            "causal",
        )
    )


def duration_s(row: dict[str, Any]) -> float | None:
    value = row.get("duration_s")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def upgrade_tier(tier: str, steps: int = 1) -> str:
    idx = TIER_ORDER.index(tier)
    return TIER_ORDER[min(len(TIER_ORDER) - 1, idx + steps)]


def downgrade_tier(tier: str, steps: int = 1) -> str:
    idx = TIER_ORDER.index(tier)
    return TIER_ORDER[max(0, idx - steps)]


def choose_tier(
    row: dict[str, Any],
    *,
    base_tier: str,
    allow_hard_query_upgrade: bool,
    long_video_s: float,
) -> tuple[str, str]:
    tier = base_tier
    reasons = [f"base_gpu_tier={base_tier}"]

    dur = duration_s(row)
    if dur is not None and dur >= long_video_s:
        tier = min(
            tier,
            "scan0.0039_k8_budget32",
            key=TIER_ORDER.index,
        )
        reasons.append(f"long_video_s={dur:.1f}")

    if allow_hard_query_upgrade and is_hard_query(row):
        if tier != "budget2":
            tier = upgrade_tier(tier)
            reasons.append("hard_query_upgrade")
        else:
            reasons.append("hard_query_but_gpu_critical")

    return tier, ";".join(reasons)


def schedule_row(
    row: dict[str, Any],
    *,
    tier: str,
    reason: str,
    gpu_state: dict[str, Any],
    selector_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(PARETO_CONFIGS[tier])
    hard_query = is_hard_query(row)
    question = str(row.get("question") or "").lower()
    query_class = (
        "temporal_sequence"
        if hard_query
        and any(term in question for term in ("sequence", "before", "after", "then"))
        else "global_process"
        if hard_query
        else "generic"
    )
    return {
        "dataset": row.get("dataset"),
        "qid": row.get("qid") or row.get("id"),
        "video_id": row.get("video_id"),
        "duration_s": row.get("duration_s"),
        "question": row.get("question"),
        "choices": row.get("choices"),
        "answer_idx": row.get("answer_idx"),
        "answer_label": row.get("answer_label"),
        "answer": row.get("answer"),
        "chosen_config": config["name"],
        "requested_config": config["name"],
        "scheduler_reason": reason,
        "scheduler_query_class": query_class,
        "scheduler_gpu_state": gpu_state,
        "scheduler_selector_meta": selector_meta or {},
        "probe_fps": config["probe_fps"],
        "probe_topk": config["probe_topk"],
        "window_len_s": config["window_len_s"],
        "vlm_budget": config["vlm_budget"],
        "expand_neighbors": False,
        "preserve_order": hard_query,
        "include_uniform_anchors": hard_query,
        "use_choice_sequence_verifier": False,
        "enable_evidence_fallback": False,
        "evidence_type": "generic",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-tier", choices=TIER_ORDER)
    parser.add_argument(
        "--selector-model",
        help=(
            "Optional model from train_config_selector.py. "
            "When set, learned per-question selection replaces keyword rules."
        ),
    )
    parser.add_argument(
        "--budget2-gate-model",
        help=(
            "Optional binary model from train_budget2_gate_selector.py. "
            "When set, budget2 is used only when the gate predicts it is safe."
        ),
    )
    parser.add_argument(
        "--budget2-gate-threshold",
        type=float,
        default=0.5,
        help="Minimum predicted budget2_ok probability needed to use budget2.",
    )
    parser.add_argument(
        "--budget2-gate-upgrade-config",
        choices=TIER_ORDER,
        default="scan0.0039_k8_budget32",
        help="Config to use when the budget2 gate predicts budget2 is unsafe.",
    )
    parser.add_argument("--low-util-pct", type=float, default=35.0)
    parser.add_argument("--high-util-pct", type=float, default=75.0)
    parser.add_argument("--critical-util-pct", type=float, default=92.0)
    parser.add_argument("--high-mem-pct", type=float, default=88.0)
    parser.add_argument("--long-video-s", type=float, default=1200.0)
    parser.add_argument(
        "--no-hard-query-upgrade",
        action="store_true",
        help="Disable question-driven upgrades and use GPU state only.",
    )
    args = parser.parse_args()
    if args.selector_model and args.budget2_gate_model:
        raise SystemExit(
            "use either --selector-model or --budget2-gate-model, not both"
        )

    dataset = load_jsonl(Path(args.dataset))
    gpu_state = query_gpu_state()
    selector = (
        ResourceAwareSelector.load(args.selector_model)
        if args.selector_model and not args.force_tier
        else None
    )
    budget2_gate = (
        Budget2GateSelector.load(args.budget2_gate_model)
        if args.budget2_gate_model and not args.force_tier
        else None
    )
    base_tier = args.force_tier or base_tier_from_gpu(
        gpu_state,
        low_util_pct=args.low_util_pct,
        high_util_pct=args.high_util_pct,
        critical_util_pct=args.critical_util_pct,
        high_mem_pct=args.high_mem_pct,
    )

    scheduled = []
    counts: dict[str, int] = {}
    for row in dataset:
        selector_meta: dict[str, Any] | None = None
        if budget2_gate is not None:
            prob, selector_meta = budget2_gate.predict_probability(row)
            predicted = (
                "budget2"
                if prob >= args.budget2_gate_threshold
                else args.budget2_gate_upgrade_config
            )
            tier, gpu_reason = apply_gpu_pressure(predicted, gpu_state)
            reason = (
                f"budget2_gate p_ok={prob:.3f} "
                f"threshold={args.budget2_gate_threshold:.3f} "
                f"predicted={predicted};{gpu_reason}"
            )
            dur = duration_s(row)
            if dur is not None and dur >= args.long_video_s:
                capped = min(
                    tier,
                    "scan0.0039_k8_budget32",
                    key=TIER_ORDER.index,
                )
                if capped != tier:
                    reason = (
                        f"{reason};long_video_downgrade "
                        f"{tier}->{capped} duration_s={dur:.1f}"
                    )
                    tier = capped
        elif selector is not None:
            tier, reason, selector_meta = selector.choose_config(
                row,
                gpu_state,
            )
            dur = duration_s(row)
            if dur is not None and dur >= args.long_video_s:
                capped = min(
                    tier,
                    "scan0.0039_k8_budget32",
                    key=TIER_ORDER.index,
                )
                if capped != tier:
                    reason = (
                        f"{reason};long_video_downgrade "
                        f"{tier}->{capped} duration_s={dur:.1f}"
                    )
                    tier = capped
        else:
            tier, reason = choose_tier(
                row,
                base_tier=base_tier,
                allow_hard_query_upgrade=not args.no_hard_query_upgrade,
                long_video_s=args.long_video_s,
            )
        counts[tier] = counts.get(tier, 0) + 1
        scheduled.append(
            schedule_row(
                row,
                tier=tier,
                reason=reason,
                gpu_state=gpu_state,
                selector_meta=selector_meta,
            )
        )

    write_jsonl(scheduled, Path(args.output))
    print(f"wrote: {args.output}")
    print(f"questions: {len(scheduled)}")
    print(
        "gpu_state:",
        json.dumps(
            {
                "available": gpu_state.get("available"),
                "avg_gpu_util_pct": gpu_state.get("avg_gpu_util_pct"),
                "avg_mem_used_pct": gpu_state.get("avg_mem_used_pct"),
            },
            sort_keys=True,
        ),
    )
    print(f"base_tier: {base_tier}")
    print(f"selector_model: {args.selector_model or 'none'}")
    print("selected configs:")
    for name in TIER_ORDER:
        if counts.get(name):
            print(f"  {name}: {counts[name]}")


if __name__ == "__main__":
    main()
