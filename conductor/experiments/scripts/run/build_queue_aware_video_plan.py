#!/usr/bin/env python3
"""Build an auditable, bounded-queue dispatch plan for video QA.

This is deliberately a controller, not another decoder.  It chooses the
*next* request to prepare using inexpensive metadata (duration, question
shape, and a supplied medium-video schedule), then writes route-specific
JSONL inputs that the existing native, retrieval, and codec runners consume.

The plan is useful in two ways:
  1. it makes routing/priority choices reproducible and inspectable; and
  2. it supplies the input order for the existing bounded preparation queue.

It does not pretend to be a GPU runtime scheduler: port placement and true
live queue feedback belong in the serving loop.  Every decision records the
queue state assumed by this first rule-based policy, so that can be replaced
by real telemetry later without changing the experiment format.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def qid(row: dict[str, Any]) -> str:
    value = row.get("qid") or row.get("question_id") or row.get("id")
    if value is None:
        raise ValueError("row has no qid/question_id/id")
    return str(value)


def duration(row: dict[str, Any]) -> float | None:
    for key in ("_measured_duration_s", "duration_s"):
        try:
            value = float(row[key])
            if value > 0:
                return value
        except (KeyError, TypeError, ValueError):
            continue
    return None


def ffprobe_duration(video: str, timeout_s: float) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video],
        check=True, capture_output=True, text=True, timeout=timeout_s,
    )
    value = float(result.stdout.strip())
    if value <= 0:
        raise ValueError(f"non-positive duration {value}")
    return value


def question_is_temporal(row: dict[str, Any]) -> bool:
    text = str(row.get("question", "")).lower()
    words = ("after", "before", "then", "first", "last", "sequence", "during", "when")
    return any(word in text for word in words)


def choose_route(
    duration_s: float,
    *,
    short_max_s: float,
    medium_max_s: float,
) -> tuple[str, str, str]:
    if duration_s <= short_max_s:
        return "short_native", "native_uniform_8", "short videos have negligible preparation cost"
    if duration_s <= medium_max_s:
        return "medium_retrieval", "schedule_selected", "medium videos use query-conditioned retrieval"
    return "long_codec_refined", "codec_refined_8", "long videos avoid full-video pixel decode"


def preparation_cost(route: str, duration_s: float, temporal: bool) -> float:
    """Relative CPU-preparation cost, used only to order pending work."""
    if route == "short_native":
        return 0.25
    if route == "medium_retrieval":
        return 1.0 + min(duration_s / 1200.0, 1.0)
    # Long codec jobs still need packet indexing/seeks; temporal questions get
    # a small boost because they benefit most from early refinement.
    return 2.0 + min(duration_s / 3600.0, 2.0) + (0.25 if temporal else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-max-s", type=float, default=300.0)
    parser.add_argument("--medium-max-s", type=float, default=1200.0)
    parser.add_argument("--duration-probe-timeout-s", type=float, default=30.0)
    parser.add_argument("--prepared-queue-depth", type=int, default=18)
    parser.add_argument("--cpu-slots", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}; use --overwrite")
    if not 0 < args.short_max_s < args.medium_max_s:
        parser.error("require 0 < short-max-s < medium-max-s")
    if args.cpu_slots < 1 or args.prepared_queue_depth < 1:
        parser.error("cpu-slots and prepared-queue-depth must be positive")

    pending: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in load_jsonl(args.dataset):
        value = duration(row)
        source = "dataset"
        if value is None:
            try:
                value = ffprobe_duration(str(row["video"]), args.duration_probe_timeout_s)
                source = "ffprobe"
            except Exception as exc:
                errors.append({"qid": qid(row), "error": repr(exc)})
                # A failure should not silently create an unbounded long job.
                value, source = 0.0, f"fallback:{type(exc).__name__}"
        route, config, reason = choose_route(
            value, short_max_s=args.short_max_s, medium_max_s=args.medium_max_s
        )
        temporal = question_is_temporal(row)
        pending.append({
            "row": row, "qid": qid(row), "duration_s": value,
            "duration_source": source, "route": route, "config": config,
            "reason": reason, "temporal": temporal,
            "cost": preparation_cost(route, value, temporal),
        })

    # Preparation-aware ordering: when there is no prepared work, start the
    # longest expensive job first so its CPU work overlaps later VLM service.
    # Once the bounded queue is populated, cheaper requests prevent head-of-
    # line blocking. This is deterministic and records its assumed state.
    dispatched: list[dict[str, Any]] = []
    prepared_assumed = 0
    while pending:
        if prepared_assumed == 0:
            chosen = max(pending, key=lambda item: (item["cost"], item["duration_s"], item["qid"]))
            rule = "hide_long_preparation_behind_future_vlm_service"
        else:
            chosen = min(pending, key=lambda item: (item["cost"], item["duration_s"], item["qid"]))
            rule = "avoid_head_of_line_blocking_in_bounded_prepare_queue"
        pending.remove(chosen)
        decision = {
            "qid": chosen["qid"], "dispatch_rank": len(dispatched),
            "route": chosen["route"], "config": chosen["config"],
            "duration_s": chosen["duration_s"], "duration_source": chosen["duration_source"],
            "question_is_temporal": chosen["temporal"],
            "assumed_cpu_slots": args.cpu_slots,
            "assumed_prepared_queue_depth": args.prepared_queue_depth,
            "assumed_prepared_queue_before": prepared_assumed,
            "decision_rule": rule,
            "route_reason": chosen["reason"],
            "estimated_relative_prepare_cost": chosen["cost"],
        }
        dispatched.append({**chosen, "decision": decision})
        prepared_assumed = min(args.prepared_queue_depth, prepared_assumed + 1)
        # This is a planning approximation: each dispatch reserves a CPU slot;
        # the downstream runner enforces the actual worker bound.
        if prepared_assumed >= args.cpu_slots:
            prepared_assumed -= 1

    args.output.mkdir(parents=True, exist_ok=True)
    route_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    for item in dispatched:
        row = dict(item["row"])
        row["_dispatch_rank"] = item["decision"]["dispatch_rank"]
        row["_dispatch_route"] = item["route"]
        route_rows[item["route"]].append(row)
        decisions.append(item["decision"])
    for route, rows in route_rows.items():
        write_jsonl(args.output / f"{route}.dataset.jsonl", rows)
    write_jsonl(args.output / "dispatch_plan.jsonl", decisions)
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "questions": len(dispatched),
        "route_counts": dict(Counter(item["route"] for item in dispatched)),
        "short_max_s": args.short_max_s,
        "medium_max_s": args.medium_max_s,
        "cpu_slots": args.cpu_slots,
        "prepared_queue_depth": args.prepared_queue_depth,
        "duration_probe_errors": errors,
        "implementation_note": (
            "This is a deterministic preparation-priority controller. The "
            "downstream execution runners enforce real decode/VLM queues."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
