#!/usr/bin/env python3
"""Evaluate a duration-aware hybrid video-QA serving policy.

The router deliberately uses only cheap pre-inference information: container
duration and a supplied retrieval schedule. It sends short videos to stock
native vLLM, medium videos through the existing retrieval pipeline, and long
videos through the codec-guided coarse-to-fine path. Each child runner writes
its normal artifacts under ``OUTPUT/parts``; this script merges their results
and records the chosen route on every row.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
NATIVE_RUNNER = ROOT / "conductor/experiments/scripts/run/run_native_vllm_video_baseline.py"
RETRIEVAL_RUNNER = ROOT / "conductor/experiments/scripts/run/run_batched_clip_streaming_vlm.py"
CODEC_RUNNER = ROOT / "conductor/experiments/scripts/run/run_codec_guided_vllm_baseline.py"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def known_duration(row: dict[str, Any]) -> float | None:
    for key in ("_measured_duration_s", "duration_s"):
        try:
            value = row.get(key)
            if value is not None and float(value) > 0:
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def probe_duration(row: dict[str, Any], timeout_s: float) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(row["video"]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    value = float(result.stdout.strip())
    if value <= 0:
        raise ValueError(f"non-positive duration {value}")
    return value


def route_for(duration_s: float, short_max_s: float, medium_max_s: float) -> str:
    if duration_s <= short_max_s:
        return "short_native"
    if duration_s <= medium_max_s:
        return "medium_retrieval"
    return "long_codec_refined"


def filter_schedule(
    source: Path,
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    wanted = {row_qid(row) for row in rows}
    found: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(source):
        key = row_qid(row)
        if key in wanted:
            if key in found:
                raise ValueError(f"duplicate qid in schedule: {key}")
            found[key] = row
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"schedule missing {len(missing)} routed qids; first={sorted(missing)[0]}")
    write_jsonl(destination, [found[row_qid(row)] for row in rows])


def run_logged(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def route_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["hybrid_route"])].append(row)
    result: dict[str, Any] = {}
    for route, values in sorted(groups.items()):
        errors = sum(value.get("error") is not None for value in values)
        correct = sum(bool(value.get("correct")) for value in values)
        latencies = [float(value.get("latency_s") or 0.0) for value in values]
        result[route] = {
            "questions": len(values),
            "correct": correct,
            "errors": errors,
            "accuracy_percent": 100 * correct / len(values) if values else 0.0,
            "mean_request_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--medium-schedule", type=Path, required=True)
    parser.add_argument(
        "--dispatch-plan",
        type=Path,
        help=(
            "Optional dispatch_plan.jsonl from build_queue_aware_video_plan.py. "
            "The plan's route validation and priority order are applied before "
            "each existing bounded execution pipeline starts."
        ),
    )
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--short-max-s", type=float, default=300.0)
    parser.add_argument("--medium-max-s", type=float, default=1200.0)
    parser.add_argument("--short-frames", type=int, default=8)
    parser.add_argument("--long-frames", type=int, default=8)
    parser.add_argument("--video-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--video-map", action="append", default=[])
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--duration-probe-timeout-s", type=float, default=30.0)
    parser.add_argument("--missing-duration-route", choices=["short_native", "medium_retrieval", "long_codec_refined"], default="short_native")
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--answer-frame-workers", type=int, default=1)
    parser.add_argument("--decode-ahead-batches", type=int, default=2)
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--clip-image-batch-size", type=int, default=16)
    parser.add_argument("--codec-window-s", type=float, default=8.0)
    parser.add_argument("--codec-refine-candidates", type=int, default=8)
    parser.add_argument("--codec-refine-regions", type=int, default=2)
    parser.add_argument("--codec-refine-radius-s", type=float, default=16.0)
    parser.add_argument("--codec-probe-max-side", type=int, default=224)
    parser.add_argument("--codec-decode-max-side", type=int, default=448)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}; use --overwrite to replace it")
    if not 0 < args.short_max_s < args.medium_max_s:
        parser.error("require 0 < --short-max-s < --medium-max-s")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    rows = load_jsonl(args.dataset)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    dispatch_by_qid: dict[str, dict[str, Any]] = {}
    if args.dispatch_plan:
        for decision in load_jsonl(args.dispatch_plan):
            key = str(decision["qid"])
            if key in dispatch_by_qid:
                raise SystemExit(f"duplicate qid in dispatch plan: {key}")
            dispatch_by_qid[key] = decision
        missing = {row_qid(row) for row in rows} - set(dispatch_by_qid)
        if missing:
            raise SystemExit(
                f"dispatch plan missing {len(missing)} dataset qids; "
                f"first={sorted(missing)[0]}"
            )
    routed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    routing_errors: list[dict[str, Any]] = []
    routing_started = time.perf_counter()
    for row in rows:
        duration_s = known_duration(row)
        duration_source = "dataset"
        if duration_s is None:
            try:
                duration_s = probe_duration(row, args.duration_probe_timeout_s)
                duration_source = "ffprobe"
            except Exception as exc:
                duration_source = f"fallback:{type(exc).__name__}"
                route = args.missing_duration_route
                routing_errors.append({"qid": row_qid(row), "error": repr(exc)})
                enriched = dict(row)
                enriched["_router_duration_s"] = None
                enriched["_router_duration_source"] = duration_source
                routed[route].append(enriched)
                continue
        route = route_for(duration_s, args.short_max_s, args.medium_max_s)
        decision = dispatch_by_qid.get(row_qid(row))
        if decision is not None and str(decision["route"]) != route:
            raise SystemExit(
                f"dispatch route mismatch qid={row_qid(row)} "
                f"plan={decision['route']} router={route}"
            )
        enriched = dict(row)
        enriched["_measured_duration_s"] = duration_s
        enriched["_router_duration_s"] = duration_s
        enriched["_router_duration_source"] = duration_source
        if decision is not None:
            enriched["_dispatch_rank"] = decision["dispatch_rank"]
            enriched["_dispatch_decision_rule"] = decision["decision_rule"]
        routed[route].append(enriched)
    routing_wall_s = time.perf_counter() - routing_started

    args.output.mkdir(parents=True, exist_ok=True)
    for route_rows in routed.values():
        # This is the order in which a child runner fills its bounded decode /
        # prepared queues. Dataset order is retained when no plan is supplied.
        route_rows.sort(key=lambda row: int(row.get("_dispatch_rank", 10**12)))
    parts = args.output / "parts"
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "short_max_s": args.short_max_s,
        "medium_max_s": args.medium_max_s,
        "short_frames": args.short_frames,
        "long_frames": args.long_frames,
        "route_counts": {route: len(values) for route, values in routed.items()},
        "routing_wall_s": routing_wall_s,
        "routing_errors": routing_errors,
        "dispatch_plan": str(args.dispatch_plan.resolve()) if args.dispatch_plan else None,
    }
    (args.output / "routing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    python = sys.executable
    route_outputs: dict[str, Path] = {}
    run_started = time.perf_counter()
    for route, route_rows in routed.items():
        if not route_rows:
            continue
        route_dir = parts / route
        dataset_path = route_dir / "dataset.jsonl"
        write_jsonl(dataset_path, route_rows)
        result_path = route_dir / "results.jsonl"
        route_outputs[route] = result_path
        common = ["--ports", args.ports, "--concurrency", str(args.concurrency), "--max-pixels", str(args.max_pixels), "--max-tokens", str(args.max_tokens)]
        if route == "short_native":
            command = [
                python, str(NATIVE_RUNNER), "--dataset", str(dataset_path),
                "--output", str(result_path), "--summary-output", str(route_dir / "summary.json"),
                "--frame-counts", str(args.short_frames), "--video-root", str(args.video_root),
                "--request-timeout-s", str(args.request_timeout_s), *common,
            ]
            for mapping in args.video_map:
                command.extend(["--video-map", mapping])
        elif route == "medium_retrieval":
            schedule_path = route_dir / "schedule.jsonl"
            filter_schedule(args.medium_schedule, route_rows, schedule_path)
            command = [
                python, str(RETRIEVAL_RUNNER), "--dataset", str(dataset_path),
                "--schedule", str(schedule_path), "--prepared-output", str(route_dir / "prepared.jsonl"),
                "--results-output", str(result_path), "--ports", args.ports,
                "--concurrency", str(args.concurrency), "--decode-workers", str(args.decode_workers),
                "--answer-frame-workers", str(args.answer_frame_workers),
                "--decode-ahead-batches", str(args.decode_ahead_batches),
                "--clip-image-batch-size", str(args.clip_image_batch_size),
                "--decode-timeout-s", "60", "--max-tokens", str(args.max_tokens),
            ]
        else:
            command = [
                python, str(CODEC_RUNNER), "--dataset", str(dataset_path),
                "--output", str(result_path), "--summary-output", str(route_dir / "summary.json"),
                "--frame-counts", str(args.long_frames), "--ports", args.ports,
                "--concurrency", str(args.concurrency), "--max-pixels", str(args.max_pixels),
                "--max-tokens", str(args.max_tokens), "--window-s", str(args.codec_window_s),
                "--decode-max-side", str(args.codec_decode_max_side),
                "--request-timeout-s", str(args.request_timeout_s), "--refine-with-clip",
                "--clip-device", args.clip_device, "--refine-candidates", str(args.codec_refine_candidates),
                "--refine-regions", str(args.codec_refine_regions), "--refine-radius-s", str(args.codec_refine_radius_s),
                "--probe-max-side", str(args.codec_probe_max_side),
            ]
        print(f"[route-start] route={route} questions={len(route_rows)}", flush=True)
        env = os.environ.copy()
        if route == "medium_retrieval":
            env["CLIP_DEVICE"] = args.clip_device
        try:
            run_logged(command, route_dir / "run.log", env=env)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"route {route} failed; see {route_dir / 'run.log'}") from exc
        print(f"[route-finish] route={route}", flush=True)
    execution_wall_s = time.perf_counter() - run_started

    merged: list[dict[str, Any]] = []
    for route, result_path in route_outputs.items():
        for row in load_jsonl(result_path):
            row["hybrid_route"] = route
            decision = dispatch_by_qid.get(row_qid(row))
            if decision is not None:
                row["dispatch_rank"] = decision["dispatch_rank"]
                row["dispatch_decision_rule"] = decision["decision_rule"]
            merged.append(row)
    expected = Counter(row_qid(row) for row in rows)
    received = Counter(row_qid(row) for row in merged)
    if expected != received:
        missing = list((expected - received).elements())
        extra = list((received - expected).elements())
        raise SystemExit(f"hybrid merge mismatch: missing={missing[:3]} extra={extra[:3]}")
    write_jsonl(args.output / "results.jsonl", merged)

    total_wall_s = routing_wall_s + execution_wall_s
    correct = sum(bool(row.get("correct")) for row in merged)
    errors = sum(row.get("error") is not None for row in merged)
    summary = {
        "dataset": str(args.dataset.resolve()),
        "output": str((args.output / "results.jsonl").resolve()),
        "method": "duration_aware_hybrid_video_qa",
        "questions": len(merged),
        "correct": correct,
        "errors": errors,
        "accuracy_percent": 100 * correct / len(merged) if merged else 0.0,
        "routing_wall_s": routing_wall_s,
        "execution_wall_s": execution_wall_s,
        "wall_time_s": total_wall_s,
        "throughput_qps": len(merged) / total_wall_s if total_wall_s else 0.0,
        "routes": route_summary(merged),
        "short_max_s": args.short_max_s,
        "medium_max_s": args.medium_max_s,
        "short_frames": args.short_frames,
        "long_frames": args.long_frames,
        "medium_schedule": str(args.medium_schedule.resolve()),
        "ports": [value for value in args.ports.split(",") if value],
        "concurrency": args.concurrency,
        "clip_device": args.clip_device,
    }
    summary_path = args.summary_output or args.output / "summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
