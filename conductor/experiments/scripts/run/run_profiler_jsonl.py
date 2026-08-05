from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from conductor.profiler.llm_profiler import profile_query_llm


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def get_duration_s(row: dict[str, Any]) -> float:
    value = row.get("duration_s")
    if value is not None:
        try:
            duration = float(value)
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            pass

    video = row.get("video") or row.get("video_path")
    if not video:
        raise KeyError("row has no duration_s, video, or video_path")

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"bad video metadata: {video}")
        return float(frames / fps)
    finally:
        cap.release()


def existing_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["qid"])
        for row in load_jsonl(path)
        if row.get("qid") is not None
    }


def flatten_result(
    *,
    source: dict[str, Any],
    duration_s: float,
    result: Any,
) -> dict[str, Any]:
    analysis = result.analysis
    policy = result.execution_policy
    requested = result.requested_config
    chosen = result.chosen_config

    return {
        "dataset": source.get("dataset"),
        "qid": source.get("qid") or source.get("id"),
        "video_id": source.get("video_id"),
        "duration_s": duration_s,
        "question": source.get("question"),
        "choices": source.get("choices"),
        "answer_idx": source.get("answer_idx"),
        "answer_label": source.get("answer_label"),
        "answer": source.get("answer"),
        "reasoning_type": analysis.get("reasoning_type"),
        "answer_type": analysis.get("answer_type"),
        "coverage_requirement": analysis.get("coverage_requirement"),
        "selection_mode": analysis.get("selection_mode"),
        "temporal_requirement": analysis.get("temporal_requirement"),
        "temporal_operation": analysis.get("temporal_operation"),
        "candidate_requirement": analysis.get("candidate_requirement"),
        "context_requirement": analysis.get("context_requirement"),
        "precision_requirement": analysis.get("precision_requirement"),
        "aggregation_type": analysis.get("aggregation_type"),
        "identity_requirement": analysis.get("identity_requirement"),
        "spatial_strategy": analysis.get("spatial_strategy"),
        "required_modalities": analysis.get("required_modalities"),
        "event_density": analysis.get("event_density"),
        "ambiguity": analysis.get("ambiguity"),
        "profile_confidence": analysis.get("profile_confidence"),
        "miss_risk": analysis.get("miss_risk"),
        "evidence_type": analysis.get("evidence_type"),
        "risk_triggers": analysis.get("_risk_triggers", []),
        "answer_sensitivity": analysis.get("answer_sensitivity"),
        "fallback_requirement": analysis.get("fallback_requirement"),
        "requested_config": requested.name,
        "chosen_config": chosen.name,
        "probe_fps": policy.get("probe_fps"),
        "probe_topk": policy.get("probe_topk"),
        "action_topk": policy.get("action_topk"),
        "window_len_s": policy.get("window_len_s"),
        "vlm_budget": policy.get("answer_max_images_total"),
        "expand_neighbors": policy.get("expand_neighbors"),
        "preserve_order": policy.get("preserve_order"),
        "include_uniform_anchors": policy.get("include_uniform_anchors"),
        "execution_policy": policy,
        "analysis": analysis,
        "requested_config_full": asdict(requested),
        "chosen_config_full": asdict(chosen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base_url",
        default="http://127.0.0.1:9000/v1",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    output = Path(args.output)
    done = set() if args.no_resume else existing_qids(output)
    rows = [
        row
        for idx, row in enumerate(dataset)
        if idx % args.num_shards == args.shard_index
    ]

    print(
        f"rows={len(rows)} shard={args.shard_index}/{args.num_shards} "
        f"output={output}",
        flush=True,
    )

    for idx, row in enumerate(rows, 1):
        qid = str(row.get("qid") or row.get("id"))
        if qid in done:
            print(f"[{idx}/{len(rows)}] skip {qid}", flush=True)
            continue

        print(f"[{idx}/{len(rows)}] profile {qid}", flush=True)
        duration_s = get_duration_s(row)
        result = profile_query_llm(
            query=str(row["question"]),
            duration_s=duration_s,
            choices=row.get("choices"),
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            verbose=False,
        )
        append_jsonl(
            flatten_result(
                source=row,
                duration_s=duration_s,
                result=result,
            ),
            output,
        )


if __name__ == "__main__":
    main()
