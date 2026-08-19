#!/usr/bin/env python
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = ROOT / "conductor/experiments/scripts/run/run_synthesis_grid_ultra.py"


def import_runner():
    spec = importlib.util.spec_from_file_location(
        "run_synthesis_grid_ultra",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def qid(row):
    return str(row.get("qid") or row.get("question_id") or row.get("id"))


def build_config_from_schedule(schedule_row):
    return {
        "name": schedule_row.get(
            "chosen_config",
            schedule_row.get("selected_config", "vimio"),
        ),
        "method": "clip_oneshot",
        "scan_fps": schedule_row["probe_fps"],
        "clip_topk": schedule_row["probe_topk"],
        "window_len_s": schedule_row["window_len_s"],
        "vlm_budget": schedule_row["vlm_budget"],
        "evidence_type": schedule_row.get("evidence_type"),
        "expand_neighbors": bool(schedule_row.get("expand_neighbors")),
        "preserve_order": bool(schedule_row.get("preserve_order")),
        "include_uniform_anchors": bool(schedule_row.get("include_uniform_anchors")),
        "use_choice_sequence_verifier": bool(
            schedule_row.get("use_choice_sequence_verifier")
        ),
        "scheduler_reason": schedule_row.get("scheduler_reason"),
        "scheduler_query_class": schedule_row.get("scheduler_query_class"),
        "scheduler_gpu_state": schedule_row.get("scheduler_gpu_state"),
        "answer_with_confidence": True,
        "enable_evidence_fallback": bool(
            schedule_row.get("enable_evidence_fallback", True)
        ),
    }


def apply_existing_policy_rewrites(runner, item, config):
    config = dict(config)

    if config["evidence_type"] == "sequence_ordering":
        if config.get("use_choice_sequence_verifier"):
            config["method"] = "choice_sequence_verifier"
        else:
            config = runner.apply_sequence_oracle_clip_policy(config, item)
        config["map_max_windows"] = 12
        config["map_frames_per_window"] = 3
        config["target_choice_events"] = True
        config["sequence_event_topk"] = config["clip_topk"]
        config["timestamp_frames_per_window"] = 3
        config["sequence_choice_topk"] = min(config["clip_topk"], 6)
        config["sequence_choice_max_events"] = 5
        config["sequence_choice_max_windows"] = 6
        config["sequence_choice_frames_per_window"] = 2

    elif config["evidence_type"] == "screen_state_change":
        config["method"] = "clip_map_answer"
        config["map_max_windows"] = 10
        config["map_frames_per_window"] = 4

    elif config["evidence_type"] == "global_process":
        config["method"] = "map_summary"
        config["num_windows"] = 8
        config["frames_per_window"] = 3
        config["query_conditioned"] = False
        config["choice_compare_answer"] = True
        config["structured_evidence_summary"] = True

    config = runner.apply_oracle_edge_policy(config, item)
    config = runner.clamp_vimio_config_cost_aware(config, item)
    return config


def prepare_clip_oneshot_job(runner, item, config):
    if config.get("method") != "clip_oneshot":
        raise NotImplementedError(
            "prepared VLM jobs currently support only clip_oneshot configs; "
            f"got method={config.get('method')}"
        )

    runner.LATENCY_STATS.clear()
    start = time.time()

    query = runner.build_clip_query(item, config=config)
    retrieval = runner.clip_topk_windows(
        video_path=item["video"],
        query=query,
        k=config["clip_topk"],
        window_len_s=config["window_len_s"],
        scan_fps=config["scan_fps"],
    )

    top_windows = retrieval["top_windows"]
    selected_windows = runner.apply_profiler_window_hints(
        item=item,
        top_windows=top_windows,
        config=config,
    )
    if not selected_windows:
        raise RuntimeError("No windows selected")

    total_frames = int(config["vlm_budget"])
    base = total_frames // len(selected_windows)
    extra = total_frames % len(selected_windows)
    frame_allocations = [
        base + (1 if idx < extra else 0)
        for idx in range(len(selected_windows))
    ]

    frames = runner.sample_frames_from_windows(
        video_path=item["video"],
        windows=selected_windows,
        frame_allocations=frame_allocations,
    )

    if len(frames["images"]) > 64:
        raise RuntimeError(f"Too many images: {len(frames['images'])}")

    evidence_hint = runner.build_profiler_evidence_context(
        selected_windows=selected_windows,
        top_windows=top_windows,
        config=config,
    )

    if config.get("answer_with_confidence"):
        prompt = runner.build_answer_prompt_with_confidence(
            item["question"],
            item["choices"],
            extra_context=evidence_hint,
        )
    else:
        prompt = runner.build_answer_prompt(
            item["question"],
            item["choices"],
            extra_context=evidence_hint,
        )

    prep_latency_s = time.time() - start
    stage_latency_s = runner.build_stage_latency_summary(prep_latency_s)
    stage_latency_s["vlm_generation_s"] = 0.0
    stage_latency_s["total_s"] = prep_latency_s
    stage_latency_s["other_s"] = max(
        0.0,
        prep_latency_s
        - stage_latency_s.get("clip_retrieval_s", 0.0)
        - stage_latency_s.get("answer_frame_pack_s", 0.0),
    )

    return {
        "qid": item["qid"],
        "video_id": item.get("video_id"),
        "video": item.get("video"),
        "dataset": item.get("dataset"),
        "question": item["question"],
        "choices": item["choices"],
        "answer_idx": item.get("answer_idx"),
        "answer_label": item.get("answer_label"),
        "answer": item.get("answer"),
        "question_category": item.get("question_category"),
        "topic_category": item.get("topic_category"),
        "vimio_profile": item.get("vimio_profile"),
        "duration_s": item.get("duration_s"),
        "duration_bucket": item.get("duration_bucket"),
        "lvb_duration_bucket": item.get("lvb_duration_bucket"),
        "config_name": config["name"],
        "method": config["method"],
        "scan_fps": config.get("scan_fps"),
        "clip_topk": config.get("clip_topk"),
        "window_len_s": config.get("window_len_s"),
        "vlm_budget": config.get("vlm_budget"),
        "evidence_type": config.get("evidence_type"),
        "expand_neighbors": config.get("expand_neighbors"),
        "preserve_order": config.get("preserve_order"),
        "include_uniform_anchors": config.get("include_uniform_anchors"),
        "answer_with_confidence": config.get("answer_with_confidence"),
        "scheduler_reason": config.get("scheduler_reason"),
        "scheduler_query_class": config.get("scheduler_query_class"),
        "scheduler_gpu_state": config.get("scheduler_gpu_state"),
        "prompt": prompt,
        "frames": frames,
        "prep_latency_s": prep_latency_s,
        "latency_breakdown": dict(runner.LATENCY_STATS),
        "stage_latency_s": stage_latency_s,
        "decord_ctx": runner.DECORD_CTX,
        "clip_device": runner.CLIP_DEVICE,
        "retrieval_effort": {
            "candidate_windows": retrieval["candidate_windows_examined"],
            "selected_windows": len(selected_windows),
            "selected_frames": len(frames["frame_indices"]),
        },
        "evidence": {
            "clip_top_windows": top_windows,
            "selected_windows": selected_windows,
            "selected_frame_indices": frames["frame_indices"],
            "selected_timestamps": frames["timestamps"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    runner = import_runner()
    rows = load_jsonl(args.dataset)
    schedule_rows = load_jsonl(args.schedule)
    schedule_by_qid = {qid(row): row for row in schedule_rows}

    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    output = Path(args.output)
    done = set()
    if output.exists() and not args.no_resume:
        for row in load_jsonl(output):
            done.add((qid(row), row.get("config_name")))

    total = len(rows)
    for idx, item in enumerate(rows, start=1):
        if qid(item) not in schedule_by_qid:
            print(f"[skip] missing schedule qid={qid(item)}", flush=True)
            continue

        config = build_config_from_schedule(schedule_by_qid[qid(item)])
        config = apply_existing_policy_rewrites(runner, item, config)
        key = (qid(item), config["name"])
        if key in done:
            print(f"[{idx}/{total}] skip qid={qid(item)} config={config['name']}")
            continue

        print(f"[{idx}/{total}] prepare qid={qid(item)} config={config['name']}")
        try:
            job = prepare_clip_oneshot_job(runner, item, config)
            job["prepare_error"] = None
        except Exception as exc:
            import traceback

            traceback.print_exc()
            job = {
                "qid": item.get("qid"),
                "video_id": item.get("video_id"),
                "video": item.get("video"),
                "dataset": item.get("dataset"),
                "question": item.get("question"),
                "choices": item.get("choices"),
                "answer_idx": item.get("answer_idx"),
                "answer_label": item.get("answer_label"),
                "answer": item.get("answer"),
                "config_name": config.get("name"),
                "method": config.get("method"),
                "prepare_error": repr(exc),
                "latency_breakdown": dict(runner.LATENCY_STATS),
            }
        append_jsonl(output, job)

    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
