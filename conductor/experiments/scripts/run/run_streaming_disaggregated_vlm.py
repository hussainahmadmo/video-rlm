#!/usr/bin/env python
import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BUILD_PREPARED_PATH = (
    ROOT / "conductor/experiments/scripts/build/build_prepared_vlm_jobs.py"
)
RUN_PREPARED_PATH = (
    ROOT / "conductor/experiments/scripts/run/run_prepared_vlm_jobs.py"
)
_WORKER_PREP = None
_WORKER_RUNNER = None


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def completed_keys(path):
    keys = set()
    if not Path(path).exists():
        return keys
    for row in load_jsonl(path):
        keys.add((qid(row), row.get("config_name")))
    return keys


def prepare_job_worker(item, schedule_row):
    global _WORKER_PREP
    global _WORKER_RUNNER

    if _WORKER_PREP is None:
        _WORKER_PREP = import_from_path(
            "build_prepared_vlm_jobs_worker",
            BUILD_PREPARED_PATH,
        )
        _WORKER_RUNNER = _WORKER_PREP.import_runner()

    config = _WORKER_PREP.build_config_from_schedule(schedule_row)
    config = _WORKER_PREP.apply_existing_policy_rewrites(
        _WORKER_RUNNER,
        item,
        config,
    )

    try:
        job = _WORKER_PREP.prepare_clip_oneshot_job(
            _WORKER_RUNNER,
            item,
            config,
        )
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
            "latency_breakdown": dict(_WORKER_RUNNER.LATENCY_STATS),
        }
    return job


def flush_completed(futures, output_path, *, block=False):
    if not futures:
        return 0

    if block:
        done = set(futures)
    else:
        done, _ = wait(
            list(futures),
            timeout=0,
            return_when=FIRST_COMPLETED,
        )

    count = 0
    for future in list(done):
        job = futures.pop(future)
        try:
            row = future.result()
        except Exception as exc:
            row = {
                "qid": job.get("qid"),
                "video_id": job.get("video_id"),
                "video": job.get("video"),
                "dataset": job.get("dataset"),
                "question": job.get("question"),
                "choices": job.get("choices"),
                "answer_idx": job.get("answer_idx"),
                "answer_label": job.get("answer_label"),
                "answer": job.get("answer"),
                "config_name": job.get("config_name"),
                "method": job.get("method"),
                "prediction_label": None,
                "prediction_text": None,
                "prediction_confidence": None,
                "correct": False,
                "num_vlm_calls": 0,
                "latency_s": job.get("prep_latency_s"),
                "prep_latency_s": job.get("prep_latency_s"),
                "latency_breakdown": job.get("latency_breakdown"),
                "stage_latency_s": job.get("stage_latency_s"),
                "error": repr(exc),
            }
        append_jsonl(output_path, row)
        count += 1
        print(
            f"[done] qid={row.get('qid')} "
            f"config={row.get('config_name')} "
            f"pred={row.get('prediction_label')} "
            f"gold={row.get('answer_label')} "
            f"correct={row.get('correct')} "
            f"latency={float(row.get('latency_s') or 0.0):.2f}s",
            flush=True,
        )
    return count


def finish_send_future(future, job, output_path):
    try:
        row = future.result()
    except Exception as exc:
        row = {
            "qid": job.get("qid"),
            "video_id": job.get("video_id"),
            "video": job.get("video"),
            "dataset": job.get("dataset"),
            "question": job.get("question"),
            "choices": job.get("choices"),
            "answer_idx": job.get("answer_idx"),
            "answer_label": job.get("answer_label"),
            "answer": job.get("answer"),
            "config_name": job.get("config_name"),
            "method": job.get("method"),
            "prediction_label": None,
            "prediction_text": None,
            "prediction_confidence": None,
            "correct": False,
            "num_vlm_calls": 0,
            "latency_s": job.get("prep_latency_s"),
            "prep_latency_s": job.get("prep_latency_s"),
            "latency_breakdown": job.get("latency_breakdown"),
            "stage_latency_s": job.get("stage_latency_s"),
            "error": repr(exc),
        }

    append_jsonl(output_path, row)
    print(
        f"[done] qid={row.get('qid')} "
        f"config={row.get('config_name')} "
        f"pred={row.get('prediction_label')} "
        f"gold={row.get('answer_label')} "
        f"correct={row.get('correct')} "
        f"latency={float(row.get('latency_s') or 0.0):.2f}s",
        flush=True,
    )
    return row


def port_from_base_url(base_url):
    return base_url.rsplit(":", 1)[-1].split("/", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--prepared-output", required=True)
    parser.add_argument("--results-output", required=True)
    parser.add_argument("--ports", default="9000")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--queue-depth", type=int, default=18)
    parser.add_argument("--prep-workers", type=int, default=1)
    parser.add_argument("--prep-queue-depth", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    send = import_from_path("run_prepared_vlm_jobs", RUN_PREPARED_PATH)
    prep = import_from_path("build_prepared_vlm_jobs", BUILD_PREPARED_PATH)
    runner = None
    if args.prep_workers <= 1:
        runner = prep.import_runner()

    examples = load_jsonl(args.dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    schedule_rows = load_jsonl(args.schedule)
    schedule_by_qid = {qid(row): row for row in schedule_rows}

    ports = [port.strip() for port in args.ports.split(",") if port.strip()]
    base_urls = [f"http://localhost:{port}/v1" for port in ports]
    concurrency = args.concurrency or max(1, len(base_urls) * 2)
    queue_depth = max(concurrency, args.queue_depth)
    prep_queue_depth = args.prep_queue_depth or max(1, args.prep_workers * 2)

    result_done = set()
    prepared_done = set()
    if not args.no_resume:
        result_done = completed_keys(args.results_output)
        prepared_done = completed_keys(args.prepared_output)

    print(
        f"examples={len(examples)} ports={','.join(ports)} "
        f"concurrency={concurrency} queue_depth={queue_depth} "
        f"prep_workers={args.prep_workers} "
        f"prep_queue_depth={prep_queue_depth}",
        flush=True,
    )

    if args.prep_workers > 1:
        run_multi_prep_stream(
            args=args,
            prep=prep,
            send=send,
            examples=examples,
            schedule_by_qid=schedule_by_qid,
            base_urls=base_urls,
            queue_depth=queue_depth,
            prep_queue_depth=prep_queue_depth,
            result_done=result_done,
            prepared_done=prepared_done,
        )
        return

    total_started = 0
    total_completed = 0
    started = time.time()
    futures = {}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for idx, item in enumerate(examples, start=1):
            item_qid = qid(item)
            if item_qid not in schedule_by_qid:
                print(f"[skip] missing schedule qid={item_qid}", flush=True)
                continue

            config = prep.build_config_from_schedule(schedule_by_qid[item_qid])
            config = prep.apply_existing_policy_rewrites(runner, item, config)
            key = (item_qid, config["name"])
            if key in result_done:
                print(f"[{idx}/{len(examples)}] skip done qid={item_qid}")
                continue

            while len(futures) >= queue_depth:
                total_completed += flush_completed(
                    futures,
                    args.results_output,
                    block=False,
                )
                if len(futures) >= queue_depth:
                    total_completed += flush_completed(
                        futures,
                        args.results_output,
                        block=True,
                    )

            print(
                f"[{idx}/{len(examples)}] prepare qid={item_qid} "
                f"config={config['name']}",
                flush=True,
            )
            try:
                job = prep.prepare_clip_oneshot_job(runner, item, config)
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

            if key not in prepared_done:
                append_jsonl(args.prepared_output, job)
                prepared_done.add(key)

            base_url = base_urls[total_started % len(base_urls)]
            future = pool.submit(
                send.run_one,
                job,
                base_url,
                args.max_tokens,
            )
            futures[future] = job
            total_started += 1
            print(
                f"[send] qid={item_qid} port={port_from_base_url(base_url)} "
                f"inflight={len(futures)}",
                flush=True,
            )

            total_completed += flush_completed(
                futures,
                args.results_output,
                block=False,
            )

        while futures:
            total_completed += flush_completed(
                futures,
                args.results_output,
                block=True,
            )

    print(
        f"prepared={len(prepared_done)} sent={total_started} "
        f"completed={total_completed} elapsed={time.time() - started:.2f}s",
        flush=True,
    )
    print(f"prepared_output={args.prepared_output}")
    print(f"results_output={args.results_output}")


def run_multi_prep_stream(
    *,
    args,
    prep,
    send,
    examples,
    schedule_by_qid,
    base_urls,
    queue_depth,
    prep_queue_depth,
    result_done,
    prepared_done,
):
    pending = []
    for idx, item in enumerate(examples, start=1):
        item_qid = qid(item)
        schedule_row = schedule_by_qid.get(item_qid)
        if schedule_row is None:
            print(f"[skip] missing schedule qid={item_qid}", flush=True)
            continue

        config_name = schedule_row.get(
            "chosen_config",
            schedule_row.get("selected_config", "vimio"),
        )
        if (item_qid, config_name) in result_done:
            print(f"[{idx}/{len(examples)}] skip done qid={item_qid}")
            continue
        pending.append((idx, item, schedule_row))

    total_started = 0
    total_completed = 0
    total_prepared = 0
    started = time.time()
    prep_futures = {}
    send_futures = {}
    next_idx = 0

    def submit_prep(pool):
        nonlocal next_idx
        while (
            next_idx < len(pending)
            and len(prep_futures) < prep_queue_depth
        ):
            idx, item, schedule_row = pending[next_idx]
            next_idx += 1
            print(
                f"[{idx}/{len(examples)}] enqueue-prep qid={qid(item)}",
                flush=True,
            )
            future = pool.submit(
                prepare_job_worker,
                item,
                schedule_row,
            )
            prep_futures[future] = (idx, item, schedule_row)

    with ProcessPoolExecutor(max_workers=args.prep_workers) as prep_pool:
        with ThreadPoolExecutor(max_workers=args.concurrency) as send_pool:
            submit_prep(prep_pool)

            while prep_futures or send_futures:
                all_futures = list(prep_futures) + list(send_futures)
                done, _ = wait(
                    all_futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in list(done):
                    if future in send_futures:
                        job = send_futures.pop(future)
                        finish_send_future(
                            future,
                            job,
                            args.results_output,
                        )
                        total_completed += 1
                        continue

                    if future not in prep_futures:
                        continue

                    idx, item, _ = prep_futures.pop(future)
                    try:
                        job = future.result()
                    except Exception as exc:
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
                            "config_name": None,
                            "method": None,
                            "prepare_error": repr(exc),
                        }

                    key = (qid(job), job.get("config_name"))
                    if key not in prepared_done:
                        append_jsonl(args.prepared_output, job)
                        prepared_done.add(key)
                    total_prepared += 1

                    while len(send_futures) >= queue_depth:
                        send_done, _ = wait(
                            list(send_futures),
                            return_when=FIRST_COMPLETED,
                        )
                        for send_future in send_done:
                            send_job = send_futures.pop(send_future)
                            finish_send_future(
                                send_future,
                                send_job,
                                args.results_output,
                            )
                            total_completed += 1

                    base_url = base_urls[total_started % len(base_urls)]
                    send_future = send_pool.submit(
                        send.run_one,
                        job,
                        base_url,
                        args.max_tokens,
                    )
                    send_futures[send_future] = job
                    total_started += 1
                    print(
                        f"[send] qid={qid(job)} "
                        f"port={port_from_base_url(base_url)} "
                        f"inflight={len(send_futures)} "
                        f"prep_pending={len(prep_futures)}",
                        flush=True,
                    )

                submit_prep(prep_pool)

    print(
        f"prepared={total_prepared} sent={total_started} "
        f"completed={total_completed} elapsed={time.time() - started:.2f}s",
        flush=True,
    )
    print(f"prepared_output={args.prepared_output}")
    print(f"results_output={args.results_output}")


if __name__ == "__main__":
    main()
