#!/usr/bin/env python3
"""Measure vLLM priority with raw videos prepared entirely inside vLLM.

The client sends every request as ``video_url`` and performs no frame decode,
sampling, resize, or preparation.  All arrival tasks are created immediately,
so an urgent request is not hidden behind a client-side worker pool.  The only
ordering signal is the OpenAI-compatible request's ``priority`` field.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from openai import AsyncOpenAI

from run_native_vllm_video_baseline import make_video_url, parse_label, prompt, qid


FRAME_BUDGET_QUERY_KEY = "vllm_num_frames"


def add_frame_budget(url: str, frame_budget: int) -> str:
    """Carry a per-request frame budget to the opt-in vLLM wrapper."""
    parts = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key != FRAME_BUDGET_QUERY_KEY
    ]
    query.append((FRAME_BUDGET_QUERY_KEY, str(frame_budget)))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def duration_s(row: dict[str, Any]) -> float | None:
    """Read existing metadata only; never inspect or decode the video client-side."""
    for field in ("_measured_duration_s", "duration_s", "video_duration_s", "duration"):
        if row.get(field) is not None:
            return float(row[field])
    return None


def workload(row: dict[str, Any]) -> str:
    return str(row.get("workload") or row.get("class") or "background")


def backend_key() -> str:
    """Support both current and legacy vLLM media-IO keyword spellings."""
    try:
        from vllm.multimodal.media.video import VideoMediaIO

        source = inspect.getsource(VideoMediaIO.__init__)
    except Exception:
        return "backend"
    return "video_backend" if 'kwargs.pop("video_backend"' in source else "backend"


def check_server(port: int) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM on port {port} is unhealthy")


async def append_jsonl(path: Path, row: dict[str, Any], lock: asyncio.Lock) -> None:
    line = json.dumps(row) + "\n"
    async with lock:
        with path.open("a") as handle:
            handle.write(line)
            handle.flush()


async def run_request(
    row: dict[str, Any],
    *,
    client: AsyncOpenAI,
    args: argparse.Namespace,
    experiment_start: float,
    results_path: Path,
    write_lock: asyncio.Lock,
    video_mappings: list[tuple[Path, str]],
    media_backend_key: str,
) -> dict[str, Any]:
    arrival_s = float(row.get("arrival_s", 0.0))
    await asyncio.sleep(max(0.0, experiment_start + arrival_s - time.perf_counter()))

    request_start = time.perf_counter()
    request_priority = (
        int(row.get("priority", 0)) if args.priority_mode == "trace" else 0
    )
    video_url = make_video_url(row, video_mappings)

    video_kwargs: dict[str, Any] = {}
    frames_value = row.get("frame_count")
    frames = int(frames_value) if frames_value is not None else None
    duration = duration_s(row)

    # vLLM 0.17's OpenAI schema ignores per-request media_io_kwargs. The
    # opt-in server wrapper consumes this URL metadata before fetching.
    if frames is not None:
        video_url = add_frame_budget(video_url, frames)
    if args.video_backend:
        video_kwargs[media_backend_key] = args.video_backend
    video_kwargs.update(args.video_backend_kwargs)

    extra_body: dict[str, Any] = {"priority": request_priority}
    if video_kwargs:
        extra_body["media_io_kwargs"] = {"video": video_kwargs}
    if args.max_pixels is not None:
        extra_body["mm_processor_kwargs"] = {"max_pixels": args.max_pixels}

    first_token_s: float | None = None
    pieces: list[str] = []
    error: str | None = None
    try:
        stream = await client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_url}},
                        {"type": "text", "text": prompt(row)},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=args.max_tokens,
            stream=True,
            extra_body=extra_body,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                if first_token_s is None:
                    first_token_s = time.perf_counter()
                pieces.append(content)
    except Exception as exc:
        error = repr(exc)

    completed_s = time.perf_counter()
    prediction_text = "".join(pieces) if pieces else None
    choices = row.get("choices") or []
    prediction = parse_label(prediction_text, len(choices))
    gold = row.get("answer_label")
    if gold is None and row.get("answer_idx") is not None:
        gold = chr(ord("A") + int(row["answer_idx"]))

    result = {
        "request_id": str(row.get("request_id") or qid(row)),
        "qid": qid(row),
        "workload": workload(row),
        "trace_priority": int(row.get("priority", 0)),
        "submitted_priority": request_priority,
        "priority_mode": args.priority_mode,
        "arrival_s": arrival_s,
        "submit_s": request_start - experiment_start,
        "frame_count": frames,
        "duration_s": duration,
        "video_url": video_url,
        "ttft_s": (
            first_token_s - (experiment_start + arrival_s)
            if first_token_s is not None
            else None
        ),
        "request_to_first_token_s": (
            first_token_s - request_start if first_token_s is not None else None
        ),
        "end_to_end_s": completed_s - (experiment_start + arrival_s),
        "prediction_label": prediction,
        "prediction_text": prediction_text,
        "correct": prediction == gold if gold is not None else None,
        "error": error,
    }
    await append_jsonl(results_path, result, write_lock)
    status = "error" if error else "done"
    print(
        f"[{status}] {result['request_id']} workload={result['workload']} "
        f"priority={request_priority} ttft={result['ttft_s']}",
        flush=True,
    )
    return result


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["error"] is None and row["ttft_s"] is not None]
    ttfts = [float(row["ttft_s"]) for row in successful]
    e2e = [float(row["end_to_end_s"]) for row in successful]
    scored = [row for row in successful if row["correct"] is not None]
    return {
        "requests": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "mean_ttft_s": mean(ttfts),
        "p50_ttft_s": percentile(ttfts, 0.50),
        "p95_ttft_s": percentile(ttfts, 0.95),
        "mean_end_to_end_s": mean(e2e),
        "accuracy_percent": (
            100.0 * sum(bool(row["correct"]) for row in scored) / len(scored)
            if scored
            else None
        ),
    }


async def async_main(args: argparse.Namespace) -> None:
    check_server(args.port)
    rows = load_jsonl(args.arrival_trace)
    if not rows:
        raise SystemExit("arrival trace is empty")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    mappings: list[tuple[Path, str]] = []
    for value in args.video_map:
        if "=" not in value:
            raise SystemExit("--video-map requires LOCAL_ROOT=HTTP_BASE")
        local_root, http_base = value.split("=", 1)
        mappings.append((Path(local_root).resolve(), http_base))
    if not mappings:
        mappings = [(args.video_root.resolve(), args.video_base_url)]
    mappings.sort(key=lambda item: len(str(item[0])), reverse=True)

    # Validate the complete trace before creating output or issuing a request.
    # This catches mixed datasets whose files need more than one HTTP mapping.
    mapping_errors = []
    for row in rows:
        try:
            make_video_url(row, mappings)
        except Exception as exc:
            mapping_errors.append(f"{row.get('request_id', qid(row))}: {exc}")
    if mapping_errors:
        preview = "\n".join(mapping_errors[:10])
        raise SystemExit(
            f"{len(mapping_errors)} videos have no HTTP mapping; first errors:\n{preview}"
        )

    args.output.mkdir(parents=True)
    results_path = args.output / "results.jsonl"
    limits = httpx.Limits(
        max_connections=max(args.http_connections, len(rows)),
        max_keepalive_connections=args.http_connections,
    )
    # Never route localhost vLLM traffic through cluster HTTP proxy settings.
    http_client = httpx.AsyncClient(limits=limits, trust_env=False)
    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{args.port}/v1",
        api_key="EMPTY",
        timeout=args.request_timeout_s,
        max_retries=0,
        http_client=http_client,
    )
    lock = asyncio.Lock()
    started = time.perf_counter()
    try:
        results = await asyncio.gather(
            *[
                run_request(
                    row,
                    client=client,
                    args=args,
                    experiment_start=started,
                    results_path=results_path,
                    write_lock=lock,
                    video_mappings=mappings,
                    media_backend_key=backend_key(),
                )
                for row in rows
            ]
        )
    finally:
        await client.close()

    wall_time_s = time.perf_counter() - started
    summary: dict[str, Any] = {
        "method": "native_vllm_raw_video_priority_burst",
        "priority_mode": args.priority_mode,
        "server_requirement": "--scheduling-policy priority",
        "port": args.port,
        "total_requests": len(results),
        "errors": sum(row["error"] is not None for row in results),
        "wall_time_s": wall_time_s,
        "throughput_qps": len(results) / wall_time_s,
        "video_processing": "native_vllm_video_url",
        "client_preparation_workers": 0,
    }
    for name in ("urgent", "background"):
        summary[name] = summarize_group(
            [row for row in results if row["workload"] == name]
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--priority-mode",
        choices=["uniform", "trace"],
        required=True,
        help="uniform submits priority 0 for every request; trace uses each row's priority",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--http-connections", type=int, default=128)
    parser.add_argument("--video-root", type=Path, default=Path("/dataheart/hussainahmad/datasets"))
    parser.add_argument("--video-base-url", default="http://127.0.0.1:8088")
    parser.add_argument(
        "--video-map",
        action="append",
        default=[],
        help="LOCAL_ROOT=HTTP_BASE; repeat for traces spanning multiple roots",
    )
    parser.add_argument(
        "--video-backend",
        choices=["opencv", "pyav", "torchcodec", "pynvvideocodec", "deepstream"],
    )
    parser.add_argument("--video-backend-kwargs", default="{}")
    args = parser.parse_args()
    args.video_backend_kwargs = json.loads(args.video_backend_kwargs)
    if not isinstance(args.video_backend_kwargs, dict):
        parser.error("--video-backend-kwargs must be a JSON object")
    if args.http_connections < 1:
        parser.error("--http-connections must be positive")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
