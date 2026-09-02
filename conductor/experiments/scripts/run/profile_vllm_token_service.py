#!/usr/bin/env python3
"""Profile text-prefill, autoregressive decode, and visual-input service.

The profiler deliberately prepares synthetic images before starting a timed
request.  Reported request time therefore covers only the OpenAI-compatible
vLLM call.  For every configuration it launches a matched concurrent batch
and reports batch wall time divided by batch size, following the profiling
methodology used by VTC's profiled cost-function experiment.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image


def integer_list(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError("expected non-negative integers")
    return values


def make_text_prompt(target: int) -> str:
    # The API-reported prompt-token count, rather than ``target``, is used in
    # analysis. Repetition only supplies stable prompts of increasing size.
    return (
        "Read the following synthetic payload and answer with neutral text. "
        + " measurement" * max(1, target)
    )


def make_jpeg_data_url(side: int) -> str:
    image = Image.new("RGB", (side, side), color=(96, 128, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def visual_content(frame_count: int, image_url: str) -> Any:
    if frame_count == 0:
        return "Describe the synthetic visual input briefly."
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for _ in range(frame_count)
    ]
    content.append(
        {"type": "text", "text": "Describe the synthetic visual input briefly."}
    )
    return content


def one_request(
    *,
    base_url: str,
    model: str,
    content: Any,
    max_tokens: int,
    ignore_eos: bool,
    max_pixels: int,
    timeout_s: float,
) -> dict[str, Any]:
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=timeout_s,
        max_retries=0,
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={
            "ignore_eos": ignore_eos,
            "mm_processor_kwargs": {"max_pixels": max_pixels},
        },
    )
    elapsed_s = time.perf_counter() - started
    usage = response.usage
    if usage is None:
        raise RuntimeError("vLLM response did not include token usage")
    return {
        "elapsed_s": elapsed_s,
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "total_tokens": int(usage.total_tokens),
    }


def run_batch(
    *,
    family: str,
    target: int,
    content: Any,
    max_tokens: int,
    ignore_eos: bool,
    concurrency: int,
    repeat: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                one_request,
                base_url=f"http://{args.host}:{args.port}/v1",
                model=args.model,
                content=content,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
                max_pixels=args.max_pixels,
                timeout_s=args.timeout_s,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    batch_wall_s = time.perf_counter() - started
    return {
        "family": family,
        "target": target,
        "concurrency": concurrency,
        "repeat": repeat,
        "batch_wall_s": batch_wall_s,
        "amortized_service_s": batch_wall_s / concurrency,
        "mean_request_elapsed_s": statistics.mean(
            item["elapsed_s"] for item in results
        ),
        "max_request_elapsed_s": max(item["elapsed_s"] for item in results),
        "mean_prompt_tokens": statistics.mean(
            item["prompt_tokens"] for item in results
        ),
        "mean_completion_tokens": statistics.mean(
            item["completion_tokens"] for item in results
        ),
        "request_results": results,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def completed_keys(path: Path) -> set[tuple[str, int, int, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int, int, int]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(
                (
                    str(row["family"]),
                    int(row["target"]),
                    int(row["concurrency"]),
                    int(row["repeat"]),
                )
            )
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-lengths", type=integer_list, default=[64, 256, 1024])
    parser.add_argument("--output-lengths", type=integer_list, default=[8, 32, 128, 256])
    parser.add_argument("--visual-frames", type=integer_list, default=[0, 1, 4, 8, 16])
    parser.add_argument("--concurrency", type=integer_list, default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image-side", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.image_side < 1:
        parser.error("--repeats and --image-side must be positive")
    if min(args.input_lengths + args.output_lengths) < 1:
        parser.error("input and output lengths must be positive")
    if min(args.concurrency) < 1:
        parser.error("concurrency must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.output)
    image_url = make_jpeg_data_url(args.image_side)

    if args.warmup and not done:
        one_request(
            base_url=f"http://{args.host}:{args.port}/v1",
            model=args.model,
            content="Warm up the model and answer with one token.",
            max_tokens=1,
            ignore_eos=True,
            max_pixels=args.max_pixels,
            timeout_s=args.timeout_s,
        )

    cases: list[tuple[str, int, Any, int, bool]] = []
    for length in args.input_lengths:
        cases.append(("prefill", length, make_text_prompt(length), 1, True))
    decode_prompt = make_text_prompt(32)
    for length in args.output_lengths:
        cases.append(("decode", length, decode_prompt, length, True))
    for frames in args.visual_frames:
        cases.append(
            ("visual", frames, visual_content(frames, image_url), 1, True)
        )

    for family, target, content, max_tokens, ignore_eos in cases:
        for concurrency in args.concurrency:
            for repeat in range(1, args.repeats + 1):
                key = (family, target, concurrency, repeat)
                if key in done:
                    print(f"SKIP family={family} target={target} "
                          f"concurrency={concurrency} repeat={repeat}", flush=True)
                    continue
                print(f"START family={family} target={target} "
                      f"concurrency={concurrency} repeat={repeat}", flush=True)
                row = run_batch(
                    family=family,
                    target=target,
                    content=content,
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    concurrency=concurrency,
                    repeat=repeat,
                    args=args,
                )
                row.update(
                    {
                        "model": args.model,
                        "host": args.host,
                        "port": args.port,
                        "image_side": args.image_side,
                        "max_pixels": args.max_pixels,
                    }
                )
                append_jsonl(args.output, row)
                print(
                    f"DONE family={family} target={target} "
                    f"concurrency={concurrency} repeat={repeat} "
                    f"amortized_s={row['amortized_service_s']:.4f} "
                    f"prompt_tokens={row['mean_prompt_tokens']:.1f} "
                    f"completion_tokens={row['mean_completion_tokens']:.1f}",
                    flush=True,
                )

    print(f"PROFILE_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
