from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import PyNvVideoCodec as nvc
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoProcessor

import queue
import threading
from dataclasses import dataclass

@dataclass
class SampledFrame:
    frame_index: int
    timestamp_s: float
    tensor: torch.Tensor

END_OF_STREAM = object()

@dataclass
class VideoResult:
    label: str
    path: str

    width: int
    height: int
    source_fps: float
    duration_s: float

    decoded_frames: int
    sampled_frames: int
    encoded_frames: int

    metadata_s: float
    model_load_s: float
    decoder_init_s: float
    decode_loop_s: float
    sampling_decision_s: float
    frame_materialization_s: float
    resize_normalize_s: float
    clip_image_encode_s: float
    clip_text_encode_s: float
    retrieval_s: float
    total_s: float

    decode_fps: float
    end_to_end_decode_fps: float
    sampled_fps: float

    top_indices: list[int]
    top_timestamps_s: list[float]
    top_scores: list[float]


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_cuda_call(function):
    synchronize_cuda()
    start = time.perf_counter()

    result = function()

    synchronize_cuda()
    elapsed = time.perf_counter() - start
    return result, elapsed


def get_video_metadata(path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    if not payload.get("streams"):
        raise RuntimeError(f"No video stream found in {path}")

    stream = payload["streams"][0]
    video_format = payload["format"]

    numerator_text, denominator_text = (
        stream["avg_frame_rate"].split("/")
    )

    numerator = float(numerator_text)
    denominator = float(denominator_text)

    source_fps = (
        numerator / denominator
        if denominator != 0
        else 0.0
    )

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "source_fps": source_fps,
        "duration_s": float(video_format["duration"]),
    }


def frame_to_chw_tensor(frame: Any) -> torch.Tensor:
    """
    Convert a PyNvVideoCodec GPU frame to a PyTorch tensor through
    DLPack.

    RGBP is normally planar RGB and commonly arrives as [3, H, W].
    This function also handles [H, W, 3].
    """
    tensor = torch.from_dlpack(frame)

    if tensor.ndim != 3:
        raise RuntimeError(
            f"Expected a 3D decoded frame, got shape={tuple(tensor.shape)}"
        )

    if tensor.shape[0] == 3:
        chw = tensor
    elif tensor.shape[-1] == 3:
        chw = tensor.permute(2, 0, 1)
    else:
        raise RuntimeError(
            "Cannot identify RGB channel dimension for frame with "
            f"shape={tuple(tensor.shape)}"
        )

    # Clone because the decoder may reuse the underlying frame surface.
    return chw.contiguous().clone()


def preprocess_batch(
    frames: list[torch.Tensor],
    *,
    image_size: int,
    device: torch.device,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    model_dtype: torch.dtype,
) -> torch.Tensor:

    if not frames:
        raise ValueError("Cannot preprocess an empty frame batch.")
    
    batch = torch.stack(frames, dim=0)

    if batch.device != device:
        batch = batch.to(device, non_blocking=False)

    batch = batch.to(dtype=torch.float32).div_(255.0)

    batch = functional.interpolate(
        batch,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    batch = batch.sub(image_mean).div(image_std)
    batch = batch.to(dtype=model_dtype)

    return batch

def decode_producer(
    *,
    path: Path,
    gpu_id: int,
    source_fps: float,
    target_sample_fps: float,
    output_queue: queue.Queue,
    stats: dict[str, Any],
    max_frames: int | None,
    progress_every: int,
) -> None:
    """
    Decode frames and place selected frames into output_queue.

    Each selected frame is cloned before being queued because the
    decoder may reuse its underlying output surface.
    """
    try:
        torch.cuda.set_device(gpu_id)

        decoder_init_start = time.perf_counter()

        decoder = nvc.SimpleDecoder(
            str(path),
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,
        )

        stats["decoder_init_s"] = (
            time.perf_counter() - decoder_init_start
        )

        decoded_frames = 0
        sampled_frames = 0
        sampling_decision_s = 0.0
        frame_materialization_s = 0.0

        next_sample_time_s = 0.0
        sample_period_s = 1.0 / target_sample_fps

        decode_start = time.perf_counter()

        for frame_index, frame in enumerate(decoder):
            decoded_frames += 1

            sampling_start = time.perf_counter()

            frame_time_s = frame_index / source_fps

            keep = (
                frame_time_s + 1e-9
                >= next_sample_time_s
            )

            sampling_decision_s += (
                time.perf_counter() - sampling_start
            )

            if keep:
                materialization_start = time.perf_counter()

                # Make an owned tensor before the decoder reuses
                # the underlying frame surface.
                tensor = frame_to_chw_tensor(frame)

                # Ensure the clone/materialization has completed
                # before another thread consumes the tensor.
                torch.cuda.synchronize(gpu_id)

                frame_materialization_s += (
                    time.perf_counter()
                    - materialization_start
                )

                output_queue.put(
                    SampledFrame(
                        frame_index=frame_index,
                        timestamp_s=frame_time_s,
                        tensor=tensor,
                    )
                )

                sampled_frames += 1
                next_sample_time_s += sample_period_s

            if (
                progress_every > 0
                and decoded_frames % progress_every == 0
            ):
                print(
                    f"decoded={decoded_frames}, "
                    f"sampled={sampled_frames}",
                    flush=True,
                )

            if (
                max_frames is not None
                and decoded_frames >= max_frames
            ):
                break

        torch.cuda.synchronize(gpu_id)

        stats["decode_producer_s"] = (
            time.perf_counter() - decode_start
        )
        stats["decoded_frames"] = decoded_frames
        stats["sampled_frames"] = sampled_frames
        stats["sampling_decision_s"] = sampling_decision_s
        stats["frame_materialization_s"] = (
            frame_materialization_s
        )

    except BaseException as error:
        stats["producer_error"] = error

    finally:
        # Always notify the consumer, including after an error.
        output_queue.put(END_OF_STREAM)


def encode_image_batch(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        features = model.get_image_features(
            pixel_values=pixel_values,
        )

        features = functional.normalize(
            features,
            p=2,
            dim=-1,
        )

    return features

def encode_text_query(
    model: torch.nn.Module,
    processor: Any,
    query: str,
    device: torch.device,
) -> torch.Tensor:
    tokens = processor(
        text=[query],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    tokens = {
        key: value.to(device)
        for key, value in tokens.items()
    }

    with torch.inference_mode():
        features = model.get_text_features(**tokens)
        features = functional.normalize(
            features,
            p=2,
            dim=-1,
        )

    return features


def should_sample_frame(
    frame_index: int,
    *,
    source_fps: float,
    target_sample_fps: float,
    next_sample_time_s: float,
) -> bool:
    frame_time_s = frame_index / source_fps

    return (
        frame_time_s + 1e-9
        >= next_sample_time_s
    )


def benchmark_video(
    *,
    label: str,
    path: Path,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    model_load_s: float,
    target_sample_fps: float,
    batch_size: int,
    image_size: int,
    top_k: int,
    query: str,
    gpu_id: int,
    max_frames: int | None,
    progress_every: int,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    model_dtype: torch.dtype,
) -> VideoResult:
    if not path.is_file():
        raise FileNotFoundError(path)

    end_to_end_start = time.perf_counter()

    metadata_start = time.perf_counter()
    metadata = get_video_metadata(path)
    metadata_s = time.perf_counter() - metadata_start

    source_fps = float(metadata["source_fps"])

    if source_fps <= 0:
        raise RuntimeError(
            f"Invalid source FPS for {path}: {source_fps}"
        )

    if target_sample_fps <= 0:
        raise ValueError("target_sample_fps must be positive")

    if target_sample_fps > source_fps:
        raise ValueError(
            f"Target sampling FPS {target_sample_fps} exceeds "
            f"source FPS {source_fps} for {path}"
        )

    decoder_init_start = time.perf_counter()

    decoder = nvc.SimpleDecoder(
        str(path),
        gpu_id=gpu_id,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGBP,
    )

    decoder_init_s = (
        time.perf_counter()
        - decoder_init_start
    )

    decoded_frames = 0
    sampled_frames = 0
    encoded_frames = 0

    sampling_decision_s = 0.0
    frame_materialization_s = 0.0
    resize_normalize_s = 0.0
    clip_image_encode_s = 0.0

    next_sample_time_s = 0.0
    sample_period_s = 1.0 / target_sample_fps

    pending_frames: list[torch.Tensor] = []
    pending_indices: list[int] = []
    pending_timestamps: list[float] = []

    embedding_batches: list[torch.Tensor] = []
    all_sampled_indices: list[int] = []
    all_sampled_timestamps: list[float] = []

    def process_pending_batch() -> None:
        nonlocal resize_normalize_s
        nonlocal clip_image_encode_s
        nonlocal encoded_frames

        if not pending_frames:
            return

        pixel_values, preprocess_elapsed = timed_cuda_call(
            lambda: preprocess_batch(
                pending_frames,
                image_size=image_size,
                device=device,
                image_mean=image_mean,
                image_std=image_std,
                model_dtype=model_dtype,
            )
        )

        resize_normalize_s += preprocess_elapsed

        features, encode_elapsed = timed_cuda_call(
            lambda: encode_image_batch(
                model,
                pixel_values,
            )
        )

        clip_image_encode_s += encode_elapsed
        encoded_frames += len(pending_frames)

        embedding_batches.append(features.cpu())

        all_sampled_indices.extend(pending_indices)
        all_sampled_timestamps.extend(pending_timestamps)

        pending_frames.clear()
        pending_indices.clear()
        pending_timestamps.clear()

    decode_start = time.perf_counter()
    previous_progress_time = decode_start

    for frame_index, frame in enumerate(decoder):
        decoded_frames += 1

        sampling_start = time.perf_counter()

        keep = should_sample_frame(
            frame_index,
            source_fps=source_fps,
            target_sample_fps=target_sample_fps,
            next_sample_time_s=next_sample_time_s,
        )

        sampling_decision_s += (
            time.perf_counter()
            - sampling_start
        )

        if keep:
            materialization_start = time.perf_counter()

            tensor = frame_to_chw_tensor(frame)

            synchronize_cuda()
            frame_materialization_s += (
                time.perf_counter()
                - materialization_start
            )

            timestamp_s = frame_index / source_fps

            pending_frames.append(tensor)
            pending_indices.append(frame_index)
            pending_timestamps.append(timestamp_s)

            sampled_frames += 1
            next_sample_time_s += sample_period_s

            if len(pending_frames) >= batch_size:
                process_pending_batch()

        if (
            progress_every > 0
            and decoded_frames % progress_every == 0
        ):
            now = time.perf_counter()

            print(
                f"[{label}] decoded={decoded_frames}, "
                f"sampled={sampled_frames}, "
                f"last_{progress_every}_s="
                f"{now - previous_progress_time:.3f}, "
                f"decode_wall_s={now - decode_start:.3f}",
                flush=True,
            )

            previous_progress_time = now

        if (
            max_frames is not None
            and decoded_frames >= max_frames
        ):
            break

    process_pending_batch()

    decode_loop_s = time.perf_counter() - decode_start

    text_features, clip_text_encode_s = timed_cuda_call(
        lambda: encode_text_query(
            model,
            processor,
            query,
            device,
        )
    )

    if embedding_batches:
        image_features = torch.cat(
            embedding_batches,
            dim=0,
        )

        retrieval_start = time.perf_counter()

        scores = (
            image_features
            @ text_features.cpu().T
        ).squeeze(1)

        actual_top_k = min(top_k, scores.numel())

        top_scores_tensor, top_positions = torch.topk(
            scores,
            k=actual_top_k,
        )

        retrieval_s = (
            time.perf_counter()
            - retrieval_start
        )

        top_indices = [
            all_sampled_indices[position]
            for position in top_positions.tolist()
        ]

        top_timestamps_s = [
            all_sampled_timestamps[position]
            for position in top_positions.tolist()
        ]

        top_scores = [
            float(value)
            for value in top_scores_tensor.tolist()
        ]
    else:
        retrieval_s = 0.0
        top_indices = []
        top_timestamps_s = []
        top_scores = []

    total_s = time.perf_counter() - end_to_end_start

    decode_fps = (
        decoded_frames / decode_loop_s
        if decode_loop_s > 0
        else 0.0
    )

    decode_end_to_end_s = (
        decoder_init_s + decode_loop_s
    )

    end_to_end_decode_fps = (
        decoded_frames / decode_end_to_end_s
        if decode_end_to_end_s > 0
        else 0.0
    )

    measured_video_duration_s = (
        decoded_frames / source_fps
    )

    sampled_fps = (
        sampled_frames / measured_video_duration_s
        if measured_video_duration_s > 0
        else 0.0
    )

    return VideoResult(
        label=label,
        path=str(path),
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        source_fps=source_fps,
        duration_s=float(metadata["duration_s"]),
        decoded_frames=decoded_frames,
        sampled_frames=sampled_frames,
        encoded_frames=encoded_frames,
        metadata_s=metadata_s,
        model_load_s=model_load_s,
        decoder_init_s=decoder_init_s,
        decode_loop_s=decode_loop_s,
        sampling_decision_s=sampling_decision_s,
        frame_materialization_s=frame_materialization_s,
        resize_normalize_s=resize_normalize_s,
        clip_image_encode_s=clip_image_encode_s,
        clip_text_encode_s=clip_text_encode_s,
        retrieval_s=retrieval_s,
        total_s=total_s,
        decode_fps=decode_fps,
        end_to_end_decode_fps=end_to_end_decode_fps,
        sampled_fps=sampled_fps,
        top_indices=top_indices,
        top_timestamps_s=top_timestamps_s,
        top_scores=top_scores,
    )


def print_result(result: VideoResult) -> None:
    print("\n" + "=" * 80)
    print(result.label.upper())
    print("=" * 80)

    print(f"Path:                 {result.path}")
    print(
        f"Resolution:           "
        f"{result.width}x{result.height}"
    )
    print(f"Duration:             {result.duration_s:.3f} s")
    print(f"Source FPS:           {result.source_fps:.3f}")
    print(f"Decoded frames:       {result.decoded_frames}")
    print(f"Sampled frames:       {result.sampled_frames}")
    print(f"Encoded frames:       {result.encoded_frames}")
    print(f"Effective sample FPS: {result.sampled_fps:.3f}")

    print("\nLatency breakdown")
    print("-" * 80)
    print(f"Metadata:              {result.metadata_s:.6f} s")
    print(f"Decoder initialization:{result.decoder_init_s:.6f} s")
    print(f"Decode loop:           {result.decode_loop_s:.6f} s")
    print(
        f"Sampling decisions:    "
        f"{result.sampling_decision_s:.6f} s"
    )
    print(
        f"Frame materialization: "
        f"{result.frame_materialization_s:.6f} s"
    )
    print(
        f"Resize + normalization:"
        f"{result.resize_normalize_s:.6f} s"
    )

    print(
        f"Retriever image encode:"
        f"{result.clip_image_encode_s:.6f} s"
    )
    print(
        f"Retriever text encode: "
        f"{result.clip_text_encode_s:.6f} s"
    )
    print(f"Retrieval/top-k:       {result.retrieval_s:.6f} s")
    print(f"Total pipeline:        {result.total_s:.6f} s")

    print("\nThroughput")
    print("-" * 80)
    print(f"Decode FPS:            {result.decode_fps:.2f}")
    print(
        f"Init + decode FPS:     "
        f"{result.end_to_end_decode_fps:.2f}"
    )

    print("\nTop retrieved frames")
    print("-" * 80)

    for rank, (
        frame_index,
        timestamp_s,
        score,
    ) in enumerate(
        zip(
            result.top_indices,
            result.top_timestamps_s,
            result.top_scores,
        ),
        start=1,
    ):
        print(
            f"{rank:>2}. frame={frame_index:<8} "
            f"time={timestamp_s:>9.3f}s "
            f"score={score:.5f}"
        )


def validate_common_resolution(
    results: list[VideoResult],
) -> None:
    resolutions = {
        (result.width, result.height)
        for result in results
    }

    if len(resolutions) != 1:
        found = ", ".join(
            f"{width}x{height}"
            for width, height in sorted(resolutions)
        )

        raise ValueError(
            "Input resolutions do not match. "
            f"Found: {found}"
        )

    width, height = next(iter(resolutions))

    print(
        f"\nVerified common resolution: "
        f"{width}x{height}"
    )


def save_csv(
    results: list[VideoResult],
    path: Path,
) -> None:
    rows = []

    excluded = {
        "top_indices",
        "top_timestamps_s",
        "top_scores",
    }

    for result in results:
        result_dict = asdict(result)

        rows.append(
            {
                key: value
                for key, value in result_dict.items()
                if key not in excluded
            }
        )

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    results: list[VideoResult],
    path: Path,
) -> None:
    with path.open("w") as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=2,
        )


def plot_total_latency(
    results: list[VideoResult],
    output_path: Path,
) -> None:
    ordered = sorted(
        results,
        key=lambda result: result.duration_s,
    )

    durations = [
        result.duration_s
        for result in ordered
    ]

    totals = [
        result.total_s
        for result in ordered
    ]

    decode_latencies = [
        result.decode_loop_s
        for result in ordered
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        durations,
        totals,
        marker="o",
        label="Full retrieval pipeline",
    )

    plt.plot(
        durations,
        decode_latencies,
        marker="s",
        label="Decode loop",
    )

    for result in ordered:
        plt.annotate(
            result.label,
            (result.duration_s, result.total_s),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Video duration (seconds)")
    plt.ylabel("Latency (seconds)")
    plt.title("Pipeline Latency vs. Video Duration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def exclusive_decode_time(
    result: VideoResult,
) -> float:
    """
    Estimate decoder/iteration time excluding stages already
    measured inside the decode loop.
    """
    return max(
        0.0,
        result.decode_loop_s
        - result.sampling_decision_s
        - result.frame_materialization_s
        - result.resize_normalize_s
        - result.clip_image_encode_s,
    )

def plot_stage_breakdown(
    results: list[VideoResult],
    output_path: Path,
) -> None:
    labels = [
        result.label
        for result in results
    ]

    stages = {
        "Metadata": [
            result.metadata_s
            for result in results
        ],
        "Decoder init": [
            result.decoder_init_s
            for result in results
        ],
        "Decode/iteration": [
            exclusive_decode_time(result)
            for result in results
        ],
        "Sampling decisions": [
            result.sampling_decision_s
            for result in results
        ],
        "Materialize": [
            result.frame_materialization_s
            for result in results
        ],
        "Resize/normalize": [
            result.resize_normalize_s
            for result in results
        ],
        "Retriever image": [
            result.clip_image_encode_s
            for result in results
        ],
        "Retriever text": [
            result.clip_text_encode_s
            for result in results
        ],
        "Top-k": [
            result.retrieval_s
            for result in results
        ],
    }

    x_positions = list(range(len(labels)))
    bottoms = [0.0] * len(labels)

    plt.figure(figsize=(9, 6))

    for stage_name, values in stages.items():
        plt.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=stage_name,
        )

        bottoms = [
            bottom + value
            for bottom, value in zip(bottoms, values)
        ]

    plt.xticks(x_positions, labels)
    plt.xlabel("Video duration category")
    plt.ylabel("Latency (seconds)")
    plt.title("End-to-End Pipeline Latency Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_summary(results: list[VideoResult]) -> None:
    print("\n" + "=" * 150)
    print("SUMMARY")
    print("=" * 150)

    print(
        f"{'Dataset':<10}"
        f"{'Duration':>12}"
        f"{'Decoded':>12}"
        f"{'Sampled':>12}"
        f"{'Decode':>12}"
        f"{'Resize':>12}"
        f"{'Retriever':>12}"
        f"{'Retrieve':>12}"
        f"{'Total':>12}"
        f"{'Decode FPS':>14}"
    )

    print("-" * 150)

    for result in results:
        print(
            f"{result.label:<10}"
            f"{result.duration_s:>12.2f}"
            f"{result.decoded_frames:>12}"
            f"{result.sampled_frames:>12}"
            f"{result.decode_loop_s:>12.3f}"
            f"{result.resize_normalize_s:>12.3f}"
            f"{result.clip_image_encode_s:>12.3f}"
            f"{result.retrieval_s:>12.6f}"
            f"{result.total_s:>12.3f}"
            f"{result.decode_fps:>14.2f}"
        )

    total_latencies = [
        result.total_s
        for result in results
    ]

    print(
        f"\nMedian total latency: "
        f"{statistics.median(total_latencies):.3f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument(
                "--runs",
                type=int,
                default=3,
                help="Number of measured runs per video.",
            )

    parser.add_argument(
        "--query",
        default="the event most relevant to the question",
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--retrieval-model",
        default="google/siglip2-so400m-patch14-384",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("nvdec_pipeline"),
    )

    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    model_load_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(
        args.retrieval_model,
    )

    image_processor = processor.image_processor
    processor_size = image_processor.size

    if isinstance(processor_size, dict):
        fallback_size = args.image_size

        model_image_height = int(
            processor_size.get(
                "height",
                processor_size.get(
                    "shortest_edge",
                    fallback_size,
                ),
            )
        )

        model_image_width = int(
            processor_size.get(
                "width",
                processor_size.get(
                    "shortest_edge",
                    fallback_size,
                ),
            )
        )
    else:
        model_image_height = args.image_size
        model_image_width = args.image_size

    if model_image_height != model_image_width:
        raise ValueError(
            "This benchmark requires a square model input, "
            f"but the processor requested "
            f"{model_image_height}x{model_image_width}."
        )

    image_size = model_image_height

    print(
        f"Retriever image size: "
        f"{image_size}x{image_size}"
    )

    image_mean = torch.tensor(
        image_processor.image_mean,
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    image_std = torch.tensor(
        image_processor.image_std,
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    model = AutoModel.from_pretrained(
        args.retrieval_model,
        torch_dtype=torch.float16,
    )
    model = model.to(device)
    model.eval()
    model_dtype = next(model.parameters()).dtype

    synchronize_cuda()
    model_load_s = (
        time.perf_counter()
        - model_load_start
    )

    print(
        f"Loaded {args.retrieval_model} in "
        f"{model_load_s:.3f} seconds"
    )

    # Warm up the complete preprocessing and retriever path so
    # first-run CUDA costs do not distort measurements.

    # Warm up the complete resize, normalization, and CLIP path.
    dummy_frame = torch.zeros(
        3,
        360,
        480,
        dtype=torch.uint8,
        device=device,
    )

    dummy_frames = [
        dummy_frame.clone()
        for _ in range(args.batch_size)
    ]

    warmup_pixels = preprocess_batch(
        dummy_frames,
        image_size=image_size,
        device=device,
        image_mean=image_mean,
        image_std=image_std,
        model_dtype=model_dtype,
    )

    with torch.inference_mode():
        _ = encode_image_batch(
            model,
            warmup_pixels,
        )

    _ = encode_text_query(
        model,
        processor,
        args.query,
        device,
    )

    synchronize_cuda()

    print("Completed preprocessing and retriever warm-up.")

    results: list[VideoResult] = []

    for label, path in [
        ("short", args.short),
        ("medium", args.medium),
        ("long", args.long),
    ]:
        print("\n" + "=" * 80)
        print(f"BENCHMARKING {label.upper()}")
        print("=" * 80)

        measured_runs: list[VideoResult] = []

        for run_index in range(args.runs):
            print(
                f"\n[{label}] run "
                f"{run_index + 1}/{args.runs}"
            )

            run_result = benchmark_video(
                label=label,
                path=path,
                model=model,
                processor=processor,
                device=device,
                model_load_s=model_load_s,
                target_sample_fps=args.sample_fps,
                batch_size=args.batch_size,
                image_size=image_size,
                image_mean=image_mean,
                image_std=image_std,
                model_dtype=model_dtype,
                top_k=args.top_k,
                query=args.query,
                gpu_id=args.gpu_id,
                max_frames=args.max_frames,
                progress_every=args.progress_every,
            )

            measured_runs.append(run_result)

        measured_runs.sort(
            key=lambda item: item.total_s
        )

        median_result = measured_runs[
            len(measured_runs) // 2
        ]

        total_values = [
            item.total_s
            for item in measured_runs
        ]

        standard_deviation = (
            statistics.stdev(total_values)
            if len(total_values) > 1
            else 0.0
        )

        print(
            f"\n[{label}] latency statistics: "
            f"median={statistics.median(total_values):.3f}s, "
            f"mean={statistics.mean(total_values):.3f}s, "
            f"std={standard_deviation:.3f}s"
        )

        results.append(median_result)
        print_result(median_result)



    validate_common_resolution(results)
    print_summary(results)

    csv_path = Path(f"{args.output_prefix}.csv")
    json_path = Path(f"{args.output_prefix}.json")

    latency_plot_path = Path(
        f"{args.output_prefix}_latency.png"
    )

    breakdown_plot_path = Path(
        f"{args.output_prefix}_breakdown.png"
    )

    save_csv(results, csv_path)
    save_json(results, json_path)

    plot_total_latency(
        results,
        latency_plot_path,
    )

    plot_stage_breakdown(
        results,
        breakdown_plot_path,
    )

    print("\nSaved outputs:")
    print(f"  CSV:             {csv_path}")
    print(f"  JSON:            {json_path}")
    print(f"  Latency graph:   {latency_plot_path}")
    print(f"  Breakdown graph: {breakdown_plot_path}")


if __name__ == "__main__":
    main()