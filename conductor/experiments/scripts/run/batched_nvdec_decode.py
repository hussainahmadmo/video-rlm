"""One-pass NVDEC video decoding for uniformly sampled JPEG frames."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import av
import PyNvVideoCodec as nvc
import torch
from PIL import Image


def _source_fps(video_path: str | Path) -> float:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or stream.guessed_rate
        if rate is None or float(rate) <= 0:
            raise RuntimeError(f"cannot determine frame rate: {video_path}")
        return float(rate)


def _jpeg(frame: object, max_side: int) -> bytes:
    tensor = torch.from_dlpack(frame)
    if tensor.ndim != 3:
        raise RuntimeError(f"expected a 3D NVDEC frame, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] != 3:
        raise RuntimeError(
            f"cannot identify RGB channel dimension in {tuple(tensor.shape)}"
        )
    array = tensor.contiguous().to(device="cpu", dtype=torch.uint8).numpy()
    image = Image.fromarray(array, mode="RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def decode_jpegs_batch_cpu(
    video_path: str | Path,
    timestamps_s: list[float],
    max_side: int,
    timeout_s: float,
) -> list[bytes]:
    """Decode requested timestamps through one sequential NVDEC session.

    The exported function name intentionally matches the runner's batch-decoder
    interface. ``VIDEO_RLM_NVDEC_GPU_ID`` selects the physical CUDA device.
    """
    if not timestamps_s:
        return []
    targets = [float(value) for value in timestamps_s]
    fps = _source_fps(video_path)
    target_indices = [max(0, round(value * fps)) for value in targets]
    order = sorted(range(len(targets)), key=lambda index: target_indices[index])
    selected: list[bytes | None] = [None] * len(targets)
    cursor = 0
    deadline = time.monotonic() + timeout_s
    gpu_id = int(os.environ.get("VIDEO_RLM_NVDEC_GPU_ID", "0"))

    decoder = nvc.SimpleDecoder(
        str(video_path),
        gpu_id=gpu_id,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGBP,
    )
    last_jpeg: bytes | None = None
    for frame_index, frame in enumerate(decoder):
        if time.monotonic() > deadline:
            raise TimeoutError(f"NVDEC exceeded {timeout_s}s: {video_path}")
        if cursor >= len(order):
            break
        if frame_index < target_indices[order[cursor]]:
            continue
        last_jpeg = _jpeg(frame, max_side)
        while cursor < len(order) and target_indices[order[cursor]] <= frame_index:
            selected[order[cursor]] = last_jpeg
            cursor += 1

    if last_jpeg is None:
        raise RuntimeError(f"no decodable NVDEC frames: {video_path}")
    while cursor < len(order):
        selected[order[cursor]] = last_jpeg
        cursor += 1
    return [jpeg for jpeg in selected if jpeg is not None]
