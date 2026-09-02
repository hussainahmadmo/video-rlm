"""Indexed NVDEC decoding for sparsely sampled video frames.

Unlike ``batched_nvdec_decode``, this backend does not iterate through every
frame from the beginning of a video to the final requested timestamp.  It asks
PyNvVideoCodec for the requested frame indices as one batch.  This matches the
access pattern used by recent native-vLLM PyNvVideoCodec loaders and is a
better fit for long videos with small frame budgets.
"""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import av
import PyNvVideoCodec as nvc
import torch
from PIL import Image


_THREAD_LOCAL = threading.local()


def _new_decoder(video_path: str | Path, gpu_id: int) -> object:
    return nvc.SimpleDecoder(
        str(video_path),
        gpu_id=gpu_id,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGBP,
        need_scanned_stream_metadata=True,
    )


def _decoder_for(video_path: str | Path, gpu_id: int) -> object:
    """Reuse one decoder per preparation thread when reconfiguration is safe."""
    source = str(video_path)
    decoder = getattr(_THREAD_LOCAL, "decoder", None)
    previous_source = getattr(_THREAD_LOCAL, "source", None)
    previous_gpu_id = getattr(_THREAD_LOCAL, "gpu_id", None)
    if decoder is None or previous_gpu_id != gpu_id:
        decoder = _new_decoder(source, gpu_id)
    elif previous_source != source:
        try:
            decoder.reconfigure_decoder(source)
        except Exception:
            # Some containers cannot safely reset their demuxer. Rebuilding is
            # slower than reconfiguration but avoids aborting the whole suite.
            try:
                decoder.stop()
            except Exception:
                pass
            decoder = _new_decoder(source, gpu_id)
    _THREAD_LOCAL.decoder = decoder
    _THREAD_LOCAL.source = source
    _THREAD_LOCAL.gpu_id = gpu_id
    return decoder


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
    """Decode exactly the requested indices with one NVDEC batch call.

    The exported name intentionally follows the runner's batch-decoder
    interface. ``VIDEO_RLM_NVDEC_GPU_ID`` selects the physical CUDA device.
    """
    if not timestamps_s:
        return []

    started = time.monotonic()
    fps = _source_fps(video_path)
    requested = [max(0, round(float(value) * fps)) for value in timestamps_s]
    unique_indices = sorted(set(requested))
    gpu_id = int(os.environ.get("VIDEO_RLM_NVDEC_GPU_ID", "0"))

    decoder = _decoder_for(video_path, gpu_id)
    frames = decoder.get_batch_frames_by_index(unique_indices)
    if time.monotonic() - started > timeout_s:
        raise TimeoutError(f"indexed NVDEC exceeded {timeout_s}s: {video_path}")
    if len(frames) != len(unique_indices):
        raise RuntimeError(
            "indexed NVDEC returned "
            f"{len(frames)}/{len(unique_indices)} frames: {video_path}"
        )

    encoded = {
        index: _jpeg(frame, max_side)
        for index, frame in zip(unique_indices, frames, strict=True)
    }
    if time.monotonic() - started > timeout_s:
        raise TimeoutError(f"indexed NVDEC exceeded {timeout_s}s: {video_path}")
    return [encoded[index] for index in requested]
