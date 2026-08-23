"""One-pass CPU video decoding for uniformly sampled JPEG frames."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import av
from PIL import Image


def _frame_time_s(frame: av.VideoFrame, stream: av.video.stream.VideoStream) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return None


def _jpeg(frame: av.VideoFrame, max_side: int) -> bytes:
    image = frame.to_image()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def decode_jpegs_batch_cpu(
    video_path: str | Path,
    timestamps_s: list[float],
    max_side: int,
    timeout_s: float,
) -> list[bytes]:
    """Decode sorted timestamps in one sequential CPU decoder session.

    For each requested timestamp, retain whichever of the adjacent decoded
    frames is temporally closest. The video is opened and decoded only once.
    """
    if not timestamps_s:
        return []
    targets = sorted(float(value) for value in timestamps_s)
    deadline = time.monotonic() + timeout_s
    selected: list[av.VideoFrame] = []
    previous: av.VideoFrame | None = None
    previous_time: float | None = None
    target_index = 0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        thread_count = int(os.environ.get("VIDEO_RLM_FFMPEG_THREADS", "0"))
        if thread_count > 0:
            stream.codec_context.thread_count = thread_count
        stream.codec_context.thread_type = "AUTO"

        for frame in container.decode(stream):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"batched CPU decode exceeded {timeout_s}s: {video_path}"
                )
            current_time = _frame_time_s(frame, stream)
            if current_time is None:
                continue
            while target_index < len(targets) and current_time >= targets[target_index]:
                target = targets[target_index]
                if previous is not None and previous_time is not None:
                    chosen = (
                        previous
                        if abs(previous_time - target) <= abs(current_time - target)
                        else frame
                    )
                else:
                    chosen = frame
                selected.append(chosen)
                target_index += 1
            previous = frame
            previous_time = current_time
            if target_index == len(targets):
                break

    if target_index < len(targets):
        if previous is None:
            raise RuntimeError(f"no decodable frames: {video_path}")
        selected.extend(previous for _ in range(len(targets) - target_index))

    return [_jpeg(frame, max_side) for frame in selected]
