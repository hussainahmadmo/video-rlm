#!/usr/bin/env python3
"""Report native vLLM video-loader capabilities in the active environment."""

from __future__ import annotations

import importlib.util
import inspect
import json

import vllm
from vllm.multimodal.media import VIDEO_LOADER_REGISTRY
from vllm.multimodal.media.video import VideoMediaIO


def main() -> None:
    source = inspect.getsource(VideoMediaIO.__init__)
    backend_key = (
        "video_backend" if 'kwargs.pop("video_backend"' in source else "backend"
    )
    registered = sorted(VIDEO_LOADER_REGISTRY.name2class)
    print(json.dumps({
        "vllm_version": vllm.__version__,
        "media_io_backend_key": backend_key,
        "registered_video_backends": registered,
        "pynvvideocodec_importable": (
            importlib.util.find_spec("PyNvVideoCodec") is not None
        ),
        "native_pynvvideocodec_registered": "pynvvideocodec" in registered,
        "native_deepstream_registered": "deepstream" in registered,
        "note": (
            "PyNvVideoCodec being importable does not mean that this vLLM "
            "build registered its native pynvvideocodec loader."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
