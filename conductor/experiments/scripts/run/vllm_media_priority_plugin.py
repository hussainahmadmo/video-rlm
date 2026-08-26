#!/usr/bin/env python3
"""Opt-in, bounded priority admission for vLLM native media loading.

This module deliberately does not modify vLLM's default behavior.  The
``install`` function monkey-patches the API-server process only when called by
``run_vllm_with_media_priority.py``.  It propagates the OpenAI request's
``priority`` field into native media loading and admits at most
``max_active_jobs`` concurrent fetch/decode operations.

Smaller numeric values have higher priority, matching vLLM's priority
scheduler.  Work that has already started remains non-preemptive.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import itertools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


LOGGER = logging.getLogger("vllm.media_priority")

_T = TypeVar("_T")
_REQUEST_PRIORITY: contextvars.ContextVar[int] = contextvars.ContextVar(
    "vllm_media_request_priority", default=0
)


class BoundedPriorityMediaScheduler:
    """Priority queue in front of asynchronous native media preparation.

    ``max_active_jobs`` bounds fetching plus decoding. ``max_pending_jobs=0``
    means that the pending heap is unbounded; a positive value applies
    backpressure to API handlers once that many media items are queued.
    """

    def __init__(self, max_active_jobs: int, max_pending_jobs: int = 0) -> None:
        if max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1")
        if max_pending_jobs < 0:
            raise ValueError("max_pending_jobs cannot be negative")

        self.max_active_jobs = max_active_jobs
        self.max_pending_jobs = max_pending_jobs
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, Callable[[], Awaitable[Any]], asyncio.Future[Any]]
        ] = asyncio.PriorityQueue(maxsize=max_pending_jobs)
        self._sequence = itertools.count()
        self._workers: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_workers(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("media scheduler cannot be shared across event loops")
        if self._workers:
            return

        self._loop = loop
        self._workers = [
            loop.create_task(self._worker(index), name=f"media-priority-{index}")
            for index in range(self.max_active_jobs)
        ]

    async def submit(
        self,
        priority: int,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        self._ensure_workers()
        assert self._loop is not None
        future: asyncio.Future[_T] = self._loop.create_future()
        await self._queue.put((int(priority), next(self._sequence), operation, future))
        return await future

    async def _worker(self, worker_index: int) -> None:
        del worker_index
        while True:
            priority, sequence, operation, future = await self._queue.get()
            del priority, sequence
            try:
                if future.cancelled():
                    continue
                result = await operation()
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    @property
    def pending_jobs(self) -> int:
        return self._queue.qsize()


_INSTALLED = False
_SCHEDULER: BoundedPriorityMediaScheduler | None = None
_FRAME_BUDGET_INSTALLED = False

FRAME_BUDGET_QUERY_KEY = "vllm_num_frames"


def extract_frame_budget(url: str) -> tuple[str, int | None]:
    """Remove and return the opt-in per-request video frame budget."""
    parts = urlsplit(url)
    kept: list[tuple[str, str]] = []
    values: list[str] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == FRAME_BUDGET_QUERY_KEY:
            values.append(value)
        else:
            kept.append((key, value))

    if not values:
        return url, None
    if len(values) != 1:
        raise ValueError(f"{FRAME_BUDGET_QUERY_KEY} must occur exactly once")

    try:
        frame_budget = int(values[0])
    except ValueError as exc:
        raise ValueError(f"invalid {FRAME_BUDGET_QUERY_KEY}: {values[0]!r}") from exc
    if frame_budget < 1:
        raise ValueError(f"{FRAME_BUDGET_QUERY_KEY} must be positive")

    clean_url = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment)
    )
    return clean_url, frame_budget


def install_frame_budget_adapter() -> None:
    """Allow an OpenAI video URL to select its own native frame budget.

    vLLM 0.17 exposes ``--media-io-kwargs`` only as a server-wide setting; its
    OpenAI request schema ignores a per-request ``media_io_kwargs`` field. The
    experiment runner therefore carries ``vllm_num_frames`` as URL metadata.
    This hook removes that metadata before fetching and constructs the video
    loader with the requested budget. It does not change request ordering.
    """
    global _FRAME_BUDGET_INSTALLED
    if _FRAME_BUDGET_INSTALLED:
        return

    from vllm import envs
    from vllm.multimodal.media.connector import MediaConnector
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.media.video import VideoMediaIO

    original_fetch_video = MediaConnector.fetch_video
    original_fetch_video_async = MediaConnector.fetch_video_async

    def video_io(connector: Any, image_mode: str, frame_budget: int) -> Any:
        image_io = ImageMediaIO(
            image_mode=image_mode,
            **connector.media_io_kwargs.get("image", {}),
        )
        kwargs = dict(connector.media_io_kwargs.get("video", {}))
        kwargs["num_frames"] = frame_budget
        return VideoMediaIO(image_io, **kwargs)

    @functools.wraps(original_fetch_video)
    def fetch_video_with_budget(
        self: Any,
        video_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Any:
        clean_url, frame_budget = extract_frame_budget(video_url)
        if frame_budget is None:
            return original_fetch_video(self, video_url, image_mode=image_mode)
        return self.load_from_url(
            clean_url,
            video_io(self, image_mode, frame_budget),
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    @functools.wraps(original_fetch_video_async)
    async def fetch_video_async_with_budget(
        self: Any,
        video_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Any:
        clean_url, frame_budget = extract_frame_budget(video_url)
        if frame_budget is None:
            return await original_fetch_video_async(
                self,
                video_url,
                image_mode=image_mode,
            )
        return await self.load_from_url_async(
            clean_url,
            video_io(self, image_mode, frame_budget),
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    MediaConnector.fetch_video = fetch_video_with_budget
    MediaConnector.fetch_video_async = fetch_video_async_with_budget
    _FRAME_BUDGET_INSTALLED = True
    LOGGER.warning(
        "Enabled per-request video frame budgets via URL key %s",
        FRAME_BUDGET_QUERY_KEY,
    )


def install(max_active_jobs: int, max_pending_jobs: int = 0) -> None:
    """Install the API-server hooks. Safe to call more than once."""
    global _INSTALLED, _SCHEDULER
    if _INSTALLED:
        return

    # Imports are intentionally delayed: importing this file alone must not
    # initialize vLLM or CUDA.
    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm.multimodal.media.connector import MediaConnector

    _SCHEDULER = BoundedPriorityMediaScheduler(
        max_active_jobs=max_active_jobs,
        max_pending_jobs=max_pending_jobs,
    )

    original_preprocess_chat = OpenAIServing._preprocess_chat
    original_load_from_url_async = MediaConnector.load_from_url_async

    @functools.wraps(original_preprocess_chat)
    async def preprocess_chat_with_priority(
        self: Any,
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        token = _REQUEST_PRIORITY.set(int(getattr(request, "priority", 0)))
        try:
            return await original_preprocess_chat(self, request, *args, **kwargs)
        finally:
            _REQUEST_PRIORITY.reset(token)

    @functools.wraps(original_load_from_url_async)
    async def load_from_url_with_priority(
        self: Any,
        url: str,
        media_io: Any,
        *,
        fetch_timeout: int | None = None,
    ) -> Any:
        assert _SCHEDULER is not None
        priority = _REQUEST_PRIORITY.get()

        async def fetch_and_decode() -> Any:
            return await original_load_from_url_async(
                self,
                url,
                media_io,
                fetch_timeout=fetch_timeout,
            )

        return await _SCHEDULER.submit(priority, fetch_and_decode)

    OpenAIServing._preprocess_chat = preprocess_chat_with_priority
    MediaConnector.load_from_url_async = load_from_url_with_priority
    _INSTALLED = True

    LOGGER.warning(
        "Enabled bounded native-media priority admission: active=%d pending=%s",
        max_active_jobs,
        "unbounded" if max_pending_jobs == 0 else max_pending_jobs,
    )
