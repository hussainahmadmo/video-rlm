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
