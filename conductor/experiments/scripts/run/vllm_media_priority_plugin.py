#!/usr/bin/env python3
"""Opt-in fair or priority admission for vLLM native media loading.

This module deliberately does not modify vLLM's default behavior.  The
``install`` function monkey-patches the API-server process only when called by
``run_vllm_with_media_priority.py``.  It propagates the OpenAI request's
``priority`` or ``user`` field into native media loading and admits at most
``max_active_jobs`` concurrent media operations. The admission point can
either cover fetching plus decoding or decoding alone.

Smaller numeric values have higher priority, matching vLLM's priority
scheduler.  Work that has already started remains non-preemptive.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import itertools
import logging
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


LOGGER = logging.getLogger("vllm.media_priority")

_T = TypeVar("_T")
_REQUEST_PRIORITY: contextvars.ContextVar[int] = contextvars.ContextVar(
    "vllm_media_request_priority", default=0
)
_REQUEST_TENANT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vllm_media_request_tenant", default="default"
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


class BoundedMaxMinMediaScheduler:
    """Work-conserving max-min admission using measured media service.

    Each tenant has a FIFO queue. The least-served backlogged tenant is chosen
    whenever a native media slot becomes free. An estimated cost is charged at
    dispatch and reconciled with measured operation time on completion.
    """

    def __init__(
        self,
        max_active_jobs: int,
        max_pending_jobs: int = 0,
        *,
        initial_service_estimate_s: float = 1.0,
        ewma_alpha: float = 0.25,
    ) -> None:
        if max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1")
        if max_pending_jobs < 0:
            raise ValueError("max_pending_jobs cannot be negative")
        if initial_service_estimate_s <= 0:
            raise ValueError("initial_service_estimate_s must be positive")
        if not 0 < ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self.max_active_jobs = max_active_jobs
        self.max_pending_jobs = max_pending_jobs
        self.initial_service_estimate_s = initial_service_estimate_s
        self.ewma_alpha = ewma_alpha
        self._queues: dict[
            str,
            deque[tuple[int, Callable[[], Awaitable[Any]], asyncio.Future[Any]]],
        ] = defaultdict(deque)
        self._service_s: dict[str, float] = defaultdict(float)
        self._estimate_s: dict[str, float] = {}
        self._outstanding: dict[str, int] = defaultdict(int)
        self._sequence = itertools.count()
        self._queued_jobs = 0
        self._condition: asyncio.Condition | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_workers(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("media scheduler cannot be shared across event loops")
        if self._workers:
            return
        self._loop = loop
        self._condition = asyncio.Condition()
        self._workers = [
            loop.create_task(self._worker(index), name=f"media-max-min-{index}")
            for index in range(self.max_active_jobs)
        ]

    async def submit(
        self, tenant: str, operation: Callable[[], Awaitable[_T]]
    ) -> _T:
        self._ensure_workers()
        assert self._loop is not None and self._condition is not None
        tenant = str(tenant or "default")
        future: asyncio.Future[_T] = self._loop.create_future()
        async with self._condition:
            await self._condition.wait_for(
                lambda: self.max_pending_jobs == 0
                or self._queued_jobs < self.max_pending_jobs
            )
            if self._outstanding[tenant] == 0:
                active_service = [
                    self._service_s[name]
                    for name, count in self._outstanding.items()
                    if count > 0 and name != tenant
                ]
                if active_service:
                    self._service_s[tenant] = max(
                        self._service_s[tenant], min(active_service)
                    )
            self._outstanding[tenant] += 1
            self._queues[tenant].append((next(self._sequence), operation, future))
            self._queued_jobs += 1
            self._condition.notify_all()
        return await future

    async def _worker(self, worker_index: int) -> None:
        del worker_index
        assert self._condition is not None
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._queued_jobs > 0)
                tenant = min(
                    (name for name, queue in self._queues.items() if queue),
                    key=lambda name: (
                        self._service_s[name], self._queues[name][0][0], name
                    ),
                )
                _, operation, future = self._queues[tenant].popleft()
                self._queued_jobs -= 1
                estimate = self._estimate_s.get(
                    tenant, self.initial_service_estimate_s
                )
                self._service_s[tenant] += estimate
                self._condition.notify_all()

            started = time.perf_counter()
            try:
                if not future.cancelled():
                    result = await operation()
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                observed = 0.0 if future.cancelled() else time.perf_counter() - started
                async with self._condition:
                    self._service_s[tenant] = max(
                        0.0, self._service_s[tenant] + observed - estimate
                    )
                    if observed > 0:
                        previous = self._estimate_s.get(
                            tenant, self.initial_service_estimate_s
                        )
                        self._estimate_s[tenant] = (
                            self.ewma_alpha * observed
                            + (1.0 - self.ewma_alpha) * previous
                        )
                    self._outstanding[tenant] -= 1
                    self._condition.notify_all()

    @property
    def pending_jobs(self) -> int:
        return self._queued_jobs

    def service_snapshot(self) -> dict[str, float]:
        return dict(self._service_s)


_INSTALLED = False
_SCHEDULER: BoundedPriorityMediaScheduler | BoundedMaxMinMediaScheduler | None = None
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


def install(
    max_active_jobs: int,
    max_pending_jobs: int = 0,
    *,
    admission_stage: str = "fetch_decode",
    policy: str = "priority",
) -> None:
    """Install the API-server hooks. Safe to call more than once."""
    global _INSTALLED, _SCHEDULER
    if _INSTALLED:
        return
    if admission_stage not in {"fetch_decode", "decode"}:
        raise ValueError(
            "admission_stage must be either 'fetch_decode' or 'decode'"
        )
    if policy not in {"priority", "max_min"}:
        raise ValueError("policy must be either 'priority' or 'max_min'")

    # Imports are intentionally delayed: importing this file alone must not
    # initialize vLLM or CUDA.
    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm import envs
    from vllm.multimodal.media.connector import MediaConnector
    from vllm.multimodal.media.connector import global_thread_pool
    from vllm.multimodal.media.video import VideoMediaIO
    from urllib3.util import parse_url

    scheduler_class = (
        BoundedMaxMinMediaScheduler
        if policy == "max_min"
        else BoundedPriorityMediaScheduler
    )
    _SCHEDULER = scheduler_class(max_active_jobs, max_pending_jobs)

    original_preprocess_chat = OpenAIServing._preprocess_chat
    original_load_from_url_async = MediaConnector.load_from_url_async

    @functools.wraps(original_preprocess_chat)
    async def preprocess_chat_with_scheduling_context(
        self: Any,
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        priority_token = _REQUEST_PRIORITY.set(int(getattr(request, "priority", 0)))
        tenant_token = _REQUEST_TENANT.set(
            str(getattr(request, "user", None) or "default")
        )
        try:
            return await original_preprocess_chat(self, request, *args, **kwargs)
        finally:
            _REQUEST_TENANT.reset(tenant_token)
            _REQUEST_PRIORITY.reset(priority_token)

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
        tenant = _REQUEST_TENANT.get()

        async def admit(operation: Callable[[], Awaitable[Any]]) -> Any:
            if policy == "max_min":
                assert isinstance(_SCHEDULER, BoundedMaxMinMediaScheduler)
                return await _SCHEDULER.submit(tenant, operation)
            assert isinstance(_SCHEDULER, BoundedPriorityMediaScheduler)
            return await _SCHEDULER.submit(priority, operation)

        # Native vLLM overlaps HTTP downloads through async_get_bytes and
        # performs CPU decoding in its global media thread pool. In decode-only
        # mode, retain that asynchronous fetch path and place priority
        # admission immediately before VideoMediaIO.load_bytes. Other media
        # types and non-HTTP sources retain the combined admission path.
        if admission_stage == "decode" and isinstance(media_io, VideoMediaIO):
            url_spec = parse_url(url)
            if url_spec.scheme and url_spec.scheme.startswith("http"):
                self._assert_url_in_allowed_media_domains(url_spec)
                data = await self.connection.async_get_bytes(
                    url_spec.url,
                    timeout=fetch_timeout,
                    allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
                )
                loop = asyncio.get_running_loop()

                async def decode_bytes() -> Any:
                    return await loop.run_in_executor(
                        global_thread_pool,
                        media_io.load_bytes,
                        data,
                    )

                return await admit(decode_bytes)

        async def fetch_and_decode() -> Any:
            return await original_load_from_url_async(
                self,
                url,
                media_io,
                fetch_timeout=fetch_timeout,
            )

        return await admit(fetch_and_decode)

    OpenAIServing._preprocess_chat = preprocess_chat_with_scheduling_context
    MediaConnector.load_from_url_async = load_from_url_with_priority
    _INSTALLED = True

    LOGGER.warning(
        "Enabled bounded native-media %s admission: stage=%s "
        "active=%d pending=%s",
        policy,
        admission_stage,
        max_active_jobs,
        "unbounded" if max_pending_jobs == 0 else max_pending_jobs,
    )
