#!/usr/bin/env python3
"""Small dependency-free checks for bounded priority media admission."""

from __future__ import annotations

import asyncio

from vllm_media_priority_plugin import BoundedPriorityMediaScheduler
from vllm_media_priority_plugin import BoundedMaxMinMediaScheduler
from vllm_media_priority_plugin import extract_frame_budget


async def _check_priority_and_non_preemption() -> None:
    scheduler = BoundedPriorityMediaScheduler(max_active_jobs=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first_background() -> str:
        order.append("background-running")
        first_started.set()
        await release_first.wait()
        return "background-running"

    async def operation(name: str) -> str:
        order.append(name)
        return name

    first = asyncio.create_task(scheduler.submit(10, first_background))
    await first_started.wait()
    queued_background = asyncio.create_task(
        scheduler.submit(10, lambda: operation("background-queued"))
    )
    urgent = asyncio.create_task(scheduler.submit(0, lambda: operation("urgent")))
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.gather(first, queued_background, urgent)

    assert order == ["background-running", "urgent", "background-queued"], order


async def _check_fetch_overlap_before_decode_priority() -> None:
    scheduler = BoundedPriorityMediaScheduler(max_active_jobs=1)
    running_decode_started = asyncio.Event()
    release_running_decode = asyncio.Event()
    all_fetches_finished = asyncio.Event()
    fetched: list[str] = []
    decoded: list[str] = []

    async def fetch_then_decode(name: str, priority: int) -> str:
        # Fetching occurs outside the scheduler, so all three requests can
        # finish their asynchronous I/O while one decode remains active.
        fetched.append(name)
        if len(fetched) == 3:
            all_fetches_finished.set()

        async def decode() -> str:
            decoded.append(name)
            if name == "background-running":
                running_decode_started.set()
                await release_running_decode.wait()
            return name

        return await scheduler.submit(priority, decode)

    running = asyncio.create_task(fetch_then_decode("background-running", 10))
    await running_decode_started.wait()
    background = asyncio.create_task(fetch_then_decode("background-queued", 10))
    urgent = asyncio.create_task(fetch_then_decode("urgent", 0))
    await all_fetches_finished.wait()
    assert decoded == ["background-running"]
    release_running_decode.set()
    await asyncio.gather(running, background, urgent)
    assert decoded == ["background-running", "urgent", "background-queued"]


async def _check_max_min_selects_least_served_tenant() -> None:
    scheduler = BoundedMaxMinMediaScheduler(max_active_jobs=1)
    blocker_started = asyncio.Event()
    release_blocker = asyncio.Event()
    order: list[str] = []

    async def blocker() -> str:
        blocker_started.set()
        await release_blocker.wait()
        return "blocker"

    async def operation(name: str) -> str:
        order.append(name)
        return name

    running = asyncio.create_task(scheduler.submit("blocker", blocker))
    await blocker_started.wait()
    tenant_a = asyncio.create_task(
        scheduler.submit("a", lambda: operation("a"))
    )
    tenant_b = asyncio.create_task(
        scheduler.submit("b", lambda: operation("b"))
    )
    await asyncio.sleep(0)
    # Model prior observed service: a has already consumed more preparation.
    scheduler._service_s["a"] = 5.0
    scheduler._service_s["b"] = 1.0
    release_blocker.set()
    await asyncio.gather(running, tenant_a, tenant_b)
    assert order == ["b", "a"], order


def main() -> None:
    clean, frames = extract_frame_budget(
        "http://127.0.0.1:8090/video.mp4?token=x&vllm_num_frames=128"
    )
    assert clean == "http://127.0.0.1:8090/video.mp4?token=x"
    assert frames == 128
    asyncio.run(_check_priority_and_non_preemption())
    asyncio.run(_check_fetch_overlap_before_decode_priority())
    asyncio.run(_check_max_min_selects_least_served_tenant())
    print("bounded priority and max-min schedulers: PASS")


if __name__ == "__main__":
    main()
