#!/usr/bin/env python3
"""Small dependency-free checks for bounded priority media admission."""

from __future__ import annotations

import asyncio

from vllm_media_priority_plugin import BoundedPriorityMediaScheduler


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


def main() -> None:
    asyncio.run(_check_priority_and_non_preemption())
    print("bounded priority scheduler: PASS")


if __name__ == "__main__":
    main()
