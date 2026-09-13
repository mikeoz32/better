import asyncio
from collections.abc import Awaitable, Callable

import pytest

from better_agent import ExecutionClaims, InlineExecutionScheduler
from better_agent.kernel.scheduler import CapabilityScheduler


async def wait_for_started(started: asyncio.Event) -> None:
    async with asyncio.timeout(1):
        await started.wait()


def claim(resource: str) -> ExecutionClaims:
    return ExecutionClaims(resources=frozenset({resource}), exclusive=False)


@pytest.mark.asyncio
async def test_compatible_claims_run_concurrently() -> None:
    scheduler = CapabilityScheduler()
    started = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def operation(index: int) -> int:
        started[index].set()
        await release.wait()
        return index

    tasks = [
        asyncio.create_task(
            scheduler.run(
                claim(f"model-{index}"),
                lambda index=index: operation(index),
            ),
        )
        for index in range(2)
    ]

    await asyncio.gather(*(wait_for_started(event) for event in started))
    release.set()

    assert await asyncio.gather(*tasks) == [0, 1]


@pytest.mark.asyncio
async def test_conflicting_claims_preserve_fifo_order() -> None:
    scheduler = CapabilityScheduler()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    third_started = asyncio.Event()
    first_release = asyncio.Event()
    second_release = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        order.append("first")
        first_started.set()
        await first_release.wait()

    async def second() -> None:
        order.append("second")
        second_started.set()
        await second_release.wait()

    async def third() -> None:
        order.append("third")
        third_started.set()

    first_task = asyncio.create_task(scheduler.run(claim("db"), first))
    await wait_for_started(first_started)
    second_task = asyncio.create_task(scheduler.run(claim("db"), second))
    third_task = asyncio.create_task(scheduler.run(claim("db"), third))

    first_release.set()
    await wait_for_started(second_started)
    assert not third_started.is_set()
    second_release.set()

    await asyncio.gather(first_task, second_task, third_task)

    assert order == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_unrelated_claim_bypasses_older_blocked_claim() -> None:
    scheduler = CapabilityScheduler()
    active_started = asyncio.Event()
    blocked_started = asyncio.Event()
    unrelated_started = asyncio.Event()
    active_release = asyncio.Event()
    blocked_release = asyncio.Event()

    async def active() -> None:
        active_started.set()
        await active_release.wait()

    async def blocked() -> None:
        blocked_started.set()
        await blocked_release.wait()

    async def unrelated() -> None:
        unrelated_started.set()

    active_task = asyncio.create_task(scheduler.run(claim("db"), active))
    await wait_for_started(active_started)
    blocked_task = asyncio.create_task(scheduler.run(claim("db"), blocked))
    unrelated_task = asyncio.create_task(scheduler.run(claim("network"), unrelated))

    await wait_for_started(unrelated_started)
    assert not blocked_started.is_set()
    active_release.set()
    await wait_for_started(blocked_started)
    blocked_release.set()

    await asyncio.gather(active_task, blocked_task, unrelated_task)


@pytest.mark.asyncio
async def test_unknown_claims_block_known_claims_conservatively() -> None:
    scheduler = CapabilityScheduler()
    unknown_started = asyncio.Event()
    known_started = asyncio.Event()
    release = asyncio.Event()

    async def unknown() -> None:
        unknown_started.set()
        await release.wait()

    async def known() -> None:
        known_started.set()

    unknown_task = asyncio.create_task(scheduler.run(ExecutionClaims(), unknown))
    await wait_for_started(unknown_started)
    known_task = asyncio.create_task(scheduler.run(claim("model"), known))
    await asyncio.sleep(0)
    assert not known_started.is_set()

    release.set()
    await asyncio.gather(unknown_task, known_task)
    assert known_started.is_set()


@pytest.mark.asyncio
async def test_scheduler_releases_claims_after_failure() -> None:
    scheduler = CapabilityScheduler()
    second_started = asyncio.Event()

    async def fail() -> None:
        raise ValueError("boom")

    async def second() -> None:
        second_started.set()

    with pytest.raises(ValueError, match="boom"):
        await scheduler.run(claim("db"), fail)

    await scheduler.run(claim("db"), second)
    assert second_started.is_set()


@pytest.mark.asyncio
async def test_scheduler_can_be_replaced_by_a_deterministic_double() -> None:
    class RecordingScheduler:
        def __init__(self) -> None:
            self.claims: list[ExecutionClaims] = []

        async def run[T](
            self,
            claims: ExecutionClaims,
            operation: Callable[[], Awaitable[T]],
            /,
        ) -> T:
            self.claims.append(claims)
            return await operation()

    scheduler = RecordingScheduler()
    claims = claim("model")

    result = await scheduler.run(claims, lambda: asyncio.sleep(0, result="ok"))

    assert result == "ok"
    assert scheduler.claims == [claims]


@pytest.mark.asyncio
async def test_inline_scheduler_is_a_deterministic_replacement() -> None:
    scheduler = InlineExecutionScheduler()

    result = await scheduler.run(ExecutionClaims(), lambda: asyncio.sleep(0, result="ok"))

    assert result == "ok"
