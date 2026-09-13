import asyncio
from collections.abc import Awaitable, Callable

import pytest

from better_agent import ExecutionClaims, InlineExecutionScheduler
from better_agent.kernel.scheduler import CapabilityScheduler


def read(resource: str) -> ExecutionClaims:
    return ExecutionClaims(reads=frozenset({resource}), exclusive=False)


def write(resource: str) -> ExecutionClaims:
    return ExecutionClaims(writes=frozenset({resource}), exclusive=False)


async def wait_for_started(started: asyncio.Event) -> None:
    async with asyncio.timeout(1):
        await started.wait()


async def noop() -> None:
    return None


@pytest.mark.asyncio
async def test_same_resource_reads_run_concurrently() -> None:
    scheduler = CapabilityScheduler()
    started = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def operation(index: int) -> int:
        started[index].set()
        await release.wait()
        return index

    tasks = [
        asyncio.create_task(
            _run_admitted(
                scheduler,
                read("session"),
                lambda index=index: operation(index),
            ),
        )
        for index in range(2)
    ]

    await asyncio.gather(*(wait_for_started(event) for event in started))
    release.set()

    assert await asyncio.gather(*tasks) == [0, 1]


@pytest.mark.asyncio
async def test_read_write_conflict_preserves_fifo_order() -> None:
    scheduler = CapabilityScheduler()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    third_started = asyncio.Event()
    unrelated_started = asyncio.Event()
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

    async def unrelated() -> None:
        order.append("unrelated")
        unrelated_started.set()

    first_task = asyncio.create_task(_run_admitted(scheduler, write("db"), first))
    await wait_for_started(first_started)
    second_task = asyncio.create_task(_run_admitted(scheduler, read("db"), second))
    third_task = asyncio.create_task(_run_admitted(scheduler, write("db"), third))
    unrelated_task = asyncio.create_task(
        _run_admitted(scheduler, read("network"), unrelated),
    )

    await wait_for_started(unrelated_started)
    assert not second_started.is_set()
    assert not third_started.is_set()
    first_release.set()
    await wait_for_started(second_started)
    assert not third_started.is_set()
    second_release.set()

    await asyncio.gather(first_task, second_task, third_task, unrelated_task)

    assert order == ["first", "unrelated", "second", "third"]


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

    active_task = asyncio.create_task(_run_admitted(scheduler, write("db"), active))
    await wait_for_started(active_started)
    blocked_task = asyncio.create_task(_run_admitted(scheduler, write("db"), blocked))
    unrelated_task = asyncio.create_task(
        _run_admitted(scheduler, read("network"), unrelated),
    )

    await wait_for_started(unrelated_started)
    assert not blocked_started.is_set()
    active_release.set()
    await wait_for_started(blocked_started)
    blocked_release.set()

    await asyncio.gather(active_task, blocked_task, unrelated_task)


@pytest.mark.asyncio
async def test_waiting_cancellation_reschedules_younger_claim() -> None:
    class ObservableScheduler(CapabilityScheduler):
        def __init__(self) -> None:
            super().__init__()
            self.admission_requests = 0
            self.third_request = asyncio.Event()

        def admit(self, claims: ExecutionClaims):
            self.admission_requests += 1
            if self.admission_requests == 3:
                self.third_request.set()
            return super().admit(claims)

    scheduler = ObservableScheduler()
    active_started = asyncio.Event()
    younger_started = asyncio.Event()
    active_release = asyncio.Event()

    async def active() -> None:
        active_started.set()
        await active_release.wait()

    async def older() -> None:
        raise AssertionError("cancelled waiting request must not start")

    async def younger() -> None:
        younger_started.set()

    active_task = asyncio.create_task(_run_admitted(scheduler, write("x"), active))
    await wait_for_started(active_started)
    older_task = asyncio.create_task(
        _run_admitted(
            scheduler,
            ExecutionClaims(
                reads=frozenset({"x", "y"}),
                exclusive=False,
            ),
            older,
        ),
    )
    younger_task = asyncio.create_task(_run_admitted(scheduler, write("y"), younger))
    await wait_for_started(scheduler.third_request)

    older_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await older_task
    await wait_for_started(younger_started)

    active_release.set()
    await asyncio.gather(active_task, younger_task)


@pytest.mark.asyncio
async def test_unknown_claims_block_known_claims_conservatively() -> None:
    class ObservableScheduler(CapabilityScheduler):
        def __init__(self) -> None:
            super().__init__()
            self.admission_requests = 0
            self.second_request = asyncio.Event()

        def admit(self, claims: ExecutionClaims):
            self.admission_requests += 1
            if self.admission_requests == 2:
                self.second_request.set()
            return super().admit(claims)

    scheduler = ObservableScheduler()
    unknown_started = asyncio.Event()
    known_started = asyncio.Event()
    release = asyncio.Event()

    async def unknown() -> None:
        unknown_started.set()
        await release.wait()

    async def known() -> None:
        known_started.set()

    unknown_task = asyncio.create_task(
        _run_admitted(scheduler, ExecutionClaims(), unknown),
    )
    await wait_for_started(unknown_started)
    known_task = asyncio.create_task(_run_admitted(scheduler, read("model"), known))
    await wait_for_started(scheduler.second_request)
    assert not known_started.is_set()
    release.set()
    await asyncio.gather(unknown_task, known_task)
    assert known_started.is_set()


@pytest.mark.asyncio
async def test_waiting_cancellation_removes_request() -> None:
    scheduler = CapabilityScheduler()
    active_started = asyncio.Event()
    unrelated_started = asyncio.Event()
    active_release = asyncio.Event()
    blocked_started = False

    async def active() -> None:
        active_started.set()
        await active_release.wait()

    async def blocked() -> None:
        nonlocal blocked_started
        blocked_started = True

    async def unrelated() -> None:
        unrelated_started.set()

    active_task = asyncio.create_task(_run_admitted(scheduler, write("db"), active))
    await wait_for_started(active_started)
    blocked_task = asyncio.create_task(_run_admitted(scheduler, write("db"), blocked))
    unrelated_task = asyncio.create_task(
        _run_admitted(scheduler, read("network"), unrelated),
    )
    await wait_for_started(unrelated_started)

    blocked_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked_task
    active_release.set()
    await active_task
    await unrelated_task

    await _run_admitted(scheduler, write("db"), noop)
    assert blocked_started is False


@pytest.mark.asyncio
async def test_active_cancellation_releases_claims() -> None:
    scheduler = CapabilityScheduler()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def active() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(_run_admitted(scheduler, write("db"), active))
    await wait_for_started(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_for_started(cancelled)

    await _run_admitted(scheduler, write("db"), noop)


@pytest.mark.asyncio
async def test_scheduler_can_be_replaced_by_a_deterministic_double() -> None:
    class RecordingScheduler:
        def __init__(self) -> None:
            self.claims: list[ExecutionClaims] = []

        def admit(self, claims: ExecutionClaims):
            self.claims.append(claims)

            class Admission:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, exc_type, exc, traceback):
                    return False

            return Admission()

    scheduler = RecordingScheduler()
    claims = read("model")

    async with scheduler.admit(claims):
        result = "ok"

    assert result == "ok"
    assert scheduler.claims == [claims]


@pytest.mark.asyncio
async def test_inline_scheduler_is_a_deterministic_replacement() -> None:
    scheduler = InlineExecutionScheduler()

    async with scheduler.admit(ExecutionClaims()):
        result = "ok"

    assert result == "ok"


async def _run_admitted[T](
    scheduler: CapabilityScheduler,
    claims: ExecutionClaims,
    operation: Callable[[], Awaitable[T]],
) -> T:
    async with scheduler.admit(claims):
        return await operation()
