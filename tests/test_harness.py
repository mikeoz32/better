import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from better_agent import (
    Command,
    CommandBinding,
    CommitRunOutcome,
    CompletedOutcome,
    DefaultEnvelopeFactory,
    Event,
    FailedOutcome,
    Harness,
    InlineExecutionScheduler,
    Origin,
    RunCompleted,
    RunFailed,
    RunFinalizationError,
    RunLimits,
    RunOutcomeReady,
    RunStarted,
    RunCancelled,
    RuntimeFault,
    RuntimeFaulted,
    StepBudgetExceeded,
)
from better_agent.kernel.runtime import InMemoryCommandBus, InMemoryEventBus


@dataclass(frozen=True, slots=True)
class StartRun(Command):
    name: str = "run"


@dataclass(frozen=True, slots=True)
class SpawnChild(Command):
    value: int


@dataclass(frozen=True, slots=True)
class WorkProduced(Event):
    value: int


@dataclass(frozen=True, slots=True)
class DomainFailure(Event):
    message: str


async def one_work_event(_: object) -> AsyncIterator[Event]:
    yield WorkProduced(1)


def make_harness(
    root_handler,
    commit_handler,
    *,
    event_bus: InMemoryEventBus | None = None,
) -> Harness:
    event_bus = event_bus or InMemoryEventBus()
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())
    command_bus.bind(StartRun, CommandBinding(root_handler))
    command_bus.bind(CommitRunOutcome, CommandBinding(commit_handler))
    return Harness(event_bus, command_bus, DefaultEnvelopeFactory())


@pytest.mark.asyncio
async def test_harness_streams_run_started_and_committed_completion() -> None:
    commit_started = asyncio.Event()
    commit_after_yield = asyncio.Event()
    release_commit = asyncio.Event()

    async def commit_handler(command) -> AsyncIterator[Event]:
        commit_started.set()
        yield RunCompleted(command.payload.outcome)
        commit_after_yield.set()
        await release_commit.wait()

    harness = make_harness(one_work_event, commit_handler)
    stream = harness.run(StartRun(), origin=Origin(component="test"))

    started = await anext(stream)
    work = await anext(stream)
    ready = await anext(stream)
    assert isinstance(started.payload, RunStarted)
    assert work.payload == WorkProduced(1)
    assert isinstance(ready.payload, RunOutcomeReady)

    terminal_task = asyncio.create_task(anext(stream))
    await commit_started.wait()
    await commit_after_yield.wait()
    assert not terminal_task.done()
    release_commit.set()
    terminal = await terminal_task

    assert isinstance(terminal.payload, RunCompleted)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_unexpected_handler_error_becomes_runtime_fault_and_failed_outcome() -> None:
    async def failing_handler(_: object) -> AsyncIterator[Event]:
        raise ValueError("model exploded")
        yield WorkProduced(0)

    async def commit_handler(command) -> AsyncIterator[Event]:
        yield RunFailed(command.payload.outcome)

    harness = make_harness(failing_handler, commit_handler)
    events = [
        event
        async for event in harness.run(
            StartRun(),
            origin=Origin(component="test"),
        )
    ]

    assert [type(event.payload) for event in events] == [
        RunStarted,
        RuntimeFaulted,
        RunOutcomeReady,
        RunFailed,
    ]
    fault_event = events[1]
    ready_event = events[2]
    assert isinstance(fault_event.payload, RuntimeFaulted)
    assert isinstance(ready_event.payload, RunOutcomeReady)
    assert fault_event.payload.fault.exception_type == "ValueError"
    assert fault_event.payload.fault.message == "model exploded"
    assert isinstance(ready_event.payload.outcome, FailedOutcome)
    assert isinstance(ready_event.payload.outcome.reason, RuntimeFault)
    assert ready_event.payload.outcome.reason.exception_type == "ValueError"
    assert fault_event.causation_id == events[0].id


@pytest.mark.asyncio
async def test_semantic_cancellation_finalizes_as_run_cancelled() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_handler(_: object) -> AsyncIterator[Event]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        yield WorkProduced(0)

    async def commit_handler(command) -> AsyncIterator[Event]:
        yield RunCancelled(command.payload.outcome)

    harness = make_harness(blocking_handler, commit_handler)
    stream = harness.run(StartRun(), origin=Origin(component="test"))
    run_started = await anext(stream)
    await asyncio.wait_for(started.wait(), timeout=1)

    await harness.cancel(run_started.correlation_id)
    ready = await anext(stream)
    terminal = await anext(stream)

    assert isinstance(ready.payload, RunOutcomeReady)
    assert isinstance(terminal.payload, RunCancelled)
    assert type(ready.payload.outcome) is type(terminal.payload.outcome)
    await asyncio.wait_for(cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_consumer_cancellation_propagates_after_owned_work_cleanup() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_handler(_: object) -> AsyncIterator[Event]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        yield WorkProduced(0)

    async def commit_handler(command) -> AsyncIterator[Event]:
        yield RunCompleted(command.payload.outcome)

    harness = make_harness(blocking_handler, commit_handler)
    stream = harness.run(StartRun(), origin=Origin(component="test"))
    await anext(stream)
    await asyncio.wait_for(started.wait(), timeout=1)

    next_event = asyncio.create_task(anext(stream))
    next_event.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_event
    await asyncio.wait_for(cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_step_budget_counts_root_and_spawned_commands() -> None:
    child_started = asyncio.Event()

    async def root_handler(_: object) -> AsyncIterator[Event]:
        yield WorkProduced(1)

    async def child_handler(_: object) -> AsyncIterator[Event]:
        child_started.set()
        yield WorkProduced(2)

    async def commit_handler(command) -> AsyncIterator[Event]:
        yield RunFailed(command.payload.outcome)

    event_bus = InMemoryEventBus()
    event_bus.subscribe(WorkProduced, lambda _: (SpawnChild(2),))
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())
    command_bus.bind(StartRun, CommandBinding(root_handler))
    command_bus.bind(SpawnChild, CommandBinding(child_handler))
    command_bus.bind(CommitRunOutcome, CommandBinding(commit_handler))
    harness = Harness(event_bus, command_bus, DefaultEnvelopeFactory())

    events = [
        event
        async for event in harness.run(
            StartRun(),
            origin=Origin(component="test"),
            limits=RunLimits(max_steps=1),
        )
    ]

    assert not child_started.is_set()
    budget_events = [
        event.payload
        for event in events
        if isinstance(event.payload, StepBudgetExceeded)
    ]
    assert len(budget_events) == 1
    assert budget_events[0].limit == 1
    assert budget_events[0].attempted_step == 2
    assert isinstance(events[-1].payload, RunFailed)


@pytest.mark.asyncio
async def test_step_budget_cancels_already_started_concurrent_work() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def root_handler(_: object) -> AsyncIterator[Event]:
        yield WorkProduced(1)

    async def child_handler(_: object) -> AsyncIterator[Event]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        yield WorkProduced(2)

    async def commit_handler(command) -> AsyncIterator[Event]:
        yield RunFailed(command.payload.outcome)

    event_bus = InMemoryEventBus()
    event_bus.subscribe(WorkProduced, lambda _: (SpawnChild(1), SpawnChild(2)))
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())
    command_bus.bind(StartRun, CommandBinding(root_handler))
    command_bus.bind(SpawnChild, CommandBinding(child_handler))
    command_bus.bind(CommitRunOutcome, CommandBinding(commit_handler))
    harness = Harness(event_bus, command_bus, DefaultEnvelopeFactory())

    events = [
        event
        async for event in harness.run(
            StartRun(),
            origin=Origin(component="test"),
            limits=RunLimits(max_steps=2),
        )
    ]

    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    budget_events = [
        event.payload
        for event in events
        if isinstance(event.payload, StepBudgetExceeded)
    ]
    assert len(budget_events) == 1
    assert budget_events[0].limit == 2
    assert budget_events[0].attempted_step == 3


@pytest.mark.asyncio
async def test_event_handler_error_becomes_runtime_fault_without_exception_group_leak() -> None:
    def failing_reaction(_: object) -> tuple[Command, ...]:
        raise LookupError("reaction failed")

    async def root_handler(_: object) -> AsyncIterator[Event]:
        yield WorkProduced(1)

    async def commit_handler(command) -> AsyncIterator[Event]:
        yield RunFailed(command.payload.outcome)

    event_bus = InMemoryEventBus()
    event_bus.subscribe(WorkProduced, failing_reaction)
    harness = make_harness(root_handler, commit_handler, event_bus=event_bus)

    events = [
        event
        async for event in harness.run(
            StartRun(),
            origin=Origin(component="test"),
        )
    ]

    assert [type(event.payload) for event in events] == [
        RunStarted,
        RuntimeFaulted,
        RunOutcomeReady,
        RunFailed,
    ]
    fault = events[1].payload
    assert isinstance(fault, RuntimeFaulted)
    assert fault.fault.phase == "event_handler"


@pytest.mark.asyncio
async def test_cancellation_after_finalizing_does_not_replace_selected_outcome() -> None:
    harness = make_harness(one_work_event, lambda command: _completed(command))
    stream = harness.run(StartRun(), origin=Origin(component="test"))
    started = await anext(stream)
    await anext(stream)
    ready = await anext(stream)

    await harness.cancel(started.correlation_id)
    terminal = await anext(stream)

    assert isinstance(ready.payload, RunOutcomeReady)
    assert isinstance(ready.payload.outcome, CompletedOutcome)
    assert isinstance(terminal.payload, RunCompleted)


async def _completed(command) -> AsyncIterator[Event]:
    yield RunCompleted(command.payload.outcome)


@pytest.mark.asyncio
async def test_lifecycle_envelopes_preserve_run_correlation_and_causation() -> None:
    harness = make_harness(one_work_event, _completed)
    events = [
        event
        async for event in harness.run(
            StartRun(),
            origin=Origin(component="test"),
        )
    ]

    correlation = events[0].correlation_id
    assert all(event.correlation_id == correlation for event in events)
    assert events[0].causation_id is not None
    assert all(event.causation_id is not None for event in events[1:])
    assert events[2].causation_id == events[1].id


@pytest.mark.asyncio
async def test_typed_domain_failure_event_is_not_normalized_to_runtime_fault() -> None:
    async def root_handler(_: object) -> AsyncIterator[Event]:
        yield DomainFailure("expected")

    async def commit_handler(command) -> AsyncIterator[Event]:
        yield RunCompleted(command.payload.outcome)

    harness = make_harness(root_handler, commit_handler)
    events = [
        event
        async for event in harness.run(
            StartRun(),
            origin=Origin(component="test"),
        )
    ]

    assert any(isinstance(event.payload, DomainFailure) for event in events)
    assert not any(isinstance(event.payload, RuntimeFaulted) for event in events)
    assert isinstance(events[-1].payload, RunCompleted)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing", "mismatch", "multiple", "failure"])
async def test_invalid_commit_stream_exposes_no_terminal(mode: str) -> None:
    async def root_handler(_: object) -> AsyncIterator[Event]:
        yield WorkProduced(1)

    async def commit_handler(command) -> AsyncIterator[Event]:
        if mode == "mismatch":
            yield RunCancelled(command.payload.outcome)
        elif mode == "multiple":
            yield RunCompleted(command.payload.outcome)
            yield RunCompleted(command.payload.outcome)
        else:
            raise RuntimeError("commit failed")

    harness = make_harness(root_handler, commit_handler)
    stream = harness.run(StartRun(), origin=Origin(component="test"))
    observed = []
    with pytest.raises(RunFinalizationError, match="finalization"):
        async for event in stream:
            observed.append(event)

    assert not any(isinstance(event.payload, RunCompleted) for event in observed)
    assert not any(isinstance(event.payload, RunCancelled) for event in observed)
