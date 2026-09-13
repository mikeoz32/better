import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import cast

import pytest

from better_agent import Command, Envelope, Event, ExecutionClaims, MessageId, Origin
from better_agent.kernel import (
    CommandBinding,
    CommandBus,
    EventBus,
    EventHandler,
    ExecutionScheduler,
    InlineExecutionScheduler,
)
from better_agent.kernel.errors import DuplicateCommandBindingError, MissingCommandHandlerError
from better_agent.kernel.runtime import (
    DefaultEnvelopeFactory,
    InMemoryCommandBus,
    InMemoryEventBus,
    RuntimePump,
)
from better_agent.kernel.scheduler import CapabilityScheduler
from tests.support.kernel import RecordingEnvelopeFactory, exclusive_claims


@dataclass(frozen=True, slots=True)
class Start(Command):
    depth: int = 0


@dataclass(frozen=True, slots=True)
class Continue(Command):
    depth: int


@dataclass(frozen=True, slots=True)
class Started(Event):
    depth: int


@dataclass(frozen=True, slots=True)
class Continued(Event):
    depth: int


async def start_handler(command: Envelope[Start]) -> AsyncIterator[Event]:
    yield Started(command.payload.depth)


async def continue_handler(command: Envelope[Continue]) -> AsyncIterator[Event]:
    yield Continued(command.payload.depth)


class ScriptedEventBus(EventBus):
    def __init__(self, commands_by_event: dict[type[Event], tuple[Command, ...]]) -> None:
        self._commands_by_event = commands_by_event
        self.published: list[Envelope[Event]] = []

    def subscribe[E: Event](self, event_type: type[E], handler: EventHandler[E], /) -> None:
        pass

    async def publish[E: Event](
        self,
        event: Envelope[E],
        /,
    ) -> tuple[Command, ...]:
        self.published.append(cast(Envelope[Event], event))
        return self._commands_by_event.get(type(event.payload), ())


class ScriptedCommandBus(CommandBus):
    def __init__(self, events_by_command: dict[type[Command], tuple[Event, ...]]) -> None:
        self._events_by_command = events_by_command
        self.dispatched: list[Envelope[Command]] = []

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C], /) -> None:
        pass

    def dispatch[C: Command](self, command: Envelope[C], /) -> AsyncIterator[Event]:
        self.dispatched.append(cast(Envelope[Command], command))

        async def stream() -> AsyncIterator[Event]:
            for event in self._events_by_command.get(type(command.payload), ()):
                yield event

        return stream()


class RecordingScheduler(ExecutionScheduler):
    def __init__(self) -> None:
        self.claims: list[ExecutionClaims] = []
        self.active = 0

    def admit(self, claims: ExecutionClaims) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def admission() -> AsyncIterator[None]:
            self.claims.append(claims)
            self.active += 1
            try:
                yield
            finally:
                self.active -= 1

        return admission()


def read_claim(_: Command) -> ExecutionClaims:
    return ExecutionClaims(reads=frozenset({"session"}), exclusive=False)


@pytest.mark.asyncio
async def test_runtime_pump_uses_event_and_command_bus_protocols() -> None:
    event_bus = ScriptedEventBus({Started: (Continue(2),)})
    command_bus = ScriptedCommandBus(
        {
            Start: (Started(1),),
            Continue: (Continued(2),),
        },
    )
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())

    events = [event async for event in pump.run(Start(1), origin=Origin(component="test"))]

    assert [event.payload for event in events] == [Started(1), Continued(2)]
    assert [command.payload for command in command_bus.dispatched] == [Start(1), Continue(2)]
    assert [event.payload for event in event_bus.published] == [Started(1), Continued(2)]
    assert all(event.origin == Origin(component="test") for event in events)


@pytest.mark.asyncio
async def test_event_bus_fans_out_in_registration_order_and_collects_commands() -> None:
    bus = InMemoryEventBus()
    calls: list[str] = []

    def first(event: Envelope[Started]) -> tuple[Command, ...]:
        calls.append(f"first:{event.payload.depth}")
        return (Continue(event.payload.depth + 1),)

    def second(event: Envelope[Started]) -> tuple[Command, ...]:
        calls.append(f"second:{event.payload.depth}")
        return (Continue(event.payload.depth + 2),)

    bus.subscribe(Started, first)
    bus.subscribe(Started, second)
    event = DefaultEnvelopeFactory().create(Started(1), origin=Origin(component="test"))

    commands = await bus.publish(event)

    assert calls == ["first:1", "second:1"]
    assert commands == (Continue(2), Continue(3))


@pytest.mark.asyncio
async def test_event_bus_allows_events_without_subscribers() -> None:
    bus = InMemoryEventBus()
    event = DefaultEnvelopeFactory().create(Started(1), origin=Origin(component="test"))

    assert await bus.publish(event) == ()


@pytest.mark.asyncio
async def test_command_bus_derives_claims_once_and_holds_admission_for_stream() -> None:
    scheduler = RecordingScheduler()
    bus = InMemoryCommandBus(scheduler)
    release = asyncio.Event()

    async def streaming_handler(_: Envelope[Start]) -> AsyncIterator[Event]:
        yield Started(1)
        await release.wait()
        yield Continued(2)

    bus.bind(Start, CommandBinding(streaming_handler, read_claim))
    command = DefaultEnvelopeFactory().create(Start(), origin=Origin(component="test"))
    events = bus.dispatch(command)

    assert await anext(events) == Started(1)
    assert scheduler.claims == [read_claim(Start())]
    assert scheduler.active == 1

    release.set()
    assert await anext(events) == Continued(2)
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    assert scheduler.active == 0


@pytest.mark.asyncio
async def test_command_bus_uses_conservative_claim_when_planner_is_missing() -> None:
    scheduler = RecordingScheduler()
    bus = InMemoryCommandBus(scheduler)
    bus.bind(Start, CommandBinding(start_handler))
    command = DefaultEnvelopeFactory().create(Start(), origin=Origin(component="test"))

    assert [event async for event in bus.dispatch(command)] == [Started(0)]
    assert scheduler.claims == [ExecutionClaims()]


@pytest.mark.asyncio
async def test_command_bus_requires_exactly_one_binding() -> None:
    bus = InMemoryCommandBus(InlineExecutionScheduler())
    binding = CommandBinding(handler=start_handler, execution=exclusive_claims)
    bus.bind(Start, binding)

    with pytest.raises(DuplicateCommandBindingError):
        bus.bind(Start, binding)

    command = DefaultEnvelopeFactory().create(Continue(1), origin=Origin(component="test"))
    with pytest.raises(MissingCommandHandlerError):
        bus.dispatch(command)


@pytest.mark.asyncio
async def test_runtime_pump_forwards_first_streamed_event_before_handler_finishes() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())
    release = asyncio.Event()
    finished = False

    async def streaming_handler(_: Envelope[Start]) -> AsyncIterator[Event]:
        nonlocal finished
        yield Started(1)
        await release.wait()
        finished = True
        yield Continued(2)

    command_bus.bind(Start, CommandBinding(streaming_handler, exclusive_claims))
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())
    events = pump.run(Start(), origin=Origin(component="test"))

    try:
        first = await anext(events)
        assert first.payload == Started(1)
        assert finished is False
    finally:
        release.set()
        await events.aclose()


@pytest.mark.asyncio
async def test_handler_failure_is_reported_on_next_read_not_at_public_yield() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())
    fail_handler = asyncio.Event()
    failure_observed = asyncio.Event()

    async def failing_handler(_: Envelope[Start]) -> AsyncIterator[Event]:
        try:
            yield Started(1)
            await fail_handler.wait()
            raise ValueError("handler failed")
        finally:
            failure_observed.set()

    command_bus.bind(Start, CommandBinding(failing_handler, exclusive_claims))
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())
    events = pump.run(Start(), origin=Origin(component="test"))

    assert (await anext(events)).payload == Started(1)
    fail_handler.set()
    consumer_await_completed = False
    try:
        await failure_observed.wait()
        consumer_await_completed = True
    except asyncio.CancelledError:
        pytest.fail("handler failure cancelled the consumer between yields")
    assert consumer_await_completed is True
    with pytest.raises(ExceptionGroup) as error:
        await anext(events)
    assert isinstance(error.value.exceptions[0], ValueError)
    assert str(error.value.exceptions[0]) == "handler failed"
    await events.aclose()


@pytest.mark.asyncio
async def test_runtime_pump_overlaps_compatible_commands_through_real_dispatch_path() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus(CapabilityScheduler())
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def concurrent_handler(command: Envelope[Continue]) -> AsyncIterator[Event]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            started.set()
        try:
            await release.wait()
            yield Continued(command.payload.depth)
        finally:
            active -= 1

    def start_reaction(_: Envelope[Started]) -> tuple[Command, ...]:
        return Continue(1), Continue(2)

    event_bus.subscribe(Started, start_reaction)
    command_bus.bind(Start, CommandBinding(start_handler, read_claim))
    command_bus.bind(Continue, CommandBinding(concurrent_handler, read_claim))
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())
    events = pump.run(Start(), origin=Origin(component="test"))

    assert (await anext(events)).payload == Started(0)
    await wait_for_event(started)
    assert active == 2
    release.set()
    remaining = [event async for event in events]

    assert {event.payload for event in remaining} == {Continued(1), Continued(2)}
    assert max_active == 2


@pytest.mark.asyncio
async def test_runtime_pump_reenters_concurrent_events_in_emission_order() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus(CapabilityScheduler())
    first_release = asyncio.Event()
    second_release = asyncio.Event()
    both_started = asyncio.Event()
    started_count = 0

    async def concurrent_handler(command: Envelope[Continue]) -> AsyncIterator[Event]:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            both_started.set()
        await (first_release if command.payload.depth == 1 else second_release).wait()
        yield Continued(command.payload.depth)

    event_bus.subscribe(Started, lambda _: (Continue(1), Continue(2)))
    command_bus.bind(Start, CommandBinding(start_handler, read_claim))
    command_bus.bind(Continue, CommandBinding(concurrent_handler, read_claim))
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())
    events = pump.run(Start(), origin=Origin(component="test"))

    assert (await anext(events)).payload == Started(0)
    await wait_for_event(both_started)
    second_release.set()
    assert (await anext(events)).payload == Continued(2)
    first_release.set()
    assert (await anext(events)).payload == Continued(1)
    with pytest.raises(StopAsyncIteration):
        await anext(events)


@pytest.mark.asyncio
async def test_runtime_waits_for_all_event_reactions_before_running_commands() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())
    factory = RecordingEnvelopeFactory()
    log: list[str] = []

    def first(event: Envelope[Started]) -> tuple[Command, ...]:
        log.append("first-reaction")
        return (Continue(event.payload.depth),)

    def second(_: Envelope[Started]) -> tuple[Command, ...]:
        log.append("second-reaction")
        return ()

    async def continue_and_record(_: Envelope[Continue]) -> AsyncIterator[Event]:
        log.append("command-handler")
        yield Continued(1)

    event_bus.subscribe(Started, first)
    event_bus.subscribe(Started, second)
    command_bus.bind(Start, CommandBinding(start_handler, exclusive_claims))
    command_bus.bind(Continue, CommandBinding(continue_and_record, exclusive_claims))
    pump = RuntimePump(event_bus, command_bus, factory)

    events = [event async for event in pump.run(Start(), origin=Origin(component="test"))]

    assert log == ["first-reaction", "second-reaction", "command-handler"]
    assert [event.payload for event in events] == [Started(0), Continued(1)]
    assert events[0].correlation_id == events[1].correlation_id
    assert events[0].causation_id == MessageId("message-1")
    assert events[1].causation_id == MessageId("message-3")
    assert all(event.origin == Origin(component="test") for event in events)


@pytest.mark.asyncio
async def test_runtime_pump_aclose_cancels_owned_command_tasks() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus(CapabilityScheduler())
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def cancellable_handler(_: Envelope[Start]) -> AsyncIterator[Event]:
        try:
            started.set()
            yield Started(1)
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    command_bus.bind(Start, CommandBinding(cancellable_handler, read_claim))
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())
    events = pump.run(Start(), origin=Origin(component="test"))

    assert (await anext(events)).payload == Started(1)
    await wait_for_event(started)
    await events.aclose()
    await wait_for_event(cancelled)


@pytest.mark.asyncio
async def test_runtime_pump_handles_deep_event_command_chains_without_recursion() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())
    factory = DefaultEnvelopeFactory()
    reaction_depth = 0
    max_reaction_depth = 0

    def continue_chain(event: Envelope[Continued]) -> tuple[Command, ...]:
        nonlocal reaction_depth, max_reaction_depth
        reaction_depth += 1
        max_reaction_depth = max(max_reaction_depth, reaction_depth)
        reaction_depth -= 1
        if event.payload.depth == 0:
            return ()
        return (Continue(event.payload.depth - 1),)

    async def chain_handler(command: Envelope[Continue]) -> AsyncIterator[Event]:
        yield Continued(command.payload.depth)

    event_bus.subscribe(Continued, continue_chain)
    command_bus.bind(Continue, CommandBinding(chain_handler, exclusive_claims))
    pump = RuntimePump(event_bus, command_bus, factory)

    events = [
        event
        async for event in pump.run(Continue(2_000), origin=Origin(component="test"))
    ]

    assert len(events) == 2_001
    assert max_reaction_depth == 1


async def wait_for_event(event: asyncio.Event) -> None:
    async with asyncio.timeout(1):
        await event.wait()
