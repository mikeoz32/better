import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

import pytest

from better_agent import Command, Envelope, Event, MessageId, Origin
from better_agent.kernel import CommandBinding, CommandBus, EventBus, EventHandler
from better_agent.kernel.errors import (
    DuplicateCommandBindingError,
    MissingCommandHandlerError,
)
from better_agent.kernel.runtime import (
    DefaultEnvelopeFactory,
    emit_command,
    emit_event,
    InMemoryCommandBus,
    InMemoryEventBus,
    RuntimePump,
)
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

    async def publish[E: Event](self, event: Envelope[E], /) -> None:
        self.published.append(cast(Envelope[Event], event))
        for command in self._commands_by_event.get(type(event.payload), ()):
            emit_command(command, origin=Origin(component="scripted-event-bus"))


class ScriptedCommandBus(CommandBus):
    def __init__(self, events_by_command: dict[type[Command], tuple[Event, ...]]) -> None:
        self._events_by_command = events_by_command
        self.dispatched: list[Envelope[Command]] = []

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C], /) -> None:
        pass

    async def dispatch[C: Command](self, command: Envelope[C], /) -> None:
        self.dispatched.append(cast(Envelope[Command], command))
        for event in self._events_by_command.get(type(command.payload), ()):
            await emit_event(event, origin=Origin(component="scripted-command-bus"))


@pytest.mark.asyncio
async def test_runtime_pump_uses_event_and_command_bus_protocols() -> None:
    event_bus = ScriptedEventBus({Started: (Continue(2),)})
    command_bus = ScriptedCommandBus(
        {
            Start: (Started(1),),
            Continue: (Continued(2),),
        }
    )
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())

    events = [event async for event in pump.run(Start(1), origin=Origin(component="test"))]

    assert [event.payload for event in events] == [Started(1), Continued(2)]
    assert [command.payload for command in command_bus.dispatched] == [Start(1), Continue(2)]
    assert [event.payload for event in event_bus.published] == [Started(1), Continued(2)]


@pytest.mark.asyncio
async def test_event_bus_fans_out_in_registration_order_and_collects_commands() -> None:
    bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus()
    calls: list[str] = []
    command_origins: list[str] = []

    def first(event: Envelope[Started]) -> tuple[Command, ...]:
        calls.append(f"first:{event.payload.depth}")
        return (Continue(event.payload.depth + 1),)

    def second(event: Envelope[Started]) -> tuple[Command, ...]:
        calls.append(f"second:{event.payload.depth}")
        return (Continue(event.payload.depth + 2),)

    async def continue_handler(command: Envelope[Continue]) -> AsyncIterator[Event]:
        command_origins.append(command.origin.component)
        yield Continued(command.payload.depth)

    bus.subscribe(Started, first)
    bus.subscribe(Started, second)
    command_bus.bind(
        Start,
        CommandBinding(start_handler, exclusive_claims),
    )
    command_bus.bind(
        Continue,
        CommandBinding(continue_handler, exclusive_claims),
    )
    pump = RuntimePump(bus, command_bus, DefaultEnvelopeFactory())
    events = [event async for event in pump.run(Start(1), origin=Origin(component="test"))]

    assert calls == ["first:1", "second:1"]
    assert [event.payload for event in events] == [Started(1), Continued(2), Continued(3)]
    assert len(command_origins) == 2
    assert command_origins[0].endswith(".first")
    assert command_origins[1].endswith(".second")


@pytest.mark.asyncio
async def test_event_bus_allows_events_without_subscribers() -> None:
    command_bus = InMemoryCommandBus()
    command_bus.bind(Start, CommandBinding(start_handler, exclusive_claims))
    pump = RuntimePump(InMemoryEventBus(), command_bus, DefaultEnvelopeFactory())

    events = [event async for event in pump.run(Start(1), origin=Origin(component="test"))]

    assert [event.payload for event in events] == [Started(1)]


@pytest.mark.asyncio
async def test_command_bus_requires_exactly_one_binding() -> None:
    bus = InMemoryCommandBus()
    binding = CommandBinding(
        handler=start_handler,
        execution=exclusive_claims,
    )
    bus.bind(Start, binding)

    with pytest.raises(DuplicateCommandBindingError):
        bus.bind(Start, binding)

    with pytest.raises(MissingCommandHandlerError):
        pump = RuntimePump(InMemoryEventBus(), bus, DefaultEnvelopeFactory())
        await anext(pump.run(Continue(1), origin=Origin(component="test")))


@pytest.mark.asyncio
async def test_command_bus_streams_events_before_handler_completion() -> None:
    bus = InMemoryCommandBus()
    release = asyncio.Event()
    finished = False

    async def streaming_handler(_: Envelope[Start]) -> AsyncIterator[Event]:
        nonlocal finished
        yield Started(1)
        await release.wait()
        finished = True
        yield Continued(2)

    bus.bind(
        Start,
        CommandBinding(
            streaming_handler,
            exclusive_claims,
        ),
    )
    command = DefaultEnvelopeFactory().create(
        Start(),
        origin=Origin(component="test"),
    )

    pump = RuntimePump(InMemoryEventBus(), bus, DefaultEnvelopeFactory())
    events = pump.run_envelope(command)
    first = await anext(events)

    assert first.payload == Started(1)
    assert first.origin.component.endswith("streaming_handler")
    assert finished is False

    release.set()
    second = await anext(events)
    assert second.payload == Continued(2)
    await events.aclose()


@pytest.mark.asyncio
async def test_runtime_pump_forwards_first_streamed_event_before_handler_finishes() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus()
    release = asyncio.Event()
    finished = False

    async def streaming_handler(_: Envelope[Start]) -> AsyncIterator[Event]:
        nonlocal finished
        yield Started(1)
        await release.wait()
        finished = True
        yield Continued(2)

    command_bus.bind(
        Start,
        CommandBinding(
            streaming_handler,
            exclusive_claims,
        ),
    )
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())
    events = pump.run(Start(), origin=Origin(component="test"))

    try:
        async with asyncio.timeout(1):
            first = await anext(events)
    finally:
        release.set()
        await events.aclose()

    assert first.payload == Started(1)
    assert finished is False


@pytest.mark.asyncio
async def test_runtime_pump_applies_backpressure_to_streamed_events() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus()
    pulled = 0

    async def streaming_handler(_: Envelope[Start]) -> AsyncIterator[Event]:
        nonlocal pulled
        pulled += 1
        yield Started(1)
        pulled += 1
        yield Continued(2)

    command_bus.bind(Start, CommandBinding(streaming_handler, exclusive_claims))
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory())
    events = pump.run(Start(), origin=Origin(component="test"))

    first = await anext(events)
    assert first.payload == Started(1)
    assert pulled == 1

    second = await anext(events)
    assert second.payload == Continued(2)
    assert pulled == 2

    await events.aclose()


@pytest.mark.asyncio
async def test_runtime_waits_for_all_event_reactions_before_running_commands() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus()
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
    command_bus.bind(
        Start,
        CommandBinding(
            start_handler,
            exclusive_claims,
        ),
    )
    command_bus.bind(
        Continue,
        CommandBinding(
            continue_and_record,
            exclusive_claims,
        ),
    )
    pump = RuntimePump(event_bus, command_bus, factory)

    events = [event async for event in pump.run(Start(), origin=Origin(component="test"))]

    assert log == ["first-reaction", "second-reaction", "command-handler"]
    assert [event.payload for event in events] == [Started(0), Continued(1)]
    assert events[0].correlation_id == events[1].correlation_id
    assert events[0].causation_id == MessageId("message-1")
    assert events[1].causation_id == MessageId("message-3")
    assert events[0].origin.component.endswith("start_handler")
    assert events[1].origin.component.endswith("continue_and_record")


@pytest.mark.asyncio
async def test_runtime_pump_handles_deep_event_command_chains_without_recursion() -> None:
    event_bus = InMemoryEventBus()
    command_bus = InMemoryCommandBus()
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
    command_bus.bind(
        Continue,
        CommandBinding(
            chain_handler,
            exclusive_claims,
        ),
    )
    pump = RuntimePump(event_bus, command_bus, factory)

    events = [
        event
        async for event in pump.run(Continue(2_000), origin=Origin(component="test"))
    ]

    assert len(events) == 2_001
    assert max_reaction_depth == 1
