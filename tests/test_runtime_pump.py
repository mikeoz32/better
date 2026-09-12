import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from better_agent import Command, Envelope, Event, MessageId, Origin
from better_agent.kernel import CommandBinding
from better_agent.kernel.errors import (
    DuplicateCommandBindingError,
    MissingCommandHandlerError,
)
from better_agent.kernel.runtime import (
    DefaultEnvelopeFactory,
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

    bus.subscribe(Started, first, origin=Origin(component="first-subscriber"))
    bus.subscribe(Started, second, origin=Origin(component="second-subscriber"))
    event = DefaultEnvelopeFactory().create(
        Started(1),
        origin=Origin(component="test"),
    )

    commands = await bus.publish(event)

    assert calls == ["first:1", "second:1"]
    assert [command.payload for command in commands] == [Continue(2), Continue(3)]
    assert [command.origin for command in commands] == [
        Origin(component="first-subscriber"),
        Origin(component="second-subscriber"),
    ]


@pytest.mark.asyncio
async def test_event_bus_allows_events_without_subscribers() -> None:
    event = DefaultEnvelopeFactory().create(
        Started(1),
        origin=Origin(component="test"),
    )

    assert await InMemoryEventBus().publish(event) == ()


@pytest.mark.asyncio
async def test_command_bus_requires_exactly_one_binding() -> None:
    bus = InMemoryCommandBus()
    binding = CommandBinding(
        handler=start_handler,
        execution=exclusive_claims,
        origin=Origin(component="start-handler"),
    )
    bus.bind(Start, binding)

    with pytest.raises(DuplicateCommandBindingError):
        bus.bind(Start, binding)

    with pytest.raises(MissingCommandHandlerError):
        async for _ in bus.dispatch(
            DefaultEnvelopeFactory().create(
                Continue(1),
                origin=Origin(component="test"),
            )
        ):
            pass


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
            origin=Origin(component="stream-handler"),
        ),
    )
    command = DefaultEnvelopeFactory().create(
        Start(),
        origin=Origin(component="test"),
    )

    events = bus.dispatch(command)
    first = await anext(events)

    assert first.payload == Started(1)
    assert first.origin == Origin(component="stream-handler")
    assert finished is False

    release.set()
    second = await anext(events)
    assert second.payload == Continued(2)


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
            origin=Origin(component="stream-handler"),
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

    event_bus.subscribe(Started, first, origin=Origin(component="first-subscriber"))
    event_bus.subscribe(Started, second, origin=Origin(component="second-subscriber"))
    command_bus.bind(
        Start,
        CommandBinding(
            start_handler,
            exclusive_claims,
            origin=Origin(component="start-handler"),
        ),
    )
    command_bus.bind(
        Continue,
        CommandBinding(
            continue_and_record,
            exclusive_claims,
            origin=Origin(component="continue-handler"),
        ),
    )
    pump = RuntimePump(event_bus, command_bus, factory)

    events = [event async for event in pump.run(Start(), origin=Origin(component="test"))]

    assert log == ["first-reaction", "second-reaction", "command-handler"]
    assert [event.payload for event in events] == [Started(0), Continued(1)]
    assert events[0].correlation_id == events[1].correlation_id
    assert events[0].causation_id == MessageId("message-1")
    assert events[1].causation_id == MessageId("message-3")
    assert events[0].origin == Origin(component="start-handler")
    assert events[1].origin == Origin(component="continue-handler")


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

    event_bus.subscribe(Continued, continue_chain, origin=Origin(component="chain-subscriber"))
    command_bus.bind(
        Continue,
        CommandBinding(
            chain_handler,
            exclusive_claims,
            origin=Origin(component="chain-handler"),
        ),
    )
    pump = RuntimePump(event_bus, command_bus, factory)

    events = [
        event
        async for event in pump.run(Continue(2_000), origin=Origin(component="test"))
    ]

    assert len(events) == 2_001
    assert max_reaction_depth == 1
