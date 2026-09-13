import asyncio
from dataclasses import dataclass
from collections.abc import AsyncIterator

from better_agent import Command, Event, Envelope, ExecutionClaims, Origin
from better_agent.kernel import CommandBinding
from better_agent.kernel.runtime import (
    DefaultEnvelopeFactory,
    InMemoryCommandBus,
    InMemoryEventBus,
    RendezvousChannel,
    RuntimePump,
)


@dataclass(frozen=True, slots=True)
class BenchmarkCommand(Command):
    value: int


@dataclass(frozen=True, slots=True)
class BenchmarkEvent(Event):
    value: int


async def command_handler(_: Envelope[BenchmarkCommand]) -> AsyncIterator[Event]:
    yield BenchmarkEvent(1)


def exclusive_claims(_: Command) -> ExecutionClaims:
    return ExecutionClaims()


def test_runtime_pump_dispatch_benchmark(benchmark) -> None:
    channel = RendezvousChannel()
    event_bus = InMemoryEventBus(channel)
    command_bus = InMemoryCommandBus(channel)

    def subscriber(_: Envelope[BenchmarkEvent]) -> tuple[Command, ...]:
        return ()

    for _ in range(4):
        event_bus.subscribe(BenchmarkEvent, subscriber)
    command_bus.bind(
        BenchmarkCommand,
        CommandBinding(command_handler, exclusive_claims),
    )
    pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory(), channel)

    loop = asyncio.new_event_loop()
    try:
        async def dispatch() -> None:
            async for _ in pump.run(
                BenchmarkCommand(1),
                origin=Origin(component="benchmark"),
            ):
                pass

        def dispatch_once() -> None:
            loop.run_until_complete(dispatch())

        benchmark(dispatch_once)
    finally:
        loop.close()
