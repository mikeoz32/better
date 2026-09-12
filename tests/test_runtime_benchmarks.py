import asyncio
from dataclasses import dataclass

from better_agent import Command, Event, Envelope, Origin
from better_agent.kernel.runtime import DefaultEnvelopeFactory, InMemoryEventBus


@dataclass(frozen=True, slots=True)
class BenchmarkEvent(Event):
    value: int


def test_event_bus_publish_benchmark(benchmark) -> None:
    bus = InMemoryEventBus()

    def subscriber(_: Envelope[BenchmarkEvent]) -> tuple[Command, ...]:
        return ()

    for _ in range(4):
        bus.subscribe(BenchmarkEvent, subscriber)
    event = DefaultEnvelopeFactory().create(
        BenchmarkEvent(1),
        origin=Origin(component="benchmark"),
    )

    def publish() -> tuple[Command, ...]:
        return asyncio.run(bus.publish(event))

    benchmark(publish)
