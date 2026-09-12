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

    loop = asyncio.new_event_loop()
    try:
        def publish() -> None:
            loop.run_until_complete(bus.publish(event))

        benchmark(publish)
    finally:
        loop.close()
