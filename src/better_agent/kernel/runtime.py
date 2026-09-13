"""Concrete event-command runtime primitives."""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

from better_agent.kernel.contracts import (
    Command,
    CommandBinding,
    CorrelationId,
    Envelope,
    EnvelopeFactory,
    Event,
    EventHandler,
    Message,
    MessageId,
    Origin,
    RuntimeChannel,
    RuntimeSink,
)
from better_agent.kernel.contracts import CommandBus, EventBus
from better_agent.kernel.errors import (
    DuplicateCommandBindingError,
    MissingCommandHandlerError,
    RuntimeDispatchContextError,
)


class DefaultEnvelopeFactory(EnvelopeFactory):
    """Create production envelopes with unique message and correlation IDs."""

    def create[T: Message, C: Message](
        self,
        payload: T,
        *,
        origin: Origin,
        cause: Envelope[C] | None = None,
    ) -> Envelope[T]:
        return Envelope(
            id=MessageId(uuid4().hex),
            payload=payload,
            correlation_id=cause.correlation_id if cause else CorrelationId(uuid4().hex),
            causation_id=cause.id if cause else None,
            origin=origin,
        )


@dataclass(frozen=True, slots=True)
class _Subscription:
    event_type: type[Event]
    handler: EventHandler[Event]
    origin: Origin


@dataclass(slots=True)
class _EventDelivery:
    event: Event
    origin: Origin
    acknowledged: asyncio.Future[None]


class RendezvousChannel(RuntimeChannel):
    """One-slot event delivery that blocks producers until the consumer resumes."""

    def __init__(self) -> None:
        self._event: _EventDelivery | None = None
        self._in_flight: _EventDelivery | None = None
        self._available = asyncio.Event()
        self._closed = False
        self._commands: list[tuple[Command, Origin]] = []

    def begin(self) -> None:
        if self._event is not None or self._in_flight is not None:
            raise RuntimeDispatchContextError("runtime channel has an outstanding event")
        self._available.clear()
        self._commands.clear()
        self._closed = False

    def emit_command(self, command: Command, *, origin: Origin) -> None:
        if self._closed:
            raise RuntimeDispatchContextError("runtime emitter is closed")
        self._commands.append((command, origin))

    async def emit_event(self, event: Event, *, origin: Origin) -> None:
        if self._closed:
            raise RuntimeDispatchContextError("runtime emitter is closed")
        if self._event is not None:
            raise RuntimeDispatchContextError("runtime emitter already has an unconsumed event")
        delivery = _EventDelivery(
            event,
            origin,
            asyncio.get_running_loop().create_future(),
        )
        self._event = delivery
        self._available.set()
        await delivery.acknowledged

    async def receive_event(self) -> tuple[Event, Origin] | None:
        while self._event is None and not self._closed:
            await self._available.wait()
        if self._event is None:
            return None
        delivery = self._event
        self._event = None
        self._in_flight = delivery
        self._available.clear()
        return delivery.event, delivery.origin

    def finish(self) -> None:
        self._closed = True
        self._available.set()

    def acknowledge_event(self) -> None:
        delivery = self._in_flight
        if delivery is None:
            raise RuntimeDispatchContextError("runtime channel has no event to acknowledge")
        self._in_flight = None
        if not delivery.acknowledged.done():
            delivery.acknowledged.set_result(None)

    def take_commands(self) -> tuple[tuple[Command, Origin], ...]:
        commands = tuple(self._commands)
        self._commands.clear()
        return commands

    def close(self) -> None:
        self._closed = True
        self._available.set()
        if self._event is not None and not self._event.acknowledged.done():
            self._event.acknowledged.cancel()
        if self._in_flight is not None and not self._in_flight.acknowledged.done():
            self._in_flight.acknowledged.cancel()
        self._event = None
        self._in_flight = None


def _producer_origin(producer: object) -> Origin:
    """Derive provenance without expanding the public subscription contract."""
    producer_type = type(producer)
    module = cast(str, getattr(producer, "__module__", producer_type.__module__))
    name = cast(str, getattr(producer, "__qualname__", producer_type.__qualname__))
    return Origin(component=f"{module}.{name}")


class InMemoryEventBus(EventBus):
    """Fan out events to matching handlers in registration order."""

    def __init__(self, sink: RuntimeSink) -> None:
        self._sink = sink
        self._subscriptions: list[_Subscription] = []

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        /,
    ) -> None:
        subscription = _Subscription(
            event_type,
            cast(EventHandler[Event], handler),
            _producer_origin(handler),
        )
        self._subscriptions.append(subscription)

    async def publish[E: Event](
        self,
        event: Envelope[E],
        /,
    ) -> None:
        for subscription in self._subscriptions:
            if isinstance(event.payload, subscription.event_type):
                for command in subscription.handler(cast(Envelope[Event], event)):
                    self._sink.emit_command(
                        command,
                        origin=subscription.origin,
                    )


class InMemoryCommandBus(CommandBus):
    """Resolve one command binding and expose its stream to the serial pump."""

    def __init__(self, sink: RuntimeSink) -> None:
        self._sink = sink
        self._bindings: dict[type[Command], CommandBinding[Command]] = {}
        self._origins: dict[type[Command], Origin] = {}

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C]) -> None:
        if command_type in self._bindings:
            raise DuplicateCommandBindingError(
                f"command type {command_type.__name__} already has an active binding"
            )
        self._bindings[command_type] = cast(CommandBinding[Command], binding)
        self._origins[command_type] = _producer_origin(binding.handler)

    async def dispatch[C: Command](self, command: Envelope[C], /) -> None:
        """Execute a command and deliver each handler event to the active pump."""
        command_type = type(command.payload)
        binding = self._bindings.get(command_type)
        if binding is None:
            raise MissingCommandHandlerError(
                f"no command handler is bound for {command_type.__name__}"
            )

        async for event in binding.handler(cast(Envelope[Command], command)):
            await self._sink.emit_event(event, origin=self._origins[command_type])


class RuntimePump:
    """Process event reactions and command streams serially with backpressure."""

    def __init__(
        self,
        event_bus: EventBus,
        command_bus: CommandBus,
        envelope_factory: EnvelopeFactory,
        channel: RuntimeChannel,
    ) -> None:
        self._event_bus = event_bus
        self._command_bus = command_bus
        self._envelope_factory = envelope_factory
        self._channel = channel

    async def run[C: Command](
        self,
        command: C,
        *,
        origin: Origin,
    ) -> AsyncGenerator[Envelope[Event], None]:
        """Run one command and stream the resulting event envelopes."""
        initial = self._envelope_factory.create(command, origin=origin)
        async for event in self.run_envelope(initial):
            yield event

    async def run_envelope[C: Command](
        self,
        command: Envelope[C],
    ) -> AsyncGenerator[Envelope[Event], None]:
        """Run an already enveloped command without recursive dispatch."""
        pending: deque[Envelope[Command]] = deque([cast(Envelope[Command], command)])
        while pending:
            current = pending.popleft()
            self._channel.begin()
            queued_commands: list[tuple[Command, Origin, Envelope[Event]]] = []

            async def dispatch() -> None:
                try:
                    await self._command_bus.dispatch(current)
                finally:
                    self._channel.finish()

            dispatch_task = asyncio.create_task(dispatch())
            completed = False
            try:
                while True:
                    delivery = await self._channel.receive_event()
                    if delivery is None:
                        break
                    payload, origin = delivery
                    event = self._envelope_factory.create(
                        payload,
                        origin=origin,
                        cause=current,
                    )
                    await self._event_bus.publish(event)
                    queued_commands.extend(
                        (command, origin, event)
                        for command, origin in self._channel.take_commands()
                    )
                    yield event
                    self._channel.acknowledge_event()

                await dispatch_task
                for produced_command, origin, cause in queued_commands:
                    pending.append(
                        self._envelope_factory.create(
                            produced_command,
                            origin=origin,
                            cause=cause,
                        )
                    )
                completed = True
            finally:
                if not dispatch_task.done():
                    dispatch_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await dispatch_task
                if not completed:
                    self._channel.close()
