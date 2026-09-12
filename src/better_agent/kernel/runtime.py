"""Concrete event-command runtime primitives."""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

from better_agent.kernel.contracts import (
    Command,
    CommandBinding,
    CommandBus,
    CorrelationId,
    Envelope,
    EnvelopeFactory,
    Event,
    EventBus,
    EventHandler,
    Message,
    MessageId,
    Origin,
    ProducedMessage,
)
from better_agent.kernel.errors import DuplicateCommandBindingError, MissingCommandHandlerError


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


class InMemoryEventBus(EventBus):
    """Fan out events to matching handlers in registration order."""

    def __init__(self) -> None:
        self._subscriptions: list[_Subscription] = []

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        *,
        origin: Origin,
    ) -> None:
        subscription = _Subscription(event_type, cast(EventHandler[Event], handler), origin)
        self._subscriptions.append(subscription)

    async def publish[E: Event](
        self,
        event: Envelope[E],
    ) -> tuple[ProducedMessage[Command], ...]:
        commands: list[ProducedMessage[Command]] = []
        for subscription in self._subscriptions:
            if isinstance(event.payload, subscription.event_type):
                commands.extend(
                    ProducedMessage(command, subscription.origin)
                    for command in subscription.handler(cast(Envelope[Event], event))
                )
        return tuple(commands)


class InMemoryCommandBus(CommandBus):
    """Resolve one command binding and collect its emitted event payloads."""

    def __init__(self) -> None:
        self._bindings: dict[type[Command], CommandBinding[Command]] = {}

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C]) -> None:
        if command_type in self._bindings:
            raise DuplicateCommandBindingError(
                f"command type {command_type.__name__} already has an active binding"
            )
        self._bindings[command_type] = cast(CommandBinding[Command], binding)

    def dispatch[C: Command](self, command: Envelope[C]) -> AsyncIterator[ProducedMessage[Event]]:
        command_type = type(command.payload)
        binding = self._bindings.get(command_type)
        if binding is None:
            raise MissingCommandHandlerError(
                f"no command handler is bound for {command_type.__name__}"
            )

        async def stream() -> AsyncIterator[ProducedMessage[Event]]:
            async for event in binding.handler(cast(Envelope[Command], command)):
                yield ProducedMessage(event, binding.origin)

        return stream()


class RuntimePump:
    """Process event reactions and command execution through an explicit queue."""

    def __init__(
        self,
        event_bus: EventBus,
        command_bus: CommandBus,
        envelope_factory: EnvelopeFactory,
    ) -> None:
        self._event_bus = event_bus
        self._command_bus = command_bus
        self._envelope_factory = envelope_factory

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
            stream: asyncio.Queue[Envelope[Event] | _CommandStreamCompleted] = asyncio.Queue()
            async with asyncio.TaskGroup() as group:
                group.create_task(self._forward_command(current, stream))
                while True:
                    item = await stream.get()
                    if isinstance(item, _CommandStreamCompleted):
                        break

                    event = item
                    commands = await self._event_bus.publish(event)
                    for produced in commands:
                        pending.append(
                            self._envelope_factory.create(
                                produced.payload,
                                origin=produced.origin,
                                cause=event,
                            )
                        )
                    yield event

    async def _forward_command(
        self,
        command: Envelope[Command],
        stream: asyncio.Queue[Envelope[Event] | _CommandStreamCompleted],
    ) -> None:
        try:
            async for produced in self._command_bus.dispatch(command):
                await stream.put(
                    self._envelope_factory.create(
                        produced.payload,
                        origin=produced.origin,
                        cause=command,
                    )
                )
        finally:
            await stream.put(_CommandStreamCompleted())


@dataclass(frozen=True, slots=True)
class _CommandStreamCompleted:
    """Sentinel indicating that one command handler has finished streaming."""
