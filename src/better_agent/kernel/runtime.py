"""Concrete event-command runtime primitives."""

from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
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
)
from better_agent.kernel.contracts import CommandBus, EventBus
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


@dataclass(frozen=True, slots=True)
class _Produced[T: Message]:
    """Internal payload/provenance pair used only while pumping messages."""

    payload: T
    origin: Origin


def _producer_origin(producer: object) -> Origin:
    """Derive provenance without expanding the public subscription contract."""
    producer_type = type(producer)
    module = cast(str, getattr(producer, "__module__", producer_type.__module__))
    name = cast(str, getattr(producer, "__qualname__", producer_type.__qualname__))
    return Origin(component=f"{module}.{name}")


class InMemoryEventBus(EventBus):
    """Fan out events to matching handlers in registration order."""

    def __init__(self) -> None:
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
        await self._react(event)

    async def _react[E: Event](
        self,
        event: Envelope[E],
        /,
    ) -> tuple[_Produced[Command], ...]:
        commands: list[_Produced[Command]] = []
        for subscription in self._subscriptions:
            if isinstance(event.payload, subscription.event_type):
                commands.extend(
                    _Produced(command, subscription.origin)
                    for command in subscription.handler(cast(Envelope[Event], event))
                )
        return tuple(commands)


class InMemoryCommandBus(CommandBus):
    """Resolve one command binding and expose its stream to the serial pump."""

    def __init__(self) -> None:
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
        """Execute a command at the ADR boundary and discard emitted events."""
        async for _ in self._stream(command):
            pass

    def _stream[C: Command](self, command: Envelope[C], /) -> AsyncIterator[_Produced[Event]]:
        command_type = type(command.payload)
        binding = self._bindings.get(command_type)
        if binding is None:
            raise MissingCommandHandlerError(
                f"no command handler is bound for {command_type.__name__}"
            )

        async def stream() -> AsyncIterator[_Produced[Event]]:
            async for event in binding.handler(cast(Envelope[Command], command)):
                yield _Produced(event, self._origins[command_type])

        return stream()


class RuntimePump:
    """Process event reactions and command streams serially with backpressure."""

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        command_bus: InMemoryCommandBus,
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
            async for produced in self._command_bus._stream(current):
                event = self._envelope_factory.create(
                    produced.payload,
                    origin=produced.origin,
                    cause=current,
                )
                commands = await self._event_bus._react(event)
                for produced_command in commands:
                    pending.append(
                        self._envelope_factory.create(
                            produced_command.payload,
                            origin=produced_command.origin,
                            cause=event,
                        )
                    )
                yield event
