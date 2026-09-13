"""Concrete event-command runtime primitives."""

import asyncio
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
    ExecutionClaims,
    ExecutionScheduler,
    Message,
    MessageId,
    Origin,
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


class InMemoryEventBus(EventBus):
    """Fan out events and return produced commands in registration order."""

    def __init__(self) -> None:
        self._subscriptions: list[_Subscription] = []

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        /,
    ) -> None:
        self._subscriptions.append(
            _Subscription(event_type, cast(EventHandler[Event], handler)),
        )

    async def publish[E: Event](
        self,
        event: Envelope[E],
        /,
    ) -> tuple[Command, ...]:
        commands: list[Command] = []
        for subscription in self._subscriptions:
            if isinstance(event.payload, subscription.event_type):
                commands.extend(subscription.handler(cast(Envelope[Event], event)))
        return tuple(commands)


class InMemoryCommandBus(CommandBus):
    """Resolve bindings and hold scheduler admission across each handler stream."""

    def __init__(self, scheduler: ExecutionScheduler) -> None:
        self._scheduler = scheduler
        self._bindings: dict[type[Command], CommandBinding[Command]] = {}

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C]) -> None:
        if command_type in self._bindings:
            raise DuplicateCommandBindingError(
                f"command type {command_type.__name__} already has an active binding",
            )
        self._bindings[command_type] = cast(CommandBinding[Command], binding)

    def dispatch[C: Command](
        self,
        command: Envelope[C],
        /,
    ) -> AsyncIterator[Event]:
        """Return a lazy stream with one admission covering its full lifetime."""
        command_type = type(command.payload)
        binding = self._bindings.get(command_type)
        if binding is None:
            raise MissingCommandHandlerError(
                f"no command handler is bound for {command_type.__name__}",
            )
        claims = binding.execution(command.payload) if binding.execution else ExecutionClaims()

        async def stream() -> AsyncIterator[Event]:
            async with self._scheduler.admit(claims):
                async for event in binding.handler(cast(Envelope[Command], command)):
                    yield event

        return stream()


class RuntimePump:
    """Own concurrent command drains and serialize event reactions."""

    def __init__(
        self,
        event_bus: EventBus,
        command_bus: CommandBus,
        envelope_factory: EnvelopeFactory,
        *,
        inbox_size: int = 64,
    ) -> None:
        if inbox_size < 1:
            raise ValueError("inbox_size must be positive")
        self._event_bus = event_bus
        self._command_bus = command_bus
        self._envelope_factory = envelope_factory
        self._inbox_size = inbox_size

    async def run[C: Command](
        self,
        command: C,
        *,
        origin: Origin,
    ) -> AsyncGenerator[Envelope[Event], None]:
        """Run one command and stream the resulting event envelopes."""
        initial = self._envelope_factory.create(command, origin=origin)
        stream = self.run_envelope(initial)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def run_envelope[C: Command](
        self,
        command: Envelope[C],
    ) -> AsyncGenerator[Envelope[Event], None]:
        """Run an already enveloped command with structured task ownership."""
        inbox: asyncio.Queue[Envelope[Event]] = asyncio.Queue(maxsize=self._inbox_size)
        state_changed = asyncio.Condition()
        active_tasks = 0

        async def drain(current: Envelope[Command]) -> None:
            nonlocal active_tasks
            try:
                async for produced_event in self._command_bus.dispatch(current):
                    event = self._envelope_factory.create(
                        produced_event,
                        origin=current.origin,
                        cause=current,
                    )
                    await inbox.put(event)
                    async with state_changed:
                        state_changed.notify_all()
            finally:
                async with state_changed:
                    active_tasks -= 1
                    state_changed.notify_all()

        async with asyncio.TaskGroup() as tasks:
            owned_tasks: set[asyncio.Task[None]] = set()

            def start(current: Envelope[Command]) -> None:
                nonlocal active_tasks
                active_tasks += 1
                task = tasks.create_task(drain(current))
                owned_tasks.add(task)
                task.add_done_callback(owned_tasks.discard)

            start(cast(Envelope[Command], command))
            while True:
                async with state_changed:
                    while inbox.empty() and active_tasks:
                        await state_changed.wait()
                    if inbox.empty():
                        break
                    event = inbox.get_nowait()

                produced_commands = await self._event_bus.publish(event)
                for produced_command in produced_commands:
                    child = self._envelope_factory.create(
                        produced_command,
                        origin=event.origin,
                        cause=event,
                    )
                    start(child)
                try:
                    yield event
                except GeneratorExit:
                    for task in owned_tasks:
                        task.cancel()
                    return
