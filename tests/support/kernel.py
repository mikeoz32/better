from collections.abc import AsyncIterator
from typing import cast

from better_agent.kernel import (
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
    MessageId,
    Origin,
    ProducedMessage,
    RuntimePort,
)


class RecordingEnvelopeFactory(EnvelopeFactory):
    def __init__(self) -> None:
        self._next_id = 0

    def create[T: Event | Command, C: Event | Command](
        self,
        payload: T,
        *,
        origin: Origin,
        cause: Envelope[C] | None = None,
    ) -> Envelope[T]:
        self._next_id += 1
        message_id = MessageId(f"message-{self._next_id}")
        correlation_id = cause.correlation_id if cause else CorrelationId("correlation-1")
        causation_id = cause.id if cause else None
        return Envelope(
            id=message_id,
            payload=payload,
            correlation_id=correlation_id,
            causation_id=causation_id,
            origin=origin,
        )


class RecordingEventBus(EventBus):
    def __init__(self) -> None:
        self.subscriptions: list[tuple[type[Event], EventHandler[Event]]] = []
        self.published: list[Envelope[Event]] = []

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        *,
        origin: Origin,
    ) -> None:
        self.subscriptions.append((event_type, cast(EventHandler[Event], handler)))

    async def publish[E: Event](
        self,
        event: Envelope[E],
    ) -> tuple[ProducedMessage[Command], ...]:
        self.published.append(cast(Envelope[Event], event))
        return ()


class RecordingCommandBus(CommandBus):
    def __init__(self) -> None:
        self.bindings: dict[type[Command], CommandBinding[Command]] = {}
        self.dispatched: list[Envelope[Command]] = []

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C]) -> None:
        self.bindings[command_type] = cast(CommandBinding[Command], binding)

    def dispatch[C: Command](self, command: Envelope[C]) -> AsyncIterator[ProducedMessage[Event]]:
        self.dispatched.append(cast(Envelope[Command], command))

        async def empty() -> AsyncIterator[ProducedMessage[Event]]:
            if False:
                yield ProducedMessage(Event(), Origin(component="test"))

        return empty()


class RecordingRuntimePort(RuntimePort):
    def __init__(self) -> None:
        self.submitted: list[Command] = []

    async def submit[C: Command](self, command: C) -> MessageId:
        self.submitted.append(command)
        return MessageId(f"submitted-{len(self.submitted)}")


def exclusive_claims(_: Command) -> ExecutionClaims:
    return ExecutionClaims()
