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
from better_agent.kernel.errors import (
    DuplicateCommandBindingError,
    MissingCommandHandlerError,
    RuntimeExecutionError,
    StepBudgetLimitReached,
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


@dataclass(frozen=True, slots=True)
class _StartCommand:
    envelope: Envelope[Command]


@dataclass(frozen=True, slots=True)
class _EventAcknowledged:
    pass


@dataclass(frozen=True, slots=True)
class _DrainFinished:
    pass


type _ControlMessage = _StartCommand | _EventAcknowledged | _DrainFinished


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
        max_steps: int | None = None,
    ) -> AsyncGenerator[Envelope[Event], None]:
        """Run one command and stream the resulting event envelopes."""
        initial = self._envelope_factory.create(command, origin=origin)
        stream = self.run_envelope(initial, max_steps=max_steps)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def run_envelope[C: Command](
        self,
        command: Envelope[C],
        *,
        max_steps: int | None = None,
    ) -> AsyncGenerator[Envelope[Event], None]:
        """Run an already enveloped command with structured task ownership."""
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive or None")
        inbox: asyncio.Queue[Envelope[Event]] = asyncio.Queue(maxsize=self._inbox_size)
        control: asyncio.Queue[_ControlMessage] = asyncio.Queue(maxsize=self._inbox_size)
        state_lock = asyncio.Lock()
        active_tasks = 0
        pending_events = 0
        supervisor_done = asyncio.Event()
        supervisor_error: BaseException | None = None
        steps = 0

        async def drain(current: Envelope[Command]) -> None:
            nonlocal pending_events
            try:
                try:
                    async for produced_event in self._command_bus.dispatch(current):
                        event = self._envelope_factory.create(
                            produced_event,
                            origin=current.origin,
                            cause=current,
                        )
                        async with state_lock:
                            pending_events += 1
                        try:
                            await inbox.put(event)
                        except BaseException:
                            async with state_lock:
                                pending_events -= 1
                            raise
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise RuntimeExecutionError("command_handler", current, error) from error
            finally:
                current_task = asyncio.current_task()
                if current_task is None or not current_task.cancelling():
                    await control.put(_DrainFinished())

        async def supervise() -> None:
            nonlocal active_tasks, pending_events, supervisor_error

            try:
                async with asyncio.TaskGroup() as tasks:
                    def start(current: Envelope[Command]) -> None:
                        nonlocal active_tasks, steps
                        attempted_step = steps + 1
                        if max_steps is not None and attempted_step > max_steps:
                            raise StepBudgetLimitReached(
                                max_steps,
                                attempted_step,
                                current,
                            )
                        steps = attempted_step
                        active_tasks += 1
                        tasks.create_task(drain(current))

                    start(cast(Envelope[Command], command))
                    while True:
                        message = await control.get()
                        if isinstance(message, _StartCommand):
                            start(message.envelope)
                        elif isinstance(message, _EventAcknowledged):
                            async with state_lock:
                                pending_events -= 1
                        else:
                            active_tasks -= 1

                        async with state_lock:
                            is_idle = (
                                active_tasks == 0
                                and pending_events == 0
                                and control.empty()
                            )
                        if is_idle:
                            break
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                supervisor_error = error
            finally:
                supervisor_done.set()

        supervisor = asyncio.create_task(supervise())

        async def send_control(message: _ControlMessage) -> None:
            put_task = asyncio.create_task(control.put(message))
            done_task = asyncio.create_task(supervisor_done.wait())
            try:
                done, pending = await asyncio.wait(
                    (put_task, done_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                put_task.cancel()
                done_task.cancel()
                await asyncio.gather(put_task, done_task, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if put_task in done:
                return
            if supervisor_error is not None:
                raise supervisor_error
            raise RuntimeError("runtime supervisor stopped before accepting control")

        async def next_event() -> Envelope[Event] | None:
            get_task = asyncio.create_task(inbox.get())
            done_task = asyncio.create_task(supervisor_done.wait())
            try:
                done, pending = await asyncio.wait(
                    (get_task, done_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                get_task.cancel()
                done_task.cancel()
                await asyncio.gather(get_task, done_task, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if get_task in done:
                return get_task.result()
            if supervisor_error is not None:
                raise supervisor_error
            return None

        try:
            while (event := await next_event()) is not None:
                try:
                    produced_commands = await self._event_bus.publish(event)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise RuntimeExecutionError("event_handler", event, error) from error
                for produced_command in produced_commands:
                    child = self._envelope_factory.create(
                        produced_command,
                        origin=event.origin,
                        cause=event,
                    )
                    await send_control(_StartCommand(child))
                await send_control(_EventAcknowledged())
                yield event
        finally:
            if not supervisor.done():
                supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)
