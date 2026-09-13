"""Small, dependency-free contracts shared by the Better Agent kernel."""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class _Identifier:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class MessageId(_Identifier):
    """Identity of one runtime message."""


@dataclass(frozen=True, slots=True)
class CorrelationId(_Identifier):
    """Identity of one logical run or workflow."""


@dataclass(frozen=True, slots=True)
class ExtensionId(_Identifier):
    """Stable identity of an installed extension."""


@dataclass(frozen=True, slots=True)
class Origin:
    """Kernel-owned source metadata for a runtime message."""

    component: str
    extension_id: ExtensionId | None = None


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable fact emitted by the runtime."""


@dataclass(frozen=True, slots=True)
class Command:
    """Immutable intent routed to exactly one command handler."""


type Message = Event | Command


@dataclass(frozen=True, slots=True)
class Envelope[T: Message]:
    """Kernel-owned message metadata wrapped around an event or command."""

    id: MessageId
    payload: T
    correlation_id: CorrelationId
    causation_id: MessageId | None
    origin: Origin


@dataclass(frozen=True, slots=True)
class ExecutionClaims:
    """Opaque scheduler claim contract until the scheduler story defines it."""


class EventHandler[E: Event](Protocol):
    """Synchronous event reactor returning commands to enqueue."""

    def __call__(self, event: Envelope[E], /) -> tuple[Command, ...]: ...


class CommandHandler[C: Command](Protocol):
    """Asynchronous effect boundary that can stream resulting events."""

    def __call__(self, command: Envelope[C], /) -> AsyncIterator[Event]: ...


class ExecutionPlanner[C: Command](Protocol):
    """Derives execution claims from a concrete command payload."""

    def __call__(self, command: C, /) -> ExecutionClaims: ...


@dataclass(frozen=True, slots=True)
class CommandBinding[C: Command]:
    """One command handler paired with its execution-claim planner."""

    handler: CommandHandler[C]
    execution: ExecutionPlanner[C]


class RuntimePort(Protocol):
    """Narrow ingress for active adapters to submit commands."""

    async def submit[C: Command](self, command: C, /) -> MessageId: ...


class EventBus(Protocol):
    """Event subscription and publication boundary."""

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        /,
    ) -> None: ...

    async def publish[E: Event](
        self,
        event: Envelope[E],
        /,
    ) -> None: ...


class CommandBus(Protocol):
    """Exactly-one command binding and dispatch boundary."""

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C], /) -> None: ...

    async def dispatch[C: Command](
        self,
        command: Envelope[C],
        /,
    ) -> None: ...


class RuntimeSink(Protocol):
    """Explicit message output channel used by bus implementations during a run."""

    def emit_command(self, command: Command, *, origin: Origin) -> None: ...

    async def emit_event(self, event: Event, *, origin: Origin) -> None: ...


class RuntimeChannel(RuntimeSink, Protocol):
    """Runtime-owned rendezvous channel shared by the pump and its buses."""

    def begin(self) -> None: ...

    async def receive_event(self) -> tuple[Event, Origin] | None: ...

    def acknowledge_event(self) -> None: ...

    def finish(self) -> None: ...

    def take_commands(self) -> tuple[tuple[Command, Origin], ...]: ...

    def close(self) -> None: ...


class ExecutionScheduler(Protocol):
    """Kernel-owned service controlling effectful command admission."""

    async def run[T](
        self,
        claims: ExecutionClaims,
        operation: Callable[[], Awaitable[T]],
        /,
    ) -> T: ...


class EnvelopeFactory(Protocol):
    """Creates envelopes and propagates trace metadata."""

    def create[T: Message, C: Message](
        self,
        payload: T,
        *,
        origin: Origin,
        cause: Envelope[C] | None = None,
    ) -> Envelope[T]: ...
