"""Small, dependency-free contracts shared by the Better Agent kernel."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
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
    """Resource access claims used to admit a command for execution."""

    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    exclusive: bool = True


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
    execution: ExecutionPlanner[C] | None = None


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
    ) -> tuple[Command, ...]: ...


class CommandBus(Protocol):
    """Exactly-one command binding and dispatch boundary."""

    def bind[C: Command](self, command_type: type[C], binding: CommandBinding[C], /) -> None: ...

    def dispatch[C: Command](
        self,
        command: Envelope[C],
        /,
    ) -> AsyncIterator[Event]: ...


class RunPump(Protocol):
    """Primary work-graph stream used by a Harness run."""

    def run_envelope[C: Command](
        self,
        command: Envelope[C],
        *,
        max_steps: int | None = None,
    ) -> AsyncIterator[Envelope[Event]]: ...


class ExecutionScheduler(Protocol):
    """Kernel-owned service controlling effectful command admission."""

    def admit(
        self,
        claims: ExecutionClaims,
    ) -> AbstractAsyncContextManager[None]: ...


class EnvelopeFactory(Protocol):
    """Creates envelopes and propagates trace metadata."""

    def create[T: Message, C: Message](
        self,
        payload: T,
        *,
        origin: Origin,
        cause: Envelope[C] | None = None,
    ) -> Envelope[T]: ...
