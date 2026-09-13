"""Run lifecycle, outcome and terminal message contracts."""

from dataclasses import dataclass
from enum import StrEnum

from better_agent.kernel.contracts import Command, Event
from better_agent.kernel.errors import InvalidRunTransitionError


class RunState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or None")


@dataclass(frozen=True, slots=True)
class RuntimeFault:
    phase: str
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class CompletedOutcome:
    pass


@dataclass(frozen=True, slots=True)
class StepBudgetExceeded(Event):
    limit: int
    attempted_step: int


@dataclass(frozen=True, slots=True)
class FailedOutcome:
    reason: RuntimeFault | StepBudgetExceeded


@dataclass(frozen=True, slots=True)
class CancelledOutcome:
    reason: str | None = None


type RunOutcome = CompletedOutcome | FailedOutcome | CancelledOutcome


@dataclass(frozen=True, slots=True)
class RunStarted(Event):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeFaulted(Event):
    fault: RuntimeFault


@dataclass(frozen=True, slots=True)
class RunOutcomeReady(Event):
    outcome: RunOutcome


@dataclass(frozen=True, slots=True)
class RunCompleted(Event):
    outcome: CompletedOutcome


@dataclass(frozen=True, slots=True)
class RunFailed(Event):
    outcome: FailedOutcome


@dataclass(frozen=True, slots=True)
class RunCancelled(Event):
    outcome: CancelledOutcome


@dataclass(frozen=True, slots=True)
class CommitRunOutcome(Command):
    outcome: RunOutcome


_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.RUNNING}),
    RunState.RUNNING: frozenset({RunState.FINALIZING}),
    RunState.FINALIZING: frozenset(
        {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED},
    ),
}


@dataclass(slots=True)
class RunLifecycle:
    state: RunState = RunState.CREATED

    def transition(self, target: RunState) -> None:
        if target not in _TRANSITIONS.get(self.state, frozenset()):
            raise InvalidRunTransitionError(
                f"cannot transition run from {self.state.value} to {target.value}",
            )
        self.state = target
