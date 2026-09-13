from dataclasses import FrozenInstanceError

import pytest

from better_agent import (
    CancelledOutcome,
    CompletedOutcome,
    FailedOutcome,
    RunLifecycle,
    RunLimits,
    RunState,
    RuntimeFault,
    StepBudgetExceeded,
)
from better_agent.kernel.errors import InvalidRunTransitionError


def test_run_lifecycle_accepts_only_the_declared_transitions() -> None:
    lifecycle = RunLifecycle()

    lifecycle.transition(RunState.RUNNING)
    lifecycle.transition(RunState.FINALIZING)
    lifecycle.transition(RunState.COMPLETED)

    assert lifecycle.state is RunState.COMPLETED


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunState.CREATED, RunState.FAILED),
        (RunState.RUNNING, RunState.COMPLETED),
        (RunState.FINALIZING, RunState.RUNNING),
        (RunState.COMPLETED, RunState.COMPLETED),
    ],
)
def test_run_lifecycle_rejects_illegal_or_repeated_transitions(
    current: RunState,
    target: RunState,
) -> None:
    lifecycle = RunLifecycle(current)

    with pytest.raises(InvalidRunTransitionError):
        lifecycle.transition(target)


def test_run_limits_validate_positive_budgets() -> None:
    assert RunLimits().max_steps is None
    assert RunLimits(max_steps=1).max_steps == 1

    with pytest.raises(ValueError, match="max_steps"):
        RunLimits(max_steps=0)

    with pytest.raises(ValueError, match="max_steps"):
        RunLimits(max_steps=-1)


def test_outcomes_and_faults_are_immutable_typed_values() -> None:
    fault = RuntimeFault(
        phase="command_handler",
        exception_type="ValueError",
        message="bad input",
    )
    budget = StepBudgetExceeded(limit=2, attempted_step=3)
    outcomes = (
        CompletedOutcome(),
        FailedOutcome(fault),
        FailedOutcome(budget),
        CancelledOutcome(reason="user requested"),
    )

    assert outcomes[1].reason == fault
    assert outcomes[2].reason == budget
    with pytest.raises(FrozenInstanceError):
        setattr(fault, "message", "changed")
