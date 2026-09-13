"""Runtime errors raised by the kernel dispatch boundaries."""


class KernelError(RuntimeError):
    """Base class for expected kernel configuration and dispatch failures."""


class DuplicateCommandBindingError(KernelError):
    """Raised when a command type receives more than one active binding."""


class MissingCommandHandlerError(KernelError):
    """Raised when a command has no active binding."""


class InvalidRunTransitionError(KernelError):
    """Raised when a run lifecycle transition is not legal."""


class RuntimeExecutionError(KernelError):
    """Internal error carrying the phase and message that failed at runtime."""

    def __init__(self, phase: str, cause: object, error: BaseException) -> None:
        self.phase = phase
        self.cause = cause
        self.error = error
        super().__init__(f"runtime {phase} execution failed: {error}")


class StepBudgetLimitReached(KernelError):
    """Internal signal raised before a command would exceed the work budget."""

    def __init__(self, limit: int, attempted_step: int, cause: object) -> None:
        self.limit = limit
        self.attempted_step = attempted_step
        self.cause = cause
        super().__init__(
            f"step budget {limit} exceeded at attempted step {attempted_step}",
        )


class RunFinalizationError(KernelError):
    """Raised when outcome commit violates the terminal-event contract."""
