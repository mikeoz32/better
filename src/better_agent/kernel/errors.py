"""Runtime errors raised by the kernel dispatch boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from better_agent.kernel.contracts import ExtensionId

if TYPE_CHECKING:
    from better_agent.kernel.extension_host import ExtensionLoadFailure


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


class DuplicateExtensionIdError(KernelError):
    """Raised when multiple extensions declare the same identity."""

    def __init__(self, extension_id: ExtensionId) -> None:
        self.extension_id = extension_id
        super().__init__(f"extension id {extension_id} is declared more than once")


class MissingExtensionDependencyError(KernelError):
    """Raised when an extension requires an unavailable extension."""

    def __init__(self, extension_id: ExtensionId, required_id: ExtensionId) -> None:
        self.extension_id = extension_id
        self.required_id = required_id
        super().__init__(f"extension {extension_id} requires missing extension {required_id}")


class ExtensionDependencyCycleError(KernelError):
    """Raised when extension requirements contain a dependency cycle."""

    def __init__(self, cycle: tuple[ExtensionId, ...]) -> None:
        self.cycle = cycle
        super().__init__("extension dependency cycle: " + " -> ".join(map(str, cycle)))


class RequiredExtensionLoadError(KernelError):
    """Raised when discovery cannot load one or more required extensions."""

    def __init__(self, failures: tuple[ExtensionLoadFailure, ...]) -> None:
        self.failures = failures
        super().__init__(f"required extension load failed for {len(failures)} extension(s)")


class ExtensionInstallError(KernelError):
    """Raised when an extension fails while installing into the staging registrar."""

    def __init__(self, extension_id: ExtensionId) -> None:
        self.extension_id = extension_id
        super().__init__(f"extension {extension_id} installation failed")


class ExtensionRegistrationError(KernelError):
    """Raised when staged registration cannot be replayed into the real registrar."""

    def __init__(self, extension_id: ExtensionId, operation: str) -> None:
        self.extension_id = extension_id
        self.operation = operation
        super().__init__(f"extension {extension_id} {operation} registration failed")


class ExtensionHostStateError(KernelError):
    """Raised when host lifecycle operations are attempted in an invalid state."""

    def __init__(self, operation: str, state: str) -> None:
        self.operation = operation
        self.state = state
        super().__init__(f"cannot {operation} extension host in {state} state")
