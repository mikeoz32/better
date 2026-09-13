"""Runtime errors raised by the kernel dispatch boundaries."""


class KernelError(RuntimeError):
    """Base class for expected kernel configuration and dispatch failures."""


class DuplicateCommandBindingError(KernelError):
    """Raised when a command type receives more than one active binding."""


class MissingCommandHandlerError(KernelError):
    """Raised when a command has no active binding."""
