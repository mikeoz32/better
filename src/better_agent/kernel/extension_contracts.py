"""Shared contracts for extension discovery results."""

from dataclasses import dataclass
from enum import StrEnum

from better_agent.kernel.contracts import ExtensionId
from better_agent.kernel.errors import KernelError
from better_agent.kernel.extensions import Extension


@dataclass(frozen=True, slots=True)
class ExtensionRequirement:
    """Declare whether discovery must find a named extension."""

    id: ExtensionId
    required: bool = True


class ExtensionLoadFailureKind(StrEnum):
    """Classify a source failure without applying startup policy."""

    MISSING = "missing"
    LOAD = "load"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ExtensionLoadFailure:
    """One extension that could not be loaded by a source."""

    extension_id: ExtensionId
    required: bool
    kind: ExtensionLoadFailureKind
    message: str


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    """Extensions and load failures reported by one discovery source."""

    extensions: tuple[Extension, ...] = ()
    failures: tuple[ExtensionLoadFailure, ...] = ()


class RequiredExtensionLoadError(KernelError):
    """Raised when discovery cannot load one or more required extensions."""

    def __init__(self, failures: tuple[ExtensionLoadFailure, ...]) -> None:
        self.failures = failures
        super().__init__(f"required extension load failed for {len(failures)} extension(s)")
