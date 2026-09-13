"""Extension discovery, composition and host lifecycle."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from typing import Protocol, cast

from better_agent.kernel.contracts import (
    Command,
    CommandBinding,
    Event,
    EventHandler,
    ExecutionPlanner,
    ExtensionId,
)
from better_agent.kernel.errors import (
    ExtensionHostStateError,
    ExtensionInstallError,
    ExtensionRegistrationError,
)
from better_agent.kernel.extension_contracts import (
    ExtensionLoadFailure,
    ExtensionLoadFailureKind,
    ExtensionLoadResult,
    ExtensionRequirement,
    RequiredExtensionLoadError,
)
from better_agent.kernel.extensions import (
    Extension,
    ExtensionRegistrar,
    ExtensionResolver,
    ExtensionSpec,
)


class ExtensionSource(Protocol):
    """Synchronous source of structural extensions and load diagnostics."""

    def load(self, /) -> ExtensionLoadResult: ...


class StaticExtensionSource:
    """Return a fixed extension collection for tests and bundled composition."""

    def __init__(self, extensions: Iterable[Extension], /) -> None:
        self._extensions = tuple(extensions)

    def load(self, /) -> ExtensionLoadResult:
        return ExtensionLoadResult(self._extensions)


class EntryPointExtensionSource:
    """Discover extensions from the fixed ``better_agent.extensions`` group."""

    GROUP = "better_agent.extensions"

    def __init__(self, requirements: Iterable[ExtensionRequirement] = ()) -> None:
        required_by_id: dict[ExtensionId, bool] = {}
        for requirement in requirements:
            required_by_id[requirement.id] = (
                required_by_id.get(requirement.id, False) or requirement.required
            )
        self._required_by_id = required_by_id

    def load(self, /) -> ExtensionLoadResult:
        entry_points = tuple(metadata.entry_points(group=self.GROUP))
        ordered = tuple(sorted(entry_points, key=_entry_point_sort_key))
        by_name: dict[str, list[metadata.EntryPoint]] = {}
        for entry_point in ordered:
            by_name.setdefault(str(entry_point.name), []).append(entry_point)

        names = sorted(
            set(by_name) | {extension_id.value for extension_id in self._required_by_id},
        )
        extensions: list[Extension] = []
        failures: list[ExtensionLoadFailure] = []
        for name in names:
            extension_id = ExtensionId(name)
            required = self._required_by_id.get(extension_id, False)
            candidates = by_name.get(name, [])
            if not candidates:
                failures.append(
                    ExtensionLoadFailure(
                        extension_id,
                        required,
                        ExtensionLoadFailureKind.MISSING,
                        "entry point is not installed",
                    ),
                )
                continue
            if len(candidates) > 1:
                failures.append(
                    ExtensionLoadFailure(
                        extension_id,
                        required,
                        ExtensionLoadFailureKind.INVALID,
                        "multiple entry points have the same name",
                    ),
                )
                continue

            extension, failure = _load_entry_point(candidates[0], extension_id, required)
            if failure is not None:
                failures.append(failure)
            elif extension is not None:
                extensions.append(extension)

        return ExtensionLoadResult(tuple(extensions), tuple(failures))


def _entry_point_sort_key(entry_point: object) -> tuple[str, str, str]:
    return (
        str(getattr(entry_point, "name", "")),
        str(getattr(entry_point, "value", "")),
        _distribution_name(entry_point),
    )


def _distribution_name(entry_point: object) -> str:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return ""
    name = getattr(distribution, "name", None)
    if name is not None:
        return str(name)
    metadata_values = getattr(distribution, "metadata", {})
    return str(metadata_values.get("Name", ""))


def _load_entry_point(
    entry_point: metadata.EntryPoint,
    extension_id: ExtensionId,
    required: bool,
) -> tuple[Extension | None, ExtensionLoadFailure | None]:
    try:
        factory = entry_point.load()
    except Exception as error:
        return None, ExtensionLoadFailure(
            extension_id,
            required,
            ExtensionLoadFailureKind.LOAD,
            str(error),
        )

    if not callable(factory):
        return None, ExtensionLoadFailure(
            extension_id,
            required,
            ExtensionLoadFailureKind.INVALID,
            "entry point target is not callable",
        )

    try:
        extension = factory()
    except Exception as error:
        return None, ExtensionLoadFailure(
            extension_id,
            required,
            ExtensionLoadFailureKind.LOAD,
            str(error),
        )

    try:
        spec = extension.spec
        install = extension.install
        valid_spec = (
            isinstance(spec, ExtensionSpec)
            and isinstance(spec.id, ExtensionId)
            and isinstance(spec.id.value, str)
            and isinstance(spec.requires, tuple)
            and all(
                isinstance(required_id, ExtensionId) and isinstance(required_id.value, str)
                for required_id in spec.requires
            )
        )
        valid_extension = valid_spec and callable(install)
    except Exception:
        valid_extension = False
        spec = None

    if not valid_extension:
        return None, ExtensionLoadFailure(
            extension_id,
            required,
            ExtensionLoadFailureKind.INVALID,
            "entry point did not return a valid extension",
        )
    if cast(ExtensionSpec, spec).id.value != extension_id.value:
        return None, ExtensionLoadFailure(
            extension_id,
            required,
            ExtensionLoadFailureKind.INVALID,
            "extension spec id does not match entry point name",
        )
    return cast(Extension, extension), None


@dataclass(frozen=True, slots=True)
class ExtensionHostReport:
    """Result of a successful host startup."""

    installed: tuple[ExtensionId, ...]
    skipped: tuple[ExtensionLoadFailure, ...] = ()


class ExtensionHost(Protocol):
    """Compose a fixed extension set around a real registrar."""

    async def start(self, /) -> ExtensionHostReport: ...

    async def stop(self, /) -> None: ...


@dataclass(frozen=True, slots=True)
class _StagedRegistration:
    owner: ExtensionId
    operation: str
    arguments: tuple[object, ...]


class _StagingRegistrar(ExtensionRegistrar):
    def __init__(self, owner: ExtensionId, registrations: list[_StagedRegistration]) -> None:
        self._owner = owner
        self._registrations = registrations

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        /,
    ) -> None:
        self._registrations.append(
            _StagedRegistration(self._owner, "subscribe", (event_type, handler)),
        )

    def handle[C: Command](
        self,
        command_type: type[C],
        handler,
        /,
        *,
        execution: ExecutionPlanner[C] | None = None,
    ) -> None:
        self._registrations.append(
            _StagedRegistration(
                self._owner,
                "handle",
                (command_type, CommandBinding(handler, execution)),
            ),
        )


class _HostState(StrEnum):
    CREATED = "created"
    STARTED = "started"
    STOPPED = "stopped"
    FAILED = "failed"


class DefaultExtensionHost:
    """Load, resolve, stage and atomically replay a composed extension set."""

    def __init__(
        self,
        sources: Iterable[ExtensionSource],
        resolver: ExtensionResolver,
        registrar: ExtensionRegistrar,
        /,
    ) -> None:
        self._sources = tuple(sources)
        self._resolver = resolver
        self._registrar = registrar
        self._state = _HostState.CREATED
        self._report: ExtensionHostReport | None = None

    async def start(self, /) -> ExtensionHostReport:
        if self._state is _HostState.STARTED:
            return cast(ExtensionHostReport, self._report)
        if self._state is not _HostState.CREATED:
            raise ExtensionHostStateError("start", self._state.value)

        try:
            results = tuple(source.load() for source in self._sources)
            failures = tuple(failure for result in results for failure in result.failures)
            required_failures = tuple(failure for failure in failures if failure.required)
            if required_failures:
                raise RequiredExtensionLoadError(required_failures)

            extensions = tuple(extension for result in results for extension in result.extensions)
            resolved = self._resolver.resolve(extensions)
            staged: list[_StagedRegistration] = []
            resolved_ids: list[ExtensionId] = []
            for extension in resolved:
                owner = extension.spec.id
                resolved_ids.append(owner)
                try:
                    extension.install(_StagingRegistrar(owner, staged))
                except Exception as error:
                    raise ExtensionInstallError(owner) from error

            for registration in staged:
                try:
                    self._replay(registration)
                except Exception as error:
                    raise ExtensionRegistrationError(
                        registration.owner,
                        registration.operation,
                    ) from error

            report = ExtensionHostReport(
                tuple(resolved_ids),
                tuple(failure for failure in failures if not failure.required),
            )
        except Exception:
            self._state = _HostState.FAILED
            raise

        self._report = report
        self._state = _HostState.STARTED
        return report

    def _replay(self, registration: _StagedRegistration) -> None:
        if registration.operation == "subscribe":
            event_type, handler = registration.arguments
            self._registrar.subscribe(
                cast(type[Event], event_type),
                cast(EventHandler[Event], handler),
            )
            return

        command_type, binding = registration.arguments
        typed_binding = cast(CommandBinding[Command], binding)
        self._registrar.handle(
            cast(type[Command], command_type),
            typed_binding.handler,
            execution=typed_binding.execution,
        )

    async def stop(self, /) -> None:
        if self._state is _HostState.STOPPED:
            return
        self._report = None
        self._state = _HostState.STOPPED
