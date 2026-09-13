from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata

import pytest

from better_agent import (
    Command,
    Event,
    ExecutionClaims,
    Extension,
    ExtensionHost,
    ExtensionHostReport,
    ExtensionId,
    ExtensionLoadFailure,
    ExtensionLoadFailureKind,
    ExtensionLoadResult,
    ExtensionRegistrar,
    ExtensionRequirement,
    ExtensionSource,
    ExtensionSpec,
    RequiredExtensionLoadError,
    StaticExtensionSource,
    TopologicalExtensionResolver,
    DefaultExtensionHost,
    ExtensionHostStateError,
    ExtensionInstallError,
    ExtensionRegistrationError,
    EntryPointExtensionSource,
)
from tests.support.extensions import RecordingRegistrar


@dataclass(frozen=True, slots=True)
class HostEvent(Event):
    value: str


@dataclass(frozen=True, slots=True)
class HostCommand(Command):
    value: str


def event_handler(_: object) -> tuple[Command, ...]:
    return ()


async def command_handler(_: object):
    if False:
        yield HostEvent("unused")


def execution_planner(_: HostCommand) -> ExecutionClaims:
    return ExecutionClaims()


class FakeExtension:
    def __init__(
        self,
        spec: ExtensionSpec,
        installer: Callable[[ExtensionRegistrar], None] | None = None,
    ) -> None:
        self.spec = spec
        self.installer = installer or (lambda _: None)
        self.install_count = 0

    def install(self, registrar: ExtensionRegistrar, /) -> None:
        self.install_count += 1
        self.installer(registrar)


def extension(
    name: str, *requires: str, installer: Callable[[ExtensionRegistrar], None] | None = None
) -> FakeExtension:
    return FakeExtension(
        ExtensionSpec(
            ExtensionId(name),
            tuple(ExtensionId(required) for required in requires),
        ),
        installer,
    )


class FakeEntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        target: object = None,
        *,
        group: str = "better_agent.extensions",
        distribution_name: str = "test-distribution",
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.target = target
        self.distribution_name = distribution_name
        self.error = error

    @property
    def dist(self):
        return type("Distribution", (), {"name": self.distribution_name})()

    def load(self) -> object:
        if self.error is not None:
            raise self.error
        return self.target


@dataclass(frozen=True, slots=True)
class ResultSource:
    result: ExtensionLoadResult

    def load(self, /) -> ExtensionLoadResult:
        return self.result


def test_static_source_is_structural_and_deterministic() -> None:
    first = extension("first")
    second = extension("second")
    source: ExtensionSource = StaticExtensionSource((first, second))

    assert source.load() == ExtensionLoadResult((first, second))
    assert source.load() == ExtensionLoadResult((first, second))


def test_entry_point_source_filters_group_and_sorts_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered: list[str] = []
    alpha = extension("alpha")
    zulu = extension("zulu")
    entry_points = (
        FakeEntryPoint("zulu", "pkg:zulu", lambda: zulu),
        FakeEntryPoint("alpha", "pkg:alpha", lambda: alpha),
    )

    def fake_entry_points(*, group: str):
        discovered.append(group)
        return entry_points

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    source: ExtensionSource = EntryPointExtensionSource()

    result = source.load()

    assert discovered == ["better_agent.extensions"]
    assert tuple(item.spec.id.value for item in result.extensions) == ("alpha", "zulu")
    assert result.failures == ()


def test_entry_point_source_supports_zero_argument_classes_and_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClassExtension:
        spec = ExtensionSpec(ExtensionId("class-extension"))

        def install(self, registrar: ExtensionRegistrar, /) -> None:
            pass

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: (FakeEntryPoint("class-extension", "pkg:ClassExtension", ClassExtension),),
    )
    source = EntryPointExtensionSource(
        (
            ExtensionRequirement(ExtensionId("class-extension")),
            ExtensionRequirement(ExtensionId("required-missing")),
            ExtensionRequirement(ExtensionId("optional-missing"), required=False),
        ),
    )

    result = source.load()

    assert tuple(item.spec.id.value for item in result.extensions) == ("class-extension",)
    assert result.failures == (
        ExtensionLoadFailure(
            ExtensionId("optional-missing"),
            False,
            ExtensionLoadFailureKind.MISSING,
            "entry point is not installed",
        ),
        ExtensionLoadFailure(
            ExtensionId("required-missing"),
            True,
            ExtensionLoadFailureKind.MISSING,
            "entry point is not installed",
        ),
    )


@pytest.mark.parametrize(
    ("target", "error", "kind"),
    [
        (object(), None, ExtensionLoadFailureKind.INVALID),
        (lambda: object(), None, ExtensionLoadFailureKind.INVALID),
        (
            lambda: (_ for _ in ()).throw(ValueError("factory failed")),
            None,
            ExtensionLoadFailureKind.LOAD,
        ),
        (None, ImportError("module failed"), ExtensionLoadFailureKind.LOAD),
    ],
)
def test_entry_point_source_reports_invalid_and_load_failures(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    error: Exception | None,
    kind: ExtensionLoadFailureKind,
) -> None:
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: (FakeEntryPoint("broken", "pkg:broken", target, error=error),),
    )

    result = EntryPointExtensionSource().load()

    assert result.extensions == ()
    assert result.failures[0].extension_id == ExtensionId("broken")
    assert result.failures[0].kind is kind
    assert result.failures[0].required is False


def test_entry_point_source_reports_spec_id_mismatch_and_duplicate_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong = extension("wrong-id")
    first = extension("duplicate")
    second = extension("duplicate")
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: (
            FakeEntryPoint("wrong-name", "pkg:wrong", lambda: wrong),
            FakeEntryPoint("duplicate", "pkg:second", lambda: second, distribution_name="z"),
            FakeEntryPoint("duplicate", "pkg:first", lambda: first, distribution_name="a"),
        ),
    )

    result = EntryPointExtensionSource(
        (ExtensionRequirement(ExtensionId("duplicate")),),
    ).load()

    assert result.extensions == ()
    assert result.failures == (
        ExtensionLoadFailure(
            ExtensionId("duplicate"),
            True,
            ExtensionLoadFailureKind.INVALID,
            "multiple entry points have the same name",
        ),
        ExtensionLoadFailure(
            ExtensionId("wrong-name"),
            False,
            ExtensionLoadFailureKind.INVALID,
            "extension spec id does not match entry point name",
        ),
    )


@pytest.mark.asyncio
async def test_host_stages_installation_and_replays_in_resolved_order() -> None:
    registrar = RecordingRegistrar()
    observed_during_install: list[int] = []

    def install_base(target: ExtensionRegistrar) -> None:
        observed_during_install.append(len(registrar.calls))
        target.subscribe(HostEvent, event_handler)

    def install_dependent(target: ExtensionRegistrar) -> None:
        observed_during_install.append(len(registrar.calls))
        target.handle(HostCommand, command_handler, execution=execution_planner)

    base = extension("base", installer=install_base)
    dependent = extension("dependent", "base", installer=install_dependent)
    host: ExtensionHost = DefaultExtensionHost(
        (StaticExtensionSource((dependent, base)),),
        TopologicalExtensionResolver(),
        registrar,
    )

    report = await host.start()

    assert report == ExtensionHostReport((ExtensionId("base"), ExtensionId("dependent")))
    assert observed_during_install == [0, 0]
    assert [call[0] for call in registrar.calls] == ["subscribe", "handle"]
    assert registrar.handlers[0][1].execution is execution_planner


@pytest.mark.asyncio
async def test_host_does_not_commit_when_install_fails() -> None:
    registrar = RecordingRegistrar()
    failing = ValueError("install failed")
    extension_with_failure = extension(
        "broken",
        installer=lambda _: (_ for _ in ()).throw(failing),
    )
    host = DefaultExtensionHost(
        (StaticExtensionSource((extension_with_failure,)),),
        TopologicalExtensionResolver(),
        registrar,
    )

    with pytest.raises(ExtensionInstallError) as error:
        await host.start()

    assert error.value.extension_id == ExtensionId("broken")
    assert error.value.__cause__ is failing
    assert registrar.calls == []


@pytest.mark.asyncio
async def test_host_aborts_required_load_failures_before_resolution_or_install() -> None:
    extension_to_skip = extension("loaded")
    failure = ExtensionLoadFailure(
        ExtensionId("missing"),
        True,
        ExtensionLoadFailureKind.MISSING,
        "not installed",
    )

    class SpyResolver:
        called = False

        def resolve(self, extensions: object, /) -> tuple[Extension, ...]:
            self.called = True
            return ()

    resolver = SpyResolver()
    host = DefaultExtensionHost(
        (ResultSource(ExtensionLoadResult((extension_to_skip,), (failure,))),),
        resolver,
        RecordingRegistrar(),
    )

    with pytest.raises(RequiredExtensionLoadError) as error:
        await host.start()

    assert error.value.failures == (failure,)
    assert resolver.called is False
    assert extension_to_skip.install_count == 0


@pytest.mark.asyncio
async def test_host_reports_optional_failures_and_start_stop_are_idempotent() -> None:
    optional_failure = ExtensionLoadFailure(
        ExtensionId("optional"),
        False,
        ExtensionLoadFailureKind.LOAD,
        "could not import",
    )
    installed = extension("installed")
    source = ResultSource(ExtensionLoadResult((installed,), (optional_failure,)))
    host = DefaultExtensionHost((source,), TopologicalExtensionResolver(), RecordingRegistrar())

    first = await host.start()
    second = await host.start()
    await host.stop()
    await host.stop()

    assert first is second
    assert first == ExtensionHostReport((ExtensionId("installed"),), (optional_failure,))
    assert installed.install_count == 1
    with pytest.raises(ExtensionHostStateError):
        await host.start()


@pytest.mark.asyncio
async def test_host_translates_registration_failure_with_owner_and_operation() -> None:
    original = RuntimeError("registrar failed")

    class FailingRegistrar(RecordingRegistrar):
        def subscribe(self, event_type, handler, /) -> None:
            raise original

    host = DefaultExtensionHost(
        (
            StaticExtensionSource(
                (extension("owner", installer=lambda r: r.subscribe(HostEvent, event_handler)),)
            ),
        ),
        TopologicalExtensionResolver(),
        FailingRegistrar(),
    )

    with pytest.raises(ExtensionRegistrationError) as error:
        await host.start()

    assert error.value.extension_id == ExtensionId("owner")
    assert error.value.operation == "subscribe"
    assert error.value.__cause__ is original
