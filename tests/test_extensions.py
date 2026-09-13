from dataclasses import FrozenInstanceError, dataclass

import pytest

from better_agent import (
    Command,
    Event,
    ExecutionClaims,
    Extension,
    ExtensionId,
    ExtensionRegistrar,
    ExtensionResolver,
    ExtensionSpec,
    ExtensionDependencyCycleError,
    MissingExtensionDependencyError,
    DuplicateExtensionIdError,
    TopologicalExtensionResolver,
)
from tests.support.extensions import RecordingRegistrar


@dataclass(frozen=True, slots=True)
class ExtensionEvent(Event):
    value: str


@dataclass(frozen=True, slots=True)
class ExtensionCommand(Command):
    value: str


def event_handler(_: object) -> tuple[Command, ...]:
    return ()


async def command_handler(_: object):
    if False:
        yield ExtensionEvent("unused")


def claims(_: ExtensionCommand) -> ExecutionClaims:
    return ExecutionClaims(reads=frozenset({"test"}), exclusive=False)


class PlainExtension:
    @property
    def spec(self) -> ExtensionSpec:
        return ExtensionSpec(ExtensionId("plain"))

    def install(self, registrar: ExtensionRegistrar, /) -> None:
        registrar.subscribe(ExtensionEvent, event_handler)
        registrar.handle(ExtensionCommand, command_handler, execution=claims)


def extension(name: str, *requires: str) -> ExtensionSpec:
    return ExtensionSpec(
        ExtensionId(name),
        tuple(ExtensionId(required) for required in requires),
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (ExtensionSpec(ExtensionId("empty")), ()),
        (extension("child"), ()),
    ],
)
def test_extension_spec_is_immutable_and_defaults_requirements(spec: ExtensionSpec, expected: tuple) -> None:
    assert spec.requires == expected
    with pytest.raises(FrozenInstanceError):
        setattr(spec, "id", ExtensionId("changed"))


def test_extension_protocol_accepts_plain_structural_implementation() -> None:
    extension: Extension = PlainExtension()

    assert extension.spec.id == ExtensionId("plain")


def test_topological_resolver_is_structural_and_does_not_install() -> None:
    resolver: ExtensionResolver = TopologicalExtensionResolver()

    resolved = resolver.resolve([PlainSpecExtension(extension("plain"))])

    assert len(resolved) == 1


def test_recording_registrar_preserves_typed_contributions_and_call_order() -> None:
    registrar = RecordingRegistrar()
    PlainExtension().install(registrar)

    assert [call[0] for call in registrar.calls] == ["subscribe", "handle"]
    assert registrar.subscriptions == [(ExtensionEvent, event_handler)]
    assert len(registrar.handlers) == 1
    command_type, binding = registrar.handlers[0]
    assert command_type is ExtensionCommand
    assert binding.handler is command_handler
    assert binding.execution is claims


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], ()),
        ([extension("single")], ("single",)),
        ([extension("dependent", "base"), extension("base")], ("base", "dependent")),
        (
            [
                extension("app", "left", "right"),
                extension("right", "base"),
                extension("left", "base"),
                extension("base"),
            ],
            ("base", "left", "right", "app"),
        ),
    ],
)
def test_resolver_returns_dependencies_before_dependents(items: list[ExtensionSpec], expected: tuple[str, ...]) -> None:
    resolved = TopologicalExtensionResolver().resolve(
        PlainSpecExtension(spec) for spec in items
    )

    assert tuple(item.spec.id.value for item in resolved) == expected


def test_resolver_order_is_independent_of_input_permutation() -> None:
    items = [
        extension("zulu"),
        extension("alpha"),
        extension("middle", "alpha"),
    ]
    resolver = TopologicalExtensionResolver()

    assert tuple(item.spec.id.value for item in resolver.resolve(PlainSpecExtension(spec) for spec in items)) == (
        "alpha",
        "middle",
        "zulu",
    )
    assert tuple(item.spec.id.value for item in resolver.resolve(PlainSpecExtension(spec) for spec in reversed(items))) == (
        "alpha",
        "middle",
        "zulu",
    )


def test_resolver_rejects_duplicate_ids_before_other_validation() -> None:
    with pytest.raises(DuplicateExtensionIdError) as error:
        TopologicalExtensionResolver().resolve(
            [PlainSpecExtension(extension("duplicate")), PlainSpecExtension(extension("duplicate", "missing"))],
        )

    assert error.value.extension_id == ExtensionId("duplicate")


def test_resolver_rejects_missing_dependencies_after_duplicate_validation() -> None:
    with pytest.raises(MissingExtensionDependencyError) as error:
        TopologicalExtensionResolver().resolve(
            [PlainSpecExtension(extension("consumer", "missing"))],
        )

    assert error.value.extension_id == ExtensionId("consumer")
    assert error.value.required_id == ExtensionId("missing")


@pytest.mark.parametrize(
    ("items", "expected_cycle"),
    [
        ([extension("self", "self")], ("self", "self")),
        (
            [extension("alpha", "beta"), extension("beta", "gamma"), extension("gamma", "alpha")],
            ("alpha", "beta", "gamma", "alpha"),
        ),
    ],
)
def test_resolver_reports_deterministic_closed_cycles(
    items: list[ExtensionSpec],
    expected_cycle: tuple[str, ...],
) -> None:
    with pytest.raises(ExtensionDependencyCycleError) as error:
        TopologicalExtensionResolver().resolve(
            PlainSpecExtension(spec) for spec in items
        )

    assert tuple(item.value for item in error.value.cycle) == expected_cycle


@dataclass(frozen=True, slots=True)
class PlainSpecExtension:
    spec: ExtensionSpec

    def install(self, registrar: ExtensionRegistrar, /) -> None:
        raise AssertionError("resolver must not install extensions")
