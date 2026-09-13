"""Typed extension contracts and deterministic dependency resolution."""

import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from better_agent.kernel.contracts import (
    Command,
    CommandHandler,
    Event,
    EventHandler,
    ExecutionPlanner,
    ExtensionId,
)
from better_agent.kernel.errors import (
    DuplicateExtensionIdError,
    ExtensionDependencyCycleError,
    MissingExtensionDependencyError,
)


@dataclass(frozen=True, slots=True)
class ExtensionSpec:
    """Immutable identity and dependency declaration for one extension."""

    id: ExtensionId
    requires: tuple[ExtensionId, ...] = ()


class ExtensionRegistrar(Protocol):
    """Registration-only surface exposed while installing an extension."""

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        /,
    ) -> None: ...

    def handle[C: Command](
        self,
        command_type: type[C],
        handler: CommandHandler[C],
        /,
        *,
        execution: ExecutionPlanner[C] | None = None,
    ) -> None: ...


class Extension(Protocol):
    """Structural extension contract with synchronous installation."""

    @property
    def spec(self) -> ExtensionSpec: ...

    def install(self, registrar: ExtensionRegistrar, /) -> None: ...


class ExtensionResolver(Protocol):
    """Resolve extension dependencies into an activation order."""

    def resolve(
        self,
        extensions: Iterable[Extension],
        /,
    ) -> tuple[Extension, ...]: ...


class TopologicalExtensionResolver:
    """Pure, deterministic topological resolver for extension dependencies."""

    def resolve(
        self,
        extensions: Iterable[Extension],
        /,
    ) -> tuple[Extension, ...]:
        materialized = tuple(extensions)
        by_id: dict[ExtensionId, Extension] = {}
        specs: dict[ExtensionId, ExtensionSpec] = {}
        duplicate_ids: set[ExtensionId] = set()
        for extension in materialized:
            spec = extension.spec
            extension_id = spec.id
            if extension_id in by_id:
                duplicate_ids.add(extension_id)
            else:
                by_id[extension_id] = extension
                specs[extension_id] = spec

        if duplicate_ids:
            duplicate_id = min(duplicate_ids, key=lambda item: item.value)
            raise DuplicateExtensionIdError(duplicate_id)

        requirements = {
            extension_id: set(spec.requires)
            for extension_id, spec in specs.items()
        }
        for extension_id in sorted(by_id, key=lambda item: item.value):
            for required_id in sorted(requirements[extension_id], key=lambda item: item.value):
                if required_id not in by_id:
                    raise MissingExtensionDependencyError(extension_id, required_id)

        dependents: dict[ExtensionId, set[ExtensionId]] = {
            extension_id: set() for extension_id in by_id
        }
        remaining_requirements = {}
        for extension_id, required_ids in requirements.items():
            remaining_requirements[extension_id] = set(required_ids)
            for required_id in required_ids:
                dependents[required_id].add(extension_id)

        ready = [extension_id.value for extension_id, required_ids in remaining_requirements.items() if not required_ids]
        heapq.heapify(ready)
        ordered_ids: list[ExtensionId] = []
        by_value = {extension_id.value: extension_id for extension_id in by_id}
        while ready:
            extension_id = by_value[heapq.heappop(ready)]
            ordered_ids.append(extension_id)
            for dependent_id in sorted(dependents[extension_id], key=lambda item: item.value):
                remaining_requirements[dependent_id].remove(extension_id)
                if not remaining_requirements[dependent_id]:
                    heapq.heappush(ready, dependent_id.value)

        if len(ordered_ids) != len(by_id):
            remaining = set(by_id) - set(ordered_ids)
            cycle = _find_cycle(remaining, requirements)
            raise ExtensionDependencyCycleError(cycle)

        return tuple(by_id[extension_id] for extension_id in ordered_ids)


def _find_cycle(
    remaining: set[ExtensionId],
    requirements: dict[ExtensionId, set[ExtensionId]],
) -> tuple[ExtensionId, ...]:
    def visit(
        extension_id: ExtensionId,
        path: list[ExtensionId],
        active: set[ExtensionId],
    ) -> tuple[ExtensionId, ...] | None:
        if extension_id in active:
            start = path.index(extension_id)
            return tuple(path[start:] + [extension_id])

        active.add(extension_id)
        path.append(extension_id)
        for required_id in sorted(requirements[extension_id], key=lambda item: item.value):
            if required_id in remaining:
                cycle = visit(required_id, path, active)
                if cycle is not None:
                    return cycle
        path.pop()
        active.remove(extension_id)
        return None

    for extension_id in sorted(remaining, key=lambda item: item.value):
        cycle = visit(extension_id, [], set())
        if cycle is not None:
            return cycle
    raise RuntimeError("dependency graph was cyclic but no cycle was found")
