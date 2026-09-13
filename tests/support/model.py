from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from better_agent import ModelEvent, ModelRequest, ModelResponseCompleted, ModelResponseFailed


@dataclass(frozen=True, slots=True)
class ScriptedModelStep:
    events: tuple[ModelEvent, ...]
    expected_request: ModelRequest | None = None


class ScriptedModelRuntime:
    """Deterministic test runtime for the Better-owned model contract."""

    def __init__(self, steps: Iterable[ScriptedModelStep]) -> None:
        self._steps = list(steps)
        self.requests: list[ModelRequest] = []

    def run(self, request: ModelRequest, /) -> AsyncIterator[ModelEvent]:
        if not self._steps:
            raise AssertionError("script exhausted")
        step = self._steps.pop(0)
        self.requests.append(request)
        if step.expected_request is not None and request != step.expected_request:
            raise AssertionError(
                f"expected request {step.expected_request!r}, got {request!r}",
            )
        _validate_terminal_event(step.events)

        async def stream() -> AsyncIterator[ModelEvent]:
            for event in step.events:
                yield event

        return stream()


def _validate_terminal_event(events: tuple[ModelEvent, ...]) -> None:
    terminal_types = (ModelResponseCompleted, ModelResponseFailed)
    terminals = tuple(event for event in events if isinstance(event, terminal_types))
    if len(terminals) != 1 or not isinstance(events[-1] if events else None, terminal_types):
        raise AssertionError("scripted model step must contain exactly one terminal event last")
