"""Provider-neutral bridge between the model contract and the Better runtime."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from better_agent.kernel.contracts import Command, CommandBus, Envelope, EnvelopeFactory, Event
from better_agent.kernel.errors import StepBudgetLimitReached
from better_agent.kernel.model import (
    ModelAssistantMessage,
    ModelEvent,
    ModelRequest,
    ModelResponseCompleted,
    ModelToolCall,
    ModelToolResultMessage,
    ModelRuntime,
)


@dataclass(frozen=True, slots=True)
class ModelEventObserved(Event):
    event: ModelEvent


@dataclass(frozen=True, slots=True)
class ModelToolCallCommand(Command):
    call: ModelToolCall


@dataclass(frozen=True, slots=True)
class ModelToolResult(Event):
    call_id: str
    name: str
    content: str
    is_error: bool = False

    def message(self) -> ModelToolResultMessage:
        return ModelToolResultMessage(self.call_id, self.name, self.content, self.is_error)


class ModelLoopPump:
    """Run model turns and route each tool call through the command boundary."""

    def __init__(
        self,
        model: ModelRuntime,
        request_for: Callable[[Command], ModelRequest],
        command_bus: CommandBus,
        envelope_factory: EnvelopeFactory,
    ) -> None:
        self._model = model
        self._request_for = request_for
        self._command_bus = command_bus
        self._factory = envelope_factory

    def run_envelope[C: Command](
        self, command: Envelope[C], *, max_steps: int | None = None
    ) -> AsyncIterator[Envelope[Event]]:
        async def stream() -> AsyncIterator[Envelope[Event]]:
            request = self._request_for(command.payload)
            history = request.messages
            turn = 0
            while True:
                turn += 1
                if max_steps is not None and turn > max_steps:
                    raise StepBudgetLimitReached(max_steps, turn, command)
                response: ModelAssistantMessage | None = None
                async for model_event in self._model.run(ModelRequest(history, request.tools)):
                    yield self._factory.create(
                        ModelEventObserved(model_event), origin=command.origin, cause=command
                    )
                    if isinstance(model_event, ModelResponseCompleted):
                        response = model_event.response
                if response is None or not response.tool_calls:
                    return
                history += (response,)
                for call in response.tool_calls:
                    tool_command = self._factory.create(
                        ModelToolCallCommand(call), origin=command.origin, cause=command
                    )
                    results = [event async for event in self._command_bus.dispatch(tool_command)]
                    if len(results) != 1 or not isinstance(results[0], ModelToolResult):
                        raise RuntimeError("tool handler must emit exactly one ModelToolResult")
                    result = results[0]
                    yield self._factory.create(result, origin=command.origin, cause=tool_command)
                    history += (result.message(),)

        return stream()
