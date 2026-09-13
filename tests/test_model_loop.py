from dataclasses import dataclass

import pytest

from better_agent import (
    Command,
    CommandBinding,
    DefaultEnvelopeFactory,
    Envelope,
    Harness,
    InlineExecutionScheduler,
    ModelAssistantMessage,
    ModelRequest,
    ModelResponseCompleted,
    ModelToolCall,
    ModelToolResultMessage,
    ModelUserMessage,
    Origin,
    RunLimits,
    RunCompleted,
    RunPump,
)
from better_agent.kernel.model_loop import ModelLoopPump, ModelToolCallCommand, ModelToolResult
from better_agent.kernel.runtime import InMemoryCommandBus, InMemoryEventBus
from tests.support.model import ScriptedModelRuntime, ScriptedModelStep


@dataclass(frozen=True, slots=True)
class _Start(Command):
    pass


@pytest.mark.asyncio
async def test_model_loop_appends_tool_results_in_call_order() -> None:
    first = ModelRequest((ModelUserMessage("find"),))
    calls = (ModelToolCall("a", "lookup", "{}"), ModelToolCall("b", "lookup", "{}"))
    assistant = ModelAssistantMessage(tool_calls=calls)
    continuation = ModelRequest(
        first.messages
        + (
            assistant,
            ModelToolResultMessage("a", "lookup", "A"),
            ModelToolResultMessage("b", "lookup", "B"),
        )
    )
    runtime = ScriptedModelRuntime(
        (
            ScriptedModelStep((ModelResponseCompleted(assistant),)),
            ScriptedModelStep(
                (ModelResponseCompleted(ModelAssistantMessage("done")),), continuation
            ),
        )
    )
    command_bus = InMemoryCommandBus(InlineExecutionScheduler())

    async def tool(command: Envelope[ModelToolCallCommand]):
        yield ModelToolResult(
            command.payload.call.id, command.payload.call.name, command.payload.call.id.upper()
        )

    command_bus.bind(ModelToolCallCommand, CommandBinding(tool))
    pump: RunPump = ModelLoopPump(runtime, lambda _: first, command_bus, DefaultEnvelopeFactory())
    events = [
        event
        async for event in pump.run_envelope(
            DefaultEnvelopeFactory().create(_Start(), origin=Origin(component="test"))
        )
    ]
    assert runtime.requests == [first, continuation]
    assert [
        event.payload.content for event in events if isinstance(event.payload, ModelToolResult)
    ] == ["A", "B"]


@pytest.mark.asyncio
async def test_model_loop_is_usable_by_harness() -> None:
    runtime = ScriptedModelRuntime(
        (ScriptedModelStep((ModelResponseCompleted(ModelAssistantMessage("done")),)),)
    )
    bus = InMemoryCommandBus(InlineExecutionScheduler())

    async def commit(command):
        yield RunCompleted(command.payload.outcome)

    from better_agent import CommitRunOutcome

    bus.bind(CommitRunOutcome, CommandBinding(commit))
    harness = Harness(
        InMemoryEventBus(),
        bus,
        DefaultEnvelopeFactory(),
        run_pump=ModelLoopPump(
            runtime,
            lambda _: ModelRequest((ModelUserMessage("hi"),)),
            bus,
            DefaultEnvelopeFactory(),
        ),
    )
    result = [
        event
        async for event in harness.run(
            _Start(), origin=Origin(component="test"), limits=RunLimits(2)
        )
    ]
    assert isinstance(result[-1].payload, RunCompleted)
