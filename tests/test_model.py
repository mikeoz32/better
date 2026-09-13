import asyncio
from dataclasses import FrozenInstanceError

import pytest

from better_agent import (
    ModelAssistantMessage,
    ModelEvent,
    ModelFailure,
    ModelFailureKind,
    ModelRequest,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelRuntime,
    ModelSystemMessage,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelToolCall,
    ModelToolCallReady,
    ModelToolDefinition,
    ModelToolResultMessage,
    ModelUsage,
    ModelUsageUpdated,
    ModelUserMessage,
)
from tests.support.model import ScriptedModelRuntime, ScriptedModelStep


def tool_definition(name: str = "lookup") -> ModelToolDefinition:
    return ModelToolDefinition(
        name=name,
        description="Look something up",
        parameters_json_schema='{"type":"object","properties":{}}',
    )


def tool_call(call_id: str = "call-1") -> ModelToolCall:
    return ModelToolCall(call_id, "lookup", '{"query":"value"}')


def request() -> ModelRequest:
    return ModelRequest(
        messages=(ModelUserMessage("Find a value"),),
        tools=(tool_definition(),),
    )


def test_model_values_are_immutable_and_request_history_is_replayable() -> None:
    assistant = ModelAssistantMessage(
        text="I will look that up",
        tool_calls=(tool_call(),),
    )
    result = ModelToolResultMessage("call-1", "lookup", "found")
    continuation = ModelRequest(request().messages + (assistant, result), request().tools)

    assert continuation.messages == (
        ModelUserMessage("Find a value"),
        assistant,
        result,
    )
    assert continuation.tools == request().tools
    with pytest.raises(FrozenInstanceError):
        setattr(continuation, "messages", ())


def test_model_request_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        ModelRequest(messages=(), tools=(tool_definition(), tool_definition()))


@pytest.mark.parametrize(
    "value",
    [
        ("", "lookup", "{}"),
        ("call-1", "", "{}"),
        ("call-1", "lookup", "[]"),
        ("call-1", "lookup", "not-json"),
    ],
)
def test_tool_call_requires_stable_identity_and_complete_json_object(
    value: tuple[str, str, str],
) -> None:
    with pytest.raises(ValueError):
        ModelToolCall(*value)


def test_assistant_tool_call_ids_must_be_non_empty_and_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        ModelAssistantMessage(tool_calls=(tool_call(), tool_call()))


def test_tool_definition_requires_json_object_schema() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        ModelToolDefinition("lookup", "Look something up", "[]")


def test_response_can_contain_text_and_tool_calls_and_usage_is_optional() -> None:
    response = ModelAssistantMessage(text="Here is the answer", tool_calls=(tool_call(),))

    assert ModelResponseCompleted(response).usage is None
    assert (
        ModelResponseCompleted(response, ModelUsage(input_tokens=4, output_tokens=2)).response
        == response
    )


@pytest.mark.parametrize("event_type", [ModelTextDelta, ModelThinkingDelta])
def test_empty_stream_deltas_are_rejected(
    event_type: type[ModelTextDelta] | type[ModelThinkingDelta],
) -> None:
    with pytest.raises(ValueError, match="empty"):
        event_type("")


def test_model_runtime_is_structural_and_streams_scripted_events_in_order() -> None:
    events: tuple[ModelEvent, ...] = (
        ModelThinkingDelta("thinking"),
        ModelTextDelta("answer"),
        ModelToolCallReady(tool_call()),
        ModelUsageUpdated(ModelUsage(input_tokens=3, output_tokens=1)),
        ModelResponseCompleted(ModelAssistantMessage("answer", (tool_call(),))),
    )
    runtime: ModelRuntime = ScriptedModelRuntime((ScriptedModelStep(events),))

    async def collect() -> list[ModelEvent]:
        return [event async for event in runtime.run(request())]

    assert asyncio.run(collect()) == list(events)


@pytest.mark.parametrize(
    "events",
    [
        (),
        (ModelTextDelta("text"),),
        (ModelResponseCompleted(ModelAssistantMessage()), ModelTextDelta("late")),
        (
            ModelResponseCompleted(ModelAssistantMessage()),
            ModelResponseFailed(ModelFailure(ModelFailureKind.MODEL, "failed")),
        ),
    ],
)
def test_scripted_runtime_requires_exactly_one_terminal_last(
    events: tuple[ModelEvent, ...],
) -> None:
    runtime = ScriptedModelRuntime((ScriptedModelStep(events),))

    with pytest.raises(AssertionError, match="terminal"):
        runtime.run(request())


def test_scripted_runtime_records_exact_requests_and_exhaustion_fails_immediately() -> None:
    first = request()
    second = ModelRequest((ModelSystemMessage("system"), ModelUserMessage("next")))
    runtime = ScriptedModelRuntime(
        (ScriptedModelStep((ModelResponseCompleted(ModelAssistantMessage("ok")),), first),),
    )

    stream = runtime.run(first)
    assert runtime.requests == [first]
    asyncio.run(stream.__anext__())
    with pytest.raises(AssertionError, match="exhausted"):
        runtime.run(second)


def test_scripted_runtime_expected_request_mismatch_fails_without_silent_replay() -> None:
    expected = request()
    actual = ModelRequest((ModelUserMessage("different"),))
    runtime = ScriptedModelRuntime(
        (ScriptedModelStep((ModelResponseCompleted(ModelAssistantMessage()),), expected),),
    )

    with pytest.raises(AssertionError, match="expected request"):
        runtime.run(actual)


def test_scripted_runtime_models_two_invocation_tool_continuation() -> None:
    first_request = request()
    calls = (tool_call("call-1"), tool_call("call-2"))
    assistant = ModelAssistantMessage(tool_calls=calls)
    results = (
        ModelToolResultMessage("call-1", "lookup", "first"),
        ModelToolResultMessage("call-2", "lookup", "second", is_error=True),
    )
    continuation = ModelRequest(
        messages=first_request.messages + (assistant,) + results,
        tools=first_request.tools,
    )
    runtime = ScriptedModelRuntime(
        (
            ScriptedModelStep(
                (ModelResponseCompleted(assistant),),
                expected_request=first_request,
            ),
            ScriptedModelStep(
                (ModelResponseCompleted(ModelAssistantMessage("final answer")),),
                expected_request=continuation,
            ),
        ),
    )

    async def invoke_twice() -> tuple[list[ModelEvent], list[ModelEvent]]:
        first_events = [event async for event in runtime.run(first_request)]
        continuation_events = [event async for event in runtime.run(continuation)]
        return first_events, continuation_events

    first_events, continuation_events = asyncio.run(invoke_twice())

    assert first_events == [ModelResponseCompleted(assistant)]
    assert continuation_events == [ModelResponseCompleted(ModelAssistantMessage("final answer"))]
    assert runtime.requests == [first_request, continuation]


def test_typed_model_failure_is_distinct_from_unexpected_exception() -> None:
    failure = ModelResponseFailed(
        ModelFailure(ModelFailureKind.PROVIDER, "provider rejected request"),
        ModelUsage(input_tokens=2),
    )
    runtime = ScriptedModelRuntime((ScriptedModelStep((failure,)),))

    async def collect() -> list[ModelEvent]:
        return [event async for event in runtime.run(request())]

    assert asyncio.run(collect()) == [failure]

    class UnexpectedRuntime:
        def run(self, _: ModelRequest, /):
            async def stream():
                raise RuntimeError("programming error")
                yield ModelTextDelta("unreachable")

            return stream()

    async def collect_unexpected() -> None:
        async for _ in UnexpectedRuntime().run(request()):
            pass

    with pytest.raises(RuntimeError, match="programming error"):
        asyncio.run(collect_unexpected())


def test_model_runtime_cancellation_is_not_translated() -> None:
    class CancellingRuntime:
        def run(self, _: ModelRequest, /):
            async def stream():
                raise asyncio.CancelledError
                yield ModelTextDelta("unreachable")

            return stream()

    async def collect() -> None:
        async for _ in CancellingRuntime().run(request()):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collect())
