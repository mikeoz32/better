"""Pydantic AI adapter for the Better model runtime contract."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai.messages import (
    ModelRequest as PydanticModelRequest,
    ModelResponse as PydanticModelResponse,
    ModelResponsePart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from better_agent.kernel.model import (
    ModelAssistantMessage,
    ModelFailure,
    ModelFailureKind,
    ModelRequest,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelEvent,
    ModelRuntime,
    ModelSystemMessage,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelToolCall,
    ModelToolCallReady,
    ModelUsage,
    ModelUsageUpdated,
    ModelToolResultMessage,
    ModelUserMessage,
)


class PydanticAIRuntime(ModelRuntime):
    """Adapt an injected Pydantic AI agent without exposing provider objects."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def run(self, request: ModelRequest, /) -> AsyncIterator[ModelEvent]:
        return self._stream(request)

    async def _stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        text = ""
        calls: list[ModelToolCall] = []
        usage: ModelUsage | None = None
        try:
            async with self._agent.run_stream(
                message_history=_messages(request),
            ) as stream:
                source = stream.stream_events() if hasattr(stream, "stream_events") else stream
                async for item in source:
                    event = _event(item)
                    if isinstance(event, ModelTextDelta):
                        text += event.delta
                    elif isinstance(event, ModelToolCallReady):
                        calls.append(event.call)
                    elif isinstance(event, ModelUsageUpdated):
                        usage = event.usage
                    if event is not None:
                        yield event
                if usage is None:
                    usage = _usage(getattr(stream, "usage", None))
                    if usage is not None:
                        yield ModelUsageUpdated(usage)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            yield ModelResponseFailed(ModelFailure(ModelFailureKind.PROVIDER, str(error)), usage)
            return
        yield ModelResponseCompleted(ModelAssistantMessage(text, tuple(calls)), usage)


def _messages(request: ModelRequest) -> list[object]:
    result: list[object] = []
    for message in request.messages:
        if isinstance(message, ModelSystemMessage):
            result.append(PydanticModelRequest(parts=[SystemPromptPart(message.content)]))
        elif isinstance(message, ModelUserMessage):
            result.append(PydanticModelRequest(parts=[UserPromptPart(message.content)]))
        elif isinstance(message, ModelAssistantMessage):
            parts: list[ModelResponsePart] = []
            if message.text:
                parts.append(TextPart(message.text))
            parts.extend(
                ToolCallPart(
                    tool_name=call.name,
                    args=call.arguments_json,
                    tool_call_id=call.id,
                )
                for call in message.tool_calls
            )
            result.append(PydanticModelResponse(parts=parts))
        elif isinstance(message, ModelToolResultMessage):
            result.append(
                PydanticModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=message.name,
                            content=message.content,
                            tool_call_id=message.call_id,
                            outcome="failed" if message.is_error else "success",
                        ),
                    ],
                ),
            )
    return result


def _event(item: Any) -> ModelEvent | None:
    if isinstance(item, PartStartEvent):
        if isinstance(item.part, TextPart):
            return _text_delta(item.part.content)
        if isinstance(item.part, ThinkingPart):
            return _thinking_delta(item.part.content)
        return None
    if isinstance(item, PartDeltaEvent):
        if isinstance(item.delta, TextPartDelta):
            return _text_delta(item.delta.content_delta)
        if isinstance(item.delta, ThinkingPartDelta):
            return _thinking_delta(item.delta.content_delta)
        return None
    if isinstance(item, PartEndEvent) and isinstance(item.part, ToolCallPart):
        arguments = item.part.args
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        return ModelToolCallReady(
            ModelToolCall(item.part.tool_call_id, item.part.tool_name, arguments),
        )

    name = item.__class__.__name__.lower()
    if "thinking" in name or "reasoning" in name:
        return _thinking_delta(getattr(item, "delta", getattr(item, "content", "")))
    if "text" in name or "partdelta" in name:
        return _text_delta(getattr(item, "delta", getattr(item, "content", "")))
    if "tool" in name and (hasattr(item, "id") or hasattr(item, "tool_call_id")):
        call_id = getattr(item, "id", getattr(item, "tool_call_id", ""))
        arguments = getattr(item, "args", getattr(item, "arguments", "{}"))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        return ModelToolCallReady(ModelToolCall(call_id, item.name, arguments))
    if "usage" in name:
        return ModelUsageUpdated(
            ModelUsage(getattr(item, "input_tokens", None), getattr(item, "output_tokens", None))
        )
    return None


def _text_delta(value: object) -> ModelTextDelta | None:
    return ModelTextDelta(value) if isinstance(value, str) and value else None


def _thinking_delta(value: object) -> ModelThinkingDelta | None:
    return ModelThinkingDelta(value) if isinstance(value, str) and value else None


def _usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    if callable(value):
        value = value()
    input_tokens = getattr(value, "input_tokens", None)
    output_tokens = getattr(value, "output_tokens", None)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return ModelUsage(input_tokens, output_tokens)
    return None
