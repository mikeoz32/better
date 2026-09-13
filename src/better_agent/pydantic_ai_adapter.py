"""Pydantic AI adapter for the Better model runtime contract."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from better_agent.kernel.model import (
    ModelAssistantMessage,
    ModelFailure,
    ModelFailureKind,
    ModelRequest,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelEvent,
    ModelRuntime,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelToolCall,
    ModelToolCallReady,
    ModelUsage,
    ModelUsageUpdated,
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
                model_settings={"tools": _tools(request)},
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
        except asyncio.CancelledError:
            raise
        except Exception as error:
            yield ModelResponseFailed(ModelFailure(ModelFailureKind.PROVIDER, str(error)), usage)
            return
        yield ModelResponseCompleted(ModelAssistantMessage(text, tuple(calls)), usage)


def _messages(request: ModelRequest) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in request.messages:
        if hasattr(message, "content"):
            role = "system" if message.__class__.__name__ == "ModelSystemMessage" else "user"
            result.append({"role": role, "content": str(message.content)})
        elif isinstance(message, ModelAssistantMessage):
            result.append({"role": "assistant", "content": message.text})
    return result


def _tools(request: ModelRequest) -> list[dict[str, object]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": json.loads(tool.parameters_json_schema),
        }
        for tool in request.tools
    ]


def _event(item: Any) -> ModelEvent | None:
    name = item.__class__.__name__.lower()
    if "thinking" in name or "reasoning" in name:
        value = getattr(item, "delta", getattr(item, "content", ""))
        return ModelThinkingDelta(value) if value else None
    if "text" in name or "partdelta" in name:
        value = getattr(item, "delta", getattr(item, "content", ""))
        return ModelTextDelta(value) if value else None
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
