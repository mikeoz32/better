"""Better-owned model request, response and streaming contracts."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")


def _require_json_object(value: str, field_name: str) -> None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")


@dataclass(frozen=True, slots=True)
class ModelToolDefinition:
    name: str
    description: str
    parameters_json_schema: str

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "tool name")
        if not isinstance(self.description, str):
            raise TypeError("tool description must be a string")
        _require_json_object(self.parameters_json_schema, "parameters_json_schema")


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "tool call id")
        _require_non_empty(self.name, "tool call name")
        _require_json_object(self.arguments_json, "arguments_json")


@dataclass(frozen=True, slots=True)
class ModelSystemMessage:
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("system message content must be a string")


@dataclass(frozen=True, slots=True)
class ModelUserMessage:
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("user message content must be a string")


@dataclass(frozen=True, slots=True)
class ModelAssistantMessage:
    text: str = ""
    tool_calls: tuple[ModelToolCall, ...] = ()

    def __post_init__(self) -> None:
        calls = tuple(self.tool_calls)
        object.__setattr__(self, "tool_calls", calls)
        if not all(isinstance(call, ModelToolCall) for call in calls):
            raise TypeError("assistant tool calls must contain ModelToolCall values")
        call_ids = [call.id for call in calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool call ids must be unique")


@dataclass(frozen=True, slots=True)
class ModelToolResultMessage:
    call_id: str
    name: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.call_id, "tool result call id")
        _require_non_empty(self.name, "tool result name")
        if not isinstance(self.content, str):
            raise TypeError("tool result content must be a string")


type ModelMessage = (
    ModelSystemMessage | ModelUserMessage | ModelAssistantMessage | ModelToolResultMessage
)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[ModelMessage, ...]
    tools: tuple[ModelToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        tools = tuple(self.tools)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)
        if not all(
            isinstance(
                message,
                (
                    ModelSystemMessage,
                    ModelUserMessage,
                    ModelAssistantMessage,
                    ModelToolResultMessage,
                ),
            )
            for message in messages
        ):
            raise TypeError("messages must contain Better model message values")
        if not all(isinstance(tool, ModelToolDefinition) for tool in tools):
            raise TypeError("tools must contain ModelToolDefinition values")
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("model tool definition names must be unique")


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer or None")


class ModelFailureKind(StrEnum):
    PROVIDER = "provider"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class ModelFailure:
    kind: ModelFailureKind
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ModelFailureKind):
            raise TypeError("model failure kind must be ModelFailureKind")
        if not isinstance(self.message, str):
            raise TypeError("model failure message must be a string")


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    delta: str

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str) or not self.delta:
            raise ValueError("model text delta must not be empty")


@dataclass(frozen=True, slots=True)
class ModelThinkingDelta:
    delta: str

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str) or not self.delta:
            raise ValueError("model thinking delta must not be empty")


@dataclass(frozen=True, slots=True)
class ModelToolCallReady:
    call: ModelToolCall

    def __post_init__(self) -> None:
        if not isinstance(self.call, ModelToolCall):
            raise TypeError("tool call ready must contain a ModelToolCall")


@dataclass(frozen=True, slots=True)
class ModelUsageUpdated:
    usage: ModelUsage

    def __post_init__(self) -> None:
        if not isinstance(self.usage, ModelUsage):
            raise TypeError("usage update must contain ModelUsage")


@dataclass(frozen=True, slots=True)
class ModelResponseCompleted:
    response: ModelAssistantMessage
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, ModelAssistantMessage):
            raise TypeError("completed response must contain ModelAssistantMessage")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("completed response usage must contain ModelUsage")


@dataclass(frozen=True, slots=True)
class ModelResponseFailed:
    failure: ModelFailure
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.failure, ModelFailure):
            raise TypeError("failed response must contain ModelFailure")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("failed response usage must contain ModelUsage")


type ModelEvent = (
    ModelTextDelta
    | ModelThinkingDelta
    | ModelToolCallReady
    | ModelUsageUpdated
    | ModelResponseCompleted
    | ModelResponseFailed
)


class ModelRuntime(Protocol):
    def run(
        self,
        request: ModelRequest,
        /,
    ) -> AsyncIterator[ModelEvent]: ...
