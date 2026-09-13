import asyncio
from dataclasses import dataclass
from typing import cast

from pydantic_ai.messages import (
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)

from better_agent import (
    ModelRequest,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelAssistantMessage,
    ModelSystemMessage,
    ModelTextDelta,
    ModelThinkingDelta,
    ModelToolCallReady,
    ModelUsageUpdated,
    ModelUserMessage,
    ModelUsage,
    ModelToolCall,
    ModelToolDefinition,
    ModelToolResultMessage,
)
from better_agent.pydantic_ai_adapter import PydanticAIRuntime


@dataclass
class TextDeltaEvent:
    delta: str


@dataclass
class ThinkingDeltaEvent:
    delta: str


@dataclass
class ToolCallEvent:
    id: str
    name: str
    args: str


@dataclass
class UsageEvent:
    input_tokens: int
    output_tokens: int


class FakeAgent:
    def __init__(self, events: list[object] | Exception) -> None:
        self.events = events
        self.calls: list[dict[str, object]] = []

    def run_stream(self, **kwargs: object):
        self.calls.append(kwargs)
        events = self.events

        class Stream:
            async def __aenter__(self):
                if isinstance(events, Exception):
                    raise events
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            def __aiter__(self):
                async def values():
                    for event in events if not isinstance(events, Exception) else ():
                        yield event

                return values()

            def stream_events(self):
                return self.__aiter__()

            @property
            def usage(self):
                return UsageEvent(4, 2)

        return Stream()


def test_adapter_translates_stream_events_to_better_events() -> None:
    agent = FakeAgent(
        [
            ThinkingDeltaEvent("reason"),
            TextDeltaEvent("answer"),
            ToolCallEvent("call-1", "lookup", '{"q":"x"}'),
            UsageEvent(4, 2),
        ],
    )
    runtime = PydanticAIRuntime(agent)

    events = asyncio.run(
        _collect(runtime.run(ModelRequest((ModelUserMessage("question"),)))),
    )

    assert events == [
        ModelThinkingDelta("reason"),
        ModelTextDelta("answer"),
        ModelToolCallReady(ModelToolCall("call-1", "lookup", '{"q":"x"}')),
        ModelUsageUpdated(ModelUsage(4, 2)),
        ModelResponseCompleted(
            ModelAssistantMessage("answer", (ModelToolCall("call-1", "lookup", '{"q":"x"}'),)),
            ModelUsage(4, 2),
        ),
    ]


def test_provider_exception_becomes_provider_failure() -> None:
    runtime = PydanticAIRuntime(FakeAgent(RuntimeError("offline")))
    events = asyncio.run(_collect(runtime.run(ModelRequest((ModelUserMessage("question"),)))))
    assert isinstance(events[0], ModelResponseFailed)
    assert events[0].failure.message == "offline"


def test_real_pydantic_stream_parts_translate_without_leaking_upstream_types() -> None:
    agent = FakeAgent(
        [
            PartStartEvent(index=0, part=ThinkingPart(content="reason")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="ing")),
            PartStartEvent(index=1, part=TextPart(content="ans")),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="wer")),
            PartEndEvent(
                index=2,
                part=ToolCallPart(
                    tool_name="lookup",
                    args={"q": "x"},
                    tool_call_id="call-1",
                ),
            ),
        ],
    )
    runtime = PydanticAIRuntime(agent)

    events = asyncio.run(
        _collect(
            runtime.run(
                ModelRequest(
                    (
                        ModelSystemMessage("be concise"),
                        ModelUserMessage("question"),
                        ModelAssistantMessage("old"),
                        ModelToolResultMessage("old-call", "lookup", "old result"),
                    ),
                    (ModelToolDefinition("lookup", "look up", '{"type":"object"}'),),
                ),
            ),
        ),
    )

    assert events == [
        ModelThinkingDelta("reason"),
        ModelThinkingDelta("ing"),
        ModelTextDelta("ans"),
        ModelTextDelta("wer"),
        ModelToolCallReady(ModelToolCall("call-1", "lookup", '{"q": "x"}')),
        ModelUsageUpdated(ModelUsage(4, 2)),
        ModelResponseCompleted(
            ModelAssistantMessage("answer", (ModelToolCall("call-1", "lookup", '{"q": "x"}'),)),
            ModelUsage(4, 2),
        ),
    ]
    history = cast(list[object], agent.calls[0]["message_history"])
    assert [type(message).__name__ for message in history] == [
        "ModelRequest",
        "ModelRequest",
        "ModelResponse",
        "ModelRequest",
    ]
    assert "tools" not in agent.calls[0]


async def _collect(stream):
    return [event async for event in stream]
