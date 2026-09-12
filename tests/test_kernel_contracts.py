from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError, dataclass
from typing import cast

import pytest

from better_agent.kernel import (
    Command,
    CommandBinding,
    CorrelationId,
    Envelope,
    Event,
    ExtensionId,
    MessageId,
    Origin,
)
from better_agent import Message
from tests.support.kernel import (
    RecordingCommandBus,
    RecordingEnvelopeFactory,
    RecordingEventBus,
    RecordingRuntimePort,
    exclusive_claims,
)


@dataclass(frozen=True, slots=True)
class PromptReceived(Event):
    text: str


@dataclass(frozen=True, slots=True)
class RunPrompt(Command):
    text: str


async def handle_prompt(command: Envelope[RunPrompt]) -> AsyncIterator[Event]:
    yield PromptReceived(command.payload.text)


def test_messages_are_immutable_and_runtime_distinct() -> None:
    event = PromptReceived("inspect")
    command = RunPrompt("inspect")

    assert isinstance(event, Event)
    assert isinstance(command, Command)
    assert Message is not None
    assert not isinstance(event, Command)
    assert not isinstance(command, Event)
    assert event.text == command.text

    with pytest.raises(FrozenInstanceError):
        setattr(event, "text", "mutate")


def test_identity_and_origin_values_are_typed_immutable_values() -> None:
    message_id = MessageId("message-1")
    correlation_id = CorrelationId("run-1")
    extension_id = ExtensionId("better.test")
    origin = Origin(component="test", extension_id=extension_id)

    assert message_id.value == "message-1"
    assert correlation_id.value == "run-1"
    assert origin.component == "test"
    assert origin.extension_id == extension_id
    assert hash(message_id) == hash(MessageId("message-1"))


def test_envelope_keeps_kernel_owned_trace_metadata() -> None:
    factory = RecordingEnvelopeFactory()
    origin = Origin(component="test")
    root = factory.create(RunPrompt("inspect"), origin=origin)
    child = factory.create(
        PromptReceived("inspect"),
        origin=origin,
        cause=cast(Envelope[Event | Command], root),
    )

    assert root.id == MessageId("message-1")
    assert root.correlation_id == CorrelationId("correlation-1")
    assert root.causation_id is None
    assert child.causation_id == root.id
    assert child.correlation_id == root.correlation_id
    assert child.origin == origin

    with pytest.raises(FrozenInstanceError):
        setattr(child, "origin", Origin(component="other"))


@pytest.mark.anyio
async def test_recording_kernel_doubles_capture_contract_interactions() -> None:
    event_bus = RecordingEventBus()
    command_bus = RecordingCommandBus()
    runtime_port = RecordingRuntimePort()

    def react(_: Envelope[PromptReceived]) -> tuple[Command, ...]:
        return (RunPrompt("continue"),)

    event_bus.subscribe(PromptReceived, react)
    command_bus.bind(
        RunPrompt,
        CommandBinding(handler=handle_prompt, execution=exclusive_claims),
    )

    factory = RecordingEnvelopeFactory()
    event = factory.create(PromptReceived("inspect"), origin=Origin(component="test"))
    command = factory.create(
        RunPrompt("continue"),
        origin=Origin(component="test"),
        cause=cast(Envelope[Event | Command], event),
    )

    await event_bus.publish(event)
    await command_bus.dispatch(command)
    message_id = await runtime_port.submit(RunPrompt("submitted"))

    assert event_bus.published == [event]
    assert command_bus.bindings[RunPrompt].execution(RunPrompt("run")).exclusive
    assert command_bus.dispatched == [command]
    assert runtime_port.submitted == [RunPrompt("submitted")]
    assert message_id == MessageId("submitted-1")
