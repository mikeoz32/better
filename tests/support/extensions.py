from typing import cast

from better_agent import (
    Command,
    CommandBinding,
    Event,
    EventHandler,
    ExecutionPlanner,
    ExtensionRegistrar,
)


class RecordingRegistrar(ExtensionRegistrar):
    """Record extension contributions without starting runtime infrastructure."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []
        self.subscriptions: list[tuple[type[Event], EventHandler[Event]]] = []
        self.handlers: list[tuple[type[Command], CommandBinding[Command]]] = []

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: EventHandler[E],
        /,
    ) -> None:
        self.calls.append(("subscribe", event_type, handler))
        self.subscriptions.append(
            (event_type, cast(EventHandler[Event], handler)),
        )

    def handle[C: Command](
        self,
        command_type: type[C],
        handler,
        /,
        *,
        execution: ExecutionPlanner[C] | None = None,
    ) -> None:
        binding = CommandBinding(handler, execution)
        self.calls.append(("handle", command_type, binding))
        self.handlers.append((command_type, binding))
