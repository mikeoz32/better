"""Public run lifecycle orchestration above the event-command runtime pump."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import cast

from better_agent.kernel.contracts import (
    Command,
    CommandBus,
    CorrelationId,
    Envelope,
    EnvelopeFactory,
    Event,
    EventBus,
    Message,
    Origin,
)
from better_agent.kernel.errors import (
    RunFinalizationError,
    RuntimeExecutionError,
    StepBudgetLimitReached,
)
from better_agent.kernel.lifecycle import (
    CancelledOutcome,
    CommitRunOutcome,
    CompletedOutcome,
    FailedOutcome,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunLifecycle,
    RunLimits,
    RunOutcome,
    RunOutcomeReady,
    RunStarted,
    RunState,
    RuntimeFault,
    RuntimeFaulted,
    StepBudgetExceeded,
)
from better_agent.kernel.runtime import RuntimePump


@dataclass(slots=True)
class _ActiveRun:
    lifecycle: RunLifecycle
    cancel_requested: asyncio.Event


def _find_error(error: BaseException, error_type: type[BaseException]) -> BaseException | None:
    if isinstance(error, error_type):
        return error
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            found = _find_error(nested, error_type)
            if found is not None:
                return found
    return None


class Harness:
    """Own run lifecycle, semantic cancellation and outcome finalization."""

    def __init__(
        self,
        event_bus: EventBus,
        command_bus: CommandBus,
        envelope_factory: EnvelopeFactory,
        *,
        runtime_pump: RuntimePump | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._command_bus = command_bus
        self._envelope_factory = envelope_factory
        self._runtime_pump = runtime_pump or RuntimePump(
            event_bus,
            command_bus,
            envelope_factory,
        )
        self._active_runs: dict[CorrelationId, _ActiveRun] = {}

    async def cancel(self, correlation_id: CorrelationId) -> None:
        run = self._active_runs.get(correlation_id)
        if run is not None and run.lifecycle.state is RunState.RUNNING:
            run.cancel_requested.set()

    async def run[C: Command](
        self,
        initial_command: C,
        *,
        origin: Origin,
        limits: RunLimits = RunLimits(),
    ) -> AsyncGenerator[Envelope[Event], None]:
        root = self._envelope_factory.create(initial_command, origin=origin)
        run = _ActiveRun(RunLifecycle(), asyncio.Event())
        self._active_runs[root.correlation_id] = run
        run.lifecycle.transition(RunState.RUNNING)
        work_stream: AsyncIterator[Envelope[Event]] = self._runtime_pump.run_envelope(
            root,
            before_command_start=self._step_counter(limits),
        )
        first_event_task = asyncio.create_task(anext(work_stream))
        last_message: Envelope[Message] = cast(Envelope[Message], root)

        try:
            started = self._envelope_factory.create(
                RunStarted(),
                origin=origin,
                cause=root,
            )
            last_message = cast(Envelope[Message], started)
            yield cast(Envelope[Event], started)

            outcome: RunOutcome
            try:
                async for event in self._primary_events(
                    work_stream,
                    run.cancel_requested,
                    initial_task=first_event_task,
                ):
                    last_message = cast(Envelope[Message], event)
                    yield event
                outcome = CompletedOutcome()
            except _SemanticCancellation:
                await work_stream.aclose()
                outcome = CancelledOutcome(reason="semantic cancellation requested")
            except asyncio.CancelledError:
                await work_stream.aclose()
                raise
            except Exception as error:
                await work_stream.aclose()
                budget_error = _find_error(error, StepBudgetLimitReached)
                if budget_error is not None:
                    budget_event = cast(StepBudgetLimitReached, budget_error).event
                    budget_envelope = self._envelope_factory.create(
                        cast(StepBudgetExceeded, budget_event),
                        origin=origin,
                        cause=last_message,
                    )
                    last_message = cast(Envelope[Message], budget_envelope)
                    yield cast(Envelope[Event], budget_envelope)
                    outcome = FailedOutcome(cast(StepBudgetExceeded, budget_event))
                else:
                    runtime_error = _find_error(error, RuntimeExecutionError)
                    if runtime_error is not None:
                        runtime_error = cast(RuntimeExecutionError, runtime_error)
                        phase = runtime_error.phase
                        original = runtime_error.__cause__ or runtime_error.error
                        cause = cast(Envelope[Message], runtime_error.cause)
                    else:
                        phase = "runtime"
                        original = error
                        cause = last_message
                    fault = RuntimeFault(
                        phase=phase,
                        exception_type=type(original).__name__,
                        message=str(original),
                    )
                    fault_envelope = self._envelope_factory.create(
                        RuntimeFaulted(fault),
                        origin=origin,
                        cause=cause,
                    )
                    last_message = cast(Envelope[Message], fault_envelope)
                    yield cast(Envelope[Event], fault_envelope)
                    outcome = FailedOutcome(fault)

            run.lifecycle.transition(RunState.FINALIZING)
            ready = self._envelope_factory.create(
                RunOutcomeReady(outcome),
                origin=origin,
                cause=last_message,
            )
            yield cast(Envelope[Event], ready)
            terminal = await self._commit(outcome, cast(Envelope[Event], ready), origin)
            if isinstance(outcome, CompletedOutcome):
                run.lifecycle.transition(RunState.COMPLETED)
            elif isinstance(outcome, FailedOutcome):
                run.lifecycle.transition(RunState.FAILED)
            else:
                run.lifecycle.transition(RunState.CANCELLED)
            yield terminal
        finally:
            first_event_task.cancel()
            await asyncio.gather(first_event_task, return_exceptions=True)
            await work_stream.aclose()
            self._active_runs.pop(root.correlation_id, None)

    def _step_counter(self, limits: RunLimits):
        steps = 0

        def before_command_start(_: Envelope[Command]) -> None:
            nonlocal steps
            attempted = steps + 1
            if limits.max_steps is not None and attempted > limits.max_steps:
                raise StepBudgetLimitReached(
                    StepBudgetExceeded(limits.max_steps, attempted),
                )
            steps = attempted

        return before_command_start

    async def _primary_events(
        self,
        stream: AsyncIterator[Envelope[Event]],
        cancel_requested: asyncio.Event,
        *,
        initial_task: asyncio.Task[Envelope[Event]] | None = None,
    ) -> AsyncIterator[Envelope[Event]]:
        async def read_next() -> Envelope[Event]:
            return await anext(stream)

        while True:
            next_task = initial_task or asyncio.create_task(read_next())
            initial_task = None
            cancel_task = asyncio.create_task(cancel_requested.wait())
            try:
                done, pending = await asyncio.wait(
                    (next_task, cancel_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                next_task.cancel()
                cancel_task.cancel()
                await asyncio.gather(next_task, cancel_task, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if cancel_task in done:
                next_task.cancel()
                await asyncio.gather(next_task, return_exceptions=True)
                raise _SemanticCancellation
            try:
                event = next_task.result()
            except StopAsyncIteration:
                return
            yield event

    async def _commit(
        self,
        outcome: RunOutcome,
        ready: Envelope[Event],
        origin: Origin,
    ) -> Envelope[Event]:
        command = self._envelope_factory.create(
            CommitRunOutcome(outcome),
            origin=origin,
            cause=ready,
        )
        expected: Event
        if isinstance(outcome, CompletedOutcome):
            expected = RunCompleted(outcome)
        elif isinstance(outcome, FailedOutcome):
            expected = RunFailed(outcome)
        else:
            expected = RunCancelled(outcome)

        observed: Envelope[Event] | None = None
        try:
            for_payloads = self._command_bus.dispatch(command)
            async for payload in for_payloads:
                if observed is not None:
                    raise RunFinalizationError("finalization handler emitted multiple terminal events")
                if payload != expected:
                    raise RunFinalizationError("finalization handler emitted mismatched terminal event")
                observed = self._envelope_factory.create(
                    payload,
                    origin=origin,
                    cause=command,
                )
        except asyncio.CancelledError:
            raise
        except RunFinalizationError:
            raise
        except BaseException as error:
            raise RunFinalizationError("finalization handler failed") from error
        if observed is None:
            raise RunFinalizationError("finalization handler emitted no terminal event")
        return observed


class _SemanticCancellation(Exception):
    pass
