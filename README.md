# Better Agent

Better Agent is a Python-native coding-agent harness built as an event-command
microkernel with bundled core extensions. The kernel is library-first and keeps
feature behavior behind typed event, command and extension contracts.

## Current Slice

The first implementation slice establishes:

- immutable `Event`, `Command` and `Envelope` message contracts;
- typed message identity and origin values;
- kernel-owned correlation and causation metadata;
- event bus, command bus, runtime ingress and scheduler-facing protocols;
- typed execution claims and a capability-aware scheduler with conservative defaults;
- deterministic in-memory event/command buses and a non-recursive runtime pump;
- streamed command events with propagated origin and causation metadata;
- explicit run lifecycle, semantic cancellation and step-budget enforcement;
- typed runtime faults and gated terminal outcome finalization through `Harness`;
- structural extension contracts with deterministic dependency resolution;
- static and Python entry-point extension discovery with typed load diagnostics;
- staged extension host startup with explicit failure and lifecycle semantics;
- Better-owned model request/response contracts with deterministic scripted runtime tests;
- deterministic recording test doubles;
- offline event-dispatch regression benchmark;
- the `better-agent` distribution and `ba` CLI entry point.

The agent loop, Pydantic AI adapter, bundled tools, sessions and TUI are
implemented in subsequent backlog slices. The model boundary is deliberately
provider-neutral; Pydantic AI integration belongs to BA-14.

## Development

Better Agent targets Python 3.14 and uses `uv` for environment and command
execution:

```bash
uv sync
uv run ba --help
uv run pytest
uv run ruff check .
uv run ty check
```

The import package is `better_agent`; the distribution package is
`better-agent`; the command-line entry point is `ba`.

Execution claims are derived by each command binding's planner. Set
`exclusive=False` for operations that can overlap, then describe read and write
keys with `reads` and `writes`. Same-resource reads may overlap; any write
conflicts with reads or writes on that key. The default `ExecutionClaims()` is
unknown and is scheduled conservatively as exclusive.

## Architecture

The kernel owns lifecycle mechanics, typed event-command dispatch, message
envelopes, scheduling boundaries, extension resolution and cancellation. Core
capabilities such as tools, sessions, context, compaction, policy and TUI are
bundled extensions using the same public extension model as external packages.

Runtime composition wires the two public buses and the envelope factory into the pump:

```python
pump = RuntimePump(
    InMemoryEventBus(),
    InMemoryCommandBus(CapabilityScheduler()),
    DefaultEnvelopeFactory(),
)
```

The public run boundary is an async event stream. `RunStarted` is emitted first;
work then runs through the same injected buses, and terminal events are withheld
until `CommitRunOutcome` completes successfully:

```python
harness = Harness(event_bus, command_bus, DefaultEnvelopeFactory())
events = harness.run(command, origin=Origin(component="cli"))
async for envelope in events:
    handle(envelope)
```

`Harness.cancel(correlation_id)` is semantic run cancellation. Cancelling the
consumer task is transport abandonment and only guarantees cleanup of owned
work. `RunLimits(max_steps=...)` bounds primary work commands; finalization is
always separate from that budget.

Extensions declare an immutable `ExtensionSpec` and synchronously contribute
typed subscriptions and command bindings through `ExtensionRegistrar`. The
`TopologicalExtensionResolver` validates duplicate IDs, missing requirements
and cycles, then returns dependencies before dependents with lexical ordering
for otherwise independent extensions. `StaticExtensionSource` supports bundled
and test composition, while `EntryPointExtensionSource` discovers the fixed
`better_agent.extensions` group. `DefaultExtensionHost` resolves, stages and
replays extension registrations before marking the host started; optional load
failures are reported and required failures abort startup.

The model boundary uses replayable `ModelRequest` history and Better-owned
stream events. A `ModelRuntime` emits text/thinking deltas, complete tool calls,
usage updates and exactly one terminal response event. `ScriptedModelRuntime`
is test-only and does not import Pydantic AI or retain hidden continuation state.

The default verification suite is deterministic and offline. No FastAPI server,
database, Redis service or Docker Compose stack is part of the current product
surface.
