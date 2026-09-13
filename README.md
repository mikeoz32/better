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
- deterministic in-memory event/command buses and a non-recursive runtime pump;
- streamed command events with internal producer origin and causation metadata;
- deterministic recording test doubles;
- offline event-dispatch regression benchmark;
- the `better-agent` distribution and `ba` CLI entry point.

The agent loop, Pydantic AI adapter, bundled tools, sessions, TUI and external
extension discovery are implemented in subsequent backlog slices.

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

## Architecture

The kernel owns lifecycle mechanics, typed event-command dispatch, message
envelopes, scheduling boundaries, extension resolution and cancellation. Core
capabilities such as tools, sessions, context, compaction, policy and TUI are
bundled extensions using the same public extension model as external packages.

Runtime composition wires one explicit channel into both buses and the pump:

```python
channel = RendezvousChannel()
event_bus = InMemoryEventBus(channel)
command_bus = InMemoryCommandBus(channel)
pump = RuntimePump(event_bus, command_bus, DefaultEnvelopeFactory(), channel)
```

The channel is a bounded rendezvous for streamed events. Custom bus
implementations receive the `RuntimeSink` dependency explicitly and can be
substituted in runtime tests without depending on an in-memory bus.

The default verification suite is deterministic and offline. No FastAPI server,
database, Redis service or Docker Compose stack is part of the current product
surface.
