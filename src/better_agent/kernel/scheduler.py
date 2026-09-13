"""Capability-aware command execution admission."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from better_agent.kernel.contracts import ExecutionClaims, ExecutionScheduler


@dataclass(slots=True)
class _Ticket:
    claims: ExecutionClaims
    granted: asyncio.Future[None]


def _conflicts(left: ExecutionClaims, right: ExecutionClaims) -> bool:
    """Return whether two admitted operations must not overlap."""
    if left.exclusive or right.exclusive:
        return True
    return bool(
        left.writes & (right.reads | right.writes)
        or right.writes & (left.reads | left.writes),
    )


class CapabilityScheduler(ExecutionScheduler):
    """Admit compatible operations concurrently while preserving conflict FIFO."""

    def __init__(self) -> None:
        self._pending: deque[_Ticket] = deque()
        self._active: list[_Ticket] = []
        self._lock = asyncio.Lock()

    def admit(self, claims: ExecutionClaims) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def admission() -> AsyncIterator[None]:
            ticket = _Ticket(
                claims=claims,
                granted=asyncio.get_running_loop().create_future(),
            )
            async with self._lock:
                self._pending.append(ticket)
                self._schedule_ready()

            try:
                await ticket.granted
                yield
            finally:
                async with self._lock:
                    if ticket in self._pending:
                        self._pending.remove(ticket)
                        self._schedule_ready()
                    elif ticket in self._active:
                        self._active.remove(ticket)
                        self._schedule_ready()

        return admission()

    def _schedule_ready(self) -> None:
        self._pending = deque(
            ticket for ticket in self._pending if not ticket.granted.cancelled()
        )
        while True:
            for index, ticket in enumerate(self._pending):
                if self._is_ready(index, ticket):
                    del self._pending[index]
                    self._active.append(ticket)
                    if ticket.granted.cancelled():
                        self._active.remove(ticket)
                        continue
                    ticket.granted.set_result(None)
                    break
            else:
                return

    def _is_ready(self, index: int, ticket: _Ticket) -> bool:
        if any(_conflicts(ticket.claims, active.claims) for active in self._active):
            return False
        return not any(
            _conflicts(ticket.claims, older.claims)
            for older in list(self._pending)[:index]
        )


class InlineExecutionScheduler(ExecutionScheduler):
    """Deterministic admission double that never blocks a request."""

    def admit(self, claims: ExecutionClaims) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def admission() -> AsyncIterator[None]:
            yield

        return admission()
