"""Capability-aware command execution scheduling."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from better_agent.kernel.contracts import ExecutionClaims, ExecutionScheduler


@dataclass(slots=True)
class _Request[T]:
    claims: ExecutionClaims
    operation: Callable[[], Awaitable[T]]
    result: asyncio.Future[T]


def _conflicts(left: ExecutionClaims, right: ExecutionClaims) -> bool:
    """Return whether two operations must not overlap."""
    if left.exclusive or right.exclusive:
        return True
    if left.resources is None or right.resources is None:
        return True
    return bool(left.resources & right.resources)


class CapabilityScheduler(ExecutionScheduler):
    """Run compatible operations concurrently while preserving conflict FIFO."""

    def __init__(self) -> None:
        self._pending: deque[_Request[Any]] = deque()
        self._active: list[_Request[Any]] = []
        self._lock = asyncio.Lock()

    async def run[T](
        self,
        claims: ExecutionClaims,
        operation: Callable[[], Awaitable[T]],
        /,
    ) -> T:
        loop = asyncio.get_running_loop()
        request: _Request[Any] = _Request(claims, operation, loop.create_future())
        async with self._lock:
            self._pending.append(request)
            self._schedule_ready()

        try:
            return cast(T, await asyncio.shield(request.result))
        except asyncio.CancelledError:
            async with self._lock:
                if request in self._pending:
                    self._pending.remove(request)
                    self._schedule_ready()
            raise

    def _schedule_ready(self) -> None:
        while True:
            for index, request in enumerate(self._pending):
                if self._is_ready(index, request):
                    del self._pending[index]
                    self._active.append(request)
                    asyncio.create_task(self._execute(request))
                    break
            else:
                return

    def _is_ready(self, index: int, request: _Request[object]) -> bool:
        if any(_conflicts(request.claims, active.claims) for active in self._active):
            return False
        return not any(
            _conflicts(request.claims, older.claims)
            for older in list(self._pending)[:index]
        )

    async def _execute(self, request: _Request[object]) -> None:
        try:
            result = await request.operation()
        except BaseException as error:
            if not request.result.done():
                request.result.set_exception(error)
        else:
            if not request.result.done():
                request.result.set_result(result)
        finally:
            async with self._lock:
                self._active.remove(request)
                self._schedule_ready()


class InlineExecutionScheduler(ExecutionScheduler):
    """Deterministic scheduler double that executes one operation inline."""

    async def run[T](
        self,
        claims: ExecutionClaims,
        operation: Callable[[], Awaitable[T]],
        /,
    ) -> T:
        return await operation()
