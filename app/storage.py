from collections.abc import Mapping, Sequence
from typing import Protocol

import asyncpg

from app.models import Bidder


class CatalogRepository(Protocol):
    async def eligible_bidders(self, supply_id: str, country: str) -> list[Bidder] | None: ...

    async def close(self) -> None: ...


class MemoryCatalogRepository:
    def __init__(
        self,
        *,
        supplies: Mapping[str, Sequence[str]],
        bidders: Mapping[str, Mapping[str, str]],
    ) -> None:
        self._supplies = {supply_id: tuple(bidder_ids) for supply_id, bidder_ids in supplies.items()}
        self._bidders = {bidder_id: dict(data) for bidder_id, data in bidders.items()}

    async def eligible_bidders(self, supply_id: str, country: str) -> list[Bidder] | None:
        bidder_ids = self._supplies.get(supply_id)
        if bidder_ids is None:
            return None

        return [
            Bidder(id=bidder_id, country=bidder["country"])
            for bidder_id in bidder_ids
            if (bidder := self._bidders.get(bidder_id)) and bidder["country"] == country
        ]

    async def close(self) -> None:
        return None


class PostgresCatalogRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, database_url: str) -> "PostgresCatalogRepository":
        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=10,
            statement_cache_size=256,
        )
        return cls(pool)

    async def eligible_bidders(self, supply_id: str, country: str) -> list[Bidder] | None:
        rows = await self._pool.fetch(
            """
            SELECT b.id, b.country
            FROM supplies s
            LEFT JOIN supply_bidders sb ON sb.supply_id = s.id
            LEFT JOIN bidders b ON b.id = sb.bidder_id
            WHERE s.id = $1
            ORDER BY sb.bidder_id
            """,
            supply_id,
        )
        if not rows:
            return None

        return [
            Bidder(id=row["id"], country=row["country"])
            for row in rows
            if row["id"] is not None and row["country"] == country
        ]

    async def close(self) -> None:
        await self._pool.close()

