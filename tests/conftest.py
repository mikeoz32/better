from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.stats import MemoryStatsStore
from app.storage import MemoryCatalogRepository


class FixedRandom:
    def __init__(self, *, skip_values: list[float], prices: list[float], delays: list[float] | None = None):
        self._skip_values = skip_values
        self._prices = prices
        self._delays = delays or []

    def random(self) -> float:
        return self._skip_values.pop(0) if self._skip_values else 0.99

    def uniform(self, start: float, end: float) -> float:
        if end <= 1.0:
            return self._prices.pop(0) if self._prices else start
        return self._delays.pop(0) if self._delays else start


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def catalog() -> MemoryCatalogRepository:
    return MemoryCatalogRepository(
        supplies={
            "supply1": ["bidder1", "bidder2", "bidder3"],
            "supply2": ["bidder2", "bidder3"],
        },
        bidders={
            "bidder1": {"country": "US"},
            "bidder2": {"country": "GB"},
            "bidder3": {"country": "US"},
        },
    )


@pytest.fixture
async def api_client(catalog: MemoryCatalogRepository) -> AsyncIterator[AsyncClient]:
    app = create_app(
        catalog=catalog,
        stats=MemoryStatsStore(),
        rng=FixedRandom(skip_values=[0.99, 0.99], prices=[0.10, 0.23]),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


async def bid(client: AsyncClient, payload: dict[str, Any], *, ip: str = "203.0.113.10"):
    return await client.post("/bid", json=payload, headers={"x-forwarded-for": ip})
