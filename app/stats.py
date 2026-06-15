from collections import defaultdict
from time import monotonic
from typing import Any, Protocol

import orjson
from redis.asyncio import Redis

from app.models import Bidder


class StatsStore(Protocol):
    async def check_rate_limit(self, ip: str, *, limit: int, window_seconds: int) -> bool: ...

    async def record_auction(
        self,
        *,
        supply_id: str,
        country: str,
        eligible_bidders: list[Bidder],
        winner_id: str | None,
        price: float | None,
        skipped_bidder_ids: tuple[str, ...],
        timed_out_bidder_ids: tuple[str, ...],
    ) -> None: ...

    async def snapshot(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def _empty_bidder_stats() -> dict[str, int | float]:
    return {"wins": 0, "total_revenue": 0.0, "no_bids": 0, "timeouts": 0}


class MemoryStatsStore:
    def __init__(self) -> None:
        self._requests: dict[str, int] = defaultdict(int)
        self._countries: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._bidders: dict[str, dict[str, dict[str, int | float]]] = defaultdict(dict)
        self._rate_limits: dict[str, tuple[float, int]] = {}

    async def check_rate_limit(self, ip: str, *, limit: int, window_seconds: int) -> bool:
        now = monotonic()
        reset_at, count = self._rate_limits.get(ip, (now + window_seconds, 0))
        if now >= reset_at:
            reset_at, count = now + window_seconds, 0

        count += 1
        self._rate_limits[ip] = (reset_at, count)
        return count <= limit

    async def record_auction(
        self,
        *,
        supply_id: str,
        country: str,
        eligible_bidders: list[Bidder],
        winner_id: str | None,
        price: float | None,
        skipped_bidder_ids: tuple[str, ...],
        timed_out_bidder_ids: tuple[str, ...],
    ) -> None:
        self._requests[supply_id] += 1
        self._countries[supply_id][country] += 1

        bidder_stats = self._bidders[supply_id]
        for bidder in eligible_bidders:
            bidder_stats.setdefault(bidder.id, _empty_bidder_stats())

        for bidder_id in skipped_bidder_ids:
            bidder_stats.setdefault(bidder_id, _empty_bidder_stats())["no_bids"] += 1

        for bidder_id in timed_out_bidder_ids:
            bidder_stats.setdefault(bidder_id, _empty_bidder_stats())["timeouts"] += 1

        if winner_id and price is not None:
            winner_stats = bidder_stats.setdefault(winner_id, _empty_bidder_stats())
            winner_stats["wins"] += 1
            winner_stats["total_revenue"] = round(float(winner_stats["total_revenue"]) + price, 2)

    async def snapshot(self) -> dict[str, Any]:
        return {
            supply_id: {
                "total_reqs": self._requests[supply_id],
                "reqs_per_country": dict(self._countries[supply_id]),
                "bidders": {
                    bidder_id: dict(stats)
                    for bidder_id, stats in sorted(self._bidders[supply_id].items())
                },
            }
            for supply_id in sorted(self._requests)
        }

    async def close(self) -> None:
        return None


class RedisStatsStore:
    def __init__(self, redis: Redis, *, stats_cache_ttl_seconds: int) -> None:
        self._redis = redis
        self._stats_cache_ttl_seconds = stats_cache_ttl_seconds

    @classmethod
    def connect(cls, redis_url: str, *, stats_cache_ttl_seconds: int) -> "RedisStatsStore":
        redis = Redis.from_url(redis_url, decode_responses=True)
        return cls(redis, stats_cache_ttl_seconds=stats_cache_ttl_seconds)

    async def check_rate_limit(self, ip: str, *, limit: int, window_seconds: int) -> bool:
        key = f"rate:{ip}"
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, window_seconds)
        return count <= limit

    async def record_auction(
        self,
        *,
        supply_id: str,
        country: str,
        eligible_bidders: list[Bidder],
        winner_id: str | None,
        price: float | None,
        skipped_bidder_ids: tuple[str, ...],
        timed_out_bidder_ids: tuple[str, ...],
    ) -> None:
        pipe = self._redis.pipeline(transaction=False)
        pipe.sadd("stats:supplies", supply_id)
        pipe.incr(f"stats:{supply_id}:total")
        pipe.hincrby(f"stats:{supply_id}:countries", country, 1)

        for bidder in eligible_bidders:
            pipe.sadd(f"stats:{supply_id}:bidders", bidder.id)
            pipe.hsetnx(f"stats:{supply_id}:bidder:{bidder.id}", "wins", 0)
            pipe.hsetnx(f"stats:{supply_id}:bidder:{bidder.id}", "total_revenue", 0.0)
            pipe.hsetnx(f"stats:{supply_id}:bidder:{bidder.id}", "no_bids", 0)
            pipe.hsetnx(f"stats:{supply_id}:bidder:{bidder.id}", "timeouts", 0)

        for bidder_id in skipped_bidder_ids:
            pipe.hincrby(f"stats:{supply_id}:bidder:{bidder_id}", "no_bids", 1)

        for bidder_id in timed_out_bidder_ids:
            pipe.hincrby(f"stats:{supply_id}:bidder:{bidder_id}", "timeouts", 1)

        if winner_id and price is not None:
            pipe.hincrby(f"stats:{supply_id}:bidder:{winner_id}", "wins", 1)
            pipe.hincrbyfloat(f"stats:{supply_id}:bidder:{winner_id}", "total_revenue", price)

        pipe.delete("stats:snapshot")
        await pipe.execute()

    async def snapshot(self) -> dict[str, Any]:
        cached = await self._redis.get("stats:snapshot")
        if cached:
            return orjson.loads(cached)

        supplies = sorted(await self._redis.smembers("stats:supplies"))
        data: dict[str, Any] = {}
        for supply_id in supplies:
            total = int(await self._redis.get(f"stats:{supply_id}:total") or 0)
            countries = await self._redis.hgetall(f"stats:{supply_id}:countries")
            bidder_ids = sorted(await self._redis.smembers(f"stats:{supply_id}:bidders"))
            bidders = {}
            for bidder_id in bidder_ids:
                stats = await self._redis.hgetall(f"stats:{supply_id}:bidder:{bidder_id}")
                bidders[bidder_id] = {
                    "wins": int(stats.get("wins", 0)),
                    "total_revenue": round(float(stats.get("total_revenue", 0.0)), 2),
                    "no_bids": int(stats.get("no_bids", 0)),
                    "timeouts": int(stats.get("timeouts", 0)),
                }
            data[supply_id] = {
                "total_reqs": total,
                "reqs_per_country": {country: int(count) for country, count in countries.items()},
                "bidders": bidders,
            }

        await self._redis.set(
            "stats:snapshot",
            orjson.dumps(data),
            ex=self._stats_cache_ttl_seconds,
        )
        return data

    async def close(self) -> None:
        await self._redis.aclose()

