from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from random import Random
from typing import Annotated

import msgspec
from fastapi import FastAPI, HTTPException, Query, Request

from app.auction import AuctionError, AuctionService
from app.codec import decode_bid_payload
from app.config import Settings
from app.responses import ORJSONResponse
from app.stats import RedisStatsStore, StatsStore
from app.storage import CatalogRepository, PostgresCatalogRepository


def create_app(
    *,
    catalog: CatalogRepository | None = None,
    stats: StatsStore | None = None,
    rng: Random | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    _configure_logging()
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if catalog is None:
            app.state.catalog = await PostgresCatalogRepository.connect(
                resolved_settings.database_url,
            )
        if stats is None:
            app.state.stats = RedisStatsStore.connect(
                resolved_settings.redis_url,
                stats_cache_ttl_seconds=resolved_settings.stats_cache_ttl_seconds,
            )

        app.state.auction = AuctionService(
            catalog=app.state.catalog,
            stats=app.state.stats,
            settings=resolved_settings,
            rng=rng or Random(),
        )
        yield

        if catalog is None:
            await app.state.catalog.close()
        if stats is None:
            await app.state.stats.close()

    app = FastAPI(
        title="Ad Exchange Auction Service",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.state.catalog = catalog
    app.state.stats = stats
    if catalog is not None and stats is not None:
        app.state.auction = AuctionService(
            catalog=catalog,
            stats=stats,
            settings=resolved_settings,
            rng=rng or Random(),
        )

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/bid")
    async def bid(
        request: Request,
        max_timeout_ms: Annotated[int, Query(ge=0, le=5000)] = 0,
    ) -> ORJSONResponse:
        try:
            bid_request = decode_bid_payload(await request.body())
        except msgspec.DecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid bid payload") from exc

        if not bid_request.supply_id or not bid_request.ip or not bid_request.country:
            raise HTTPException(status_code=422, detail="Missing or empty bid fields")

        client_ip = _client_ip(request, fallback=bid_request.ip)
        try:
            outcome = await request.app.state.auction.run(
                bid_request,
                client_ip=client_ip,
                max_timeout_ms=max_timeout_ms,
            )
        except AuctionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        return ORJSONResponse(
            {"winner": outcome.winner.bidder_id, "price": outcome.winner.price},
        )

    @app.get("/stat")
    async def stat(request: Request) -> ORJSONResponse:
        return ORJSONResponse(await request.app.state.stats.snapshot())

    return app


def _client_ip(request: Request, *, fallback: str) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return fallback


def _configure_logging() -> None:
    auction_logger = logging.getLogger("auction")
    if not auction_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        auction_logger.addHandler(handler)
    auction_logger.setLevel(logging.INFO)
    auction_logger.propagate = False


app = create_app()
