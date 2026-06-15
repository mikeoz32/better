import asyncio
import logging
import random
from dataclasses import dataclass, field
from random import Random

from app.config import Settings
from app.models import AuctionOutcome, Bid, BidRequest, Bidder
from app.stats import StatsStore
from app.storage import CatalogRepository

logger = logging.getLogger("auction")


class AuctionError(Exception):
    status_code = 400
    detail = "Auction error"


class RateLimitExceeded(AuctionError):
    status_code = 429
    detail = "Rate limit exceeded"


class UnknownSupply(AuctionError):
    status_code = 404
    detail = "Unknown supply_id"


class NoEligibleBidders(AuctionError):
    status_code = 404
    detail = "No eligible bidders"


class NoBids(AuctionError):
    status_code = 409
    detail = "No bids"


@dataclass(slots=True)
class AuctionService:
    catalog: CatalogRepository
    stats: StatsStore
    settings: Settings
    rng: Random = field(default_factory=random.Random)

    async def run(
        self,
        bid_request: BidRequest,
        *,
        client_ip: str,
        max_timeout_ms: int = 0,
    ) -> AuctionOutcome:
        allowed = await self.stats.check_rate_limit(
            client_ip,
            limit=self.settings.rate_limit_per_minute,
            window_seconds=self.settings.rate_limit_window_seconds,
        )
        if not allowed:
            raise RateLimitExceeded

        eligible_bidders = await self.catalog.eligible_bidders(
            bid_request.supply_id,
            bid_request.country,
        )
        if eligible_bidders is None:
            raise UnknownSupply
        if not eligible_bidders:
            raise NoEligibleBidders

        bids, skipped_ids, timed_out_ids = await self._collect_bids(
            eligible_bidders,
            max_timeout_ms=max_timeout_ms,
        )
        if not bids:
            await self.stats.record_auction(
                supply_id=bid_request.supply_id,
                country=bid_request.country,
                eligible_bidders=eligible_bidders,
                winner_id=None,
                price=None,
                skipped_bidder_ids=tuple(skipped_ids),
                timed_out_bidder_ids=tuple(timed_out_ids),
            )
            raise NoBids

        winner = max(bids, key=lambda bid: bid.price)
        await self.stats.record_auction(
            supply_id=bid_request.supply_id,
            country=bid_request.country,
            eligible_bidders=eligible_bidders,
            winner_id=winner.bidder_id,
            price=winner.price,
            skipped_bidder_ids=tuple(skipped_ids),
            timed_out_bidder_ids=tuple(timed_out_ids),
        )
        logger.info(
            "Auction for %s (country=%s): winner=%s price=%.2f",
            bid_request.supply_id,
            bid_request.country,
            winner.bidder_id,
            winner.price,
        )
        return AuctionOutcome(
            winner=winner,
            skipped_bidder_ids=tuple(skipped_ids),
            timed_out_bidder_ids=tuple(timed_out_ids),
        )

    async def _collect_bids(
        self,
        eligible_bidders: list[Bidder],
        *,
        max_timeout_ms: int,
    ) -> tuple[list[Bid], list[str], list[str]]:
        bids: list[Bid] = []
        skipped_ids: list[str] = []
        timed_out_ids: list[str] = []

        for bidder in eligible_bidders:
            if max_timeout_ms > 0:
                delay_ms = self.rng.uniform(0.0, float(max_timeout_ms) * 1.5)
                if delay_ms > max_timeout_ms:
                    timed_out_ids.append(bidder.id)
                    logger.info("bidder=%s timed out after %.2f ms", bidder.id, delay_ms)
                    continue
                if delay_ms:
                    await asyncio.sleep(delay_ms / 1000)

            if self.rng.random() < 0.30:
                skipped_ids.append(bidder.id)
                logger.info("bidder=%s skipped bidding", bidder.id)
                continue

            price = round(self.rng.uniform(0.01, 1.00), 2)
            logger.info("bidder=%s price=%.2f", bidder.id, price)
            bids.append(Bid(bidder_id=bidder.id, price=price))

        return bids, skipped_ids, timed_out_ids
