from dataclasses import dataclass

import msgspec


class BidRequest(msgspec.Struct, frozen=True):
    supply_id: str
    ip: str
    country: str


@dataclass(frozen=True, slots=True)
class Bidder:
    id: str
    country: str


@dataclass(frozen=True, slots=True)
class Bid:
    bidder_id: str
    price: float


@dataclass(frozen=True, slots=True)
class AuctionOutcome:
    winner: Bid
    skipped_bidder_ids: tuple[str, ...]
    timed_out_bidder_ids: tuple[str, ...]

