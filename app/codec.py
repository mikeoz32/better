import msgspec

from app.models import BidRequest

_bid_request_decoder = msgspec.json.Decoder(type=BidRequest)


def decode_bid_payload(payload: bytes) -> BidRequest:
    return _bid_request_decoder.decode(payload)

