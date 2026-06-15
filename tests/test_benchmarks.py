from app.codec import decode_bid_payload


def test_bid_payload_decode_benchmark(benchmark) -> None:
    payload = b'{"supply_id":"supply1","ip":"123.45.67.89","country":"US"}'

    result = benchmark(decode_bid_payload, payload)

    assert result.supply_id == "supply1"
    assert result.country == "US"
