import pytest
from httpx import AsyncClient

from tests.conftest import bid


@pytest.mark.anyio
async def test_health_endpoint_returns_ok_status(api_client: AsyncClient) -> None:
    response = await api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_bid_runs_auction_for_matching_country_bidders(api_client: AsyncClient) -> None:
    response = await bid(
        api_client,
        {"supply_id": "supply1", "ip": "123.45.67.89", "country": "US"},
    )

    assert response.status_code == 200
    assert response.json() == {"winner": "bidder3", "price": 0.23}


@pytest.mark.anyio
async def test_bid_rejects_unknown_supply(api_client: AsyncClient) -> None:
    response = await bid(
        api_client,
        {"supply_id": "missing", "ip": "123.45.67.89", "country": "US"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown supply_id"


@pytest.mark.anyio
async def test_bid_rate_limits_by_ip(api_client: AsyncClient) -> None:
    payload = {"supply_id": "supply1", "ip": "123.45.67.89", "country": "US"}

    for _ in range(3):
        assert (await bid(api_client, payload, ip="198.51.100.7")).status_code == 200

    response = await bid(api_client, payload, ip="198.51.100.7")

    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded"


@pytest.mark.anyio
async def test_stat_returns_grouped_supply_statistics(api_client: AsyncClient) -> None:
    await bid(api_client, {"supply_id": "supply1", "ip": "1.1.1.1", "country": "US"})
    await bid(api_client, {"supply_id": "supply2", "ip": "2.2.2.2", "country": "GB"})

    response = await api_client.get("/stat")

    assert response.status_code == 200
    assert response.json() == {
        "supply1": {
            "total_reqs": 1,
            "reqs_per_country": {"US": 1},
            "bidders": {
                "bidder1": {"wins": 0, "total_revenue": 0.0, "no_bids": 0, "timeouts": 0},
                "bidder3": {"wins": 1, "total_revenue": 0.23, "no_bids": 0, "timeouts": 0},
            },
        },
        "supply2": {
            "total_reqs": 1,
            "reqs_per_country": {"GB": 1},
            "bidders": {
                "bidder2": {"wins": 1, "total_revenue": 0.01, "no_bids": 0, "timeouts": 0},
            },
        },
    }
