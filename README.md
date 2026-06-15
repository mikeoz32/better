# better

FastAPI implementation of a simplified Ad Exchange Auction Service.

## Stack

- FastAPI with `uvicorn` and `uvloop`
- `msgspec` for fast request JSON decoding
- `orjson` for response serialization
- PostgreSQL via `asyncpg` for supply/bidder data
- Redis with `hiredis` for rate limiting, counters, and cached `/stat` snapshots

## Run

Start the app, PostgreSQL, and Redis:

```bash
docker compose up --build
```

The API listens on `http://localhost:8000`.

Run one auction:

```bash
curl -X POST 'http://localhost:8000/bid?max_timeout_ms=50' \
  -H 'content-type: application/json' \
  -d '{"supply_id":"supply1","ip":"123.45.67.89","country":"US"}'
```

Read statistics:

```bash
curl http://localhost:8000/stat
```

## Local Development

Install dependencies:

```bash
uv sync
```

Run tests and benchmarks:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Run without Docker only if PostgreSQL and Redis are available at `DATABASE_URL` and `REDIS_URL`:

```bash
uv run uvicorn app.main:app --reload
```

## Notes On Scaling

PostgreSQL stores normalized supply and bidder data and can be scaled with read replicas if the catalog becomes large. Redis handles request-rate counters, auction statistics, and the cached `/stat` response; the cache TTL is intentionally short so writes stay simple while repeated stats reads avoid rebuilding the response on every request. The app is stateless outside PostgreSQL and Redis, so more API containers can be added behind a load balancer without changing application code.
