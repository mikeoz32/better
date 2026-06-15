from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    redis_url: str
    rate_limit_per_minute: int = 3
    rate_limit_window_seconds: int = 60
    stats_cache_ttl_seconds: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=getenv(
                "DATABASE_URL",
                "postgresql://auction:auction@postgres:5432/auction",
            ),
            redis_url=getenv("REDIS_URL", "redis://redis:6379/0"),
            rate_limit_per_minute=int(getenv("RATE_LIMIT_PER_MINUTE", "3")),
            rate_limit_window_seconds=int(getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            stats_cache_ttl_seconds=int(getenv("STATS_CACHE_TTL_SECONDS", "1")),
        )

