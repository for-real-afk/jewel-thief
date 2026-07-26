"""
Redis-backed rate limiting, tier-aware, applied per (endpoint scope, client).

Implemented as a fixed-window counter rather than a literal token bucket --
same practical effect (N requests per rolling-ish window, 429 past it) with a
much simpler, easier-to-test implementation (one INCR + one conditional
EXPIRE per request, no separate refill bookkeeping). Revisit only if bursty
traffic right at a window boundary turns out to matter in practice.

When REDIS_URL isn't set, rate limiting is a deliberate no-op -- same
tradeoff as cache.py/job_store.py's in-memory fallbacks: a single dev
instance doesn't need it, and a fake single-process limiter would mean
something different (and misleadingly reassuring) once a second instance
exists.
"""
import redis

from config import get_settings

settings = get_settings()

_WINDOW_SECONDS = 60

# requests per _WINDOW_SECONDS, per (scope, tier). Search traffic can run
# much higher than bulk indexing -- a single admin bulk upload is naturally
# low-frequency, and a low cap here bounds the worst case of a runaway retry
# loop hammering the embedding pipeline.
_LIMITS: dict[str, dict[str, int]] = {
    "search": {"standard": 60, "legacy": 60, "bulk_admin": 60},
    "index": {"standard": 10, "legacy": 10, "bulk_admin": 10},
}
_DEFAULT_TIER = "standard"

_client = redis.from_url(settings.redis_url) if settings.redis_url else None


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit exceeded, retry after {retry_after_seconds}s")


def _limit_for(scope: str, tier: str) -> int:
    tiers = _LIMITS[scope]
    return tiers.get(tier, tiers[_DEFAULT_TIER])


def check(scope: str, client_name: str, tier: str) -> None:
    """Raises RateLimitExceeded if client_name has exceeded its per-window
    budget for this scope ("search" or "index"). No-op if Redis isn't
    configured (see module docstring)."""
    if _client is None:
        return

    key = f"ratelimit:{scope}:{client_name}"
    count = _client.incr(key)
    if count == 1:
        _client.expire(key, _WINDOW_SECONDS)
    if count > _limit_for(scope, tier):
        ttl = _client.ttl(key)
        raise RateLimitExceeded(retry_after_seconds=max(int(ttl), 1))
