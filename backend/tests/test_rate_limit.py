import pytest

from app import rate_limit


class _FakeRedis:
    """In-memory stand-in for the handful of redis-py calls rate_limit.py
    makes -- INCR, EXPIRE, TTL. Real TTL countdown isn't simulated; tests
    that care about the Retry-After value set it explicitly."""

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._ttl = 60

    def incr(self, key):
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def expire(self, key, seconds):
        self._ttl = seconds

    def ttl(self, key):
        return self._ttl


@pytest.fixture(autouse=True)
def fake_redis(mocker):
    fake = _FakeRedis()
    mocker.patch.object(rate_limit, "_client", fake)
    return fake


def test_check_allows_requests_under_the_limit():
    for _ in range(rate_limit._limit_for("search", "standard")):
        rate_limit.check("search", "acme-mobile", "standard")  # should not raise


def test_check_raises_once_limit_exceeded():
    limit = rate_limit._limit_for("index", "standard")
    for _ in range(limit):
        rate_limit.check("index", "acme-mobile", "standard")

    with pytest.raises(rate_limit.RateLimitExceeded):
        rate_limit.check("index", "acme-mobile", "standard")


def test_check_is_scoped_per_client():
    limit = rate_limit._limit_for("search", "standard")
    for _ in range(limit):
        rate_limit.check("search", "client-a", "standard")

    rate_limit.check("search", "client-b", "standard")  # different client, should not raise


def test_check_is_scoped_per_endpoint():
    limit = rate_limit._limit_for("index", "standard")
    for _ in range(limit):
        rate_limit.check("index", "acme-mobile", "standard")

    rate_limit.check("search", "acme-mobile", "standard")  # different scope, should not raise


def test_check_unknown_tier_falls_back_to_standard_limit():
    assert rate_limit._limit_for("search", "nonexistent-tier") == rate_limit._limit_for("search", "standard")


def test_check_no_op_when_redis_not_configured(mocker):
    mocker.patch.object(rate_limit, "_client", None)

    for _ in range(1000):
        rate_limit.check("search", "acme-mobile", "standard")  # should never raise


def test_retry_after_exceeded_uses_key_ttl(fake_redis):
    limit = rate_limit._limit_for("search", "standard")
    for _ in range(limit):
        rate_limit.check("search", "acme-mobile", "standard")
    # expire() is only called on the first INCR (count == 1) -- set the TTL
    # only after the window is already established, so it isn't immediately
    # overwritten back to the default set during the loop above.
    fake_redis._ttl = 42

    with pytest.raises(rate_limit.RateLimitExceeded) as exc_info:
        rate_limit.check("search", "acme-mobile", "standard")
    assert exc_info.value.retry_after_seconds == 42
