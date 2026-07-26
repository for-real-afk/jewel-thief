"""
Search-vector caching.

This module is built around an abstract interface specifically so a storage
swap doesn't require a rewrite. InMemoryCache is the single-instance
stopgap, exactly the same tradeoff already documented for the job-tracking
dict in main.py and the (now Supabase-backed) catalog store used to have —
it disappears on restart and isn't shared across multiple backend instances.
RedisCache is the real, shared alternative, selected automatically at import
time via settings.redis_url: empty means Redis isn't configured yet (falls
back to InMemoryCache), set means every backend instance shares one cache.
Either way callers only ever touch the SearchCache interface, never a
concrete class.

Only wired into the text-query path (main.py) — repeat identical image
uploads are rare, and hashing image bytes for a cache key is out of scope
here.
"""
import hashlib
import json
import time
from abc import ABC, abstractmethod

import redis

from config import get_settings

settings = get_settings()


class SearchCache(ABC):
    @abstractmethod
    def get(self, key: str) -> list[float] | None: ...

    @abstractmethod
    def set(self, key: str, vector: list[float], ttl_seconds: int = 3600) -> None: ...


class InMemoryCache(SearchCache):
    """Process-local dict with TTL eviction. NOT shared across multiple
    backend instances or restarts — this is a stopgap, used automatically
    whenever REDIS_URL isn't set (see get_cache() below)."""

    def __init__(self):
        self._store: dict[str, tuple[list[float], float]] = {}

    def get(self, key: str) -> list[float] | None:
        entry = self._store.get(key)
        if not entry:
            return None
        vector, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return vector

    def set(self, key: str, vector: list[float], ttl_seconds: int = 3600) -> None:
        self._store[key] = (vector, time.time() + ttl_seconds)


class RedisCache(SearchCache):
    """Shared cache backed by any Redis-compatible service (Upstash, Redis
    Cloud, self-hosted, ...) — the real, multi-instance-safe implementation.
    Connection is lazy (redis-py's from_url doesn't connect until the first
    command), so constructing this with a bad URL doesn't fail at import
    time; it fails on the first get()/set() call, same as any other external
    API call in this codebase (and is not wrapped in external_api_retry: a
    cache is a performance optimization, not a critical path — better to let
    a transient cache failure surface immediately than silently retry and
    delay a search)."""

    def __init__(self, redis_url: str):
        self._client = redis.from_url(redis_url)

    def get(self, key: str) -> list[float] | None:
        raw = self._client.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, vector: list[float], ttl_seconds: int = 3600) -> None:
        self._client.set(key, json.dumps(vector), ex=ttl_seconds)


def ping() -> None:
    """Cheap Redis reachability check for GET /health. No-op if REDIS_URL
    isn't set (InMemoryCache is in use, not a failure state -- see main.py's
    /health, which reports this as "not_configured" rather than "down").
    Raises on failure when Redis IS configured; callers decide how to
    report that."""
    if isinstance(_cache, RedisCache):
        _cache._client.ping()


def cache_key(query_text: str, filters: dict) -> str:
    normalized = query_text.strip().lower()
    payload = f"{normalized}:{json.dumps(filters, sort_keys=True)}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _make_cache() -> SearchCache:
    if settings.redis_url:
        return RedisCache(settings.redis_url)
    return InMemoryCache()


_cache: SearchCache = _make_cache()
