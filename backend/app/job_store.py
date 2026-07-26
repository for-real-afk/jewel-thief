"""
Indexing job status tracking.

Mirrors cache.py's storage-swap-via-interface pattern: JobStore is the
abstract contract, InMemoryJobStore is the single-instance stopgap (was
previously just a plain dict in main.py -- see README.md §7/§11), RedisJobStore
is the real, shared alternative, selected automatically at import time via
settings.redis_url (empty means Redis isn't configured yet, falls back to
InMemoryJobStore; set means every backend instance shares one job store --
same switch cache.py uses for search-vector caching, same Redis instance,
two independent uses of it). Callers only ever touch the JobStore interface,
never a concrete class.
"""
import json
from abc import ABC, abstractmethod

import redis

from .config import get_settings

settings = get_settings()

# Job status doesn't need to live forever -- a day is generous headroom for
# an admin to notice a bulk upload finished and check the result.
_JOB_TTL_SECONDS = 24 * 60 * 60


class JobStore(ABC):
    @abstractmethod
    def get(self, job_id: str) -> dict | None: ...

    @abstractmethod
    def set(self, job_id: str, job: dict) -> None: ...


class InMemoryJobStore(JobStore):
    """Process-local dict, no TTL eviction (same behavior as the plain dict
    this replaces). NOT shared across multiple backend instances or restarts
    -- this is a stopgap, used automatically whenever REDIS_URL isn't set."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def get(self, job_id: str) -> dict | None:
        return self._store.get(job_id)

    def set(self, job_id: str, job: dict) -> None:
        self._store[job_id] = job


class RedisJobStore(JobStore):
    """Shared job store backed by any Redis-compatible service -- the real,
    multi-instance-safe implementation. Connection is lazy (redis-py's
    from_url doesn't connect until the first command), matching cache.py's
    RedisCache."""

    def __init__(self, redis_url: str):
        self._client = redis.from_url(redis_url)

    def get(self, job_id: str) -> dict | None:
        raw = self._client.get(f"job:{job_id}")
        return json.loads(raw) if raw is not None else None

    def set(self, job_id: str, job: dict) -> None:
        self._client.set(f"job:{job_id}", json.dumps(job), ex=_JOB_TTL_SECONDS)


def _make_job_store() -> JobStore:
    if settings.redis_url:
        return RedisJobStore(settings.redis_url)
    return InMemoryJobStore()


_job_store: JobStore = _make_job_store()
