"""
Per-client API key management, backed by Supabase (the same Postgres
instance already used by catalog_store.py -- no new database needed).

Raw keys are never stored, only a SHA-256 hash, so a leaked table dump alone
can't be used to authenticate as an existing client. create_key() returns
the raw key exactly once, at issuance time -- there is no way to recover it
afterward, only to revoke and issue a new one.

Table: api_keys(key_id PK, hashed_key, client_name, created_at, revoked_at,
rate_limit_tier).
"""
import hashlib
import secrets
import uuid
from datetime import datetime, timezone

from supabase import create_client

from .config import get_settings

settings = get_settings()
_client = create_client(settings.supabase_url, settings.supabase_key)
_TABLE = "api_keys"

DEFAULT_RATE_LIMIT_TIER = "standard"


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def create_key(client_name: str, rate_limit_tier: str = DEFAULT_RATE_LIMIT_TIER) -> tuple[str, str]:
    """Issues a new key for client_name. Returns (key_id, raw_key) -- raw_key
    is shown to the caller exactly once (e.g. by scripts/create_api_key.py)
    and is not recoverable afterward, only revocable."""
    key_id = str(uuid.uuid4())
    raw_key = secrets.token_urlsafe(32)
    _client.table(_TABLE).insert({
        "key_id": key_id,
        "hashed_key": _hash_key(raw_key),
        "client_name": client_name,
        "rate_limit_tier": rate_limit_tier,
        "revoked_at": None,
    }).execute()
    return key_id, raw_key


def lookup_key(raw_key: str) -> dict | None:
    """Returns {"client_name", "rate_limit_tier"} for a valid, non-revoked
    key, or None if the key is unknown or has been revoked."""
    result = (
        _client.table(_TABLE)
        .select("client_name, rate_limit_tier, revoked_at")
        .eq("hashed_key", _hash_key(raw_key))
        .execute()
    )
    if not result.data:
        return None
    row = result.data[0]
    if row["revoked_at"] is not None:
        return None
    return {"client_name": row["client_name"], "rate_limit_tier": row["rate_limit_tier"]}


def revoke_key(key_id: str) -> None:
    _client.table(_TABLE).update(
        {"revoked_at": datetime.now(timezone.utc).isoformat()}
    ).eq("key_id", key_id).execute()
