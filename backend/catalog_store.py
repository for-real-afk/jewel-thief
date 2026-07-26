"""
Catalog metadata store for GET /api/v1/catalog/items, backed by Supabase.

Pinecone doesn't support convenient arbitrary listing — its list() call
returns IDs only (no metadata) and isn't meant for admin-table pagination.
This mirrors every Pinecone upsert into a `catalog_items` table so the
admin UI can page through metadata without touching the vector index.
"""
from supabase import create_client

from config import get_settings

settings = get_settings()
_client = create_client(settings.supabase_url, settings.supabase_key)
_TABLE = "catalog_items"


def record_item(item_id: str, metadata: dict) -> None:
    """Upsert one item's metadata into the catalog table (mirrors the Pinecone upsert)."""
    _client.table(_TABLE).upsert({"item_id": item_id, "metadata": metadata}).execute()


def list_items(limit: int, offset: int) -> tuple[list[dict], int]:
    """Newest-first page of {item_id, **metadata} dicts, plus the total count."""
    result = (
        _client.table(_TABLE)
        .select("item_id, metadata", count="exact")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    items = [{"item_id": row["item_id"], **row["metadata"]} for row in result.data]
    return items, result.count or 0


def get_item(item_id: str) -> dict | None:
    """Single {item_id, **metadata} dict, or None if item_id isn't in the catalog."""
    result = _client.table(_TABLE).select("item_id, metadata").eq("item_id", item_id).execute()
    if not result.data:
        return None
    row = result.data[0]
    return {"item_id": row["item_id"], **row["metadata"]}


def delete_item(item_id: str) -> None:
    _client.table(_TABLE).delete().eq("item_id", item_id).execute()
