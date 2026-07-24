"""
Local catalog index for GET /api/v1/catalog/items.

Pinecone doesn't support convenient arbitrary listing — its list() call
returns IDs only (no metadata) and isn't meant for admin-table pagination.
A JSON file is enough for a single-process dev/admin tool; move this to a
real table (Postgres/SQLite) before running multiple backend instances, same
caveat as main.py's in-memory job dict.
"""
import json
import threading
from pathlib import Path

_STORE_PATH = Path(__file__).resolve().parent / "catalog_store.json"
_lock = threading.Lock()


def _load() -> dict:
    if _STORE_PATH.exists():
        return json.loads(_STORE_PATH.read_text())
    return {}


def _save(data: dict) -> None:
    _STORE_PATH.write_text(json.dumps(data, indent=2))


def record_item(item_id: str, metadata: dict) -> None:
    """Upsert one item's metadata into the local store (mirrors the Pinecone upsert)."""
    with _lock:
        data = _load()
        data[item_id] = metadata
        _save(data)


def list_items(limit: int, offset: int) -> tuple[list[dict], int]:
    """Newest-first page of {item_id, **metadata} dicts, plus the total count."""
    with _lock:
        data = _load()
    items = [{"item_id": item_id, **metadata} for item_id, metadata in data.items()]
    items.reverse()  # dict preserves insertion order in Python 3.7+ -> newest last -> reverse
    total = len(items)
    return items[offset:offset + limit], total
