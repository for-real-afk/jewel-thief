#!/usr/bin/env python
"""
One-off migration: copies every item from the old local catalog_store.json
(pre-Supabase catalog metadata store) into the Supabase catalog_items table
via catalog_store.record_item(), so it doesn't need to duplicate the upsert
logic. Safe to re-run — record_item() upserts by item_id.

Usage:
    python scripts/migrate_catalog_to_supabase.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import catalog_store

LEGACY_STORE_PATH = Path(__file__).resolve().parent.parent / "catalog_store.json"


def main() -> int:
    if not LEGACY_STORE_PATH.exists():
        print(f"No legacy store found at {LEGACY_STORE_PATH} — nothing to migrate.")
        return 0

    data = json.loads(LEGACY_STORE_PATH.read_text())
    print(f"Migrating {len(data)} item(s) from {LEGACY_STORE_PATH.name} to Supabase...")

    failed = []
    for i, (item_id, metadata) in enumerate(data.items(), 1):
        try:
            catalog_store.record_item(item_id, metadata)
        except Exception as exc:
            failed.append((item_id, str(exc)))
            continue
        if i % 10 == 0 or i == len(data):
            print(f"  [{i}/{len(data)}] migrated", flush=True)

    if failed:
        print(f"\nFailed to migrate {len(failed)} item(s):")
        for item_id, err in failed:
            print(f"  - {item_id}: {err}")

    migrated = len(data) - len(failed)
    print(f"\nDone. {migrated} item(s) migrated to Supabase.")

    _, total = catalog_store.list_items(limit=1, offset=0)
    print(f"Supabase catalog_items table now has {total} row(s) total.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
