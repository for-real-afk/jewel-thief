#!/usr/bin/env python
"""
One-off, destructive full-catalog re-embed.

Every catalog vector currently in Pinecone was produced either by the old
(broken) task_type convention, or without text fusion, or both -- these are
not comparable to vectors produced by embeddings.embed_catalog_item() and
must ALL be regenerated, not left mixed with new-style vectors in the same
index (mixed vector "generations" in one index would make cosine scores
incomparable between old and new items -- see README.md §4.3 on why the
whole catalog must share one embedding process).

Metadata source of truth is catalog_store.py (Supabase-backed) -- NOT any
local JSON file. Every currently-indexed item already has name, category,
price, caption, description, tags, image_url, and (optionally) material
there; this script re-derives nothing, it only re-embeds.

Upsert is a full replace (see vector_db.py::upsert_batch -> Pinecone's
index.upsert()), so re-embedding under the SAME item_id correctly overwrites
the old vector in place -- no separate delete step needed.

This is a manual, one-off operational script -- NOT imported into main.py or
called from any endpoint.

Usage:
    python scripts/reembed_catalog.py --dry-run   # embeds for real, skips the Pinecone write, shows what would happen
    python scripts/reembed_catalog.py             # embeds AND upserts for real -- destructive, overwrites the live index
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import catalog_store
from app import embeddings
from app import vector_db
from app.main import build_catalog_text_description

STATIC_CATALOG_DIR = Path(__file__).resolve().parent.parent / "app" / "static" / "catalog"
PAGE_SIZE = 100
PROGRESS_EVERY = 10


def fetch_all_catalog_items() -> list[dict]:
    """Every item currently in the catalog store, paginated -- Supabase's
    list_items() is the source of truth for metadata (see module docstring)."""
    items: list[dict] = []
    offset = 0
    while True:
        page, total = catalog_store.list_items(limit=PAGE_SIZE, offset=offset)
        if not page:
            break
        items.extend(page)
        offset += len(page)
        if offset >= total:
            break
    return items


def re_embed_item(item: dict) -> dict:
    """Load the item's stored image, build the fused text description exactly
    as new items are indexed with (main.py's build_catalog_text_description),
    and produce one new fused vector. Raises on any failure -- caller catches
    per-item so one bad item doesn't abort the whole run."""
    item_id = item["item_id"]
    image_path = STATIC_CATALOG_DIR / f"{item_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"no stored image at {image_path}")

    image_bytes = image_path.read_bytes()
    text_description = build_catalog_text_description(
        item.get("name", ""), item.get("caption", ""), item.get("description", ""), item.get("tags", [])
    )
    vector = embeddings.embed_catalog_item(image_bytes, text_description)

    metadata = {k: v for k, v in item.items() if k != "item_id"}
    return {"id": item_id, "vector": vector, "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Embed every item for real but skip the Pinecone upsert -- prints what would happen.",
    )
    args = parser.parse_args()

    items = fetch_all_catalog_items()
    total = len(items)
    print(f"Found {total} catalog item(s) to re-embed{' (DRY RUN)' if args.dry_run else ''}.")
    if total == 0:
        print("Nothing to do.")
        return 0

    to_upsert = []
    failed = []
    for i, item in enumerate(items, 1):
        item_id = item.get("item_id", "<unknown>")
        try:
            to_upsert.append(re_embed_item(item))
        except Exception as exc:
            print(f"  [{i}/{total}] FAILED item_id={item_id}: {exc!r}", flush=True)
            failed.append(item_id)
            continue
        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"  [{i}/{total}] re-embedded (last: item_id={item_id})", flush=True)

    succeeded = len(to_upsert)
    if args.dry_run:
        print(f"\nDRY RUN -- would upsert {succeeded} item(s) to Pinecone index "
              f"'{vector_db.settings.pinecone_index_name}':")
        for item in to_upsert[:10]:
            print(f"  - {item['id']}")
        if succeeded > 10:
            print(f"  ... and {succeeded - 10} more")
    elif to_upsert:
        count = vector_db.upsert_batch(to_upsert)
        print(f"\nUpserted {count} item(s) to Pinecone (full replace, same item_ids).")
    else:
        print("\nNothing to upsert -- every item failed re-embedding.")

    print(f"\n{succeeded}/{total} items re-embedded successfully, {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
