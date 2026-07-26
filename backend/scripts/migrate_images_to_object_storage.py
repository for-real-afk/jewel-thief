#!/usr/bin/env python
"""
One-off migration: uploads every already-committed catalog image
(backend/static/catalog/*.jpg) to R2 (object_storage.py) and backfills
image_url on the corresponding Supabase catalog_items row (via
vector_db.update_metadata + catalog_store.record_item -- merge semantics,
NOT a re-embed: the vector itself is untouched, only the image_url metadata
changes, same as a metadata-only PATCH /api/v1/catalog/items/{item_id}).

After this runs (for real, not --dry-run) and is verified, the git-tracked
static/catalog/*.jpg files are no longer the source of truth going forward
-- new items already upload straight to R2 (main.py, since the object
storage cutover). This script only catches up items indexed before that.

Usage:
    python scripts/migrate_images_to_object_storage.py --dry-run   # uploads for real, skips the metadata write
    python scripts/migrate_images_to_object_storage.py             # uploads AND backfills image_url for real
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import catalog_store
from app import object_storage
from app import vector_db

STATIC_CATALOG_DIR = Path(__file__).resolve().parent.parent / "app" / "static" / "catalog"
PAGE_SIZE = 100
PROGRESS_EVERY = 10


def fetch_all_catalog_items() -> list[dict]:
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


def migrate_item(item: dict, dry_run: bool) -> str:
    """Uploads item's stored image to R2 and (unless dry_run) backfills
    image_url in Pinecone + Supabase. Returns the new image_url. Raises on
    any failure; caller catches per-item so one bad item doesn't abort the
    whole run."""
    item_id = item["item_id"]
    image_path = STATIC_CATALOG_DIR / f"{item_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"no stored image at {image_path}")

    image_bytes = image_path.read_bytes()
    if dry_run:
        return f"{object_storage.settings.r2_public_url_base}/{object_storage._key_for(item_id)}"

    image_url = object_storage.upload_catalog_image(item_id, image_bytes)
    metadata = {k: v for k, v in item.items() if k != "item_id"}
    metadata["image_url"] = image_url
    vector_db.update_metadata(item_id, metadata)
    catalog_store.record_item(item_id, metadata)
    return image_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded and which image_url would result, without uploading or writing anything.",
    )
    args = parser.parse_args()

    items = fetch_all_catalog_items()
    total = len(items)
    print(f"Found {total} catalog item(s){' (DRY RUN)' if args.dry_run else ''}.")
    if total == 0:
        print("Nothing to do.")
        return 0

    migrated = []
    failed = []
    for i, item in enumerate(items, 1):
        item_id = item.get("item_id", "<unknown>")
        try:
            image_url = migrate_item(item, args.dry_run)
            migrated.append((item_id, image_url))
        except Exception as exc:
            print(f"  [{i}/{total}] FAILED item_id={item_id}: {exc!r}", flush=True)
            failed.append(item_id)
            continue
        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"  [{i}/{total}] {'would migrate' if args.dry_run else 'migrated'} item_id={item_id}", flush=True)

    print(f"\n{len(migrated)}/{total} item(s) {'would be migrated' if args.dry_run else 'migrated'}, "
          f"{len(failed)} failed: {failed}")
    if args.dry_run and migrated:
        print("\nSample resulting image_url values:")
        for item_id, image_url in migrated[:5]:
            print(f"  - {item_id}: {image_url}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
