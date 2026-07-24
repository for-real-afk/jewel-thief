#!/usr/bin/env python
"""
One-off migration: copies every already-indexed catalog image into
static/catalog/<item_id>.<ext> (served by main.py's StaticFiles mount) and
backfills an "image_url" field onto each item's existing Pinecone metadata
via index.update(set_metadata=...) — a partial merge, so name/category/price
already stored are left untouched and no re-embedding is needed.

Reads the two manifests written by fetch_sample_images.py and
index_local_folder.py to know where each item's source file actually lives.

Usage:
    python scripts/backfill_image_urls.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vector_db

BACKEND_DIR = Path(__file__).resolve().parent.parent
STATIC_CATALOG_DIR = BACKEND_DIR / "static" / "catalog"

WIKIMEDIA_MANIFEST = BACKEND_DIR / "sample_data" / "catalog" / "metadata.json"
WIKIMEDIA_SOURCE_DIR = BACKEND_DIR / "sample_data" / "catalog"

LOCAL_MANIFEST = BACKEND_DIR / "sample_data" / "real_catalog" / "metadata.json"
LOCAL_SOURCE_DIR = Path(r"C:\Users\deepa\Downloads\files (1)")


def copy_and_collect(manifest_path: Path, source_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    urls = {}
    for item_id, meta in manifest.items():
        src = source_dir / meta["filename"]
        if not src.exists():
            print(f"  -> MISSING source file for {item_id}: {src}")
            continue
        dest_name = f"{item_id}{src.suffix.lower()}"
        dest = STATIC_CATALOG_DIR / dest_name
        shutil.copyfile(src, dest)
        urls[item_id] = f"/static/catalog/{dest_name}"
    return urls


def main() -> int:
    STATIC_CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Copying Wikimedia demo catalog images...")
    urls = copy_and_collect(WIKIMEDIA_MANIFEST, WIKIMEDIA_SOURCE_DIR)
    print(f"  -> {len(urls)} copied")

    print("Copying local folder catalog images...")
    local_urls = copy_and_collect(LOCAL_MANIFEST, LOCAL_SOURCE_DIR)
    print(f"  -> {len(local_urls)} copied")
    urls.update(local_urls)

    print(f"\nBackfilling image_url metadata for {len(urls)} item(s) in Pinecone...")
    index = vector_db.get_or_create_index()
    failed = []
    for i, (item_id, url) in enumerate(urls.items(), 1):
        try:
            index.update(id=item_id, set_metadata={"image_url": url})
        except Exception as exc:
            failed.append((item_id, str(exc)))
            continue
        if i % 10 == 0 or i == len(urls):
            print(f"  [{i}/{len(urls)}] updated", flush=True)

    if failed:
        print(f"\nFailed to update {len(failed)} item(s):")
        for item_id, err in failed:
            print(f"  - {item_id}: {err}")

    print(f"\nDone. {len(urls) - len(failed)} item(s) now have image_url metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
