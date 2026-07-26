#!/usr/bin/env python
"""
Re-runs catalog indexing against a REMOTE, already-deployed backend (e.g.
Render) via real HTTP POSTs to /api/v1/catalog/index — as opposed to every
other indexing script here, which writes to Pinecone directly via a local
Python import.

Why this needs to exist at all: Render's web service disk is ephemeral (see
DEPLOYMENT.md, README.md §11). Catalog images indexed locally during
development never existed on the deployed instance's disk — only the
Pinecone vectors + metadata (including image_url) did — so every thumbnail
404s despite search itself working correctly. POSTing through the real
endpoint is what actually writes the image files server-side.

This is a stopgap, not a permanent fix: images written this way vanish
again on Render's next restart/redeploy, same as the original problem. The
real fix is object storage (S3/R2/Render Disk) — out of scope here.

Sources enumerated (item_id -> source image path):
  - sample_data/catalog/metadata.json      (4 Wikimedia demo items)
  - sample_data/real_catalog/metadata.json (75 local-folder items)
  - catalog_store.json                      (items added via the live
                                              /catalog admin page)

Items from the first two sources predate the price/caption/description/tags
fields (indexed before the catalog admin feature existed) and have no real
price in Pinecone — PLACEHOLDER_PRICE is used for those and is clearly
marked as such in the request; replace with real pricing before this is a
real storefront.

Usage:
    python scripts/reindex_to_remote.py <base_url> <api_key>
    python scripts/reindex_to_remote.py https://jewel-thief.onrender.com sk-...
"""
import json
import sys
import time
from pathlib import Path

import requests

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
from app.main import infer_category  # noqa: E402
WIKIMEDIA_MANIFEST = BACKEND_DIR / "sample_data" / "catalog" / "metadata.json"
WIKIMEDIA_SOURCE_DIR = BACKEND_DIR / "sample_data" / "catalog"
LOCAL_MANIFEST = BACKEND_DIR / "sample_data" / "real_catalog" / "metadata.json"
LOCAL_SOURCE_DIR = Path(r"C:\Users\deepa\Downloads\files (1)")
CATALOG_STORE = BACKEND_DIR / "catalog_store.json"

PLACEHOLDER_PRICE = 99.0
BATCH_SIZE = 10
POLL_INTERVAL_SECONDS = 3


def _display_name_from_id(item_id: str) -> str:
    return item_id.replace("-", " ").title()


def load_items() -> list[dict]:
    """Returns [{"item_id", "source_path", "name", "category", "price",
    "caption", "description", "tags", "material"}, ...]."""
    items = []

    for item_id, meta in json.loads(WIKIMEDIA_MANIFEST.read_text()).items():
        # The original Wikimedia title (e.g. "File:Anita-bespoke-custom
        # made-salt and pepper diamond-engagement ring-...") is unusable as a
        # display name and these items were never categorized at fetch time
        # — derive both from the item_id instead (e.g. "ring-gold-diamond").
        items.append({
            "item_id": item_id,
            "source_path": WIKIMEDIA_SOURCE_DIR / meta["filename"],
            "name": _display_name_from_id(item_id),
            "category": infer_category(item_id),
            "price": PLACEHOLDER_PRICE,
        })

    for item_id, meta in json.loads(LOCAL_MANIFEST.read_text()).items():
        items.append({
            "item_id": item_id,
            "source_path": LOCAL_SOURCE_DIR / meta["filename"],
            "name": meta.get("name", item_id),
            "category": meta.get("category", "unknown"),
            "price": PLACEHOLDER_PRICE,
        })

    if CATALOG_STORE.exists():
        for item_id, meta in json.loads(CATALOG_STORE.read_text()).items():
            # These filenames are the ORIGINAL upload filename, which may
            # collide across sources (e.g. reused a Wikimedia sample) — try
            # both plausible source directories.
            candidate = WIKIMEDIA_SOURCE_DIR / meta["filename"]
            if not candidate.exists():
                candidate = LOCAL_SOURCE_DIR / meta["filename"]
            if not candidate.exists():
                # Fall back to Downloads root (e.g. a one-off admin upload).
                candidate = Path.home() / "Downloads" / meta["filename"]
            items.append({
                "item_id": item_id,
                "source_path": candidate,
                "name": meta.get("name", item_id),
                "category": meta.get("category", "unknown"),
                "price": meta.get("price") or PLACEHOLDER_PRICE,
                "caption": meta.get("caption", ""),
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "material": meta.get("material"),
            })

    return items


def post_batch(base_url: str, api_key: str, batch: list[dict]) -> str | None:
    files = []
    items_json = []
    for item in batch:
        if not item["source_path"].exists():
            print(f"  -> MISSING source file for {item['item_id']}: {item['source_path']}")
            continue
        files.append(("images", (item["source_path"].name, item["source_path"].read_bytes(), "image/jpeg")))
        entry = {
            "item_id": item["item_id"],
            "name": item["name"],
            "category": item["category"],
            "price": item["price"],
            "caption": item.get("caption", ""),
            "description": item.get("description", ""),
            "tags": item.get("tags", []),
        }
        if item.get("material"):
            entry["material"] = item["material"]
        items_json.append(entry)

    if not files:
        return None

    resp = requests.post(
        f"{base_url}/api/v1/catalog/index",
        headers={"x-api-key": api_key},
        files=files,
        data={"items_json": json.dumps(items_json)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def wait_for_job(base_url: str, api_key: str, job_id: str) -> dict:
    while True:
        resp = requests.get(
            f"{base_url}/api/v1/catalog/jobs/{job_id}", headers={"x-api-key": api_key}, timeout=30
        )
        resp.raise_for_status()
        status = resp.json()
        if status["status"] != "pending":
            return status
        print(f"  -> {status['processed']}/{status['total']} processed...", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/reindex_to_remote.py <base_url> <api_key>", file=sys.stderr)
        return 1

    base_url = sys.argv[1].rstrip("/")
    api_key = sys.argv[2]

    items = load_items()
    print(f"Loaded {len(items)} items from local manifests.")

    total_ok, total_failed = 0, 0
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"\nBatch {batch_num} ({len(batch)} items): uploading...", flush=True)
        job_id = post_batch(base_url, api_key, batch)
        if job_id is None:
            print("  -> nothing to upload in this batch, skipping.")
            continue
        result = wait_for_job(base_url, api_key, job_id)
        ok = result["total"] - len(result["failed_items"])
        total_ok += ok
        total_failed += len(result["failed_items"])
        print(f"  -> done: {ok}/{result['total']} succeeded.")
        for f in result["failed_items"]:
            print(f"     FAILED {f['item_id']}: {f['error']}")

    print(f"\nTotal: {total_ok} succeeded, {total_failed} failed.")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
