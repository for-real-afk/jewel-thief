#!/usr/bin/env python
"""
Downloads a small set of real, appropriately-licensed jewellery photos from
Wikimedia Commons into sample_data/catalog/ (plus one held-out query image),
for use with scripts/live_demo.py. Not part of the pytest suite — this hits
the real internet and only needs to be run once (or re-run to refresh).

Writes sample_data/catalog/metadata.json recording title/artist/license/source
URL for each image, since Commons content requires attribution.
"""
import json
import sys
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "jewel-thief-sample-data-fetcher/1.0 (local dev/testing use)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

CATALOG_SEARCHES = {
    "ring-gold-diamond": "gold diamond engagement ring jewellery",
    "necklace-pearl": "pearl necklace jewellery",
    "earrings-gemstone": "gemstone earrings jewellery",
    "bracelet-gold": "gold bracelet jewellery",
}
# Fetched by exact title (not fuzzy search) since it's a specific, verified,
# non-portrait macro shot of a diamond ring — a meaningful visual match
# against catalog/ring-gold-diamond.jpg for the live search demo.
QUERY_TITLES = {"query-ring": "File:Diamond princess cut.jpg"}

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "catalog"
QUERY_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "query"


def search_commons_image(query: str) -> dict | None:
    resp = requests.get(
        COMMONS_API,
        params={
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": 600,
        },
        headers=HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        meta = info.get("extmetadata", {})
        return {
            "title": page.get("title", ""),
            "url": info.get("thumburl") or info.get("url"),
            "mime": info["mime"],
            "artist": _strip_html(meta.get("Artist", {}).get("value", "unknown")),
            "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "source_page": info.get("descriptionurl", ""),
        }
    return None


def fetch_by_title(title: str) -> dict | None:
    resp = requests.get(
        COMMONS_API,
        params={
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": title,
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": 600,
        },
        headers=HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        if not info:
            return None
        meta = info.get("extmetadata", {})
        return {
            "title": page.get("title", ""),
            "url": info.get("thumburl") or info.get("url"),
            "mime": info["mime"],
            "artist": _strip_html(meta.get("Artist", {}).get("value", "unknown")),
            "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "source_page": info.get("descriptionurl", ""),
        }
    return None


def _strip_html(s: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", s).strip()


def download(url: str, dest: Path) -> None:
    for attempt in range(5):
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            print(f"  -> rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return
    raise RuntimeError(f"Gave up downloading {url} after repeated 429s.")


def _load_existing_metadata(out_dir: Path) -> dict:
    meta_path = out_dir / "metadata.json"
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def fetch_set(searches: dict, out_dir: Path) -> dict:
    """Commons full-text search is not deterministic across calls — re-running
    it for an item whose file already exists on disk could return a *different*
    photo than the one actually saved, silently mismatching the attribution.
    So: search (and download) only for items not already on disk; for
    already-downloaded items, carry forward their previously recorded metadata
    unchanged rather than re-querying."""
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_metadata = _load_existing_metadata(out_dir)
    metadata = {}
    for item_id, query in searches.items():
        existing = list(out_dir.glob(f"{item_id}.*"))
        if existing and item_id in previous_metadata:
            print(f"Already have {existing[0].name} with recorded metadata, skipping.")
            metadata[item_id] = previous_metadata[item_id]
            continue
        print(f"Searching Commons for '{query}'...")
        result = search_commons_image(query)
        if not result:
            print(f"  -> no usable result for {item_id}, skipping.")
            continue
        ext = ".jpg" if result["mime"] == "image/jpeg" else ".png"
        filename = f"{item_id}{ext}"
        download(result["url"], out_dir / filename)
        print(f"  -> saved {filename} ({result['title']}, {result['license']})")
        metadata[item_id] = {"filename": filename, **result}
        time.sleep(5)
    return metadata


def fetch_titles(titles: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_metadata = _load_existing_metadata(out_dir)
    metadata = {}
    for item_id, title in titles.items():
        existing = list(out_dir.glob(f"{item_id}.*"))
        if existing and item_id in previous_metadata:
            print(f"Already have {existing[0].name} with recorded metadata, skipping.")
            metadata[item_id] = previous_metadata[item_id]
            continue
        print(f"Fetching Commons title '{title}'...")
        result = fetch_by_title(title)
        if not result or result["mime"] not in ("image/jpeg", "image/png"):
            print(f"  -> no usable result for {item_id}, skipping.")
            continue
        ext = ".jpg" if result["mime"] == "image/jpeg" else ".png"
        filename = f"{item_id}{ext}"
        download(result["url"], out_dir / filename)
        print(f"  -> saved {filename} ({result['title']}, {result['license']})")
        metadata[item_id] = {"filename": filename, **result}
        time.sleep(5)
    return metadata


def main() -> int:
    # Some Commons titles/artists contain characters outside the Windows
    # console's default cp1252 codepage (e.g. Polish "ł") — don't let a
    # print() crash the download after the file is already saved.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    catalog_meta = fetch_set(CATALOG_SEARCHES, OUT_DIR)
    query_meta = fetch_titles(QUERY_TITLES, QUERY_DIR)

    (OUT_DIR / "metadata.json").write_text(json.dumps(catalog_meta, indent=2))
    (QUERY_DIR / "metadata.json").write_text(json.dumps(query_meta, indent=2))

    if not catalog_meta or not query_meta:
        print("FAILED: could not fetch a complete sample image set.", file=sys.stderr)
        return 1
    print(f"\nDone. {len(catalog_meta)} catalog images, {len(query_meta)} query image(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
