#!/usr/bin/env python
"""
Index every image in a local folder into the real catalog (real Pinecone,
real embedding provider from .env — currently LM Studio).

Since these files don't come with real product copy, name/category are
derived straight from the filename (e.g. "necklace (14).jpg" -> category
"necklace") rather than spending a vision-model caption call per image —
this dataset's filenames already say what they are.

Writes sample_data/real_catalog/metadata.json incrementally as it goes, and
skips any item_id already recorded there on a re-run, so an interrupted run
can just be restarted.

Usage:
    python scripts/index_local_folder.py "C:\\path\\to\\folder"
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embeddings
import vector_db
from main import infer_category
from preprocessing import prepare_image_bytes, InvalidImageError

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "real_catalog"
MANIFEST_PATH = MANIFEST_DIR / "metadata.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/index_local_folder.py <folder>", file=sys.stderr)
        return 1

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"Found {len(files)} image(s) in {folder}")

    manifest = load_manifest()
    seen_ids = set()
    to_upsert = []
    skipped = 0
    failed = []

    for i, path in enumerate(files):
        stem = path.stem
        item_id = slugify(stem)
        if item_id in seen_ids:
            item_id = f"{item_id}-{i}"
        seen_ids.add(item_id)

        if item_id in manifest:
            skipped += 1
            continue

        category = infer_category(stem)
        print(f"[{i+1}/{len(files)}] {path.name} -> id={item_id} category={category}", flush=True)

        try:
            raw = path.read_bytes()
            clean = prepare_image_bytes(raw)
            vector = embeddings.embed_catalog_image(clean)
        except InvalidImageError as exc:
            print(f"  -> SKIPPED (invalid image): {exc}", flush=True)
            failed.append({"file": path.name, "error": str(exc)})
            continue
        except Exception as exc:
            print(f"  -> FAILED: {exc!r}", flush=True)
            failed.append({"file": path.name, "error": repr(exc)})
            continue

        metadata = {"filename": path.name, "name": stem, "category": category}
        to_upsert.append({"id": item_id, "vector": vector, "metadata": metadata})
        manifest[item_id] = metadata
        save_manifest(manifest)  # incremental, so a crash mid-run doesn't lose progress

    if to_upsert:
        count = vector_db.upsert_batch(to_upsert)
        print(f"\nUpserted {count} new item(s) to Pinecone.")
    else:
        print("\nNothing new to upsert.")

    print(f"Skipped (already indexed): {skipped}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f['file']}: {f['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
