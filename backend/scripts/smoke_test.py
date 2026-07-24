#!/usr/bin/env python
"""
End-to-end smoke test for the jewellery visual search pipeline.

Runs the real ingestion -> embedding -> vector search -> rerank flow using
the project's actual code, with every external API (Gemini, Pinecone)
replaced by a lightweight in-process fake. No real credentials or network
access are required.

Usage:
    python scripts/smoke_test.py

Exits non-zero if any step raises, or if the final result set is empty.
"""
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Same rationale as conftest.py: Settings' fields bind os.getenv(...) at
# class-definition time, so env vars must exist before `config` (or anything
# that imports it) is imported for the first time.
os.environ.setdefault("GEMINI_API_KEY", "fake-gemini-key")
os.environ.setdefault("EMBEDDING_MODEL", "gemini-embedding-2")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "9")
os.environ.setdefault("RERANKER_MODEL", "gemini-2.5-flash")
os.environ.setdefault("PINECONE_API_KEY", "fake-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "smoke-test-catalog")
os.environ.setdefault("PINECONE_CLOUD", "aws")
os.environ.setdefault("PINECONE_REGION", "us-east-1")
os.environ.setdefault("TOP_K", "3")
os.environ.setdefault("MIN_SIMILARITY_THRESHOLD", "0.55")
os.environ.setdefault("APP_API_KEY", "smoke-test-key")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

import embeddings
import preprocessing
import reranker
import vector_db
from config import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Fakes standing in for Gemini / Pinecone. Embeddings are derived from real
# image content (mean RGB, L2-normalized) rather than being random, so the
# similarity ranking below reflects genuine visual similarity between the
# fake catalog images and the fake query image.
# ---------------------------------------------------------------------------

def _color_embedding(image_bytes: bytes, dims: int) -> list:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    vec = np.tile(mean_rgb, dims // 3 + 1)[:dims]
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


def _fake_embed_content(*, model, contents, config):
    image_bytes = contents.inline_data.data
    values = _color_embedding(image_bytes, config.output_dimensionality)
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


class _FakeIndex:
    """In-memory stand-in for a Pinecone index: real cosine-similarity search
    over whatever has actually been upserted through vector_db.upsert_batch."""

    def __init__(self):
        self.store = {}

    def upsert(self, vectors):
        for item_id, vector, metadata in vectors:
            self.store[item_id] = (np.asarray(vector, dtype=np.float32), metadata)

    def query(self, vector, top_k, include_metadata, filter):
        q = np.asarray(vector, dtype=np.float32)
        scored = []
        for item_id, (vec, metadata) in self.store.items():
            score = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec) + 1e-9))
            scored.append({"id": item_id, "score": score, "metadata": metadata})
        scored.sort(key=lambda m: -m["score"])
        return {"matches": scored[:top_k]}


def _fake_generate_content(*, model, contents):
    """Stand-in reranker LLM call: derives a plausible confidence tier from
    the cosine score already computed by the fake vector search."""
    import json
    import re

    prompt = contents[-1]
    ids = re.findall(r"id: ([\w-]+), cosine_similarity: ([\d.]+)", prompt)
    judgments = []
    for item_id, score_str in ids:
        score = float(score_str)
        if score >= 0.97:
            confidence, reason = "high", "near-identical color profile"
        elif score >= 0.85:
            confidence, reason = "medium", "similar overall tone"
        else:
            confidence, reason = "low", "different color profile"
        judgments.append({"id": item_id, "confidence": confidence, "reason": reason})
    return SimpleNamespace(text=json.dumps(judgments))


def _make_jpeg_bytes(color, shape="rectangle") -> bytes:
    img = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse((60, 60, 196, 196), fill=tuple(min(c + 40, 255) for c in color))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def main() -> int:
    with patch.object(embeddings._client.models, "embed_content", side_effect=_fake_embed_content), \
         patch.object(reranker._client.models, "generate_content", side_effect=_fake_generate_content), \
         patch.object(vector_db, "get_or_create_index", return_value=_FakeIndex()):

        print("Ingesting 3 sample catalog images...")
        catalog = [
            {"id": "ring-red-01", "raw": _make_jpeg_bytes((200, 30, 30)), "metadata": {"category": "ring", "price": 199.0}},
            {"id": "necklace-blue-01", "raw": _make_jpeg_bytes((30, 60, 200)), "metadata": {"category": "necklace", "price": 299.0}},
            {"id": "earring-green-01", "raw": _make_jpeg_bytes((40, 160, 60)), "metadata": {"category": "earring", "price": 49.0}},
        ]
        to_upsert = []
        for item in catalog:
            clean_bytes = preprocessing.prepare_image_bytes(item["raw"])
            vector = embeddings.embed_catalog_image(clean_bytes)
            to_upsert.append({"id": item["id"], "vector": vector, "metadata": item["metadata"]})
        indexed_count = vector_db.upsert_batch(to_upsert)
        print(f"  -> indexed {indexed_count} items.")

        print("Running search against 1 query image...")
        # Same red as ring-red-01 plus a highlight circle, simulating a
        # customer photo of the same item under different lighting/angle.
        query_raw = _make_jpeg_bytes((210, 35, 25), shape="circle")
        query_clean = preprocessing.prepare_image_bytes(query_raw)
        query_vector = embeddings.embed_query_image(query_clean)

        raw_matches = vector_db.search(query_vector, top_k=settings.top_k)
        strong_matches = [m for m in raw_matches if m["score"] >= settings.min_similarity_threshold]
        if not strong_matches:
            ranked = []
        else:
            ranked = reranker.rerank(query_clean, strong_matches)

    print("\nFinal ranked results:")
    header = f"{'id':<20} {'similarity_percent':>18} {'confidence':>10}  reason"
    print(header)
    print("-" * len(header))
    for m in ranked:
        similarity_percent = round(m["score"] * 100, 1)
        print(f"{m['id']:<20} {similarity_percent:>18} {m['confidence']:>10}  {m['reason']}")

    if not ranked:
        print("\nFAILED: no results returned.")
        return 1

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nFAILED: smoke test raised an exception: {exc!r}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
