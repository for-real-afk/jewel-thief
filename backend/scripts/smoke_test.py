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


# Maps a color word in a text query to the same RGB used to build the
# matching catalog image below, so a text query like "red ring" fakes a
# semantically-plausible embedding without needing real Gemini text
# understanding. Real gemini-embedding-2 quality on color/material terms is
# NOT verified by this fake -- see README.md's Known Limitations note.
_COLOR_WORDS = {"red": (200, 30, 30), "blue": (30, 60, 200), "green": (40, 160, 60)}


def _fake_embed_content(*, model, contents, config):
    if isinstance(contents, str):
        text = contents.lower()
        color = next((rgb for word, rgb in _COLOR_WORDS.items() if word in text), (128, 128, 128))
        image_bytes = _make_jpeg_bytes(color)
    else:
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


def _print_results(title: str, ranked: list) -> None:
    print(f"\n{title}")
    header = f"{'id':<20} {'similarity_percent':>18} {'confidence':>10}  reason"
    print(header)
    print("-" * len(header))
    for m in ranked:
        similarity_percent = round(m["score"] * 100, 1)
        print(f"{m['id']:<20} {similarity_percent:>18} {m['confidence']:>10}  {m['reason']}")


def main() -> int:
    # Force the Gemini reranker path regardless of the real LLM_PROVIDER in
    # .env (e.g. "groq") -- otherwise reranker.rerank() would call the
    # unpatched utils.groq_*_chat functions and make a real network call
    # with real credentials, defeating the point of this fake-only script.
    with patch.object(embeddings._client.models, "embed_content", side_effect=_fake_embed_content), \
         patch.object(reranker._client.models, "generate_content", side_effect=_fake_generate_content), \
         patch.object(reranker.settings, "llm_provider", "gemini"), \
         patch.object(vector_db, "get_or_create_index", return_value=_FakeIndex()):

        print("Ingesting 3 sample catalog images...")
        catalog = [
            {"id": "ring-red-01", "raw": _make_jpeg_bytes((200, 30, 30)),
             "metadata": {"category": "ring", "price": 199.0, "name": "Ruby Halo Ring", "caption": "a red gemstone ring"}},
            {"id": "necklace-blue-01", "raw": _make_jpeg_bytes((30, 60, 200)),
             "metadata": {"category": "necklace", "price": 299.0, "name": "Sapphire Drop Necklace", "caption": "a blue gemstone necklace"}},
            {"id": "earring-green-01", "raw": _make_jpeg_bytes((40, 160, 60)),
             "metadata": {"category": "earring", "price": 49.0, "name": "Emerald Stud Earring", "caption": "a green gemstone earring"}},
        ]
        to_upsert = []
        for item in catalog:
            clean_bytes = preprocessing.prepare_image_bytes(item["raw"])
            vector = embeddings.embed_catalog_image(clean_bytes)
            to_upsert.append({"id": item["id"], "vector": vector, "metadata": item["metadata"]})
        indexed_count = vector_db.upsert_batch(to_upsert)
        print(f"  -> indexed {indexed_count} items.")

        print("Running image search against 1 query image...")
        # Same red as ring-red-01 plus a highlight circle, simulating a
        # customer photo of the same item under different lighting/angle.
        query_raw = _make_jpeg_bytes((210, 35, 25), shape="circle")
        query_clean = preprocessing.prepare_image_bytes(query_raw)
        query_vector = embeddings.embed_query_image(query_clean)

        raw_matches = vector_db.search(query_vector, top_k=settings.top_k)
        strong_matches = [m for m in raw_matches if m["score"] >= settings.min_similarity_threshold]
        image_ranked = [] if not strong_matches else reranker.rerank(
            {"type": "image", "bytes": query_clean}, strong_matches
        )

        print("Running text search against 1 query string...")
        query_text = "a blue gemstone necklace"
        text_query_vector = embeddings.embed_text_query(query_text)

        text_raw_matches = vector_db.search(text_query_vector, top_k=settings.top_k)
        text_strong_matches = [m for m in text_raw_matches if m["score"] >= settings.min_similarity_threshold]
        text_ranked = [] if not text_strong_matches else reranker.rerank(
            {"type": "text", "text": query_text}, text_strong_matches
        )

    _print_results("Image search - final ranked results:", image_ranked)
    _print_results(f'Text search ("{query_text}") - final ranked results:', text_ranked)

    if not image_ranked or not text_ranked:
        print("\nFAILED: no results returned for one or both search flows.")
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
