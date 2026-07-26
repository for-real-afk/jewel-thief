#!/usr/bin/env python
"""
End-to-end smoke test for the jewellery visual search pipeline.

Calls main.py's actual _search_image()/_search_text() functions -- not a
reimplementation of the search flow -- so this exercises the real domain
gate, conditional-rerank, and caching logic, with every external API
(Gemini, Pinecone) replaced by a lightweight in-process fake. No real
credentials or network access are required.

Usage:
    python scripts/smoke_test.py

Exits non-zero if any step raises, or if a search that should return results
returns none.
"""
import io
import json
import os
import re
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
os.environ.setdefault("TOP_K", "5")
os.environ.setdefault("MIN_SIMILARITY_THRESHOLD", "0.55")
os.environ.setdefault("APP_API_KEY", "smoke-test-key")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")
os.environ.setdefault("SUPABASE_URL", "https://fake-project.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "fake-supabase-service-key")
# object_storage.py builds a boto3 client at import time and validates its
# endpoint URL immediately -- same rationale as conftest.py's R2_* defaults.
os.environ.setdefault("R2_ACCOUNT_ID", "fake-account-id")
os.environ.setdefault("R2_ACCESS_KEY_ID", "fake-access-key-id")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "fake-secret-access-key")
os.environ.setdefault("R2_BUCKET_NAME", "smoke-test-catalog")
os.environ.setdefault("R2_PUBLIC_URL_BASE", "https://pub-fake.r2.dev")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageDraw

import embeddings
import main
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


# Maps a color word in a text query to an RGB used to fake a
# semantically-plausible embedding without needing real Gemini text
# understanding. Real gemini-embedding-2 quality on color/material terms is
# NOT verified by this fake -- see README.md's Known Limitations note.
_COLOR_WORDS = {"red": (200, 30, 30), "blue": (30, 60, 200), "green": (40, 160, 60)}

_TEXT_QUERY_PREFIX = embeddings._TEXT_QUERY_TASK_PREFIX

embed_text_query_calls = {"count": 0}


def _fake_embed_content(*, model, contents, config):
    if isinstance(contents, str):
        text = contents
        if text.startswith(_TEXT_QUERY_PREFIX):
            embed_text_query_calls["count"] += 1
            text = text[len(_TEXT_QUERY_PREFIX):]
        color = next((rgb for word, rgb in _COLOR_WORDS.items() if word in text.lower()), (128, 128, 128))
        image_bytes = _make_jpeg_bytes(color)
    elif isinstance(contents, list):
        # embed_catalog_item: [Part, text_description] -- fuse by embedding
        # the image (the fake ignores the text half, same as the real model
        # would still be dominated by pixel content for a solid-color image).
        image_bytes = contents[0].inline_data.data
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


def _fake_generate_content(*, model, contents, config=None):
    """Stand-in for both reranker.is_plausibly_jewelry's gate call and
    reranker._judge_gemini's judge call -- distinguished by prompt content,
    same way a single real model handles both, just via one endpoint here."""
    prompt = contents[-1] if isinstance(contents, list) else contents

    if "yes or no" in prompt.lower():
        return SimpleNamespace(text="yes")  # every fake query image "is jewelry"

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


def _print_response(title: str, resp) -> None:
    print(f"\n{title}")
    print(f"  no_match={resp.no_match} query_type={resp.query_type} reason={resp.reason}")
    header = f"  {'id':<22} {'similarity_percent':>18} {'confidence':>10}  reason"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in resp.matches:
        print(f"  {m.id:<22} {m.similarity_percent:>18} {m.confidence:>10}  {m.reason}")


async def main_() -> int:
    # Force the Gemini reranker path regardless of the real LLM_PROVIDER in
    # .env (e.g. "groq") -- otherwise reranker.rerank() would call the
    # unpatched utils.groq_*_chat functions and make a real network call
    # with real credentials, defeating the point of this fake-only script.
    with patch.object(embeddings._client.models, "embed_content", side_effect=_fake_embed_content), \
         patch.object(reranker._client.models, "generate_content", side_effect=_fake_generate_content), \
         patch.object(reranker.settings, "llm_provider", "gemini"), \
         patch.object(vector_db, "get_or_create_index", return_value=_FakeIndex()):

        print("Ingesting 5 sample catalog images (fused image+text embedding)...")
        catalog = [
            {"id": "ring-red-01", "raw": _make_jpeg_bytes((200, 30, 30)),
             "metadata": {"category": "ring", "price": 199.0, "name": "Ruby Halo Ring", "caption": "a red gemstone ring"}},
            {"id": "necklace-blue-01", "raw": _make_jpeg_bytes((30, 60, 200)),
             "metadata": {"category": "necklace", "price": 299.0, "name": "Sapphire Drop Necklace", "caption": "a blue gemstone necklace"}},
            {"id": "earring-green-01", "raw": _make_jpeg_bytes((40, 160, 60)),
             "metadata": {"category": "earrings", "price": 49.0, "name": "Emerald Stud Earring", "caption": "a green gemstone earring"}},
            {"id": "earring-green-02", "raw": _make_jpeg_bytes((42, 158, 58)),
             "metadata": {"category": "earrings", "price": 59.0, "name": "Peridot Stud Earring", "caption": "a green gemstone earring"}},
            {"id": "earring-green-03", "raw": _make_jpeg_bytes((38, 162, 62)),
             "metadata": {"category": "earrings", "price": 55.0, "name": "Jade Stud Earring", "caption": "a green gemstone earring"}},
        ]
        to_upsert = []
        for item in catalog:
            clean_bytes = preprocessing.prepare_image_bytes(item["raw"])
            text_description = main.build_catalog_text_description(
                item["metadata"]["name"], item["metadata"]["caption"], "", []
            )
            vector = embeddings.embed_catalog_item(clean_bytes, text_description)
            to_upsert.append({"id": item["id"], "vector": vector, "metadata": item["metadata"]})
        indexed_count = vector_db.upsert_batch(to_upsert)
        print(f"  -> indexed {indexed_count} items.")

        print("\n--- Image search (domain gate passes, always LLM rerank) ---")
        # Same red as ring-red-01 plus a highlight circle, simulating a
        # customer photo of the same item under different lighting/angle.
        query_raw = _make_jpeg_bytes((210, 35, 25), shape="circle")
        upload = UploadFile(file=io.BytesIO(query_raw), filename="query.jpg")
        image_resp, _ = await main._search_image(upload, {})
        _print_response("Image search results:", image_resp)

        print("\n--- Text search: well-separated query (expect cheap path, zero LLM calls) ---")
        before = _fake_generate_content_calls()
        separated_resp, _ = main._search_text("a blue gemstone necklace", {})
        separated_gen_calls = _fake_generate_content_calls() - before
        _print_response('Text search ("a blue gemstone necklace") results:', separated_resp)
        print(f"  generate_content calls during this search: {separated_gen_calls} "
              f"(expected 0 -- cheap path should skip the LLM entirely)")

        print("\n--- Text search: same query again (expect cache hit, zero embed calls) ---")
        before = embed_text_query_calls["count"]
        main._search_text("a blue gemstone necklace", {})
        repeat_embed_calls = embed_text_query_calls["count"] - before
        print(f"  embed_text_query calls during repeat search: {repeat_embed_calls} "
              f"(expected 0 -- second identical search should hit the cache)")

        print("\n--- Text search: ambiguous query (expect LLM rerank to fire) ---")
        before = _fake_generate_content_calls()
        ambiguous_resp, _ = main._search_text("a green gemstone earring", {})
        ambiguous_gen_calls = _fake_generate_content_calls() - before
        _print_response('Text search ("a green gemstone earring") results:', ambiguous_resp)
        print(f"  generate_content calls during this search: {ambiguous_gen_calls} "
              f"(expected 1 -- ambiguous top results should trigger the LLM judge)")

    ok = (
        not image_resp.no_match and len(image_resp.matches) > 0
        and not separated_resp.no_match and len(separated_resp.matches) > 0
        and separated_gen_calls == 0  # cheap path really skipped the LLM
        and repeat_embed_calls == 0  # cache really skipped re-embedding
        and ambiguous_gen_calls >= 1  # ambiguous case really called the LLM
        and not ambiguous_resp.no_match and len(ambiguous_resp.matches) > 0
    )
    if not ok:
        print("\nFAILED: one or more search flows didn't behave as expected.")
        return 1

    print("\nSmoke test passed.")
    return 0


def _fake_generate_content_calls() -> int:
    return reranker._client.models.generate_content.call_count


if __name__ == "__main__":
    import asyncio

    try:
        sys.exit(asyncio.run(main_()))
    except Exception as exc:
        print(f"\nFAILED: smoke test raised an exception: {exc!r}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
