"""
Pinecone vector database: index setup, batch upsert, filtered ANN search.
"""
from pinecone import Pinecone, ServerlessSpec

from config import get_settings
from utils import external_api_retry

settings = get_settings()
_pc = Pinecone(api_key=settings.pinecone_api_key)


def ping() -> None:
    """Cheap, read-only Pinecone reachability check for GET /health --
    list_indexes(), not get_or_create_index() (which can have the side
    effect of creating an index on a misconfigured name). Raises on
    failure; callers decide how to report that."""
    _pc.list_indexes()


def get_or_create_index():
    existing = [idx["name"] for idx in _pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        _pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return _pc.Index(settings.pinecone_index_name)


def upsert_batch(items: list[dict]) -> int:
    """
    items: [{"id": str, "vector": list[float], "metadata": {...}}, ...]
    Batches in chunks of 100 (Pinecone's recommended upsert batch size).
    """
    index = get_or_create_index()
    count = 0
    for i in range(0, len(items), 100):
        chunk = items[i:i + 100]
        vectors = [(item["id"], item["vector"], item["metadata"]) for item in chunk]
        index.upsert(vectors=vectors)
        count += len(chunk)
    return count


def update_metadata(item_id: str, metadata: dict) -> None:
    """Replace an existing vector's metadata in place, without touching its
    embedding — used for metadata-only catalog edits (no new image), so an
    edit doesn't cost a re-embedding call for a vector that hasn't changed."""
    index = get_or_create_index()
    index.update(id=item_id, set_metadata=metadata)


def delete_by_id(item_id: str) -> None:
    index = get_or_create_index()
    index.delete(ids=[item_id])


@external_api_retry
def search(
    query_vector: list[float],
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    ANN search with optional metadata filtering (price range, category, material).
    Returns raw Pinecone matches: [{"id", "score", "metadata"}, ...]
    """
    index = get_or_create_index()
    result = index.query(
        vector=query_vector,
        top_k=top_k or settings.top_k,
        include_metadata=True,
        filter=metadata_filter or {},
    )
    # Pinecone's approximate cosine computation can overshoot slightly past
    # [0, 1] (e.g. 1.004) due to floating-point error in the ANN index -- this
    # is the one place every result flows through, so the clamp guarantee
    # belongs here rather than scattered across every caller.
    return [
        {"id": m["id"], "score": max(0.0, min(m["score"], 1.0)), "metadata": m.get("metadata", {})}
        for m in result["matches"]
    ]
