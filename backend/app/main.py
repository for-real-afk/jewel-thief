"""
Multimodal Visual Jewellery Search Engine — FastAPI backend.

Endpoints:
  POST   /api/v1/search                    — upload a reference image, get ranked matches
  POST   /api/v1/catalog/index             — batch-upsert catalog items (images + items_json)
  GET    /api/v1/catalog/jobs/{job_id}     — indexing job progress/result
  GET    /api/v1/catalog/items             — paginated list of indexed catalog items
  GET    /api/v1/catalog/items/{item_id}   — single catalog item
  PATCH  /api/v1/catalog/items/{item_id}   — edit an item's metadata (and optionally its image)
  DELETE /api/v1/catalog/items/{item_id}   — remove an item from the catalog
  GET    /health                           — healthcheck
"""
import hashlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import sentry_sdk
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import api_keys
from . import cache
from . import catalog_store
from . import embeddings
from . import job_store
from . import logging_config
from . import object_storage
from . import rate_limit
from . import search_events
from . import vector_db
from . import reranker
from .config import get_settings
from .preprocessing import prepare_image_bytes, InvalidImageError

settings = get_settings()
logging_config.configure_logging()
logger = logging.getLogger("jewellery_search")

# No-op (sentry_sdk.init is never called) when SENTRY_DSN is unset -- capture
# calls below become harmless no-ops too (sentry_sdk's documented behavior
# when the SDK isn't initialized), so this is safe to leave in for local dev.
if settings.sentry_dsn:
    sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.0)

app = FastAPI(title="Jewellery Visual Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=settings.allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Without this, an unhandled exception (e.g. a downstream API outage)
    propagates past CORSMiddleware entirely to Starlette's default error
    handler, which returns a response with NO CORS headers at all — the
    browser then reports a misleading "CORS policy" error that has nothing
    to do with CORS configuration, masking the real 500 and its actual cause.
    Catching it here keeps the response inside the normal middleware chain,
    so CORSMiddleware still gets to attach its headers."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    sentry_sdk.capture_exception(exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})

# Search results need something to actually show the user — catalog images
# are persisted here at indexing time and served back as image_url metadata,
# rather than only ever existing transiently as an upload the pipeline embeds
# and discards.
STATIC_DIR = Path(__file__).resolve().parent / "static"
CATALOG_IMAGE_DIR = STATIC_DIR / "catalog"
CATALOG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def require_api_key(x_api_key: Optional[str] = Header(None)) -> tuple[str, str]:
    """Validates the incoming key and returns (client_name, rate_limit_tier)
    -- used by rate_limit.py's per-tier checks and, later, structured
    logging. The legacy shared APP_API_KEY is still accepted, mapped to
    ("legacy", "legacy"), as a deprecation path during rollout of per-client
    keys (api_keys.py).

    RETIREMENT DATE: 2026-10-27 (90 days from this fallback shipping). Once
    every real client has been issued a per-client key via
    scripts/create_api_key.py, remove the `x_api_key == settings.api_key`
    branch below and drop APP_API_KEY from .env.example/render.yaml. See
    README.md §14 and PRODUCTION_HARDENING_PLAN.md's Risks table -- this is
    a dated commitment, not an "eventually"."""
    if x_api_key == settings.api_key:
        return "legacy", "legacy"
    key_record = x_api_key and api_keys.lookup_key(x_api_key)
    if key_record is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key_record["client_name"], key_record["rate_limit_tier"]


def _enforce_rate_limit(scope: str, client_name: str, tier: str) -> None:
    try:
        rate_limit.check(scope, client_name, tier)
    except rate_limit.RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )


class MatchResult(BaseModel):
    id: str
    similarity_percent: float
    confidence: str
    reason: str
    metadata: dict


class SearchResponse(BaseModel):
    query_id: str
    matches: list[MatchResult]
    no_match: bool
    query_type: Literal["image", "text"]
    reason: Optional[str] = None


class FeedbackRequest(BaseModel):
    result_id: str = Field(min_length=1)
    action: Literal["clicked", "purchased", "dismissed"]


class IndexResponse(BaseModel):
    indexed_count: int
    job_id: str


class CatalogItemFields(BaseModel):
    """A catalog item's editable metadata. Only name/category/price are
    required — caption/description/tags/material are free-form catalog copy
    an admin may not always fill in."""
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price: float = Field(gt=0)
    caption: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    material: Optional[str] = None


class IndexItemMetadata(CatalogItemFields):
    """One catalog item's metadata, positionally aligned with the `images`
    upload list."""
    item_id: str = Field(min_length=1)


class FailedItem(BaseModel):
    item_id: str
    error: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "done" | "failed"
    total: int
    processed: int
    failed_items: list[FailedItem]


class CatalogItemsResponse(BaseModel):
    items: list[dict]
    total: int
    limit: int
    offset: int


_CATEGORY_KEYWORDS = {
    "ring": ["ring", "rings"],
    "necklace": ["necklace", "choker", "pendant", "collar"],
    "earrings": ["earring", "earrings", "stud", "jhumka"],
    "bracelet": ["bracelet", "braclet", "bangle", "cuff", "kada"],
}


def infer_category(caption: str) -> str:
    """Keyword match over an auto-generated caption — a cheap stand-in for a
    real classifier, good enough for the handful of jewellery categories this
    catalog covers. Falls back to "other" rather than guessing wrong.

    Matches whole words only: a plain substring check would match "ring"
    inside "earrings" and miscategorize every pair of earrings as a ring.
    """
    words = set(re.findall(r"[a-z]+", caption.lower()))
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if words & set(keywords):
            return category
    return "other"


@app.get("/health")
def health():
    """Actually verifies Pinecone, Supabase, and (if configured) Redis are
    reachable, rather than a static {"status": "ok"} -- this is what a real
    uptime monitor should hit. Redis is reported "not_configured" rather
    than "down" when REDIS_URL is unset, since that's a deliberate dev/
    single-instance choice (see cache.py/job_store.py), not a failure."""
    checks = {}
    healthy = True

    try:
        vector_db.ping()
        checks["pinecone"] = "ok"
    except Exception:
        logger.exception("Health check: Pinecone unreachable.")
        checks["pinecone"] = "unreachable"
        healthy = False

    try:
        catalog_store.ping()
        checks["supabase"] = "ok"
    except Exception:
        logger.exception("Health check: Supabase unreachable.")
        checks["supabase"] = "unreachable"
        healthy = False

    if settings.redis_url:
        try:
            cache.ping()
            checks["redis"] = "ok"
        except Exception:
            logger.exception("Health check: Redis unreachable.")
            checks["redis"] = "unreachable"
            healthy = False
    else:
        checks["redis"] = "not_configured"

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code, content={"status": "ok" if healthy else "degraded", "checks": checks}
    )


def build_catalog_text_description(name: str, caption: str, description: str, tags: list[str]) -> str:
    """The text half of a fused catalog embedding (see
    embeddings.embed_catalog_item). Single source of truth for this string's
    shape -- also imported by scripts/reembed_catalog.py so a full-catalog
    re-embed produces text in exactly the same form new items get indexed
    with, not a subtly different one."""
    return f"name: {name}. {caption}. {description}. Tags: {', '.join(tags)}."


# Text-query cheap-scoring gap (reranker.score_candidates_cheap's
# blended_score, top result vs. 3rd place or the last available) below which
# results are treated as genuinely ambiguous and escalated to a real LLM
# judgment via reranker.rerank(). See main.py's search() and reranker.py's
# module docstring for the full conditional-rerank flow.
_TEXT_RERANK_GAP_THRESHOLD = 0.1


def _not_jewelry_response(query_id: str) -> SearchResponse:
    return SearchResponse(
        query_id=query_id,
        matches=[],
        no_match=True,
        query_type="image",
        reason="The uploaded photo doesn't appear to show a piece of jewellery.",
    )


def _matches_from_ranked(ranked: list[dict]) -> list[MatchResult]:
    return [
        MatchResult(
            id=m["id"],
            similarity_percent=round(m["score"] * 100, 1),
            confidence=m["confidence"],
            reason=m["reason"],
            metadata=m["metadata"],
        )
        for m in ranked
    ]


async def _search_image(
    image: UploadFile, metadata_filter: dict, query_id: str | None = None
) -> tuple[SearchResponse, dict]:
    """Image-query path: unchanged from before this consolidation pass except
    for the domain gate below -- score >= min_similarity_threshold filter,
    then always reranker.rerank() (no cheap-scoring equivalent exists for
    judging visual traits against a photo).

    Returns (response, log_fields) -- log_fields carries path_taken for the
    structured request-completed log line in search() (see logging_config.py).
    """
    query_id = query_id or str(uuid.uuid4())
    raw_bytes = await image.read()
    # Query representation for search_events (§14/Phase 6) -- a hash, not the
    # raw bytes, so a jsonb log table never holds actual image data.
    query_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        clean_bytes = prepare_image_bytes(raw_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Domain gate BEFORE the expensive embed -> search -> rerank pipeline --
    # both a correctness fix (an unrelated photo can still clear the cosine
    # threshold against SOME catalog item in a shared embedding space) and a
    # cost optimization (rejects obviously irrelevant uploads before spending
    # an embedding call, a Pinecone query, and a potential rerank call).
    if not reranker.is_plausibly_jewelry(clean_bytes):
        return _not_jewelry_response(query_id), {
            "path_taken": "domain_gate_rejected", "query_representation": query_hash, "candidates": [],
        }

    query_vector = embeddings.embed_image(clean_bytes)
    raw_matches = vector_db.search(query_vector, metadata_filter=metadata_filter or None)
    candidates_log = [{"id": m["id"], "score": m["score"]} for m in raw_matches]

    strong_matches = [m for m in raw_matches if m["score"] >= settings.min_similarity_threshold]
    if not strong_matches:
        response = SearchResponse(query_id=query_id, matches=[], no_match=True, query_type="image")
        return response, {
            "path_taken": "no_strong_match", "query_representation": query_hash, "candidates": candidates_log,
        }

    ranked = reranker.rerank({"type": "image", "bytes": clean_bytes}, strong_matches)
    response = SearchResponse(
        query_id=query_id, no_match=False, query_type="image", matches=_matches_from_ranked(ranked)
    )
    return response, {
        "path_taken": "image_llm_rerank", "query_representation": query_hash, "candidates": candidates_log,
    }


def _search_text(
    text: str, metadata_filter: dict, query_id: str | None = None
) -> tuple[SearchResponse, dict]:
    """Text-query path: no absolute cosine floor (Google's own guidance
    against a fixed cutoff for this model, and cross-modal scores run
    structurally lower than image-vs-image scores -- see config.py). Default
    ranking is score_candidates_cheap() (zero API calls); reranker.rerank()
    only fires when the cheap-scored top results are genuinely ambiguous.

    Returns (response, log_fields) -- log_fields carries cache_hit and
    path_taken for the structured request-completed log line in search()
    (see logging_config.py).
    """
    query_id = query_id or str(uuid.uuid4())

    key = cache.cache_key(text, metadata_filter)
    cached_vector = cache._cache.get(key)
    cache_hit = cached_vector is not None
    query_vector = cached_vector
    if query_vector is None:
        query_vector = embeddings.embed_text_query(text)
        cache._cache.set(key, query_vector)

    raw_matches = vector_db.search(query_vector, metadata_filter=metadata_filter or None)

    # no_match now only fires for an empty result set outright (empty
    # catalog, or a metadata filter matching nothing) -- NOT for low scores,
    # which is the exact bug that silently zeroed out every text search
    # before this fix (see README.md §11).
    if not raw_matches:
        response = SearchResponse(query_id=query_id, matches=[], no_match=True, query_type="text")
        return response, {
            "cache_hit": cache_hit, "path_taken": "empty_result_set", "query_representation": text, "candidates": [],
        }

    candidates = raw_matches[:settings.top_k]
    candidates_log = [{"id": m["id"], "score": m["score"]} for m in candidates]
    cheap_scored = reranker.score_candidates_cheap(text, candidates)

    if len(cheap_scored) >= 3:
        gap = cheap_scored[0]["blended_score"] - cheap_scored[2]["blended_score"]
    else:
        gap = cheap_scored[0]["blended_score"] - cheap_scored[-1]["blended_score"]

    if gap >= _TEXT_RERANK_GAP_THRESHOLD:
        logger.info("text search: cheap path used (gap=%.3f)", gap)
        ranked = cheap_scored
        path_taken = "cheap"
    else:
        logger.info("text search: LLM rerank triggered, gap=%.3f", gap)
        ranked = reranker.rerank({"type": "text", "text": text}, candidates)
        path_taken = "llm_rerank"

    response = SearchResponse(
        query_id=query_id, no_match=False, query_type="text", matches=_matches_from_ranked(ranked)
    )
    return response, {
        "cache_hit": cache_hit, "path_taken": path_taken, "query_representation": text, "candidates": candidates_log,
    }


@app.post("/api/v1/search", response_model=SearchResponse, dependencies=[])
async def search(
    image: Optional[UploadFile] = File(None),
    query_text: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    max_price: Optional[float] = Form(None),
    min_price: Optional[float] = Form(None),
    x_api_key: Optional[str] = Header(None),
):
    """
    Accepts EITHER an image OR a text query, never both -- the two paths
    share the same Pinecone index and the same embedding model, but diverge
    sharply downstream: image queries always use an absolute cosine floor
    plus a mandatory LLM rerank; text queries use rank-based cheap scoring by
    default and only escalate to an LLM rerank when results are ambiguous
    (see _search_image / _search_text).
    """
    client_name, tier = require_api_key(x_api_key)
    _enforce_rate_limit("search", client_name, tier)

    has_image = image is not None
    text = (query_text or "").strip()
    has_text = bool(text)

    if not has_image and not has_text:
        raise HTTPException(status_code=400, detail="Provide either an image or a text query.")
    if has_image and has_text:
        raise HTTPException(status_code=400, detail="Provide only one of image or text query, not both.")

    metadata_filter: dict = {}
    if category:
        metadata_filter["category"] = {"$eq": category}
    if min_price is not None or max_price is not None:
        price_filter = {}
        if min_price is not None:
            price_filter["$gte"] = min_price
        if max_price is not None:
            price_filter["$lte"] = max_price
        metadata_filter["price"] = price_filter

    query_id = str(uuid.uuid4())
    start = time.perf_counter()
    if has_image:
        response, log_fields = await _search_image(image, metadata_filter, query_id)
    else:
        response, log_fields = _search_text(text, metadata_filter, query_id)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    logger.info(
        "search request completed",
        extra={"structured_fields": {
            "request_id": query_id,
            "client_name": client_name,
            "query_type": response.query_type,
            "no_match": response.no_match,
            "result_count": len(response.matches),
            "latency_ms": latency_ms,
            **log_fields,
        }},
    )

    # Best-effort: a Supabase hiccup here must never break a real search --
    # this is telemetry for future training data (Phase 6), not core
    # functionality. See search_events.py.
    try:
        search_events.record_search_event(
            request_id=query_id,
            client_name=client_name,
            query_type=response.query_type,
            query_text_or_image_hash=log_fields.get("query_representation", ""),
            retrieved_candidates=log_fields.get("candidates", []),
            path_taken=log_fields.get("path_taken", ""),
            result_ids_returned_in_order=[m.id for m in response.matches],
            no_match=response.no_match,
        )
    except Exception:
        logger.exception("Failed to record search event for request_id=%s.", query_id)
        sentry_sdk.capture_exception()

    return response


@app.post("/api/v1/search/{query_id}/feedback", status_code=204)
def submit_search_feedback(
    query_id: str, feedback: FeedbackRequest, x_api_key: Optional[str] = Header(None)
):
    """
    Records client-reported feedback on a specific search result (clicked /
    purchased / dismissed) -- ground truth for the future ranking work
    search_events.py's data collection exists to support (§14/Phase 6).
    Frontend integration is a stretch goal, not required this phase; the
    endpoint and its table exist now so it's ready the moment that lands.
    """
    require_api_key(x_api_key)
    search_events.record_feedback(query_id, feedback.result_id, feedback.action)
    return Response(status_code=204)


async def _index_job(job_id: str, items_payload: list[dict]):
    """Background task: embed + upsert catalog items one at a time, updating
    the job_store entry as it goes so GET /api/v1/catalog/jobs/{job_id} can
    report live progress (see job_store.py -- Redis-backed when REDIS_URL is
    set, so progress is visible across instances/restarts, not just to the
    process that started the job).

    Each item is wrapped in its own try/except: one bad image (corrupt file,
    a transient embedding-API failure that exhausts retries, ...) is recorded
    as a failure for that item_id and the batch continues, rather than one
    bad item aborting everything else the admin uploaded alongside it.
    """
    job = job_store._job_store.get(job_id)
    for item in items_payload:
        meta: IndexItemMetadata = item["meta"]
        try:
            clean_bytes = prepare_image_bytes(item["raw_bytes"])
            text_description = build_catalog_text_description(
                meta.name, meta.caption, meta.description, meta.tags
            )
            vector = embeddings.embed_catalog_item(clean_bytes, text_description)

            image_url = object_storage.upload_catalog_image(meta.item_id, clean_bytes)

            metadata = {
                "filename": item["filename"],
                "name": meta.name,
                "caption": meta.caption,
                "description": meta.description,
                "tags": meta.tags,
                "category": meta.category,
                "price": meta.price,
                "image_url": image_url,
            }
            if meta.material:
                metadata["material"] = meta.material

            vector_db.upsert_batch([{"id": meta.item_id, "vector": vector, "metadata": metadata}])
            catalog_store.record_item(meta.item_id, metadata)
        except Exception as exc:
            logger.exception("Failed to index catalog item %s.", meta.item_id)
            sentry_sdk.capture_exception(exc)
            job["failed_items"].append({"item_id": meta.item_id, "error": str(exc)})
        finally:
            job["processed"] += 1
            job_store._job_store.set(job_id, job)

    job["status"] = "failed" if len(job["failed_items"]) == job["total"] else "done"
    job_store._job_store.set(job_id, job)


@app.post("/api/v1/catalog/index", response_model=IndexResponse)
async def index_catalog(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(...),
    items_json: str = Form(...),
    x_api_key: Optional[str] = Header(None),
):
    """
    Batch catalog upsert. Runs as a background task so bulk loads (thousands
    of items) don't block the request or time out; poll the returned job_id
    via GET /api/v1/catalog/jobs/{job_id} for progress/results.

    items_json is a JSON array of per-item metadata objects (see
    IndexItemMetadata), positionally aligned 1:1 with the `images` list.
    """
    client_name, tier = require_api_key(x_api_key)
    _enforce_rate_limit("index", client_name, tier)

    try:
        raw_items = json.loads(items_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"items_json is not valid JSON: {exc}")

    if not isinstance(raw_items, list):
        raise HTTPException(status_code=400, detail="items_json must be a JSON array.")
    if len(raw_items) != len(images):
        raise HTTPException(
            status_code=400,
            detail=(
                f"items_json has {len(raw_items)} entries but {len(images)} images were "
                "uploaded — they must match 1:1 in the same order."
            ),
        )

    parsed_items: list[IndexItemMetadata] = []
    for i, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise HTTPException(
                status_code=400, detail=f"items_json[{i}] must be an object, got {type(raw).__name__}."
            )
        try:
            parsed_items.append(IndexItemMetadata(**raw))
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={"message": f"items_json[{i}] failed validation.", "errors": exc.errors()},
            )

    items_payload = []
    for img, meta in zip(images, parsed_items):
        raw_bytes = await img.read()
        items_payload.append({"raw_bytes": raw_bytes, "filename": img.filename, "meta": meta})

    job_id = str(uuid.uuid4())
    job_store._job_store.set(
        job_id, {"status": "pending", "total": len(items_payload), "processed": 0, "failed_items": []}
    )
    background_tasks.add_task(_index_job, job_id, items_payload)

    return IndexResponse(indexed_count=len(items_payload), job_id=job_id)


@app.get("/api/v1/catalog/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    job = job_store._job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return JobStatus(job_id=job_id, **job)


@app.get("/api/v1/catalog/items", response_model=CatalogItemsResponse)
def list_catalog_items(limit: int = 20, offset: int = 0, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    items, total = catalog_store.list_items(limit=limit, offset=offset)
    return CatalogItemsResponse(items=items, total=total, limit=limit, offset=offset)


@app.get("/api/v1/catalog/items/{item_id}", response_model=dict)
def get_catalog_item(item_id: str, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    item = catalog_store.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unknown item_id.")
    return item


@app.patch("/api/v1/catalog/items/{item_id}", response_model=dict)
async def update_catalog_item(
    item_id: str,
    fields: str = Form(...),
    image: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
):
    """
    Edit an existing item's metadata, optionally replacing its image too.

    fields is a JSON object matching CatalogItemFields (name/category/price
    required). When no image is uploaded, only Supabase + Pinecone metadata
    are updated in place — no re-embedding, since the vector is unchanged.
    Uploading a new image re-embeds and overwrites the stored file, same as
    a fresh index.
    """
    require_api_key(x_api_key)

    existing = catalog_store.get_item(item_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Unknown item_id.")

    try:
        raw = json.loads(fields)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"fields is not valid JSON: {exc}")
    try:
        edited = CatalogItemFields(**raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": "fields failed validation.", "errors": exc.errors()},
        )

    metadata = {
        "filename": existing.get("filename"),
        "name": edited.name,
        "caption": edited.caption,
        "description": edited.description,
        "tags": edited.tags,
        "category": edited.category,
        "price": edited.price,
        "image_url": existing.get("image_url"),
    }
    if edited.material:
        metadata["material"] = edited.material

    if image is not None:
        raw_bytes = await image.read()
        try:
            clean_bytes = prepare_image_bytes(raw_bytes)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        text_description = build_catalog_text_description(
            edited.name, edited.caption, edited.description, edited.tags
        )
        vector = embeddings.embed_catalog_item(clean_bytes, text_description)
        metadata["filename"] = image.filename
        metadata["image_url"] = object_storage.upload_catalog_image(item_id, clean_bytes)
        vector_db.upsert_batch([{"id": item_id, "vector": vector, "metadata": metadata}])
    else:
        vector_db.update_metadata(item_id, metadata)

    catalog_store.record_item(item_id, metadata)
    return {"item_id": item_id, **metadata}


@app.delete("/api/v1/catalog/items/{item_id}", status_code=204)
def delete_catalog_item(item_id: str, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)

    existing = catalog_store.get_item(item_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Unknown item_id.")

    vector_db.delete_by_id(item_id)
    catalog_store.delete_item(item_id)
    object_storage.delete_catalog_image(item_id)

    return Response(status_code=204)
