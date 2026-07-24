"""
Multimodal Visual Jewellery Search Engine — FastAPI backend.

Endpoints:
  POST /api/v1/search               — upload a reference image, get ranked matches
  POST /api/v1/catalog/index        — batch-upsert catalog items (images + items_json)
  GET  /api/v1/catalog/jobs/{job_id} — indexing job progress/result
  GET  /api/v1/catalog/items        — paginated list of indexed catalog items
  GET  /health                      — healthcheck
"""
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

import catalog_store
import embeddings
import vector_db
import reranker
from config import get_settings
from preprocessing import prepare_image_bytes, InvalidImageError

settings = get_settings()
logger = logging.getLogger("jewellery_search")
app = FastAPI(title="Jewellery Visual Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Search results need something to actually show the user — catalog images
# are persisted here at indexing time and served back as image_url metadata,
# rather than only ever existing transiently as an upload the pipeline embeds
# and discards.
STATIC_DIR = Path(__file__).resolve().parent / "static"
CATALOG_IMAGE_DIR = STATIC_DIR / "catalog"
CATALOG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def require_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


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


class IndexResponse(BaseModel):
    indexed_count: int
    job_id: str


class IndexItemMetadata(BaseModel):
    """One catalog item's metadata, positionally aligned with the `images`
    upload list. Only name/category/price are required — caption/description/
    tags/material are free-form catalog copy an admin may not always fill in."""
    item_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price: float = Field(gt=0)
    caption: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    material: Optional[str] = None


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
    return {"status": "ok"}


@app.post("/api/v1/search", response_model=SearchResponse, dependencies=[])
async def search(
    image: UploadFile = File(...),
    category: Optional[str] = Form(None),
    max_price: Optional[float] = Form(None),
    min_price: Optional[float] = Form(None),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)

    raw_bytes = await image.read()
    try:
        clean_bytes = prepare_image_bytes(raw_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    query_vector = embeddings.embed_query_image(clean_bytes)

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

    raw_matches = vector_db.search(query_vector, metadata_filter=metadata_filter or None)

    # Apply similarity threshold before spending a reranker call on weak candidates.
    strong_matches = [m for m in raw_matches if m["score"] >= settings.min_similarity_threshold]

    if not strong_matches:
        return SearchResponse(query_id=str(uuid.uuid4()), matches=[], no_match=True)

    ranked = reranker.rerank(clean_bytes, strong_matches)

    return SearchResponse(
        query_id=str(uuid.uuid4()),
        no_match=False,
        matches=[
            MatchResult(
                id=m["id"],
                similarity_percent=round(m["score"] * 100, 1),
                confidence=m["confidence"],
                reason=m["reason"],
                metadata=m["metadata"],
            )
            for m in ranked
        ],
    )


# job_id -> {"status", "total", "processed", "failed_items"}. In-memory and
# process-local — fine for a single dev/admin backend instance, but move this
# to Redis/a DB before running more than one worker process, since a second
# instance (or a restart) won't see jobs recorded by another.
_jobs: dict[str, dict] = {}


async def _index_job(job_id: str, items_payload: list[dict]):
    """Background task: embed + upsert catalog items one at a time, updating
    _jobs[job_id] as it goes so GET /api/v1/catalog/jobs/{job_id} can report
    live progress.

    Each item is wrapped in its own try/except: one bad image (corrupt file,
    a transient embedding-API failure that exhausts retries, ...) is recorded
    as a failure for that item_id and the batch continues, rather than one
    bad item aborting everything else the admin uploaded alongside it.
    """
    job = _jobs[job_id]
    for item in items_payload:
        meta: IndexItemMetadata = item["meta"]
        try:
            clean_bytes = prepare_image_bytes(item["raw_bytes"])
            vector = embeddings.embed_catalog_image(clean_bytes)

            image_path = CATALOG_IMAGE_DIR / f"{meta.item_id}.jpg"
            image_path.write_bytes(clean_bytes)

            metadata = {
                "filename": item["filename"],
                "name": meta.name,
                "caption": meta.caption,
                "description": meta.description,
                "tags": meta.tags,
                "category": meta.category,
                "price": meta.price,
                "image_url": f"/static/catalog/{meta.item_id}.jpg",
            }
            if meta.material:
                metadata["material"] = meta.material

            vector_db.upsert_batch([{"id": meta.item_id, "vector": vector, "metadata": metadata}])
            catalog_store.record_item(meta.item_id, metadata)
        except Exception as exc:
            logger.exception("Failed to index catalog item %s.", meta.item_id)
            job["failed_items"].append({"item_id": meta.item_id, "error": str(exc)})
        finally:
            job["processed"] += 1

    job["status"] = "failed" if len(job["failed_items"]) == job["total"] else "done"


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
    require_api_key(x_api_key)

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
    _jobs[job_id] = {"status": "pending", "total": len(items_payload), "processed": 0, "failed_items": []}
    background_tasks.add_task(_index_job, job_id, items_payload)

    return IndexResponse(indexed_count=len(items_payload), job_id=job_id)


@app.get("/api/v1/catalog/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    job = _jobs.get(job_id)
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
