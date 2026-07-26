"""
Multimodal embedding generation via gemini-embedding-2.

gemini-embedding-2 is a natively multimodal preview model (text, image, video,
audio, documents in one vector space). IMPORTANT: task_type is NOT supported
by this model and is silently ignored if passed — confirmed against the
official Google docs and a filed llama_index bug report. (An earlier version
of this module passed task_type="RETRIEVAL_QUERY"/"RETRIEVAL_DOCUMENT" the
way the older text-only gemini-embedding-001 API expects; that parameter was
doing nothing.) Since there's no task_type to lean on for asymmetric
query/document encoding, a TEXT query instead gets an explicit task
instruction baked directly into the input string ("task: search result |
query: ..." — see embed_text_query). Images have no text to prefix onto and
are embedded with no task hint at all.

Catalog items are embedded with TEXT FUSION: embed_catalog_item() sends the
image bytes and a text description (name/caption/description/tags) in one
interleaved multimodal call, producing a single vector that captures both
pixel content and stated metadata. This exists specifically to raise
cross-modal (text-query vs. catalog-item) cosine scores — a catalog vector
built from pixels alone has nothing of a text query's own modality to align
with beyond whatever the model infers visually, which is the structural
reason text search cosine scores ran lower than image search's ever could
(see README.md §11 for the measured gap before this fix).

Gemini is the only embedding provider: it's the only one of this project's
providers with an image-embedding endpoint (Groq has none). An LM Studio
caption-then-embed fallback was evaluated and removed after comparison — see
MODEL_COMPARISON.md for the measured accuracy/determinism cost of that
workaround relative to embedding pixels directly.
"""
from google import genai
from google.genai import types

from config import get_settings
from utils import external_api_retry

settings = get_settings()
_client = genai.Client(api_key=settings.gemini_api_key)

_CAPTION_PROMPT = (
    "Describe this jewellery item in one factual sentence, covering gemstone cut, "
    "metal color/finish, chain or band pattern, and overall silhouette. No opinions."
)

_TEXT_QUERY_TASK_PREFIX = "task: search result | query: "


@external_api_retry
def embed_image(image_bytes: bytes) -> list[float]:
    """Embed a single query image — used for query-time image search only.
    No task_type parameter: unsupported by gemini-embedding-2 (see module
    docstring), so none is sent."""
    response = _client.models.embed_content(
        model=settings.embedding_model,
        contents=types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
    )
    return response.embeddings[0].values


@external_api_retry
def embed_text_query(text: str) -> list[float]:
    """Embed a natural-language search query. Since gemini-embedding-2 has no
    task_type parameter, the query/document asymmetry is instead encoded as a
    prompt prefix on the input string itself — the convention Google's docs
    describe for this model. No task_type parameter is sent."""
    prefixed = f"{_TEXT_QUERY_TASK_PREFIX}{text}"
    response = _client.models.embed_content(
        model=settings.embedding_model,
        contents=prefixed,
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
    )
    return response.embeddings[0].values


@external_api_retry
def embed_catalog_item(image_bytes: bytes, text_description: str) -> list[float]:
    """THE catalog indexing function. Sends the image and a text description
    of the item in one interleaved multimodal call, producing a single fused
    vector per item — not two vectors, not an average of two separate calls.
    Callers build text_description from the item's stored metadata, e.g.
    f"name: {name}. {caption}. {description}. Tags: {', '.join(tags)}." — see
    main.py's _index_job and scripts/reembed_catalog.py for the exact shape
    used against real catalog metadata. No task_type parameter is sent."""
    response = _client.models.embed_content(
        model=settings.embedding_model,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), text_description],
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
    )
    return response.embeddings[0].values


@external_api_retry
def caption_image(image_bytes: bytes) -> str:
    """Short factual description of an image — used to auto-fill catalog
    metadata (name/category) when the caller doesn't supply one."""
    response = _client.models.generate_content(
        model=settings.reranker_model,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), _CAPTION_PROMPT],
    )
    return response.text
