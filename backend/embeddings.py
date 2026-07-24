"""
Multimodal embedding generation via gemini-embedding-2.

gemini-embedding-2 is a natively multimodal preview model (text, image, video,
audio, documents in one vector space). Its request shape may differ from the
older text-only gemini-embedding-001 API (e.g. the RETRIEVAL_QUERY /
RETRIEVAL_DOCUMENT task_type convention was designed for text embeddings) —
verify the current API reference for this model before relying on task_type
being interpreted the same way for image inputs.

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


@external_api_retry
def embed_image(image_bytes: bytes, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a single image into the shared multimodal vector space."""
    response = _client.models.embed_content(
        model=settings.embedding_model,
        contents=types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        config=types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions,
            task_type=task_type,
        ),
    )
    return response.embeddings[0].values


def embed_query_image(image_bytes: bytes) -> list[float]:
    """Convenience wrapper for a search-query image (as opposed to a catalog item)."""
    return embed_image(image_bytes, task_type="RETRIEVAL_QUERY")


def embed_catalog_image(image_bytes: bytes) -> list[float]:
    """Convenience wrapper for a catalog item being indexed."""
    return embed_image(image_bytes, task_type="RETRIEVAL_DOCUMENT")


def embed_batch_catalog_images(images: list[bytes]) -> list[list[float]]:
    """Embed multiple catalog images. Sequential for clarity; parallelize with
    asyncio/gather behind a semaphore if indexing volume demands it."""
    return [embed_catalog_image(img) for img in images]


@external_api_retry
def caption_image(image_bytes: bytes) -> str:
    """Short factual description of an image — used to auto-fill catalog
    metadata (name/category) when the caller doesn't supply one."""
    response = _client.models.generate_content(
        model=settings.reranker_model,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), _CAPTION_PROMPT],
    )
    return response.text
