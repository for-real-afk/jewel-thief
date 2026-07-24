"""
Shared pytest fixtures.

Environment variables MUST be set before any project module is imported:
config.py's Settings fields default to `os.getenv(...)` expressions evaluated
once at class-definition time (i.e. at `import config`), not lazily per
instantiation. pytest always fully executes conftest.py before it imports any
test module, so setting them here at module scope guarantees they're in place
before `config`, `embeddings`, `vector_db`, `reranker`, or `main` are ever
imported anywhere in the suite.
"""
import io
import os

os.environ.setdefault("GEMINI_API_KEY", "fake-gemini-key")
# Pin the reranker provider to "gemini" regardless of what a developer's
# local .env has set (e.g. "groq") — the suite must stay hermetic and never
# depend on a real external service being reachable.
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("EMBEDDING_MODEL", "gemini-embedding-2")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "8")
os.environ.setdefault("RERANKER_MODEL", "gemini-2.5-flash")
os.environ.setdefault("PINECONE_API_KEY", "fake-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-jewellery-catalog")
os.environ.setdefault("PINECONE_CLOUD", "aws")
os.environ.setdefault("PINECONE_REGION", "us-east-1")
os.environ.setdefault("TOP_K", "20")
os.environ.setdefault("MIN_SIMILARITY_THRESHOLD", "0.55")
os.environ.setdefault("MIN_SIMILARITY_THRESHOLD_TEXT", "0.35")
os.environ.setdefault("APP_API_KEY", "test-api-key")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")

import pytest
from PIL import Image


def make_image_bytes(size=(200, 200), color=(255, 0, 0), fmt="JPEG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def image_bytes_factory():
    """Factory fixture so tests can build images with whatever shape/format they need."""
    return make_image_bytes


@pytest.fixture
def valid_jpeg_bytes():
    return make_image_bytes(fmt="JPEG")


@pytest.fixture
def valid_png_bytes():
    return make_image_bytes(fmt="PNG")


@pytest.fixture
def valid_webp_bytes():
    return make_image_bytes(fmt="WEBP")


@pytest.fixture
def corrupt_image_bytes():
    return b"not a real image, just some garbage bytes 0123456789"


@pytest.fixture
def no_sleep(mocker):
    """Skip tenacity's real backoff delay so retry tests run instantly."""
    return mocker.patch("time.sleep", return_value=None)
