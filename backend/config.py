"""
Environment / configuration management.

All secrets are read from environment variables (see .env.example).
Never hardcode API keys — this module is the single source of truth
for configuration so the rest of the app never touches os.environ directly.
"""
import os
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    # -- Google Gemini --
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "gemini-2.5-flash")

    # -- Provider switch --
    # llm_provider (reranker judge only): "gemini" or "groq". Embedding is
    # always Gemini — it's the only provider here with an image-embedding
    # endpoint (Groq has none). LM Studio support was removed after
    # evaluation; see MODEL_COMPARISON.md for why.
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")

    # -- Groq (OpenAI-compatible cloud inference) — reranker judge only, no
    # embeddings endpoint. Cloud inference is fast, so a much shorter timeout
    # than LM Studio's is reasonable even for a full top_k batch. --
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    groq_chat_timeout_seconds: int = int(os.getenv("GROQ_CHAT_TIMEOUT_SECONDS", "60"))

    # -- Pinecone --
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "jewellery-catalog")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # -- Supabase (catalog metadata table) --
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # -- Search behavior --
    top_k: int = int(os.getenv("TOP_K", "20"))
    # Applies to IMAGE queries only. Text queries do NOT use an absolute
    # cosine floor (see main.py's search()) -- per Google's own guidance
    # against a fixed cutoff, and because cross-modal (text-vs-image) cosine
    # scores run structurally lower than same-modality scores even for
    # genuinely relevant matches (a real, measured production incident: every
    # text query was returning no_match against a single shared threshold).
    # Text-query filtering instead happens via
    # reranker.score_candidates_cheap()'s blended score plus the conditional
    # LLM-rerank gap check in main.py.
    min_similarity_threshold: float = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.55"))

    # -- API auth --
    api_key: str = os.getenv("APP_API_KEY", "change-me-in-production")

    # -- Caching (see cache.py) --
    # Empty string = use InMemoryCache (single-instance stopgap). Set once
    # Redis is available in deployment -- cache.py's RedisCache (drop-in,
    # same SearchCache interface) is the swap point; no other code changes.
    redis_url: str = os.getenv("REDIS_URL", "")

    # -- CORS --
    allowed_origins: list[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    # Optional regex for origins that vary per-deploy (e.g. Vercel preview URLs
    # like https://jewel-thief-git-<branch>-<team>.vercel.app), so every new
    # preview doesn't require manually editing ALLOWED_ORIGINS. Example:
    # ALLOWED_ORIGIN_REGEX=https://jewel-thief.*\.vercel\.app
    allowed_origin_regex: str | None = os.getenv("ALLOWED_ORIGIN_REGEX") or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
