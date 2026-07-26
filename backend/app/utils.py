"""Shared utilities: retry/backoff for flaky external APIs (Gemini, Pinecone, Groq)."""
import base64
import io

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_settings

external_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)

settings = get_settings()

# Real production incident: Groq's free/on_demand tier enforces an 8000
# tokens-PER-REQUEST-MINUTE budget, and its rate limiter appears to count
# prompt_tokens + max_tokens (the theoretical worst case), not actual usage
# — a request measured at "Requested 8117" with max_tokens=4096 implies the
# prompt itself cost ~4021 tokens, and max_tokens=4096 alone pushed the
# *requested* total over the 8000 cap even though real usage was far lower.
# A ~20-item {id, confidence, reason} JSON array response realistically
# needs a few hundred tokens; 2048 leaves generous headroom without
# needlessly inflating the requested-token count that gates the limit.
# Shared by both the vision (image attached) and text-only chat calls below
# — it bounds the response, which has the same shape either way.
_CHAT_MAX_TOKENS = 2048

# preprocessing.py caps catalog/query images at 1024x1024 for embedding
# quality, but a vision-chat *judge* call doesn't need that much resolution
# to compare gemstone cut/metal color/silhouette — and the base64-encoded
# image is itself a meaningful chunk of the token budget above. Downscale
# specifically for this HTTP path; the stored catalog image and the Gemini
# embedding call are untouched.
_VISION_CHAT_MAX_DIMENSION = 384
_VISION_CHAT_JPEG_QUALITY = 75


def _downscale_for_vision_api(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((_VISION_CHAT_MAX_DIMENSION, _VISION_CHAT_MAX_DIMENSION), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=_VISION_CHAT_JPEG_QUALITY)
    return buf.getvalue()


def _openai_compatible_chat(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    image_bytes: bytes | None = None,
    headers: dict | None = None,
    extra_params: dict | None = None,
) -> str:
    """Chat against an OpenAI-compatible /chat/completions endpoint, with or
    without an attached image (text-only when image_bytes is None — the
    reranker's text-query path has no image to send). Only Groq uses this
    now, but kept as a named function separate from groq_vision_chat/
    groq_text_chat since it's a generically useful request shape, not a
    Groq-specific one."""
    if image_bytes is not None:
        b64 = base64.b64encode(_downscale_for_vision_api(image_bytes)).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
    else:
        content = prompt

    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": _CHAT_MAX_TOKENS,
    }
    if extra_params:
        body.update(extra_params)

    response = requests.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=timeout)
    if not response.ok:
        # raise_for_status()'s default message doesn't include the response
        # body — often the only place the *actual* rejection reason lives
        # (e.g. Groq's 413 body names the exact size limit hit).
        raise requests.exceptions.HTTPError(
            f"{response.status_code} error from {base_url}: {response.text[:500]}", response=response
        )
    return response.json()["choices"][0]["message"]["content"]


@external_api_retry
def groq_vision_chat(image_bytes: bytes, prompt: str, model: str) -> str:
    """Single-turn vision chat against Groq's cloud API (OpenAI-compatible). No
    embeddings endpoint exists on Groq — this is reranker-judge use only.

    reasoning_effort="none": Groq's reasoning models (this account's vision
    model, qwen/qwen3.6-27b, is one) emit an extended <think>...</think>
    block by default. Left on, that block can consume the whole token budget
    before ever producing the requested JSON, and even when it doesn't, it
    has to be stripped before parsing. Verified this param is accepted by
    the current model rather than assumed.
    """
    return _openai_compatible_chat(
        settings.groq_base_url,
        model,
        prompt,
        settings.groq_chat_timeout_seconds,
        image_bytes=image_bytes,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        extra_params={"reasoning_effort": "none"},
    )


@external_api_retry
def groq_text_chat(prompt: str, model: str) -> str:
    """Text-only chat against Groq's cloud API (OpenAI-compatible) — the
    reranker's judge call for a text search query, where there's no query
    image to send alongside the prompt. See groq_vision_chat's docstring for
    why reasoning_effort="none" is required."""
    return _openai_compatible_chat(
        settings.groq_base_url,
        model,
        prompt,
        settings.groq_chat_timeout_seconds,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        extra_params={"reasoning_effort": "none"},
    )
