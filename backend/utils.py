"""Shared utilities: retry/backoff for flaky external APIs (Gemini, Pinecone, Groq)."""
import base64

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_settings

external_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)

settings = get_settings()

# Generous enough for a full top_k batch of {id, confidence, reason} JSON
# objects, with headroom to spare — not meant to bound reasoning tokens.
_VISION_CHAT_MAX_TOKENS = 4096


def _openai_compatible_vision_chat(
    base_url: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    timeout: int,
    headers: dict | None = None,
    extra_params: dict | None = None,
) -> str:
    """Single-turn vision chat against an OpenAI-compatible /chat/completions
    endpoint. Only Groq uses this now, but kept as a named function separate
    from groq_vision_chat since it's a generically useful request shape, not
    a Groq-specific one."""
    b64 = base64.b64encode(image_bytes).decode()
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0.2,
        "max_tokens": _VISION_CHAT_MAX_TOKENS,
    }
    if extra_params:
        body.update(extra_params)

    response = requests.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=timeout)
    response.raise_for_status()
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
    return _openai_compatible_vision_chat(
        settings.groq_base_url,
        model,
        image_bytes,
        prompt,
        settings.groq_chat_timeout_seconds,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        extra_params={"reasoning_effort": "none"},
    )
