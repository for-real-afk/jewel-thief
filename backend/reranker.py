"""
RAG reranking.

Design choice: cosine similarity (a real, computed number) is the primary
score shown to the user, not an LLM-invented percentage. An LLM asked to
output "94.3% match" is producing a plausible-looking number it has no
calibrated way to compute. Instead, gemini-2.5-flash reviews the Top-K
candidates and returns a categorical confidence (high/medium/low) plus a
short rationale — grounded judgment, not a fake precise figure.

All K candidates are sent in a SINGLE batched call, not one call per
candidate — this keeps reranking latency and cost roughly constant
regardless of K.

Provider is selected via settings.llm_provider: "gemini" (default/production)
or "groq" (fast cloud inference — see MODEL_COMPARISON.md for measured
speed/accuracy against Gemini and the now-removed LM Studio option). Only the
raw-text-generation step differs between providers; JSON parsing, fallback,
and sorting are shared.

Two query shapes, since search accepts either an uploaded photo or a text
description (see main.py's /api/v1/search):
  - {"type": "image", "bytes": <jpeg bytes>} — the original path. The
    reference photo is sent as a multimodal Part alongside the prompt, and
    the LLM judges visual traits (gemstone cut, metal color/finish, band
    pattern, silhouette) it can actually see.
  - {"type": "text", "text": <query string>} — no image exists to send.
    The LLM instead judges each candidate's METADATA (name/category/
    material/caption/description/tags) against the stated query terms
    (material, color, style words) — it is explicitly told not to invent
    visual details it has no way to observe from text alone.
"""
import json
import re

from google import genai
from google.genai import types

import utils
from config import get_settings
from utils import external_api_retry

settings = get_settings()
_client = genai.Client(api_key=settings.gemini_api_key)

_IMAGE_PROMPT_TEMPLATE = """You are evaluating jewellery visual similarity for a search system.

You are shown exactly ONE image: the customer's reference photo. You do NOT get to see
images of the {n} candidates below — they were already retrieved by vector search and are
described only by their catalog ID, cosine similarity score, and metadata. Judge each
candidate's plausibility as a visual match to the reference image using that information,
reasoning about concrete visual traits: gemstone cut, metal color/finish, chain or band
pattern, and overall silhouette.

Respond with ONLY a JSON array — no thinking, no markdown fences, no preamble — one object
per candidate, in this exact form:
[{{"id": "<candidate id>", "confidence": "high"|"medium"|"low", "reason": "<one short phrase citing a specific visual trait>"}}]

Candidates:
{candidate_list}
"""

_TEXT_PROMPT_TEMPLATE = """You are evaluating jewellery search relevance for a search system.

The customer searched for: "{query_text}". There is no image — you do NOT get to see the
{n} candidates below or the item the customer has in mind, only their catalog metadata
(name, category, material, caption, description, tags). Judge how well each candidate's
STATED metadata matches the terms in the customer's description (material, color, gemstone,
style words, etc.). Reason only from what the metadata actually says — do not invent or
assume visual details it doesn't state.

Respond with ONLY a JSON array — no thinking, no markdown fences, no preamble — one object
per candidate, in this exact form:
[{{"id": "<candidate id>", "confidence": "high"|"medium"|"low", "reason": "<one short phrase citing which stated term(s) matched or didn't>"}}]

Candidates:
{candidate_list}
"""

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(raw_text: str) -> str:
    """Defensive cleanup for reasoning models that leak a <think>...</think>
    block into the response content regardless of provider-level flags."""
    return _THINK_BLOCK_RE.sub("", raw_text).strip()


# Image queries: the LLM judges what it can SEE, so only enough metadata to
# disambiguate (name/category/material) is worth the token cost — dumping the
# full dict here was a real contributor to a production Groq rate-limit
# error (see utils.py's _CHAT_MAX_TOKENS comment). Text queries: there is no
# image, so caption/description/tags are the primary signal, not filler —
# they're the only place material/color terms the customer searched for
# would actually appear. (Removing the image itself more than offsets this
# for total token budget — see the "hasn't been measured" caveat in
# embeddings.py/README.md though: real prompt sizes for rich catalog copy
# aren't verified against Groq's limit the way the image path now is.)
_IMAGE_METADATA_FIELDS = ("name", "category", "material")
_TEXT_METADATA_FIELDS = ("name", "category", "material", "caption", "description", "tags")


def _summarize_for_prompt(metadata: dict, fields: tuple) -> str:
    parts = []
    for key in fields:
        value = metadata.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(value)
        parts.append(f"{key}={value}")
    return f"metadata: {', '.join(parts)}" if parts else "metadata: none"


@external_api_retry
def _judge_gemini(query: dict, prompt: str) -> str:
    if query["type"] == "image":
        contents = [types.Part.from_bytes(data=query["bytes"], mime_type="image/jpeg"), prompt]
    else:
        contents = [prompt]
    response = _client.models.generate_content(model=settings.reranker_model, contents=contents)
    return response.text


def _judge_groq(query: dict, prompt: str) -> str:
    if query["type"] == "image":
        return utils.groq_vision_chat(query["bytes"], prompt, settings.groq_model)
    return utils.groq_text_chat(prompt, settings.groq_model)


def rerank(query: dict, candidates: list[dict]) -> list[dict]:
    """
    query: {"type": "image", "bytes": <jpeg bytes>} or {"type": "text", "text": <str>}
    candidates: [{"id": str, "score": float, "metadata": dict}, ...] from vector_db.search
    Returns candidates enriched with "confidence" and "reason", sorted by
    (confidence tier, cosine score) — cosine score remains the primary ranking signal.
    """
    if not candidates:
        return []

    is_text_query = query["type"] == "text"
    fields = _TEXT_METADATA_FIELDS if is_text_query else _IMAGE_METADATA_FIELDS
    candidate_list = "\n".join(
        f"- id: {c['id']}, cosine_similarity: {c['score']:.3f}, {_summarize_for_prompt(c['metadata'], fields)}"
        for c in candidates
    )

    if is_text_query:
        prompt = _TEXT_PROMPT_TEMPLATE.format(
            n=len(candidates), query_text=query["text"], candidate_list=candidate_list
        )
    else:
        prompt = _IMAGE_PROMPT_TEMPLATE.format(n=len(candidates), candidate_list=candidate_list)

    if settings.llm_provider == "groq":
        raw_text = _judge_groq(query, prompt)
    else:
        raw_text = _judge_gemini(query, prompt)

    try:
        judgments = {j["id"]: j for j in json.loads(_strip_reasoning(raw_text))}
    except (json.JSONDecodeError, KeyError, TypeError):
        # If the reranker output is malformed, fall back to cosine-only ranking
        # rather than failing the whole search.
        judgments = {}

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    enriched = []
    for c in candidates:
        judgment = judgments.get(c["id"], {"confidence": "medium", "reason": "no reranker judgment available"})
        enriched.append({
            **c,
            "confidence": judgment["confidence"],
            "reason": judgment["reason"],
        })

    enriched.sort(key=lambda c: (confidence_rank.get(c["confidence"], 1), -c["score"]))
    return enriched
