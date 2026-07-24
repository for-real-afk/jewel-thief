"""
RAG reranking.

Design choice: cosine similarity (a real, computed number) is the primary
score shown to the user, not an LLM-invented percentage. An LLM asked to
output "94.3% match" is producing a plausible-looking number it has no
calibrated way to compute. Instead, gemini-2.5-flash reviews the Top-K
candidates and returns a categorical confidence (high/medium/low) plus a
short rationale on concrete visual traits — grounded judgment, not a fake
precise figure.

All K candidates are sent in a SINGLE batched call, not one call per
candidate — this keeps reranking latency and cost roughly constant
regardless of K.

Provider is selected via settings.llm_provider: "gemini" (default/production)
or "groq" (fast cloud inference — see MODEL_COMPARISON.md for measured
speed/accuracy against Gemini and the now-removed LM Studio option). Only the
raw-text-generation step differs between providers; JSON parsing, fallback,
and sorting are shared.
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

_PROMPT_TEMPLATE = """You are evaluating jewellery visual similarity for a search system.

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

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(raw_text: str) -> str:
    """Defensive cleanup for reasoning models that leak a <think>...</think>
    block into the response content regardless of provider-level flags."""
    return _THINK_BLOCK_RE.sub("", raw_text).strip()


@external_api_retry
def _judge_gemini(query_image_bytes: bytes, prompt: str) -> str:
    response = _client.models.generate_content(
        model=settings.reranker_model,
        contents=[
            types.Part.from_bytes(data=query_image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    return response.text


def _judge_groq(query_image_bytes: bytes, prompt: str) -> str:
    return utils.groq_vision_chat(query_image_bytes, prompt, settings.groq_model)


def rerank(
    query_image_bytes: bytes,
    candidates: list[dict],
) -> list[dict]:
    """
    candidates: [{"id": str, "score": float, "metadata": dict}, ...] from vector_db.search
    Returns candidates enriched with "confidence" and "reason", sorted by
    (confidence tier, cosine score) — cosine score remains the primary ranking signal.
    """
    if not candidates:
        return []

    candidate_list = "\n".join(
        f"- id: {c['id']}, cosine_similarity: {c['score']:.3f}, "
        f"metadata: {c['metadata']}"
        for c in candidates
    )
    prompt = _PROMPT_TEMPLATE.format(n=len(candidates), candidate_list=candidate_list)

    if settings.llm_provider == "groq":
        raw_text = _judge_groq(query_image_bytes, prompt)
    else:
        raw_text = _judge_gemini(query_image_bytes, prompt)

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
