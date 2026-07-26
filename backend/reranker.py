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
    pattern, silhouette) it can actually see. ALWAYS goes through this LLM
    rerank() path — there's no cheap-scoring equivalent for judging visual
    traits against a photo.
  - {"type": "text", "text": <query string>} — no image exists to send. This
    is now the LLM rerank() path's fallback only: the default text-query path
    is score_candidates_cheap() (zero API calls), and rerank() is called on
    text candidates only when main.py finds the cheap-scored results are
    ambiguous (see main.py's conditional-rerank logic). When it IS called for
    a text query, the LLM instead judges each candidate's METADATA
    (name/category/material/caption/description/tags) against the stated
    query terms (material, color, style words) — it is explicitly told not to
    invent visual details it has no way to observe from text alone.

Sorting: every code path that produces a final ranked list (rerank() and
score_candidates_cheap()) sorts by the SAME final_rank_score() function below,
not two independently-maintained sort implementations. Confidence is a
bounded tiebreaker, not a primary sort key — see final_rank_score's docstring
for why.
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

_JEWELLERY_GATE_PROMPT = (
    "Does this image show a piece of jewellery (ring, earring, necklace, "
    "bracelet, pendant, brooch, etc.) as the main subject? "
    "Answer with exactly one word: yes or no."
)

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
# would actually appear.
_IMAGE_METADATA_FIELDS = ("name", "category", "material")
_TEXT_METADATA_FIELDS = ("name", "category", "material", "caption", "description", "tags")

# Confidence is a BOUNDED tiebreaker, never a primary sort key. Sorting by
# (confidence_rank, -score) directly -- the previous scheme -- let a "medium"
# candidate with a much higher cosine score sink below several "high"
# candidates with lower scores (observed for real: a high-scoring result
# landing 6th). A flat additive weight instead means confidence can only ever
# nudge a close call, never override a large score gap: a 0.75 "medium"
# (0.75 + 0.0 = 0.75) still outranks a 0.55 "high" (0.55 + 0.05 = 0.60)... but
# a 0.60 "high" (0.65) does edge out a 0.58 "medium" (0.58) when scores are
# genuinely close. Cosine score is the real retrieval signal and should
# dominate; confidence is the LLM's read on it and should only refine.
CONFIDENCE_WEIGHT = {"high": 0.05, "medium": 0.0, "low": -0.05}


def final_rank_score(candidate: dict) -> float:
    """The one sort key used by every code path that produces a final ranked
    list (rerank() and score_candidates_cheap()) -- see module docstring."""
    return candidate["score"] + CONFIDENCE_WEIGHT.get(candidate.get("confidence"), 0.0)


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
    final_rank_score() -- cosine score remains the dominant ranking signal,
    confidence only a bounded tiebreaker (see final_rank_score docstring).
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

    enriched = []
    for c in candidates:
        judgment = judgments.get(c["id"], {"confidence": "medium", "reason": "no reranker judgment available"})
        enriched.append({
            **c,
            "confidence": judgment["confidence"],
            "reason": judgment["reason"],
        })

    enriched.sort(key=lambda c: -final_rank_score(c))
    return enriched


def score_candidates_cheap(query_text: str, candidates: list[dict]) -> list[dict]:
    """Blend cosine similarity with lexical overlap against stored metadata
    text. Zero external calls — this is the DEFAULT ranking path for text
    queries (see main.py: the LLM rerank() path only fires when this cheap
    score leaves the top results ambiguous)."""
    if not candidates:
        return []

    query_terms = set(query_text.lower().split())
    scored = []
    for c in candidates:
        meta_text = " ".join(str(v) for v in c["metadata"].values()).lower()
        overlap = sum(1 for t in query_terms if t in meta_text) / max(len(query_terms), 1)
        blended = 0.7 * c["score"] + 0.3 * overlap
        confidence = "high" if blended > 0.75 else "medium" if blended > 0.5 else "low"
        scored.append({
            **c,
            "confidence": confidence,
            "reason": "matched on similarity and description overlap",
            "blended_score": blended,
        })
    return sorted(scored, key=lambda c: -final_rank_score(c))


@external_api_retry
def is_plausibly_jewelry(image_bytes: bytes) -> bool:
    """Cheap one-word classification gate, run BEFORE the expensive
    embed -> search -> rerank pipeline for image queries. Reuses the existing
    fast reranker model, no new provider. This is NOT the same as rerank()'s
    judgment call: it runs earlier, on a single image, before any candidates
    even exist, and only answers "is this worth searching for at all" — it
    doubles as a cost optimization (rejects an obviously irrelevant upload
    before spending an embedding call + Pinecone query + potential rerank
    call on it).

    Heuristic, not a guarantee: a single fast classification call can produce
    both false negatives (real jewelry rejected, e.g. an unusually abstract
    or sculptural piece) and false positives (an ambiguous non-jewelry photo
    passing through) — see README.md §13. Does not apply to text queries:
    there's no image to classify, and a vague text query isn't the same
    failure mode as an entirely unrelated photo.
    """
    response = _client.models.generate_content(
        model=settings.reranker_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            _JEWELLERY_GATE_PROMPT,
        ],
        config=types.GenerateContentConfig(max_output_tokens=5),
    )
    return response.text.strip().lower().startswith("y")
