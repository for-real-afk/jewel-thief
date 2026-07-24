#!/usr/bin/env python
"""
Side-by-side comparison of reranker LLM providers (Gemini vs Groq) against
the exact same candidate set — same query image, same vector-search results,
same prompt. Only the judge model differs. Reports timing and how each
ranked results.

Does NOT change the app's default LLM_PROVIDER — settings.llm_provider is
overridden in-process just for the duration of each call.

Usage:
    python scripts/compare_rerankers.py "<path to query image>"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embeddings
import reranker
import vector_db
from config import get_settings
from preprocessing import prepare_image_bytes

settings = get_settings()

PROVIDERS_TO_COMPARE = ["gemini", "groq"]


def run_with_provider(provider: str, query_image_bytes: bytes, candidates: list[dict]) -> tuple[list[dict], float]:
    original = settings.llm_provider
    settings.llm_provider = provider
    try:
        t0 = time.time()
        ranked = reranker.rerank(query_image_bytes, candidates)
        elapsed = time.time() - t0
        return ranked, elapsed
    finally:
        settings.llm_provider = original


def print_ranking(provider: str, ranked: list[dict], elapsed: float) -> None:
    print(f"\n=== {provider} ({elapsed:.1f}s for {len(ranked)} candidates) ===")
    header = f"{'rank':<5}{'id':<20}{'cosine':>8}{'confidence':>12}  reason"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked, 1):
        print(f"{i:<5}{r['id']:<20}{r['score']:>8.3f}{r['confidence']:>12}  {r['reason']}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_rerankers.py <query image path>", file=sys.stderr)
        return 1

    query_path = Path(sys.argv[1])
    raw = query_path.read_bytes()
    clean = prepare_image_bytes(raw)

    print("Embedding query image...", flush=True)
    query_vector = embeddings.embed_query_image(clean)

    print("Searching catalog...", flush=True)
    raw_matches = vector_db.search(query_vector, metadata_filter=None)
    candidates = [m for m in raw_matches if m["score"] >= settings.min_similarity_threshold]
    print(f"{len(candidates)} candidate(s) above threshold ({settings.min_similarity_threshold}).")

    results = {}
    for provider in PROVIDERS_TO_COMPARE:
        print(f"\nRunning reranker via {provider}...", flush=True)
        try:
            ranked, elapsed = run_with_provider(provider, clean, candidates)
        except Exception as exc:
            print(f"  -> {provider} FAILED: {exc!r}")
            continue
        results[provider] = (ranked, elapsed)
        print_ranking(provider, ranked, elapsed)

    if len(results) == 2:
        print("\n=== Top pick comparison ===")
        for provider, (ranked, elapsed) in results.items():
            top = ranked[0]
            print(f"{provider:<10} -> #1: {top['id']} ({top['confidence']}, cosine {top['score']:.3f}) in {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
