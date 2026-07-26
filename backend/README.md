# Facet — Jewellery Visual Search

A multimodal visual search engine for a jewellery catalog: upload a photo, get back
visually similar catalog items ranked by real vector similarity and refined by an LLM's
categorical judgment. Plus a catalog management page for bulk-adding inventory.

This document covers the full architecture — how the pieces connect, why the vector
space is built the way it is, and how indexing/retrieval/reranking actually work.
For a chronological list of every real bug hit while building this and how it was
fixed, see **[ISSUES.md](../ISSUES.md)**. For measured model-vs-model performance data,
see **[MODEL_COMPARISON.md](../MODEL_COMPARISON.md)**.

---

## 1. Architecture overview

```text
┌─────────────────┐         ┌──────────────────────────────────────────┐
│   Frontend        │         │              FastAPI backend              │
│   (React + Vite)   │         │                (main.py)                  │
│                    │         │                                          │
│  /        chat search │────▶│ POST /api/v1/search                       │
│  /catalog admin page  │────▶│ POST /api/v1/catalog/index                │
│                    │         │ GET  /api/v1/catalog/jobs/{job_id}        │
│                    │◀────────│ GET  /api/v1/catalog/items                 │
│                    │         │ GET  /static/catalog/{item_id}.jpg        │
└─────────────────┘         └───────────────┬──────────────────────────┘
                                             │
                    ┌────────────────────────┼──────────────────────────┐
                    ▼                        ▼                          ▼
        ┌───────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
        │  preprocessing.py    │   │    embeddings.py      │   │      reranker.py         │
        │  (Pillow)             │   │                        │   │                          │
        │  normalize any        │   │  always Gemini —       │   │  provider-switched:      │
        │  upload before it     │   │  the only provider     │   │  Gemini or Groq          │
        │  ever reaches a model │   │  with an image-embed   │   │                          │
        └───────────────────┘   │  endpoint              │   └──────────────────────┘
                                    └──────────┬─────────┘
                                               │ 768-dim vector
                                               ▼
                                    ┌────────────────────┐
                                    │     vector_db.py      │
                                    │   Pinecone (serverless)│
                                    │  cosine metric, 768-dim │
                                    └────────────────────┘

                    ┌────────────────────────┐   ┌──────────────────────────┐
                    │      catalog_store.py     │   │   static/catalog/*.jpg     │
                    │  local JSON, mirrors what  │   │  persisted catalog photos, │
                    │  is in Pinecone, for the    │   │  served back as image_url   │
                    │  admin "Recently added" UI  │   │  in every search result     │
                    └────────────────────────┘   └──────────────────────────┘
```

**One provider switch**, set via `.env`:

- `LLM_PROVIDER` — `gemini` or `groq`. Affects only the reranker's judgment call.
  Embedding is always Gemini — it's the only provider here with an image-embedding
  endpoint (Groq has none).

An LM Studio (local, no-API-key) path was built and evaluated for both embedding
(caption-then-embed workaround) and reranking, then removed after comparison: Groq was
~48x faster and measurably more accurate on every real test run. See
[MODEL_COMPARISON.md](../MODEL_COMPARISON.md) for the actual numbers and
[ISSUES.md](../ISSUES.md) for what broke along the way — both are kept as a record even
though the code itself is gone.

---

## 2. Repository layout

```text
backend/
  app/
    config.py         Settings — every env var, one place, loaded via python-dotenv
    preprocessing.py   Image normalization (the "chunking" analog — see §4)
    embeddings.py      Image -> vector via Gemini (gemini-embedding-2)
    vector_db.py       Pinecone client: index creation, upsert, ANN search
    reranker.py        LLM-judged reranking, provider-switched (Gemini / Groq)
    utils.py           Shared retry decorator + Groq's OpenAI-compatible HTTP helper
    catalog_store.py   Supabase-backed index of catalog items, for admin listing (§7)
    job_store.py       Indexing job status -- Redis-backed when REDIS_URL is set (§7, §14)
    api_keys.py        Per-client API key issuance/lookup/revocation, Supabase-backed (§14)
    rate_limit.py       Redis-backed, tier-aware rate limiting per endpoint (§14)
    logging_config.py  Structured (JSON) logging setup (§14)
    object_storage.py  Cloudflare R2 (S3-compatible) catalog image storage (§14)
    search_events.py   Search-event + feedback logging, Supabase-backed (§14)
    main.py            FastAPI app: routes, request validation, job tracking, static files
    static/catalog/    Persisted catalog photos, served at /static/catalog/{item_id}.jpg
  tests/             pytest suite — every external call mocked, no real API keys needed
  scripts/           One-off/utility scripts (see §9)
frontend/
  src/pages/SearchPage.jsx      Chat search page ("/")
  src/pages/CatalogPage.jsx     Catalog admin page ("/catalog")
  src/components/ShimmerLine.jsx  Shared loading-state indicator (used by both pages)
  src/TopNav.jsx        Shared nav between the two routes
  src/theme.css        Shared design tokens (see note in §8)
  src/main.jsx         react-router-dom route table
```

---

## 3. Data flow

### 3.1 Indexing (adding a catalog item)

```text
Admin uploads image + metadata (via /catalog page or directly to the API)
  │
  ▼
POST /api/v1/catalog/index
  images: list[UploadFile]
  items_json: JSON array, one {item_id, name, category, price, caption,
                               description, tags, material} per image,
                               positionally aligned with `images`
  │
  ▼
main.py validates items_json against IndexItemMetadata (Pydantic) —
  400 with field-level errors on any mismatch, BEFORE any background work starts
  │
  ▼
Returns {indexed_count, job_id} immediately; actual work happens in a
BackgroundTask (_index_job) so a bulk upload doesn't time out the request
  │
  ▼
For EACH item independently (one bad image doesn't abort the rest):
  1. preprocessing.prepare_image_bytes()      — normalize (§4)
  2. main.build_catalog_text_description()    — "name: {name}. {caption}.
                                                 {description}. Tags: {tags}."
  3. embeddings.embed_catalog_item(bytes, text) — ONE fused vector: image
                                                 bytes + text description sent
                                                 in a single interleaved call
                                                 (§4.1) -- not two vectors, not
                                                 an average of two calls
  4. write normalized JPEG to static/catalog/{item_id}.jpg
  5. vector_db.upsert_batch([{id, vector, metadata}])  — write to Pinecone
  6. catalog_store.record_item(item_id, metadata)      — mirror to Supabase (§7)
  7. update _jobs[job_id] progress (processed count, or failed_items entry)
  │
  ▼
Admin polls GET /api/v1/catalog/jobs/{job_id} every 2s until status is
"done" (partial failures still count as done) or "failed" (every item failed)
```

`scripts/reembed_catalog.py` re-runs steps 1-3+5 above against every already-indexed
item (metadata from Supabase, image from `static/catalog/`) — see the scripts table in
§9 for when this is needed and why it's destructive.

### 3.2 Search (querying with a photo or a text description)

`POST /api/v1/search` accepts exactly one of two inputs — a multipart `image` file, or
a `query_text` form field — never both, never neither (both return `400`, see the exact
messages in `main.py::search`). Whichever one is provided decides everything downstream
via a `query_type: "image" | "text"` field echoed back on `SearchResponse`, but **both
paths converge on the same Pinecone index and the same 768-dim vector space** — see §4.2
for why that's possible without a second embedding pipeline.

The two paths diverge sharply past embedding — image queries always use an absolute
cosine floor plus a mandatory LLM rerank; text queries use free-ranking cheap scoring
by default and only escalate to an LLM when results are genuinely ambiguous:

```text
  IMAGE                                          TEXT
  ─────                                          ────
  preprocessing.prepare_image_bytes()            text = query_text.strip()  (blank
    — same normalization as indexing               after stripping is treated as
                    │                               "not provided" → 400)
                    ▼                                             │
  reranker.is_plausibly_jewelry(bytes)                            │
    cheap ONE-WORD gate call, BEFORE the                          │
    expensive pipeline (§13) -- "no" short-                       │
    circuits straight to no_match=True with                       │
    a reason string, no embedding/search spent                    │
                    │ "yes"                                       │
                    ▼                                              ▼
  embeddings.embed_image(bytes)                  cache.cache_key(text, filters) lookup
    gemini-embedding-2, NO task_type               (§12) -- on a hit, skip straight to
    param (§4.2 -- unsupported by this model)       vector_db.search() with the cached
                    │                               vector; on a miss:
                    │                               embeddings.embed_text_query(text)
                    │                                 same model, prompts with
                    │                                 "task: search result | query: "
                    │                                 instead of a task_type param (§4.2),
                    │                               then cache.set() the result
                    │                                             │
                    └─────────────────────┬───────────────────────┘
                                           ▼
                  vector_db.search()  — Pinecone ANN query against the SAME
                    index the catalog was indexed into (top_k=20 by default),
                    optional metadata filter ({"category": {"$eq": ...}},
                    {"price": {"$gte"/"$lte": ...}}) — identical call shape for
                    both paths; scores clamped to [0, 1] at the source (§5)
                    │                                             │
                    ▼                                             ▼
  Filter to score >= MIN_SIMILARITY_THRESHOLD    no_match=True ONLY if raw_matches is
    (0.55 default) BEFORE reranking -- image-      empty outright (empty catalog, or a
    vs-image scores are reliable enough for an     filter matching nothing) -- NO absolute
    absolute cutoff. Empty -> no_match=True,        cosine floor (Google's own guidance,
    reranker.rerank() never called.                 and cross-modal scores run structurally
                    │                               lower -- a real production incident
                    │                               with a single shared threshold; §11).
                    │                               Otherwise: reranker.score_candidates_
                    │                               cheap(text, candidates) -- ZERO API
                    │                               calls, blended cosine+lexical score (§6)
                    │                                             │
                    │                               gap = top blended_score - 3rd place's
                    │                               (or last, if <3 candidates)
                    │                                     │                    │
                    │                              gap >= 0.1              gap < 0.1
                    │                              (well separated)     (ambiguous)
                    │                                     │                    │
                    │                              use cheap-scored    reranker.rerank()
                    │                              results directly   on just these
                    │                              (no LLM call)      candidates
                    ▼                                     │                    │
  reranker.rerank({"type": "image", ...}, candidates)     └────────┬───────────┘
    ALWAYS runs for image queries -- no cheap-                     │
    scoring equivalent exists for judging visual                   │
    traits against a photo. See §6 for the prompt.                 │
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                             ▼
        Every path's results (whichever ranking method produced them) are sorted by
        reranker.final_rank_score() -- cosine score dominant, confidence a bounded
        +/-0.05 tiebreaker, never able to override a real score gap (§6, resolved).
                                             ▼
                      SearchResponse: {query_id, no_match, query_type, reason,
                        matches: [{id, similarity_percent, confidence, reason,
                        metadata}]}  — similarity_percent is the REAL cosine
                        score * 100, never an LLM-invented number (see §6);
                        `reason` is non-null only for a domain-gate rejection
```

---

## 4. Embedding strategy & vector space

### 4.1 What actually gets embedded

There is no sub-image chunking and no text-chunking, because this isn't a text-document
RAG system — a jewellery photo isn't split into patches or tiles before embedding.
Query vectors (image or text) are each produced by a single `embed_content()` call.

**Catalog vectors are different: they're fused, not pixels-only.**
`embeddings.embed_catalog_item(image_bytes, text_description)` sends the normalized
image AND a text description — `f"name: {name}. {caption}. {description}. Tags:
{tags}."`, built from the item's own stored metadata (`main.py::
build_catalog_text_description`) — in ONE interleaved multimodal call
(`contents=[Part.from_bytes(...), text_description]`), producing a single vector that
captures both pixel content and stated metadata. This exists specifically to raise
cross-modal (text-query vs. catalog-item) cosine scores: a catalog vector built from
pixels alone has nothing of a text query's own modality to align with beyond whatever
the model infers purely visually — a real, measured production gap before this fix (see
§11's history and §12's conditional-rerank cost note). Query-time image search still
embeds the query photo alone (`embeddings.embed_image`, no text) — there's no metadata
to fuse for an unindexed upload.

**The closest analog to "chunking" here is `preprocessing.py`'s normalization step**,
and it exists for the same underlying reason chunking strategy matters in text RAG:
consistent input granularity. Before any image reaches an embedding model:

1. **EXIF rotation is corrected** (`ImageOps.exif_transpose`) — a phone photo shot
   sideways must be embedded right-side-up, or visual features land in the wrong place.
2. **Converted to RGB** — strips alpha channels / CMYK inconsistencies that would
   otherwise be an arbitrary source of variance between images.
3. **Center-cropped to square** — jewellery product shots are approximately square;
   this removes background padding that would otherwise dilute the embedding with
   irrelevant content.
4. **Resized to a max dimension of 1024px** (`MAX_DIMENSION`) — keeps embedding
   latency/cost predictable regardless of whether the upload was a 400px thumbnail or a
   12MP phone photo.
5. **Re-encoded to JPEG** — one consistent format reaching every embedding call.

Both the catalog-indexing path and the query-search path call the exact same
`prepare_image_bytes()` function. This matters: if a customer's query photo were
normalized differently than the catalog photos it's being compared against (e.g.
different crop logic), the *systematic* difference would show up as noise in every
similarity score, not just occasional bad matches.

### 4.2 The embedding provider

`gemini-embedding-2` (`embeddings.py`) is a natively multimodal model — image and/or
text bytes go in, a 768-dim vector comes out, in one call.

**`task_type` is NOT supported by this model and was previously sent for nothing.** An
earlier version of this codebase passed `task_type="RETRIEVAL_QUERY"` /
`"RETRIEVAL_DOCUMENT"` the way the older text-only `gemini-embedding-001` API expects —
confirmed against the official Google docs and a filed llama_index bug report that this
parameter is silently ignored by `gemini-embedding-2`. It has been removed from every
call in this module; no `EmbedContentConfig` here sets it. Since there's no `task_type`
to lean on for the query/document asymmetry, a **text query** instead gets an explicit
task instruction baked directly into the input string:
`f"task: search result | query: {text}"` (`embeddings.py::embed_text_query`,
`_TEXT_QUERY_TASK_PREFIX`) — the prefix convention Google's own docs describe for this
model. Images have no text to prefix onto and are embedded with no task hint at all
(`embeddings.py::embed_image`).

**Text search reuses the exact same model and space.** Because `gemini-embedding-2` is
natively multimodal — not a text model and an image model bolted together — a prefixed
query string handed to `embed_content()` lands in the *same* 768-dim space as the
catalog's fused vectors (§4.1). No second Pinecone index, no second embedding call path
with different dimensions, no provider switch: a text query is compared against the
catalog's fused vectors directly.

**Why not a local/offline option too?** One was built and tested: a caption-then-embed
workaround (a local vision model describes the image in one sentence, a local
text-embedding model embeds that description) for `EMBEDDING_PROVIDER=lmstudio`,
back when no working Gemini key was available. It genuinely worked end-to-end, but
measurement showed real costs: the derived vector is a function of a **text
description** of the image, not the pixels — information the caption omits or gets
wrong is invisible to retrieval, and free-text generation isn't fully deterministic (a
duplicate image embedded twice measured **0.886** self-similarity against itself, not
~1.0). See [MODEL_COMPARISON.md](../MODEL_COMPARISON.md) for the full comparison. Once
a real Gemini key was available, the workaround was removed rather than kept as a
fallback — see [ISSUES.md §2.2](../ISSUES.md#22-no-local-model-can-embed-images-directly).

### 4.3 Why the whole catalog must share one embedding space

Cosine similarity between two vectors is only meaningful if both vectors were produced
by the *same* process. Query and catalog vectors coming from different embedding
techniques (different models, different techniques) can coincidentally share a
dimension count while living in completely unrelated vector spaces — Pinecone will
happily compute a cosine score between them, and that score will be numerically valid
but semantically meaningless. This is exactly what would have happened switching from
the (now-removed) LM Studio embedding path to Gemini without re-embedding the existing
catalog first — see
[ISSUES.md §4.1](../ISSUES.md#41-switching-embedding-providers-without-re-embedding-the-catalog-would-have-silently-broken-search)
for how that was actually handled at the time. Worth remembering if another embedding
provider is ever added: **any embedding-provider change requires re-embedding the
entire catalog**, not just switching the setting for future queries.

The same principle applied within Gemini itself, not just across providers: fixing the
task_type bug and adding text fusion (§4.1) both changed what a catalog vector actually
represents, so every existing vector was regenerated via `scripts/reembed_catalog.py`
(§9) rather than left mixed with new-style vectors in the same index.

---

## 5. Vector database (Pinecone)

`vector_db.py` is the only module that talks to Pinecone.

- **Index**: serverless, created lazily on first use (`get_or_create_index()`) if it
  doesn't already exist — `dimension=768` (matching `EMBEDDING_DIMENSIONS`),
  `metric="cosine"`, cloud/region from `.env` (`aws` / `us-east-1` by default).
- **Upsert** (`upsert_batch`): takes `[{"id", "vector", "metadata"}, ...]`, chunks into
  groups of 100 (Pinecone's recommended batch size), calls `index.upsert()` per chunk.
  **This is a full replace** — upserting an existing ID overwrites its metadata
  entirely, not merges it. (Contrast with `index.update(set_metadata=...)`, used
  once for the `image_url` backfill in [ISSUES.md §5.1](../ISSUES.md#51-search-results-had-no-way-to-show-an-actual-photo),
  which *does* merge.) `main.py`'s indexing path calls this once per item (not once for
  the whole batch) specifically so a Pinecone-level failure on one item doesn't affect
  the others.
- **Metadata schema actually stored** per catalog item: `filename`, `name`, `caption`,
  `description`, `tags` (list), `category`, `price`, `image_url`, `material` (optional).
  This is what makes `category`/`price` search filters and the admin "Recently added"
  table possible — see [ISSUES.md §6.1](../ISSUES.md#61-catalog-items-only-ever-stored-a-filename--categoryprice-search-filters-were-dead-code)
  for why this wasn't always true.
- **Search** (`search`): `index.query(vector, top_k, include_metadata=True, filter)`,
  reshaped into `[{"id", "score", "metadata"}, ...]`. `filter` defaults to `{}` (not
  `None`) when the caller doesn't supply one — Pinecone's API is picky about this.
  Every score is clamped to `[0.0, 1.0]` here, at the source, before it leaves this
  function — Pinecone's approximate cosine computation can overshoot slightly (e.g.
  `1.004`) due to floating-point error in the ANN index; clamping once here means no
  caller (or the frontend's `similarity_percent` display) ever has to guard against it.
- **Listing** (for the admin UI): Pinecone's `list()` call returns IDs only, no
  metadata, and isn't designed for admin-table pagination — so `GET /api/v1/catalog/items`
  reads from `catalog_store.py`'s local JSON file instead, which is written alongside
  every Pinecone upsert (see §7).

---

## 6. Reranking strategy

`reranker.py` has two ranking paths now, not one:

- **`score_candidates_cheap(query_text, candidates)`** — the DEFAULT for text queries.
  Zero external API calls. Blends cosine similarity with lexical overlap against each
  candidate's stringified metadata: `blended = 0.7 * cosine_score + 0.3 * overlap`,
  where `overlap` is the fraction of the query's (lowercased, whitespace-split) terms
  that appear anywhere in the candidate's metadata values. Confidence buckets off the
  blended score (`high` > 0.75, `medium` > 0.5, else `low`) — a heuristic label, not an
  LLM judgment.
- **`rerank(query, candidates)`** — the LLM path. `query` is `{"type": "image", "bytes":
  <jpeg bytes>}` or `{"type": "text", "text": <query string>}`, branching which prompt
  template is used:
  - **Image query** (`_IMAGE_PROMPT_TEMPLATE`): the reference photo is sent as an
    actual multimodal `Part` alongside the prompt. The LLM reasons about what it can
    literally *see* — gemstone cut, metal color/finish, chain or band pattern,
    silhouette — against a deliberately narrow metadata slice (`name`/`category`/
    `material` only, to control token cost — see `utils.py`'s `_CHAT_MAX_TOKENS`
    comment on the production Groq TPM incident this avoids).
  - **Text query** (`_TEXT_PROMPT_TEMPLATE`): there is no image to send — the prompt
    contains only the query string and each candidate's metadata, widened to
    `name`/`category`/`material`/`caption`/`description`/`tags` since those are the
    only place a searched-for color/material term could actually appear. The LLM is
    told **not** to invent or assume visual details it has no way to observe from text
    alone — this guards against fabricating a "gemstone cut" judgment it never saw.

### Which path runs, and why (cost model)

- **Image queries always use `rerank()`** — no cheap-scoring equivalent exists for
  judging visual traits against a photo; there's nothing lexical to compare a photo
  against. Cost per image search: 1 embedding call + 1 LLM call, unchanged.
- **Text queries default to `score_candidates_cheap()`**, escalating to `rerank()` only
  when results are genuinely ambiguous (`main.py`'s `_search_text`): compute the gap
  between the top result's `blended_score` and the 3rd-place result's (or the last
  available, if fewer than 3 candidates). A gap `>= 0.1` means the top results are
  clearly separated — use the cheap-scored results directly, `rerank()` is never
  called. A gap `< 0.1` means the top results are too close to trust a heuristic blend
  — escalate to a real LLM judgment on just those candidates. Each search logs which
  path ran at INFO level (`"text search: cheap path used"` / `"text search: LLM rerank
  triggered, gap=..."`) so this is observable in production without extra
  instrumentation.
  **Cost impact: most text searches now cost 1 embedding call + 0 LLM calls**, down
  from 1 embedding + 1 LLM call for every search before this change (and a text-query
  cache hit — §12 — drops even the embedding call for a repeated query).

### Shared properties of both paths

1. **Never invents a percentage.** The percentage shown to users (`similarity_percent`)
   is always the *real* cosine score (clamped at the source — §5). Both ranking paths
   produce a **categorical confidence** (`high`/`medium`/`low`) plus a short reason
   instead — a grounded (or heuristic) judgment, never a fake-precise figure.
2. **`rerank()`'s LLM call is always batched**, not one call per candidate — see
   [MODEL_COMPARISON.md](../MODEL_COMPARISON.md) for measured latency across providers.
   Provider-switched (`gemini` / `groq`) at the raw-text-generation step only; JSON
   parsing, the malformed-response fallback, and sorting are identical either way.
3. **Sorting is unified and RESOLVED (previously an open question).**
   `reranker.final_rank_score(candidate) = candidate["score"] + CONFIDENCE_WEIGHT[confidence]`,
   with `CONFIDENCE_WEIGHT = {"high": 0.05, "medium": 0.0, "low": -0.05}` — ONE shared
   function, imported and used by both `rerank()` and `score_candidates_cheap()`, not
   two independently-maintained sorts that could drift apart. This replaced the
   previous `(confidence_rank, -score)` scheme, where confidence tier was the *primary*
   sort key and cosine score only broke ties within a tier — that let a wrongly "high
   confidence" result outrank a correctly "low confidence" one with genuinely higher
   cosine similarity, observed for real (a high-scoring result landing 6th; see
   [ISSUES.md §3.5](../ISSUES.md#35-confidence-tier-first-sorting-can-rank-a-wrong-but-confident-guess-above-a-right-but-uncertain-one)).
   `final_rank_score` fixes this by making confidence a **bounded tiebreaker**: it can
   nudge a close call (a 0.60 "high" edges out a 0.58 "medium") but can never override a
   real score gap (a 0.75 "medium" still outranks a 0.55 "high": 0.75 vs. 0.60). Cosine
   score is the actual retrieval signal and should dominate; confidence is the LLM's
   (or heuristic's) read on it and should only refine.
4. **Fails soft.** If the model's response doesn't parse as JSON (malformed, wrapped in
   markdown fences, or — as found with Groq's reasoning model — cut off mid-`<think>`
   block with no answer at all), every candidate falls back to
   `confidence="medium", reason="no reranker judgment available"` rather than the whole
   search failing. A `<think>...</think>`-stripping step runs before parsing, as
   defensive insurance for any reasoning-style model.
5. **Empty input short-circuits before any LLM call** in both `rerank([])` and
   `score_candidates_cheap()` — matters for cost: a `no_match` search never spends a
   reranker call on candidates that already didn't clear the bar (image queries) or on
   an empty result set (text queries).

---

## 7. Job tracking & the catalog store

Two pieces of state exist outside Pinecone:

- **Job status** (`job_store.py`, **RESOLVED** — see §14): `job_id -> {status, total,
  processed, failed_items}`. Written to by `_index_job` as it processes each item; read
  by `GET /api/v1/catalog/jobs/{job_id}`. `status` is `"pending"` while running, `"done"`
  once every item has been *attempted* (even if some failed — partial success still
  counts as done), `"failed"` only if *every single item* failed. Previously a plain dict
  in `main.py` (in-process, single-instance); now behind the same storage-swap-via-
  interface pattern as `cache.py` — `InMemoryJobStore` (single-instance, `REDIS_URL`
  unset) or `RedisJobStore` (shared across instances, `REDIS_URL` set). The underlying
  job-processing loop still runs in a `BackgroundTask` in whichever process started it —
  a real queue (Celery) that lets a *different* instance resume a killed job is still
  outstanding, see `PRODUCTION_HARDENING_PLAN.md` Phase 3.
- **`catalog_store.py`**: item metadata (`item_id -> metadata`) mirrored into a Supabase
  `catalog_items` table alongside every Pinecone upsert (`record_item`), read back
  paginated for `GET /api/v1/catalog/items` (`list_items`) and by
  `scripts/reembed_catalog.py` as the metadata source of truth for a full re-embed (§4.3,
  §9). Exists purely because Pinecone isn't a good fit for "list everything, paginated"
  — its own listing API returns IDs without metadata. Unlike `_jobs`, this IS shared
  across instances/restarts, since Supabase is external persistent storage — the earlier
  local-JSON-file version of this module was replaced for exactly that reason.
  `backend/catalog_store.json` still exists in the repo as the one-time migration
  source for `scripts/migrate_catalog_to_supabase.py`, not as a live data path anymore.

### Catalog images are committed to the repo (historical; RESOLVED going forward, see §14)

`backend/app/static/catalog/*.jpg` (the served images) is tracked in git, not ignored — a
deliberate exception to "runtime state doesn't belong in the repo." This was forced by a
real deployment problem: Render's disk is ephemeral (see `DEPLOYMENT.md` §1), so images
written at indexing time vanished on every restart, breaking every thumbnail even though
search itself kept working (the Pinecone vectors + `image_url` metadata are unaffected
by a restart — only the actual image files were gone). Committing the current set means
it ships with every deploy regardless of restarts.

This only covers whatever was committed as of when this stopgap was in effect. **As of
§14, new/updated catalog images upload straight to Cloudflare R2** (`object_storage.py`)
instead of local disk — the actual permanent fix `DEPLOYMENT.md` §5 previously pointed
to. `scripts/migrate_images_to_object_storage.py --dry-run` (then for real) backfills
`image_url` for every item indexed before this cutover, using the still-committed
`static/catalog/*.jpg` files as its source.

---

## 8. Frontend architecture

Two routes (`react-router-dom`), sharing one design system:

- **`/`** (`pages/SearchPage.jsx`) — the chat-style search page. Drag/drop or attach a
  photo, get back ranked results as cards. Falls back to canned demo data if the backend
  is unreachable, specifically so the UI is still inspectable/reviewable standalone.
- **`/catalog`** (`pages/CatalogPage.jsx`) — the admin bulk-upload page. Drag/drop multiple
  images, fill in an editable row per image (name/category/price required; caption/
  description/tags/material optional), submit, and watch a real job-status poll to
  completion. **Deliberately has no demo fallback** — silently showing fake "success" on
  an admin tool that manages real inventory would be actively misleading, so backend
  failures surface as a visible inline error instead.

**Shared design tokens live in `theme.css`**, imported once, globally, in `main.jsx` —
`:root` CSS variables, fonts, `.app-shell`, `.header`, `.wordmark`, `.shimmer-line`.
This is not a stylistic choice, it's a correctness requirement: React's inline
`<style>` tags aren't scoped, but since only one route's component tree is ever mounted
at a time, a page-specific `<style>` block (like `pages/SearchPage.jsx`'s) simply doesn't exist in
the DOM when a *different* route is active. Anything both pages need has to live
somewhere that's always mounted — see
[ISSUES.md §6.3](../ISSUES.md#63-the-new-admin-page-would-have-rendered-completely-unstyled)
for how this was actually caught (rendering `/catalog` in a real browser, not by
reading the code).

No `localStorage`/`sessionStorage` anywhere on either page — all state is in-memory
React state for the session. A page refresh mid-upload on `/catalog` loses any
unsubmitted rows; accepted as a v1 tradeoff.

---

## 9. Setup

```bash
cp .env.example .env   # fill in real GEMINI_API_KEY, PINECONE_API_KEY, APP_API_KEY,
                        # and GROQ_API_KEY if LLM_PROVIDER=groq
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Also run `backend/scripts/sql/001_api_keys_search_events_feedback.sql` once in the
Supabase SQL editor -- `api_keys.py` and `search_events.py` read/write those three
tables but nothing creates them automatically (unlike `catalog_items`, which predates
this). Skipping it means `POST /api/v1/search` silently fails to log to `search_events`
(caught, logged, doesn't break the search response) and any real per-client API key
issued via `create_api_key.py` won't validate.

API docs at `http://localhost:8000/docs` once running.

```bash
cd ../frontend
npm install
npm run dev   # http://localhost:5173
```

### Key `.env` variables

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `gemini` or `groq` — see §6 |
| `EMBEDDING_DIMENSIONS` | Must match what `gemini-embedding-2` actually outputs, and must match the Pinecone index's fixed dimension |
| `TOP_K` | How many ANN candidates `vector_db.search()` retrieves; also the cap on how many text candidates `score_candidates_cheap`/`rerank` see (§6) |
| `MIN_SIMILARITY_THRESHOLD` | Cosine score floor for **image** queries only before a candidate is sent to the reranker. Text queries have NO absolute floor — see §3.2/§11 |
| `REDIS_URL` | Empty = `InMemoryCache` (single-instance stopgap, §12). Set to a real Redis URL (a free [Upstash](https://upstash.com) database works) to switch to `RedisCache` automatically |
| `GROQ_CHAT_TIMEOUT_SECONDS` | Must be generous enough for a full `TOP_K`-candidate batch in one prompt, not just a single small request (see §6) — Groq's cloud inference needs far less headroom here than the local model that used to fill this role did |
| `GROQ_MODEL` | Whatever vision-capable model is available on the account; verify via `GET /v1/models`, don't assume a name |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | Catalog metadata table (§7) — required, no fallback |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET_NAME` / `R2_PUBLIC_URL_BASE` | Cloudflare R2 object storage for catalog images (§14) — required, no fallback (a boto3 client is built at import time and fails fast on a missing account ID) |
| `SENTRY_DSN` | Empty = Sentry disabled. Set to enable error tracking (§14) |

### Scripts (`scripts/`)

| Script | Purpose |
| --- | --- |
| `smoke_test.py` | End-to-end flow, fully mocked, no real API keys — for CI/quick sanity checks |
| `reembed_catalog.py` | ⚠️ **Destructive full-catalog overwrite.** Re-embeds every already-indexed item with the current `embed_catalog_item` (fused image+text) and upserts under the same `item_id`s, replacing every existing vector in Pinecone. Needed any time the embedding process itself changes (task_type fix, adding text fusion — see §4.3) since old and new vectors aren't comparable in the same index. **Always run `--dry-run` first** to verify the item count and see what would be upserted before committing to the real run |
| `migrate_catalog_to_supabase.py` | One-off: copies every item from the legacy local `catalog_store.json` into the Supabase `catalog_items` table. Safe to re-run (upserts by `item_id`) |
| `compare_rerankers.py` | Runs the exact same candidate set through both reranker providers (Gemini, Groq) back-to-back and prints a side-by-side comparison — how [MODEL_COMPARISON.md](../MODEL_COMPARISON.md)'s reranker numbers were actually produced |
| `backfill_image_urls.py` | One-off: adds `image_url` metadata to already-indexed items without re-embedding (uses Pinecone's merge-semantics `update()`, not `upsert()`) |
| `fetch_sample_images.py` | Downloads a small set of properly-licensed (Wikimedia Commons) sample jewellery photos for local testing |
| `index_local_folder.py` | Bulk-indexes every image in a local folder, deriving name/category from filenames |

## 10. Testing

No real API keys required — every external call (Gemini, Pinecone, Groq) is mocked, and
`conftest.py` explicitly pins `LLM_PROVIDER=gemini` regardless of what a developer's
local `.env` says (this wasn't always true — see
[ISSUES.md §1.4](../ISSUES.md#14-test-suite-briefly-executed-against-the-real-local-llm-server)
for why that pin exists; it was originally added because a real local LLM server was
reachable and the pin was missing, back when LM Studio was still a valid provider).

```bash
pip install -r requirements-dev.txt
pytest -v
python scripts/smoke_test.py
```

## 11. Known limitations / production considerations

- ~~`_jobs` (indexing progress) is in-memory/single-instance~~ — **RESOLVED**, see §7/§14
  (`job_store.py`, Redis-backed when configured). The `BackgroundTask` itself still runs
  in a single process; a real queue (Celery, so a *different* instance can resume a
  killed job) is still outstanding — `PRODUCTION_HARDENING_PLAN.md` Phase 3.
- ~~A single shared `APP_API_KEY` for all clients, no rate limiting~~ — **RESOLVED**, see
  §14 (`api_keys.py`, `rate_limit.py`). `APP_API_KEY` is still accepted as a "legacy"
  client during rollout — retire it once every real client has a per-client key.
- Object storage for catalog images, a real task queue (Celery), Sentry error tracking, a
  metrics dashboard, CI/CD + staging, and search-event data collection are still
  outstanding — see `PRODUCTION_HARDENING_PLAN.md` for the full phased plan and
  sequencing rationale.
- ~~The reranker's confidence-tier-first sort can let a wrong LLM judgment override a
  correct cosine-similarity signal~~ — **RESOLVED.** `final_rank_score()` replaced the
  confidence-tier-first sort with a bounded additive weight; see §6, point 3.
- There is no offline/no-API-key fallback anymore — both `GEMINI_API_KEY` and (if
  `LLM_PROVIDER=groq`) `GROQ_API_KEY` are required. A local caption-then-embed path
  existed and worked but was removed after comparison showed it was measurably less
  accurate and non-deterministic — see [MODEL_COMPARISON.md](../MODEL_COMPARISON.md).
- `gemini-2.5-flash` is a valid reranker option (`LLM_PROVIDER=gemini`) but was never
  live-tested with real image data this session — only Groq was actively measured
  against the (now-removed) local model. See the caveat in
  [MODEL_COMPARISON.md](../MODEL_COMPARISON.md) before assuming it performs like
  `gemini-embedding-2` just because they're the same vendor.
- `BackgroundTasks` (stdlib, in-process) is fine at this scale; swap for a real queue
  (Celery/RQ/Cloud Tasks) before indexing volume grows large enough that a backend
  restart mid-batch becomes a real operational risk.
- **Text search: how the no-`no_match`-for-low-scores fix actually evolved.** Text
  queries were compared cross-modal (text vector vs. image-derived catalog vectors) and
  measured lower in absolute cosine score than image-vs-image comparisons — confirmed
  against the live production catalog after text search initially returned `no_match`
  for every real query. The first fix was a separate, lower `MIN_SIMILARITY_THRESHOLD_TEXT`
  (0.35). This consolidation pass replaced that entirely: text queries now use fused
  catalog embeddings (§4.1, which measurably raises cross-modal scores) and have **no
  absolute cosine floor at all** — filtering is rank-based (`score_candidates_cheap`'s
  blended score + the conditional LLM-rerank gap check, §6), matching Google's own
  guidance against a fixed cutoff for this model. `MIN_SIMILARITY_THRESHOLD_TEXT` no
  longer exists as a setting.
- **`score_candidates_cheap`'s lexical overlap is naive**: a plain substring check
  against stringified metadata values, no stemming, no synonym matching, no handling
  for multi-word phrases as a unit. "gold" won't match "golden"; "diamond ring" as a
  two-word query gets scored as two independent single-word overlaps. This is
  acceptable because it's a coarse pre-filter for the ambiguity-gap check (§6), not the
  final word on ranking — genuinely ambiguous results still get a real LLM judgment.
- **The 0.1 conditional-rerank gap threshold (§6) is a starting heuristic, not an
  empirically tuned value.** It has not been measured against a labeled set of "should
  have escalated to LLM" vs. "was fine with the cheap path" real queries — if text
  search quality complaints emerge in production, check whether the gap threshold
  itself (rather than the scoring formula) is the actual problem before changing the
  formula.

See also §12 (caching) and §13 (image domain gating) for their own limitations.

---

## 12. Caching (`cache.py`)

Text queries embed the query string on every search unless the exact same query (text
and metadata filter) has been searched recently — `cache.py` exists to skip that repeat
embedding call.

- **`SearchCache`** (`abc.ABC`): a two-method interface, `get(key) -> vector | None` and
  `set(key, vector, ttl_seconds=3600) -> None`. Both concrete implementations below only
  ever get called through this interface — nothing in `main.py` touches
  `InMemoryCache`/`RedisCache` directly.
- **`InMemoryCache`**: a process-local dict with TTL eviction, checked lazily on `get()`
  (no background sweep). This is a single-instance stopgap with **exactly the same
  limitation already documented for `_jobs` in §7**: it disappears on restart and isn't
  shared across multiple backend processes — a second instance won't see another
  instance's cached vectors, so cache hit rate degrades (but correctness doesn't — a
  miss just re-embeds) as soon as more than one backend process is running. Used
  automatically whenever `REDIS_URL` isn't set.
- **`RedisCache`**: the real, multi-instance-safe implementation — a thin wrapper over
  `redis-py`, storing each vector as a JSON string with Redis's own `EX` TTL. Connection
  is lazy (`redis.from_url()` doesn't connect until the first command), so a bad/missing
  `REDIS_URL` fails on first use, not at import time. **Free hosting exists**: Upstash
  (<https://upstash.com>) gives a serverless Redis database on its free tier (no credit
  card, 10K commands/day, 256MB) — copy its `rediss://` connection URL into `REDIS_URL`
  and it works with no code change, since `_make_cache()` (`cache.py`) picks
  `RedisCache` over `InMemoryCache` automatically whenever `settings.redis_url` is
  non-empty.
- **`cache_key(query_text, filters)`**: `sha256` of the lowercased/stripped query text
  plus the metadata filter dict (JSON-serialized with `sort_keys=True`, so key order in
  the filter dict doesn't produce a different cache key for an equivalent filter).
  Different filters for the same text string are different cache entries on purpose —
  `"gold ring"` filtered to `category=ring` is a different search than `"gold ring"`
  filtered to `category=necklace`.
- **Wiring** (`main.py::_search_text`): before calling `embeddings.embed_text_query()`,
  check `cache._cache.get(cache_key(text, metadata_filter))`; on a hit, skip the
  embedding call entirely and go straight to `vector_db.search()` with the cached
  vector; on a miss, embed normally and `cache._cache.set()` the result. **Image queries
  are NOT cached** — repeat identical image uploads are rare, and hashing image bytes
  for a cache key is out of scope here.

**Limitation**: no cache invalidation exists. If a catalog item's metadata changes (or
the item is removed) within a cached vector's TTL, a cached search still returns
whatever `vector_db.search()` finds for that stale vector — this is generally fine since
the *query* vector doesn't depend on catalog contents, but worth knowing if catalog
churn and search-result staleness ever need to be reasoned about together.

---

## 13. Domain gating for image queries (`reranker.is_plausibly_jewelry`)

**The problem this fixes**: nothing in the pipeline previously verified a query image
was actually jewellery before running the full embed → search → rerank flow. A shared
general-purpose embedding space has enough ambient/positive cosine bias that an
unrelated photo (a puppy, a car, a landscape) can still clear `MIN_SIMILARITY_THRESHOLD`
against SOME catalog item — and the reranker's confidence buckets aren't designed to say
"none of these are remotely relevant," only to rank what's already been retrieved
relative to each other. Retrieving nonsense still produces confidently-labeled nonsense.

**The fix**: `reranker.is_plausibly_jewelry(image_bytes)` runs BEFORE
`embeddings.embed_image()` for every image query (`main.py::_search_image`) — a single,
cheap, tightly-constrained classification call (`max_output_tokens=5`, reuses the
existing `settings.reranker_model`, no new provider) asking exactly one question:
"Does this image show a piece of jewellery ... as the main subject? Answer with exactly
one word: yes or no." A "no" short-circuits straight to `no_match=True` with
`reason="The uploaded photo doesn't appear to show a piece of jewellery."` — no
embedding call, no Pinecone query, no rerank call spent on it. This doubles as a cost
optimization for exactly that reason.

**Explicitly does NOT apply to text queries.** There's no image to classify, and a
short or vague text query (e.g. "gold") isn't the same failure mode as an entirely
unrelated photo — rejecting it outright would be a false-positive-prone overreach in a
way the image case isn't.

**This is a heuristic gate, not a guarantee.** A single fast classification call can
produce both:

- **False negatives** — real jewellery rejected, most likely for an unusually abstract,
  sculptural, or non-standard piece the model doesn't confidently recognize as
  jewellery from a photo alone.
- **False positives** — a genuinely ambiguous but non-jewellery photo passing through
  and proceeding to the full (wasted) search pipeline.

Neither failure mode has been measured against a labeled set of real product photos
this session. Before trusting this gate for a real product surface, measure its
false-negative rate on real (including unusual/abstract) jewellery photos — a gate that
silently rejects legitimate customer uploads is a worse user experience than the puppy
photo problem it's meant to solve.

---

## 14. Production infrastructure hardening

Implemented against the phased plan in `PRODUCTION_HARDENING_PLAN.md`: Phase 1 (all of
it, including 1.3), Phase 2, Phase 4.1/4.2/4.4, Phase 5's CI half, and Phase 6. See that
document for what's still outstanding (Celery/a real job queue, a metrics dashboard, and
a staging environment) and why those specifically are deferred until real usage justifies
their cost.

- **`object_storage.py`** — catalog images upload to Cloudflare R2 instead of the
  container's local disk. This fixed a **live data-loss bug**, not a scaling nice-to-have:
  Render's disk is ephemeral (DEPLOYMENT.md §1), so any image added/replaced through the
  live `/catalog` admin page was lost on the next restart/redeploy. `scripts/
  migrate_images_to_object_storage.py --dry-run` (then for real) backfills `image_url`
  for items indexed before this cutover, from the still-git-committed `static/catalog/`
  images. Resolves the object-storage limitation noted in §7/`DEPLOYMENT.md` §5.
- **Sentry** (`SENTRY_DSN`) — uncaught exceptions in `unhandled_exception_handler` and
  per-item indexing failures in `_index_job` now report to Sentry (`sentry_sdk.
  capture_exception`), not just a log line. No-op when `SENTRY_DSN` is unset (local dev).
- **`job_store.py`** — job-status tracking (`_jobs`, formerly a plain dict in `main.py`)
  moved behind the same storage-swap-via-interface pattern as `cache.py`:
  `InMemoryJobStore` (single-instance, `REDIS_URL` unset) or `RedisJobStore` (shared
  across instances, 24h TTL, `REDIS_URL` set). Resolves the `_jobs` limitation in §7/§11.
- **`api_keys.py`** — per-client API keys, hashed (SHA-256) in a new Supabase `api_keys`
  table (`key_id`, `hashed_key`, `client_name`, `rate_limit_tier`, `revoked_at`). Issue
  with `scripts/create_api_key.py --client-name <name> [--tier <tier>]` (prints the raw
  key once); revoke with `scripts/revoke_api_key.py --key-id <uuid>`. The legacy shared
  `APP_API_KEY` is still accepted, mapped to client_name `"legacy"`, as a deprecation
  path. **Retirement date: 2026-10-27** — by then, every real client should have a
  per-client key; remove the legacy branch in `require_api_key` (`main.py`) and drop
  `APP_API_KEY` from `.env.example`/`render.yaml`. Resolves the single-shared-key
  limitation in §11.
- **`search_events.py`** (Phase 6) — one row per search written to Supabase
  `search_events` (query type, candidates retrieved, path taken, results returned,
  no_match), plus `POST /api/v1/search/{query_id}/feedback` writing to
  `search_feedback`. Best-effort at the call site in `main.py`'s `search()` — a Supabase
  hiccup here is logged and reported to Sentry but never breaks the actual search
  response. No training pipeline consumes this data yet; it exists so real search
  behavior isn't lost before there's something to train on it.
- **CI** — `.github/workflows/ci.yml` runs the full `pytest` suite and
  `scripts/smoke_test.py` on every PR touching `backend/**`, blocking merge on failure.
  No deploy step: Render's own GitHub integration handles deploys from `main`. A
  separate staging environment (its own Pinecone index, Supabase project, Redis
  instance) is deliberately deferred — see `PRODUCTION_HARDENING_PLAN.md` Phase 5.
- **`rate_limit.py`** — Redis-backed, tier-aware rate limiting, applied separately to
  `/api/v1/search` and `/api/v1/catalog/index` (bulk indexing gets a much lower budget
  than search). Implemented as a fixed-window counter rather than a literal token bucket
  — same practical effect, simpler to reason about and test. No-op when `REDIS_URL` isn't
  set, same tradeoff as `cache.py`/`job_store.py`'s in-memory fallbacks. Returns `429`
  with a `Retry-After` header on limit, not a bare `500`.
- **`GET /health`** — now actually checks Pinecone (`vector_db.ping()`), Supabase
  (`catalog_store.ping()`), and Redis (`cache.ping()`, only when `REDIS_URL` is set)
  reachability, returning `{"status": "ok"|"degraded", "checks": {...}}` with a `503` on
  any failure, instead of a static `{"status": "ok"}`.
- **`logging_config.py`** — structured (JSON) logging, installed at `main.py` import
  time. Existing `logger.info`/`logger.exception` call sites are unchanged; pass
  `extra={"structured_fields": {...}}` to attach queryable fields to a log line. The
  `/api/v1/search` endpoint logs one `"search request completed"` line per request with
  `request_id`, `client_name`, `query_type`, `cache_hit` (text queries only),
  `path_taken`, `result_count`, `no_match`, and total `latency_ms`. Per-stage latency
  (embed / vector search / rerank individually) was scoped out of this pass — it would
  require threading timers through `embeddings.py`/`vector_db.py`/`reranker.py`; the
  request-level total is what shipped here.
