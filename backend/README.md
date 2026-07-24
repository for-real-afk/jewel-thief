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
  config.py         Settings — every env var, one place, loaded via python-dotenv
  preprocessing.py   Image normalization (the "chunking" analog — see §4)
  embeddings.py      Image -> vector via Gemini (gemini-embedding-2)
  vector_db.py       Pinecone client: index creation, upsert, ANN search
  reranker.py        LLM-judged reranking, provider-switched (Gemini / Groq)
  utils.py           Shared retry decorator + Groq's OpenAI-compatible HTTP helper
  catalog_store.py   Local JSON-backed index of catalog items, for admin listing
  main.py            FastAPI app: routes, request validation, job tracking, static files
  static/catalog/    Persisted catalog photos, served at /static/catalog/{item_id}.jpg
  tests/             pytest suite — every external call mocked, no real API keys needed
  scripts/           One-off/utility scripts (see §9)
frontend/
  src/App.jsx          Chat search page ("/")
  src/CatalogUpload.jsx  Catalog admin page ("/catalog")
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
  1. preprocessing.prepare_image_bytes()  — normalize (§4)
  2. embeddings.embed_catalog_image()     — normalized bytes -> 768-dim vector
  3. write normalized JPEG to static/catalog/{item_id}.jpg
  4. vector_db.upsert_batch([{id, vector, metadata}])  — write to Pinecone
  5. catalog_store.record_item(item_id, metadata)      — mirror to local JSON
  6. update _jobs[job_id] progress (processed count, or failed_items entry)
  │
  ▼
Admin polls GET /api/v1/catalog/jobs/{job_id} every 2s until status is
"done" (partial failures still count as done) or "failed" (every item failed)
```

### 3.2 Search (querying with a photo)

```text
User uploads a reference photo (chat page, or POST /api/v1/search directly)
  │
  ▼
preprocessing.prepare_image_bytes()  — same normalization as indexing;
  query and catalog images MUST go through the identical pipeline, or
  systematic differences (crop, scale) would bias similarity scores
  │
  ▼
embeddings.embed_query_image()  — gemini-embedding-2, task_type="RETRIEVAL_QUERY"
  │
  ▼
vector_db.search()  — Pinecone ANN query, top_k=20 by default, optional
  metadata filter ({"category": {"$eq": ...}}, {"price": {"$gte"/"$lte": ...}})
  │
  ▼
Filter to matches with score >= MIN_SIMILARITY_THRESHOLD (0.55 default) —
  BEFORE reranking, specifically to avoid spending an LLM call on weak
  candidates that were never going to be shown. If nothing clears the bar,
  return no_match=True immediately; reranker.rerank() is never called.
  │
  ▼
reranker.rerank()  — single batched LLM call judging ALL surviving
  candidates at once (not one call per candidate — keeps latency/cost
  roughly constant regardless of K). Returns each candidate enriched with
  confidence (high/medium/low) + a one-line reason, sorted by
  (confidence tier, then cosine score) — see §6 for why this sort order
  is a live design tradeoff, not a settled decision.
  │
  ▼
SearchResponse: {query_id, no_match, matches: [{id, similarity_percent,
  confidence, reason, metadata}]}  — similarity_percent is the REAL cosine
  score * 100, never an LLM-invented number (see §6)
```

---

## 4. Embedding strategy & vector space

### 4.1 What actually gets embedded

Every vector in this system represents **one whole image** — there is no sub-image
chunking, and there's no text-chunking either, because this isn't a text-document RAG
system. A jewellery photo isn't split into patches or tiles before embedding; the whole
normalized image goes into a single `embed_content()` call and produces exactly one
768-dimensional vector.

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

`gemini-embedding-2` (`embeddings.py::embed_image`) is a natively multimodal model —
the normalized image bytes go in, a 768-dim vector comes out, in one call. `task_type`
is set to `RETRIEVAL_QUERY` for search-time queries and `RETRIEVAL_DOCUMENT` for
catalog items — this is an *asymmetric* retrieval convention (the query and the thing
being retrieved are embedded slightly differently on purpose, to optimize for "find
documents relevant to this query" rather than "find documents identical to this
query"). This vector is a direct function of pixel content — not a description of it.

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
- **Listing** (for the admin UI): Pinecone's `list()` call returns IDs only, no
  metadata, and isn't designed for admin-table pagination — so `GET /api/v1/catalog/items`
  reads from `catalog_store.py`'s local JSON file instead, which is written alongside
  every Pinecone upsert (see §7).

---

## 6. Reranking strategy

The reranker exists because raw cosine similarity alone doesn't capture everything a
human would notice — it can rank two items close in embedding space that differ in a
way a shopper would care about, or vice versa. `reranker.py::rerank()`:

1. **Never invents a percentage.** The design docstring says it directly: an LLM asked
   for "94.3% match" is producing a plausible-looking number it has no calibrated way to
   compute. The percentage shown to users (`similarity_percent`) is always the *real*
   cosine score. The LLM instead returns a **categorical confidence** (`high`/`medium`/
   `low`) plus a short, concrete reason — a grounded judgment call, not a fake-precise
   figure.
2. **One batched call, not one call per candidate.** All surviving candidates (after
   the similarity threshold filter) go into a single prompt, so latency and API cost
   stay roughly constant regardless of how many candidates there are — the whole batch
   rides on one model call, which is exactly why a slow model is a real problem: the
   now-removed local LM Studio reranker took 96 seconds for 20 candidates, vs. Groq's
   2 seconds for the same batch (see
   [ISSUES.md §2.6](../ISSUES.md#26-lm-studio-reranker-calls-timed-out-on-real-non-toy-candidate-counts)
   and [MODEL_COMPARISON.md](../MODEL_COMPARISON.md)).
3. **Provider-switched** (`gemini` / `groq`) at the raw-text-generation step only — JSON
   parsing, the malformed-response fallback, and sorting are identical regardless of
   which model answered.
4. **Sorts by `(confidence_rank, -score)`.** Confidence tier is the *primary* sort key;
   cosine score only breaks ties within a tier. This means a wrongly "high confidence"
   result can outrank a correctly "low confidence" one with genuinely higher cosine
   similarity — this was observed for real (see
   [ISSUES.md §3.5](../ISSUES.md#35-confidence-tier-first-sorting-can-rank-a-wrong-but-confident-guess-above-a-right-but-uncertain-one))
   and is flagged here explicitly as an open design question, not a settled one.
5. **Fails soft.** If the model's response doesn't parse as JSON (malformed, wrapped in
   markdown fences, or — as found with Groq's reasoning model — cut off mid-`<think>`
   block with no answer at all), every candidate falls back to
   `confidence="medium", reason="no reranker judgment available"` rather than the whole
   search failing. A `<think>...</think>`-stripping step runs before parsing, as
   defensive insurance for any reasoning-style model.
6. **Empty input short-circuits before any LLM call** — `rerank([])` returns `[]`
   immediately, which matters for cost: a `no_match` search never spends a reranker call
   on candidates that already didn't clear the similarity bar.

---

## 7. Job tracking & the local catalog store

Two pieces of state exist outside Pinecone, both **in-process and single-instance**:

- **`_jobs`** (a plain dict in `main.py`): `job_id -> {status, total, processed,
  failed_items}`. Written to by `_index_job` as it processes each item; read by
  `GET /api/v1/catalog/jobs/{job_id}`. `status` is `"pending"` while running, `"done"`
  once every item has been *attempted* (even if some failed — partial success still
  counts as done), `"failed"` only if *every single item* failed.
- **`catalog_store.py`**: a JSON file (`catalog_store.json`) mapping `item_id ->
  metadata`, written alongside every Pinecone upsert. Exists purely because Pinecone
  isn't a good fit for "list everything, paginated" — its own listing API returns IDs
  without metadata.

**Both are explicitly a v1 tradeoff, not a production design**: an in-memory dict
disappears on restart, and neither is shared across multiple backend processes. Move to
Redis (or a real table) before running more than one backend instance — a second
instance, or a restart, simply won't see jobs or catalog entries another instance
recorded.

---

## 8. Frontend architecture

Two routes (`react-router-dom`), sharing one design system:

- **`/`** (`App.jsx`) — the chat-style search page. Drag/drop or attach a photo, get
  back ranked results as cards. Falls back to canned demo data if the backend is
  unreachable, specifically so the UI is still inspectable/reviewable standalone.
- **`/catalog`** (`CatalogUpload.jsx`) — the admin bulk-upload page. Drag/drop multiple
  images, fill in an editable row per image (name/category/price required; caption/
  description/tags/material optional), submit, and watch a real job-status poll to
  completion. **Deliberately has no demo fallback** — silently showing fake "success" on
  an admin tool that manages real inventory would be actively misleading, so backend
  failures surface as a visible inline error instead.

**Shared design tokens live in `theme.css`**, imported once, globally, in `main.jsx` —
`:root` CSS variables, fonts, `.app-shell`, `.header`, `.wordmark`, `.shimmer-line`.
This is not a stylistic choice, it's a correctness requirement: React's inline
`<style>` tags aren't scoped, but since only one route's component tree is ever mounted
at a time, a page-specific `<style>` block (like `App.jsx`'s) simply doesn't exist in
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
uvicorn main:app --reload
```

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
| `TOP_K` | How many ANN candidates `vector_db.search()` retrieves before threshold filtering |
| `MIN_SIMILARITY_THRESHOLD` | Cosine score floor before a candidate is even sent to the reranker |
| `GROQ_CHAT_TIMEOUT_SECONDS` | Must be generous enough for a full `TOP_K`-candidate batch in one prompt, not just a single small request (see §6, point 2) — Groq's cloud inference needs far less headroom here than the local model that used to fill this role did |
| `GROQ_MODEL` | Whatever vision-capable model is available on the account; verify via `GET /v1/models`, don't assume a name |

### Scripts (`scripts/`)

| Script | Purpose |
| --- | --- |
| `smoke_test.py` | End-to-end flow, fully mocked, no real API keys — for CI/quick sanity checks |
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

- Job tracking and the catalog store are in-memory/single-file — move to Redis/a real
  DB before running more than one backend instance (§7).
- The reranker's confidence-tier-first sort can let a wrong LLM judgment override a
  correct cosine-similarity signal (§6, point 4) — worth reconsidering the sort
  weighting before relying on it for real merchandising decisions.
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
