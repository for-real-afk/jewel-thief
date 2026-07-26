# Issues Log

Every real problem hit while building and operating Facet, in the order encountered.
For each: what broke, why, and what actually fixed it. This is a record of genuine
debugging, not a hypothetical — every entry here was reproduced and confirmed before
being called "fixed."

---

## 1. Backend setup & auth

### 1.1 Missing `x-api-key` header returned 422, not 401
**What:** The task spec (and the test suite) required a missing `x-api-key` header to
return `401 Unauthorized`. It returned `422 Unprocessable Entity` instead.
**Why:** `x_api_key: str = Header(...)` makes the header itself a *required FastAPI
parameter*. When it's absent, FastAPI's own request validation rejects the request
before the handler body (and its `require_api_key()` check) ever runs.
**Fix:** Changed to `x_api_key: Optional[str] = Header(None)` on every authenticated
endpoint, so the request always reaches `require_api_key()`, which then raises 401 for
both "missing" and "wrong" cases uniformly.

### 1.2 `.env` values were silently ignored
**What:** After creating `backend/.env` from `.env.example` and setting real values,
`uvicorn` still crashed with `ValueError: API key must be set when using the Google AI
API` — using the *default* empty-string key, not the one in `.env`.
**Why:** `config.py` read `os.environ` via `os.getenv(...)` directly but never called
`load_dotenv()`. Nothing ever loaded the `.env` file into the process environment in
the first place.
**Fix:** Added `from dotenv import load_dotenv; load_dotenv()` at the top of
`config.py`, and pinned `python-dotenv` in `requirements.txt` (it had only been present
as a transitive dependency of `google-genai` before).

### 1.3 Editing `.env` had no effect on an already-running server
**What:** After fixing 1.2, changing `EMBEDDING_PROVIDER` in `.env` and expecting
`uvicorn --reload` to pick it up did nothing — the server kept using the old value.
**Why:** `uvicorn --reload`'s file-watcher only watches `.py` files by default. `.env`
changes never trigger a reload, and `load_dotenv()` only runs once, at import time.
**Fix:** No code fix — this is inherent to `--reload`. The operational fix is to fully
kill and restart the process after any `.env` change, which is what surfaced issue 1.4.

### 1.4 Test suite briefly executed against the real local LLM server
**What:** A full `pytest` run took 84 seconds (should be ~2s) and LM Studio's own logs
showed it receiving real requests with literal placeholder bytes (`b"query-bytes"`,
the fixture value from a reranker test) as if they were image data — it correctly
rejected them, but the "test" had escaped its sandbox.
**Why:** `conftest.py` set fake `GEMINI_API_KEY`/`PINECONE_API_KEY`/etc. but never set
`EMBEDDING_PROVIDER`/`LLM_PROVIDER`. When the developer's real `.env` had
`LLM_PROVIDER=lmstudio`, `config.py`'s `load_dotenv()` (which doesn't override
already-set env vars, but *does* fill in ones `conftest.py` left unset) let that real
value leak into the "hermetic" test run, so `reranker.rerank()` dispatched to the real
LM Studio HTTP call instead of the mocked Gemini client the tests expected.
**Fix:** `conftest.py` now explicitly pins `EMBEDDING_PROVIDER=gemini` and
`LLM_PROVIDER=gemini` via `os.environ.setdefault(...)`, before any project module is
imported — regardless of what a developer's local `.env` says, tests can never reach a
real network service again.

---

## 2. LM Studio integration

### 2.1 The user-supplied curl example didn't match the real server
**What:** `curl http://localhost:1234/api/v1/chat` returned `404`.
**Why:** That endpoint shape doesn't exist. LM Studio exposes a standard
OpenAI-compatible API at `/v1/chat/completions` and `/v1/embeddings`, confirmed by
querying `/v1/models` directly.
**Fix:** Built `utils.py`'s HTTP calls against the real, verified endpoint shape —
never assumed the example curl was correct without checking.

### 2.2 No local model can embed images directly
**What:** The two local embedding models available (`text-embedding-nomic-embed-text-v1.5`,
`text-embedding-google_embeddinggemma-300m-qat`) are **text-only**. Only the chat model
(`google/gemma-4-e4b`) accepts image input, and it doesn't expose an embeddings endpoint.
**Why:** This is a real capability gap in the local stack, not a bug — there's no local
multimodal embedding model to call.
**Fix:** Built a "caption-then-embed" path for `EMBEDDING_PROVIDER=lmstudio`: the vision
chat model captions the image in one factual sentence, then the text-embedding model
embeds that caption. Two HTTP round-trips instead of one direct call. See
[MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the real accuracy/consistency cost of
this workaround.

### 2.3 Windows-native `curl.exe` can't read MSYS-style paths
**What:** `curl -F "image=@/c/Users/deepa/Downloads/photo.jpg"` failed with
`curl: (26) Failed to open/read local data from file/application`, even though the file
demonstrably existed at that path via `ls`.
**Why:** This shell's `curl` (`/mingw64/bin/curl`, an `x86_64-w64-mingw32` build) is a
native Windows binary — it doesn't understand `/c/...` MSYS path translation the way
`ls`/`cat`/etc. do. It needs a literal Windows path.
**Fix:** Use `C:\Users\deepa\Downloads\photo.jpg` (native Windows syntax) in any `curl
-F @path` argument, not the MSYS-translated form. Not a code bug — a shell-environment
gotcha worth remembering for every future `curl` file upload in this environment.

### 2.4 A stuck "ghost" listener on port 8000 silently ate requests
**What:** After several backend restarts across a long session, requests to
`localhost:8000` intermittently returned connection errors, HTTP 500s with no
corresponding server log entry, or appeared to hang — with no consistent pattern.
**Why:** Every earlier restart attempt had used `taskkill /PID <n>` against a PID read
from `tasklist`, but `netstat -ano` (a different tool, possibly a different PID
namespace in this environment) kept reporting *five* separate processes still `LISTEN`ing
on port 8000 — none of which `taskkill` could find ("process not found") even though
something was clearly still bound to the port. Requests were landing on stale/broken
listeners non-deterministically.
**Fix:** Stopped trying to reconcile git-bash's `netstat`/`tasklist` output and switched
to PowerShell's `Get-NetTCPConnection` + `Stop-Process` (a single, authoritative,
native-Windows view of what owns a port) for all future process management. Also just
moved the backend to port **8001** to sidestep whatever state port 8000 was stuck in
entirely, rather than continuing to fight it.

### 2.5 Backend log output vanished when launched via the harness's background-task capture
**What:** Requests were demonstrably succeeding (confirmed via `curl` response bodies)
but neither the request nor the server's normal startup log lines ever appeared in the
harness's captured output file for that background task — even minutes later.
**Why:** Python's stdout is block-buffered (not line-buffered) when not attached to a
real terminal, and the harness's `run_in_background` output-capture mechanism appears
not to flush/forward that buffer reliably on Windows for a long-lived process. Setting
`PYTHONUNBUFFERED=1` did not fully resolve it either.
**Fix:** Reverted to launching the server as plain `nohup ... > uvicorn.log 2>&1 &` via
Bash, which reliably captured every log line (this is the same technique — and the same
reason it works — as any Unix log-redirection setup: the shell owns the file descriptor
directly, no intermediate capture layer).

### 2.6 LM Studio reranker calls timed out on real (non-toy) candidate counts
**What:** `reranker.rerank()` against a real 20-item candidate batch raised
`requests.exceptions.ReadTimeoutError` after retrying 3 times (each waiting the full
60s) — ~3 minutes wasted before finally failing.
**Why:** `utils.lmstudio_vision_chat()` had a hardcoded `timeout=60`. That was plenty
for the single-image captioning case (and for the earlier small 3-4 candidate tests),
but the reranker's whole design point is sending *all* candidates in one batched
prompt — with real `TOP_K=20` catalog data, the prompt is much larger, and generation
time on modest local hardware scaled well past 60s.
**Fix:** Made the timeout a setting (`LMSTUDIO_CHAT_TIMEOUT_SECONDS`, default 180) and
raised it. Confirmed with real 20-candidate data afterward: reranking completed in
52–99 seconds depending on run.

### 2.7 The BackgroundTasks indexing job appeared to fail silently
**What:** Early catalog-indexing attempts returned a `job_id` immediately (as designed)
but no vectors ever appeared in Pinecone afterward, and no exception was visible
anywhere.
**Why:** Turned out to be two compounding issues, not one: (a) `_index_job` originally
wrapped the *entire* batch in one try/except with no logging inside it, so any
exception vanished without a trace; separately (b) at least one attempt was interrupted
mid-run by a `uvicorn --reload` restart triggered by an unrelated code edit landing
while the background task was still executing. Reproducing the exact same logic in a
synchronous foreground script (bypassing BackgroundTasks entirely) showed it actually
worked fine standalone — confirming the pipeline itself was never the problem.
**Fix:** Added `logger.exception(...)` inside `_index_job`'s except block so failures
are never silent again, and — more substantively, once the catalog-management feature
was built properly — restructured it to catch **per-item** exceptions (one bad image no
longer aborts the whole batch) and record `{item_id, error}` into a queryable job-status
dict instead of a bare log line.

---

## 3. Reranker correctness (category confusion, JSON parsing, sort order)

### 3.1 `infer_category()` matched "ring" inside "earrings"
**What:** Every auto-categorized pair of earrings was being filed under category
`"ring"`.
**Why:** The keyword matcher used a plain substring check (`"ring" in caption.lower()`),
and the literal substring `"ring"` occurs inside `"earrings"` (e-a-r-**ring**-s). Since
`"ring"` was checked before `"earrings"` in iteration order, it won every time.
**Fix:** Switched to whole-word matching — tokenize the caption into words via regex
and intersect with a keyword set, instead of substring search.

### 3.2 Groq's reasoning model never produced usable JSON
**What:** Every reranked result from `LLM_PROVIDER=groq` came back with
`confidence="medium"` and `reason="no reranker judgment available"` — the code's
fallback for "the LLM's response didn't parse as JSON."
**Why:** `qwen/qwen3.6-27b` (the only vision-capable model on this Groq account) is a
reasoning model that emits an extended `<think>...reasoning...</think>` block before
any real answer, by default. With no `max_tokens` cap set on the request, that block
alone consumed the entire response budget — the model was cut off mid-thought and
never reached the requested JSON array at all.
**Fix:** Two changes, both verified empirically rather than guessed: added
`"reasoning_effort": "none"` to the Groq request body (confirmed this Groq account's
model actually honors it — response came back with `finish_reason: "stop"` and clean
JSON), and set an explicit `max_tokens: 4096` as headroom. Also added a defensive
`<think>...</think>`-stripping step before JSON parsing in `reranker.py`, as cheap
insurance for any other reasoning-style model that might leak a thinking block despite
a provider-level flag.

### 3.3 The reranker prompt was ambiguous about how many images it was looking at
**What:** Inspecting Qwen's raw (unsuppressed) reasoning trace showed it spending most
of its "thinking" confused about whether the single provided image was the reference
photo, one of the candidates, or a collage of several candidates — before eventually
guessing correctly.
**Why:** The original prompt said "Below are N candidate items retrieved by vector
search" without ever clarifying that those candidates are described by ID/score/metadata
only, with no accompanying images.
**Fix:** Rewrote the prompt to state explicitly: "You are shown exactly ONE image: the
customer's reference photo. You do NOT get to see images of the N candidates below."

### 3.4 A local vision model repeatedly misclassified the same item as the wrong category
**What:** Across multiple independent real tests (different runs, different candidate
sets), `google/gemma-4-e4b` consistently described a wide, ornate gold choker necklace
as a "cuff bracelet" — not a one-off mistake.
**Why:** This is a genuine model-capability limitation (see
[MODEL_COMPARISON.md](MODEL_COMPARISON.md)), not a bug in this codebase — a wide curved
gold band is visually ambiguous between "choker" and "cuff" for a small local model
without stronger context cues.
**Fix:** No code fix applies to a model's perception error. This finding is exactly why
`MODEL_COMPARISON.md` exists as a separate, explicit record — and directly motivated
testing Groq as an alternative reranker provider, which did *not* reproduce this
specific error on the same image.

### 3.5 Confidence-tier-first sorting can rank a wrong-but-confident guess above a right-but-uncertain one
**What:** With the local model's category confusion (3.4) in play, its "high
confidence" (wrong) bracelet picks outranked "low confidence" (correct) necklace picks
that had a genuinely *higher* raw cosine similarity score.
**Why:** `reranker.rerank()`'s sort key is `(confidence_rank, -score)` — confidence tier
dominates the sort completely; cosine score only breaks ties *within* a tier. When the
LLM's tier judgment is wrong, it doesn't just add noise, it actively overrides a
correct signal that was already available.
**Fix:** Not yet changed — flagged explicitly to the user as a design tradeoff worth
reconsidering (e.g., weighting cosine score more heavily, or only letting confidence
reorder results within a similar score band) rather than silently "fixing" a
scoring-philosophy decision without sign-off.

### 3.6 Caption-then-embed is not deterministic — a photo doesn't fully match itself
**What:** Searching with the *exact same file* already present in the catalog returned
a self-similarity of ~0.886, not ~1.0 — and it wasn't even the top match.
**Why:** `EMBEDDING_PROVIDER=lmstudio`'s embedding is derived from an LLM-generated
caption of the image, not the pixels directly. Free-text generation has sampling
variance even at low temperature, so the *same* image can produce slightly different
wording — and therefore a slightly different embedding — on different calls.
**Fix:** No code fix (this is inherent to the caption-then-embed architecture, not a
bug). Documented as a real, measured limitation in `MODEL_COMPARISON.md`; resolved
functionally by switching `EMBEDDING_PROVIDER=gemini`, which embeds pixels directly.

---

## 4. Switching to real Gemini embeddings

### 4.1 Switching embedding providers without re-embedding the catalog would have silently broken search
**What:** Not an observed failure — an issue caught and prevented before it happened.
**Why:** Query and catalog vectors must live in the *same* embedding space for cosine
similarity to be meaningful. The 79 already-indexed items were embedded via LM Studio's
caption-then-embed pipeline; flipping `EMBEDDING_PROVIDER=gemini` for future queries
alone would have compared Gemini query vectors against LM Studio catalog vectors —
different models, different techniques, coincidentally the same dimensionality (768),
which would have produced plausible-looking but semantically meaningless scores with no
obvious error to signal it.
**Fix:** Wrote `scripts/reembed_catalog.py` to re-embed all 79 items under the new
provider *before* switching it on for real use, explicitly preserving each item's
existing metadata (fetched from Pinecone first) rather than reconstructing it —
Pinecone's `upsert()` fully replaces a vector's metadata, so skipping this step would
also have silently wiped the `image_url` field fixed in section 5.

---

## 5. Missing image-serving layer

### 5.1 Search results had no way to show an actual photo
**What:** Every result card in the frontend showed a generic gem-shaped placeholder
icon, never a real photo — for every search, regardless of what was uploaded.
**Why:** No part of the system ever persisted a catalog image anywhere web-accessible.
Images were read, embedded, and discarded; `/api/v1/search` only ever returned
`{id, score, metadata}`, and metadata was never given an image URL because nothing
served one.
**Fix:** Added a FastAPI `StaticFiles` mount (`/static/catalog/`), persisted each
catalog image to disk at indexing time, and added `image_url` to stored metadata.
Backfilled the 79 already-indexed items via `Pinecone.Index.update(set_metadata=...)`
(a **partial merge**, unlike `upsert()`) so existing `name`/`category`/`price` fields
were left untouched. Updated the frontend's `ResultCard` to render the real `<img>`
when `image_url` is present, falling back to the placeholder icon only when it's
missing or fails to load.

---

## 6. Catalog management page

### 6.1 Catalog items only ever stored a filename — category/price search filters were dead code
**What:** `/api/v1/search`'s `category`/`min_price`/`max_price` filters built a valid
Pinecone metadata filter, but it could never match anything, because no indexed item's
metadata ever had a `category` or `price` key in the first place.
**Why:** The original `/api/v1/catalog/index` endpoint only recorded
`{"filename": img.filename}` as metadata — no code path existed to accept or store
richer per-item data.
**Fix:** Replaced the endpoint's contract with `items_json` (a JSON array of full
per-item metadata, validated against a Pydantic model requiring `name`/`category`/
`price`), built the `/catalog` admin page around it, and added `catalog_store.py` +
`GET /api/v1/catalog/items` so indexed data is actually inspectable.

### 6.2 A non-dict `items_json` entry would have crashed with the wrong error
**What:** Caught during implementation, before shipping — not an observed runtime
failure.
**Why:** An early version's fallback for a malformed entry tried
`IndexItemMetadata(raw)` (a bare positional argument) when `raw` wasn't a dict; Pydantic
models don't accept positional construction like that, so this would have raised an
unrelated `TypeError` instead of the intended, informative 400 response.
**Fix:** Added an explicit `isinstance(raw, dict)` check ahead of construction, with its
own clear 400 error naming the offending array index.

### 6.3 The new admin page would have rendered completely unstyled
**What:** Caught by actually rendering `/catalog` in a real (headless) browser and
looking at the screenshot — not by code review alone.
**Why:** The whole design system (`:root` CSS variables, fonts, `.app-shell`, `.header`)
lived inside the chat page's own `<App>` component, inside a React `<style>` JSX tag.
React doesn't scope inline `<style>` tags — but since `react-router-dom` only ever
mounts *one* matched route's component tree at a time, `<App>` (and its `<style>` tag)
never renders at all when visiting `/catalog` directly. Every shared token would have
been simply absent from the document.
**Fix:** Extracted `:root`, `.app-shell`, `.header`, `.wordmark`, `.facet-line`, and
`.shimmer-line` into a new `theme.css`, imported once, globally, in `main.jsx` — so
every route gets the same tokens regardless of which page component is mounted.

### 6.4 A stray zombie process meant "restarted" servers weren't always running the new code
**What:** Recurring theme across the whole session (related to 2.4) — several times, a
restarted server appeared to still be running old behavior.
**Why:** Same root cause as 2.4: process management on this Windows/git-bash
combination is unreliable via `tasklist`/`taskkill` alone.
**Fix:** Standardized on PowerShell `Get-NetTCPConnection` + `Stop-Process` to positively
identify and kill whatever actually owns a port before every restart, for the rest of
the session.

---

## 7. Production hardening & the `backend/app/` restructure

### 7.1 An unconfigured object-storage client crashed the entire app on startup, not just image uploads
**What:** After shipping `object_storage.py` (Cloudflare R2 image storage), the live
Render service crash-looped: `ValueError: Invalid endpoint:
https://.r2.cloudflarestorage.com`, `==> Exited with status 1`. Every endpoint was
down, not just image uploads — the crash happened before the FastAPI app object even
finished constructing.
**Why:** `object_storage.py` built its boto3 `s3` client as a *module-level* statement,
executed immediately when `main.py`'s unconditional `import object_storage` ran at
process startup. `R2_ACCOUNT_ID` had never been added to `render.yaml`/Render's
dashboard for this newly-shipped feature, so it defaulted to `""`, producing the
literally-invalid endpoint `https://.r2.cloudflarestorage.com` — and boto3 validates the
endpoint URL eagerly at client-construction time, not on first real request.
**Fix:** Moved client construction into a `_get_client()` function, called lazily on
first actual upload/delete, with a clear `RuntimeError` naming the missing env vars if
still unconfigured — instead of at import time. Search and every other endpoint now work
regardless of R2's configuration state; only the catalog-image path is affected by a
missing R2 setup, and it now fails with an intelligible error instead of taking the
whole process down.

### 7.2 Shipped code referenced three Supabase tables that no migration ever created
**What:** Every real search request logged a caught (non-fatal) error and fired a Sentry
event: `postgrest.exceptions.APIError: {'message': "Could not find the table
'public.search_events' in the schema cache", 'code': 'PGRST205', ...}`.
**Why:** `search_events.py` and `api_keys.py` were both written against Supabase tables
(`search_events`, `search_feedback`, `api_keys`) that were never actually created by any
script or documented SQL — unlike `catalog_items`, which had a real one-off migration.
`api_keys`'s version of this gap hadn't surfaced yet only because `require_api_key()`'s
legacy `APP_API_KEY` fallback short-circuits before ever querying that table.
**Fix:** Wrote `backend/scripts/sql/001_api_keys_search_events_feedback.sql` (idempotent
`CREATE TABLE IF NOT EXISTS`, matching the exact columns each module already
reads/writes) for a human to run once in Supabase's SQL editor — no migration runner
exists in this repo, and this agent has no Supabase credentials to apply it directly.
**Not yet resolved** — the SQL has been provided but not yet run as of this entry.

### 7.3 A test-suite import sweep missed function-local imports during the restructure
**What:** After moving all 15 backend modules into `backend/app/` and converting every
module-level `import X` to `from app import X` across `tests/`/`scripts/`, one test still
failed: `test_cache.py::test_search_cache_hit_skips_embed_text_query` —
`ModuleNotFoundError: No module named 'embeddings'`.
**Why:** The grep sweep used to find stale imports was anchored to line-start
(`^import X$`), which correctly caught every module-level import but missed three
imports written *inside* a test function's body (indented `import embeddings` / `import
main` / `import vector_db`) — a pattern used nowhere else in the suite.
**Fix:** Re-ran the sweep with an indentation-tolerant pattern (`^\s+import X$`) across
`tests/`, `scripts/`, and `app/`, confirmed no further hits, and re-ran the full suite
back to 185/185 passing — the same count as the pre-restructure baseline.

### 7.4 The restructure had two silent-breakage risks, both caught before any deploy
**What:** Neither of these was ever observed failing in production — both were caught
while planning the restructure, specifically because of the precedent set by 7.1's
crash-loop.
**Why:** (a) `render.yaml`'s `startCommand` and `Dockerfile`'s `CMD` both hardcoded
`uvicorn main:app`; moving `main.py` to `backend/app/main.py` without updating either
would have caused the exact same crash-loop as 7.1 on the next deploy. (b) `main.py`'s
`STATIC_DIR = Path(__file__).resolve().parent / "static"` is relative to wherever
`main.py` itself lives — moving `main.py` into `app/` without also moving `static/` would
have silently pointed this at a *new, empty* `backend/app/static/` directory
(auto-created by the next line's `mkdir(parents=True, exist_ok=True)`, no error raised),
quietly serving broken images instead of failing loudly.
**Fix:** (a) Updated both `render.yaml` and `Dockerfile` to `uvicorn app.main:app` in the
same change that moved `main.py`. (b) Moved `static/` into `backend/app/static/`
alongside `main.py`, keeping the existing relative-path code unchanged, then verified
with a real running server that `/static/catalog/ring-10.jpg` returned `200` — not just
that `mkdir` hadn't errored.

---

## Recurring themes worth naming directly

- **Verify, don't assume, for anything crossing a process/network boundary.** Nearly
  every "mysterious" bug (2.4, 2.5, 2.7) came from state living somewhere other than the
  code being read — a stale process, a buffered log, an unloaded `.env`. The fix was
  never a subtler code change; it was checking what was *actually* running.
- **Model behavior needs to be tested with real data, not assumed from a small sample.**
  The category-confusion (3.4) and non-determinism (3.6) findings only surfaced once
  real catalog-scale data (20 candidates, 79+ items) replaced the original 3–4-item
  smoke tests.
- **A vector space is only meaningful if every vector in it was produced the same way.**
  Both 4.1 (embedding provider switch) and 5.1 (missing `image_url` backfill) are
  instances of the same underlying discipline: know whether an operation is a full
  replace or a partial merge before touching data that's already live.
- **A module-level side effect at import time can crash the whole app, not just the
  feature it belongs to.** 7.1's R2 client (and any future external-service client)
  should be constructed lazily, on first real use — not eagerly at import — so one
  misconfigured integration can't take down request paths that never touch it.
