# Plan: Production Infrastructure Hardening

**Status**: Phase 1 (all of it, including 1.3), Phase 2, Phase 4.1/4.2/4.4, Phase 5's CI
half, and Phase 6 are **implemented** (see `backend/README.md` §14 for details —
`job_store.py`, `api_keys.py`, `rate_limit.py`, `object_storage.py`, Sentry,
`logging_config.py`, deeper `/health`, `search_events.py`, `.github/workflows/ci.yml`).
184 tests passing. Outstanding: Phase 3 (Celery), Phase 4.3 (metrics dashboard), and
Phase 5's staging-environment half — all deliberately deferred until real usage/budget
justifies them (see updated Risks table below).

**Scope note**: this was requested as a "small feature" but is a full production-readiness
program — persistent job state, auth/rate-limiting, a real task queue, observability,
CI/CD + staging, and a data-collection pipeline. Treated here as a phased plan; nothing
is implemented until each phase (or the whole plan) is explicitly confirmed.

**Complexity**: Large (multi-week). See "Sequencing recommendation" at the end for a
smaller first slice.

## Corrections to the requested scope

Two sub-tasks in the original ask are **already done** in this codebase — implementing
them again would be redundant work:

| Requested | Actual state |
| --- | --- |
| 1.1 — Move `catalog_store.py` off local JSON onto Postgres/Supabase | **Already done.** `catalog_store.py:9-46` is fully Supabase-backed (`record_item`, `list_items`, `get_item`, `delete_item`). `backend/catalog_store.json` is kept only as the historical input to `scripts/migrate_catalog_to_supabase.py` (README §7, §9), which **already exists** — no new migration script needed. |
| 1.2 — Build a `SearchCache` abstraction, add `RedisCache` | **Already done.** `cache.py:31-92` already has the `SearchCache` ABC, `InMemoryCache`, and a fully implemented `RedisCache`, auto-selected via `settings.redis_url` (empty → in-memory, set → Redis). `redis==5.2.1` is already in `requirements.txt`. |

What's still genuinely true from 1.2: **only the `_jobs` dict** (job-status tracking in
`main.py`) is still in-process/single-instance (confirmed at `main.py:453-461` /
README §7, §11). That part of 1.2 is real work — see Phase 1 below.

Also confirmed still real and outstanding: 1.3 (images on ephemeral disk — `main.py:65-68`,
`DEPLOYMENT.md` §1/§5 already documents this and names S3/R2/Render Disk as the fix),
and all of Phases 2-6 (single shared `APP_API_KEY` at `main.py:71-73`, `BackgroundTasks`
not Celery, plain `logging` not structured JSON, no Sentry, no CI workflow file, no
`search_events`/`search_feedback` tables).

## Patterns to mirror

| Category | Source | Pattern |
| --- | --- | --- |
| Storage swap via interface | `cache.py:31-92` | Define an ABC (`SearchCache`), keep concrete impls interchangeable behind one factory (`_make_cache`) gated by a settings flag — same shape to reuse for the job store |
| Retry on external calls | `utils.py:11-16` (`external_api_retry`) | `tenacity` decorator, 3 attempts, exponential backoff, `reraise=True` — apply to any new outbound call (object storage upload, Sentry is exempt — see Phase 4 notes) |
| Migration scripts | `scripts/migrate_catalog_to_supabase.py`, `scripts/reembed_catalog.py` (README §9) | One-off script, **required `--dry-run` flag**, safe to re-run (upsert semantics, not insert-only) |
| Settings | `config.py:16-81` | Every new env var gets one line in `Settings`, `os.getenv` with an explicit default, documented inline; never read `os.environ` outside this module |
| Auth dependency | `main.py:71-73` (`require_api_key`) | FastAPI `Header(...)`-based dependency, raised as `HTTPException`; new per-key auth extends this shape rather than replacing the call sites |
| Logging | `main.py:36, 58, 269, 272, 370` | `logging.getLogger("jewellery_search")`, `logger.info`/`logger.exception` at existing decision points — Phase 4 upgrades the format, not these call sites |
| Root docs | `DEPLOYMENT.md`, `ISSUES.md`, `MODEL_COMPARISON.md` | ALL-CAPS root-level markdown per topic — this plan follows that convention |

## Files to change (by phase)

| File | Action | Why |
| --- | --- | --- |
| `backend/main.py` | UPDATE | Job store swap, per-key auth, rate limiting, Celery dispatch, structured logging, deeper `/health` |
| `backend/cache.py` | UPDATE | Add a `JobStore` ABC + `RedisJobStore`/`InMemoryJobStore` alongside the existing `SearchCache` (same file, same pattern — or a new `job_store.py` if you'd rather not overload `cache.py`; flagging as a decision, not deciding it here) |
| `backend/config.py` | UPDATE | New settings: object storage creds, `api_keys` table toggle, rate-limit tiers, Sentry DSN, Celery broker URL |
| `backend/catalog_store.py` | UPDATE | Add `image_url` write-through to object storage upload result |
| NEW `backend/object_storage.py` | CREATE | Upload/URL-generation wrapper for the chosen provider (S3/R2 — decision needed, see Risks) |
| NEW `backend/api_keys.py` | CREATE | Hash/lookup/issue/revoke logic for per-client keys |
| NEW `backend/rate_limit.py` | CREATE | Redis token-bucket, tier-aware |
| NEW `backend/tasks.py` | CREATE | Celery app + `index_catalog_item` task (body moved from `_index_job`) |
| NEW `backend/scripts/migrate_images_to_object_storage.py` | CREATE | `--dry-run` required, per existing script convention |
| NEW `backend/scripts/create_api_key.py`, `revoke_api_key.py` | CREATE | CLI issuance/revocation |
| `backend/Dockerfile`, `docker-compose.yml` (repo root) | UPDATE | Add Celery worker as a second process/service |
| `render.yaml` | UPDATE | Add worker service, new env vars (object storage, Sentry, Celery broker) |
| `backend/tests/` | UPDATE | New tests per phase (rate-limit 429 behavior, job store, auth) |
| `backend/README.md` | UPDATE | New "Production Infrastructure" section; mark §7/§11 limitations resolved |
| `DEPLOYMENT.md` | UPDATE | Object storage decision, staging setup, Celery worker deploy step |
| NEW `.github/workflows/ci.yml` | CREATE | Test-on-PR, migrate-and-deploy-on-merge |

## Phases

### Phase 1 — Remaining persistent state
- **1.2 (remainder)**: Move `_jobs` to Redis. Add `JobStore` ABC mirroring `SearchCache`'s
  shape; `RedisJobStore` key `job:{job_id}`, 24h TTL; `InMemoryJobStore` stays the
  fallback when `REDIS_URL` is unset (dev must keep working without Redis).
  - **Validate**: existing job-polling tests still pass against `InMemoryJobStore`; new
    test against a mocked Redis client for `RedisJobStore`.
- **1.3**: Real object storage for catalog images.
  - Pick one provider (S3 or Cloudflare R2 — DEPLOYMENT.md already names both plus
    Render Disks; recommend R2 for no egress fees, but this is your call, not mine to
    silently pick).
  - Indexing path (`main.py`'s `_index_job`) uploads to the bucket instead of
    `static/catalog/*.jpg`; `image_url` becomes the bucket's public/signed URL.
  - `scripts/migrate_images_to_object_storage.py`, `--dry-run` required, backfills
    `image_url` for every existing row (mirrors `backfill_image_urls.py`'s Pinecone-only
    backfill, but writing to Supabase instead).
  - Update README §7's "Catalog images are committed to the repo" section — mark
    resolved, stop committing new images once migrated (existing committed ones can stay
    or be pruned after).

### Phase 2 — Access control and abuse prevention
- **2.1**: Per-client API keys. New Supabase table `api_keys` (same Postgres already
  backing `catalog_items` — no new database). `require_api_key` (`main.py:71-73`) extended
  to hash-lookup instead of a single string compare; legacy `APP_API_KEY` still accepted
  as a fallback client during rollout (documented deprecation path in README, not silently
  dropped).
- **2.2**: Redis token-bucket rate limiting, applied separately to `/api/v1/search` and
  `/api/v1/catalog/index`, tier-driven from `api_keys.rate_limit_tier`. 429 + `Retry-After`
  on limit, not a bare 500.
  - **Validate**: new test hammering a mocked-Redis-backed endpoint past its limit,
    asserting 429 then recovery after the window.

### Phase 3 — Real job queue
- Move `_index_job`'s body into a Celery task (`backend/tasks.py`), Redis as broker
  (same Redis instance as Phases 1/2 — one connection config, not two). Job status writes
  go to the Phase 1 `RedisJobStore` so `GET /api/v1/catalog/jobs/{job_id}` is unchanged
  from the caller's/frontend's perspective.
  - Adds a second deployable process (worker) — real infra/cost change, see Risks.
  - **Validate**: kill the API process mid-batch, confirm the worker completes the
    remaining items and job status reflects it correctly on restart.

### Phase 4 — Observability
- **4.1**: Structured (JSON) logs across `main.py`, `embeddings.py`, `vector_db.py`,
  `reranker.py`, `cache.py` — same `logger.info`/`logger.exception` call sites, upgraded
  formatter/fields (`request_id`, `client_name`, `query_type`, `cache_hit`, `path_taken`,
  per-stage `latency_ms`, `result_count`, `no_match`).
- **4.2**: Sentry for uncaught exceptions (the existing `unhandled_exception_handler` at
  `main.py:49-59` is the integration point — add `sentry_sdk.capture_exception` there and
  in the Celery task's failure path).
- **4.3**: Metrics dashboard (Grafana/Prometheus or hosting-provider equivalent) — largest
  unknown-cost item in this plan; needs a hosting decision before scoping further.
- **4.4**: `/health` (`main.py:159-160`) actually pings Redis, Supabase, and Pinecone,
  returning which dependency (if any) is down, not a static `{"status": "ok"}`.

### Phase 5 — CI/CD and environments
- GitHub Actions: run `pytest` + `scripts/smoke_test.py` on every PR (both already exist
  and need no real API keys — README §10); migrate (idempotent) + deploy on merge to main.
- Staging: separate Pinecone index, separate Supabase project/DB, separate Redis, separate
  API keys — document the staging vs. prod env var diff in `.env.example`.
- Load test (Locust or k6) against staging for `/api/v1/search` and `/api/v1/catalog/index`;
  document the measured first bottleneck in README rather than guessing.

### Phase 6 — Data collection groundwork
- `search_events` table (Supabase/Postgres): one row per search, capturing retrieval +
  rerank output for future LTR/domain-classifier training. No training pipeline built now
  — data collection only.
- `POST /api/v1/search/{query_id}/feedback` + `search_feedback` table. Frontend wiring is
  a stretch goal, not required this phase.

## Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Render free plan doesn't run a persistent background worker — Celery (Phase 3) likely forces a paid tier | High | Confirm hosting budget before committing to Phase 3; `BackgroundTasks` already works for current volume, so Phase 3 can be deferred independently of Phases 1/2/4 |
| Object storage provider choice (S3 vs R2) left open by the original ask | Certain (needs a decision) | Recommend R2 (no egress fees, S3-compatible API); confirm before Phase 1.3 starts |
| Metrics dashboard (4.3) has no named provider/budget | High | Scope after seeing Phase 4.1/4.2 running for a week of real traffic — cheaper to pick metrics once structured logs exist |
| Staging environment (Phase 5.2) multiplies Pinecone/Supabase/Redis costs | Medium | Confirm this is worth it before non-prod usage volume justifies it |
| Legacy `APP_API_KEY` fallback (2.1) means the "old" auth path never gets fully retired unless someone follows up | Medium | **RESOLVED** — retirement date set: **2026-10-27** (documented in `require_api_key`'s docstring and README §14). Remove the legacy branch and drop `APP_API_KEY` from `.env.example`/`render.yaml` on or before that date. |
| Large surface area (6 phases) reviewed/merged as one PR would be unreviewable | High | Land as one PR per phase minimum, per phase's own Validate step, not one giant PR |

## Sequencing recommendation

Given the corrected scope (Phase 1 is now much smaller than originally written) and that
Phase 3/4.3/5.2 carry real new infra cost, the highest-value/lowest-risk first slice is:

**Phase 1 (remainder) → Phase 2 → Phase 4.1/4.4** — job durability, real auth/rate-limiting,
and structured logging + a real health check. This closes the two `_jobs`/auth gaps that
are actual production risks today, without committing to Celery, a metrics stack, or a
second environment until there's real traffic to justify them.

Phases 3, 4.2/4.3, 5, and 6 are independent enough to schedule later, in any order, once
the above ships.

## Acceptance
- [ ] Corrected scope agreed (1.1 and cache-in-1.2 confirmed already done, not re-implemented)
- [ ] Object storage provider decided (S3 vs R2) before Phase 1.3 starts
- [ ] Hosting budget confirmed before Phase 3 (Celery worker) starts
- [ ] Each phase ships as its own PR with its own passing tests
- [ ] README.md / DEPLOYMENT.md updated per phase, cross-referencing which §7/§11 limitation each resolves
