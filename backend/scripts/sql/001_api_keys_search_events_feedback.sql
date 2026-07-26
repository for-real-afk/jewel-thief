-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query).
--
-- api_keys.py and search_events.py reference these three tables, but no
-- migration ever created them (unlike catalog_items, §1.1 of
-- PRODUCTION_HARDENING_PLAN.md, which predates this file and was created
-- manually before that script existed). api_keys.lookup_key() hasn't hit
-- this gap yet in production because require_api_key() short-circuits on
-- the legacy APP_API_KEY before ever querying it. search_events did hit it:
-- every search logs "Could not find the table 'public.search_events' in the
-- schema cache" (PGRST205) -- caught, so search itself doesn't break, but it
-- fires sentry_sdk.capture_exception() on every single request until this
-- runs.
--
-- Safe to run once; CREATE TABLE IF NOT EXISTS makes re-running a no-op.

CREATE TABLE IF NOT EXISTS api_keys (
    key_id UUID PRIMARY KEY,
    hashed_key TEXT NOT NULL,
    client_name TEXT NOT NULL,
    rate_limit_tier TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

-- lookup_key() filters by hashed_key on every authenticated request.
CREATE UNIQUE INDEX IF NOT EXISTS api_keys_hashed_key_idx ON api_keys (hashed_key);

CREATE TABLE IF NOT EXISTS search_events (
    request_id TEXT PRIMARY KEY,
    "timestamp" TIMESTAMPTZ NOT NULL,
    client_name TEXT NOT NULL,
    query_type TEXT NOT NULL,
    query_text_or_image_hash TEXT NOT NULL,
    retrieved_candidates JSONB NOT NULL,
    path_taken TEXT NOT NULL,
    result_ids_returned_in_order JSONB NOT NULL,
    no_match BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS search_feedback (
    id UUID PRIMARY KEY,
    -- Not a foreign key to search_events.request_id: the search_events write
    -- is best-effort (main.py wraps it in try/except so a Supabase hiccup
    -- never breaks a real search), so a feedback row's parent event isn't
    -- guaranteed to exist.
    query_id TEXT NOT NULL,
    result_id TEXT NOT NULL,
    action TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS search_feedback_query_id_idx ON search_feedback (query_id);
