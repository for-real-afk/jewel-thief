import json

import pytest
from fastapi.testclient import TestClient

from app import cache
from app import main

client = TestClient(main.app)

VALID_KEY = main.settings.api_key


@pytest.fixture(autouse=True)
def _fresh_cache(mocker):
    """Every test gets an isolated cache -- otherwise a cached vector from one
    test's fake data could leak into another test via the module-level
    singleton and skip embed_text_query when the test expects it to run."""
    mocker.patch.object(cache, "_cache", cache.InMemoryCache())


@pytest.fixture(autouse=True)
def _no_real_search_event_writes(mocker):
    """search() calls search_events.record_search_event() best-effort after
    every request (Phase 6 data collection) -- default it to a no-op so
    unrelated tests don't attempt a real Supabase call. Tests exercising the
    call itself override this explicitly."""
    mocker.patch.object(main.search_events, "record_search_event")


@pytest.fixture(autouse=True)
def _no_real_api_key_lookups(mocker):
    """Any x-api-key that isn't the legacy VALID_KEY falls through to
    api_keys.lookup_key() -- default that to "unknown" (None) rather than a
    real Supabase call, so tests that only care about the wrong-key/401 path
    stay hermetic (README §10). Tests exercising a real per-client key
    override this explicitly."""
    mocker.patch.object(main.api_keys, "lookup_key", return_value=None)


def _image_file(jpeg_bytes, filename="query.jpg"):
    return {"image": (filename, jpeg_bytes, "image/jpeg")}


def _fake_match(id_, score, category="ring"):
    return {"id": id_, "score": score, "metadata": {"category": category}}


def _fake_ranked(match, confidence="high", reason="matching cut"):
    return {**match, "confidence": confidence, "reason": reason}


def _pass_jewelry_gate(mocker):
    return mocker.patch.object(main.reranker, "is_plausibly_jewelry", return_value=True)


@pytest.mark.parametrize("caption,expected", [
    ("a rose gold ring with a pear-cut diamond", "ring"),
    ("a silver choker necklace with pearls", "necklace"),
    ("gold jhumka earrings with green stones", "earrings"),
    ("an engraved gold bangle bracelet", "bracelet"),
    ("a decorative brooch with no clear category", "other"),
])
def test_infer_category_matches_keywords(caption, expected):
    assert main.infer_category(caption) == expected


def test_unhandled_exception_reports_to_sentry(mocker):
    import asyncio

    from starlette.requests import Request as StarletteRequest

    mock_capture = mocker.patch.object(main.sentry_sdk, "capture_exception")
    request = StarletteRequest({"type": "http", "method": "GET", "path": "/whatever", "headers": []})

    try:
        raise ValueError("boom")
    except ValueError as exc:
        asyncio.run(main.unhandled_exception_handler(request, exc))
        mock_capture.assert_called_once_with(exc)


def test_catalog_index_per_item_failure_reports_to_sentry(mocker, valid_jpeg_bytes):
    mocker.patch.object(main.object_storage, "upload_catalog_image", return_value="https://cdn.example.com/x.jpg")
    boom = Exception("embedding API down")
    mocker.patch.object(main.embeddings, "embed_catalog_item", side_effect=boom)
    mock_capture = mocker.patch.object(main.sentry_sdk, "capture_exception")

    resp = client.post(
        "/api/v1/catalog/index",
        files=[("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg"))],
        data={"items_json": json.dumps([_item("id-1")])},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 200

    mock_capture.assert_called_once_with(boom)


def test_health_reports_ok_when_all_dependencies_reachable(mocker):
    mocker.patch.object(main.vector_db, "ping")
    mocker.patch.object(main.catalog_store, "ping")

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["pinecone"] == "ok"
    assert body["checks"]["supabase"] == "ok"
    assert body["checks"]["redis"] == "not_configured"  # REDIS_URL unset in test env


def test_health_reports_degraded_when_pinecone_unreachable(mocker):
    mocker.patch.object(main.vector_db, "ping", side_effect=Exception("connection refused"))
    mocker.patch.object(main.catalog_store, "ping")

    resp = client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["pinecone"] == "unreachable"
    assert body["checks"]["supabase"] == "ok"


def test_health_reports_degraded_when_supabase_unreachable(mocker):
    mocker.patch.object(main.vector_db, "ping")
    mocker.patch.object(main.catalog_store, "ping", side_effect=Exception("connection refused"))

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["checks"]["supabase"] == "unreachable"


def test_health_reports_redis_ok_when_configured_and_reachable(mocker):
    mocker.patch.object(main.vector_db, "ping")
    mocker.patch.object(main.catalog_store, "ping")
    mocker.patch.object(main.settings, "redis_url", "redis://fake:6379")
    mocker.patch.object(main.cache, "ping")

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["checks"]["redis"] == "ok"


def test_search_without_api_key_header_returns_401(valid_jpeg_bytes):
    resp = client.post("/api/v1/search", files=_image_file(valid_jpeg_bytes))
    assert resp.status_code == 401


def test_search_with_wrong_api_key_returns_401(valid_jpeg_bytes):
    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": "totally-wrong"}
    )
    assert resp.status_code == 401


def test_require_api_key_accepts_legacy_shared_key_without_supabase_lookup(mocker):
    mock_lookup = mocker.patch.object(main.api_keys, "lookup_key")

    result = main.require_api_key(VALID_KEY)

    assert result == ("legacy", "legacy")
    mock_lookup.assert_not_called()


def test_require_api_key_accepts_valid_per_client_key(mocker):
    mocker.patch.object(
        main.api_keys, "lookup_key", return_value={"client_name": "acme-mobile", "rate_limit_tier": "standard"}
    )

    assert main.require_api_key("some-per-client-key") == ("acme-mobile", "standard")


def test_require_api_key_rejects_revoked_or_unknown_key(mocker):
    mocker.patch.object(main.api_keys, "lookup_key", return_value=None)

    with pytest.raises(main.HTTPException) as exc_info:
        main.require_api_key("revoked-key")
    assert exc_info.value.status_code == 401


def test_search_logs_structured_completion_summary(mocker):
    """The search endpoint should log one structured summary line per
    request carrying request_id, client_name, query_type, and path_taken --
    see main.py's search() and logging_config.py."""
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    mocker.patch.object(main.vector_db, "search", return_value=[])
    mock_log_info = mocker.patch.object(main.logger, "info")

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )
    query_id = resp.json()["query_id"]

    completion_calls = [
        c for c in mock_log_info.call_args_list if c.args and c.args[0] == "search request completed"
    ]
    assert len(completion_calls) == 1
    fields = completion_calls[0].kwargs["extra"]["structured_fields"]
    assert fields["request_id"] == query_id
    assert fields["client_name"] == "legacy"
    assert fields["query_type"] == "text"
    assert fields["no_match"] is True
    assert fields["path_taken"] == "empty_result_set"
    assert "latency_ms" in fields


def test_search_records_search_event(mocker):
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    mocker.patch.object(main.vector_db, "search", return_value=[])
    mock_record = mocker.patch.object(main.search_events, "record_search_event")

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )
    query_id = resp.json()["query_id"]

    mock_record.assert_called_once()
    kwargs = mock_record.call_args.kwargs
    assert kwargs["request_id"] == query_id
    assert kwargs["client_name"] == "legacy"
    assert kwargs["query_type"] == "text"
    assert kwargs["query_text_or_image_hash"] == "gold ring"
    assert kwargs["no_match"] is True
    assert kwargs["result_ids_returned_in_order"] == []


def test_search_event_recording_failure_does_not_break_the_search_response(mocker):
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    mocker.patch.object(main.vector_db, "search", return_value=[])
    mocker.patch.object(main.search_events, "record_search_event", side_effect=Exception("supabase down"))
    mock_capture = mocker.patch.object(main.sentry_sdk, "capture_exception")

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200  # best-effort: telemetry failure must not break the response
    mock_capture.assert_called_once()


def test_submit_search_feedback_requires_api_key():
    resp = client.post("/api/v1/search/req-1/feedback", json={"result_id": "ring-1", "action": "clicked"})
    assert resp.status_code == 401


def test_submit_search_feedback_happy_path(mocker):
    mock_record = mocker.patch.object(main.search_events, "record_feedback")

    resp = client.post(
        "/api/v1/search/req-1/feedback",
        json={"result_id": "ring-1", "action": "purchased"},
        headers={"x-api-key": VALID_KEY},
    )

    assert resp.status_code == 204
    mock_record.assert_called_once_with("req-1", "ring-1", "purchased")


def test_submit_search_feedback_rejects_invalid_action():
    resp = client.post(
        "/api/v1/search/req-1/feedback",
        json={"result_id": "ring-1", "action": "not-a-real-action"},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 422


def test_search_returns_429_with_retry_after_when_rate_limited(mocker, valid_jpeg_bytes):
    mocker.patch.object(
        main.rate_limit, "check", side_effect=main.rate_limit.RateLimitExceeded(retry_after_seconds=30)
    )

    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "30"


def test_search_with_corrupt_image_returns_400(corrupt_image_bytes):
    resp = client.post(
        "/api/v1/search",
        files=_image_file(corrupt_image_bytes),
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400
    assert "detail" in resp.json()


# --- Image queries ---

def test_search_happy_path(mocker, valid_jpeg_bytes):
    _pass_jewelry_gate(mocker)
    mocker.patch.object(main.embeddings, "embed_image", return_value=[0.1] * 8)
    raw_matches = [_fake_match("a", 0.91), _fake_match("b", 0.80), _fake_match("c", 0.61)]
    mocker.patch.object(main.vector_db, "search", return_value=raw_matches)
    mocker.patch.object(main.reranker, "rerank", return_value=[_fake_ranked(m) for m in raw_matches])

    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["no_match"] is False
    assert len(body["matches"]) == 3
    for match, raw in zip(body["matches"], raw_matches):
        assert match["similarity_percent"] == round(raw["score"] * 100, 1)


def test_search_all_below_threshold_returns_no_match_without_reranking(mocker, valid_jpeg_bytes):
    _pass_jewelry_gate(mocker)
    mocker.patch.object(main.embeddings, "embed_image", return_value=[0.1] * 8)
    weak_matches = [_fake_match("a", 0.10), _fake_match("b", 0.05)]
    mocker.patch.object(main.vector_db, "search", return_value=weak_matches)
    mock_rerank = mocker.patch.object(main.reranker, "rerank")

    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["no_match"] is True
    assert body["matches"] == []
    mock_rerank.assert_not_called()


def test_search_builds_correct_metadata_filter_from_category_and_price(mocker, valid_jpeg_bytes):
    _pass_jewelry_gate(mocker)
    mocker.patch.object(main.embeddings, "embed_image", return_value=[0.1] * 8)
    mock_search = mocker.patch.object(main.vector_db, "search", return_value=[])

    resp = client.post(
        "/api/v1/search",
        files=_image_file(valid_jpeg_bytes),
        data={"category": "ring", "min_price": "100", "max_price": "500"},
        headers={"x-api-key": VALID_KEY},
    )

    assert resp.status_code == 200
    kwargs = mock_search.call_args.kwargs
    assert kwargs["metadata_filter"] == {
        "category": {"$eq": "ring"},
        "price": {"$gte": 100.0, "$lte": 500.0},
    }


def test_search_image_response_has_query_type_image(mocker, valid_jpeg_bytes):
    _pass_jewelry_gate(mocker)
    mocker.patch.object(main.embeddings, "embed_image", return_value=[0.1] * 8)
    raw_matches = [_fake_match("a", 0.91)]
    mocker.patch.object(main.vector_db, "search", return_value=raw_matches)
    mock_rerank = mocker.patch.object(main.reranker, "rerank", return_value=[_fake_ranked(m) for m in raw_matches])

    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    assert resp.json()["query_type"] == "image"
    query_arg = mock_rerank.call_args.args[0]
    assert query_arg == {"type": "image", "bytes": mocker.ANY}


# --- Domain gate (Step 12) ---

def test_search_image_rejected_by_domain_gate_never_embeds_or_searches(mocker, valid_jpeg_bytes):
    mocker.patch.object(main.reranker, "is_plausibly_jewelry", return_value=False)
    mock_embed = mocker.patch.object(main.embeddings, "embed_image")
    mock_search = mocker.patch.object(main.vector_db, "search")

    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["no_match"] is True
    assert body["matches"] == []
    assert body["query_type"] == "image"
    assert body["reason"] == "The uploaded photo doesn't appear to show a piece of jewellery."
    mock_embed.assert_not_called()
    mock_search.assert_not_called()


def test_search_image_accepted_by_domain_gate_proceeds_normally(mocker, valid_jpeg_bytes):
    mocker.patch.object(main.reranker, "is_plausibly_jewelry", return_value=True)
    mocker.patch.object(main.embeddings, "embed_image", return_value=[0.1] * 8)
    raw_matches = [_fake_match("a", 0.91)]
    mocker.patch.object(main.vector_db, "search", return_value=raw_matches)
    mocker.patch.object(main.reranker, "rerank", return_value=[_fake_ranked(m) for m in raw_matches])

    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    assert resp.json()["no_match"] is False


# --- Text queries: default cheap path vs. conditional LLM rerank (Step 6) ---

def test_search_text_query_well_separated_never_calls_llm_rerank(mocker):
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    raw_matches = [_fake_match("a", 0.9), _fake_match("b", 0.8), _fake_match("c", 0.7)]
    mocker.patch.object(main.vector_db, "search", return_value=raw_matches)
    cheap_scored = [
        {**raw_matches[0], "confidence": "high", "reason": "x", "blended_score": 0.90},
        {**raw_matches[1], "confidence": "medium", "reason": "x", "blended_score": 0.50},
        {**raw_matches[2], "confidence": "low", "reason": "x", "blended_score": 0.40},
    ]
    mocker.patch.object(main.reranker, "score_candidates_cheap", return_value=cheap_scored)
    mock_rerank = mocker.patch.object(main.reranker, "rerank")

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["no_match"] is False
    assert [m["id"] for m in body["matches"]] == ["a", "b", "c"]
    mock_rerank.assert_not_called()


def test_search_text_query_ambiguous_top_results_triggers_llm_rerank(mocker):
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    raw_matches = [_fake_match("a", 0.9), _fake_match("b", 0.85), _fake_match("c", 0.83)]
    mocker.patch.object(main.vector_db, "search", return_value=raw_matches)
    # gap between 1st and 3rd place blended_score is 0.90 - 0.86 = 0.04 < 0.1
    cheap_scored = [
        {**raw_matches[0], "confidence": "medium", "reason": "x", "blended_score": 0.90},
        {**raw_matches[1], "confidence": "medium", "reason": "x", "blended_score": 0.88},
        {**raw_matches[2], "confidence": "medium", "reason": "x", "blended_score": 0.86},
    ]
    mocker.patch.object(main.reranker, "score_candidates_cheap", return_value=cheap_scored)
    llm_ranked = [_fake_ranked(m) for m in raw_matches]
    mock_rerank = mocker.patch.object(main.reranker, "rerank", return_value=llm_ranked)

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["no_match"] is False
    mock_rerank.assert_called_once()
    query_arg = mock_rerank.call_args.args[0]
    assert query_arg == {"type": "text", "text": "gold ring"}
    assert [m["id"] for m in body["matches"]] == ["a", "b", "c"]


def test_search_text_query_fewer_than_3_candidates_uses_first_vs_last_gap(mocker):
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    raw_matches = [_fake_match("a", 0.9), _fake_match("b", 0.5)]
    mocker.patch.object(main.vector_db, "search", return_value=raw_matches)
    # Only 2 candidates -> gap is 1st vs last (not 3rd, which doesn't exist).
    # 0.90 - 0.40 = 0.50 >= 0.1 -> cheap path.
    cheap_scored = [
        {**raw_matches[0], "confidence": "high", "reason": "x", "blended_score": 0.90},
        {**raw_matches[1], "confidence": "low", "reason": "x", "blended_score": 0.40},
    ]
    mocker.patch.object(main.reranker, "score_candidates_cheap", return_value=cheap_scored)
    mock_rerank = mocker.patch.object(main.reranker, "rerank")

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    mock_rerank.assert_not_called()


def test_search_text_query_empty_raw_matches_returns_no_match(mocker):
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    mocker.patch.object(main.vector_db, "search", return_value=[])
    mock_cheap = mocker.patch.object(main.reranker, "score_candidates_cheap")
    mock_rerank = mocker.patch.object(main.reranker, "rerank")

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["no_match"] is True
    assert body["query_type"] == "text"
    mock_cheap.assert_not_called()
    mock_rerank.assert_not_called()


def test_search_text_query_low_scores_but_nonempty_does_not_trigger_no_match(mocker):
    """The old threshold-based no_match must NOT fire for text queries
    anymore -- see main.py's _search_text and README.md §11."""
    mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    low_matches = [_fake_match("a", 0.05), _fake_match("b", 0.03)]
    mocker.patch.object(main.vector_db, "search", return_value=low_matches)
    # Gap deliberately >= 0.1 so this test stays on the cheap path -- the
    # ambiguous/LLM-rerank branch is covered separately above.
    cheap_scored = [
        {**low_matches[0], "confidence": "low", "reason": "x", "blended_score": 0.30},
        {**low_matches[1], "confidence": "low", "reason": "x", "blended_score": 0.05},
    ]
    mocker.patch.object(main.reranker, "score_candidates_cheap", return_value=cheap_scored)

    resp = client.post(
        "/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["no_match"] is False
    assert len(body["matches"]) == 2


def test_search_text_query_happy_path(mocker):
    mock_embed = mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    raw_matches = [_fake_match("a", 0.91), _fake_match("b", 0.80)]
    mocker.patch.object(main.vector_db, "search", return_value=raw_matches)
    cheap_scored = [
        {**raw_matches[0], "confidence": "high", "reason": "x", "blended_score": 0.9},
        {**raw_matches[1], "confidence": "low", "reason": "x", "blended_score": 0.3},
    ]
    mocker.patch.object(main.reranker, "score_candidates_cheap", return_value=cheap_scored)

    resp = client.post(
        "/api/v1/search",
        data={"query_text": "gold pink enamel earrings"},
        headers={"x-api-key": VALID_KEY},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["query_type"] == "text"
    assert body["no_match"] is False
    assert len(body["matches"]) == 2
    mock_embed.assert_called_once_with("gold pink enamel earrings")


def test_search_with_both_image_and_query_text_returns_400(valid_jpeg_bytes):
    resp = client.post(
        "/api/v1/search",
        files=_image_file(valid_jpeg_bytes),
        data={"query_text": "gold ring"},
        headers={"x-api-key": VALID_KEY},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Provide only one of image or text query, not both."


def test_search_with_neither_image_nor_query_text_returns_400():
    resp = client.post("/api/v1/search", headers={"x-api-key": VALID_KEY})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Provide either an image or a text query."


def test_search_with_blank_query_text_and_no_image_returns_400():
    resp = client.post(
        "/api/v1/search", data={"query_text": "   "}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Provide either an image or a text query."


def test_search_text_query_requires_api_key():
    resp = client.post("/api/v1/search", data={"query_text": "gold ring"})
    assert resp.status_code == 401


# --- Caching (Step 7) ---

def test_search_text_query_second_identical_search_skips_embedding(mocker):
    mock_embed = mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    mocker.patch.object(main.vector_db, "search", return_value=[])

    client.post("/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY})
    client.post("/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY})

    mock_embed.assert_called_once()


def test_search_text_query_different_text_does_not_hit_cache(mocker):
    mock_embed = mocker.patch.object(main.embeddings, "embed_text_query", return_value=[0.1] * 8)
    mocker.patch.object(main.vector_db, "search", return_value=[])

    client.post("/api/v1/search", data={"query_text": "gold ring"}, headers={"x-api-key": VALID_KEY})
    client.post("/api/v1/search", data={"query_text": "silver necklace"}, headers={"x-api-key": VALID_KEY})

    assert mock_embed.call_count == 2


# --- Catalog indexing (fused embed_catalog_item) ---

def _item(item_id, name="Item", category="ring", price=100.0, **extra):
    return {"item_id": item_id, "name": name, "category": category, "price": price, **extra}


def test_catalog_index_mismatched_lengths_returns_400(valid_jpeg_bytes):
    resp = client.post(
        "/api/v1/catalog/index",
        files=[("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg")), ("images", ("b.jpg", valid_jpeg_bytes, "image/jpeg"))],
        data={"items_json": json.dumps([_item("only-one")])},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400


def test_catalog_index_invalid_json_returns_400(valid_jpeg_bytes):
    resp = client.post(
        "/api/v1/catalog/index",
        files=[("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg"))],
        data={"items_json": "{not valid json"},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400


def test_catalog_index_items_json_not_an_array_returns_400(valid_jpeg_bytes):
    resp = client.post(
        "/api/v1/catalog/index",
        files=[("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg"))],
        data={"items_json": json.dumps(_item("id-1"))},  # object, not array
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400


def test_catalog_index_item_missing_required_field_returns_400_with_field_errors(valid_jpeg_bytes):
    bad_item = {"item_id": "id-1", "name": "Ring"}  # missing category, price
    resp = client.post(
        "/api/v1/catalog/index",
        files=[("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg"))],
        data={"items_json": json.dumps([bad_item])},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    fields_with_errors = {e["loc"][-1] for e in detail["errors"]}
    assert "category" in fields_with_errors
    assert "price" in fields_with_errors


def test_catalog_index_price_must_be_positive(valid_jpeg_bytes):
    resp = client.post(
        "/api/v1/catalog/index",
        files=[("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg"))],
        data={"items_json": json.dumps([_item("id-1", price=0)])},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400


def test_catalog_index_happy_path_stores_full_metadata(mocker, valid_jpeg_bytes):
    mock_upload = mocker.patch.object(
        main.object_storage, "upload_catalog_image",
        side_effect=lambda item_id, image_bytes: f"https://cdn.example.com/catalog/{item_id}.jpg",
    )
    mock_embed = mocker.patch.object(main.embeddings, "embed_catalog_item", return_value=[0.1] * 8)
    mock_upsert = mocker.patch.object(main.vector_db, "upsert_batch", return_value=1)
    mock_record = mocker.patch.object(main.catalog_store, "record_item")

    items = [
        _item("ring-1", name="Solitaire Ring", category="ring", price=980,
              caption="a gold ring", description="A hand-forged solitaire.",
              tags=["gold", "solitaire"], material="18k gold"),
        _item("necklace-1", name="Pearl Necklace", category="necklace", price=640),
    ]
    resp = client.post(
        "/api/v1/catalog/index",
        files=[
            ("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg")),
            ("images", ("b.jpg", valid_jpeg_bytes, "image/jpeg")),
        ],
        data={"items_json": json.dumps(items)},
        headers={"x-api-key": VALID_KEY},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["indexed_count"] == 2

    # one upsert_batch call per item (see _index_job docstring: per-item error isolation)
    assert mock_upsert.call_count == 2
    first_call_items = mock_upsert.call_args_list[0].args[0]
    assert first_call_items == [{
        "id": "ring-1",
        "vector": [0.1] * 8,
        "metadata": {
            "filename": "a.jpg",
            "name": "Solitaire Ring",
            "caption": "a gold ring",
            "description": "A hand-forged solitaire.",
            "tags": ["gold", "solitaire"],
            "category": "ring",
            "price": 980.0,
            "image_url": "https://cdn.example.com/catalog/ring-1.jpg",
            "material": "18k gold",
        },
    }]
    mock_record.assert_any_call("ring-1", first_call_items[0]["metadata"])
    mock_upload.assert_any_call("ring-1", mocker.ANY)

    # embed_catalog_item gets the fused text description built from metadata
    first_call_text = mock_embed.call_args_list[0].args[1]
    assert first_call_text == "name: Solitaire Ring. a gold ring. A hand-forged solitaire.. Tags: gold, solitaire."

    # second item had no material/caption/description/tags supplied -> defaults, no material key
    second_call_items = mock_upsert.call_args_list[1].args[0]
    assert second_call_items[0]["metadata"]["caption"] == ""
    assert second_call_items[0]["metadata"]["tags"] == []
    assert "material" not in second_call_items[0]["metadata"]


def test_catalog_index_per_item_failure_does_not_abort_batch(mocker, valid_jpeg_bytes):
    mock_upload = mocker.patch.object(
        main.object_storage, "upload_catalog_image",
        side_effect=lambda item_id, image_bytes: f"https://cdn.example.com/catalog/{item_id}.jpg",
    )
    mocker.patch.object(
        main.embeddings, "embed_catalog_item",
        side_effect=[[0.1] * 8, Exception("embedding API down"), [0.2] * 8],
    )
    mocker.patch.object(main.vector_db, "upsert_batch", return_value=1)
    mocker.patch.object(main.catalog_store, "record_item")

    items = [_item("id-1"), _item("id-2"), _item("id-3")]
    resp = client.post(
        "/api/v1/catalog/index",
        files=[
            ("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg")),
            ("images", ("b.jpg", valid_jpeg_bytes, "image/jpeg")),
            ("images", ("c.jpg", valid_jpeg_bytes, "image/jpeg")),
        ],
        data={"items_json": json.dumps(items)},
        headers={"x-api-key": VALID_KEY},
    )
    job_id = resp.json()["job_id"]

    status = client.get(f"/api/v1/catalog/jobs/{job_id}", headers={"x-api-key": VALID_KEY}).json()
    assert status["status"] == "done"  # not "failed" -- 2 of 3 succeeded
    assert status["processed"] == 3
    assert status["failed_items"] == [{"item_id": "id-2", "error": "embedding API down"}]
    uploaded_ids = {c.args[0] for c in mock_upload.call_args_list}
    assert uploaded_ids == {"id-1", "id-3"}  # id-2 failed before the upload step


def test_catalog_index_all_items_failing_marks_job_failed(mocker, valid_jpeg_bytes):
    mocker.patch.object(main.embeddings, "embed_catalog_item", side_effect=Exception("down"))

    resp = client.post(
        "/api/v1/catalog/index",
        files=[("images", ("a.jpg", valid_jpeg_bytes, "image/jpeg"))],
        data={"items_json": json.dumps([_item("id-1")])},
        headers={"x-api-key": VALID_KEY},
    )
    job_id = resp.json()["job_id"]

    status = client.get(f"/api/v1/catalog/jobs/{job_id}", headers={"x-api-key": VALID_KEY}).json()
    assert status["status"] == "failed"


def test_job_status_unknown_job_id_returns_404():
    resp = client.get("/api/v1/catalog/jobs/does-not-exist", headers={"x-api-key": VALID_KEY})
    assert resp.status_code == 404


def test_job_status_requires_api_key():
    resp = client.get("/api/v1/catalog/jobs/some-id")
    assert resp.status_code == 401


def test_catalog_items_requires_api_key():
    resp = client.get("/api/v1/catalog/items")
    assert resp.status_code == 401


def test_catalog_items_returns_paginated_list(mocker):
    mock_list = mocker.patch.object(
        main.catalog_store, "list_items", return_value=([{"item_id": "a", "name": "Ring"}], 42)
    )

    resp = client.get(
        "/api/v1/catalog/items", params={"limit": 5, "offset": 10}, headers={"x-api-key": VALID_KEY}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"items": [{"item_id": "a", "name": "Ring"}], "total": 42, "limit": 5, "offset": 10}
    mock_list.assert_called_once_with(limit=5, offset=10)


def test_catalog_items_clamps_limit_to_reasonable_bounds(mocker):
    mock_list = mocker.patch.object(main.catalog_store, "list_items", return_value=([], 0))

    client.get("/api/v1/catalog/items", params={"limit": 1000}, headers={"x-api-key": VALID_KEY})

    assert mock_list.call_args.kwargs["limit"] == 100


def test_get_catalog_item_requires_api_key():
    resp = client.get("/api/v1/catalog/items/id-1")
    assert resp.status_code == 401


def test_get_catalog_item_returns_404_for_unknown_id(mocker):
    mocker.patch.object(main.catalog_store, "get_item", return_value=None)
    resp = client.get("/api/v1/catalog/items/does-not-exist", headers={"x-api-key": VALID_KEY})
    assert resp.status_code == 404


def test_get_catalog_item_happy_path(mocker):
    mocker.patch.object(
        main.catalog_store, "get_item", return_value={"item_id": "id-1", "name": "Ring", "price": 100.0}
    )
    resp = client.get("/api/v1/catalog/items/id-1", headers={"x-api-key": VALID_KEY})
    assert resp.status_code == 200
    assert resp.json() == {"item_id": "id-1", "name": "Ring", "price": 100.0}


def _edit_fields(**overrides):
    fields = {"name": "Updated Ring", "category": "ring", "price": 150.0}
    fields.update(overrides)
    return fields


def test_update_catalog_item_requires_api_key():
    resp = client.patch("/api/v1/catalog/items/id-1", data={"fields": json.dumps(_edit_fields())})
    assert resp.status_code == 401


def test_update_catalog_item_returns_404_for_unknown_id(mocker):
    mocker.patch.object(main.catalog_store, "get_item", return_value=None)
    resp = client.patch(
        "/api/v1/catalog/items/does-not-exist",
        data={"fields": json.dumps(_edit_fields())},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 404


def test_update_catalog_item_invalid_json_returns_400(mocker):
    mocker.patch.object(main.catalog_store, "get_item", return_value=_item("id-1"))
    resp = client.patch(
        "/api/v1/catalog/items/id-1",
        data={"fields": "{not valid json"},
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400


def test_update_catalog_item_missing_required_field_returns_400(mocker):
    mocker.patch.object(main.catalog_store, "get_item", return_value=_item("id-1"))
    resp = client.patch(
        "/api/v1/catalog/items/id-1",
        data={"fields": json.dumps({"name": "Ring"})},  # missing category, price
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400


def test_update_catalog_item_metadata_only_skips_reembedding_and_reuses_existing_image(mocker):
    mocker.patch.object(
        main.catalog_store, "get_item",
        return_value={
            "item_id": "id-1", "filename": "old.jpg", "image_url": "/static/catalog/id-1.jpg",
            "name": "Ring", "category": "ring", "price": 100.0,
        },
    )
    mock_update_metadata = mocker.patch.object(main.vector_db, "update_metadata")
    mock_upsert = mocker.patch.object(main.vector_db, "upsert_batch")
    mock_embed = mocker.patch.object(main.embeddings, "embed_catalog_item")
    mock_record = mocker.patch.object(main.catalog_store, "record_item")

    resp = client.patch(
        "/api/v1/catalog/items/id-1",
        data={"fields": json.dumps(_edit_fields(name="Renamed Ring", price=200))},
        headers={"x-api-key": VALID_KEY},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Renamed Ring"
    assert body["price"] == 200.0
    assert body["image_url"] == "/static/catalog/id-1.jpg"  # unchanged

    mock_embed.assert_not_called()
    mock_upsert.assert_not_called()
    mock_update_metadata.assert_called_once()
    assert mock_update_metadata.call_args.args[0] == "id-1"
    mock_record.assert_called_once_with("id-1", mock_update_metadata.call_args.args[1])


def test_update_catalog_item_with_new_image_reembeds_and_reuploads(mocker, valid_jpeg_bytes):
    mocker.patch.object(
        main.catalog_store, "get_item",
        return_value={
            "item_id": "id-1", "filename": "old.jpg", "image_url": "https://cdn.example.com/catalog/id-1.jpg",
            "name": "Ring", "category": "ring", "price": 100.0,
        },
    )
    mock_upload = mocker.patch.object(
        main.object_storage, "upload_catalog_image", return_value="https://cdn.example.com/catalog/id-1.jpg"
    )
    mocker.patch.object(main.embeddings, "embed_catalog_item", return_value=[0.2] * 8)
    mock_upsert = mocker.patch.object(main.vector_db, "upsert_batch")
    mock_update_metadata = mocker.patch.object(main.vector_db, "update_metadata")
    mocker.patch.object(main.catalog_store, "record_item")

    resp = client.patch(
        "/api/v1/catalog/items/id-1",
        data={"fields": json.dumps(_edit_fields())},
        files={"image": ("new.jpg", valid_jpeg_bytes, "image/jpeg")},
        headers={"x-api-key": VALID_KEY},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "new.jpg"
    assert body["image_url"] == "https://cdn.example.com/catalog/id-1.jpg"

    mock_update_metadata.assert_not_called()
    mock_upload.assert_called_once_with("id-1", mocker.ANY)
    mock_upsert.assert_called_once()
    assert mock_upsert.call_args.args[0][0]["id"] == "id-1"
    assert mock_upsert.call_args.args[0][0]["vector"] == [0.2] * 8


def test_delete_catalog_item_requires_api_key():
    resp = client.delete("/api/v1/catalog/items/id-1")
    assert resp.status_code == 401


def test_delete_catalog_item_returns_404_for_unknown_id(mocker):
    mocker.patch.object(main.catalog_store, "get_item", return_value=None)
    resp = client.delete("/api/v1/catalog/items/does-not-exist", headers={"x-api-key": VALID_KEY})
    assert resp.status_code == 404


def test_delete_catalog_item_happy_path_removes_vector_row_and_image(mocker):
    mocker.patch.object(main.catalog_store, "get_item", return_value=_item("id-1"))
    mock_delete_vector = mocker.patch.object(main.vector_db, "delete_by_id")
    mock_delete_row = mocker.patch.object(main.catalog_store, "delete_item")
    mock_delete_image = mocker.patch.object(main.object_storage, "delete_catalog_image")

    resp = client.delete("/api/v1/catalog/items/id-1", headers={"x-api-key": VALID_KEY})

    assert resp.status_code == 204
    mock_delete_vector.assert_called_once_with("id-1")
    mock_delete_row.assert_called_once_with("id-1")
    mock_delete_image.assert_called_once_with("id-1")
