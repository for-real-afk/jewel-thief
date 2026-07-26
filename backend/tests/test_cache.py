import pytest

import cache


def test_in_memory_cache_set_then_get_returns_value():
    c = cache.InMemoryCache()

    c.set("key-1", [0.1, 0.2, 0.3])

    assert c.get("key-1") == [0.1, 0.2, 0.3]


def test_in_memory_cache_miss_returns_none():
    c = cache.InMemoryCache()

    assert c.get("does-not-exist") is None


def test_in_memory_cache_expires_after_ttl(mocker):
    c = cache.InMemoryCache()
    mock_time = mocker.patch.object(cache.time, "time", return_value=1000.0)

    c.set("key-1", [0.1, 0.2], ttl_seconds=60)

    mock_time.return_value = 1000.0 + 61  # past expiry
    assert c.get("key-1") is None


def test_in_memory_cache_does_not_expire_before_ttl(mocker):
    c = cache.InMemoryCache()
    mock_time = mocker.patch.object(cache.time, "time", return_value=1000.0)

    c.set("key-1", [0.1, 0.2], ttl_seconds=60)

    mock_time.return_value = 1000.0 + 59  # not yet expired
    assert c.get("key-1") == [0.1, 0.2]


def test_cache_key_is_deterministic_for_same_input():
    key1 = cache.cache_key("Gold Ring", {"category": {"$eq": "ring"}})
    key2 = cache.cache_key("Gold Ring", {"category": {"$eq": "ring"}})

    assert key1 == key2


def test_cache_key_normalizes_case_and_whitespace():
    key1 = cache.cache_key("Gold Ring", {})
    key2 = cache.cache_key("  gold ring  ", {})

    assert key1 == key2


def test_cache_key_differs_for_different_filters():
    key1 = cache.cache_key("gold ring", {"category": {"$eq": "ring"}})
    key2 = cache.cache_key("gold ring", {"category": {"$eq": "necklace"}})

    assert key1 != key2


def test_cache_key_is_order_independent_for_filter_dict():
    key1 = cache.cache_key("gold ring", {"a": 1, "b": 2})
    key2 = cache.cache_key("gold ring", {"b": 2, "a": 1})

    assert key1 == key2


def test_redis_cache_get_deserializes_json(mocker):
    c = cache.RedisCache("redis://fake")
    mocker.patch.object(c._client, "get", return_value=b"[0.1, 0.2, 0.3]")

    assert c.get("key-1") == [0.1, 0.2, 0.3]


def test_redis_cache_get_miss_returns_none(mocker):
    c = cache.RedisCache("redis://fake")
    mocker.patch.object(c._client, "get", return_value=None)

    assert c.get("does-not-exist") is None


def test_redis_cache_set_serializes_json_with_ttl(mocker):
    c = cache.RedisCache("redis://fake")
    mock_set = mocker.patch.object(c._client, "set")

    c.set("key-1", [0.1, 0.2], ttl_seconds=120)

    mock_set.assert_called_once_with("key-1", "[0.1, 0.2]", ex=120)


def test_ping_no_op_when_in_memory_cache_in_use(mocker):
    mocker.patch.object(cache, "_cache", cache.InMemoryCache())

    cache.ping()  # should not raise -- nothing to ping


def test_ping_calls_redis_client_ping_when_redis_cache_in_use(mocker):
    redis_cache = cache.RedisCache("redis://fake")
    mocker.patch.object(cache, "_cache", redis_cache)
    mock_ping = mocker.patch.object(redis_cache._client, "ping")

    cache.ping()

    mock_ping.assert_called_once()


def test_ping_propagates_failure_when_redis_unreachable(mocker):
    redis_cache = cache.RedisCache("redis://fake")
    mocker.patch.object(cache, "_cache", redis_cache)
    mocker.patch.object(redis_cache._client, "ping", side_effect=Exception("down"))

    with pytest.raises(Exception, match="down"):
        cache.ping()


def test_make_cache_uses_redis_when_url_configured(mocker):
    mocker.patch.object(cache.settings, "redis_url", "redis://fake:6379")
    mock_redis_cache = mocker.patch.object(cache, "RedisCache")

    result = cache._make_cache()

    mock_redis_cache.assert_called_once_with("redis://fake:6379")
    assert result is mock_redis_cache.return_value


def test_make_cache_uses_in_memory_when_url_not_configured(mocker):
    mocker.patch.object(cache.settings, "redis_url", "")

    result = cache._make_cache()

    assert isinstance(result, cache.InMemoryCache)


def test_search_cache_hit_skips_embed_text_query(mocker):
    """End-to-end: an identical second text search must not call
    embeddings.embed_text_query at all -- it should be served entirely from
    cache.py's InMemoryCache."""
    import embeddings
    import main
    import vector_db

    mocker.patch.object(cache, "_cache", cache.InMemoryCache())
    mock_embed = mocker.patch.object(embeddings, "embed_text_query", return_value=[0.1] * 8)
    mocker.patch.object(vector_db, "search", return_value=[])

    main._search_text("gold ring", {})
    main._search_text("gold ring", {})

    mock_embed.assert_called_once()
