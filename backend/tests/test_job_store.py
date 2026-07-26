import job_store


def test_in_memory_job_store_set_then_get_returns_value():
    s = job_store.InMemoryJobStore()

    s.set("job-1", {"status": "pending", "total": 2, "processed": 0, "failed_items": []})

    assert s.get("job-1") == {"status": "pending", "total": 2, "processed": 0, "failed_items": []}


def test_in_memory_job_store_miss_returns_none():
    s = job_store.InMemoryJobStore()

    assert s.get("does-not-exist") is None


def test_redis_job_store_get_deserializes_json(mocker):
    s = job_store.RedisJobStore("redis://fake")
    mocker.patch.object(
        s._client, "get", return_value=b'{"status": "done", "total": 1, "processed": 1, "failed_items": []}'
    )

    assert s.get("job-1") == {"status": "done", "total": 1, "processed": 1, "failed_items": []}


def test_redis_job_store_get_miss_returns_none(mocker):
    s = job_store.RedisJobStore("redis://fake")
    mocker.patch.object(s._client, "get", return_value=None)

    assert s.get("does-not-exist") is None


def test_redis_job_store_set_serializes_json_with_ttl(mocker):
    s = job_store.RedisJobStore("redis://fake")
    mock_set = mocker.patch.object(s._client, "set")

    s.set("job-1", {"status": "pending", "total": 1, "processed": 0, "failed_items": []})

    mock_set.assert_called_once_with(
        "job:job-1",
        '{"status": "pending", "total": 1, "processed": 0, "failed_items": []}',
        ex=job_store._JOB_TTL_SECONDS,
    )


def test_make_job_store_uses_redis_when_url_configured(mocker):
    mocker.patch.object(job_store.settings, "redis_url", "redis://fake:6379")
    mock_redis_job_store = mocker.patch.object(job_store, "RedisJobStore")

    result = job_store._make_job_store()

    mock_redis_job_store.assert_called_once_with("redis://fake:6379")
    assert result is mock_redis_job_store.return_value


def test_make_job_store_uses_in_memory_when_url_not_configured(mocker):
    mocker.patch.object(job_store.settings, "redis_url", "")

    result = job_store._make_job_store()

    assert isinstance(result, job_store.InMemoryJobStore)
