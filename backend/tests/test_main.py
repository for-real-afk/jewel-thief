import json

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

VALID_KEY = main.settings.api_key


def _image_file(jpeg_bytes, filename="query.jpg"):
    return {"image": (filename, jpeg_bytes, "image/jpeg")}


def _fake_match(id_, score, category="ring"):
    return {"id": id_, "score": score, "metadata": {"category": category}}


def _fake_ranked(match, confidence="high", reason="matching cut"):
    return {**match, "confidence": confidence, "reason": reason}


@pytest.mark.parametrize("caption,expected", [
    ("a rose gold ring with a pear-cut diamond", "ring"),
    ("a silver choker necklace with pearls", "necklace"),
    ("gold jhumka earrings with green stones", "earrings"),
    ("an engraved gold bangle bracelet", "bracelet"),
    ("a decorative brooch with no clear category", "other"),
])
def test_infer_category_matches_keywords(caption, expected):
    assert main.infer_category(caption) == expected


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_without_api_key_header_returns_401(valid_jpeg_bytes):
    resp = client.post("/api/v1/search", files=_image_file(valid_jpeg_bytes))
    assert resp.status_code == 401


def test_search_with_wrong_api_key_returns_401(valid_jpeg_bytes):
    resp = client.post(
        "/api/v1/search", files=_image_file(valid_jpeg_bytes), headers={"x-api-key": "totally-wrong"}
    )
    assert resp.status_code == 401


def test_search_with_corrupt_image_returns_400(corrupt_image_bytes):
    resp = client.post(
        "/api/v1/search",
        files=_image_file(corrupt_image_bytes),
        headers={"x-api-key": VALID_KEY},
    )
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_search_happy_path(mocker, valid_jpeg_bytes):
    mocker.patch.object(main.embeddings, "embed_query_image", return_value=[0.1] * 8)
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
    mocker.patch.object(main.embeddings, "embed_query_image", return_value=[0.1] * 8)
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
    mocker.patch.object(main.embeddings, "embed_query_image", return_value=[0.1] * 8)
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


def test_catalog_index_happy_path_stores_full_metadata(mocker, valid_jpeg_bytes, tmp_path):
    mocker.patch.object(main, "CATALOG_IMAGE_DIR", tmp_path)
    mocker.patch.object(main.embeddings, "embed_catalog_image", return_value=[0.1] * 8)
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
            "image_url": "/static/catalog/ring-1.jpg",
            "material": "18k gold",
        },
    }]
    mock_record.assert_any_call("ring-1", first_call_items[0]["metadata"])
    assert (tmp_path / "ring-1.jpg").exists()

    # second item had no material/caption/description/tags supplied -> defaults, no material key
    second_call_items = mock_upsert.call_args_list[1].args[0]
    assert second_call_items[0]["metadata"]["caption"] == ""
    assert second_call_items[0]["metadata"]["tags"] == []
    assert "material" not in second_call_items[0]["metadata"]


def test_catalog_index_per_item_failure_does_not_abort_batch(mocker, valid_jpeg_bytes, tmp_path):
    mocker.patch.object(main, "CATALOG_IMAGE_DIR", tmp_path)
    mocker.patch.object(
        main.embeddings, "embed_catalog_image",
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
    assert (tmp_path / "id-1.jpg").exists()
    assert (tmp_path / "id-3.jpg").exists()
    assert not (tmp_path / "id-2.jpg").exists()


def test_catalog_index_all_items_failing_marks_job_failed(mocker, valid_jpeg_bytes, tmp_path):
    mocker.patch.object(main, "CATALOG_IMAGE_DIR", tmp_path)
    mocker.patch.object(main.embeddings, "embed_catalog_image", side_effect=Exception("down"))

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
