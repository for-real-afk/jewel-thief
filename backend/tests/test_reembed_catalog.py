import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "reembed_catalog.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reembed_catalog", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reembed_catalog"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reembed_catalog():
    return _load_module()


def _item(item_id, **extra):
    return {"item_id": item_id, "name": "Item", "category": "ring", "price": 100.0, **extra}


def test_fetch_all_catalog_items_paginates_until_exhausted(reembed_catalog, mocker):
    page1 = [_item(f"id-{i}") for i in range(100)]
    page2 = [_item(f"id-{i}") for i in range(100, 150)]
    mock_list = mocker.patch.object(
        reembed_catalog.catalog_store, "list_items",
        side_effect=[(page1, 150), (page2, 150)],
    )

    items = reembed_catalog.fetch_all_catalog_items()

    assert len(items) == 150
    assert mock_list.call_count == 2
    assert mock_list.call_args_list[0].kwargs == {"limit": 100, "offset": 0}
    assert mock_list.call_args_list[1].kwargs == {"limit": 100, "offset": 100}


def test_fetch_all_catalog_items_empty_catalog_returns_empty(reembed_catalog, mocker):
    mocker.patch.object(reembed_catalog.catalog_store, "list_items", return_value=([], 0))

    assert reembed_catalog.fetch_all_catalog_items() == []


def test_re_embed_item_builds_fused_vector_from_stored_image_and_metadata(reembed_catalog, mocker, tmp_path):
    mocker.patch.object(reembed_catalog, "STATIC_CATALOG_DIR", tmp_path)
    (tmp_path / "ring-1.jpg").write_bytes(b"fake-jpeg-bytes")
    mock_embed = mocker.patch.object(reembed_catalog.embeddings, "embed_catalog_item", return_value=[0.1] * 8)

    item = {
        "item_id": "ring-1", "name": "Ruby Ring", "category": "ring", "price": 500.0,
        "caption": "a red ring", "description": "hand forged", "tags": ["gold", "ruby"],
        "image_url": "/static/catalog/ring-1.jpg",
    }
    result = reembed_catalog.re_embed_item(item)

    assert result["id"] == "ring-1"
    assert result["vector"] == [0.1] * 8
    assert "item_id" not in result["metadata"]
    assert result["metadata"]["name"] == "Ruby Ring"

    mock_embed.assert_called_once()
    assert mock_embed.call_args.args[0] == b"fake-jpeg-bytes"
    assert mock_embed.call_args.args[1] == "name: Ruby Ring. a red ring. hand forged. Tags: gold, ruby."


def test_re_embed_item_missing_image_raises(reembed_catalog, mocker, tmp_path):
    mocker.patch.object(reembed_catalog, "STATIC_CATALOG_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        reembed_catalog.re_embed_item(_item("no-image-here"))


def test_main_dry_run_never_calls_upsert_batch(reembed_catalog, mocker, tmp_path):
    mocker.patch.object(reembed_catalog, "STATIC_CATALOG_DIR", tmp_path)
    items = [_item("id-1"), _item("id-2")]
    mocker.patch.object(reembed_catalog, "fetch_all_catalog_items", return_value=items)
    for item in items:
        (tmp_path / f"{item['item_id']}.jpg").write_bytes(b"fake-bytes")
    mocker.patch.object(reembed_catalog.embeddings, "embed_catalog_item", return_value=[0.1] * 8)
    mock_upsert = mocker.patch.object(reembed_catalog.vector_db, "upsert_batch")

    mocker.patch.object(sys, "argv", ["reembed_catalog.py", "--dry-run"])
    exit_code = reembed_catalog.main()

    mock_upsert.assert_not_called()
    assert exit_code == 0


def test_main_real_run_calls_upsert_batch_once_with_all_items(reembed_catalog, mocker, tmp_path):
    mocker.patch.object(reembed_catalog, "STATIC_CATALOG_DIR", tmp_path)
    items = [_item("id-1"), _item("id-2")]
    mocker.patch.object(reembed_catalog, "fetch_all_catalog_items", return_value=items)
    for item in items:
        (tmp_path / f"{item['item_id']}.jpg").write_bytes(b"fake-bytes")
    mocker.patch.object(reembed_catalog.embeddings, "embed_catalog_item", return_value=[0.1] * 8)
    mock_upsert = mocker.patch.object(reembed_catalog.vector_db, "upsert_batch", return_value=2)

    mocker.patch.object(sys, "argv", ["reembed_catalog.py"])
    exit_code = reembed_catalog.main()

    mock_upsert.assert_called_once()
    upserted = mock_upsert.call_args.args[0]
    assert {item["id"] for item in upserted} == {"id-1", "id-2"}
    assert exit_code == 0


def test_main_per_item_failure_does_not_abort_run(reembed_catalog, mocker, tmp_path):
    mocker.patch.object(reembed_catalog, "STATIC_CATALOG_DIR", tmp_path)
    items = [_item("id-1"), _item("id-2"), _item("id-3")]
    mocker.patch.object(reembed_catalog, "fetch_all_catalog_items", return_value=items)
    for item in items:
        (tmp_path / f"{item['item_id']}.jpg").write_bytes(b"fake-bytes")
    mocker.patch.object(
        reembed_catalog.embeddings, "embed_catalog_item",
        side_effect=[[0.1] * 8, Exception("embedding API down"), [0.2] * 8],
    )
    mock_upsert = mocker.patch.object(reembed_catalog.vector_db, "upsert_batch", return_value=2)

    mocker.patch.object(sys, "argv", ["reembed_catalog.py"])
    exit_code = reembed_catalog.main()

    upserted = mock_upsert.call_args.args[0]
    assert {item["id"] for item in upserted} == {"id-1", "id-3"}
    assert exit_code == 1  # non-zero because at least one item failed
