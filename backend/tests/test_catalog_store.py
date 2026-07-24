import catalog_store


def test_record_and_list_round_trip(mocker, tmp_path):
    mocker.patch.object(catalog_store, "_STORE_PATH", tmp_path / "catalog_store.json")

    catalog_store.record_item("id-1", {"name": "Ring", "category": "ring"})
    catalog_store.record_item("id-2", {"name": "Necklace", "category": "necklace"})

    items, total = catalog_store.list_items(limit=10, offset=0)

    assert total == 2
    assert {i["item_id"] for i in items} == {"id-1", "id-2"}
    ring = next(i for i in items if i["item_id"] == "id-1")
    assert ring["name"] == "Ring"
    assert ring["category"] == "ring"


def test_record_item_overwrites_existing_id(mocker, tmp_path):
    mocker.patch.object(catalog_store, "_STORE_PATH", tmp_path / "catalog_store.json")

    catalog_store.record_item("id-1", {"name": "Old Name"})
    catalog_store.record_item("id-1", {"name": "New Name"})

    items, total = catalog_store.list_items(limit=10, offset=0)

    assert total == 1
    assert items[0]["name"] == "New Name"


def test_list_items_newest_first(mocker, tmp_path):
    mocker.patch.object(catalog_store, "_STORE_PATH", tmp_path / "catalog_store.json")

    catalog_store.record_item("first", {"name": "First"})
    catalog_store.record_item("second", {"name": "Second"})
    catalog_store.record_item("third", {"name": "Third"})

    items, _ = catalog_store.list_items(limit=10, offset=0)

    assert [i["item_id"] for i in items] == ["third", "second", "first"]


def test_list_items_pagination(mocker, tmp_path):
    mocker.patch.object(catalog_store, "_STORE_PATH", tmp_path / "catalog_store.json")

    for i in range(5):
        catalog_store.record_item(f"id-{i}", {"name": f"Item {i}"})

    page1, total = catalog_store.list_items(limit=2, offset=0)
    page2, _ = catalog_store.list_items(limit=2, offset=2)

    assert total == 5
    assert len(page1) == 2
    assert len(page2) == 2
    assert {i["item_id"] for i in page1} & {i["item_id"] for i in page2} == set()


def test_list_items_on_empty_store_returns_empty(mocker, tmp_path):
    mocker.patch.object(catalog_store, "_STORE_PATH", tmp_path / "does_not_exist.json")

    items, total = catalog_store.list_items(limit=10, offset=0)

    assert items == []
    assert total == 0
