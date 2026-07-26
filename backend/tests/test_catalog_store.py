import pytest

from app import catalog_store


class _FakeResult:
    def __init__(self, data, count):
        self.data = data
        self.count = count


class _FakeTable:
    """Minimal stand-in for the chainable supabase-py table query builder."""

    def __init__(self, store):
        self._store = store
        self._op = None
        self._desc = False
        self._range = None
        self._eq_value = None

    def upsert(self, row):
        self._op = "upsert"
        self._row = row
        return self

    def select(self, columns, count=None):
        self._op = "select"
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, column, value):
        self._eq_value = value
        return self

    def order(self, column, desc=False):
        self._desc = desc
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def limit(self, n):
        self._range = (0, n - 1)
        return self

    def execute(self):
        if self._op == "upsert":
            self._store[self._row["item_id"]] = self._row
            return _FakeResult([self._row], None)

        if self._op == "delete":
            self._store.pop(self._eq_value, None)
            return _FakeResult([], None)

        rows = list(self._store.values())
        if self._eq_value is not None:
            rows = [r for r in rows if r["item_id"] == self._eq_value]
        if self._desc:
            rows = list(reversed(rows))
        total = len(rows)
        if self._range:
            start, end = self._range
            rows = rows[start:end + 1]
        return _FakeResult(rows, total)


class _FakeClient:
    def __init__(self):
        self._store = {}

    def table(self, name):
        return _FakeTable(self._store)


@pytest.fixture(autouse=True)
def fake_supabase(mocker):
    mocker.patch.object(catalog_store, "_client", _FakeClient())


def test_ping_does_not_raise_when_supabase_reachable():
    catalog_store.ping()  # fake client always succeeds -- should not raise


def test_ping_propagates_failure(mocker):
    mocker.patch.object(catalog_store, "_client", mocker.Mock(**{
        "table.return_value.select.return_value.limit.return_value.execute.side_effect": Exception("down"),
    }))

    with pytest.raises(Exception, match="down"):
        catalog_store.ping()


def test_record_and_list_round_trip():
    catalog_store.record_item("id-1", {"name": "Ring", "category": "ring"})
    catalog_store.record_item("id-2", {"name": "Necklace", "category": "necklace"})

    items, total = catalog_store.list_items(limit=10, offset=0)

    assert total == 2
    assert {i["item_id"] for i in items} == {"id-1", "id-2"}
    ring = next(i for i in items if i["item_id"] == "id-1")
    assert ring["name"] == "Ring"
    assert ring["category"] == "ring"


def test_record_item_overwrites_existing_id():
    catalog_store.record_item("id-1", {"name": "Old Name"})
    catalog_store.record_item("id-1", {"name": "New Name"})

    items, total = catalog_store.list_items(limit=10, offset=0)

    assert total == 1
    assert items[0]["name"] == "New Name"


def test_list_items_newest_first():
    catalog_store.record_item("first", {"name": "First"})
    catalog_store.record_item("second", {"name": "Second"})
    catalog_store.record_item("third", {"name": "Third"})

    items, _ = catalog_store.list_items(limit=10, offset=0)

    assert [i["item_id"] for i in items] == ["third", "second", "first"]


def test_list_items_pagination():
    for i in range(5):
        catalog_store.record_item(f"id-{i}", {"name": f"Item {i}"})

    page1, total = catalog_store.list_items(limit=2, offset=0)
    page2, _ = catalog_store.list_items(limit=2, offset=2)

    assert total == 5
    assert len(page1) == 2
    assert len(page2) == 2
    assert {i["item_id"] for i in page1} & {i["item_id"] for i in page2} == set()


def test_list_items_on_empty_store_returns_empty():
    items, total = catalog_store.list_items(limit=10, offset=0)

    assert items == []
    assert total == 0


def test_get_item_returns_matching_item():
    catalog_store.record_item("id-1", {"name": "Ring"})
    catalog_store.record_item("id-2", {"name": "Necklace"})

    item = catalog_store.get_item("id-2")

    assert item == {"item_id": "id-2", "name": "Necklace"}


def test_get_item_returns_none_for_unknown_id():
    assert catalog_store.get_item("does-not-exist") is None


def test_delete_item_removes_only_the_matching_item():
    catalog_store.record_item("id-1", {"name": "Ring"})
    catalog_store.record_item("id-2", {"name": "Necklace"})

    catalog_store.delete_item("id-1")

    items, total = catalog_store.list_items(limit=10, offset=0)
    assert total == 1
    assert items[0]["item_id"] == "id-2"


def test_delete_item_on_unknown_id_is_a_no_op():
    catalog_store.record_item("id-1", {"name": "Ring"})

    catalog_store.delete_item("does-not-exist")

    _, total = catalog_store.list_items(limit=10, offset=0)
    assert total == 1
