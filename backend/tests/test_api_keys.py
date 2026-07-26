import pytest

import api_keys


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    """Minimal stand-in for the chainable supabase-py table query builder."""

    def __init__(self, store):
        self._store = store
        self._op = None
        self._eq_column = None
        self._eq_value = None

    def insert(self, row):
        self._op = "insert"
        self._row = row
        return self

    def update(self, fields):
        self._op = "update"
        self._fields = fields
        return self

    def select(self, columns):
        self._op = "select"
        return self

    def eq(self, column, value):
        self._eq_column = column
        self._eq_value = value
        return self

    def execute(self):
        if self._op == "insert":
            self._store[self._row["key_id"]] = self._row
            return _FakeResult([self._row])

        if self._op == "update":
            for row in self._store.values():
                if row.get(self._eq_column) == self._eq_value:
                    row.update(self._fields)
            return _FakeResult([])

        rows = [r for r in self._store.values() if r.get(self._eq_column) == self._eq_value]
        return _FakeResult(rows)


class _FakeClient:
    def __init__(self):
        self._store = {}

    def table(self, name):
        return _FakeTable(self._store)


@pytest.fixture(autouse=True)
def fake_supabase(mocker):
    mocker.patch.object(api_keys, "_client", _FakeClient())


def test_create_key_then_lookup_succeeds():
    key_id, raw_key = api_keys.create_key("acme-mobile", rate_limit_tier="bulk_admin")

    record = api_keys.lookup_key(raw_key)

    assert record == {"client_name": "acme-mobile", "rate_limit_tier": "bulk_admin"}


def test_create_key_defaults_to_standard_tier():
    _, raw_key = api_keys.create_key("acme-mobile")

    record = api_keys.lookup_key(raw_key)

    assert record["rate_limit_tier"] == "standard"


def test_lookup_key_unknown_key_returns_none():
    assert api_keys.lookup_key("not-a-real-key") is None


def test_lookup_key_wrong_raw_value_does_not_match_stored_hash():
    api_keys.create_key("acme-mobile")

    assert api_keys.lookup_key("some-other-guess") is None


def test_revoke_key_makes_lookup_fail():
    key_id, raw_key = api_keys.create_key("acme-mobile")
    assert api_keys.lookup_key(raw_key) is not None

    api_keys.revoke_key(key_id)

    assert api_keys.lookup_key(raw_key) is None


def test_create_key_returns_a_key_that_differs_each_call():
    _, raw_key_1 = api_keys.create_key("client-a")
    _, raw_key_2 = api_keys.create_key("client-b")

    assert raw_key_1 != raw_key_2
