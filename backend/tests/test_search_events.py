import pytest

from app import search_events


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, store):
        self._store = store

    def insert(self, row):
        self._row = row
        return self

    def execute(self):
        self._store.append(self._row)
        return _FakeResult([self._row])


class _FakeClient:
    def __init__(self):
        self.events = []
        self.feedback = []

    def table(self, name):
        if name == search_events._EVENTS_TABLE:
            return _FakeTable(self.events)
        if name == search_events._FEEDBACK_TABLE:
            return _FakeTable(self.feedback)
        raise ValueError(f"unexpected table {name}")


@pytest.fixture(autouse=True)
def fake_supabase(mocker):
    fake = _FakeClient()
    mocker.patch.object(search_events, "_client", fake)
    return fake


def test_record_search_event_writes_expected_fields(fake_supabase):
    search_events.record_search_event(
        request_id="req-1",
        client_name="legacy",
        query_type="text",
        query_text_or_image_hash="gold ring",
        retrieved_candidates=[{"id": "ring-1", "score": 0.9}],
        path_taken="cheap",
        result_ids_returned_in_order=["ring-1"],
        no_match=False,
    )

    assert len(fake_supabase.events) == 1
    row = fake_supabase.events[0]
    assert row["request_id"] == "req-1"
    assert row["client_name"] == "legacy"
    assert row["query_type"] == "text"
    assert row["path_taken"] == "cheap"
    assert row["result_ids_returned_in_order"] == ["ring-1"]
    assert row["no_match"] is False
    assert "timestamp" in row


def test_record_feedback_writes_expected_fields(fake_supabase):
    search_events.record_feedback("req-1", "ring-1", "clicked")

    assert len(fake_supabase.feedback) == 1
    row = fake_supabase.feedback[0]
    assert row["query_id"] == "req-1"
    assert row["result_id"] == "ring-1"
    assert row["action"] == "clicked"
    assert "id" in row and "created_at" in row
