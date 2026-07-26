"""
Search-event and feedback logging, Supabase-backed (search_events,
search_feedback tables).

Captures one row per search -- the input a future learning-to-rank or
domain-classifier training pass would need, and it cannot be reconstructed
retroactively, so it starts now regardless of whether anything trains on it
yet. No training pipeline exists yet; this module only writes, nothing reads
it back to influence ranking.

Tables:
  search_events(request_id PK, timestamp, client_name, query_type,
    query_text_or_image_hash, retrieved_candidates jsonb, path_taken,
    result_ids_returned_in_order jsonb, no_match)
  search_feedback(id PK, query_id, result_id, action, created_at)
"""
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from supabase import create_client

from config import get_settings

settings = get_settings()
_client = create_client(settings.supabase_url, settings.supabase_key)
_EVENTS_TABLE = "search_events"
_FEEDBACK_TABLE = "search_feedback"

FeedbackAction = Literal["clicked", "purchased", "dismissed"]


def record_search_event(
    request_id: str,
    client_name: str,
    query_type: str,
    query_text_or_image_hash: str,
    retrieved_candidates: list[dict],
    path_taken: str,
    result_ids_returned_in_order: list[str],
    no_match: bool,
) -> None:
    """Best-effort by design at the call site (main.py wraps this in a
    try/except so a Supabase hiccup never breaks a real search) -- this
    function itself just writes, it doesn't swallow errors."""
    _client.table(_EVENTS_TABLE).insert({
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_name": client_name,
        "query_type": query_type,
        "query_text_or_image_hash": query_text_or_image_hash,
        "retrieved_candidates": retrieved_candidates,
        "path_taken": path_taken,
        "result_ids_returned_in_order": result_ids_returned_in_order,
        "no_match": no_match,
    }).execute()


def record_feedback(query_id: str, result_id: str, action: FeedbackAction) -> None:
    _client.table(_FEEDBACK_TABLE).insert({
        "id": str(uuid4()),
        "query_id": query_id,
        "result_id": result_id,
        "action": action,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
