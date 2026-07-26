"""
Structured (JSON) logging setup.

configure_logging() is called once at process startup (main.py, at import
time) so every logger.info/logger.exception call across the app emits one
JSON object per line -- parseable by a log aggregator, not just readable by
a human tailing a file. Existing call sites (main.py, embeddings.py,
vector_db.py, reranker.py, cache.py) need no changes to keep working; pass
extra={"structured_fields": {...}} on top of the usual message/args to attach
queryable fields (request_id, client_name, latency_ms, ...) to a specific
log line.
"""
import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "structured_fields", {}))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
