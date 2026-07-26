import json
import logging

from app import logging_config


def _make_record(structured_fields=None, exc_info=None):
    record = logging.LogRecord(
        name="jewellery_search",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something happened: %s",
        args=("detail",),
        exc_info=exc_info,
    )
    if structured_fields is not None:
        record.structured_fields = structured_fields
    return record


def test_format_produces_valid_json_with_core_fields():
    formatter = logging_config.JsonFormatter()

    line = formatter.format(_make_record())
    payload = json.loads(line)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "jewellery_search"
    assert payload["message"] == "something happened: detail"


def test_format_merges_structured_fields_into_payload():
    formatter = logging_config.JsonFormatter()

    line = formatter.format(_make_record(structured_fields={"request_id": "abc-123", "latency_ms": 12.5}))
    payload = json.loads(line)

    assert payload["request_id"] == "abc-123"
    assert payload["latency_ms"] == 12.5


def test_format_without_structured_fields_does_not_raise():
    formatter = logging_config.JsonFormatter()

    line = formatter.format(_make_record())  # no structured_fields attribute at all

    assert json.loads(line)["message"] == "something happened: detail"


def test_configure_logging_installs_json_formatter_on_root_handler():
    logging_config.configure_logging()

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, logging_config.JsonFormatter)
