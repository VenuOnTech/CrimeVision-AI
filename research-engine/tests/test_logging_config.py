import json
import logging

from logging_config import JsonFormatter, configure_logging


def test_json_formatter_includes_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    record.request_id = "abc123"
    record.custom = 42
    payload = json.loads(formatter.format(record))
    assert payload["msg"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "abc123"
    assert payload["custom"] == 42


def test_configure_logging_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_JSON", "1")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "app.log"))
    configure_logging()
    root = logging.getLogger()
    first_count = len(root.handlers)
    assert first_count >= 1
    configure_logging()
    assert len(root.handlers) == first_count
    assert root.level == logging.DEBUG